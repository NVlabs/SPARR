"""Deterministic trial randomization and explicit Phase 1 trial transitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
import random
from typing import Mapping

import numpy as np

from .observations import ObservationConfig, offset_point_world
from .pose_math import Pose, quaternion_multiply_xyzw
from .robot_state import RobotStateSnapshot, timestamp_to_ns


DEFAULT_XY_NOISE_M = 0.005

#: Default metric: raw flange origin to the exact socket root.  It carries
#: the full tool offset, so its threshold is not an insertion clearance.
SUCCESS_METRIC_FLANGE_TO_SOCKET_ROOT = "flange_to_socket_root_position_l2"
#: Geometrically meaningful metric: the two points the policy actually sees,
#: i.e. the flange-local EEF keypoint against the exact socket keypoint.
SUCCESS_METRIC_EEF_KEYPOINT_TO_SOCKET_KEYPOINT = (
    "eef_keypoint_to_socket_keypoint_position_l2"
)
#: Same two points, judged by an axis-aligned box. The insertion axis needs a
#: looser bound because the plug tip sits beyond the EEF keypoint.
SUCCESS_METRIC_EEF_KEYPOINT_TO_SOCKET_KEYPOINT_AXIS_TOLERANCES = (
    "eef_keypoint_to_socket_keypoint_axis_tolerances"
)
SUPPORTED_SUCCESS_METRICS: tuple[str, ...] = (
    SUCCESS_METRIC_FLANGE_TO_SOCKET_ROOT,
    SUCCESS_METRIC_EEF_KEYPOINT_TO_SOCKET_KEYPOINT,
    SUCCESS_METRIC_EEF_KEYPOINT_TO_SOCKET_KEYPOINT_AXIS_TOLERANCES,
)
#: Metrics judged by a scalar L2 radius, i.e. the ones that require
#: ``distance_threshold_m`` and reject ``axis_tolerances_m``.
SCALAR_THRESHOLD_SUCCESS_METRICS: tuple[str, ...] = (
    SUCCESS_METRIC_FLANGE_TO_SOCKET_ROOT,
    SUCCESS_METRIC_EEF_KEYPOINT_TO_SOCKET_KEYPOINT,
)
#: Metrics judged by a per-axis tolerance box, i.e. the ones that require
#: ``axis_tolerances_m`` and reject ``distance_threshold_m``.
AXIS_TOLERANCE_SUCCESS_METRICS: tuple[str, ...] = (
    SUCCESS_METRIC_EEF_KEYPOINT_TO_SOCKET_KEYPOINT_AXIS_TOLERANCES,
)
DEFAULT_SUCCESS_METRIC = SUCCESS_METRIC_FLANGE_TO_SOCKET_ROOT


def validate_axis_tolerances(
    value: object,
    label: str = "axis_tolerances_m",
) -> tuple[float, float, float]:
    """Return ``value`` as a validated ``(x, y, z)`` tolerance triple.

    Raises :class:`TrialConfigurationError` unless it holds exactly three
    finite, strictly positive metre values.
    """

    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TrialConfigurationError(f"{label} must be a [x, y, z] list of metres")
    if len(value) != 3:
        raise TrialConfigurationError(f"{label} must contain exactly three values")
    try:
        tolerances = tuple(float(component) for component in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrialConfigurationError(f"{label} must be numeric") from exc
    if any(
        not math.isfinite(component) or component <= 0.0 for component in tolerances
    ):
        raise TrialConfigurationError(
            f"{label} must contain finite and strictly positive values"
        )
    return tolerances  # type: ignore[return-value]


class TrialConfigurationError(ValueError):
    """Raised for an invalid trial configuration or target pose."""


@dataclass(frozen=True, slots=True)
class PoseNoise:
    """Symmetric uniform pose-noise limits.

    Rotation is parameterized but disabled by default.  Phase 1 defaults apply
    independent X/Y offsets in ``[-5 mm, +5 mm]`` and no Z or rotation noise.
    """

    xy_half_range_m: tuple[float, float] = (
        DEFAULT_XY_NOISE_M,
        DEFAULT_XY_NOISE_M,
    )
    z_half_range_m: float = 0.0
    roll_half_range_rad: float = 0.0
    pitch_half_range_rad: float = 0.0
    yaw_half_range_rad: float = 0.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.xy_half_range_m, (tuple, list))
            or len(self.xy_half_range_m) != 2
        ):
            raise TrialConfigurationError(
                "xy_half_range_m must contain independent (x, y) half-ranges"
            )
        try:
            xy_half_range_m = tuple(float(value) for value in self.xy_half_range_m)
            scalar_values = {
                "z_half_range_m": float(self.z_half_range_m),
                "roll_half_range_rad": float(self.roll_half_range_rad),
                "pitch_half_range_rad": float(self.pitch_half_range_rad),
                "yaw_half_range_rad": float(self.yaw_half_range_rad),
            }
        except (TypeError, ValueError, OverflowError) as exc:
            raise TrialConfigurationError("pose-noise limits must be numeric") from exc
        object.__setattr__(self, "xy_half_range_m", xy_half_range_m)
        for name, value in scalar_values.items():
            object.__setattr__(self, name, value)
        values = {
            "x_half_range_m": xy_half_range_m[0],
            "y_half_range_m": xy_half_range_m[1],
            **scalar_values,
        }
        for name, value in values.items():
            if not math.isfinite(value) or value < 0.0:
                raise TrialConfigurationError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class SampledPoseOffset:
    x_m: float
    y_m: float
    z_m: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float


@dataclass(frozen=True, slots=True)
class SuccessEvaluation:
    #: ``distance_m`` is always the plain L2 norm of ``error_xyz_m`` so records
    #: and summaries stay comparable across metrics.  For an axis-tolerance
    #: metric it is reporting-only: the box in ``axis_tolerances_m`` decides
    #: ``succeeded`` and ``threshold_m`` is NaN because no radius applies.
    succeeded: bool
    distance_m: float
    threshold_m: float
    error_xyz_m: tuple[float, float, float]
    metric: str = DEFAULT_SUCCESS_METRIC
    axis_tolerances_m: tuple[float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class TrialTargets:
    """All targets for one seeded trial.

    ``socket_root_exact`` is immutable ground truth for success evaluation.
    ``socket_root_policy`` is the independently noised policy observation.
    There is no generic ``socket_pose`` alias, so they cannot be confused.

    ``seed`` is the base configuration seed shared by every trial, while
    ``derived_seed`` is the per-trial value that actually seeded the RNG and is
    what a reader needs to reproduce this trial.
    """

    seed: int
    derived_seed: int
    trial_index: int
    nominal_start: Pose
    randomized_start: Pose
    socket_root_exact: Pose
    socket_root_policy: Pose
    start_offset: SampledPoseOffset
    socket_policy_offset: SampledPoseOffset

    @property
    def exact_socket_root(self) -> Pose:
        return self.socket_root_exact


    def evaluate_success(
        self,
        robot_state: RobotStateSnapshot,
        *,
        threshold_m: float | None,
        metric: str = DEFAULT_SUCCESS_METRIC,
        observation_config: ObservationConfig | None = None,
        axis_tolerances_m: tuple[float, float, float] | None = None,
    ) -> SuccessEvaluation:
        """Evaluate against the exact, never the policy, socket root."""

        return evaluate_success(
            robot_state,
            self.socket_root_exact,
            threshold_m=threshold_m,
            metric=metric,
            observation_config=observation_config,
            axis_tolerances_m=axis_tolerances_m,
        )


class TrialRandomizer:
    """Generate reproducible start and policy-target perturbations.

    Each trial has a random stream derived from ``sha256(seed, trial_index)``,
    so a trial reproduces independently of invocation order and process
    restarts -- which is also what makes resuming at a later index exact.
    """

    def __init__(
        self,
        seed: int,
        *,
        start_noise: PoseNoise | None = None,
        socket_policy_noise: PoseNoise | None = None,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TrialConfigurationError("seed must be an integer")
        self._seed = seed
        self._start_noise = start_noise or PoseNoise()
        self._socket_policy_noise = socket_policy_noise or PoseNoise()

    @property
    def seed(self) -> int:
        return self._seed

    def sample(
        self,
        trial_index: int,
        nominal_start: Pose,
        exact_socket_root: Pose,
    ) -> TrialTargets:
        if isinstance(trial_index, bool) or not isinstance(trial_index, int):
            raise TrialConfigurationError("trial_index must be an integer")
        if trial_index < 0:
            raise TrialConfigurationError("trial_index must be non-negative")
        _validate_pose(nominal_start, "nominal_start")
        _validate_pose(exact_socket_root, "exact_socket_root")

        derived_seed = _derived_seed(self._seed, trial_index)
        rng = random.Random(derived_seed)
        start_offset = _sample_offset(rng, self._start_noise)
        socket_offset = _sample_offset(rng, self._socket_policy_noise)
        randomized_start = _offset_pose(nominal_start, start_offset)
        policy_socket = _offset_pose(exact_socket_root, socket_offset)

        return TrialTargets(
            seed=self._seed,
            derived_seed=derived_seed,
            trial_index=trial_index,
            nominal_start=nominal_start,
            randomized_start=randomized_start,
            socket_root_exact=exact_socket_root,
            socket_root_policy=policy_socket,
            start_offset=start_offset,
            socket_policy_offset=socket_offset,
        )


def evaluate_success(
    robot_state: RobotStateSnapshot,
    exact_socket_root: Pose,
    *,
    threshold_m: float | None,
    metric: str = DEFAULT_SUCCESS_METRIC,
    observation_config: ObservationConfig | None = None,
    axis_tolerances_m: tuple[float, float, float] | None = None,
) -> SuccessEvaluation:
    """Compute success against the exact (never the noised) socket root.

    ``flange_to_socket_root_position_l2`` (default) compares the raw flange
    origin, so it includes the whole tool offset. The ``eef_keypoint_*`` metrics
    compare the two keypoints the policy observes, using offsets from the loaded
    ``Phase1PolicySpec`` that must be supplied via ``observation_config``; the
    ``_axis_tolerances`` variant requires each of ``|dx|, |dy|, |dz|`` to be
    within ``axis_tolerances_m`` and ``threshold_m`` to be ``None``.

    Frame caveat: ``axis_tolerances_m`` is applied in world axes, which is only
    correct for an axis-vertical socket mount (socket X = world +Z). For a
    tilted mount, rotate ``error`` into the socket frame first.
    """

    if metric not in SUPPORTED_SUCCESS_METRICS:
        raise TrialConfigurationError(
            f"unsupported success metric {metric!r}; "
            f"expected one of {SUPPORTED_SUCCESS_METRICS}"
        )
    if metric in AXIS_TOLERANCE_SUCCESS_METRICS:
        if threshold_m is not None:
            raise TrialConfigurationError(
                f"success metric {metric!r} is judged by axis_tolerances_m; "
                "threshold_m must be None"
            )
        if axis_tolerances_m is None:
            raise TrialConfigurationError(
                f"success metric {metric!r} requires axis_tolerances_m"
            )
        tolerances = validate_axis_tolerances(axis_tolerances_m)
    else:
        if axis_tolerances_m is not None:
            raise TrialConfigurationError(
                f"success metric {metric!r} is judged by a scalar threshold_m; "
                "axis_tolerances_m must be None"
            )
        tolerances = None
        if (
            threshold_m is None
            or not math.isfinite(threshold_m)
            or threshold_m <= 0.0
        ):
            raise TrialConfigurationError("threshold_m must be finite and positive")
    _validate_pose(exact_socket_root, "exact_socket_root")
    flange = robot_state.success_pose
    _validate_pose(flange, "robot_state.flange_pose")

    if metric == SUCCESS_METRIC_FLANGE_TO_SOCKET_ROOT:
        measured = flange.position
        reference = exact_socket_root.position
    else:
        if observation_config is None:
            raise TrialConfigurationError(
                f"success metric {metric!r} requires the checkpoint keypoint "
                "offsets; pass observation_config"
            )
        measured = offset_point_world(flange, observation_config.eef_offset_flange)
        reference = offset_point_world(
            exact_socket_root, observation_config.socket_keypoint_offset
        )

    error = tuple(float(component) for component in (measured - reference))
    # Reported for every metric so records stay comparable across metrics.
    distance = math.sqrt(sum(component * component for component in error))
    if tolerances is None:
        succeeded = distance <= threshold_m
    else:
        # World axes; see the frame caveat in the docstring.
        succeeded = all(
            abs(component) <= tolerance
            for component, tolerance in zip(error, tolerances)
        )
    return SuccessEvaluation(
        succeeded=succeeded,
        distance_m=distance,
        threshold_m=math.nan if tolerances is not None else threshold_m,
        error_xyz_m=error,
        metric=metric,
        axis_tolerances_m=tolerances,
    )


class TrialState(str, Enum):
    IDLE = "idle"
    PAUSING = "pausing"
    RETRACTING = "retracting"
    MOVING_TO_START = "moving_to_start"
    SETTLING = "settling"
    READY = "ready"
    RUNNING = "running"
    HOLDING = "holding"
    SUCCEEDED = "succeeded"
    TIMED_OUT = "timed_out"
    REJECTED = "rejected"
    ABORTED = "aborted"


class TrialEvent(str, Enum):
    BEGIN = "begin"
    PAUSE_ACKNOWLEDGED = "pause_acknowledged"
    RETRACTED = "retracted"
    START_REACHED = "start_reached"
    SETTLED = "settled"
    POLICY_RESET_ACKNOWLEDGED = "policy_reset_acknowledged"
    SAFETY_HOLD = "safety_hold"
    STATE_FRESH = "state_fresh"
    SUCCESS = "success"
    TIMEOUT = "timeout"
    SAFETY_REJECT = "safety_reject"
    ABORT = "abort"
    CLEAR = "clear"


@dataclass(frozen=True, slots=True)
class TrialTransition:
    previous: TrialState
    event: TrialEvent
    current: TrialState
    timestamp_ns: int
    reason: str


class InvalidTrialTransition(RuntimeError):
    """Raised when a caller attempts to skip an explicit handshake."""


_TRANSITIONS: Mapping[tuple[TrialState, TrialEvent], TrialState] = {
    (TrialState.IDLE, TrialEvent.BEGIN): TrialState.PAUSING,
    (TrialState.PAUSING, TrialEvent.PAUSE_ACKNOWLEDGED): TrialState.RETRACTING,
    (TrialState.RETRACTING, TrialEvent.RETRACTED): TrialState.MOVING_TO_START,
    (TrialState.MOVING_TO_START, TrialEvent.START_REACHED): TrialState.SETTLING,
    (TrialState.SETTLING, TrialEvent.SETTLED): TrialState.READY,
    (TrialState.READY, TrialEvent.POLICY_RESET_ACKNOWLEDGED): TrialState.RUNNING,
    (TrialState.RUNNING, TrialEvent.SAFETY_HOLD): TrialState.HOLDING,
    (TrialState.HOLDING, TrialEvent.STATE_FRESH): TrialState.RUNNING,
    (TrialState.RUNNING, TrialEvent.SUCCESS): TrialState.SUCCEEDED,
    (TrialState.RUNNING, TrialEvent.TIMEOUT): TrialState.TIMED_OUT,
    (TrialState.HOLDING, TrialEvent.TIMEOUT): TrialState.TIMED_OUT,
}

_TERMINAL_STATES = frozenset(
    {
        TrialState.SUCCEEDED,
        TrialState.TIMED_OUT,
        TrialState.REJECTED,
        TrialState.ABORTED,
    }
)


class TrialStateMachine:
    """Small event-driven state machine with pause/reset acknowledgements."""

    def __init__(self) -> None:
        self._state = TrialState.IDLE
        self._history: list[TrialTransition] = []
        self._last_timestamp_ns: int | None = None

    @property
    def state(self) -> TrialState:
        return self._state

    @property
    def history(self) -> tuple[TrialTransition, ...]:
        return tuple(self._history)

    @property
    def is_terminal(self) -> bool:
        return self._state in _TERMINAL_STATES

    def transition(
        self,
        event: TrialEvent | str,
        timestamp: int | object,
        *,
        reason: str = "",
    ) -> TrialTransition:
        try:
            normalized_event = event if isinstance(event, TrialEvent) else TrialEvent(event)
        except ValueError as exc:
            raise InvalidTrialTransition(f"unknown event {event!r}") from exc
        timestamp_ns = timestamp_to_ns(timestamp)
        if timestamp_ns < 0:
            raise InvalidTrialTransition("transition timestamp must be non-negative")
        if self._last_timestamp_ns is not None and timestamp_ns < self._last_timestamp_ns:
            raise InvalidTrialTransition("transition timestamps must be monotonic")

        key = (self._state, normalized_event)
        if normalized_event is TrialEvent.ABORT and self._state not in _TERMINAL_STATES:
            next_state = TrialState.ABORTED
        elif (
            normalized_event is TrialEvent.SAFETY_REJECT
            and self._state not in _TERMINAL_STATES
            and self._state is not TrialState.IDLE
        ):
            next_state = TrialState.REJECTED
        elif normalized_event is TrialEvent.CLEAR and self._state in _TERMINAL_STATES:
            next_state = TrialState.IDLE
        else:
            try:
                next_state = _TRANSITIONS[key]
            except KeyError as exc:
                raise InvalidTrialTransition(
                    f"event {normalized_event.value!r} is invalid from {self._state.value!r}"
                ) from exc

        transition = TrialTransition(
            previous=self._state,
            event=normalized_event,
            current=next_state,
            timestamp_ns=timestamp_ns,
            reason=str(reason),
        )
        self._state = next_state
        self._last_timestamp_ns = timestamp_ns
        self._history.append(transition)
        return transition


def _derived_seed(seed: int, trial_index: int) -> int:
    # Exactly 8 unsigned bytes, so the result always fits the uint64 column the
    # recorder writes it to.
    material = f"sparr-phase1:{seed}:{trial_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False)


def _sample_offset(rng: random.Random, limits: PoseNoise) -> SampledPoseOffset:
    x_half_range_m, y_half_range_m = limits.xy_half_range_m
    return SampledPoseOffset(
        x_m=rng.uniform(-x_half_range_m, x_half_range_m),
        y_m=rng.uniform(-y_half_range_m, y_half_range_m),
        z_m=rng.uniform(-limits.z_half_range_m, limits.z_half_range_m),
        roll_rad=rng.uniform(
            -limits.roll_half_range_rad, limits.roll_half_range_rad
        ),
        pitch_rad=rng.uniform(
            -limits.pitch_half_range_rad, limits.pitch_half_range_rad
        ),
        yaw_rad=rng.uniform(-limits.yaw_half_range_rad, limits.yaw_half_range_rad),
    )


def _offset_pose(pose: Pose, offset: SampledPoseOffset) -> Pose:
    orientation = pose.quaternion_xyzw
    if any(
        value != 0.0
        for value in (offset.roll_rad, offset.pitch_rad, offset.yaw_rad)
    ):
        delta = _quaternion_from_rpy(
            offset.roll_rad, offset.pitch_rad, offset.yaw_rad
        )
        orientation = quaternion_multiply_xyzw(delta, orientation)
    return Pose(
        position=pose.position
        + np.asarray(
            (offset.x_m, offset.y_m, offset.z_m),
            dtype=np.float64,
        ),
        quaternion_xyzw=orientation,
    )


def _validate_pose(pose: Pose, label: str) -> None:
    if not isinstance(pose, Pose):
        raise TrialConfigurationError(f"{label} must be a Pose")
    if not np.all(np.isfinite(pose.position)) or not np.all(
        np.isfinite(pose.quaternion_xyzw)
    ):
        raise TrialConfigurationError(f"{label} contains a non-finite value")
    if abs(float(np.linalg.norm(pose.quaternion_xyzw)) - 1.0) > 0.01:
        raise TrialConfigurationError(f"{label} quaternion is not unit length")


def _quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return np.asarray(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        dtype=np.float64,
    )


__all__ = [
    "AXIS_TOLERANCE_SUCCESS_METRICS",
    "DEFAULT_SUCCESS_METRIC",
    "DEFAULT_XY_NOISE_M",
    "SCALAR_THRESHOLD_SUCCESS_METRICS",
    "SUCCESS_METRIC_EEF_KEYPOINT_TO_SOCKET_KEYPOINT",
    "SUCCESS_METRIC_EEF_KEYPOINT_TO_SOCKET_KEYPOINT_AXIS_TOLERANCES",
    "SUCCESS_METRIC_FLANGE_TO_SOCKET_ROOT",
    "SUPPORTED_SUCCESS_METRICS",
    "InvalidTrialTransition",
    "PoseNoise",
    "SampledPoseOffset",
    "SuccessEvaluation",
    "TrialConfigurationError",
    "TrialEvent",
    "TrialRandomizer",
    "TrialState",
    "TrialStateMachine",
    "TrialTargets",
    "TrialTransition",
    "evaluate_success",
    "validate_axis_tolerances",
]
