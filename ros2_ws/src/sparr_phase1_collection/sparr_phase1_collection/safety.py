"""Explicit safety decisions for Phase 1 state and pose-command inputs.

This module only classifies.  It never clamps a command, normalizes a source
quaternion, or silently substitutes another pose.  The runtime must record and
act on the returned ACCEPT, REJECT, or HOLD status.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

import numpy as np

from .pose_math import Pose
from .robot_state import RobotStateSnapshot, Vector3, timestamp_to_ns


class SafetyConfigurationError(ValueError):
    """Raised for an invalid guard or limit configuration."""


class GuardStatus(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    HOLD = "hold"


@dataclass(frozen=True, slots=True)
class GuardDecision:
    status: GuardStatus
    reasons: tuple[str, ...] = ()
    state_age_ns: int | None = None

    @property
    def accepted(self) -> bool:
        return self.status is GuardStatus.ACCEPT


    @property
    def rejected(self) -> bool:
        return self.status is GuardStatus.REJECT

    @property
    def reason(self) -> str:
        return "; ".join(self.reasons)


@dataclass(frozen=True, slots=True)
class StateGuardConfig:
    max_state_age_ns: int
    max_future_skew_ns: int = 0
    expected_joint_count: int | None = 7
    quaternion_norm_tolerance: float = 0.01
    require_robot_timestamp: bool = True
    hold_on_unchanged_robot_timestamp: bool = True
    expected_pose_frame_id: str | None = None

    def __post_init__(self) -> None:
        integer_fields = {
            "max_state_age_ns": self.max_state_age_ns,
            "max_future_skew_ns": self.max_future_skew_ns,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SafetyConfigurationError(f"{name} must be a non-negative integer")
        if (
            self.expected_joint_count is not None
            and (
                isinstance(self.expected_joint_count, bool)
                or not isinstance(self.expected_joint_count, int)
                or self.expected_joint_count <= 0
            )
        ):
            raise SafetyConfigurationError(
                "expected_joint_count must be positive or None"
            )
        if (
            not math.isfinite(self.quaternion_norm_tolerance)
            or self.quaternion_norm_tolerance < 0.0
        ):
            raise SafetyConfigurationError(
                "quaternion_norm_tolerance must be finite and non-negative"
            )
        if self.expected_pose_frame_id is not None and not self.expected_pose_frame_id:
            raise SafetyConfigurationError(
                "expected_pose_frame_id must be non-empty or None"
            )


class RobotStateGuard:
    """Reject invalid robot states and hold on stale robot states.

    The separate RDK timestamp check detects the Flexiv bridge failure mode in
    which an unchanged robot payload is republished with a fresh ROS header.
    """

    def __init__(self, config: StateGuardConfig) -> None:
        self._config = config
        self._last_robot_timestamp_ns: int | None = None
        self._last_ros_timestamp_ns: int | None = None

    @property
    def config(self) -> StateGuardConfig:
        return self._config

    def reset(self) -> None:
        self._last_robot_timestamp_ns = None
        self._last_ros_timestamp_ns = None

    def evaluate(
        self,
        snapshot: RobotStateSnapshot,
        now: int | object,
    ) -> GuardDecision:
        now_ns = timestamp_to_ns(now)
        errors = snapshot.validation_errors(
            expected_joint_count=self._config.expected_joint_count,
            quaternion_norm_tolerance=self._config.quaternion_norm_tolerance,
            require_robot_timestamp=self._config.require_robot_timestamp,
            expected_pose_frame_id=self._config.expected_pose_frame_id,
        )
        if errors:
            return GuardDecision(GuardStatus.REJECT, errors)

        state_age_ns = now_ns - snapshot.ros_timestamp_ns
        if state_age_ns < -self._config.max_future_skew_ns:
            return GuardDecision(
                GuardStatus.REJECT,
                ("robot state timestamp is too far in the future",),
                state_age_ns,
            )
        if state_age_ns > self._config.max_state_age_ns:
            return GuardDecision(
                GuardStatus.HOLD,
                (
                    f"robot state age {state_age_ns} ns exceeds "
                    f"{self._config.max_state_age_ns} ns",
                ),
                state_age_ns,
            )

        robot_timestamp_ns = snapshot.robot_timestamp_ns
        if robot_timestamp_ns is not None and self._last_robot_timestamp_ns is not None:
            if robot_timestamp_ns < self._last_robot_timestamp_ns:
                return GuardDecision(
                    GuardStatus.REJECT,
                    ("robot_timestamp regressed",),
                    state_age_ns,
                )
            ros_stamp_advanced = (
                self._last_ros_timestamp_ns is not None
                and snapshot.ros_timestamp_ns > self._last_ros_timestamp_ns
            )
            if (
                self._config.hold_on_unchanged_robot_timestamp
                and robot_timestamp_ns == self._last_robot_timestamp_ns
                and ros_stamp_advanced
            ):
                return GuardDecision(
                    GuardStatus.HOLD,
                    ("robot payload timestamp did not advance",),
                    state_age_ns,
                )

        self._last_robot_timestamp_ns = robot_timestamp_ns
        self._last_ros_timestamp_ns = snapshot.ros_timestamp_ns
        return GuardDecision(GuardStatus.ACCEPT, (), state_age_ns)


def guard_robot_state(
    snapshot: RobotStateSnapshot,
    now: int | object,
    *,
    max_state_age_ns: int,
    previous_snapshot: RobotStateSnapshot | None = None,
    max_future_skew_ns: int = 0,
) -> GuardDecision:
    """Stateless convenience wrapper for one robot-state comparison."""

    guard = RobotStateGuard(
        StateGuardConfig(
            max_state_age_ns=max_state_age_ns,
            max_future_skew_ns=max_future_skew_ns,
        )
    )
    if previous_snapshot is not None:
        guard._last_robot_timestamp_ns = previous_snapshot.robot_timestamp_ns
        guard._last_ros_timestamp_ns = previous_snapshot.ros_timestamp_ns
    return guard.evaluate(snapshot, now)


# ``PoseCommandLimits`` and ``guard_pose_command`` are not called by
# ``collector_node``: no workspace envelope or per-step motion limit is enforced
# on commanded poses.


@dataclass(frozen=True, slots=True)
class PoseCommandLimits:
    """Optional hard rejection limits for a requested absolute pose."""

    workspace_min: Vector3 | None = None
    workspace_max: Vector3 | None = None
    max_translation_step_m: float | None = None
    max_rotation_step_rad: float | None = None
    quaternion_norm_tolerance: float = 0.01

    def __post_init__(self) -> None:
        if (self.workspace_min is None) != (self.workspace_max is None):
            raise SafetyConfigurationError(
                "workspace_min and workspace_max must be provided together"
            )
        if self.workspace_min is not None and self.workspace_max is not None:
            if not self.workspace_min.is_finite() or not self.workspace_max.is_finite():
                raise SafetyConfigurationError("workspace bounds must be finite")
            if any(
                lower > upper
                for lower, upper in zip(
                    self.workspace_min.as_tuple(), self.workspace_max.as_tuple()
                )
            ):
                raise SafetyConfigurationError("workspace_min exceeds workspace_max")
        for name, value in {
            "max_translation_step_m": self.max_translation_step_m,
            "max_rotation_step_rad": self.max_rotation_step_rad,
        }.items():
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise SafetyConfigurationError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.quaternion_norm_tolerance)
            or self.quaternion_norm_tolerance < 0.0
        ):
            raise SafetyConfigurationError(
                "quaternion_norm_tolerance must be finite and non-negative"
            )


def guard_pose_command(
    requested: Pose,
    current_flange: Pose,
    limits: PoseCommandLimits,
) -> GuardDecision:
    """Accept or reject a pose request without returning a modified pose."""

    invalid_reasons: list[str] = []
    for label, pose in (("requested", requested), ("current_flange", current_flange)):
        if not isinstance(pose, Pose):
            invalid_reasons.append(f"{label} is not a Pose")
            continue
        if not np.all(np.isfinite(pose.position)) or not np.all(
            np.isfinite(pose.quaternion_xyzw)
        ):
            invalid_reasons.append(f"{label} contains a non-finite value")
            continue
        norm_error = abs(float(np.linalg.norm(pose.quaternion_xyzw)) - 1.0)
        if norm_error > limits.quaternion_norm_tolerance:
            invalid_reasons.append(
                f"{label} quaternion norm error {norm_error:.9g} exceeds "
                f"{limits.quaternion_norm_tolerance:.9g}"
            )
    if invalid_reasons:
        return GuardDecision(GuardStatus.REJECT, tuple(invalid_reasons))

    reasons: list[str] = []
    if limits.workspace_min is not None and limits.workspace_max is not None:
        for axis, value, lower, upper in zip(
            "xyz",
            requested.position,
            limits.workspace_min.as_tuple(),
            limits.workspace_max.as_tuple(),
        ):
            if value < lower or value > upper:
                reasons.append(
                    f"requested {axis}={value:.9g} is outside [{lower:.9g}, {upper:.9g}]"
                )

    if limits.max_translation_step_m is not None:
        delta = tuple(
            target - current
            for target, current in zip(
                requested.position, current_flange.position
            )
        )
        translation_step = math.sqrt(sum(value * value for value in delta))
        if translation_step > limits.max_translation_step_m:
            reasons.append(
                f"translation step {translation_step:.9g} m exceeds "
                f"{limits.max_translation_step_m:.9g} m"
            )

    if limits.max_rotation_step_rad is not None:
        rotation_step = _quaternion_distance_rad(
            requested.quaternion_xyzw, current_flange.quaternion_xyzw
        )
        if rotation_step > limits.max_rotation_step_rad:
            reasons.append(
                f"rotation step {rotation_step:.9g} rad exceeds "
                f"{limits.max_rotation_step_rad:.9g} rad"
            )

    if reasons:
        return GuardDecision(GuardStatus.REJECT, tuple(reasons))
    return GuardDecision(GuardStatus.ACCEPT)


def _quaternion_distance_rad(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    # Inputs were already accepted as unit quaternions.  abs(dot) accounts for
    # q and -q representing the same orientation.  The min is solely a guard
    # against floating-point roundoff outside acos' mathematical domain.
    dot = abs(sum(left * right for left, right in zip(first, second)))
    return 2.0 * math.acos(min(1.0, dot))


__all__ = [
    "GuardDecision",
    "GuardStatus",
    "PoseCommandLimits",
    "RobotStateGuard",
    "SafetyConfigurationError",
    "StateGuardConfig",
    "guard_pose_command",
    "guard_robot_state",
]
