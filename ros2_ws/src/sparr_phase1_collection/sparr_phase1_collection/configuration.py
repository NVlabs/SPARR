# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Typed configuration and startup validation for Phase 1 collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from .trial_state_machine import (
    AXIS_TOLERANCE_SUCCESS_METRICS,
    DEFAULT_SUCCESS_METRIC,
    SUPPORTED_SUCCESS_METRICS,
    TrialConfigurationError,
    validate_axis_tolerances,
)


class ConfigurationError(ValueError):
    """Raised when a collection configuration is unsafe or incomplete."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: yaml.Loader, node: yaml.Node, deep: bool = False) -> dict:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConfigurationError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a mapping")
    return value


def _pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigurationError(f"{name} must contain exactly two values")
    result = float(value[0]), float(value[1])
    if not all(math.isfinite(item) for item in result):
        raise ConfigurationError(f"{name} must contain finite values")
    return result


def _triple(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ConfigurationError(f"{name} must contain exactly three values")
    result = float(value[0]), float(value[1]), float(value[2])
    if not all(math.isfinite(item) for item in result):
        raise ConfigurationError(f"{name} must contain finite values")
    return result


def _pose7(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 7:
        raise ConfigurationError(f"{name} must be [x,y,z,qx,qy,qz,qw]")
    pose = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in pose):
        raise ConfigurationError(f"{name} must contain finite values")
    if sum(v * v for v in pose[3:]) <= 1.0e-12:
        raise ConfigurationError(f"{name} has a zero quaternion")
    return pose


def _is_todo(value: Any) -> bool:
    return isinstance(value, str) and value.strip().upper().startswith("TODO")


@dataclass(frozen=True)
class RobotConfig:
    serial: str = "Rizon4s-063459"
    #: Empty (or "auto") derives ``/<serial with '-' -> '_'>/flexiv_robot_states``,
    #: the topic the Flexiv driver publishes for that serial.
    state_topic: str = ""
    command_topic: str = "/cartesian_motion_controller/command"
    world_frame: str = "world"
    policy_body: str = "flange"
    controller_body: str = "tcp"
    max_state_age_ms: float = 20.0

    def __post_init__(self) -> None:
        if str(self.state_topic).strip().lower() in ("", "auto"):
            namespace = str(self.serial).strip().replace("-", "_")
            object.__setattr__(
                self, "state_topic", f"/{namespace}/flexiv_robot_states"
            )


@dataclass(frozen=True)
class PolicyConfig:
    checkpoint: str = ""
    agent_yaml: Optional[str] = None
    env_yaml: Optional[str] = None
    device: str = "cuda:0"
    control_rate_hz: float = 30.0
    action_scale_multiplier: float = 1.0
    stochastic: bool = True
    torch_seed: int = 0
    #: Why the policy runs off its trained rate (env.sim.dt * env.decimation).
    #: Empty forbids a mismatch; the reason is recorded with the data.
    control_rate_mismatch_reason: str = ""

    @property
    def deterministic_inference(self) -> bool:
        """True when the distribution mean, not the sample, drives the action."""

        return not bool(self.stochastic)


@dataclass(frozen=True)
class SocketConfig:
    #: Pose of the socket opening (where the plug enters; the policy's socket
    #: keypoint) in the world frame, [x, y, z, qx, qy, qz, qw]. The socket root
    #: used internally is derived from it by removing the checkpoint's socket
    #: keypoint offset.
    opening_pose_world_xyzw: tuple[float, ...] = (
        0.473,
        0.121,
        0.071,
        0.5,
        0.5,
        0.5,
        -0.5,
    )
    source: str = "parameter"


@dataclass(frozen=True)
class PoseNoiseConfig:
    xy_half_range_m: tuple[float, float] = (0.005, 0.005)
    z_half_range_m: float = 0.0
    roll_half_range_rad: float = 0.0
    pitch_half_range_rad: float = 0.0
    yaw_half_range_rad: float = 0.0


@dataclass(frozen=True)
class RandomizationConfig:
    numpy_seed: int = 0
    start: PoseNoiseConfig = field(default_factory=PoseNoiseConfig)
    socket_policy_observation: PoseNoiseConfig = field(default_factory=PoseNoiseConfig)


@dataclass(frozen=True)
class SuccessConfig:
    #: One of ``SUPPORTED_SUCCESS_METRICS``; defaults to the flange-to-socket-root
    #: distance, whose threshold includes the full tool offset.
    metric: str = DEFAULT_SUCCESS_METRIC
    #: Scalar L2 radius; required by and only valid for ``*_position_l2`` metrics.
    distance_threshold_m: Optional[float] = None
    #: Per-axis ``(x, y, z)`` half-widths in metres, world axes; required by and
    #: only valid for ``eef_keypoint_to_socket_keypoint_axis_tolerances``. See the
    #: frame caveat in :func:`trial_state_machine.evaluate_success`.
    axis_tolerances_m: Optional[tuple[float, float, float]] = None
    hold_steps: int = 1
    #: Extra steps to run and record after success fires; the geometric check is
    #: met mid-insertion, so 0 ends the trial before the plug is fully seated.
    post_success_steps: int = 0
    require_operator_confirmation: bool = True


@dataclass(frozen=True)
class CameraConfig:
    name: str
    enabled: bool = True
    image_topic: str = "TODO"
    camera_info_topic: str = "TODO"
    serial: str = "TODO"
    frame_id: str = "TODO"
    # Identity/version for the serial, stream profile, and CameraInfo intrinsics.
    # "auto" (or empty) derives realsense-camera-info-<serial>-<W>x<H>-<encoding>-v1.
    calibration_id: str = "auto"
    # Optional provenance only.  Phase 1 does not transform camera data into a
    # robot frame, so a flange-to-camera calibration is not required.
    extrinsics_file: Optional[str] = None
    encoding: str = "TODO"
    input_width: Optional[int] = None
    input_height: Optional[int] = None
    fps: Optional[float] = None
    crop_xywh: Optional[tuple[int, int, int, int]] = None
    output_width: int = 128
    output_height: int = 128
    qos: str = "TODO"
    max_age_ms: float = 50.0

    def __post_init__(self) -> None:
        if str(self.calibration_id).strip().lower() not in ("", "auto"):
            return
        parts = (self.serial, self.input_width, self.input_height, self.encoding)
        if any(value is None or not str(value).strip() or _is_todo(value) for value in parts):
            derived = "TODO"
        else:
            derived = (
                f"realsense-camera-info-{str(self.serial).strip()}-"
                f"{self.input_width}x{self.input_height}-{str(self.encoding).strip()}-v1"
            )
        object.__setattr__(self, "calibration_id", derived)

    @property
    def has_placeholders(self) -> bool:
        if not self.enabled:
            return False
        required = (
            self.image_topic,
            self.camera_info_topic,
            self.serial,
            self.frame_id,
            self.calibration_id,
            self.encoding,
            self.qos,
        )
        extrinsics_unresolved = self.extrinsics_file is not None and (
            not str(self.extrinsics_file).strip() or _is_todo(self.extrinsics_file)
        )
        return (
            any(
                value is None or not str(value).strip() or _is_todo(value)
                for value in required
            )
            or extrinsics_unresolved
            or self.input_width is None
            or self.input_height is None
            or self.fps is None
            or self.crop_xywh is None
        )


@dataclass(frozen=True)
class RecordingConfig:
    output_directory: str = "data/phase1"
    target_successes: int = 20
    trial_timeout_s: float = 10.0
    writer_queue_size: int = 128
    flush_every_steps: int = 1
    image_compression: str = "lzf"
    #: Also record each frame as captured under
    #: observations/extra/<camera>_source_rgb, alongside the cropped/resized one.
    record_source_frames: bool = False
    preserve_all_attempts: bool = True


@dataclass(frozen=True)
class SynchronizationConfig:
    max_camera_pair_skew_ms: Optional[float] = None
    #: librealsense timestamp domain every camera must use. ``auto`` can fall back
    #: to the ROS clock per camera, mixing domains and breaking the pair-skew
    #: check; naming a domain makes the driver fail instead.
    camera_timestamp_source: str = "auto"


@dataclass(frozen=True)
class TrialControlConfig:
    """Reset/reposition values for the mounted hardware."""

    retraction_offset_world_m: Optional[tuple[float, float, float]] = None
    #: Lift straight out by this much from the current pose before any lateral
    #: motion. None uses retraction_offset_world_m alone, which is anchored to the
    #: reset pose and can drag a seated plug sideways (risking a torque limit).
    retraction_clearance_m: Optional[float] = None
    #: Level the tool onto the socket axis, pivoting about the plug tip, before
    #: lifting, so a tilted plug is not levered on the connector. Requires
    #: retraction_clearance_m.
    level_before_retract: bool = False
    reposition_duration_s: Optional[float] = None
    reposition_timeout_s: Optional[float] = None
    settle_time_s: Optional[float] = None
    input_hold_timeout_s: Optional[float] = None
    #: Abort when the tool keypoint is this far laterally (world xy) from the
    #: socket keypoint; None disables the check.
    max_lateral_offset_m: Optional[float] = None
    position_tolerance_m: Optional[float] = None
    orientation_tolerance_rad: Optional[float] = None
    stable_steps: Optional[int] = None

    #: Fields for which ``None`` is a valid setting, not a placeholder.
    OPTIONAL_FIELDS = ("max_lateral_offset_m", "retraction_clearance_m")

    @property
    def has_placeholders(self) -> bool:
        return any(
            value is None
            for name, value in asdict(self).items()
            if name not in self.OPTIONAL_FIELDS
        )


@dataclass(frozen=True)
class RuntimeConfig:
    execution_mode: str = "shadow"
    require_start_confirmation: bool = True
    publish_goal_pose: bool = True
    goal_pose_topic: str = "/goal_pose"
    diagnostics_topic: str = "/sparr/phase1/diagnostics"


@dataclass(frozen=True)
class CollectionConfig:
    robot: RobotConfig = field(default_factory=RobotConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    socket: SocketConfig = field(default_factory=SocketConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    success: SuccessConfig = field(default_factory=SuccessConfig)
    cameras: tuple[CameraConfig, CameraConfig] = (
        CameraConfig(name="wrist_1"),
        CameraConfig(name="wrist_2"),
    )
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    synchronization: SynchronizationConfig = field(default_factory=SynchronizationConfig)
    trial_control: TrialControlConfig = field(default_factory=TrialControlConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @property
    def enabled_cameras(self) -> tuple[CameraConfig, ...]:
        """Return configured camera streams in their canonical wrist order."""

        return tuple(camera for camera in self.cameras if camera.enabled is True)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CollectionConfig":
        data = _mapping(data, "root")
        robot = RobotConfig(**dict(_mapping(data.get("robot", {}), "robot")))
        policy = PolicyConfig(**dict(_mapping(data.get("policy", {}), "policy")))

        socket_data = dict(_mapping(data.get("socket", {}), "socket"))
        if "root_pose_world_xyzw" in socket_data:
            raise ConfigurationError(
                "socket.root_pose_world_xyzw is no longer supported; give the pose "
                "of the socket opening as socket.opening_pose_world_xyzw "
                "(the root pose plus the socket keypoint offset, e.g. z 0.0335 -> 0.071)"
            )
        if "opening_pose_world_xyzw" in socket_data:
            socket_data["opening_pose_world_xyzw"] = _pose7(
                socket_data["opening_pose_world_xyzw"],
                "socket.opening_pose_world_xyzw",
            )
        socket = SocketConfig(**socket_data)

        random_data = _mapping(data.get("randomization", {}), "randomization")
        # Only per-trial streams are supported. Tolerate the key, but never
        # silently run a different stream than the file asks for.
        stream = random_data.get("socket_noise_stream", "per_trial")
        if stream != "per_trial":
            raise ConfigurationError(
                f"randomization.socket_noise_stream={stream!r} is no longer "
                "supported; remove the key (every trial uses its own stream)"
            )
        start_data = dict(_mapping(random_data.get("start", {}), "randomization.start"))
        socket_noise_data = dict(
            _mapping(
                random_data.get("socket_policy_observation", {}),
                "randomization.socket_policy_observation",
            )
        )
        if "xy_half_range_m" in start_data:
            start_data["xy_half_range_m"] = _pair(
                start_data["xy_half_range_m"], "randomization.start.xy_half_range_m"
            )
        if "xy_half_range_m" in socket_noise_data:
            socket_noise_data["xy_half_range_m"] = _pair(
                socket_noise_data["xy_half_range_m"],
                "randomization.socket_policy_observation.xy_half_range_m",
            )
        randomization = RandomizationConfig(
            numpy_seed=int(random_data.get("numpy_seed", 0)),
            start=PoseNoiseConfig(**start_data),
            socket_policy_observation=PoseNoiseConfig(**socket_noise_data),
        )

        success_data = dict(_mapping(data.get("success", {}), "success"))
        tolerances = success_data.get("axis_tolerances_m")
        if tolerances is not None:
            try:
                success_data["axis_tolerances_m"] = validate_axis_tolerances(
                    tolerances, "success.axis_tolerances_m"
                )
            except TrialConfigurationError as exc:
                raise ConfigurationError(str(exc)) from exc
        success = SuccessConfig(**success_data)

        cameras_data = _mapping(data.get("cameras", {}), "cameras")
        cameras = []
        for name in ("wrist_1", "wrist_2"):
            camera_data = dict(_mapping(cameras_data.get(name, {}), f"cameras.{name}"))
            for field_name in (
                "image_topic",
                "camera_info_topic",
                "serial",
                "frame_id",
                "calibration_id",
                "extrinsics_file",
                "encoding",
                "qos",
            ):
                if field_name in camera_data and camera_data[field_name] is not None:
                    camera_data[field_name] = str(camera_data[field_name]).strip()
            crop = camera_data.get("crop_xywh")
            if crop is not None:
                if not isinstance(crop, (list, tuple)) or len(crop) != 4:
                    raise ConfigurationError(f"cameras.{name}.crop_xywh must be [x,y,w,h]")
                camera_data["crop_xywh"] = tuple(int(v) for v in crop)
            cameras.append(CameraConfig(name=name, **camera_data))

        recording = RecordingConfig(
            **dict(_mapping(data.get("recording", {}), "recording"))
        )
        synchronization = SynchronizationConfig(
            **dict(_mapping(data.get("synchronization", {}), "synchronization"))
        )
        trial_control_data = dict(
            _mapping(data.get("trial_control", {}), "trial_control")
        )
        retraction = trial_control_data.get("retraction_offset_world_m")
        if retraction is not None:
            trial_control_data["retraction_offset_world_m"] = _triple(
                retraction, "trial_control.retraction_offset_world_m"
            )
        trial_control = TrialControlConfig(**trial_control_data)
        runtime = RuntimeConfig(**dict(_mapping(data.get("runtime", {}), "runtime")))
        config = cls(
            robot=robot,
            policy=policy,
            socket=socket,
            randomization=randomization,
            success=success,
            cameras=tuple(cameras),
            recording=recording,
            synchronization=synchronization,
            trial_control=trial_control,
            runtime=runtime,
        )
        config.validate(live=runtime.execution_mode == "live")
        return config

    @classmethod
    def load(cls, path: str | Path) -> "CollectionConfig":
        config_path = Path(path).expanduser().resolve()
        try:
            with config_path.open("r", encoding="utf-8") as stream:
                data = yaml.load(stream, Loader=_UniqueKeyLoader)
        except OSError as exc:
            raise ConfigurationError(f"cannot read configuration {config_path}: {exc}") from exc
        except yaml.YAMLError as exc:
            raise ConfigurationError(
                f"invalid configuration YAML in {config_path}: {exc}"
            ) from exc
        try:
            return cls.from_mapping(data or {})
        except ConfigurationError:
            raise
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigurationError(
                f"invalid value in configuration {config_path}: {exc}"
            ) from exc

    def validate(self, *, live: bool) -> None:
        errors: list[str] = []
        serial = str(self.robot.serial).strip()
        if not serial or _is_todo(serial) or any(ch.isspace() for ch in serial):
            errors.append("robot.serial must be set to your robot's serial, e.g. 'Rizon4s-012345'")
        if not str(self.robot.state_topic).strip().startswith("/"):
            errors.append("robot.state_topic must be an absolute topic name")
        if self.robot.command_topic != "/cartesian_motion_controller/command":
            errors.append("robot.command_topic must be '/cartesian_motion_controller/command'")
        if self.robot.world_frame != "world":
            errors.append("robot.world_frame must be 'world'")
        if self.robot.policy_body != "flange" or self.robot.controller_body != "tcp":
            errors.append("approved body contract is policy_body=flange, controller_body=tcp")
        if not math.isfinite(self.robot.max_state_age_ms) or self.robot.max_state_age_ms <= 0.0:
            errors.append("robot.max_state_age_ms must be finite and positive")
        if not math.isfinite(self.policy.control_rate_hz) or self.policy.control_rate_hz <= 0.0:
            errors.append("policy.control_rate_hz must be positive")
        if (
            not math.isfinite(self.policy.action_scale_multiplier)
            or self.policy.action_scale_multiplier <= 0.0
        ):
            errors.append("policy.action_scale_multiplier must be finite and positive")
        # policy.stochastic=false (mean-action debugging) is allowed; its trials
        # are recorded but never export-eligible, and preflight warns.
        try:
            socket_pose = tuple(float(value) for value in self.socket.opening_pose_world_xyzw)
        except (TypeError, ValueError, OverflowError):
            socket_pose = ()
        if len(socket_pose) != 7 or any(not math.isfinite(value) for value in socket_pose):
            errors.append(
                "socket.opening_pose_world_xyzw must be a finite "
                "[x,y,z,qx,qy,qz,qw] pose"
            )
        elif sum(value * value for value in socket_pose[3:]) <= 1.0e-12:
            errors.append("socket.opening_pose_world_xyzw has a zero quaternion")
        if self.success.metric not in SUPPORTED_SUCCESS_METRICS:
            errors.append(
                f"unsupported success metric {self.success.metric!r}; "
                f"expected one of {', '.join(SUPPORTED_SUCCESS_METRICS)}"
            )
        if (
            isinstance(self.success.hold_steps, bool)
            or not isinstance(self.success.hold_steps, int)
            or self.success.hold_steps < 1
        ):
            errors.append("success.hold_steps must be a positive integer")
        if (
            isinstance(self.success.post_success_steps, bool)
            or not isinstance(self.success.post_success_steps, int)
            or self.success.post_success_steps < 0
        ):
            errors.append("success.post_success_steps must be a non-negative integer")
        if self.success.distance_threshold_m is not None and (
            not math.isfinite(self.success.distance_threshold_m)
            or self.success.distance_threshold_m <= 0.0
        ):
            errors.append("success.distance_threshold_m must be finite and positive")
        # A metric uses either a scalar L2 radius or a per-axis box; supplying
        # the wrong one is an error, not an ignored field.
        if self.success.axis_tolerances_m is not None:
            try:
                validate_axis_tolerances(
                    self.success.axis_tolerances_m, "success.axis_tolerances_m"
                )
            except TrialConfigurationError as exc:
                errors.append(str(exc))
        if self.success.metric in AXIS_TOLERANCE_SUCCESS_METRICS:
            if self.success.axis_tolerances_m is None:
                errors.append(
                    f"success.axis_tolerances_m is required for success metric "
                    f"{self.success.metric!r}"
                )
            if self.success.distance_threshold_m is not None:
                errors.append(
                    f"success.distance_threshold_m is meaningless for success "
                    f"metric {self.success.metric!r}; remove it or set it to null"
                )
        elif (
            self.success.metric in SUPPORTED_SUCCESS_METRICS
            and self.success.axis_tolerances_m is not None
        ):
            errors.append(
                f"success.axis_tolerances_m is meaningless for success metric "
                f"{self.success.metric!r}; remove it or set it to null"
            )
        if (
            isinstance(self.recording.target_successes, bool)
            or not isinstance(self.recording.target_successes, int)
            or self.recording.target_successes < 1
        ):
            errors.append("recording.target_successes must be a positive integer")
        if (
            isinstance(self.recording.writer_queue_size, bool)
            or not isinstance(self.recording.writer_queue_size, int)
            or self.recording.writer_queue_size < 1
        ):
            errors.append("recording.writer_queue_size must be a positive integer")
        if (
            not math.isfinite(self.recording.trial_timeout_s)
            or self.recording.trial_timeout_s <= 0.0
        ):
            errors.append("recording.trial_timeout_s must be finite and positive")
        if (
            isinstance(self.recording.flush_every_steps, bool)
            or not isinstance(self.recording.flush_every_steps, int)
            or self.recording.flush_every_steps < 1
        ):
            errors.append("recording.flush_every_steps must be a positive integer")
        if self.recording.image_compression != "lzf":
            errors.append("recording.image_compression must be 'lzf'")
        if not self.recording.preserve_all_attempts:
            errors.append("Phase 1 requires recording.preserve_all_attempts=true")
        if not str(self.recording.output_directory).strip():
            errors.append("recording.output_directory must be non-empty")
        if self.runtime.execution_mode not in {"shadow", "live"}:
            errors.append("runtime.execution_mode must be shadow or live")

        for label, noise in (
            ("start", self.randomization.start),
            ("socket_policy_observation", self.randomization.socket_policy_observation),
        ):
            values = (*noise.xy_half_range_m, noise.z_half_range_m,
                      noise.roll_half_range_rad, noise.pitch_half_range_rad,
                      noise.yaw_half_range_rad)
            if any(not math.isfinite(value) or value < 0.0 for value in values):
                errors.append(
                    f"randomization.{label} half-ranges must be finite and non-negative"
                )

        if tuple(camera.name for camera in self.cameras) != ("wrist_1", "wrist_2"):
            errors.append("cameras must contain wrist_1 then wrist_2")
        for camera in self.cameras:
            if not isinstance(camera.enabled, bool):
                errors.append(f"camera {camera.name!r} enabled must be a boolean")
        if self.cameras and self.cameras[0].enabled is not True:
            errors.append("camera 'wrist_1' must be enabled")

        enabled_cameras = self.enabled_cameras
        image_topics = [camera.image_topic for camera in enabled_cameras]
        info_topics = [camera.camera_info_topic for camera in enabled_cameras]
        serials = [camera.serial for camera in enabled_cameras]
        frame_ids = [camera.frame_id for camera in enabled_cameras]
        calibration_ids = [camera.calibration_id for camera in enabled_cameras]
        for label, values in (
            ("image topics", image_topics),
            ("CameraInfo topics", info_topics),
            ("camera serials", serials),
            ("camera frame IDs", frame_ids),
            ("camera calibration IDs", calibration_ids),
        ):
            concrete = [
                value for value in values if str(value).strip() and not _is_todo(value)
            ]
            if len(set(concrete)) != len(concrete):
                errors.append(f"{label} must be unique across enabled wrist cameras")
        for camera in enabled_cameras:
            encoding = "" if camera.encoding is None else str(camera.encoding).lower()
            qos = "" if camera.qos is None else str(camera.qos).lower()
            if not _is_todo(camera.encoding) and encoding not in {
                "rgb8",
                "bgr8",
            }:
                errors.append(
                    f"camera {camera.name!r} encoding must be 'rgb8' or 'bgr8'"
                )
            if not _is_todo(camera.qos) and qos not in {
                "sensor_data",
                "best_effort",
                "reliable",
            }:
                errors.append(
                    f"camera {camera.name!r} qos must be sensor_data, best_effort, "
                    "or reliable"
                )
            for name, value in (
                ("input_width", camera.input_width),
                ("input_height", camera.input_height),
                ("output_width", camera.output_width),
                ("output_height", camera.output_height),
            ):
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                ):
                    errors.append(f"camera {camera.name!r} {name} must be a positive integer")
            if camera.fps is not None and (
                not math.isfinite(camera.fps) or camera.fps <= 0.0
            ):
                errors.append(f"camera {camera.name!r} fps must be finite and positive")
            if not math.isfinite(camera.max_age_ms) or camera.max_age_ms <= 0.0:
                errors.append(f"camera {camera.name!r} max_age_ms must be finite and positive")
            # Any stored size is allowed; preflight warns when it is not 128x128.
            if (
                camera.crop_xywh is not None
                and camera.input_width is not None
                and camera.input_height is not None
            ):
                x, y, width, height = camera.crop_xywh
                if (
                    x < 0
                    or y < 0
                    or width <= 0
                    or height <= 0
                    or x + width > camera.input_width
                    or y + height > camera.input_height
                ):
                    errors.append(
                        f"camera {camera.name!r} crop lies outside its input frame"
                    )

        pair_skew = self.synchronization.max_camera_pair_skew_ms
        if pair_skew is not None and (
            not math.isfinite(pair_skew) or pair_skew <= 0.0
        ):
            errors.append(
                "synchronization.max_camera_pair_skew_ms must be finite and positive"
            )

        if live:
            if not self.policy.checkpoint:
                errors.append("policy.checkpoint is required in live mode")
            # Box metrics use success.axis_tolerances_m instead of a threshold.
            if self.success.metric not in AXIS_TOLERANCE_SUCCESS_METRICS:
                if self.success.distance_threshold_m is None:
                    errors.append("success.distance_threshold_m is still TODO")
                elif self.success.distance_threshold_m <= 0.0:
                    errors.append("success.distance_threshold_m must be positive")
            for camera in enabled_cameras:
                if camera.has_placeholders:
                    errors.append(f"camera {camera.name!r} still contains TODO fields")
            if len(enabled_cameras) > 1 and pair_skew is None:
                errors.append("synchronization.max_camera_pair_skew_ms is still TODO")
            if self.trial_control.has_placeholders:
                errors.append("trial_control still contains TODO fields")

        control = self.trial_control
        if control.retraction_offset_world_m is not None and any(
            not isinstance(value, (int, float)) for value in control.retraction_offset_world_m
        ):
            errors.append("trial_control.retraction_offset_world_m must be numeric")
        for name, value in (
            ("reposition_duration_s", control.reposition_duration_s),
            ("reposition_timeout_s", control.reposition_timeout_s),
            ("settle_time_s", control.settle_time_s),
            ("input_hold_timeout_s", control.input_hold_timeout_s),
            ("position_tolerance_m", control.position_tolerance_m),
            ("orientation_tolerance_rad", control.orientation_tolerance_rad),
            ("max_lateral_offset_m", control.max_lateral_offset_m),
            ("retraction_clearance_m", control.retraction_clearance_m),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                errors.append(f"trial_control.{name} must be finite and positive")
        if control.stable_steps is not None and (
            isinstance(control.stable_steps, bool)
            or not isinstance(control.stable_steps, int)
            or control.stable_steps < 1
        ):
            errors.append("trial_control.stable_steps must be a positive integer")

        if errors:
            raise ConfigurationError("invalid collection configuration:\n- " + "\n- ".join(errors))

    def resolved_mapping(self) -> dict[str, Any]:
        """Return a serialization-safe snapshot for HDF5 provenance."""
        return asdict(self)
