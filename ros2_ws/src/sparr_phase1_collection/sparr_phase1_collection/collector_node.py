"""ROS 2 node for SPARR Phase 1 Flexiv data collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping, Optional

from cv_bridge import CvBridge
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from flexiv_msgs.msg import CartesianMotionForceCommand, RobotStates
from geometry_msgs.msg import PoseStamped
import numpy as np
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger

from .action_pipeline import ActionPipelineConfig, ActionTrace, build_action_trace
from .async_writer import (
    AsyncHDF5SessionWriter,
    WriterBackpressureError,
    WriterWorkerError,
)
from .camera_buffer import CameraBuffer, CameraFrameSet, CropRegion
from .camera_validation import camera_info_errors
from .checkpoint import build_phase1_policy_spec, load_yaml_mapping
from .collector_logic import (
    RolloutInputDisposition,
    TerminalObservationPlan,
    classify_rollout_input,
    plan_terminal_observation,
    policy_goal_publish_enabled,
    trial_export_eligible,
)
from .configuration import CameraConfig, CollectionConfig, ConfigurationError
from .motion import TimedPoseTrajectory, offset_position_world, pose_error
from .observations import (
    ObservationConfig,
    encode_phase1_observation,
    offset_point_world,
    socket_root_from_opening,
)
from .pose_math import (
    Pose,
    compose_poses,
    invert_pose,
    quaternion_to_matrix_xyzw,
)
from .provenance import (
    artifact_provenance,
    build_session_provenance,
    find_repository_root,
    resolve_artifact_path,
    resolve_output_directory,
)
from .records import build_action_record, build_observation_record
from .robot_state import RobotStateAdapterError, RobotStateSnapshot, adapt_robot_states
from .rsl_rl_runtime import load_rsl_rl_policy
from .safety import GuardDecision, GuardStatus, RobotStateGuard, StateGuardConfig
from .trial_state_machine import (
    AXIS_TOLERANCE_SUCCESS_METRICS,
    PoseNoise,
    TrialEvent,
    TrialRandomizer,
    TrialState,
    TrialStateMachine,
    TrialTargets,
)


GUARD_ROBOT_INVALID = 1 << 0
GUARD_ROBOT_STALE = 1 << 1
GUARD_CAMERA_AGE = 1 << 2
GUARD_CAMERA_SKEW = 1 << 3
GUARD_POLICY = 1 << 4
GUARD_COMMAND_PUBLISH = 1 << 5
GUARD_WRITER = 1 << 6


@dataclass(frozen=True)
class _InputBundle:
    snapshot: RobotStateSnapshot
    receive_monotonic_ns: int
    cameras: CameraFrameSet
    guard: GuardDecision
    camera_validity: Mapping[str, bool]
    camera_pair_valid: bool
    policy_input: np.ndarray
    policy_input_valid: bool
    record: Mapping[str, Any]
    guard_bitmask: int


def _camera_qos(config: CameraConfig) -> QoSProfile:
    normalized = config.qos.strip().lower()
    if normalized == "sensor_data":
        return qos_profile_sensor_data
    reliability = (
        ReliabilityPolicy.BEST_EFFORT
        if normalized == "best_effort"
        else ReliabilityPolicy.RELIABLE
    )
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
        reliability=reliability,
        durability=DurabilityPolicy.VOLATILE,
    )


def _set_pose_message(target: Any, pose: Pose) -> None:
    target.position.x = float(pose.position[0])
    target.position.y = float(pose.position[1])
    target.position.z = float(pose.position[2])
    target.orientation.x = float(pose.quaternion_xyzw[0])
    target.orientation.y = float(pose.quaternion_xyzw[1])
    target.orientation.z = float(pose.quaternion_xyzw[2])
    target.orientation.w = float(pose.quaternion_xyzw[3])


def _camera_info_record(message: CameraInfo, config: CameraConfig) -> dict[str, Any]:
    roi = message.roi
    configured_extrinsics = config.extrinsics_file
    if configured_extrinsics is None:
        extrinsics_status = "not_provided"
    else:
        extrinsics_status = "configured_unverified"
    return {
        "serial": config.serial,
        "configured_frame_id": config.frame_id,
        "calibration_id": config.calibration_id,
        "extrinsics_file": configured_extrinsics,
        "extrinsics_available": False,
        "extrinsics_used_by_phase1": False,
        "extrinsics_status": extrinsics_status,
        "message_frame_id": message.header.frame_id,
        "width": int(message.width),
        "height": int(message.height),
        "distortion_model": message.distortion_model,
        "d": list(message.d),
        "k": list(message.k),
        "r": list(message.r),
        "p": list(message.p),
        "binning_x": int(message.binning_x),
        "binning_y": int(message.binning_y),
        "roi": {
            "x_offset": int(roi.x_offset),
            "y_offset": int(roi.y_offset),
            "height": int(roi.height),
            "width": int(roi.width),
            "do_rectify": bool(roi.do_rectify),
        },
        "crop_xywh": list(config.crop_xywh or ()),
        "stored_size": [config.output_width, config.output_height],
    }


def _matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    """Rotation matrix -> (x, y, z, w). Branches on the largest diagonal term
    so the divisor is never near zero, which the naive trace form is."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (m[2, 1] - m[1, 2]) / scale
        y = (m[0, 2] - m[2, 0]) / scale
        z = (m[1, 0] - m[0, 1]) / scale
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        scale = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / scale
        x = 0.25 * scale
        y = (m[0, 1] + m[1, 0]) / scale
        z = (m[0, 2] + m[2, 0]) / scale
    elif m[1, 1] > m[2, 2]:
        scale = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / scale
        x = (m[0, 1] + m[1, 0]) / scale
        y = 0.25 * scale
        z = (m[1, 2] + m[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / scale
        x = (m[0, 2] + m[2, 0]) / scale
        y = (m[1, 2] + m[2, 1]) / scale
        z = 0.25 * scale
    quaternion = np.array([x, y, z, w], dtype=np.float64)
    return quaternion / float(np.linalg.norm(quaternion))


class Phase1CollectorNode(Node):
    """Single owner of policy state, trial state, commands, and recording."""

    def __init__(self) -> None:
        super().__init__("sparr_phase1")
        self.declare_parameter("config_path", "")
        self.declare_parameter("execution_mode_override", "")
        self.declare_parameter("repository_root", "")
        # Randomization index to resume from. A ROS parameter, not config, so a
        # stale value cannot silently offset later sessions.
        self.declare_parameter("first_attempt_index", 0)
        # Write this session somewhere other than recording.output_directory,
        # e.g. one folder per evaluated checkpoint. Empty keeps the config's.
        self.declare_parameter("output_directory_override", "")

        config_value = str(self.get_parameter("config_path").value)
        if not config_value:
            raise ConfigurationError("config_path ROS parameter is required")
        self._config_path = Path(config_value).expanduser().resolve()
        config = CollectionConfig.load(self._config_path)
        mode_override = str(self.get_parameter("execution_mode_override").value).strip()
        if mode_override:
            config = replace(
                config,
                runtime=replace(config.runtime, execution_mode=mode_override),
            )
        output_override = str(
            self.get_parameter("output_directory_override").value
        ).strip()
        if output_override:
            config = replace(
                config,
                recording=replace(config.recording, output_directory=output_override),
            )

        checkpoint = resolve_artifact_path(
            config.policy.checkpoint, config_path=self._config_path
        )
        agent_yaml = resolve_artifact_path(
            config.policy.agent_yaml or "", config_path=self._config_path
        )
        env_yaml = resolve_artifact_path(
            config.policy.env_yaml or "", config_path=self._config_path
        )
        config = replace(
            config,
            policy=replace(
                config.policy,
                checkpoint=str(checkpoint),
                agent_yaml=str(agent_yaml),
                env_yaml=str(env_yaml),
            ),
        )
        config.validate(live=config.runtime.execution_mode == "live")
        for label, path in (
            ("checkpoint", checkpoint),
            ("agent_yaml", agent_yaml),
            ("env_yaml", env_yaml),
        ):
            if not path.is_file():
                raise ConfigurationError(f"{label} does not exist: {path}")
        self._config = config
        self._mode = config.runtime.execution_mode
        # Debug-only mean-action mode: one act() per step and mu/std/sample are
        # still recorded, but trials are never export-eligible.
        self._deterministic_inference = config.policy.deterministic_inference
        self._repository_root = find_repository_root(
            self._config_path, str(self.get_parameter("repository_root").value)
        )

        self._policy_spec = build_phase1_policy_spec(
            load_yaml_mapping(agent_yaml), load_yaml_mapping(env_yaml)
        )
        if not math.isclose(
            config.policy.control_rate_hz,
            self._policy_spec.control_rate_hz,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            # Running off the trained rate requires a stated reason; rechecked
            # here because the node can start without preflight.
            reason = str(config.policy.control_rate_mismatch_reason or "").strip()
            if not reason:
                raise ConfigurationError(
                    "configured policy rate does not match checkpoint: "
                    f"{config.policy.control_rate_hz} vs {self._policy_spec.control_rate_hz}"
                    "; set policy.control_rate_mismatch_reason to run deliberately "
                    "off the trained rate"
                )
            self.get_logger().warning(
                f"RUNNING OFF THE TRAINED CONTROL RATE: configured "
                f"{config.policy.control_rate_hz} Hz vs checkpoint "
                f"{self._policy_spec.control_rate_hz} Hz. The recurrent policy's "
                f"state advances at a different rate relative to the world, which "
                f"action rescaling does not compensate. Reason: {reason}"
            )
        self._observation_config = ObservationConfig.from_policy_spec(self._policy_spec)
        self._action_config = ActionPipelineConfig.from_policy_spec(self._policy_spec)
        self._policy = load_rsl_rl_policy(
            checkpoint,
            agent_yaml=agent_yaml,
            env_yaml=env_yaml,
            device=config.policy.device,
            seed=config.policy.torch_seed,
        )

        self._exact_socket_root = self._configured_socket_root()
        self._randomizer = TrialRandomizer(
            config.randomization.numpy_seed,
            start_noise=PoseNoise(**asdict(config.randomization.start)),
            socket_policy_noise=PoseNoise(
                **asdict(config.randomization.socket_policy_observation)
            ),
        )
        self._state_guard = RobotStateGuard(
            StateGuardConfig(
                max_state_age_ns=int(config.robot.max_state_age_ms * 1.0e6),
                max_future_skew_ns=int(config.robot.max_state_age_ms * 1.0e6),
                expected_pose_frame_id=config.robot.world_frame,
            )
        )

        self._enabled_cameras = config.enabled_cameras
        self._enabled_camera_names = tuple(
            camera.name for camera in self._enabled_cameras
        )
        camera_crops = {
            camera.name: CropRegion(*camera.crop_xywh)
            for camera in self._enabled_cameras
            if camera.crop_xywh is not None
        }
        self._cameras_configured = all(
            not camera.has_placeholders for camera in self._enabled_cameras
        )
        self._camera_buffer: CameraBuffer | None = None
        if self._cameras_configured:
            output_sizes = {
                (camera.output_height, camera.output_width)
                for camera in self._enabled_cameras
            }
            # CameraBuffer takes a single output size shared by every camera.
            if len(output_sizes) != 1:
                raise ConfigurationError(
                    "enabled cameras must share one stored output size, got "
                    f"{sorted(output_sizes)}"
                )
            self._camera_buffer = CameraBuffer(
                self._enabled_camera_names,
                # Must match the writer's image_shape (_stored_image_shape).
                next(iter(output_sizes)),
                crops=camera_crops,
                capacity=max(
                    120,
                    int(max(camera.fps or 30.0 for camera in self._enabled_cameras) * 4),
                ),
                retain_source=bool(config.recording.record_source_frames),
            )

        self._bridge = CvBridge()
        self._lock = threading.RLock()
        self._latest_snapshot: RobotStateSnapshot | None = None
        self._latest_receive_monotonic_ns = -1
        self._camera_info: dict[str, dict[str, Any]] = {}
        self._nominal_reset_flange: Pose | None = None
        self._flange_to_tcp: Pose | None = None
        self._targets: TrialTargets | None = None
        self._state_machine = TrialStateMachine()
        self._trajectory: TimedPoseTrajectory | None = None
        #: Remaining retraction waypoints, consumed inside RETRACTING.
        self._retraction_queue: list[tuple[Pose, str]] = []
        self._trajectory_deadline_ns = -1
        self._segment_stable_steps = 0
        self._settle_deadline_ns = -1
        self._rollout_start_ns = -1
        self._input_hold_start_ns = -1
        self._success_hold_steps = 0
        self._hold_tick_count = 0
        self._hold_guard_bits = 0
        self._post_success_remaining: int | None = None
        self._success_reason = ""
        self._last_frame_stamp_ns: dict[str, int] = {}
        self._repeated_frame_counts: dict[str, int] = {}
        self._terminal_reason = ""
        self._auto_outcome = "unknown"
        self._trial_export_eligible = False
        self._trial_observation_count = 0
        self._trial_action_count = 0
        first_attempt = int(self.get_parameter("first_attempt_index").value)
        if first_attempt < 0:
            raise ConfigurationError(
                f"first_attempt_index must not be negative, got {first_attempt}"
            )
        self._attempt_index = first_attempt
        #: When set, the next attempt replays these targets at the same attempt
        #: index. Never set in Phase 1; used by subclasses to retry an attempt.
        self._repeat_targets: TrialTargets | None = None
        # A trial rejected before its rollout began (start pose not reached,
        # settle/hold timeout) is rerun at the same index rather than skipped.
        self._rollout_started = False
        self._rejected_before_rollout = False
        #: How many times the current attempt index has already been replayed.
        self._attempt_repeat_index = 0
        self._accepted_successes = 0
        self._writer: AsyncHDF5SessionWriter | None = None
        self._session_complete = False
        self._last_diagnostic_reason = "waiting for robot state"
        self._diagnostic_tick = 0

        state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._state_subscription = self.create_subscription(
            RobotStates,
            config.robot.state_topic,
            self._robot_state_callback,
            state_qos,
        )
        self._camera_subscriptions = []
        if self._cameras_configured:
            for camera in self._enabled_cameras:
                image_subscription = self.create_subscription(
                    Image,
                    camera.image_topic,
                    lambda message, name=camera.name: self._image_callback(name, message),
                    _camera_qos(camera),
                )
                info_subscription = self.create_subscription(
                    CameraInfo,
                    camera.camera_info_topic,
                    lambda message, item=camera: self._camera_info_callback(item, message),
                    _camera_qos(camera),
                )
                self._camera_subscriptions.extend(
                    (image_subscription, info_subscription)
                )
        else:
            self.get_logger().warning(
                "Enabled camera configuration contains TODO values; trials are disabled "
                "until resolved"
            )

        self._command_publisher = None
        if self._mode == "live":
            self._command_publisher = self.create_publisher(
                CartesianMotionForceCommand,
                config.robot.command_topic,
                command_qos,
            )
        self._goal_publisher = self.create_publisher(
            PoseStamped,
            config.runtime.goal_pose_topic,
            command_qos,
        )
        self._diagnostics_publisher = self.create_publisher(
            DiagnosticArray,
            config.runtime.diagnostics_topic,
            10,
        )
        self.create_service(Trigger, "~/teach_reset_pose", self._teach_reset_service)
        self.create_service(Trigger, "~/start_trial", self._start_trial_service)
        self.create_service(Trigger, "~/begin_rollout", self._begin_rollout_service)
        self.create_service(Trigger, "~/stop_trial", self._stop_trial_service)
        self.create_service(Trigger, "~/confirm_success", self._confirm_success_service)
        self.create_service(Trigger, "~/confirm_failure", self._confirm_failure_service)
        self.create_service(Trigger, "~/discard_trial", self._discard_trial_service)
        self.create_service(Trigger, "~/abort_session", self._abort_session_service)

        period = 1.0 / config.policy.control_rate_hz
        self._control_timer = self.create_timer(period, self._control_tick)
        # Report the rate the loop actually runs at, and the trained rate if
        # they differ.
        rate_detail = f"{config.policy.control_rate_hz:.1f} Hz"
        if not math.isclose(
            config.policy.control_rate_hz,
            self._policy_spec.control_rate_hz,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            rate_detail += f" (checkpoint trained at {self._policy_spec.control_rate_hz:.1f} Hz)"
        self.get_logger().info(
            f"Phase 1 collector ready in {self._mode!r} mode; "
            f"policy={rate_detail}, "
            f"clip=+-{self._policy_spec.training_action_clip}"
        )
        if self._attempt_index:
            # Loud: a wrong resume index silently repeats offsets an earlier
            # session already covered.
            self.get_logger().warning(
                f"RESUMING at attempt index {self._attempt_index}: randomization "
                f"continues from there, so trials 0..{self._attempt_index - 1} of "
                "this seed are NOT repeated. This session writes a new file; the "
                "earlier one is unchanged."
            )

    # ---- ROS callbacks -------------------------------------------------

    def _robot_state_callback(self, message: RobotStates) -> None:
        try:
            snapshot = adapt_robot_states(message)
        except RobotStateAdapterError as exc:
            self._last_diagnostic_reason = f"RobotStates rejected: {exc}"
            return
        with self._lock:
            self._latest_snapshot = snapshot
            self._latest_receive_monotonic_ns = time.monotonic_ns()

    def _image_callback(self, camera_name: str, message: Image) -> None:
        if self._camera_buffer is None:
            return
        config = next(
            camera for camera in self._enabled_cameras if camera.name == camera_name
        )
        try:
            if message.encoding.lower() != config.encoding.lower():
                raise ValueError(
                    f"encoding {message.encoding!r} does not match configured {config.encoding!r}"
                )
            if (
                int(message.width) != config.input_width
                or int(message.height) != config.input_height
            ):
                raise ValueError(
                    f"frame is {message.width}x{message.height}, configured "
                    f"{config.input_width}x{config.input_height}"
                )
            if message.header.frame_id != config.frame_id:
                raise ValueError(
                    f"frame_id {message.header.frame_id!r} does not match configured "
                    f"{config.frame_id!r}"
                )
            rgb = self._bridge.imgmsg_to_cv2(message, desired_encoding="rgb8")
            self._camera_buffer.add_frame(
                camera_name,
                message.header.stamp,
                rgb,
                metadata={
                    "frame_id": message.header.frame_id,
                    "encoding": message.encoding,
                    "step": int(message.step),
                    "is_bigendian": int(message.is_bigendian),
                },
            )
        except Exception as exc:
            self._last_diagnostic_reason = f"{camera_name} frame rejected: {exc}"

    def _camera_info_callback(self, config: CameraConfig, message: CameraInfo) -> None:
        errors = camera_info_errors(message, config)
        if errors:
            self._camera_info.pop(config.name, None)
            self._last_diagnostic_reason = (
                f"{config.name} CameraInfo rejected: " + "; ".join(errors)
            )
            return
        self._camera_info[config.name] = _camera_info_record(message, config)

    # ---- operator services --------------------------------------------

    def _teach_reset_service(self, _request: Trigger.Request, response: Trigger.Response):
        with self._lock:
            if self._state_machine.state is not TrialState.IDLE:
                return self._service_response(response, False, "cannot teach during a trial")
            snapshot = self._latest_snapshot
            if snapshot is None:
                return self._service_response(response, False, "no RobotStates received")
            errors = snapshot.validation_errors()
            if errors or snapshot.tcp_pose is None:
                detail = "; ".join(errors) or "tcp_pose is missing"
                return self._service_response(response, False, detail)
            now_ns = self.get_clock().now().nanoseconds
            age_ns = now_ns - snapshot.ros_timestamp_ns
            if abs(age_ns) > int(self._config.robot.max_state_age_ms * 1.0e6):
                return self._service_response(response, False, f"robot state age is {age_ns} ns")
            self._nominal_reset_flange = snapshot.anchor_pose
            self._flange_to_tcp = compose_poses(
                invert_pose(snapshot.anchor_pose), snapshot.tcp_pose
            )
            self._last_diagnostic_reason = "reset flange pose taught"
            return self._service_response(
                response,
                True,
                "captured taught reset flange pose and measured flange-to-TCP transform",
            )

    def _start_trial_service(self, _request: Trigger.Request, response: Trigger.Response):
        with self._lock:
            try:
                self._start_trial()
            except Exception as exc:
                self.get_logger().error(f"Cannot start trial: {exc}")
                return self._service_response(response, False, str(exc))
            return self._service_response(
                response,
                True,
                f"attempt {self._attempt_index} started",
            )

    def _begin_rollout_service(self, _request: Trigger.Request, response: Trigger.Response):
        with self._lock:
            state = self._state_machine.state
            if state is not TrialState.READY:
                return self._service_response(
                    response, False, f"not ready to start (state {state.value})"
                )
            started, detail = self._begin_rollout(time.monotonic_ns())
            return self._service_response(response, started, detail)

    #: Terminal states an operator may still label a success.
    #:
    #: TIMED_OUT is included because the geometric check cannot always tell a
    #: seated plug from one on the rim. REJECTED and ABORTED are excluded: their
    #: terminal observation is invalid.
    _OPERATOR_SUCCESS_STATES = (TrialState.SUCCEEDED, TrialState.TIMED_OUT)

    def _confirm_success_service(self, _request: Trigger.Request, response: Trigger.Response):
        with self._lock:
            state = self._state_machine.state
            if state not in self._OPERATOR_SUCCESS_STATES:
                if state is TrialState.IDLE:
                    return self._service_response(response, False, "no active trial")
                if not self._state_machine.is_terminal:
                    return self._service_response(
                        response,
                        False,
                        f"trial is still {state.value}; let it finish, or use "
                        "confirm_failure / discard_trial",
                    )
                return self._service_response(
                    response,
                    False,
                    f"a {state.value} trial cannot be relabelled a success: it stopped "
                    "on a guard or an abort, so its final observation is marked "
                    "invalid; discard it instead",
                )
            overridden = state is not TrialState.SUCCEEDED
            if self._mode == "shadow":
                # No command reached the robot, so success is meaningless.
                # Deterministic live trials may be labelled; `export_eligible`
                # already keeps them out of training data.
                return self._service_response(
                    response,
                    False,
                    "shadow-mode trials cannot be labelled a success: no command "
                    "reached the robot; confirm failure or discard it",
                )
            # Record whether success was automated or an operator override of
            # a timeout; Phase 2 trains on both but they are different evidence.
            reason = (
                "operator_override_success_after_timeout"
                if overridden
                else "operator_confirmed_success"
            )
            if overridden:
                self.get_logger().warning(
                    "operator labelled a TIMED_OUT trial a success; the automated "
                    "check did not fire. Recorded as "
                    f"termination_reason={reason!r}."
                )
            try:
                self._finish_trial(
                    accepted=True,
                    operator_label="success",
                    termination_reason=reason,
                )
            except Exception as exc:
                return self._service_response(response, False, str(exc))
            return self._service_response(
                response,
                True,
                "success accepted (operator override; automated check did not fire)"
                if overridden
                else "success accepted",
            )

    def _confirm_failure_service(self, _request: Trigger.Request, response: Trigger.Response):
        with self._lock:
            if self._state_machine.state is TrialState.IDLE:
                return self._service_response(response, False, "no active trial")
            if not self._state_machine.is_terminal:
                self._state_machine.transition(
                    TrialEvent.ABORT, time.monotonic_ns(), reason="operator failure"
                )
                self._ensure_terminal_observation(valid=False, guard_bitmask=GUARD_POLICY)
                self._auto_outcome = "operator_stopped"
                self._terminal_reason = "operator_confirmed_failure"
                self._trial_export_eligible = False
            try:
                self._finish_trial(
                    accepted=False,
                    operator_label="failure",
                    termination_reason=self._terminal_reason or "operator_confirmed_failure",
                )
            except Exception as exc:
                return self._service_response(response, False, str(exc))
            return self._service_response(response, True, "failure recorded")

    def _discard_trial_service(self, _request: Trigger.Request, response: Trigger.Response):
        with self._lock:
            if self._state_machine.state is TrialState.IDLE:
                return self._service_response(response, False, "no active trial")
            if not self._state_machine.is_terminal:
                self._state_machine.transition(
                    TrialEvent.ABORT, time.monotonic_ns(), reason="operator discard"
                )
            self._trial_export_eligible = False
            if self._rejected_before_rollout:
                reason = self._terminal_reason or "rejected_before_rollout"
            else:
                self._auto_outcome = "discarded"
                reason = "operator_discarded"
            self._terminal_reason = reason
            self._ensure_terminal_observation(valid=False, guard_bitmask=GUARD_POLICY)
            try:
                self._finish_trial(
                    accepted=False,
                    operator_label="discarded",
                    termination_reason=reason,
                )
            except Exception as exc:
                return self._service_response(response, False, str(exc))
            return self._service_response(
                response,
                True,
                "attempt retained in canonical HDF5 as discarded",
            )

    def _abort_session_service(self, _request: Trigger.Request, response: Trigger.Response):
        with self._lock:
            try:
                if self._state_machine.state is not TrialState.IDLE:
                    if not self._state_machine.is_terminal:
                        self._state_machine.transition(
                            TrialEvent.ABORT,
                            time.monotonic_ns(),
                            reason="operator session abort",
                        )
                    self._trial_export_eligible = False
                    self._auto_outcome = "session_aborted"
                    self._terminal_reason = "operator_aborted_session"
                    self._ensure_terminal_observation(
                        valid=False, guard_bitmask=GUARD_POLICY
                    )
                    self._finish_trial(
                        accepted=False,
                        operator_label="session_aborted",
                        termination_reason="operator_aborted_session",
                    )
                path = self._close_writer(clean=True)
                self._session_complete = True
            except Exception as exc:
                self._close_writer(clean=False)
                return self._service_response(response, False, str(exc))
            return self._service_response(response, True, f"session closed: {path}")

    @staticmethod
    def _service_response(response: Trigger.Response, success: bool, message: str):
        response.success = bool(success)
        response.message = str(message)
        return response

    # ---- trial lifecycle ----------------------------------------------

    def _start_trial(self) -> None:
        if self._session_complete:
            raise RuntimeError("session is already complete")
        if self._state_machine.state is not TrialState.IDLE:
            raise RuntimeError(f"trial state is {self._state_machine.state.value}")
        if self._nominal_reset_flange is None or self._flange_to_tcp is None:
            raise RuntimeError("teach_reset_pose must succeed first")
        if self._config.success.metric in AXIS_TOLERANCE_SUCCESS_METRICS:
            # Judged by a per-axis box, so distance_threshold_m does not apply.
            if self._config.success.axis_tolerances_m is None:
                raise RuntimeError("success.axis_tolerances_m is still TODO")
        elif self._config.success.distance_threshold_m is None:
            raise RuntimeError("success.distance_threshold_m is still TODO")
        if self._config.trial_control.has_placeholders:
            raise RuntimeError("trial_control still contains TODO values")
        if not self._cameras_configured or self._camera_buffer is None:
            raise RuntimeError("all enabled camera configurations must be resolved")
        if set(self._camera_info) != set(self._enabled_camera_names):
            waiting = sorted(set(self._enabled_camera_names) - set(self._camera_info))
            raise RuntimeError(
                "waiting for CameraInfo from enabled cameras: " + ", ".join(waiting)
            )
        snapshot = self._latest_snapshot
        if snapshot is None or snapshot.tcp_pose is None:
            raise RuntimeError("waiting for a RobotStates sample with flange and TCP poses")
        if snapshot.validation_errors():
            raise RuntimeError("latest RobotStates sample is invalid")
        fresh_guard = RobotStateGuard(self._state_guard.config).evaluate(
            snapshot, self.get_clock().now().nanoseconds
        )
        if not fresh_guard.accepted:
            raise RuntimeError(f"latest RobotStates sample is not fresh: {fresh_guard.reason}")
        cameras = self._nearest_camera_frames(snapshot.ros_timestamp_ns)
        if cameras is None:
            raise RuntimeError("waiting for one frame from each enabled camera")
        _, cameras_valid, _ = self._camera_status(cameras)
        if not cameras_valid:
            raise RuntimeError("latest enabled camera frames exceed configured age or skew")
        if self._mode == "live":
            for camera in self._enabled_cameras:
                configured_extrinsics = camera.extrinsics_file
                if configured_extrinsics is None:
                    continue
                value = str(configured_extrinsics).strip()
                if not value or value.upper().startswith("TODO"):
                    raise RuntimeError(
                        f"camera {camera.name!r} extrinsics must be a real path or null"
                    )
                extrinsics = resolve_artifact_path(
                    value,
                    config_path=self._config_path,
                )
                if not extrinsics.is_file():
                    raise RuntimeError(
                        f"camera {camera.name!r} extrinsics file does not exist: "
                        f"{extrinsics}"
                    )

        self._flange_to_tcp = compose_poses(invert_pose(snapshot.anchor_pose), snapshot.tcp_pose)
        if self._repeat_targets is not None:
            self._targets = self._repeat_targets
        else:
            self._targets = self._randomizer.sample(
                self._attempt_index,
                self._nominal_reset_flange,
                self._exact_socket_root,
            )
        self._ensure_writer()
        assert self._writer is not None
        start_offset = self._targets.start_offset
        socket_offset = self._targets.socket_policy_offset
        self._writer.start_trial(
            attempt_index=self._attempt_index,
            attrs={
                "execution_mode": self._mode,
                # 0 for a first attempt, 1+ for a replay of the same targets,
                # so per-trial statistics can tell replays apart.
                "repeat_index": int(self._attempt_repeat_index),
                # Only physical live stochastic attempts may become training
                # data; shadow sessions remain useful for QA.
                "export_eligible": trial_export_eligible(
                    self._mode, self._deterministic_inference
                ),
                "taught_reset_flange_pose_world_xyzw": (
                    self._nominal_reset_flange.as_vector().tolist()
                ),
                "randomized_start_flange_pose_world_xyzw": (
                    self._targets.randomized_start.as_vector().tolist()
                ),
                "flange_to_controller_tcp_xyzw": self._flange_to_tcp.as_vector().tolist(),
                "start_offset_xyz_rpy": [
                    start_offset.x_m,
                    start_offset.y_m,
                    start_offset.z_m,
                    start_offset.roll_rad,
                    start_offset.pitch_rad,
                    start_offset.yaw_rad,
                ],
                "socket_policy_offset_xyz_rpy": [
                    socket_offset.x_m,
                    socket_offset.y_m,
                    socket_offset.z_m,
                    socket_offset.roll_rad,
                    socket_offset.pitch_rad,
                    socket_offset.yaw_rad,
                ],
            },
        )
        self._trial_observation_count = 0
        self._trial_action_count = 0
        self._trial_export_eligible = trial_export_eligible(
            self._mode, self._deterministic_inference
        )
        self._success_hold_steps = 0
        self._input_hold_start_ns = -1
        self._post_success_remaining = None
        self._success_reason = ""
        self._reset_trial_sync_counters()
        self._terminal_reason = ""
        self._auto_outcome = "in_progress"
        self._rollout_started = False
        self._rejected_before_rollout = False
        self._state_machine = TrialStateMachine()
        now = time.monotonic_ns()
        self._state_machine.transition(TrialEvent.BEGIN, now)
        self._state_machine.transition(TrialEvent.PAUSE_ACKNOWLEDGED, now)
        # Retract from the current pose, not toward the reset pose, so the plug
        # is never dragged sideways while still in the socket.
        waypoints = self._retraction_waypoints(snapshot.anchor_pose)
        first, reason = waypoints[0]
        self._retraction_queue = waypoints[1:]
        self._begin_reposition(snapshot.anchor_pose, first, now)
        self._last_diagnostic_reason = reason

    def _retraction_waypoints(self, seated: Pose) -> list[tuple[Pose, str]]:
        """Retraction stages, in order, as ``(flange goal, diagnostic reason)``.

        1. Level the tool onto the socket axis, pivoting about the plug tip (a
           flange pivot would swing the plug sideways by eef_offset * sin(tilt)).
        2. Lift straight along the socket axis, clear of the connector.

        Each stage is included only when configured and needed. Falls back to
        the reset-pose-anchored target when clearance is unset.
        """
        control = self._config.trial_control
        clearance = control.retraction_clearance_m
        if not clearance:
            return [(
                offset_position_world(
                    self._nominal_reset_flange, control.retraction_offset_world_m
                ),
                "retracting before trial",
            )]

        axis = self._socket_insertion_axis()
        stages: list[tuple[Pose, str]] = []
        start = seated
        if control.level_before_retract:
            levelled = self._level_about_tool_tip(seated, axis)
            if levelled is not None:
                tilt_deg = math.degrees(
                    self._tool_axis_error(seated, axis)
                )
                stages.append((
                    levelled,
                    f"levelling {tilt_deg:.1f} deg onto the socket axis, "
                    "pivoting about the plug tip",
                ))
                start = levelled
        stages.append((
            Pose(start.position + axis * float(clearance), start.quaternion_xyzw),
            f"retracting {clearance * 1000.0:.0f} mm clear of the socket",
        ))
        return stages

    def _configured_socket_root(self) -> Pose:
        """Socket root, derived from the configured opening pose."""
        return socket_root_from_opening(
            Pose.from_vector(self._config.socket.opening_pose_world_xyzw),
            self._observation_config.socket_keypoint_offset,
        )

    def _socket_insertion_axis(self) -> np.ndarray:
        """World-frame unit vector along which the plug leaves the socket.

        Derived from the socket pose rather than assumed to be world +z.
        """
        socket = self._exact_socket_root
        # socket_keypoint_offset lies along the socket's insertion axis.
        local = np.asarray(
            self._observation_config.socket_keypoint_offset, dtype=np.float64
        )
        norm_local = float(np.linalg.norm(local))
        if norm_local <= 1.0e-9:
            return np.array([0.0, 0.0, 1.0])
        axis = quaternion_to_matrix_xyzw(socket.quaternion_xyzw) @ (local / norm_local)
        norm = float(np.linalg.norm(axis))
        return axis / norm if norm > 1.0e-9 else np.array([0.0, 0.0, 1.0])

    def _tool_axis_error(self, pose: Pose, axis: np.ndarray) -> float:
        """Angle in radians between the tool's own axis and -``axis``."""
        tool = quaternion_to_matrix_xyzw(pose.quaternion_xyzw) @ np.array([0.0, 0.0, 1.0])
        return float(np.arccos(np.clip(float(np.dot(tool, -axis)), -1.0, 1.0)))

    def _level_about_tool_tip(self, pose: Pose, axis: np.ndarray) -> Optional[Pose]:
        """``pose`` reoriented onto the socket axis, tool tip held in place.

        Yaw about the tool axis is preserved, since rotating the connector inside
        the socket must not happen. Returns None when already aligned.
        """
        tilt = self._tool_axis_error(pose, axis)
        if tilt < math.radians(0.2):
            return None
        rotation = quaternion_to_matrix_xyzw(pose.quaternion_xyzw)
        tool = rotation @ np.array([0.0, 0.0, 1.0])
        target = -axis
        cross = np.cross(tool, target)
        sin_a = float(np.linalg.norm(cross))
        if sin_a < 1.0e-9:
            return None
        # Rotation taking the tool axis onto the socket axis, applied on the
        # left because it is expressed in the world frame.
        k = cross / sin_a
        kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        delta = (
            np.eye(3)
            + math.sin(tilt) * kx
            + (1.0 - math.cos(tilt)) * (kx @ kx)
        )
        levelled_rotation = delta @ rotation
        offset = np.asarray(self._observation_config.eef_offset_flange, dtype=np.float64)
        tip = pose.position + rotation @ offset
        position = tip - levelled_rotation @ offset
        return Pose(position, _matrix_to_quaternion_xyzw(levelled_rotation))

    def _begin_reposition(self, start: Pose, goal: Pose, now_ns: int) -> None:
        control = self._config.trial_control
        assert control.reposition_duration_s is not None
        assert control.reposition_timeout_s is not None
        duration_ns = int(control.reposition_duration_s * 1.0e9)
        self._trajectory = TimedPoseTrajectory(start, goal, now_ns, duration_ns)
        self._trajectory_deadline_ns = now_ns + int(control.reposition_timeout_s * 1.0e9)
        self._segment_stable_steps = 0

    def _control_tick(self) -> None:
        with self._lock:
            try:
                if self._writer is not None:
                    self._writer.check_health()
                state = self._state_machine.state
                if state in {TrialState.RETRACTING, TrialState.MOVING_TO_START}:
                    self._tick_reposition()
                elif state is TrialState.SETTLING:
                    self._tick_settling()
                elif state is TrialState.READY:
                    self._tick_ready()
                elif state in {TrialState.RUNNING, TrialState.HOLDING}:
                    self._tick_rollout()
            except (WriterBackpressureError, WriterWorkerError) as exc:
                self.get_logger().error(f"recording failure: {exc}")
                self._last_diagnostic_reason = f"recording failure: {exc}"
                self._close_writer(clean=False)
                self._session_complete = True
            except Exception as exc:
                self.get_logger().error(f"control tick rejected: {exc}")
                self._reject_trial(f"control_tick_exception:{exc}", GUARD_POLICY)
            self._diagnostic_tick += 1
            if self._diagnostic_tick % max(1, int(self._config.policy.control_rate_hz)) == 0:
                self._publish_diagnostics()

    def _tick_reposition(self) -> None:
        trajectory = self._trajectory
        snapshot = self._latest_snapshot
        if trajectory is None or snapshot is None or self._flange_to_tcp is None:
            return
        now = time.monotonic_ns()
        decision = self._state_guard.evaluate(snapshot, self.get_clock().now().nanoseconds)
        if decision.status is GuardStatus.REJECT:
            self._reject_trial(decision.reason, GUARD_ROBOT_INVALID)
            return
        if decision.status is GuardStatus.HOLD:
            if now > self._trajectory_deadline_ns:
                self._reject_trial(decision.reason, GUARD_ROBOT_STALE)
            return

        target_flange = trajectory.sample(now)
        if self._mode == "live":
            self._publish_tcp_target(compose_poses(target_flange, self._flange_to_tcp))
        if not trajectory.complete(now):
            return
        position_error, orientation_error = pose_error(snapshot.anchor_pose, trajectory.goal)
        control = self._config.trial_control
        if self._mode == "shadow":
            reached = True
        else:
            reached = (
                position_error <= control.position_tolerance_m
                and orientation_error <= control.orientation_tolerance_rad
            )
        self._segment_stable_steps = self._segment_stable_steps + 1 if reached else 0
        required = self._config.trial_control.stable_steps
        if self._segment_stable_steps < required:
            if now > self._trajectory_deadline_ns:
                # Report which term missed and by how much, to distinguish "never
                # arrived" from a tolerance tighter than steady-state error.
                detail = (
                    f"reposition_timeout: position_error "
                    f"{position_error * 1000.0:.3f} mm (limit "
                    f"{control.position_tolerance_m * 1000.0:.3f}), "
                    f"orientation_error {orientation_error * 1000.0:.3f} mrad "
                    f"(limit {control.orientation_tolerance_rad * 1000.0:.3f})"
                )
                self.get_logger().warning(detail)
                self._last_diagnostic_reason = detail
                self._reject_trial("reposition_timeout", GUARD_ROBOT_STALE)
            return

        if self._state_machine.state is TrialState.RETRACTING:
            if self._retraction_queue:
                # Stages run sequentially; levelling and lifting at once would
                # blend into a diagonal move.
                goal, reason = self._retraction_queue.pop(0)
                self._begin_reposition(snapshot.anchor_pose, goal, now)
                self._last_diagnostic_reason = reason
                return
            self._state_machine.transition(TrialEvent.RETRACTED, now)
            assert self._targets is not None
            self._begin_reposition(snapshot.anchor_pose, self._targets.randomized_start, now)
            self._last_diagnostic_reason = "moving to randomized start flange pose"
        else:
            self._state_machine.transition(TrialEvent.START_REACHED, now)
            self._settle_deadline_ns = now + int(
                self._config.trial_control.settle_time_s * 1.0e9
            )
            self._last_diagnostic_reason = "settling at randomized start"

    def _tick_settling(self) -> None:
        snapshot = self._latest_snapshot
        if snapshot is None or self._targets is None or self._flange_to_tcp is None:
            return
        now = time.monotonic_ns()
        decision = self._state_guard.evaluate(snapshot, self.get_clock().now().nanoseconds)
        if decision.status is not GuardStatus.ACCEPT:
            if now > self._settle_deadline_ns + int(1.0e9):
                bit = GUARD_ROBOT_INVALID if decision.rejected else GUARD_ROBOT_STALE
                self._reject_trial(decision.reason, bit)
            return
        if self._mode == "live":
            self._publish_tcp_target(
                compose_poses(self._targets.randomized_start, self._flange_to_tcp)
            )
        if now < self._settle_deadline_ns:
            return
        if self._mode != "shadow":
            position_error, orientation_error = pose_error(
                snapshot.anchor_pose, self._targets.randomized_start
            )
            control = self._config.trial_control
            if (
                position_error > control.position_tolerance_m
                or orientation_error > control.orientation_tolerance_rad
            ):
                self._reject_trial("start_pose_drifted_during_settle", GUARD_ROBOT_STALE)
                return
        self._state_machine.transition(TrialEvent.SETTLED, now)
        if self._config.runtime.require_start_confirmation:
            # Hold the start pose in READY until ~/begin_rollout is called.
            self._last_diagnostic_reason = "ready: awaiting operator start confirmation"
            return
        self._begin_rollout(now)

    def _tick_ready(self) -> None:
        """Hold the randomized start pose while awaiting the operator's go.

        There is no deadline: actively commanding the start pose keeps the wait
        safe, and ``_begin_rollout`` re-checks the pose before any policy action.
        """

        snapshot = self._latest_snapshot
        if snapshot is None or self._targets is None or self._flange_to_tcp is None:
            return
        decision = self._state_guard.evaluate(snapshot, self.get_clock().now().nanoseconds)
        if decision.status is not GuardStatus.ACCEPT:
            return
        if self._mode == "live":
            self._publish_tcp_target(
                compose_poses(self._targets.randomized_start, self._flange_to_tcp)
            )

    def _begin_rollout(self, now: int) -> tuple[bool, str]:
        """Reset the policy and enter RUNNING.  Returns (started, detail)."""

        snapshot = self._latest_snapshot
        if snapshot is None or self._targets is None:
            return False, "no RobotStates received"
        if self._mode != "shadow":
            # The wait in READY is unbounded, so re-verify the start pose here
            # rather than trusting the check made at the end of settling.
            position_error, orientation_error = pose_error(
                snapshot.anchor_pose, self._targets.randomized_start
            )
            control = self._config.trial_control
            if (
                position_error > control.position_tolerance_m
                or orientation_error > control.orientation_tolerance_rad
            ):
                self._reject_trial("start_pose_drifted_before_rollout", GUARD_ROBOT_STALE)
                return False, "start pose drifted while awaiting confirmation"
        try:
            self._policy.reset()
        except Exception as exc:
            self._reject_trial(f"policy_reset_failed:{exc}", GUARD_POLICY)
            return False, f"policy reset failed: {exc}"
        self._state_machine.transition(TrialEvent.POLICY_RESET_ACKNOWLEDGED, now)
        self._rollout_started = True
        self._state_guard.reset()
        self._rollout_start_ns = now
        self._last_diagnostic_reason = "policy rollout running"
        self._publish_policy_goal()
        return True, "rollout started"

    def _compose_action(self, sample: Any, bundle: _InputBundle) -> ActionTrace:
        """Turn one policy sample into the action actually commanded.

        Phase 1 composes the base policy alone. Phase 2 overrides this to add
        the residual; the Phase 1 body is pinned by the recorded-session replay
        tests.
        """

        return build_action_trace(
            raw_mean=sample.raw_mean,
            raw_sample=sample.raw_sample,
            observed_flange_pose=bundle.snapshot.anchor_pose,
            flange_to_tcp=self._flange_to_tcp,
            config=self._action_config,
            deterministic=self._deterministic_inference,
        )

    def _tick_rollout(self) -> None:
        now = time.monotonic_ns()
        bundle = self._make_input_bundle(now)
        if bundle is None:
            if now - self._rollout_start_ns >= int(
                self._config.recording.trial_timeout_s * 1.0e9
            ):
                self._reject_trial("input_missing_until_timeout", GUARD_CAMERA_AGE)
            return
        disposition = classify_rollout_input(
            guard_status=bundle.guard.status,
            camera_pair_valid=bundle.camera_pair_valid,
            policy_input_valid=bundle.policy_input_valid,
        )
        if disposition is RolloutInputDisposition.REJECT:
            self._terminate_with_invalid_observation(
                bundle,
                reject=True,
                reason=bundle.guard.reason or "policy observation encoding is invalid",
            )
            return
        if disposition is RolloutInputDisposition.HOLD:
            # A held tick writes no rows, so count dropped steps and their reasons
            # for the trial attrs.
            self._hold_tick_count += 1
            self._hold_guard_bits |= int(bundle.guard_bitmask)
            if self._state_machine.state is TrialState.RUNNING:
                self._state_machine.transition(
                    TrialEvent.SAFETY_HOLD,
                    now,
                    reason=bundle.guard.reason or "camera synchronization hold",
                )
                self._input_hold_start_ns = now
            elif self._input_hold_start_ns < 0:
                self._input_hold_start_ns = now
            self._last_diagnostic_reason = bundle.guard.reason or "camera synchronization hold"
            hold_timeout_s = self._config.trial_control.input_hold_timeout_s
            assert hold_timeout_s is not None
            if now - self._input_hold_start_ns >= int(hold_timeout_s * 1.0e9):
                self._terminate_with_invalid_observation(
                    bundle,
                    reason=bundle.guard.reason or "camera synchronization hold timed out",
                )
            return

        if self._state_machine.state is TrialState.HOLDING:
            self._state_machine.transition(TrialEvent.STATE_FRESH, now)
            self._input_hold_start_ns = -1
        # With cameras near the control rate, jitter can match the previous
        # tick's frame again; not an error, but recorded.
        self._count_repeated_frames(bundle)
        self._append_observation(bundle.record)
        assert self._targets is not None
        success = self._targets.evaluate_success(
            bundle.snapshot,
            threshold_m=self._config.success.distance_threshold_m,
            metric=self._config.success.metric,
            observation_config=self._observation_config,
            axis_tolerances_m=self._config.success.axis_tolerances_m,
        )
        self._success_hold_steps = (
            self._success_hold_steps + 1 if success.succeeded else 0
        )
        # Success fires mid-insertion, so post_success_steps keeps recording
        # until the plug seats. Once started the dwell always completes, since
        # the rollout timeout is suppressed during it.
        if self._post_success_remaining is not None:
            if self._post_success_remaining > 0:
                self._post_success_remaining -= 1
                self._last_diagnostic_reason = (
                    f"success reached; recording {self._post_success_remaining} "
                    "more step(s)"
                )
                # Fall through: dwell steps are commanded and recorded normally.
            else:
                self._state_machine.transition(
                    TrialEvent.SUCCESS, now, reason=self._success_reason
                )
                self._auto_outcome = "success_candidate"
                self._terminal_reason = "distance_threshold_reached"
                self._last_diagnostic_reason = (
                    "success candidate awaiting operator confirmation"
                )
                return
        elif self._success_hold_steps >= self._config.success.hold_steps:
            self._success_reason = (
                f"{success.metric} distance {success.distance_m:.6g} m"
            )
            self._post_success_remaining = int(
                self._config.success.post_success_steps
            )
            if self._post_success_remaining <= 0:
                self._state_machine.transition(
                    TrialEvent.SUCCESS, now, reason=self._success_reason
                )
                self._auto_outcome = "success_candidate"
                self._terminal_reason = "distance_threshold_reached"
                self._last_diagnostic_reason = (
                    "success candidate awaiting operator confirmation"
                )
                return
            self._last_diagnostic_reason = (
                f"success reached; recording {self._post_success_remaining} "
                "more step(s)"
            )
        # Runaway guard: stop a tool driving away from the socket rather than
        # waiting out trial_timeout_s.
        lateral = self._lateral_offset_m(bundle)
        limit = self._config.trial_control.max_lateral_offset_m
        if (
            limit is not None
            and lateral is not None
            and lateral > limit
            and self._post_success_remaining is None
        ):
            self._state_machine.transition(TrialEvent.TIMEOUT, now)
            self._auto_outcome = "runaway"
            self._terminal_reason = "max_lateral_offset_exceeded"
            detail = (
                f"runaway: tool is {lateral * 1000.0:.1f} mm laterally from the socket "
                f"(limit {limit * 1000.0:.1f} mm); stopping the trial"
            )
            self.get_logger().warning(detail)
            self._last_diagnostic_reason = detail
            return

        # The rollout timeout must not turn an already-satisfied success into a
        # timeout just because the dwell ran past it.
        if self._post_success_remaining is None and now - self._rollout_start_ns >= int(
            self._config.recording.trial_timeout_s * 1.0e9
        ):
            self._state_machine.transition(TrialEvent.TIMEOUT, now)
            self._auto_outcome = "timeout"
            self._terminal_reason = "trial_timeout"
            self._last_diagnostic_reason = "timeout awaiting operator confirmation"
            return

        inference_start = time.monotonic_ns()
        try:
            sample = self._policy.sample_once(bundle.policy_input)
        except Exception as exc:
            self._reject_trial(f"policy_inference_failed:{exc}", GUARD_POLICY)
            return
        inference_ns = time.monotonic_ns() - inference_start
        composition_start = time.monotonic_ns()
        try:
            trace = self._compose_action(sample, bundle)
        except Exception as exc:
            self._reject_trial(f"action_composition_failed:{exc}", GUARD_POLICY)
            return
        composition_ns = time.monotonic_ns() - composition_start

        command_stamp_ns = -1
        published = False
        publish_ns = -1
        if self._mode == "live":
            publish_start = time.monotonic_ns()
            try:
                command_stamp_ns = self._publish_tcp_target(
                    trace.submitted_controller_tcp_target
                )
                published = True
            except Exception as exc:
                self._last_diagnostic_reason = f"command publish failed: {exc}"
            publish_ns = time.monotonic_ns() - publish_start
        command_valid = published and self._mode != "shadow"
        action_record = build_action_record(
            trace=trace,
            raw_std=sample.raw_std,
            recurrent_state_in=sample.recurrent_state_in,
            recurrent_state_out=sample.recurrent_state_out,
            recurrent_num_layers=self._policy_spec.recurrent_num_layers,
            recurrent_hidden_dim=self._policy_spec.recurrent_hidden_dim,
            step_start_monotonic_ns=now,
            command_ros_stamp_ns=command_stamp_ns,
            inference_ns=inference_ns,
            composition_ns=composition_ns,
            publish_ns=publish_ns,
            recorder_enqueue_ns=time.monotonic_ns(),
            writer_queue_depth=self._writer.queue_depth,
            command_published=published,
            command_valid=command_valid,
            guard_bitmask=0 if published or self._mode == "shadow" else GUARD_COMMAND_PUBLISH,
        )
        self._append_action(action_record)
        if self._mode != "shadow" and not published:
            self._trial_export_eligible = False
            self._ensure_terminal_observation(
                valid=False, guard_bitmask=GUARD_COMMAND_PUBLISH, bundle=bundle
            )
            self._state_machine.transition(
                TrialEvent.SAFETY_REJECT, now, reason="command publish failed"
            )
            self._auto_outcome = "command_publish_failed"
            self._terminal_reason = "command_publish_failed"
        self._publish_policy_goal()

    # ---- input, recording, and command helpers ------------------------

    def _nearest_camera_frames(self, target_stamp_ns: int) -> CameraFrameSet | None:
        if self._camera_buffer is None:
            return None
        return self._camera_buffer.closest_frames(
            target_stamp_ns,
            max_delta_ns=(1 << 62),
            max_camera_skew_ns=None,
        )

    def _camera_status(
        self, cameras: CameraFrameSet
    ) -> tuple[dict[str, bool], bool, bool]:
        deltas = cameras.per_camera_delta_ns
        camera_validity = {
            camera.name: deltas[camera.name] <= int(camera.max_age_ms * 1.0e6)
            for camera in self._enabled_cameras
        }
        if len(self._enabled_cameras) == 1:
            # Pair skew is inapplicable with a single enabled camera.
            skew_valid = True
        else:
            max_skew_ms = self._config.synchronization.max_camera_pair_skew_ms
            skew_ns = cameras.camera_skew_ns
            skew_valid = (
                max_skew_ms is not None
                and skew_ns is not None
                and skew_ns <= int(max_skew_ms * 1.0e6)
            )
        return camera_validity, all(camera_validity.values()) and skew_valid, skew_valid

    def _make_input_bundle(self, control_tick_ns: int) -> _InputBundle | None:
        snapshot = self._latest_snapshot
        targets = self._targets
        if snapshot is None or targets is None:
            return None
        cameras = self._nearest_camera_frames(snapshot.ros_timestamp_ns)
        if cameras is None:
            return None
        guard = self._state_guard.evaluate(snapshot, self.get_clock().now().nanoseconds)
        camera_validity, camera_pair_valid, skew_valid = self._camera_status(cameras)
        bitmask = 0
        if guard.status is GuardStatus.REJECT:
            bitmask |= GUARD_ROBOT_INVALID
        elif guard.status is GuardStatus.HOLD:
            bitmask |= GUARD_ROBOT_STALE
        if not all(camera_validity.values()):
            bitmask |= GUARD_CAMERA_AGE
        if not skew_valid:
            bitmask |= GUARD_CAMERA_SKEW
        try:
            policy_input = encode_phase1_observation(
                snapshot.anchor_pose,
                targets.socket_root_policy,
                self._observation_config,
            )
            policy_input_valid = guard.accepted and camera_pair_valid
        except Exception:
            policy_input = np.full(18, np.nan, dtype=np.float32)
            policy_input_valid = False
            bitmask |= GUARD_POLICY
        record = build_observation_record(
            snapshot=snapshot,
            receive_monotonic_ns=self._latest_receive_monotonic_ns,
            control_tick_monotonic_ns=control_tick_ns,
            cameras=cameras,
            targets=targets,
            policy_input=policy_input,
            observation_config=self._observation_config,
            robot_state_valid=guard.accepted,
            camera_validity=camera_validity,
            policy_input_valid=policy_input_valid,
        )
        return _InputBundle(
            snapshot=snapshot,
            receive_monotonic_ns=self._latest_receive_monotonic_ns,
            cameras=cameras,
            guard=guard,
            camera_validity=camera_validity,
            camera_pair_valid=camera_pair_valid,
            policy_input=policy_input,
            policy_input_valid=policy_input_valid,
            record=record,
            guard_bitmask=bitmask,
        )

    def _append_observation(self, record: Mapping[str, Any]) -> None:
        if self._writer is None:
            raise RuntimeError("writer is not initialized")
        self._writer.append_observation(record)
        self._trial_observation_count += 1

    def _append_action(self, record: Mapping[str, Any]) -> None:
        if self._writer is None:
            raise RuntimeError("writer is not initialized")
        self._writer.append_action(record)
        self._trial_action_count += 1

    def _terminate_with_invalid_observation(
        self,
        bundle: _InputBundle,
        *,
        reject: bool = False,
        reason: str = "",
    ) -> None:
        self._trial_export_eligible = False
        self._ensure_terminal_observation(
            valid=False,
            guard_bitmask=bundle.guard_bitmask,
            bundle=bundle,
        )
        event = (
            TrialEvent.SAFETY_REJECT
            if reject or bundle.guard.status is GuardStatus.REJECT
            else TrialEvent.TIMEOUT
        )
        terminal_reason = reason or bundle.guard.reason or "camera synchronization timeout"
        self._state_machine.transition(
            event,
            time.monotonic_ns(),
            reason=terminal_reason,
        )
        self._auto_outcome = "invalid_input"
        self._terminal_reason = terminal_reason
        self._last_diagnostic_reason = self._terminal_reason

    def _ensure_terminal_observation(
        self,
        *,
        valid: bool,
        guard_bitmask: int,
        bundle: _InputBundle | None = None,
    ) -> None:
        if self._writer is None:
            return
        plan = plan_terminal_observation(
            self._trial_observation_count,
            self._trial_action_count,
        )
        if plan is TerminalObservationPlan.ALREADY_COMPLETE:
            return
        if bundle is None:
            bundle = self._make_input_bundle(time.monotonic_ns())
        if bundle is None:
            raise RuntimeError("cannot form terminal observation from last robot/camera samples")
        self._writer.invalidate_pending_action(guard_bitmask=guard_bitmask)
        record = dict(bundle.record)
        self._append_observation(record)
        if not valid:
            self._trial_export_eligible = False

    def _reject_trial(self, reason: str, guard_bitmask: int) -> None:
        if self._state_machine.state is TrialState.IDLE or self._state_machine.is_terminal:
            return
        try:
            self._state_machine.transition(
                TrialEvent.SAFETY_REJECT,
                time.monotonic_ns(),
                reason=reason,
            )
            self._ensure_terminal_observation(
                valid=False, guard_bitmask=guard_bitmask
            )
        except Exception as exc:
            self.get_logger().error(f"could not complete rejected attempt: {exc}")
        self._trial_export_eligible = False
        self._auto_outcome = "rejected"
        self._terminal_reason = reason
        self._rejected_before_rollout = not self._rollout_started
        self._last_diagnostic_reason = reason

    def _publish_tcp_target(self, tcp_target: Pose) -> int:
        if self._command_publisher is None:
            raise RuntimeError(f"command publishing is disabled in {self._mode!r} mode")
        message = CartesianMotionForceCommand()
        stamp = self.get_clock().now()
        message.header.stamp = stamp.to_msg()
        message.header.frame_id = self._config.robot.world_frame
        _set_pose_message(message.target_pose, tcp_target)
        # Target wrench, velocity, and acceleration stay zero, as recorded.
        self._command_publisher.publish(message)
        return stamp.nanoseconds

    def _publish_policy_goal(self) -> None:
        if (
            not policy_goal_publish_enabled(
                self._mode, self._config.runtime.publish_goal_pose
            )
            or self._targets is None
        ):
            return
        message = PoseStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._config.robot.world_frame
        _set_pose_message(message.pose, self._targets.socket_root_policy)
        self._goal_publisher.publish(message)

    def _ensure_writer(self) -> None:
        if self._writer is not None:
            return
        output = resolve_output_directory(
            self._config.recording.output_directory, self._repository_root
        )
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        final_path = output / f"sparr_phase1_{timestamp}.h5"
        provenance = build_session_provenance(
            repository_root=self._repository_root,
            checkpoint=self._config.policy.checkpoint,
            agent_yaml=self._config.policy.agent_yaml,
            env_yaml=self._config.policy.env_yaml,
        )
        calibration = {
            name: dict(record) for name, record in self._camera_info.items()
        }
        for camera in self._enabled_cameras:
            record = calibration[camera.name]
            configured_extrinsics = camera.extrinsics_file
            if configured_extrinsics is None:
                record["extrinsics_available"] = False
                record["extrinsics_used_by_phase1"] = False
                record["extrinsics_status"] = "not_provided"
                continue
            value = str(configured_extrinsics).strip()
            extrinsics = resolve_artifact_path(
                value,
                config_path=self._config_path,
            )
            if extrinsics.is_file():
                record["extrinsics_available"] = True
                record["extrinsics_status"] = "provided"
                record["extrinsics_provenance"] = asdict(
                    artifact_provenance(extrinsics)
                )
            else:
                record["extrinsics_available"] = False
                record["extrinsics_status"] = "configured_missing"
        self._writer = AsyncHDF5SessionWriter(
            final_path,
            queue_size=self._config.recording.writer_queue_size,
            enabled_cameras=self._enabled_camera_names,
            metadata={
                "collection_config": self._config.resolved_mapping(),
                "policy_spec": asdict(self._policy_spec),
                "provenance": provenance,
                "label_semantics": (
                    "effective_software_action_normalized minus same-forward raw mean"
                ),
                "command_semantics": (
                    "submitted controller TCP target; no acknowledgement/execution claim"
                ),
                "environment": {
                    "ros_distro": os.environ.get("ROS_DISTRO", "unknown"),
                    "rmw_implementation": os.environ.get("RMW_IMPLEMENTATION", "default"),
                },
            },
            calibration=calibration,
            root_attrs={
                "execution_mode": self._mode,
                "robot_serial": self._config.robot.serial,
                "robot_state_topic": self._config.robot.state_topic,
                "controller_command_topic": self._config.robot.command_topic,
                **self._extra_root_attrs(),
            },
            # Must match what CameraBuffer stores, or the first image write
            # fails the writer's per-row shape check.
            image_shape=self._stored_image_shape(),
            training_action_clip=self._policy_spec.training_action_clip,
            flush_every=self._config.recording.flush_every_steps,
        )
        self.get_logger().info(f"Recording session to {self._writer.partial_path}")

    def _lateral_offset_m(self, bundle: _InputBundle) -> float | None:
        """World-xy distance from the tool keypoint to the exact socket keypoint.

        Lateral only: travel along the insertion axis is large during a normal
        approach, so a 3-D radius cannot separate descending from driving away.
        """
        targets = self._targets
        if targets is None or self._observation_config is None:
            return None
        try:
            tool = offset_point_world(
                bundle.snapshot.anchor_pose, self._observation_config.eef_offset_flange
            )
            socket = offset_point_world(
                targets.socket_root_exact,
                self._observation_config.socket_keypoint_offset,
            )
        except Exception as error:  # noqa: BLE001 - a guard must not raise mid-rollout
            # Returning None disables the runaway check, so say so rather than
            # letting the safeguard fail silently for the rest of the session.
            self.get_logger().warning(
                f"runaway guard disabled; lateral offset unavailable: {error}",
                once=True,
            )
            return None
        return float(np.linalg.norm(np.asarray(tool)[:2] - np.asarray(socket)[:2]))

    def _stop_trial_service(self, _request: Trigger.Request, response: Trigger.Response):
        """Operator abort of the running rollout, recorded as a failure.

        Unlike abort_session and discard_trial, the data is kept and the trial
        can be labelled.
        """
        with self._lock:
            state = self._state_machine.state
            if state not in {TrialState.RUNNING, TrialState.HOLDING}:
                return self._service_response(
                    response, False, f"no rollout to stop (state {state.value})"
                )
            self._state_machine.transition(TrialEvent.TIMEOUT, time.monotonic_ns())
            self._auto_outcome = "operator_stopped"
            self._terminal_reason = "operator_stopped_trial"
            self._last_diagnostic_reason = "operator stopped the rollout"
            self.get_logger().warning("operator stopped the rollout")
            return self._service_response(response, True, "rollout stopped")

    def _extra_root_attrs(self) -> dict[str, Any]:
        """Session-level attributes a subclass wants recorded. Empty here."""

        return {}

    def _stored_image_shape(self) -> tuple[int, int, int]:
        """(height, width, 3) actually stored, from the enabled cameras."""

        if not self._enabled_cameras:
            return (128, 128, 3)
        camera = self._enabled_cameras[0]
        return (int(camera.output_height), int(camera.output_width), 3)

    def _count_repeated_frames(self, bundle: _InputBundle) -> None:
        """Record how often a camera frame is reused on consecutive steps."""

        for frame in bundle.cameras.frames:
            name = frame.camera_id
            stamp = int(frame.timestamp_ns)
            if self._last_frame_stamp_ns.get(name) == stamp:
                self._repeated_frame_counts[name] = (
                    self._repeated_frame_counts.get(name, 0) + 1
                )
            self._last_frame_stamp_ns[name] = stamp

    def _reset_trial_sync_counters(self) -> None:
        self._hold_tick_count = 0
        self._hold_guard_bits = 0
        self._last_frame_stamp_ns = {}
        self._repeated_frame_counts = {}

    def _sync_trial_attrs(self) -> dict[str, Any]:
        """Per-trial synchronisation accounting written alongside the labels."""

        attrs: dict[str, Any] = {
            "held_tick_count": int(self._hold_tick_count),
            "held_guard_bitmask": int(self._hold_guard_bits),
        }
        for camera in self._enabled_cameras:
            attrs[f"{camera.name}_repeated_frame_count"] = int(
                self._repeated_frame_counts.get(camera.name, 0)
            )
        return attrs

    def _finish_trial(
        self,
        *,
        accepted: bool,
        operator_label: str,
        termination_reason: str,
    ) -> None:
        if self._writer is None:
            raise RuntimeError("writer is not initialized")
        self._ensure_terminal_observation(
            valid=self._trial_export_eligible,
            guard_bitmask=0 if self._trial_export_eligible else GUARD_POLICY,
        )
        self._writer.finish_trial(
            # `accepted` is the success verdict; `export_eligible` separately
            # says whether the trial may become training data.
            accepted=bool(accepted),
            operator_label=operator_label,
            auto_outcome=self._auto_outcome,
            termination_reason=termination_reason,
            attrs={
                "export_eligible": bool(self._trial_export_eligible),
                **self._sync_trial_attrs(),
            },
        )
        if accepted:
            self._accepted_successes += 1
        if self._rejected_before_rollout:
            # The policy never ran, so this attempt has no outcome: keep the
            # index (and any armed replay) so the same trial runs again.
            self.get_logger().warning(
                f"attempt {self._attempt_index} rejected before its rollout "
                f"({self._terminal_reason}); it will be run again"
            )
        elif self._repeat_targets is None:
            self._attempt_index += 1
            self._attempt_repeat_index = 0
        else:
            # Same index, same targets: the next attempt is a replay, and the
            # index has to stay put or the randomizer would move on.
            self._attempt_repeat_index += 1
        if self._state_machine.is_terminal:
            self._state_machine.transition(TrialEvent.CLEAR, time.monotonic_ns())
        self._targets = None
        self._trajectory = None
        #: Remaining retraction waypoints, consumed inside RETRACTING.
        self._retraction_queue: list[tuple[Pose, str]] = []
        self._trial_observation_count = 0
        self._trial_action_count = 0
        # Publish now rather than on the 1 Hz timer so consumers (run_trials)
        # see the new total as soon as the trial is confirmed.
        self._publish_diagnostics()
        if self._accepted_successes >= self._config.recording.target_successes:
            path = self._close_writer(clean=True)
            self._session_complete = True
            self._last_diagnostic_reason = f"target reached; session closed at {path}"

    def _close_writer(self, *, clean: bool) -> Path | None:
        if self._writer is None:
            return None
        writer = self._writer
        self._writer = None
        return writer.close(clean=clean)

    # ---- diagnostics and shutdown -------------------------------------

    def _publish_diagnostics(self) -> None:
        message = DiagnosticArray()
        message.header.stamp = self.get_clock().now().to_msg()
        status = DiagnosticStatus()
        status.name = "sparr_phase1_collection"
        status.hardware_id = self._config.robot.serial
        state = self._state_machine.state
        if self._session_complete:
            status.level = DiagnosticStatus.OK
            status.message = "session complete"
        elif state in {TrialState.REJECTED, TrialState.ABORTED}:
            status.level = DiagnosticStatus.ERROR
            status.message = self._last_diagnostic_reason
        elif state in {TrialState.HOLDING, TrialState.TIMED_OUT, TrialState.SUCCEEDED}:
            status.level = DiagnosticStatus.WARN
            status.message = self._last_diagnostic_reason
        else:
            status.level = DiagnosticStatus.OK
            status.message = self._last_diagnostic_reason
        values = {
            "mode": self._mode,
            "trial_state": state.value,
            "attempt_index": self._attempt_index,
            "accepted_successes": self._accepted_successes,
            "target_successes": self._config.recording.target_successes,
            "writer_queue_depth": self._writer.queue_depth if self._writer else 0,
            "robot_state_received": self._latest_snapshot is not None,
            "camera_info_count": len(self._camera_info),
            "enabled_camera_count": len(self._enabled_cameras),
            "enabled_cameras": ",".join(self._enabled_camera_names),
            "session_complete": self._session_complete,
        }
        status.values = [KeyValue(key=str(key), value=str(value)) for key, value in values.items()]
        message.status = [status]
        self._diagnostics_publisher.publish(message)

    def shutdown(self) -> Path | None:
        """Best-effort structural close; retain a partial file on any failure."""

        with self._lock:
            clean = True
            try:
                if self._state_machine.state is not TrialState.IDLE:
                    if not self._state_machine.is_terminal:
                        self._state_machine.transition(
                            TrialEvent.ABORT,
                            time.monotonic_ns(),
                            reason="node shutdown",
                        )
                    self._trial_export_eligible = False
                    self._auto_outcome = "node_shutdown"
                    self._terminal_reason = "node_shutdown"
                    self._ensure_terminal_observation(
                        valid=False, guard_bitmask=GUARD_POLICY
                    )
                    self._finish_trial(
                        accepted=False,
                        operator_label="node_shutdown",
                        termination_reason="node_shutdown",
                    )
            except Exception as exc:
                clean = False
                self.get_logger().error(f"could not structurally close active trial: {exc}")
            return self._close_writer(clean=clean)


__all__ = ["Phase1CollectorNode"]
