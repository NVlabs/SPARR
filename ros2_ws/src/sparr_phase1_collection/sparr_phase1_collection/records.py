# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Canonical HDF5 record construction from runtime snapshots and traces."""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from .action_pipeline import ActionTrace
from .camera_buffer import CameraFramePair
from .hdf5_schema import normalize_enabled_cameras
from .observations import ObservationConfig, offset_point_world
from .pose_math import Pose
from .robot_state import RobotStateSnapshot, SpatialVector, Wrench
from .trial_state_machine import TrialTargets


def _fixed(values: Any, size: int, *, dtype: Any = np.float64) -> np.ndarray:
    if values is None:
        return np.full(size, np.nan, dtype=dtype)
    array = np.asarray(values, dtype=dtype).reshape(-1)
    if array.shape != (size,):
        return np.full(size, np.nan, dtype=dtype)
    return array


def _pose(pose: Pose | None) -> np.ndarray:
    return np.full(7, np.nan, dtype=np.float64) if pose is None else pose.as_vector()


def _spatial(value: SpatialVector | None) -> np.ndarray:
    if value is None:
        return np.full(6, np.nan, dtype=np.float64)
    return np.asarray((*value.linear.as_tuple(), *value.angular.as_tuple()), dtype=np.float64)


def _wrench(value: Wrench | None) -> np.ndarray:
    if value is None:
        return np.full(6, np.nan, dtype=np.float64)
    return np.asarray((*value.force.as_tuple(), *value.torque.as_tuple()), dtype=np.float64)


def _keypoint_pose(root: Pose, offset: tuple[float, float, float]) -> np.ndarray:
    return Pose(offset_point_world(root, offset), root.quaternion_xyzw).as_vector()


def _split_lstm_state(
    values: np.ndarray,
    *,
    num_layers: int,
    hidden_dim: int,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the runtime's layer-major ``[hidden, cell]`` trace for HDF5."""

    layers = int(num_layers)
    hidden = int(hidden_dim)
    if layers < 1 or hidden < 1:
        raise ValueError("recurrent LSTM dimensions must be positive")
    state = np.asarray(values, dtype=np.float32).reshape(-1)
    expected = 2 * layers * hidden
    if state.size != expected:
        raise ValueError(
            f"{name} has {state.size} values; expected {expected} "
            f"for LSTM ({layers} layers, hidden {hidden})"
        )
    if not np.all(np.isfinite(state)):
        raise ValueError(f"{name} contains non-finite values")
    split = layers * hidden
    return state[:split].reshape(layers, hidden), state[split:].reshape(layers, hidden)


def build_observation_record(
    *,
    snapshot: RobotStateSnapshot,
    receive_monotonic_ns: int,
    control_tick_monotonic_ns: int,
    cameras: CameraFramePair,
    targets: TrialTargets,
    policy_input: np.ndarray,
    observation_config: ObservationConfig,
    robot_state_valid: bool,
    camera_validity: Mapping[str, bool],
    policy_input_valid: bool,
) -> dict[str, Any]:
    """Build one complete canonical observation without retaining source frames."""

    camera_ids = getattr(cameras, "camera_ids", None)
    if camera_ids is None:
        # Two-camera objects exposing only ``first``/``second``.
        camera_ids = (cameras.first.camera_id, cameras.second.camera_id)
    enabled_cameras = normalize_enabled_cameras(camera_ids)
    frames = {camera_id: cameras.by_camera(camera_id) for camera_id in enabled_cameras}

    timing: dict[str, Any] = {
        "ros_stamp_ns": np.int64(snapshot.ros_timestamp_ns),
        "robot_stamp_ns": np.int64(
            -1 if snapshot.robot_timestamp_ns is None else snapshot.robot_timestamp_ns
        ),
        "monotonic_receive_ns": np.int64(receive_monotonic_ns),
        "control_tick_monotonic_ns": np.int64(control_tick_monotonic_ns),
    }
    validity: dict[str, Any] = {
        "robot_state": bool(robot_state_valid),
        "policy_input": bool(policy_input_valid),
    }
    camera_records: dict[str, Any] = {}
    camera_extra: dict[str, Any] = {}
    for camera_id, frame in frames.items():
        timing[f"{camera_id}_stamp_ns"] = np.int64(frame.timestamp_ns)
        timing[f"{camera_id}_age_ns"] = np.int64(
            abs(snapshot.ros_timestamp_ns - frame.timestamp_ns)
        )
        validity[camera_id] = bool(camera_validity.get(camera_id, False))
        camera_records[camera_id] = {"rgb": frame.image}
        camera_extra[f"{camera_id}_source_shape"] = np.asarray(
            frame.source_shape, dtype=np.int32
        )
        camera_extra[f"{camera_id}_source_frame_id"] = str(
            frame.metadata.get("frame_id", "")
        )
        camera_extra[f"{camera_id}_source_encoding"] = str(
            frame.metadata.get("encoding", "")
        )
        # Unprocessed frame, if the buffer kept it; stored under the schema's
        # observations/extra/ extension namespace.
        if getattr(frame, "source_image", None) is not None:
            camera_extra[f"{camera_id}_source_rgb"] = frame.source_image
    if len(enabled_cameras) == 2:
        camera_skew_ns = cameras.camera_skew_ns
        if camera_skew_ns is None:
            raise ValueError("two-camera observations require a camera skew")
        timing["camera_skew_ns"] = np.int64(camera_skew_ns)

    exact_root = targets.socket_root_exact
    policy_root = targets.socket_root_policy
    # Invalid robot samples still need a terminal row to satisfy N == M + 1, so
    # a missing flange is recorded as NaN.
    if snapshot.flange_pose is None:
        flange_distance = float("nan")
        keypoint_distance = float("nan")
    else:
        flange_distance = float(
            np.linalg.norm(snapshot.flange_pose.position - exact_root.position)
        )
        # The same two points the policy sees: the flange shifted by the
        # checkpoint EEF offset, against the exact (never noised) socket keypoint.
        keypoint_distance = float(
            np.linalg.norm(
                offset_point_world(
                    snapshot.flange_pose, observation_config.eef_offset_flange
                )
                - offset_point_world(
                    exact_root, observation_config.socket_keypoint_offset
                )
            )
        )
    return {
        "timing": timing,
        "validity": validity,
        "camera": camera_records,
        "robot": {
            "flange_pose_world_xyzw": _pose(snapshot.flange_pose),
            "tcp_pose_world_xyzw": _pose(snapshot.tcp_pose),
            "q": _fixed(snapshot.q, 7),
            "theta": _fixed(snapshot.theta, 7),
            "dq": _fixed(snapshot.dq, 7),
            "dtheta": _fixed(snapshot.dtheta, 7),
            "tau": _fixed(snapshot.tau, 7),
            "tau_des": _fixed(snapshot.tau_des, 7),
            "tau_dot": _fixed(snapshot.tau_dot, 7),
            "tau_ext": _fixed(snapshot.tau_ext, 7),
            "tau_interact": _fixed(snapshot.tau_interact, 7),
            "temperature": _fixed(snapshot.temperature, 7),
            "tcp_velocity_world": _spatial(snapshot.tcp_velocity),
            "ft_sensor_raw_flange": _wrench(snapshot.ft_sensor_raw),
            "ext_wrench_tcp": _wrench(snapshot.ext_wrench_in_tcp),
            "ext_wrench_world": _wrench(snapshot.ext_wrench_in_world),
            "ext_wrench_tcp_raw": _wrench(snapshot.ext_wrench_in_tcp_raw),
            "ext_wrench_world_raw": _wrench(snapshot.ext_wrench_in_world_raw),
        },
        "task": {
            # Preserve the unmodified run parameter separately from every
            # derived/exact/noisy representation for later ablations.
            "socket_pose_supplied_world_xyzw": exact_root.as_vector(),
            "socket_root_exact_world_xyzw": exact_root.as_vector(),
            "socket_root_policy_world_xyzw": policy_root.as_vector(),
            "socket_keypoint_exact_world_xyzw": _keypoint_pose(
                exact_root, observation_config.socket_keypoint_offset
            ),
            "socket_keypoint_policy_world_xyzw": _keypoint_pose(
                policy_root, observation_config.socket_keypoint_offset
            ),
            "randomized_start_flange_world_xyzw": targets.randomized_start.as_vector(),
            "flange_to_socket_root_exact_distance_m": np.float64(flange_distance),
            "eef_keypoint_to_socket_keypoint_distance_m": np.float64(keypoint_distance),
        },
        "policy": {"input": np.asarray(policy_input, dtype=np.float32)},
        "extra": {
            "robot_source_frame_id": str(snapshot.frame_id),
            "raw_robot_states_json": json.dumps(
                snapshot.as_record(), sort_keys=True, separators=(",", ":"), default=str
            ),
            **camera_extra,
            "trial_seed": np.uint64(targets.derived_seed),
            "trial_index": np.int64(targets.trial_index),
            "start_offset_xyz_rpy": np.asarray(
                (
                    targets.start_offset.x_m,
                    targets.start_offset.y_m,
                    targets.start_offset.z_m,
                    targets.start_offset.roll_rad,
                    targets.start_offset.pitch_rad,
                    targets.start_offset.yaw_rad,
                ),
                dtype=np.float64,
            ),
            "socket_policy_offset_xyz_rpy": np.asarray(
                (
                    targets.socket_policy_offset.x_m,
                    targets.socket_policy_offset.y_m,
                    targets.socket_policy_offset.z_m,
                    targets.socket_policy_offset.roll_rad,
                    targets.socket_policy_offset.pitch_rad,
                    targets.socket_policy_offset.yaw_rad,
                ),
                dtype=np.float64,
            ),
        },
    }


def build_action_record(
    *,
    trace: ActionTrace,
    raw_std: np.ndarray,
    recurrent_state_in: np.ndarray,
    recurrent_state_out: np.ndarray,
    recurrent_num_layers: int,
    recurrent_hidden_dim: int,
    step_start_monotonic_ns: int,
    command_ros_stamp_ns: int,
    inference_ns: int,
    composition_ns: int,
    publish_ns: int,
    recorder_enqueue_ns: int,
    writer_queue_depth: int,
    command_published: bool,
    command_valid: bool,
    guard_bitmask: int = 0,
) -> dict[str, Any]:
    """Build the action trace and exact effective-action-minus-mean label."""

    std = np.asarray(raw_std, dtype=np.float32).reshape(6)
    hidden_in, cell_in = _split_lstm_state(
        recurrent_state_in,
        num_layers=recurrent_num_layers,
        hidden_dim=recurrent_hidden_dim,
        name="recurrent_state_in",
    )
    hidden_out, cell_out = _split_lstm_state(
        recurrent_state_out,
        num_layers=recurrent_num_layers,
        hidden_dim=recurrent_hidden_dim,
        name="recurrent_state_out",
    )
    standard_normal = (
        (trace.raw_sample - trace.raw_mean) / std
        if np.all(std > 0.0)
        else np.full(6, np.nan, dtype=np.float64)
    )
    label = (
        trace.effective_action_minus_raw_mean.astype(np.float32)
        if command_valid
        else np.full(6, np.nan, dtype=np.float32)
    )
    labels: dict[str, Any] = {
        "residual_effective_software_action_minus_raw_mean": label,
        "command_valid": bool(command_valid),
        "measured_valid": False,
    }
    if trace.requested_residual is not None:
        # Phase 2 only, under the schema's extension namespace; Phase 1 records
        # do not contain these datasets.
        labels["extra"] = {
            "requested_residual": trace.requested_residual.astype(np.float32),
            # Shadow mode records a residual it never commanded; without this
            # flag that data would be indistinguishable from active-mode data.
            "residual_applied": bool(trace.residual_applied),
            "residual_params_version": np.int64(trace.residual_params_version),
        }

    return {
        "policy": {
            "mu_raw_normalized": trace.raw_mean.astype(np.float32),
            "std_raw_normalized": std,
            "sample_raw_normalized": trace.raw_sample.astype(np.float32),
            # Always the clipped sample; in deterministic mode the effective
            # action comes from the mean instead.
            "sample_clipped_normalized": trace.clipped_sample.astype(np.float32),
            "sample_standard_normal": standard_normal.astype(np.float32),
            "recurrent_hidden_in": hidden_in,
            "recurrent_cell_in": cell_in,
            "recurrent_hidden_out": hidden_out,
            "recurrent_cell_out": cell_out,
        },
        "action": {
            "delta_scaled_pose_rel": np.concatenate(
                (trace.scaled_translation_world, trace.scaled_rotation_vector_world)
            ),
            "requested_flange_target_world_xyzw": trace.effective_flange_target.as_vector(),
            "effective_software_flange_target_world_xyzw": (
                trace.effective_flange_target.as_vector()
            ),
            "submitted_controller_tcp_target_world_xyzw": (
                trace.submitted_controller_tcp_target.as_vector()
            ),
            "effective_software_action_normalized": (
                trace.effective_software_action.astype(np.float32)
            ),
            "command_wrench": np.zeros(6, dtype=np.float64),
            "command_velocity": np.zeros(6, dtype=np.float64),
            "command_acceleration": np.zeros(6, dtype=np.float64),
            "command_published": bool(command_published),
            "guard_bitmask": np.uint32(guard_bitmask),
        },
        "timing": {
            "step_start_monotonic_ns": np.int64(step_start_monotonic_ns),
            "command_ros_stamp_ns": np.int64(command_ros_stamp_ns),
            "inference_ns": np.int64(inference_ns),
            "composition_ns": np.int64(composition_ns),
            "publish_ns": np.int64(publish_ns),
            "recorder_enqueue_ns": np.int64(recorder_enqueue_ns),
            "writer_queue_depth": np.int32(writer_queue_depth),
        },
        "labels": labels,
    }


__all__ = ["build_action_record", "build_observation_record"]
