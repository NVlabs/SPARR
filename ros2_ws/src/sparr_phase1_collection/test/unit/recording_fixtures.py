"""Small canonical records shared by recording-tool unit tests."""

from __future__ import annotations

import numpy as np


IMAGE_SHAPE = (8, 10, 3)
TRAINING_ACTION_CLIP = 0.75


def _pose(x: float, y: float, z: float) -> np.ndarray:
    return np.asarray([x, y, z, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def make_observation(
    index: int,
    *,
    image_shape: tuple[int, int, int] = IMAGE_SHAPE,
    include_extra: bool = False,
) -> dict:
    flange = _pose(0.40 + 0.002 * index, 0.10, 0.21 - 0.001 * index)
    tcp = flange.copy()
    tcp[2] -= 0.01
    exact_root = _pose(0.475, 0.128, 0.0325)
    policy_root = exact_root.copy()
    policy_root[0] += 0.003
    exact_keypoint = exact_root.copy()
    exact_keypoint[2] += 0.02
    policy_keypoint = policy_root.copy()
    policy_keypoint[2] += 0.02
    distance = np.linalg.norm(flange[:3] - exact_root[:3])
    policy_input = np.linspace(-0.5, 0.5, 18, dtype=np.float32) + index * 0.01
    # policy/input[:, :3] is the EEF keypoint in world, so keep the recorded
    # keypoint distance consistent with it for the recomputation check.
    keypoint_distance = np.linalg.norm(
        np.asarray(policy_input[:3], dtype=np.float64) - exact_keypoint[:3]
    )
    record = {
        "timing": {
            "ros_stamp_ns": np.int64(1_000_000_000 + index * 33_333_333),
            "robot_stamp_ns": np.int64(900_000_000 + index * 33_333_333),
            "monotonic_receive_ns": np.int64(2_000_000_000 + index * 33_333_333),
            "wrist_1_stamp_ns": np.int64(999_000_000 + index * 33_333_333),
            "wrist_2_stamp_ns": np.int64(998_000_000 + index * 33_333_333),
            "wrist_1_age_ns": np.int64(1_000_000),
            "wrist_2_age_ns": np.int64(2_000_000),
            "camera_skew_ns": np.int64(1_000_000),
        },
        "validity": {
            "robot_state": np.bool_(True),
            "wrist_1": np.bool_(True),
            "wrist_2": np.bool_(True),
            "policy_input": np.bool_(True),
        },
        "camera": {
            "wrist_1": {
                "rgb": np.full(image_shape, index % 255, dtype=np.uint8),
            },
            "wrist_2": {
                "rgb": np.full(image_shape, (index + 17) % 255, dtype=np.uint8),
            },
        },
        "robot": {
            "flange_pose_world_xyzw": flange,
            "tcp_pose_world_xyzw": tcp,
            "q": np.linspace(0.0, 0.6, 7, dtype=np.float64) + index * 0.001,
            "dq": np.zeros(7, dtype=np.float64),
            "ext_wrench_tcp": np.asarray(
                [1.0 + index, 2.0, 3.0, 0.1, 0.2, 0.3], dtype=np.float64
            ),
        },
        "task": {
            "socket_pose_supplied_world_xyzw": exact_root.copy(),
            "socket_root_exact_world_xyzw": exact_root,
            "socket_root_policy_world_xyzw": policy_root,
            "socket_keypoint_exact_world_xyzw": exact_keypoint,
            "socket_keypoint_policy_world_xyzw": policy_keypoint,
            "randomized_start_flange_world_xyzw": _pose(0.39, 0.102, 0.22),
            "flange_to_socket_root_exact_distance_m": np.float64(distance),
            "eef_keypoint_to_socket_keypoint_distance_m": np.float64(keypoint_distance),
        },
        "policy": {
            "input": policy_input,
        },
    }
    if include_extra:
        record["extra"] = {"ambient_temperature_c": np.float32(24.0 + index)}
    return record


def make_single_camera_observation(
    index: int,
    *,
    image_shape: tuple[int, int, int] = IMAGE_SHAPE,
    include_extra: bool = False,
) -> dict:
    record = make_observation(
        index,
        image_shape=image_shape,
        include_extra=include_extra,
    )
    for key in ("wrist_2_stamp_ns", "wrist_2_age_ns", "camera_skew_ns"):
        record["timing"].pop(key)
    record["validity"].pop("wrist_2")
    record["camera"].pop("wrist_2")
    return record


def make_action(
    index: int,
    *,
    training_action_clip: float = TRAINING_ACTION_CLIP,
    valid: bool = True,
    include_extra: bool = False,
) -> dict:
    mean = np.asarray([0.10, -0.10, 0.05, 0.0, 0.02, -0.02], dtype=np.float32)
    raw = np.asarray([1.20, -1.40, 0.30, -0.40, 0.20, -0.10], dtype=np.float32)
    raw += np.float32(index * 0.01)
    clipped = np.clip(raw, -training_action_clip, training_action_clip).astype(np.float32)
    effective = clipped.copy()
    residual = (effective - mean).astype(np.float32)
    if not valid:
        residual[:] = np.nan
    requested = _pose(0.402 + index * 0.002, 0.10, 0.208 - index * 0.001)
    effective_target = requested.copy()
    submitted_tcp = requested.copy()
    submitted_tcp[2] -= 0.01
    record = {
        "policy": {
            "mu_raw_normalized": mean,
            "std_raw_normalized": np.full(6, 0.15, dtype=np.float32),
            "sample_raw_normalized": raw,
            "sample_clipped_normalized": clipped,
            "recurrent_hidden_in": (
                np.arange(8, dtype=np.float32).reshape(2, 4) + index
            ),
            "recurrent_cell_in": (
                np.arange(8, 16, dtype=np.float32).reshape(2, 4) + index
            ),
            "recurrent_hidden_out": (
                np.arange(16, 24, dtype=np.float32).reshape(2, 4) + index
            ),
            "recurrent_cell_out": (
                np.arange(24, 32, dtype=np.float32).reshape(2, 4) + index
            ),
        },
        "action": {
            "delta_scaled_pose_rel": np.asarray(
                [0.001, 0.0, -0.001, 0.0, 0.0, 0.001], dtype=np.float64
            ),
            "requested_flange_target_world_xyzw": requested,
            "effective_software_flange_target_world_xyzw": effective_target,
            "submitted_controller_tcp_target_world_xyzw": submitted_tcp,
            "effective_software_action_normalized": effective,
            "command_published": np.bool_(valid),
            "guard_bitmask": np.uint32(0),
        },
        "timing": {
            "step_start_monotonic_ns": np.int64(2_010_000_000 + index * 33_333_333),
            "inference_ns": np.int64(2_000_000),
            "publish_ns": np.int64(150_000),
        },
        "labels": {
            "residual_effective_software_action_minus_raw_mean": residual,
            "command_valid": np.bool_(valid),
        },
    }
    if include_extra:
        record["actions"] = {"extra": {"controller_sequence": np.int64(index + 100)}}
    return record


def write_complete_trial(
    writer,
    *,
    attempt_index: int,
    accepted: bool,
    action_count: int = 2,
    trial_attrs: dict | None = None,
):
    trial = writer.start_trial(attempt_index=attempt_index, attrs=trial_attrs)
    trial.append_observation(make_observation(0))
    for index in range(action_count):
        trial.append_action(make_action(index))
        trial.append_observation(make_observation(index + 1))
    trial.finish(
        accepted=accepted,
        operator_label="confirmed_success" if accepted else "confirmed_failure",
        auto_outcome="success" if accepted else "timeout",
        termination_reason="success" if accepted else "timeout",
        accepted_success_index=action_count - 1 if accepted and action_count else None,
        reset_randomization_xyz_yaw=np.zeros(4, dtype=np.float64),
    )
    return trial
