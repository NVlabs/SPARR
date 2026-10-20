# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import h5py

from sparr_phase1_collection.async_writer import AsyncHDF5SessionWriter
from sparr_phase1_collection.validation import validate_file


def _observation(value):
    robot = np.array([value, value, value, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return {
        "timing": {
            "ros_stamp_ns": value,
            "robot_stamp_ns": value,
            "monotonic_receive_ns": value,
            "wrist_1_stamp_ns": value,
            "wrist_2_stamp_ns": value,
            "wrist_1_age_ns": 0,
            "wrist_2_age_ns": 0,
            "camera_skew_ns": 0,
        },
        "validity": {
            "robot_state": True,
            "wrist_1": True,
            "wrist_2": True,
            "policy_input": True,
        },
        "camera": {
            "wrist_1": {"rgb": np.zeros((128, 128, 3), dtype=np.uint8)},
            "wrist_2": {"rgb": np.zeros((128, 128, 3), dtype=np.uint8)},
        },
        "robot": {
            "flange_pose_world_xyzw": robot,
            "tcp_pose_world_xyzw": robot,
            "q": robot,
            "dq": robot,
            "ext_wrench_tcp": np.zeros(6),
        },
        "task": {
            "socket_root_exact_world_xyzw": robot,
            "socket_root_policy_world_xyzw": robot,
            "socket_keypoint_exact_world_xyzw": robot,
            "socket_keypoint_policy_world_xyzw": robot,
            "randomized_start_flange_world_xyzw": robot,
            "flange_to_socket_root_exact_distance_m": 0.0,
        },
        "policy": {"input": np.zeros(18, dtype=np.float32)},
    }


def _action():
    pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    return {
        "policy": {
            "mu_raw_normalized": np.zeros(6, dtype=np.float32),
            "sample_raw_normalized": np.zeros(6, dtype=np.float32),
            "sample_clipped_normalized": np.zeros(6, dtype=np.float32),
        },
        "action": {
            "delta_scaled_pose_rel": np.zeros(6),
            "requested_flange_target_world_xyzw": pose,
            "effective_software_flange_target_world_xyzw": pose,
            "submitted_controller_tcp_target_world_xyzw": pose,
            "effective_software_action_normalized": np.zeros(6, dtype=np.float32),
            "command_published": True,
            "guard_bitmask": 0,
        },
        "timing": {
            "step_start_monotonic_ns": 1,
            "inference_ns": 1,
            "publish_ns": -1,
        },
        "labels": {
            "residual_effective_software_action_minus_raw_mean": np.zeros(
                6, dtype=np.float32
            ),
            "command_valid": True,
        },
    }


def test_async_writer_preserves_transition_order(tmp_path):
    writer = AsyncHDF5SessionWriter(
        tmp_path / "session.h5",
        queue_size=8,
        training_action_clip=1.0,
        flush_every=1,
    )
    writer.start_trial(attempt_index=0)
    writer.append_observation(_observation(1))
    writer.append_action(_action())
    writer.append_observation(_observation(2))
    writer.finish_trial(
        accepted=True,
        operator_label="success",
        auto_outcome="success_candidate",
        termination_reason="operator_confirmed",
    )
    final_path = writer.close(clean=True)
    assert final_path.suffix == ".h5"
    assert validate_file(final_path).is_valid


def test_async_invalidation_is_ordered_and_quarantines_pending_label(tmp_path):
    writer = AsyncHDF5SessionWriter(
        tmp_path / "invalidated.h5",
        queue_size=8,
        training_action_clip=1.0,
        flush_every=1,
    )
    writer.start_trial(attempt_index=0, attrs={"export_eligible": False})
    writer.append_observation(_observation(1))
    action = _action()
    action["action"]["measured_next_normalized"] = np.zeros(6, dtype=np.float32)
    action["labels"]["residual_measured_minus_raw_mean"] = np.zeros(
        6, dtype=np.float32
    )
    action["labels"]["measured_valid"] = True
    writer.append_action(action)

    assert writer.invalidate_pending_action(guard_bitmask=0x20)
    writer.append_observation(_observation(2))
    writer.finish_trial(
        accepted=False,
        operator_label="input_invalid",
        auto_outcome="rejected",
        termination_reason="stale_input",
    )
    final_path = writer.close(clean=True)

    with h5py.File(final_path, "r") as file:
        trial = file["trials/trial_000000"]
        assert not bool(trial["labels/command_valid"][0])
        assert not bool(trial["labels/measured_valid"][0])
        assert np.isnan(
            trial["labels/residual_effective_software_action_minus_raw_mean"][0]
        ).all()
        assert np.isnan(trial["labels/residual_measured_minus_raw_mean"][0]).all()
        assert int(trial["actions/action/guard_bitmask"][0]) == 0x20
        assert int(trial["actions/mapping/next_observation_index"][0]) == 1
    assert validate_file(final_path).is_valid
