# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import json
from dataclasses import replace

import numpy as np
import pytest

from sparr_phase1_collection.action_pipeline import (
    ActionPipelineConfig,
    build_action_trace,
)
from sparr_phase1_collection.camera_buffer import CameraBuffer, TwoCameraBuffer
from sparr_phase1_collection.observations import ObservationConfig
from sparr_phase1_collection.pose_math import Pose
from sparr_phase1_collection.records import build_action_record, build_observation_record
from sparr_phase1_collection.robot_state import (
    RobotStateSnapshot,
    SpatialVector,
    Vector3,
    Wrench,
)
from sparr_phase1_collection.trial_state_machine import PoseNoise, TrialRandomizer


def _snapshot(flange: Pose) -> RobotStateSnapshot:
    tcp = Pose(flange.position + np.array([0.0, 0.0, -0.02]), flange.quaternion_xyzw)
    wrench = Wrench(Vector3(1.0, 2.0, 3.0), Vector3(0.1, 0.2, 0.3))
    joints = tuple(float(value) for value in range(7))
    return RobotStateSnapshot(
        ros_timestamp_ns=1_000,
        robot_timestamp_ns=900,
        frame_id="world",
        q=joints,
        theta=joints,
        dq=joints,
        dtheta=joints,
        tau=joints,
        tau_des=joints,
        tau_dot=joints,
        tau_ext=joints,
        tau_interact=joints,
        temperature=joints,
        tcp_pose=tcp,
        tcp_velocity=SpatialVector(Vector3(0.1, 0.2, 0.3), Vector3(0.4, 0.5, 0.6)),
        flange_pose=flange,
        ft_sensor_raw=wrench,
        ext_wrench_in_tcp=wrench,
        ext_wrench_in_world=wrench,
        ext_wrench_in_tcp_raw=wrench,
        ext_wrench_in_world_raw=wrench,
        source_fields={
            "header": {"frame_id": "world", "stamp": {"sec": 0, "nanosec": 1000}},
            "vendor_future_field": [7, 8, 9],
        },
    )


def test_records_preserve_wrench_socket_images_and_effective_label():
    flange = Pose.from_vector([0.4, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0])
    socket = Pose.from_vector([0.475, 0.128, 0.0325, 0.5, 0.5, 0.5, -0.5])
    zero_noise = PoseNoise(
        xy_half_range_m=(0.0, 0.0),
        z_half_range_m=0.0,
        roll_half_range_rad=0.0,
        pitch_half_range_rad=0.0,
        yaw_half_range_rad=0.0,
    )
    targets = TrialRandomizer(
        17, start_noise=zero_noise, socket_policy_noise=zero_noise
    ).sample(3, flange, socket)
    cameras = TwoCameraBuffer(
        ("wrist_1", "wrist_2"),
        (128, 128),
        crops={"wrist_1": (0, 0, 16, 12), "wrist_2": (0, 0, 16, 12)},
    )
    source = np.zeros((12, 16, 3), dtype=np.uint8)
    cameras.add_frame(
        "wrist_1", 995, source, metadata={"frame_id": "w1", "encoding": "rgb8"}
    )
    cameras.add_frame(
        "wrist_2", 996, source, metadata={"frame_id": "w2", "encoding": "rgb8"}
    )
    pair = cameras.closest_pair(1_000, max_delta_ns=10, max_camera_skew_ns=10)
    assert pair is not None
    observation_config = ObservationConfig(
        eef_offset_flange=(0.0, 0.0, 0.0),
        socket_keypoint_offset=(0.0375, 0.0, 0.0),
    )
    observation = build_observation_record(
        snapshot=_snapshot(flange),
        receive_monotonic_ns=2_000,
        control_tick_monotonic_ns=2_100,
        cameras=pair,
        targets=targets,
        policy_input=np.zeros(18, dtype=np.float32),
        observation_config=observation_config,
        robot_state_valid=True,
        camera_validity={"wrist_1": True, "wrist_2": True},
        policy_input_valid=True,
    )

    assert observation["camera"]["wrist_1"]["rgb"].shape == (128, 128, 3)
    assert observation["camera"]["wrist_2"]["rgb"].shape == (128, 128, 3)
    np.testing.assert_allclose(
        observation["robot"]["ext_wrench_tcp"], [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
    )
    np.testing.assert_allclose(
        observation["task"]["socket_pose_supplied_world_xyzw"], socket.as_vector()
    )
    np.testing.assert_allclose(
        observation["task"]["socket_root_exact_world_xyzw"], socket.as_vector()
    )
    np.testing.assert_allclose(
        observation["task"]["socket_root_policy_world_xyzw"], socket.as_vector()
    )
    raw = json.loads(observation["extra"]["raw_robot_states_json"])
    assert raw["header"]["frame_id"] == "world"
    assert raw["vendor_future_field"] == [7, 8, 9]

    single_buffer = CameraBuffer(
        ("wrist_1",),
        (128, 128),
        crops={"wrist_1": (0, 0, 16, 12)},
    )
    single_buffer.add_frame(
        "wrist_1", 995, source, metadata={"frame_id": "w1", "encoding": "rgb8"}
    )
    single_frames = single_buffer.closest_pair(
        1_000, max_delta_ns=10, max_camera_skew_ns=None
    )
    assert single_frames is not None
    single_observation = build_observation_record(
        snapshot=_snapshot(flange),
        receive_monotonic_ns=2_000,
        control_tick_monotonic_ns=2_100,
        cameras=single_frames,
        targets=targets,
        policy_input=np.zeros(18, dtype=np.float32),
        observation_config=observation_config,
        robot_state_valid=True,
        camera_validity={"wrist_1": True},
        policy_input_valid=True,
    )
    assert tuple(single_observation["camera"]) == ("wrist_1",)
    assert "wrist_2" not in single_observation["validity"]
    assert "wrist_2_stamp_ns" not in single_observation["timing"]
    assert "camera_skew_ns" not in single_observation["timing"]
    assert "wrist_2_source_shape" not in single_observation["extra"]

    invalid_observation = build_observation_record(
        snapshot=replace(_snapshot(flange), flange_pose=None),
        receive_monotonic_ns=2_000,
        control_tick_monotonic_ns=2_100,
        cameras=pair,
        targets=targets,
        policy_input=np.full(18, np.nan, dtype=np.float32),
        observation_config=observation_config,
        robot_state_valid=False,
        camera_validity={"wrist_1": True, "wrist_2": True},
        policy_input_valid=False,
    )
    assert np.isnan(invalid_observation["robot"]["flange_pose_world_xyzw"]).all()
    assert np.isnan(
        invalid_observation["task"]["flange_to_socket_root_exact_distance_m"]
    )
    assert np.isnan(
        invalid_observation["task"]["eef_keypoint_to_socket_keypoint_distance_m"]
    )

    config = ActionPipelineConfig(
        position_scale=(0.003125, 0.003125, 0.003125),
        orientation_scale=(0.003125, 0.003125, 0.003125),
        action_clip=1.0,
    )
    mean = np.asarray([0.1, -0.2, 0.0, 0.3, 0.0, -0.1])
    sample = np.asarray([2.0, -2.0, 0.5, -0.5, 0.25, 0.0])
    trace = build_action_trace(
        raw_mean=mean,
        raw_sample=sample,
        observed_flange_pose=flange,
        flange_to_tcp=Pose.from_vector([0.0, 0.0, -0.02, 0.0, 0.0, 0.0, 1.0]),
        config=config,
    )
    action = build_action_record(
        trace=trace,
        raw_std=np.full(6, 0.5),
        recurrent_state_in=np.arange(8, dtype=np.float32),
        recurrent_state_out=np.arange(8, 16, dtype=np.float32),
        recurrent_num_layers=2,
        recurrent_hidden_dim=2,
        step_start_monotonic_ns=3_000,
        command_ros_stamp_ns=1_000,
        inference_ns=100,
        composition_ns=20,
        publish_ns=10,
        recorder_enqueue_ns=5,
        writer_queue_depth=0,
        command_published=True,
        command_valid=True,
    )
    np.testing.assert_allclose(
        action["action"]["effective_software_action_normalized"],
        np.clip(sample, -1.0, 1.0),
    )
    np.testing.assert_allclose(
        action["labels"]["residual_effective_software_action_minus_raw_mean"],
        np.clip(sample, -1.0, 1.0) - mean,
    )
    assert "accepted" not in action["action"]
    assert "executed" not in action["action"]
    np.testing.assert_array_equal(
        action["policy"]["recurrent_hidden_in"], [[0.0, 1.0], [2.0, 3.0]]
    )
    np.testing.assert_array_equal(
        action["policy"]["recurrent_cell_in"], [[4.0, 5.0], [6.0, 7.0]]
    )
    np.testing.assert_array_equal(
        action["policy"]["recurrent_hidden_out"], [[8.0, 9.0], [10.0, 11.0]]
    )
    np.testing.assert_array_equal(
        action["policy"]["recurrent_cell_out"], [[12.0, 13.0], [14.0, 15.0]]
    )
    assert "extra" not in action


def test_deterministic_action_record_keeps_mu_std_and_sample():
    flange = Pose.from_vector([0.4, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0])
    config = ActionPipelineConfig(
        position_scale=(0.003125, 0.003125, 0.003125),
        orientation_scale=(0.003125, 0.003125, 0.003125),
        action_clip=1.0,
    )
    mean = np.asarray([0.1, -0.2, 0.0, 0.3, 0.0, -0.1])
    sample = np.asarray([2.0, -2.0, 0.5, -0.5, 0.25, 0.0])
    std = np.full(6, 0.5)
    trace = build_action_trace(
        raw_mean=mean,
        raw_sample=sample,
        observed_flange_pose=flange,
        flange_to_tcp=Pose.from_vector([0.0, 0.0, -0.02, 0.0, 0.0, 0.0, 1.0]),
        config=config,
        deterministic=True,
    )
    action = build_action_record(
        trace=trace,
        raw_std=std,
        recurrent_state_in=np.arange(8, dtype=np.float32),
        recurrent_state_out=np.arange(8, 16, dtype=np.float32),
        recurrent_num_layers=2,
        recurrent_hidden_dim=2,
        step_start_monotonic_ns=3_000,
        command_ros_stamp_ns=1_000,
        inference_ns=100,
        composition_ns=20,
        publish_ns=10,
        recorder_enqueue_ns=5,
        writer_queue_depth=0,
        command_published=True,
        command_valid=True,
    )

    # The full distribution trace is still recorded in deterministic mode.
    np.testing.assert_allclose(action["policy"]["mu_raw_normalized"], mean)
    np.testing.assert_allclose(action["policy"]["std_raw_normalized"], std)
    np.testing.assert_allclose(action["policy"]["sample_raw_normalized"], sample)
    np.testing.assert_allclose(
        action["policy"]["sample_clipped_normalized"], np.clip(sample, -1.0, 1.0)
    )
    np.testing.assert_allclose(
        action["policy"]["sample_standard_normal"], (sample - mean) / std, rtol=1e-6
    )
    np.testing.assert_array_equal(
        action["policy"]["recurrent_hidden_in"], [[0.0, 1.0], [2.0, 3.0]]
    )
    np.testing.assert_array_equal(
        action["policy"]["recurrent_cell_out"], [[12.0, 13.0], [14.0, 15.0]]
    )

    # Only the effective action and its label switch to the mean.
    np.testing.assert_allclose(
        action["action"]["effective_software_action_normalized"],
        np.clip(mean, -1.0, 1.0),
    )
    np.testing.assert_allclose(
        action["labels"]["residual_effective_software_action_minus_raw_mean"],
        np.zeros(6),
    )


def _trial_seed_of(trial_index: int, *, seed: int = 17) -> int:
    flange = Pose.from_vector([0.4, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0])
    socket = Pose.from_vector([0.475, 0.128, 0.0325, 0.5, 0.5, 0.5, -0.5])
    targets = TrialRandomizer(seed).sample(trial_index, flange, socket)
    cameras = CameraBuffer(
        ("wrist_1",), (128, 128), crops={"wrist_1": (0, 0, 16, 12)}
    )
    cameras.add_frame(
        "wrist_1",
        995,
        np.zeros((12, 16, 3), dtype=np.uint8),
        metadata={"frame_id": "w1", "encoding": "rgb8"},
    )
    frames = cameras.closest_pair(1_000, max_delta_ns=10, max_camera_skew_ns=None)
    assert frames is not None
    observation = build_observation_record(
        snapshot=_snapshot(flange),
        receive_monotonic_ns=2_000,
        control_tick_monotonic_ns=2_100,
        cameras=frames,
        targets=targets,
        policy_input=np.zeros(18, dtype=np.float32),
        observation_config=ObservationConfig(
            eef_offset_flange=(0.0, 0.0, 0.0),
            socket_keypoint_offset=(0.0375, 0.0, 0.0),
        ),
        robot_state_valid=True,
        camera_validity={"wrist_1": True},
        policy_input_valid=True,
    )
    recorded = observation["extra"]["trial_seed"]
    assert recorded.dtype == np.uint64
    assert recorded == np.uint64(targets.derived_seed)
    return int(recorded)


def test_recorded_trial_seed_is_the_per_trial_derived_seed():
    first = _trial_seed_of(0)
    second = _trial_seed_of(1)

    # The base config seed is not what identifies a trial's random stream.
    assert first != 0 and second != 0
    assert first != 17 and second != 17
    assert first != second
    assert _trial_seed_of(0) == first
    assert _trial_seed_of(0, seed=18) != first


def test_observation_records_the_eef_to_socket_keypoint_distance():
    # Checkpoint geometry: tool tip 19.25 cm along +Z of the flange and the
    # socket keypoint 3.75 cm along +X of the socket root.
    observation_config = ObservationConfig(
        eef_offset_flange=(0.0, 0.0, 0.1925),
        socket_keypoint_offset=(0.0375, 0.0, 0.0),
    )
    flange = Pose.from_vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    socket = Pose.from_vector([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])
    zero_noise = PoseNoise(
        xy_half_range_m=(0.0, 0.0),
        z_half_range_m=0.0,
        roll_half_range_rad=0.0,
        pitch_half_range_rad=0.0,
        yaw_half_range_rad=0.0,
    )
    targets = TrialRandomizer(
        5, start_noise=zero_noise, socket_policy_noise=zero_noise
    ).sample(0, flange, socket)
    cameras = CameraBuffer(("wrist_1",), (128, 128), crops={"wrist_1": (0, 0, 16, 12)})
    cameras.add_frame(
        "wrist_1",
        995,
        np.zeros((12, 16, 3), dtype=np.uint8),
        metadata={"frame_id": "w1", "encoding": "rgb8"},
    )
    frames = cameras.closest_pair(1_000, max_delta_ns=10, max_camera_skew_ns=None)
    assert frames is not None

    observation = build_observation_record(
        snapshot=_snapshot(flange),
        receive_monotonic_ns=2_000,
        control_tick_monotonic_ns=2_100,
        cameras=frames,
        targets=targets,
        policy_input=np.zeros(18, dtype=np.float32),
        observation_config=observation_config,
        robot_state_valid=True,
        camera_validity={"wrist_1": True},
        policy_input_valid=True,
    )

    task = observation["task"]
    expected = float(np.sqrt(0.0375**2 + (0.5 - 0.1925) ** 2))
    assert task["eef_keypoint_to_socket_keypoint_distance_m"] == pytest.approx(expected)
    # This field remains the raw flange-to-socket-root distance.
    assert task["flange_to_socket_root_exact_distance_m"] == pytest.approx(0.5)
    # The keypoint distance is measured against the exact socket keypoint.
    assert task["eef_keypoint_to_socket_keypoint_distance_m"] == pytest.approx(
        float(
            np.linalg.norm(
                np.asarray(task["socket_keypoint_exact_world_xyzw"][:3])
                - np.asarray([0.0, 0.0, 0.1925])
            )
        )
    )
