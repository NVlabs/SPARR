# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math

import numpy as np

from sparr_phase1_collection.observations import (
    ObservationConfig,
    encode_phase1_observation,
)
from sparr_phase1_collection.pose_math import Pose, rotvec_to_quaternion_xyzw


def test_exact_18d_identity_observation_with_both_keypoint_offsets():
    config = ObservationConfig(
        eef_offset_flange=(0.0, 0.0, 0.1925),
        socket_keypoint_offset=(0.0375, 0.0, 0.0),
    )
    flange = Pose(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0, 1.0]))
    socket = Pose(np.array([0.4, 0.5, 0.6]), np.array([0.0, 0.0, 0.0, 1.0]))

    observation = encode_phase1_observation(flange, socket, config, dtype=np.float64)

    expected_rot6d = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    expected = np.array(
        [
            1.0,
            2.0,
            3.1925,
            *expected_rot6d,
            0.4375,
            0.5,
            0.6,
            *expected_rot6d,
        ]
    )
    assert observation.shape == (18,)
    np.testing.assert_allclose(observation, expected, atol=1e-12)


def test_offsets_rotate_from_each_pose_local_frame_into_world():
    yaw_90 = rotvec_to_quaternion_xyzw([0.0, 0.0, math.pi / 2.0])
    config = ObservationConfig(
        eef_offset_flange=(0.1, 0.0, 0.0),
        socket_keypoint_offset=(0.2, 0.0, 0.0),
    )
    flange = Pose(np.zeros(3), yaw_90)
    socket = Pose(np.array([1.0, 2.0, 3.0]), yaw_90)

    observation = encode_phase1_observation(flange, socket, config, dtype=np.float64)

    np.testing.assert_allclose(observation[0:3], [0.0, 0.1, 0.0], atol=1e-12)
    np.testing.assert_allclose(observation[9:12], [1.0, 2.2, 3.0], atol=1e-12)
    expected_rot6d = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0]
    np.testing.assert_allclose(observation[3:9], expected_rot6d, atol=1e-12)
    np.testing.assert_allclose(observation[12:18], expected_rot6d, atol=1e-12)
