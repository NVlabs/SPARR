# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA License.
# See the LICENSE file in the repository root for the full license text.

import math

import numpy as np

from sparr_phase1_collection.pose_math import (
    Pose,
    apply_world_pose_delta,
    compose_poses,
    flange_to_tcp_pose,
    invert_pose,
    quaternion_to_matrix_xyzw,
    rotvec_to_quaternion_xyzw,
    world_pose_delta,
)


def test_world_delta_uses_world_translation_and_left_multiplied_rotation():
    base = Pose(
        np.array([0.4, -0.2, 0.6]),
        rotvec_to_quaternion_xyzw([math.pi / 2.0, 0.0, 0.0]),
    )
    translation_world = np.array([0.01, 0.0, 0.0])
    rotation_world = np.array([0.0, 0.0, math.pi / 3.0])

    target = apply_world_pose_delta(base, translation_world, rotation_world)

    # Translation is not rotated by the flange orientation.
    np.testing.assert_allclose(target.position, base.position + translation_world, atol=1e-12)
    expected_rotation = (
        quaternion_to_matrix_xyzw(rotvec_to_quaternion_xyzw(rotation_world))
        @ quaternion_to_matrix_xyzw(base.quaternion_xyzw)
    )
    np.testing.assert_allclose(
        quaternion_to_matrix_xyzw(target.quaternion_xyzw), expected_rotation, atol=1e-12
    )

    recovered_translation, recovered_rotation = world_pose_delta(base, target)
    np.testing.assert_allclose(recovered_translation, translation_world, atol=1e-12)
    np.testing.assert_allclose(recovered_rotation, rotation_world, atol=1e-12)


def test_flange_tcp_conversion_round_trips_rigid_offset():
    flange = Pose(
        np.array([0.5, 0.1, 0.3]),
        rotvec_to_quaternion_xyzw([0.0, 0.0, math.pi / 2.0]),
    )
    flange_to_tcp = Pose(
        np.array([0.1, 0.0, 0.02]),
        rotvec_to_quaternion_xyzw([0.0, math.pi / 6.0, 0.0]),
    )

    tcp = flange_to_tcp_pose(flange, flange_to_tcp)
    # Local +X rotates to world +Y under the flange's 90-degree yaw.
    np.testing.assert_allclose(tcp.position, [0.5, 0.2, 0.32], atol=1e-12)

    recovered_flange = compose_poses(tcp, invert_pose(flange_to_tcp))
    np.testing.assert_allclose(recovered_flange.position, flange.position, atol=1e-12)
    np.testing.assert_allclose(
        quaternion_to_matrix_xyzw(recovered_flange.quaternion_xyzw),
        quaternion_to_matrix_xyzw(flange.quaternion_xyzw),
        atol=1e-12,
    )
