# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA OneWay Noncommercial License (March 22, 2022).
# See the LICENSE file in the repository root for the full license text:
# https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf

import math

import numpy as np

from sparr_phase1_collection.motion import TimedPoseTrajectory, pose_error
from sparr_phase1_collection.pose_math import Pose


def test_reset_trajectory_reaches_goal_without_overshoot():
    start = Pose(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
    goal = Pose(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 1.0, 0.0]))
    trajectory = TimedPoseTrajectory(start, goal, start_time_ns=100, duration_ns=100)

    midpoint = trajectory.sample(150)
    assert np.allclose(midpoint.position, [0.5, 1.0, 1.5])
    assert math.isclose(np.linalg.norm(midpoint.quaternion_xyzw), 1.0)
    assert np.allclose(trajectory.sample(300).as_vector(), goal.as_vector())
    assert trajectory.complete(200)


def test_pose_error_treats_q_and_negative_q_as_same_orientation():
    pose = Pose(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
    equivalent = Pose(np.zeros(3), np.array([0.0, 0.0, 0.0, -1.0]))
    translation, rotation = pose_error(pose, equivalent)
    assert translation == 0.0
    assert rotation == 0.0


def test_axial_retraction_clears_the_socket_before_moving_sideways():
    """The geometry behind trial_control.retraction_clearance_m.

    A straight line to the reset pose drags the still-engaged plug sideways,
    which can bind it in the socket.
    """
    level = np.array([0.0, 0.0, 0.0, 1.0])
    seated = Pose(np.array([0.4755, 0.1210, 0.2630]), level)
    reset = Pose(np.array([0.4748, 0.1290, 0.3020]), level)   # +8 mm y, +39 mm z
    engaged_depth_m = 0.010

    # Straight line from seated to the reset pose.
    direct = TimedPoseTrajectory(seated, reset, start_time_ns=0, duration_ns=1_000)
    # Sample where the tool has risen by the engagement depth.
    fraction = engaged_depth_m / (reset.position[2] - seated.position[2])
    while_engaged = direct.sample(int(fraction * 1_000))
    lateral_drag = abs(while_engaged.position[1] - seated.position[1])
    assert lateral_drag > 0.002, "the straight-line reset drags the plug sideways"

    # Axial lift from the current tool pose.
    lifted = Pose(seated.position + np.array([0.0, 0.0, 0.040]), level)
    axial = TimedPoseTrajectory(seated, lifted, start_time_ns=0, duration_ns=1_000)
    for t in range(0, 1_001, 50):
        sample = axial.sample(t)
        assert np.allclose(sample.position[:2], seated.position[:2]), (
            "axial retraction must not move laterally at any point"
        )
    assert axial.sample(1_000).position[2] > seated.position[2] + engaged_depth_m, (
        "the lift must clear the engagement depth"
    )


def test_levelling_the_tool_pivots_about_the_plug_tip_not_the_flange():
    """The property trial_control.level_before_retract depends on.

    Levelling must keep the plug tip fixed; pivoting about the flange would swing
    the tip by about eef_offset * sin(tilt).
    """
    from sparr_phase1_collection.collector_node import _matrix_to_quaternion_xyzw
    from sparr_phase1_collection.pose_math import quaternion_to_matrix_xyzw

    eef_offset = np.array([0.0, 0.0, 0.1925])
    axis = np.array([0.0, 0.0, 1.0])            # socket insertion axis, world
    tilt = math.radians(3.89)

    # A flange tilted by `tilt` about world x, tool pointing broadly down.
    base = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    lean = np.array([[1.0, 0.0, 0.0],
                     [0.0, math.cos(tilt), -math.sin(tilt)],
                     [0.0, math.sin(tilt), math.cos(tilt)]])
    rotation = lean @ base
    seated = Pose(np.array([0.4755, 0.1210, 0.2630]),
                  _matrix_to_quaternion_xyzw(rotation))

    tip_before = seated.position + quaternion_to_matrix_xyzw(
        seated.quaternion_xyzw) @ eef_offset

    # Level it the way the collector does: rotate the tool axis onto -axis,
    # then translate so the tip lands back where it started.
    tool = quaternion_to_matrix_xyzw(seated.quaternion_xyzw) @ np.array([0.0, 0.0, 1.0])
    cross = np.cross(tool, -axis)
    sin_a = float(np.linalg.norm(cross))
    k = cross / sin_a
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    delta = np.eye(3) + math.sin(tilt) * kx + (1.0 - math.cos(tilt)) * (kx @ kx)
    levelled_rotation = delta @ quaternion_to_matrix_xyzw(seated.quaternion_xyzw)
    levelled = Pose(tip_before - levelled_rotation @ eef_offset,
                    _matrix_to_quaternion_xyzw(levelled_rotation))

    tip_after = levelled.position + quaternion_to_matrix_xyzw(
        levelled.quaternion_xyzw) @ eef_offset
    assert np.allclose(tip_before, tip_after, atol=1e-9), "the plug tip must not move"

    # ...and the tool really is on the socket axis afterwards.
    tool_after = quaternion_to_matrix_xyzw(levelled.quaternion_xyzw) @ np.array(
        [0.0, 0.0, 1.0])
    assert np.allclose(tool_after, -axis, atol=1e-9)

    # Rotating about the flange instead moves the tip.
    naive_tip = seated.position + levelled_rotation @ eef_offset
    assert np.linalg.norm(naive_tip - tip_before) > 0.010, (
        "pivoting about the flange would drag the plug >10 mm sideways"
    )
