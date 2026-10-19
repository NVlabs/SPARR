# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from types import SimpleNamespace

import numpy as np
import pytest

from sparr_phase1_collection.robot_state import (
    RobotStateAdapterError,
    RobotStateSnapshot,
    adapt_robot_states,
)


def _stamp(timestamp_ns: int) -> SimpleNamespace:
    return SimpleNamespace(
        sec=timestamp_ns // 1_000_000_000,
        nanosec=timestamp_ns % 1_000_000_000,
    )


def _vector(x: float, y: float, z: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


def _pose(x: float, y: float, z: float, *, quaternion_w: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(
        position=_vector(x, y, z),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=quaternion_w),
    )


def _pose_stamped(x: float, y: float, z: float, *, quaternion_w: float = 1.0):
    return SimpleNamespace(
        header=SimpleNamespace(stamp=_stamp(123), frame_id="world"),
        pose=_pose(x, y, z, quaternion_w=quaternion_w),
    )


def _wrench_stamped():
    return SimpleNamespace(
        header=SimpleNamespace(stamp=_stamp(123), frame_id="world"),
        wrench=SimpleNamespace(
            force=_vector(1.0, 2.0, 3.0),
            torque=_vector(0.1, 0.2, 0.3),
        ),
    )


def _message(*, flange_quaternion_w: float = 1.0) -> SimpleNamespace:
    joint = [float(index) for index in range(7)]
    wrench = _wrench_stamped()
    return SimpleNamespace(
        header=SimpleNamespace(stamp=_stamp(2_000_000_025), frame_id="world"),
        robot_timestamp=_stamp(9_000_000_125),
        q=list(joint),
        theta=list(joint),
        dq=list(joint),
        dtheta=list(joint),
        tau=list(joint),
        tau_des=list(joint),
        tau_dot=list(joint),
        tau_ext=list(joint),
        tau_interact=list(joint),
        temperature=list(joint),
        tcp_pose=_pose_stamped(9.0, 9.0, 9.0),
        tcp_vel=SimpleNamespace(
            header=SimpleNamespace(stamp=_stamp(123), frame_id="world"),
            accel=SimpleNamespace(
                linear=_vector(0.1, 0.2, 0.3),
                angular=_vector(0.4, 0.5, 0.6),
            ),
        ),
        flange_pose=_pose_stamped(
            1.0, 2.0, 3.0, quaternion_w=flange_quaternion_w
        ),
        ft_sensor_raw=wrench,
        ext_wrench_in_tcp=wrench,
        ext_wrench_in_world=wrench,
        ext_wrench_in_tcp_raw=wrench,
        ext_wrench_in_world_raw=wrench,
        vendor_extension={"mode": "rt_cartesian", "flags": [1, 2]},
    )


def test_adapter_preserves_all_fields_and_uses_flange_as_task_anchor():
    message = _message()
    snapshot = adapt_robot_states(message)

    assert isinstance(snapshot, RobotStateSnapshot)
    assert snapshot.ros_timestamp_ns == 2_000_000_025
    assert snapshot.robot_timestamp_ns == 9_000_000_125
    assert snapshot.anchor_pose is snapshot.flange_pose
    np.testing.assert_array_equal(snapshot.success_pose.position, (1.0, 2.0, 3.0))
    np.testing.assert_array_equal(snapshot.tcp_pose.position, (9.0, 9.0, 9.0))
    np.testing.assert_array_equal(
        snapshot.success_pose.quaternion_xyzw, (0.0, 0.0, 0.0, 1.0)
    )
    assert snapshot.tcp_velocity.linear.as_tuple() == (0.1, 0.2, 0.3)
    assert snapshot.ext_wrench_in_tcp.force.as_tuple() == (1.0, 2.0, 3.0)
    assert snapshot.source_frame_id("flange_pose") == "world"
    assert snapshot.source_frame_id("tcp_pose") == "world"
    assert snapshot.source_fields["vendor_extension"]["mode"] == "rt_cartesian"
    assert snapshot.as_record()["vendor_extension"]["flags"] == [1, 2]
    assert snapshot.validation_errors() == ()


def test_pose_source_frame_contract_is_explicitly_validated():
    message = _message()
    message.flange_pose.header.frame_id = "base"
    snapshot = adapt_robot_states(message)

    errors = snapshot.validation_errors(expected_pose_frame_id="world")

    assert errors == (
        "flange_pose.header.frame_id 'base' does not match 'world'",
    )


def test_adapter_owns_values_and_source_payload_is_immutable():
    message = _message()
    snapshot = adapt_robot_states(message)
    message.q[0] = 999.0
    message.vendor_extension["flags"][0] = 999

    assert snapshot.q[0] == 0.0
    assert snapshot.source_fields["vendor_extension"]["flags"][0] == 1
    with pytest.raises(TypeError):
        snapshot.source_fields["new_field"] = 1


def test_invalid_quaternion_is_rejected_without_normalizing_the_source():
    message = _message(flange_quaternion_w=2.0)

    with pytest.raises(RobotStateAdapterError, match="quaternion is not unit length"):
        adapt_robot_states(message)
    assert message.flange_pose.pose.orientation.w == 2.0
