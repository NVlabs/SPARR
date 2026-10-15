from dataclasses import replace
import math

import numpy as np

from sparr_phase1_collection.pose_math import Pose
from sparr_phase1_collection.robot_state import (
    RobotStateSnapshot,
    Vector3,
)
from sparr_phase1_collection.safety import (
    GuardStatus,
    PoseCommandLimits,
    RobotStateGuard,
    StateGuardConfig,
    guard_pose_command,
)


def _pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Pose:
    return Pose(
        np.asarray((x, y, z), dtype=np.float64),
        np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64),
    )


def _snapshot(*, ros_timestamp_ns: int = 100, robot_timestamp_ns: int = 50):
    joint = (0.0,) * 7
    return RobotStateSnapshot(
        ros_timestamp_ns=ros_timestamp_ns,
        robot_timestamp_ns=robot_timestamp_ns,
        frame_id="world",
        q=joint,
        theta=joint,
        dq=joint,
        dtheta=joint,
        tau=joint,
        tau_des=joint,
        tau_dot=joint,
        tau_ext=joint,
        tau_interact=joint,
        temperature=joint,
        tcp_pose=_pose(),
        tcp_velocity=None,
        flange_pose=_pose(),
        ft_sensor_raw=None,
        ext_wrench_in_tcp=None,
        ext_wrench_in_world=None,
        ext_wrench_in_tcp_raw=None,
        ext_wrench_in_world_raw=None,
        source_fields={},
    )


def test_valid_state_is_accepted_but_stale_state_holds():
    guard = RobotStateGuard(StateGuardConfig(max_state_age_ns=50))

    accepted = guard.evaluate(_snapshot(), 125)
    stale = guard.evaluate(_snapshot(ros_timestamp_ns=200, robot_timestamp_ns=51), 251)

    assert accepted.status is GuardStatus.ACCEPT
    assert stale.status is GuardStatus.HOLD
    assert stale.state_age_ns == 51
    assert "exceeds" in stale.reason


def test_invalid_state_is_rejected_not_clamped_or_held():
    state = _snapshot()
    invalid = replace(state, q=(math.nan,) + state.q[1:])
    guard = RobotStateGuard(StateGuardConfig(max_state_age_ns=50))

    decision = guard.evaluate(invalid, 100)

    assert decision.status is GuardStatus.REJECT
    assert "q contains a non-finite value" in decision.reasons
    assert math.isnan(invalid.q[0])


def test_unchanged_robot_payload_with_fresh_ros_stamp_holds():
    guard = RobotStateGuard(StateGuardConfig(max_state_age_ns=50))
    assert guard.evaluate(_snapshot(ros_timestamp_ns=100, robot_timestamp_ns=50), 100).accepted

    republished = _snapshot(ros_timestamp_ns=110, robot_timestamp_ns=50)
    decision = guard.evaluate(republished, 110)

    assert decision.status is GuardStatus.HOLD
    assert decision.reasons == ("robot payload timestamp did not advance",)


def test_regressed_or_far_future_timestamp_is_rejected():
    guard = RobotStateGuard(
        StateGuardConfig(max_state_age_ns=50, max_future_skew_ns=2)
    )
    assert guard.evaluate(_snapshot(ros_timestamp_ns=100, robot_timestamp_ns=50), 100).accepted
    regression = guard.evaluate(
        _snapshot(ros_timestamp_ns=101, robot_timestamp_ns=49), 101
    )
    future = RobotStateGuard(
        StateGuardConfig(max_state_age_ns=50, max_future_skew_ns=2)
    ).evaluate(_snapshot(ros_timestamp_ns=104), 100)

    assert regression.status is GuardStatus.REJECT
    assert future.status is GuardStatus.REJECT


def test_state_guard_rejects_a_mismatched_nested_pose_frame():
    state = replace(
        _snapshot(),
        source_fields={
            "flange_pose": {"header": {"frame_id": "base"}},
            "tcp_pose": {"header": {"frame_id": "world"}},
        },
    )
    guard = RobotStateGuard(
        StateGuardConfig(
            max_state_age_ns=50,
            expected_pose_frame_id="world",
        )
    )

    decision = guard.evaluate(state, 100)

    assert decision.status is GuardStatus.REJECT
    assert "flange_pose.header.frame_id 'base'" in decision.reason


def test_pose_guard_rejects_original_request_without_hidden_clamp():
    requested = _pose(2.0, 0.0, 0.0)
    limits = PoseCommandLimits(
        workspace_min=Vector3(-1.0, -1.0, -1.0),
        workspace_max=Vector3(1.0, 1.0, 1.0),
        max_translation_step_m=0.1,
    )

    decision = guard_pose_command(requested, _pose(), limits)

    assert decision.status is GuardStatus.REJECT
    assert requested.position[0] == 2.0
    assert any("outside" in reason for reason in decision.reasons)
    assert any("translation step" in reason for reason in decision.reasons)


def test_pose_guard_accepts_request_inside_explicit_limits():
    decision = guard_pose_command(
        _pose(0.01, 0.0, 0.0),
        _pose(),
        PoseCommandLimits(max_translation_step_m=0.02),
    )

    assert decision.status is GuardStatus.ACCEPT
