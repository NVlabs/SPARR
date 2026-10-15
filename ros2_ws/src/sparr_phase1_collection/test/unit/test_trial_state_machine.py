import math

import numpy as np
import pytest

from sparr_phase1_collection.observations import ObservationConfig
from sparr_phase1_collection.pose_math import Pose
from sparr_phase1_collection.robot_state import RobotStateSnapshot
from sparr_phase1_collection.trial_state_machine import (
    SUCCESS_METRIC_EEF_KEYPOINT_TO_SOCKET_KEYPOINT,
    SUCCESS_METRIC_EEF_KEYPOINT_TO_SOCKET_KEYPOINT_AXIS_TOLERANCES,
    SUCCESS_METRIC_FLANGE_TO_SOCKET_ROOT,
    InvalidTrialTransition,
    PoseNoise,
    SampledPoseOffset,
    TrialConfigurationError,
    TrialEvent,
    TrialRandomizer,
    TrialState,
    TrialStateMachine,
    TrialTargets,
    evaluate_success,
)


APPROVED_OBSERVATION_CONFIG = ObservationConfig(
    eef_offset_flange=(0.0, 0.0, 0.1925),
    socket_keypoint_offset=(0.0375, 0.0, 0.0),
)


def _pose(x: float, y: float, z: float) -> Pose:
    return Pose(
        np.asarray((x, y, z), dtype=np.float64),
        np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64),
    )


def _snapshot(*, flange: Pose, tcp: Pose | None = None) -> RobotStateSnapshot:
    joint = (0.0,) * 7
    return RobotStateSnapshot(
        ros_timestamp_ns=100,
        robot_timestamp_ns=50,
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
        tcp_pose=tcp,
        tcp_velocity=None,
        flange_pose=flange,
        ft_sensor_raw=None,
        ext_wrench_in_tcp=None,
        ext_wrench_in_world=None,
        ext_wrench_in_tcp_raw=None,
        ext_wrench_in_world_raw=None,
        source_fields={},
    )


def test_seeded_start_and_policy_socket_noise_are_reproducible_and_bounded():
    nominal_start = _pose(0.4, -0.2, 0.3)
    exact_socket = _pose(0.5, 0.1, 0.2)

    first = TrialRandomizer(1234).sample(7, nominal_start, exact_socket)
    second = TrialRandomizer(1234).sample(7, nominal_start, exact_socket)

    assert first.start_offset == second.start_offset
    assert first.socket_policy_offset == second.socket_policy_offset
    np.testing.assert_array_equal(
        first.randomized_start.as_vector(), second.randomized_start.as_vector()
    )
    np.testing.assert_array_equal(
        first.socket_root_policy.as_vector(), second.socket_root_policy.as_vector()
    )
    assert first.socket_root_exact is exact_socket
    np.testing.assert_allclose(
        first.socket_root_policy.position - first.socket_root_exact.position,
        (
            first.socket_policy_offset.x_m,
            first.socket_policy_offset.y_m,
            first.socket_policy_offset.z_m,
        ),
    )
    assert abs(first.start_offset.x_m) <= 0.005
    assert abs(first.start_offset.y_m) <= 0.005
    assert abs(first.socket_policy_offset.x_m) <= 0.005
    assert abs(first.socket_policy_offset.y_m) <= 0.005
    assert first.start_offset.z_m == 0.0
    assert first.socket_policy_offset.z_m == 0.0
    assert first.start_offset.roll_rad == 0.0
    assert first.start_offset.pitch_rad == 0.0
    assert first.start_offset.yaw_rad == 0.0
    assert first.randomized_start.position[2] == nominal_start.position[2]
    np.testing.assert_array_equal(
        first.socket_root_policy.quaternion_xyzw,
        exact_socket.quaternion_xyzw,
    )


def test_noise_supports_independent_xy_and_z_half_ranges():
    noise = PoseNoise(xy_half_range_m=(0.001, 0.002), z_half_range_m=0.003)
    targets = TrialRandomizer(
        9,
        start_noise=noise,
        socket_policy_noise=noise,
    ).sample(
        2,
        _pose(0.0, 0.0, 0.0),
        _pose(1.0, 1.0, 1.0),
    )

    assert abs(targets.start_offset.x_m) <= 0.001
    assert abs(targets.start_offset.y_m) <= 0.002
    assert abs(targets.start_offset.z_m) <= 0.003
    np.testing.assert_allclose(
        targets.randomized_start.position,
        (
            targets.start_offset.x_m,
            targets.start_offset.y_m,
            targets.start_offset.z_m,
        ),
    )
    assert targets.start_offset.roll_rad == 0.0
    assert targets.start_offset.pitch_rad == 0.0
    assert targets.start_offset.yaw_rad == 0.0
    assert abs(targets.socket_policy_offset.x_m) <= 0.001
    assert abs(targets.socket_policy_offset.y_m) <= 0.002
    assert abs(targets.socket_policy_offset.z_m) <= 0.003
    np.testing.assert_allclose(
        targets.socket_root_policy.position - targets.socket_root_exact.position,
        (
            targets.socket_policy_offset.x_m,
            targets.socket_policy_offset.y_m,
            targets.socket_policy_offset.z_m,
        ),
    )


def test_success_uses_l2_from_exact_socket_root_to_flange_only():
    zero = SampledPoseOffset(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    exact_socket = _pose(0.0, 0.0, 0.0)
    policy_socket = _pose(1.0, 0.0, 0.0)
    targets = TrialTargets(
        seed=1,
        derived_seed=2,
        trial_index=0,
        nominal_start=_pose(0.0, 0.0, 1.0),
        randomized_start=_pose(0.0, 0.0, 1.0),
        socket_root_exact=exact_socket,
        socket_root_policy=policy_socket,
        start_offset=zero,
        socket_policy_offset=zero,
    )
    # Flange is at the noisy policy socket, while TCP is at exact ground truth.
    state = _snapshot(flange=policy_socket, tcp=exact_socket)

    result = targets.evaluate_success(state, threshold_m=0.1)

    assert result.succeeded is False
    assert result.distance_m == 1.0
    assert result.error_xyz_m == (1.0, 0.0, 0.0)

    near_state = _snapshot(flange=_pose(0.003, 0.004, 0.0), tcp=policy_socket)
    near_result = evaluate_success(near_state, exact_socket, threshold_m=0.005)
    assert near_result.succeeded is True
    assert math.isclose(near_result.distance_m, 0.005)


def test_keypoint_metric_measures_the_two_points_the_policy_observes():
    # Flange at the origin, socket root 0.5 m above it, both identity-rotated.
    flange = _pose(0.0, 0.0, 0.0)
    exact_socket = _pose(0.0, 0.0, 0.5)
    state = _snapshot(flange=flange, tcp=flange)

    # EEF keypoint = (0, 0, 0.1925); socket keypoint = (0.0375, 0, 0.5).
    expected = math.sqrt(0.0375**2 + (0.5 - 0.1925) ** 2)
    result = evaluate_success(
        state,
        exact_socket,
        threshold_m=0.01,
        metric=SUCCESS_METRIC_EEF_KEYPOINT_TO_SOCKET_KEYPOINT,
        observation_config=APPROVED_OBSERVATION_CONFIG,
    )
    assert result.metric == SUCCESS_METRIC_EEF_KEYPOINT_TO_SOCKET_KEYPOINT
    assert math.isclose(result.distance_m, expected, rel_tol=0.0, abs_tol=1e-12)
    assert result.succeeded is False

    # The offsets carry the pose rotation, so the same geometry with the socket
    # yawed by 180 degrees mirrors the socket keypoint offset.
    yawed_socket = Pose(
        np.asarray((0.0, 0.0, 0.5), dtype=np.float64),
        np.asarray((0.0, 0.0, 1.0, 0.0), dtype=np.float64),
    )
    yawed = evaluate_success(
        state,
        yawed_socket,
        threshold_m=0.01,
        metric=SUCCESS_METRIC_EEF_KEYPOINT_TO_SOCKET_KEYPOINT,
        observation_config=APPROVED_OBSERVATION_CONFIG,
    )
    assert math.isclose(yawed.distance_m, expected, rel_tol=0.0, abs_tol=1e-12)
    assert yawed.error_xyz_m[0] == pytest.approx(0.0375)

    # A perfectly aligned keypoint pair succeeds under a tight threshold.
    aligned_flange = _pose(0.0375, 0.0, 0.5 - 0.1925)
    aligned = evaluate_success(
        _snapshot(flange=aligned_flange),
        exact_socket,
        threshold_m=0.001,
        metric=SUCCESS_METRIC_EEF_KEYPOINT_TO_SOCKET_KEYPOINT,
        observation_config=APPROVED_OBSERVATION_CONFIG,
    )
    assert aligned.succeeded is True
    assert aligned.distance_m == pytest.approx(0.0, abs=1e-12)


def test_flange_metric_is_unchanged_and_ignores_keypoint_offsets():
    flange = _pose(0.0, 0.0, 0.0)
    exact_socket = _pose(0.0, 0.0, 0.5)
    state = _snapshot(flange=flange, tcp=flange)

    default = evaluate_success(state, exact_socket, threshold_m=0.01)
    assert default.metric == SUCCESS_METRIC_FLANGE_TO_SOCKET_ROOT
    assert default.distance_m == 0.5
    assert default.error_xyz_m == (0.0, 0.0, -0.5)

    explicit = evaluate_success(
        state,
        exact_socket,
        threshold_m=0.01,
        metric=SUCCESS_METRIC_FLANGE_TO_SOCKET_ROOT,
        observation_config=APPROVED_OBSERVATION_CONFIG,
    )
    assert explicit == default


def test_keypoint_metric_evaluates_against_exact_socket_not_the_noisy_one():
    zero = SampledPoseOffset(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    exact_socket = _pose(0.0375, 0.0, 0.5)
    policy_socket = _pose(1.0, 0.0, 0.5)
    targets = TrialTargets(
        seed=1,
        derived_seed=2,
        trial_index=0,
        nominal_start=_pose(0.0, 0.0, 1.0),
        randomized_start=_pose(0.0, 0.0, 1.0),
        socket_root_exact=exact_socket,
        socket_root_policy=policy_socket,
        start_offset=zero,
        socket_policy_offset=zero,
    )
    # Flange placed so the EEF keypoint lands exactly on the exact socket keypoint.
    flange = _pose(0.0375 + 0.0375, 0.0, 0.5 - 0.1925)
    result = targets.evaluate_success(
        _snapshot(flange=flange),
        threshold_m=0.001,
        metric=SUCCESS_METRIC_EEF_KEYPOINT_TO_SOCKET_KEYPOINT,
        observation_config=APPROVED_OBSERVATION_CONFIG,
    )
    assert result.succeeded is True


# Socket axis vertical: socket-frame X maps to world +Z, so world Z is the
# insertion axis, as the axis-tolerance metric assumes.
_AXIS_VERTICAL_SOCKET = Pose(
    np.asarray((0.5, 0.1, 0.2), dtype=np.float64),
    np.asarray((0.0, -math.sqrt(0.5), 0.0, math.sqrt(0.5)), dtype=np.float64),
)
_MEASURED_TOLERANCES = (0.005, 0.005, 0.045)


def _flange_for_world_residual(dx: float, dy: float, dz: float) -> Pose:
    """Flange pose whose EEF keypoint is (dx, dy, dz) from the socket keypoint."""

    socket_keypoint = _AXIS_VERTICAL_SOCKET.position + np.asarray(
        (0.0, 0.0, 0.0375), dtype=np.float64
    )
    return _pose(
        float(socket_keypoint[0]) + dx,
        float(socket_keypoint[1]) + dy,
        float(socket_keypoint[2]) + dz - 0.1925,
    )


def _axis_result(dx: float, dy: float, dz: float, tolerances=_MEASURED_TOLERANCES):
    return evaluate_success(
        _snapshot(flange=_flange_for_world_residual(dx, dy, dz)),
        _AXIS_VERTICAL_SOCKET,
        threshold_m=None,
        metric=SUCCESS_METRIC_EEF_KEYPOINT_TO_SOCKET_KEYPOINT_AXIS_TOLERANCES,
        observation_config=APPROVED_OBSERVATION_CONFIG,
        axis_tolerances_m=tolerances,
    )


def test_axis_tolerance_metric_accepts_the_measured_seated_insertion():
    # Seated insertions: millimetres laterally, ~41 mm in Z because the plug tip
    # extends past the fingertip keypoint the policy reports.
    for dx, dy, dz in (
        (0.0001, -0.0017, 0.0410),
        (-0.0001, 0.0021, 0.0416),
        (0.0007, 0.0013, 0.0405),
    ):
        result = _axis_result(dx, dy, dz)
        assert result.succeeded is True
        assert result.metric == (
            SUCCESS_METRIC_EEF_KEYPOINT_TO_SOCKET_KEYPOINT_AXIS_TOLERANCES
        )
        assert result.axis_tolerances_m == _MEASURED_TOLERANCES
        assert result.error_xyz_m == pytest.approx((dx, dy, dz), abs=1e-12)
        # distance_m stays the ordinary L2 norm for reporting, even though the
        # box decided the outcome, and no scalar radius applies.
        assert result.distance_m == pytest.approx(
            math.sqrt(dx * dx + dy * dy + dz * dz), abs=1e-12
        )
        assert math.isnan(result.threshold_m)


def test_axis_tolerance_metric_fails_when_any_single_axis_is_exceeded():
    assert _axis_result(0.0007, 0.0013, 0.046).succeeded is False
    assert _axis_result(0.0007, 0.006, 0.0405).succeeded is False
    assert _axis_result(-0.0051, 0.0013, 0.0405).succeeded is False
    # Exactly on every bound passes; tolerances are taken from the computed
    # residual so the comparison is bit-exact.
    on_bound = _axis_result(0.005, -0.005, 0.045)
    assert on_bound.succeeded is False  # 0.005 + 1 ulp is outside the box
    exact = _axis_result(
        0.005,
        -0.005,
        0.045,
        tolerances=tuple(abs(component) for component in on_bound.error_xyz_m),
    )
    assert exact.succeeded is True
    # The scalar L2 metric rejects the same geometry: the Z gap exceeds any
    # usable lateral radius.
    scalar = evaluate_success(
        _snapshot(flange=_flange_for_world_residual(0.0007, 0.0013, 0.0405)),
        _AXIS_VERTICAL_SOCKET,
        threshold_m=0.005,
        metric=SUCCESS_METRIC_EEF_KEYPOINT_TO_SOCKET_KEYPOINT,
        observation_config=APPROVED_OBSERVATION_CONFIG,
    )
    assert scalar.succeeded is False


def test_axis_tolerance_metric_uses_the_exact_socket_and_the_checkpoint_offsets():
    zero = SampledPoseOffset(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    policy_socket = Pose(
        _AXIS_VERTICAL_SOCKET.position + np.asarray((1.0, 0.0, 0.0)),
        _AXIS_VERTICAL_SOCKET.quaternion_xyzw,
    )
    targets = TrialTargets(
        seed=1,
        derived_seed=2,
        trial_index=0,
        nominal_start=_pose(0.0, 0.0, 1.0),
        randomized_start=_pose(0.0, 0.0, 1.0),
        socket_root_exact=_AXIS_VERTICAL_SOCKET,
        socket_root_policy=policy_socket,
        start_offset=zero,
        socket_policy_offset=zero,
    )
    result = targets.evaluate_success(
        _snapshot(flange=_flange_for_world_residual(0.0007, 0.0013, 0.0405)),
        threshold_m=None,
        metric=SUCCESS_METRIC_EEF_KEYPOINT_TO_SOCKET_KEYPOINT_AXIS_TOLERANCES,
        observation_config=APPROVED_OBSERVATION_CONFIG,
        axis_tolerances_m=_MEASURED_TOLERANCES,
    )
    assert result.succeeded is True

    with pytest.raises(TrialConfigurationError, match="observation_config"):
        evaluate_success(
            _snapshot(flange=_pose(0.0, 0.0, 0.0)),
            _AXIS_VERTICAL_SOCKET,
            threshold_m=None,
            metric=SUCCESS_METRIC_EEF_KEYPOINT_TO_SOCKET_KEYPOINT_AXIS_TOLERANCES,
            axis_tolerances_m=_MEASURED_TOLERANCES,
        )


@pytest.mark.parametrize(
    "tolerances",
    [None, (0.005, 0.005), (0.005, 0.005, 0.045, 0.1), (0.005, 0.0, 0.045),
     (0.005, -0.005, 0.045), (0.005, 0.005, float("nan")), 0.005],
)
def test_axis_tolerance_metric_rejects_malformed_tolerances(tolerances):
    with pytest.raises(TrialConfigurationError):
        _axis_result(0.0, 0.0, 0.0, tolerances=tolerances)


def test_axis_tolerances_and_scalar_threshold_are_mutually_exclusive():
    state = _snapshot(flange=_pose(0.0, 0.0, 0.0))
    with pytest.raises(TrialConfigurationError, match="threshold_m must be None"):
        evaluate_success(
            state,
            _AXIS_VERTICAL_SOCKET,
            threshold_m=0.01,
            metric=SUCCESS_METRIC_EEF_KEYPOINT_TO_SOCKET_KEYPOINT_AXIS_TOLERANCES,
            observation_config=APPROVED_OBSERVATION_CONFIG,
            axis_tolerances_m=_MEASURED_TOLERANCES,
        )
    for metric in (
        SUCCESS_METRIC_FLANGE_TO_SOCKET_ROOT,
        SUCCESS_METRIC_EEF_KEYPOINT_TO_SOCKET_KEYPOINT,
    ):
        with pytest.raises(
            TrialConfigurationError, match="axis_tolerances_m must be None"
        ):
            evaluate_success(
                state,
                _AXIS_VERTICAL_SOCKET,
                threshold_m=0.01,
                metric=metric,
                observation_config=APPROVED_OBSERVATION_CONFIG,
                axis_tolerances_m=_MEASURED_TOLERANCES,
            )


def test_unknown_success_metric_is_rejected():
    state = _snapshot(flange=_pose(0.0, 0.0, 0.0))
    with pytest.raises(TrialConfigurationError, match="unsupported success metric"):
        evaluate_success(
            state,
            _pose(0.0, 0.0, 0.5),
            threshold_m=0.01,
            metric="tcp_z_below_socket",
        )


def test_keypoint_metric_requires_the_checkpoint_offsets():
    state = _snapshot(flange=_pose(0.0, 0.0, 0.0))
    with pytest.raises(TrialConfigurationError, match="observation_config"):
        evaluate_success(
            state,
            _pose(0.0, 0.0, 0.5),
            threshold_m=0.01,
            metric=SUCCESS_METRIC_EEF_KEYPOINT_TO_SOCKET_KEYPOINT,
        )


def test_success_threshold_is_required():
    with pytest.raises(TypeError):
        evaluate_success(_snapshot(flange=_pose(0.0, 0.0, 0.0)), _pose(0.0, 0.0, 0.0))


def test_state_machine_requires_acknowledged_sequence_and_supports_hold_resume():
    machine = TrialStateMachine()
    machine.transition(TrialEvent.BEGIN, 1)
    machine.transition(TrialEvent.PAUSE_ACKNOWLEDGED, 2)
    machine.transition(TrialEvent.RETRACTED, 3)
    machine.transition(TrialEvent.START_REACHED, 4)
    machine.transition(TrialEvent.SETTLED, 5)
    machine.transition(TrialEvent.POLICY_RESET_ACKNOWLEDGED, 6)
    assert machine.state is TrialState.RUNNING

    machine.transition(TrialEvent.SAFETY_HOLD, 7, reason="state stale")
    assert machine.state is TrialState.HOLDING
    machine.transition(TrialEvent.STATE_FRESH, 8)
    machine.transition(TrialEvent.SUCCESS, 9)
    assert machine.state is TrialState.SUCCEEDED
    assert machine.is_terminal

    machine.transition(TrialEvent.CLEAR, 10)
    assert machine.state is TrialState.IDLE
    assert [transition.timestamp_ns for transition in machine.history] == list(range(1, 11))


def test_state_machine_rejects_skipped_or_out_of_order_transition():
    machine = TrialStateMachine()
    with pytest.raises(InvalidTrialTransition):
        machine.transition(TrialEvent.SUCCESS, 1)

    machine.transition(TrialEvent.BEGIN, 2)
    with pytest.raises(InvalidTrialTransition):
        machine.transition(TrialEvent.RETRACTED, 3)
    with pytest.raises(InvalidTrialTransition, match="monotonic"):
        machine.transition(TrialEvent.PAUSE_ACKNOWLEDGED, 1)


class TestResumingASessionAtALaterIndex:
    """`first_attempt_index` continues a seed's stream instead of restarting it."""

    @staticmethod
    def _randomizer():
        return TrialRandomizer(
            0,
            start_noise=PoseNoise(xy_half_range_m=(0.005, 0.005)),
            socket_policy_noise=PoseNoise(xy_half_range_m=(0.001, 0.001)),
        )

    @staticmethod
    def _pose():
        return Pose(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))

    def test_resuming_lands_on_the_uninterrupted_sequence(self):
        nominal = socket = self._pose()
        uninterrupted = self._randomizer()
        continuous = [
            uninterrupted.sample(i, nominal, socket) for i in range(41)
        ][31:]

        resumed_rng = self._randomizer()
        resumed = [resumed_rng.sample(i, nominal, socket) for i in range(31, 41)]

        for a, b in zip(continuous, resumed):
            assert a.socket_policy_offset == b.socket_policy_offset
            assert a.start_offset == b.start_offset

    def test_a_resumed_run_does_not_repeat_the_earlier_offsets(self, ):
        nominal = socket = self._pose()
        rng = self._randomizer()
        earlier = [rng.sample(i, nominal, socket).socket_policy_offset
                   for i in range(31)]
        later = [self._randomizer().sample(i, nominal, socket).socket_policy_offset
                 for i in range(31, 41)]
        for offset in later:
            assert offset not in earlier
