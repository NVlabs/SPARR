# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""The retry policy: when a failed attempt is replayed, and what it costs."""

from __future__ import annotations

import pytest

from sparr_phase2_residual.retry import RetryError, RetryPolicy


def _policy(**kwargs) -> RetryPolicy:
    defaults = dict(
        enabled=True, success_rate_threshold=0.8, window=5, max_repeats=3
    )
    defaults.update(kwargs)
    return RetryPolicy(**defaults)


def _fill(policy: RetryPolicy, successes: int, failures: int) -> None:
    """Fill the window without tripping a replay, by keeping the gate shut."""
    for _ in range(successes):
        policy.consider(succeeded=True)
    for _ in range(failures):
        policy.consider(succeeded=False)


class TestTheGate:
    def test_a_failure_is_not_replayed_before_the_window_has_filled(self) -> None:
        policy = _policy(window=5)
        # Three successes then a failure is 75% -- but on four attempts, which
        # is exactly the small sample the window length exists to refuse.
        for _ in range(3):
            assert not policy.consider(succeeded=True).repeat
        decision = policy.consider(succeeded=False)
        assert not decision.repeat
        assert decision.success_rate is None
        assert "not established" in decision.reason

    def test_a_failure_below_the_threshold_is_not_replayed(self) -> None:
        policy = _policy(window=5, success_rate_threshold=0.8)
        _fill(policy, successes=2, failures=2)
        decision = policy.consider(succeeded=False)
        # 2/5 = 40%, well under 80%.
        assert not decision.repeat
        assert decision.success_rate == pytest.approx(0.4)

    def test_a_failure_at_the_threshold_is_replayed(self) -> None:
        policy = _policy(window=5, success_rate_threshold=0.8)
        _fill(policy, successes=4, failures=0)
        decision = policy.consider(succeeded=False)
        # The window is [T,T,T,T,F] = 80%, which is >= the threshold.
        assert decision.repeat
        assert decision.success_rate == pytest.approx(0.8)

    def test_disabled_never_replays_however_good_the_policy_is(self) -> None:
        policy = _policy(enabled=False, window=5)
        _fill(policy, successes=5, failures=0)
        decision = policy.consider(succeeded=False)
        assert not decision.repeat
        assert decision.reason == "retry disabled"


class TestReplaySequences:
    def test_a_successful_replay_is_force_promoted_and_ends_the_sequence(self) -> None:
        policy = _policy(window=5)
        _fill(policy, successes=5, failures=0)
        assert policy.consider(succeeded=False).repeat
        assert policy.replaying

        decision = policy.consider(succeeded=True)
        assert not decision.repeat
        assert decision.force_promote
        assert not policy.replaying
        assert decision.repeats_used == 1

    def test_replays_stop_at_max_repeats(self) -> None:
        policy = _policy(window=5, max_repeats=3)
        _fill(policy, successes=5, failures=0)
        for expected in (1, 2, 3):
            decision = policy.consider(succeeded=False)
            assert decision.repeat
            assert decision.repeats_used == expected
        # The fourth failure exhausts the budget rather than replaying again.
        decision = policy.consider(succeeded=False)
        assert not decision.repeat
        assert not decision.force_promote
        assert "gave up" in decision.reason
        assert not policy.replaying

    def test_replay_outcomes_do_not_move_the_success_window(self) -> None:
        # Replays run on conditions that already failed; counting them would shut
        # the gate that opened them.
        policy = _policy(window=5, max_repeats=3)
        _fill(policy, successes=5, failures=0)
        policy.consider(succeeded=False)  # the original failure: counted
        assert policy.success_rate == pytest.approx(0.8)
        for _ in range(3):
            policy.consider(succeeded=False)  # replays: not counted
        assert policy.success_rate == pytest.approx(0.8)

    def test_promotion_can_be_turned_off_while_replaying_stays_on(self) -> None:
        policy = _policy(window=5, promote_on_success=False)
        _fill(policy, successes=5, failures=0)
        policy.consider(succeeded=False)
        assert not policy.consider(succeeded=True).force_promote

    def test_an_ordinary_success_is_never_force_promoted(self) -> None:
        policy = _policy(window=5)
        _fill(policy, successes=4, failures=0)
        decision = policy.consider(succeeded=True)
        assert not decision.force_promote
        assert not decision.repeat

    def test_abandoning_a_replay_frees_the_budget(self) -> None:
        policy = _policy(window=5, max_repeats=3)
        _fill(policy, successes=5, failures=0)
        policy.consider(succeeded=False)
        assert policy.repeats_used == 1
        policy.abandon()
        assert policy.repeats_used == 0
        assert not policy.replaying


class TestConfigurationIsRejectedEarly:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"success_rate_threshold": -0.1},
            {"success_rate_threshold": 1.5},
            {"window": 0},
            {"max_repeats": 0},
        ],
    )
    def test_out_of_range_values_raise(self, kwargs) -> None:
        with pytest.raises(RetryError):
            _policy(**kwargs)
