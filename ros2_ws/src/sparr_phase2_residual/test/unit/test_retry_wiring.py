# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA License.
# See the LICENSE file in the repository root for the full license text.

"""The seam between the retry policy and Phase 1's replay hook.

``_decide_retry`` is the only place that arms a replay, and what it arms is
``_repeat_targets`` -- the attribute Phase 1's ``_finish_trial`` reads to decide
whether the attempt index advances. These tests drive the real method on a bare
instance, because constructing the node needs a ROS context and a robot.
"""

from __future__ import annotations

import pytest

from sparr_phase2_residual.collector_node import Phase2CollectorNode
from sparr_phase2_residual.retry import RetryPolicy


class _Logger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def info(self, message: str) -> None:
        self.lines.append(message)

    warning = info
    error = info


class _Targets:
    """Stands in for TrialTargets; identity is all that is under test."""


@pytest.fixture
def node():
    instance = object.__new__(Phase2CollectorNode)
    instance._retry = RetryPolicy(
        enabled=True, success_rate_threshold=0.8, window=5, max_repeats=2
    )
    instance._targets = _Targets()
    instance._repeat_targets = None
    instance._rejected_before_rollout = False
    logger = _Logger()
    instance.get_logger = lambda: logger  # type: ignore[method-assign]
    return instance


def _warm_up(node, successes: int) -> None:
    for _ in range(successes):
        node._decide_retry(accepted=True, termination_reason="distance_threshold_reached")


class TestArmingAReplay:
    def test_a_failure_past_the_gate_arms_a_replay_with_the_same_targets(self, node):
        _warm_up(node, 5)
        targets = node._targets
        decision = node._decide_retry(
            accepted=False, termination_reason="operator_confirmed_failure"
        )
        assert decision.repeat
        # The identical object, not one re-derived from the index.
        assert node._repeat_targets is targets

    def test_a_failure_before_the_gate_leaves_the_hook_clear(self, node):
        decision = node._decide_retry(
            accepted=False, termination_reason="operator_confirmed_failure"
        )
        assert not decision.repeat
        assert node._repeat_targets is None

    def test_a_successful_replay_disarms_the_hook_and_forces_promotion(self, node):
        _warm_up(node, 5)
        node._decide_retry(accepted=False, termination_reason="operator_confirmed_failure")
        assert node._repeat_targets is not None

        decision = node._decide_retry(
            accepted=True, termination_reason="distance_threshold_reached"
        )
        assert decision.force_promote
        assert node._repeat_targets is None

    def test_exhausting_the_budget_disarms_the_hook(self, node):
        _warm_up(node, 5)
        for _ in range(2):
            assert node._decide_retry(
                accepted=False, termination_reason="operator_confirmed_failure"
            ).repeat
        decision = node._decide_retry(
            accepted=False, termination_reason="operator_confirmed_failure"
        )
        assert not decision.repeat
        assert not decision.force_promote
        # Left armed, the session would replay this attempt forever.
        assert node._repeat_targets is None


class TestTrialsWithNoVerdict:
    @pytest.mark.parametrize(
        "reason", ["operator_discarded", "operator_aborted_session", "node_shutdown"]
    )
    def test_they_do_not_move_the_success_window(self, node, reason):
        _warm_up(node, 5)
        assert node._retry.success_rate == pytest.approx(1.0)
        for _ in range(5):
            node._decide_retry(accepted=False, termination_reason=reason)
        # A reposition that never reached the start pose says nothing about
        # whether the policy can do the task; counting it would shut the gate.
        assert node._retry.success_rate == pytest.approx(1.0)

    def test_they_do_not_consume_a_repeat(self, node):
        _warm_up(node, 5)
        node._decide_retry(accepted=False, termination_reason="operator_confirmed_failure")
        assert node._retry.repeats_used == 1
        node._decide_retry(accepted=False, termination_reason="operator_discarded")
        assert node._retry.repeats_used == 0
        assert node._repeat_targets is None


class TestARejectBeforeTheRollout:
    def test_it_leaves_an_armed_replay_and_its_budget_untouched(self, node):
        _warm_up(node, 5)
        node._decide_retry(accepted=False, termination_reason="operator_confirmed_failure")
        armed = node._repeat_targets
        assert armed is not None and node._retry.repeats_used == 1

        node._rejected_before_rollout = True
        decision = node._decide_retry(accepted=False, termination_reason="reposition_timeout")
        assert not decision.repeat
        assert node._repeat_targets is armed
        assert node._retry.repeats_used == 1

    def test_it_does_not_move_the_success_window(self, node):
        _warm_up(node, 5)
        node._rejected_before_rollout = True
        for _ in range(5):
            node._decide_retry(accepted=False, termination_reason="reposition_timeout")
        assert node._retry.success_rate == pytest.approx(1.0)

