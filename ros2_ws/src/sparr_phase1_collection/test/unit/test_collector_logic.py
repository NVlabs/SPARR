# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA License.
# See the LICENSE file in the repository root for the full license text.

import pytest

from sparr_phase1_collection.collector_logic import (
    RolloutInputDisposition,
    TerminalObservationPlan,
    classify_rollout_input,
    plan_terminal_observation,
    policy_goal_publish_enabled,
    trial_export_eligible,
)
from sparr_phase1_collection.safety import GuardStatus


@pytest.mark.parametrize(
    ("guard", "camera_valid", "policy_valid", "expected"),
    [
        (GuardStatus.ACCEPT, True, True, RolloutInputDisposition.ACCEPT),
        (GuardStatus.HOLD, True, True, RolloutInputDisposition.HOLD),
        (GuardStatus.ACCEPT, False, False, RolloutInputDisposition.HOLD),
        (GuardStatus.REJECT, True, False, RolloutInputDisposition.REJECT),
        (GuardStatus.ACCEPT, True, False, RolloutInputDisposition.REJECT),
    ],
)
def test_rollout_input_disposition_preserves_hold_vs_reject(
    guard, camera_valid, policy_valid, expected
):
    assert (
        classify_rollout_input(
            guard_status=guard,
            camera_pair_valid=camera_valid,
            policy_input_valid=policy_valid,
        )
        is expected
    )


def test_terminal_observation_plan_accepts_only_streaming_invariants():
    assert plan_terminal_observation(4, 3) is TerminalObservationPlan.ALREADY_COMPLETE
    assert plan_terminal_observation(3, 3) is TerminalObservationPlan.APPEND

    with pytest.raises(RuntimeError, match="N=2, M=3"):
        plan_terminal_observation(2, 3)
    with pytest.raises(RuntimeError, match="N=5, M=3"):
        plan_terminal_observation(5, 3)
    with pytest.raises(ValueError):
        plan_terminal_observation(-1, 0)


def test_only_live_trials_are_export_eligible():
    assert trial_export_eligible("live") is True
    assert trial_export_eligible("shadow") is False
    with pytest.raises(ValueError, match="unknown execution mode"):
        trial_export_eligible("other")


def test_deterministic_inference_trials_are_never_export_eligible():
    # Deterministic (mean-action) rollouts produce a ~zero residual label.
    assert trial_export_eligible("live", True) is False
    assert trial_export_eligible("shadow", True) is False
    assert trial_export_eligible("live", False) is True


def test_shadow_suppresses_goal_publication_side_channel():
    assert policy_goal_publish_enabled("live", True) is True
    assert policy_goal_publish_enabled("shadow", True) is False
    assert policy_goal_publish_enabled("live", False) is False
