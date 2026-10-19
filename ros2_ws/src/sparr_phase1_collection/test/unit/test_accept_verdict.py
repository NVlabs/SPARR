# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""``accepted`` is the success verdict; ``export_eligible`` is the data gate.

Deterministic inference is never export-eligible, so conflating the two would
score every deterministic evaluation as 0% success. The Phase 2 demo exporter
gates on ``export_eligible`` itself.
"""

from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sparr_phase1_collection.collector_node import Phase1CollectorNode  # noqa: E402
from sparr_phase1_collection.trial_state_machine import (  # noqa: E402
    TrialEvent,
    TrialState,
    TrialStateMachine,
)


class _Writer:
    def __init__(self) -> None:
        self.finished: dict | None = None

    def finish_trial(self, **kwargs) -> None:
        self.finished = kwargs


@pytest.fixture
def node():
    """A collector stripped to just what ``_finish_trial`` touches."""
    instance = object.__new__(Phase1CollectorNode)
    instance._writer = _Writer()
    instance._accepted_successes = 0
    instance._attempt_index = 0
    instance._attempt_repeat_index = 0
    instance._repeat_targets = None
    instance._targets = None
    instance._trajectory = None
    instance._retraction_queue = []
    instance._trial_observation_count = 0
    instance._trial_action_count = 0
    instance._auto_outcome = "success_candidate"
    instance._terminal_reason = ""
    instance._rollout_started = True
    instance._rejected_before_rollout = False
    instance._trial_export_eligible = True
    instance._lock = threading.Lock()
    instance.get_logger = lambda: types.SimpleNamespace(
        warning=lambda *_: None, info=lambda *_: None, error=lambda *_: None
    )
    instance._session_complete = False
    instance._last_diagnostic_reason = ""
    instance._state_machine = TrialStateMachine()
    instance._config = types.SimpleNamespace(
        recording=types.SimpleNamespace(target_successes=1000)
    )
    instance._ensure_terminal_observation = lambda **_: None
    instance._sync_trial_attrs = lambda: {}
    instance._publish_diagnostics = lambda: None
    instance._close_writer = lambda **_: Path("unused.h5")
    return instance


def _finish(node, *, accepted: bool, eligible: bool):
    node._trial_export_eligible = eligible
    node._finish_trial(
        accepted=accepted,
        operator_label="success" if accepted else "failure",
        termination_reason="distance_threshold_reached",
    )
    return node._writer.finished


class TestTheVerdictIsRecordedHonestly:
    def test_a_deterministic_success_is_accepted_though_not_exportable(self, node):
        recorded = _finish(node, accepted=True, eligible=False)
        assert recorded["accepted"] is True
        assert recorded["attrs"]["export_eligible"] is False

    def test_a_deterministic_success_counts_toward_target_successes(self, node):
        _finish(node, accepted=True, eligible=False)
        # Otherwise target_successes is unreachable in deterministic runs.
        assert node._accepted_successes == 1

    def test_an_exportable_success_is_unchanged(self, node):
        recorded = _finish(node, accepted=True, eligible=True)
        assert recorded["accepted"] is True
        assert recorded["attrs"]["export_eligible"] is True
        assert node._accepted_successes == 1

    def test_a_failure_is_still_a_failure_either_way(self, node):
        recorded = _finish(node, accepted=False, eligible=True)
        assert recorded["accepted"] is False
        assert node._accepted_successes == 0

    def test_eligibility_alone_never_creates_a_success(self, node):
        _finish(node, accepted=False, eligible=False)
        assert node._accepted_successes == 0


class TestTheTrialBoundaryPublishesDiagnostics:
    def test_the_count_is_published_before_the_next_trial_can_be_read(self, node):
        # run_trials reads the cached diagnostics sample as soon as a trial is
        # confirmed, so the count must be published at the trial boundary.
        published: list[int] = []
        node._publish_diagnostics = lambda: published.append(node._accepted_successes)
        _finish(node, accepted=True, eligible=False)
        assert published == [1]


class TestARejectBeforeTheRolloutRerunsTheSameTrial:
    """The policy never ran, so the attempt has no outcome and is run again."""

    @staticmethod
    def _repositioning(node):
        node._rollout_started = False
        node._state_machine = TrialStateMachine()
        node._state_machine.transition(TrialEvent.BEGIN, 0)
        node._state_machine.transition(TrialEvent.PAUSE_ACKNOWLEDGED, 0)
        assert node._state_machine.state is TrialState.RETRACTING

    @staticmethod
    def _discard(node):
        response = types.SimpleNamespace(success=None, message="")
        node._discard_trial_service(None, response)
        assert response.success, response.message
        return node._writer.finished

    def test_the_attempt_index_does_not_advance(self, node):
        node._attempt_index = 7
        self._repositioning(node)
        node._reject_trial("reposition_timeout", 0)
        self._discard(node)
        assert node._attempt_index == 7
        assert node._attempt_repeat_index == 0

    def test_the_recorded_reason_is_the_reject_not_a_discard(self, node):
        self._repositioning(node)
        node._reject_trial("reposition_timeout", 0)
        recorded = self._discard(node)
        assert recorded["termination_reason"] == "reposition_timeout"
        assert recorded["accepted"] is False

    def test_an_armed_replay_stays_armed(self, node):
        targets = object()
        node._repeat_targets = targets
        self._repositioning(node)
        node._reject_trial("reposition_timeout", 0)
        self._discard(node)
        assert node._repeat_targets is targets

    def test_a_reject_during_the_rollout_moves_on(self, node):
        node._attempt_index = 7
        self._repositioning(node)
        node._rollout_started = True
        node._reject_trial("robot_state_invalid", 0)
        recorded = self._discard(node)
        assert node._attempt_index == 8
        assert recorded["termination_reason"] == "operator_discarded"

    def test_an_operator_skip_at_the_start_pose_moves_on(self, node):
        node._attempt_index = 7
        self._repositioning(node)
        self._discard(node)
        assert node._attempt_index == 8

