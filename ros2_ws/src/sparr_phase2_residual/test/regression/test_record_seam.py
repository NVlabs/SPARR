"""The Phase 1 record seam: inert without a residual, explicit with one.

The residual and whether it was commanded must be absent (not empty or zero)
from Phase 1 records, so Phase 1 files keep their exact shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
PHASE1_SRC = REPO_ROOT / "ros2_ws" / "src" / "sparr_phase1_collection"
if str(PHASE1_SRC) not in sys.path:
    sys.path.insert(0, str(PHASE1_SRC))

from sparr_phase1_collection.action_pipeline import (  # noqa: E402
    ActionPipelineConfig,
    build_action_trace,
)
from sparr_phase1_collection.pose_math import Pose  # noqa: E402
from sparr_phase1_collection.records import build_action_record  # noqa: E402

CONFIG = ActionPipelineConfig(
    position_scale=(0.004, 0.004, 0.004),
    orientation_scale=(0.02, 0.02, 0.02),
    action_clip=1.0,
)
FLANGE = Pose(np.array([0.4, 0.1, 0.3]), np.array([0.0, 0.0, 0.0, 1.0]))
FLANGE_TO_TCP = Pose(np.array([0.0, 0.0, 0.15]), np.array([0.0, 0.0, 0.0, 1.0]))
# A mean that saturates on two axes, so clip(mu + r) != clip(mu) + r.
MEAN = np.array([0.5, -2.2, 2.4, -0.3, 0.7, -0.1])
SAMPLE = np.array([0.9, -0.4, 0.2, -0.1, 0.3, 0.6])
RESIDUAL = np.array([0.05, -0.05, 0.1, 0.0, -0.02, 0.03])


def trace(**kwargs):
    return build_action_trace(
        raw_mean=MEAN,
        raw_sample=SAMPLE,
        observed_flange_pose=FLANGE,
        flange_to_tcp=FLANGE_TO_TCP,
        config=CONFIG,
        **kwargs,
    )


def record(one_trace, *, command_valid=True):
    return build_action_record(
        trace=one_trace,
        raw_std=np.full(6, 0.5, np.float32),
        recurrent_state_in=np.zeros((2, 2, 256), np.float32),
        recurrent_state_out=np.zeros((2, 2, 256), np.float32),
        recurrent_num_layers=2,
        recurrent_hidden_dim=256,
        step_start_monotonic_ns=1,
        command_ros_stamp_ns=2,
        inference_ns=3,
        composition_ns=4,
        publish_ns=5,
        recorder_enqueue_ns=6,
        writer_queue_depth=0,
        command_published=True,
        command_valid=command_valid,
    )


def test_a_phase1_record_has_no_residual_fields_at_all():
    """Not present-and-zero: absent, so Phase 1 files keep their exact shape."""
    labels = record(trace())["labels"]
    assert "extra" not in labels
    assert set(labels) == {
        "residual_effective_software_action_minus_raw_mean",
        "command_valid",
        "measured_valid",
    }


def test_a_phase1_trace_reports_no_residual():
    base = trace()
    assert base.requested_residual is None
    assert base.residual_applied is False


def test_an_applied_residual_is_recorded_and_marked_applied():
    applied = trace(residual=RESIDUAL)
    assert applied.residual_applied is True
    assert np.allclose(applied.requested_residual, RESIDUAL)

    extra = record(applied)["labels"]["extra"]
    assert np.allclose(extra["requested_residual"], RESIDUAL.astype(np.float32))
    assert extra["residual_applied"] is True
    assert extra["requested_residual"].dtype == np.float32


def test_a_shadow_mode_residual_is_recorded_but_marked_not_applied():
    """Shadow data must be distinguishable from active data after the fact."""
    from dataclasses import replace

    shadow = replace(trace(), requested_residual=RESIDUAL, residual_applied=False)
    extra = record(shadow)["labels"]["extra"]
    assert np.allclose(extra["requested_residual"], RESIDUAL.astype(np.float32))
    assert extra["residual_applied"] is False
    # ...and the commanded action is exactly what Phase 1 would have commanded.
    assert np.array_equal(
        shadow.effective_software_action, trace().effective_software_action
    )


def test_clipping_happens_after_the_addition_not_before():
    """The property that lets a residual steer an already-saturated action.

    The residual must pull the axis back inside the bound, or both orderings clip
    to the same value.
    """
    mean = np.array([-1.05, 1.05, 0.0, 0.0, 0.0, 0.0])
    residual = np.array([0.2, -0.2, 0.0, 0.0, 0.0, 0.0])
    assert np.any(np.abs(mean) > CONFIG.action_clip), "fixture must saturate"

    applied = build_action_trace(
        raw_mean=mean,
        raw_sample=SAMPLE,
        observed_flange_pose=FLANGE,
        flange_to_tcp=FLANGE_TO_TCP,
        config=CONFIG,
        residual=residual,
    )
    correct = np.clip(mean + residual, -1, 1)          # clip(mu + r)
    naive = np.clip(np.clip(mean, -1, 1) + residual, -1, 1)  # clip(clip(mu) + r)

    assert np.allclose(applied.effective_software_action, correct)
    assert not np.allclose(correct, naive), "fixture must distinguish the orderings"
    assert not np.allclose(applied.effective_software_action, naive)


def test_an_invalid_command_still_records_the_residual():
    """The residual is what the policy asked for, regardless of eligibility."""
    labels = record(trace(residual=RESIDUAL), command_valid=False)["labels"]
    assert np.all(np.isnan(labels["residual_effective_software_action_minus_raw_mean"]))
    assert np.allclose(labels["extra"]["requested_residual"], RESIDUAL.astype(np.float32))


@pytest.mark.parametrize("bad", [np.zeros(5), np.zeros(7), np.full(6, np.nan)])
def test_a_malformed_residual_is_rejected_by_the_pipeline(bad):
    with pytest.raises(ValueError):
        trace(residual=bad)
