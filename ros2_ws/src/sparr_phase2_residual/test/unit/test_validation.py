"""Phase 2 validation checks.

Each check is also given a deliberately corrupted recording and must fail on it.
"""

import numpy as np
import pytest

from sparr_phase2_residual.validation import (
    DEADLINE_FACTOR,
    NOMINAL_RATE_HZ,
    ValidationError,
    analyse,
    check_action_fidelity,
    check_control_rate,
    check_shadow_integrity,
    format_report,
    summarise,
)

h5py = pytest.importorskip("h5py")

PERIOD_NS = int(1e9 / NOMINAL_RATE_HZ)
CLIP = 1.0


def write_session(
    path,
    *,
    steps=20,
    trials=1,
    accepted=True,
    residual=None,          # None = a Phase 1 recording
    residual_applied=True,
    period_ns=PERIOD_NS,
    jitter_ns=None,
    corrupt_executed=False,
    mean=None,
):
    """A minimal but faithful session, with knobs for each failure mode."""
    with h5py.File(path, "w") as handle:
        handle.attrs["training_action_clip"] = CLIP
        group = handle.create_group("trials")
        for index in range(trials):
            trial = group.create_group(f"trial_{index:06d}")
            trial.attrs["accepted"] = accepted
            rng = np.random.default_rng(index)

            mu = (
                np.asarray(mean, np.float32)
                if mean is not None
                else rng.uniform(-1.5, 1.5, (steps, 6)).astype(np.float32)
            )
            clipped_sample = np.clip(
                rng.uniform(-1, 1, (steps, 6)), -CLIP, CLIP
            ).astype(np.float32)

            if residual is None:
                executed = clipped_sample.copy()
            else:
                res = np.broadcast_to(
                    np.asarray(residual, np.float32), (steps, 6)
                ).astype(np.float32)
                if residual_applied:
                    executed = np.clip(
                        mu.astype(np.float64) + res, -CLIP, CLIP
                    ).astype(np.float32)
                else:
                    executed = clipped_sample.copy()

            if corrupt_executed:
                executed = executed + np.float32(0.05)

            actions = trial.create_group("actions")
            actions.create_group("policy").create_dataset(
                "mu_raw_normalized", data=mu
            )
            trial["actions/policy"].create_dataset(
                "sample_clipped_normalized", data=clipped_sample
            )
            action = actions.create_group("action")
            action.create_dataset(
                "effective_software_action_normalized", data=executed
            )
            action.create_dataset("command_published", data=np.ones(steps, bool))

            starts = np.arange(steps, dtype=np.int64) * period_ns
            if jitter_ns is not None:
                starts = starts + np.asarray(jitter_ns, np.int64)
            timing = actions.create_group("timing")
            timing.create_dataset("step_start_monotonic_ns", data=starts)
            timing.create_dataset(
                "inference_ns", data=np.full(steps, 2_500_000, np.int64)
            )
            timing.create_dataset(
                "composition_ns", data=np.full(steps, 200_000, np.int64)
            )

            labels = trial.create_group("labels")
            labels.create_dataset("command_valid", data=np.ones(steps, bool))
            if residual is not None:
                extra = labels.create_group("extra")
                extra.create_dataset(
                    "requested_residual",
                    data=np.broadcast_to(
                        np.asarray(residual, np.float32), (steps, 6)
                    ).astype(np.float32),
                )
                extra.create_dataset(
                    "residual_applied", data=np.full(steps, residual_applied, bool)
                )
    return path


def only(path, name="trial_000000"):
    """Open one trial for a single check."""
    handle = h5py.File(path, "r")
    return handle, handle[f"trials/{name}"]


def test_a_clean_thirty_hertz_loop_passes(tmp_path):
    handle, trial = only(write_session(tmp_path / "s.h5"))
    with handle:
        rate, misses = check_control_rate(trial)
    assert rate.passed and misses.passed
    assert "30.00 Hz" in rate.detail


def test_a_slow_loop_fails(tmp_path):
    """20 Hz is outside the 27-33 Hz band and must not slip through."""
    handle, trial = only(write_session(tmp_path / "s.h5", period_ns=int(1e9 / 20)))
    with handle:
        rate, _ = check_control_rate(trial)
    assert rate.passed is False
    assert "20.00 Hz" in rate.detail


def test_occasional_long_steps_are_reported_as_deadline_misses(tmp_path):
    steps = 40
    jitter = np.zeros(steps, np.int64)
    # Push a fifth of the steps well past the deadline.
    jitter[10:] += int(DEADLINE_FACTOR * PERIOD_NS)
    jitter[20:] += int(DEADLINE_FACTOR * PERIOD_NS)
    jitter[30:] += int(DEADLINE_FACTOR * PERIOD_NS)
    handle, trial = only(
        write_session(tmp_path / "s.h5", steps=steps, jitter_ns=jitter)
    )
    with handle:
        _, misses = check_control_rate(trial)
    assert misses.passed is False


def test_too_few_steps_is_reported_as_not_measurable(tmp_path):
    handle, trial = only(write_session(tmp_path / "s.h5", steps=1))
    with handle:
        rate, _ = check_control_rate(trial)
    assert rate.passed is None


def test_a_correctly_applied_residual_passes_fidelity(tmp_path):
    handle, trial = only(
        write_session(tmp_path / "s.h5", residual=[0.1, -0.1, 0.05, 0, 0, 0])
    )
    with handle:
        check = check_action_fidelity(trial, CLIP)
    assert check.passed, check.detail


def test_fidelity_catches_a_command_that_does_not_match_the_residual(tmp_path):
    """The check exists for exactly this: recorded residual, different command."""
    handle, trial = only(
        write_session(
            tmp_path / "s.h5",
            residual=[0.1, -0.1, 0.05, 0, 0, 0],
            corrupt_executed=True,
        )
    )
    with handle:
        check = check_action_fidelity(trial, CLIP)
    assert check.passed is False
    assert "max|clip(mu+r) - executed|" in check.detail


def test_fidelity_holds_when_the_mean_is_saturated(tmp_path):
    """clip(mu+r), not clip(mu)+r -- the case that distinguishes them."""
    mean = np.tile(np.array([-1.4, 1.4, 0, 0, 0, 0], np.float32), (20, 1))
    handle, trial = only(
        write_session(
            tmp_path / "s.h5", residual=[0.6, -0.6, 0, 0, 0, 0], mean=mean
        )
    )
    with handle:
        check = check_action_fidelity(trial, CLIP)
    assert check.passed, check.detail


def test_a_phase1_recording_reports_fidelity_as_not_applicable(tmp_path):
    handle, trial = only(write_session(tmp_path / "s.h5", residual=None))
    with handle:
        check = check_action_fidelity(trial, CLIP)
    assert check.passed is None
    assert "Phase 1" in check.detail


def test_shadow_mode_commanding_the_phase1_action_passes(tmp_path):
    handle, trial = only(
        write_session(
            tmp_path / "s.h5",
            residual=[0.3, 0.3, 0.3, 0, 0, 0],
            residual_applied=False,
        )
    )
    with handle:
        check = check_shadow_integrity(trial, CLIP)
    assert check.passed, check.detail


def test_shadow_mode_that_actually_moved_the_robot_is_caught(tmp_path):
    """A residual leaking into the command in shadow mode is the worst failure."""
    handle, trial = only(
        write_session(
            tmp_path / "s.h5",
            residual=[0.3, 0.3, 0.3, 0, 0, 0],
            residual_applied=False,
            corrupt_executed=True,
        )
    )
    with handle:
        check = check_shadow_integrity(trial, CLIP)
    assert check.passed is False


def test_active_mode_reports_shadow_integrity_as_not_applicable(tmp_path):
    handle, trial = only(
        write_session(tmp_path / "s.h5", residual=[0.1] * 6, residual_applied=True)
    )
    with handle:
        check = check_shadow_integrity(trial, CLIP)
    assert check.passed is None


def test_trials_are_classified_by_mode(tmp_path):
    write_session(tmp_path / "p1.h5", residual=None)
    write_session(tmp_path / "shadow.h5", residual=[0.1] * 6, residual_applied=False)
    write_session(tmp_path / "active.h5", residual=[0.1] * 6, residual_applied=True)
    modes = {r.where.split("/")[0]: r.mode for r in analyse([tmp_path])}
    assert modes["p1.h5"] == "phase1"
    assert modes["shadow.h5"] == "shadow"
    assert modes["active.h5"] == "active"


def test_kpis_are_reported_per_mode_not_pooled(tmp_path):
    """Pooling shadow and active success rates would be meaningless."""
    write_session(tmp_path / "a.h5", residual=[0.1] * 6, residual_applied=False,
                  accepted=True, trials=2)
    write_session(tmp_path / "b.h5", residual=[0.1] * 6, residual_applied=True,
                  accepted=False, trials=3)
    summary = summarise(analyse([tmp_path]))
    assert summary["shadow"]["trials"] == 2
    assert summary["shadow"]["success_rate"] == 1.0
    assert summary["active"]["trials"] == 3
    assert summary["active"]["success_rate"] == 0.0
    assert "cycle_time_s" in summary["shadow"]
    assert "cycle_time_s" not in summary["active"], "no successes, so no cycle time"


def test_cycle_time_is_derived_from_the_measured_period(tmp_path):
    write_session(tmp_path / "s.h5", steps=60, accepted=True)
    summary = summarise(analyse([tmp_path]))
    assert summary["phase1"]["cycle_time_s"]["mean"] == pytest.approx(2.0, abs=0.05)


def test_the_report_states_what_it_cannot_measure(tmp_path):
    """Silently omitting these would read as 'everything checked out'."""
    write_session(tmp_path / "s.h5")
    text = format_report(analyse([tmp_path]))
    assert "not measurable from a recording" in text
    assert "reset reliability" in text
    assert "sample efficiency" in text


def test_a_failing_check_makes_the_overall_verdict_fail(tmp_path):
    write_session(tmp_path / "s.h5", period_ns=int(1e9 / 5))
    text = format_report(analyse([tmp_path]))
    assert "VALIDATION: FAIL" in text
    assert "control rate" in text


def test_a_clean_session_passes_overall(tmp_path):
    write_session(tmp_path / "s.h5", residual=[0.05] * 6)
    assert "VALIDATION: PASS" in format_report(analyse([tmp_path]))


def test_a_missing_path_is_an_error_not_an_empty_pass(tmp_path):
    with pytest.raises(ValidationError, match="no such file"):
        analyse([tmp_path / "absent"])


def test_directories_are_walked_recursively(tmp_path):
    (tmp_path / "nested").mkdir()
    write_session(tmp_path / "a.h5")
    write_session(tmp_path / "nested" / "b.h5")
    assert len(analyse([tmp_path])) == 2


def test_partial_files_are_ignored(tmp_path):
    write_session(tmp_path / "good.h5")
    (tmp_path / "open.partial.h5").write_bytes(b"not hdf5")
    assert len(analyse([tmp_path])) == 1
