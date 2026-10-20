# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Regression: Phase 2 must not alter Phase 1's action pipeline.

Replays recorded Phase 1 trials through ``action_pipeline.build_action_trace``
with ``residual=None`` and asserts every output is bit-identical to what was
commanded. Recordings carry their own scales, clip and flange-to-TCP transform.
Without recordings the tests skip; :func:`test_recordings_are_available` makes
that visible.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
PHASE1_SRC = REPO_ROOT / "ros2_ws" / "src" / "sparr_phase1_collection"
RECORDINGS = REPO_ROOT / "data" / "phase1"

if str(PHASE1_SRC) not in sys.path:
    sys.path.insert(0, str(PHASE1_SRC))

h5py = pytest.importorskip("h5py")

from sparr_phase1_collection.action_pipeline import (  # noqa: E402
    ActionPipelineConfig,
    build_action_trace,
)
from sparr_phase1_collection.pose_math import Pose  # noqa: E402

#: Digest of one fixed pre-Phase-2 trial: detects change, not correctness
#: (``test_replay_reproduces_the_commanded_actions_exactly`` checks that).
GOLDEN_TRIAL = "sparr_phase1_20260820T165619_506982Z.h5/trial_000000"
GOLDEN_DIGEST = "2b2ba2fc36bda46fea183ccb7a8403c3"


@dataclass(frozen=True)
class RecordedTrial:
    """Everything one trial needs, read out while its file is still open.

    h5py datasets raise closed-file errors once their ``with`` block exits.
    """

    where: str
    stochastic: bool
    config: ActionPipelineConfig
    flange_to_tcp: Pose
    mu: np.ndarray
    sample: np.ndarray
    flange: np.ndarray
    observation_index: np.ndarray
    effective_action: np.ndarray
    flange_target: np.ndarray
    tcp_target: np.ndarray
    label: np.ndarray
    command_valid: np.ndarray

    def __len__(self) -> int:
        return len(self.mu)


def _pose(row) -> Pose:
    """A recorded [x, y, z, qx, qy, qz, qw] row as a Pose."""
    row = np.asarray(row, np.float64)
    return Pose(row[:3], row[3:])


def _sessions() -> list[Path]:
    if not RECORDINGS.is_dir():
        return []
    return sorted(p for p in RECORDINGS.glob("*.h5") if ".partial." not in p.name)


def load_trials() -> list[RecordedTrial]:
    trials: list[RecordedTrial] = []
    for session in _sessions():
        try:
            handle = h5py.File(session, "r")
        except OSError:
            continue  # a truncated or still-open session, not a regression
        with handle:
            if "metadata/policy_spec" not in handle or "trials" not in handle:
                continue
            stochastic_flag = handle.get("metadata/collection_config/policy/stochastic")
            if stochastic_flag is None:
                continue
            stochastic = bool(stochastic_flag[()])
            spec = handle["metadata/policy_spec"]
            config = ActionPipelineConfig(
                position_scale=tuple(spec["position_scale"][:]),
                orientation_scale=tuple(spec["orientation_scale"][:]),
                action_clip=float(handle.attrs["training_action_clip"]),
            )
            for name in sorted(handle["trials"]):
                trial = handle[f"trials/{name}"]
                required = (
                    "labels/residual_effective_software_action_minus_raw_mean",
                    "actions/action/effective_software_flange_target_world_xyzw",
                    "actions/action/submitted_controller_tcp_target_world_xyzw",
                    "actions/mapping/observation_index",
                    "observations/robot/flange_pose_world_xyzw",
                    "labels/command_valid",
                    "actions/policy/mu_raw_normalized",
                    "actions/policy/sample_raw_normalized",
                    "actions/action/effective_software_action_normalized",
                    "labels/residual_effective_software_action_minus_raw_mean",
                )
                if any(path not in trial for path in required):
                    continue
                if "flange_to_controller_tcp_xyzw" not in trial.attrs:
                    continue
                trials.append(
                    RecordedTrial(
                        where=f"{session.name}/{name}",
                        stochastic=stochastic,
                        config=config,
                        flange_to_tcp=_pose(trial.attrs["flange_to_controller_tcp_xyzw"]),
                        mu=trial["actions/policy/mu_raw_normalized"][:],
                        sample=trial["actions/policy/sample_raw_normalized"][:],
                        flange=trial["observations/robot/flange_pose_world_xyzw"][:],
                        observation_index=trial["actions/mapping/observation_index"][:],
                        effective_action=trial[
                            "actions/action/effective_software_action_normalized"
                        ][:],
                        flange_target=trial[
                            "actions/action/effective_software_flange_target_world_xyzw"
                        ][:],
                        tcp_target=trial[
                            "actions/action/submitted_controller_tcp_target_world_xyzw"
                        ][:],
                        label=trial[
                            "labels/residual_effective_software_action_minus_raw_mean"
                        ][:],
                        command_valid=trial["labels/command_valid"][:],
                    )
                )
    return trials


TRIALS = load_trials()
needs_recordings = pytest.mark.skipif(
    not TRIALS, reason=f"no replayable Phase 1 recordings under {RECORDINGS}"
)


def replay(trial: RecordedTrial):
    """Rerun every recorded step through the Phase 1 action pipeline."""
    return [
        build_action_trace(
            raw_mean=trial.mu[step],
            raw_sample=trial.sample[step],
            observed_flange_pose=_pose(trial.flange[trial.observation_index[step]]),
            flange_to_tcp=trial.flange_to_tcp,
            config=trial.config,
            # Sessions with stochastic: false fed the mean, not the sample,
            # into the pipeline.
            deterministic=not trial.stochastic,
            residual=None,
        )
        for step in range(len(trial))
    ]


def test_recordings_are_available():
    """Guard against this whole module silently skipping."""
    if not _sessions():
        pytest.skip(f"no Phase 1 recordings under {RECORDINGS}")
    assert TRIALS, "recordings are present but none could be replayed"


@needs_recordings
def test_replay_reproduces_the_commanded_actions_exactly():
    replayed = 0
    for trial in TRIALS:
        traces = replay(trial)
        assert len(traces) == len(trial.effective_action), trial.where
        for step, trace in enumerate(traces):
            at = f"{trial.where} step {step}"
            assert np.array_equal(
                trace.effective_software_action.astype(np.float32),
                trial.effective_action[step],
            ), f"{at}: effective action diverged"
            assert np.allclose(
                trace.effective_flange_target.as_vector(),
                trial.flange_target[step],
                atol=1e-12,
            ), f"{at}: flange target diverged"
            assert np.allclose(
                trace.submitted_controller_tcp_target.as_vector(),
                trial.tcp_target[step],
                atol=1e-12,
            ), f"{at}: TCP target diverged"
            if trial.command_valid[step]:
                assert np.array_equal(
                    trace.effective_action_minus_raw_mean.astype(np.float32),
                    trial.label[step],
                ), f"{at}: residual label diverged"
            else:
                # Phase 1 writes NaN labels for non-training-eligible steps; the
                # action itself is still checked above.
                assert np.all(np.isnan(trial.label[step])), (
                    f"{at}: an invalid step should carry a NaN label"
                )
            assert trace.requested_residual is None, f"{at}: Phase 1 records no residual"
        replayed += len(traces)
    assert replayed > 0
    print(f"\nreplayed {replayed} recorded Phase 1 steps across {len(TRIALS)} trials")


@needs_recordings
def test_replay_matches_the_golden_digest():
    """One signature over a fixed pre-Phase-2 trial's replayed outputs."""
    matching = [t for t in TRIALS if t.where == GOLDEN_TRIAL]
    if not matching:
        pytest.skip(f"golden trial {GOLDEN_TRIAL} is not present")
    digest = hashlib.md5()
    steps = 0
    for trial in matching:
        for trace in replay(trial):
            for value in (
                trace.effective_software_action,
                trace.scaled_translation_world,
                trace.scaled_rotation_vector_world,
                trace.effective_flange_target.as_vector(),
                trace.submitted_controller_tcp_target.as_vector(),
                trace.effective_action_minus_raw_mean,
            ):
                digest.update(np.ascontiguousarray(value, np.float64).tobytes())
            steps += 1
    assert digest.hexdigest() == GOLDEN_DIGEST, (
        f"Phase 1 action pipeline changed: {steps} steps of {GOLDEN_TRIAL} "
        f"hashed to {digest.hexdigest()}, expected {GOLDEN_DIGEST}"
    )


@needs_recordings
def test_a_zero_residual_takes_the_deterministic_path_not_the_phase1_path():
    """Passing a residual is a behavioural switch, not a no-op.

    ``residual=0`` composes onto the mean, so it matches the deterministic path
    rather than the stochastic one.
    """
    stochastic = [t for t in TRIALS if t.stochastic]
    if not stochastic:
        pytest.skip("no stochastic Phase 1 trial to contrast against")
    trial = stochastic[0]
    differed = 0
    for step in range(min(len(trial), 50)):
        pose = _pose(trial.flange[trial.observation_index[step]])
        common = dict(
            raw_mean=trial.mu[step],
            raw_sample=trial.sample[step],
            observed_flange_pose=pose,
            flange_to_tcp=trial.flange_to_tcp,
            config=trial.config,
        )
        zero = build_action_trace(residual=np.zeros(6), **common)
        deterministic = build_action_trace(deterministic=True, **common)
        stochastic = build_action_trace(**common)

        assert np.array_equal(
            zero.effective_software_action, deterministic.effective_software_action
        ), f"step {step}: residual=0 must equal the deterministic path"
        assert zero.requested_residual is not None
        if not np.array_equal(
            zero.effective_software_action, stochastic.effective_software_action
        ):
            differed += 1
    assert differed > 0, (
        "residual=0 never differed from the stochastic Phase 1 path; the "
        "recording may not contain genuine sampling noise"
    )
