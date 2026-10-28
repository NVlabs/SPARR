# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA License.
# See the LICENSE file in the repository root for the full license text.

"""Phase 1 HDF5 -> RLPD demonstration buffer conversion."""

import numpy as np
import pytest

from sparr_phase2_residual.demo_export import (
    DemoExportError,
    ResidualObservationSpec,
    _terminal_run_start,
    export_demos,
)

H = W = 8  # tiny frames keep the fixtures fast; the code is shape-agnostic


def write_session(
    path,
    *,
    steps=5,
    accepted=True,
    complete=True,
    export_eligible=True,
    closed_cleanly=True,
    tcp=None,
    command_valid=None,
    mapping=None,
    trials=1,
):
    """Build a minimal but schema-faithful Phase 1 session."""
    import h5py

    n_obs = steps + 1
    with h5py.File(path, "w") as f:
        f.attrs["closed_cleanly"] = closed_cleanly
        meta = f.create_group("metadata")
        meta.create_dataset("unused", data=0)
        group = f.create_group("trials")
        for t in range(trials):
            trial = group.create_group(f"trial_{t:06d}")
            trial.attrs["accepted"] = accepted
            trial.attrs["complete"] = complete
            trial.attrs["export_eligible"] = export_eligible
            trial.attrs["trial_uuid"] = f"uuid-{t}"

            rng = np.random.default_rng(t)
            obs = trial.create_group("observations")
            robot = obs.create_group("robot")
            robot.create_dataset(
                "flange_pose_world_xyzw",
                data=rng.normal(size=(n_obs, 7)).astype(np.float64),
            )
            robot.create_dataset(
                "ext_wrench_tcp", data=rng.normal(size=(n_obs, 6)).astype(np.float64)
            )
            poses = (
                np.asarray(tcp, dtype=np.float64)
                if tcp is not None
                else np.arange(n_obs, dtype=np.float64)[:, None] * np.ones((1, 3)) * 0.01
            )
            robot.create_dataset(
                "tcp_pose_world_xyzw",
                data=np.concatenate([poses, np.zeros((n_obs, 4))], axis=1),
            )
            task = obs.create_group("task")
            task.create_dataset(
                "socket_root_exact_world_xyzw",
                data=rng.normal(size=(n_obs, 7)).astype(np.float64),
            )
            cam = obs.create_group("camera").create_group("wrist_1")
            cam.create_dataset(
                "rgb",
                data=rng.integers(0, 255, size=(n_obs, H, W, 3), dtype=np.uint8),
            )

            act = trial.create_group("actions")
            act.create_group("policy").create_dataset(
                "mu_raw_normalized", data=rng.normal(size=(steps, 6)).astype(np.float32)
            )
            idx = (
                np.asarray(mapping, dtype=np.int64)
                if mapping is not None
                else np.arange(steps, dtype=np.int64)
            )
            act.create_group("mapping").create_dataset("observation_index", data=idx)

            labels = trial.create_group("labels")
            labels.create_dataset(
                "residual_effective_software_action_minus_raw_mean",
                data=rng.normal(size=(steps, 6)).astype(np.float32),
            )
            labels.create_dataset(
                "command_valid",
                data=(np.ones(steps, bool) if command_valid is None
                      else np.asarray(command_valid, bool)),
            )
    return path


def test_default_state_layout_is_nineteen_dimensional_and_ordered():
    spec = ResidualObservationSpec()
    assert spec.state_dim() == 19
    assert spec.describe() == (
        "[0:7] flange_pose, [7:13] wrench, [13:19] base_mean"
    )
    assert spec.socket_pose is False, "socket pose is excluded by design"


@pytest.mark.parametrize(
    "kwargs,dim",
    [
        ({}, 19),
        ({"wrench": False}, 13),
        ({"socket_pose": True}, 26),
        ({"wrench": False, "socket_pose": True}, 20),
        ({"base_mean": False}, 13),
    ],
)
def test_ablation_terms_change_only_the_state_width(tmp_path, kwargs, dim):
    """Term selection is configuration, not a code change."""
    session = write_session(tmp_path / "s.h5")
    spec = ResidualObservationSpec(**kwargs)
    demos, stats = export_demos([session], spec)
    assert stats.trials_exported == 1
    assert spec.state_dim() == dim
    assert demos[0]["observations"]["state"].shape == (1, dim)


def test_demo_reconstructs_the_executed_action_from_the_same_timestep_mean(tmp_path):
    import h5py

    session = write_session(tmp_path / "s.h5", steps=6)
    with h5py.File(session, "r") as f:
        trial = f["trials/trial_000000"]
        mu = trial["actions/policy/mu_raw_normalized"][:]
        residual = trial["labels/residual_effective_software_action_minus_raw_mean"][:]
    executed = mu + residual

    demos, _ = export_demos([session], ResidualObservationSpec())
    for i, transition in enumerate(demos):
        recomposed = transition["observations"]["state"][0, 13:19] + transition["actions"]
        assert np.allclose(recomposed, executed[i], atol=1e-6), f"step {i}"


def test_next_observation_chains_and_only_the_last_step_is_terminal(tmp_path):
    session = write_session(tmp_path / "s.h5", steps=6)
    demos, _ = export_demos([session], ResidualObservationSpec())
    assert sum(t["dones"] for t in demos) == 1
    assert demos[-1]["dones"] is True
    assert demos[-1]["masks"] == 0.0
    assert all(t["masks"] == 1.0 for t in demos[:-1])
    for i in range(len(demos) - 1):
        assert np.array_equal(
            demos[i]["next_observations"]["state"], demos[i + 1]["observations"]["state"]
        )
        assert np.array_equal(
            demos[i]["next_observations"]["wrist_1"], demos[i + 1]["observations"]["wrist_1"]
        )


def test_carried_forward_terminal_mean_is_always_masked_out(tmp_path):
    """The terminal observation has no recorded mean; SAC must never use it."""
    import h5py

    session = write_session(tmp_path / "s.h5", steps=4)
    with h5py.File(session, "r") as f:
        mu = f["trials/trial_000000/actions/policy/mu_raw_normalized"][:]
    demos, _ = export_demos([session], ResidualObservationSpec())
    assert np.array_equal(demos[-1]["next_observations"]["state"][0, 13:19], mu[-1])
    assert demos[-1]["masks"] == 0.0  # r + gamma * mask * Q(s',a') -> next_obs unused


def test_reward_is_one_only_on_the_terminal_step_of_an_accepted_trial(tmp_path):
    accepted, _ = export_demos([write_session(tmp_path / "a.h5")], ResidualObservationSpec())
    assert accepted[-1]["rewards"] == 1.0
    assert all(t["rewards"] == 0.0 for t in accepted[:-1])

    failed, _ = export_demos(
        [write_session(tmp_path / "b.h5", accepted=False)],
        ResidualObservationSpec(), accepted_only=False,
    )
    assert all(t["rewards"] == 0.0 for t in failed)


def test_stationary_tail_is_truncated_and_its_reward_backfilled(tmp_path):
    """The trailing run inside the threshold collapses."""
    # moves for 4 steps, then holds the same pose for the last 3 observations
    poses = np.array([[0.0, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [0.03, 0, 0],
                      [0.04, 0, 0], [0.04, 0, 0], [0.04, 0, 0]])
    session = write_session(tmp_path / "s.h5", steps=6, tcp=poses)
    demos, stats = export_demos(
        [session], ResidualObservationSpec(), stationary_threshold_m=0.001
    )
    assert stats.truncated_steps > 0
    assert len(demos) < 6
    assert demos[-1]["dones"] is True and demos[-1]["rewards"] == 1.0


def test_terminal_run_start_finds_the_first_stationary_index():
    # indices 2, 3, 4 all sit at the terminal pose, so the run starts at 2
    poses = np.array([[0.0, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [0.02, 0, 0], [0.02, 0, 0]])
    assert _terminal_run_start(poses, 0.001) == 2
    moving = np.arange(5, dtype=float)[:, None] * np.ones((1, 3)) * 0.01
    assert _terminal_run_start(moving, 0.001) == 4


def test_entirely_stationary_episode_is_rejected_not_silently_truncated(tmp_path):
    """Keeping 2 transitions here would quietly discard the episode."""
    session = write_session(
        tmp_path / "s.h5", steps=5, tcp=np.zeros((6, 3))
    )
    demos, stats = export_demos([session], ResidualObservationSpec())
    assert demos == []
    assert any("stationary" in note for note in stats.skipped)


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"accepted": False}, "not accepted"),
        ({"complete": False}, "incomplete"),
        ({"export_eligible": False}, "not export-eligible"),
        ({"closed_cleanly": False}, "not closed cleanly"),
    ],
)
def test_ineligible_sessions_are_skipped_with_a_reason(tmp_path, kwargs, reason):
    session = write_session(tmp_path / "s.h5", **kwargs)
    demos, stats = export_demos([session], ResidualObservationSpec())
    assert demos == []
    assert any(reason in note for note in stats.skipped), stats.skipped


def test_deterministic_debug_data_cannot_reach_training(tmp_path):
    """Deterministic trials are marked export_eligible=False by the collector."""
    session = write_session(tmp_path / "s.h5", export_eligible=False)
    demos, _ = export_demos(
        [session], ResidualObservationSpec(), accepted_only=False
    )
    assert demos == [], "debug-only data must never be exported"


def test_non_identity_action_mapping_is_rejected(tmp_path):
    """Misaligned means would silently pair the wrong base action with a state."""
    session = write_session(tmp_path / "s.h5", steps=4, mapping=[0, 1, 3, 2])
    demos, stats = export_demos([session], ResidualObservationSpec())
    assert demos == []
    assert any("mapping is not identity" in n for n in stats.skipped)


def test_invalid_command_rows_are_rejected(tmp_path):
    session = write_session(
        tmp_path / "s.h5", steps=4, command_valid=[True, True, False, True]
    )
    demos, stats = export_demos([session], ResidualObservationSpec())
    assert demos == []
    assert any("not training-eligible" in n for n in stats.skipped)


def test_transition_matches_the_reference_dict_schema(tmp_path):
    session = write_session(tmp_path / "s.h5", steps=3)
    demos, _ = export_demos([session], ResidualObservationSpec())
    t = demos[0]
    assert set(t) == {
        "observations", "actions", "next_observations",
        "rewards", "masks", "dones", "infos",
    }
    assert set(t["observations"]) == {"state", "wrist_1"}
    assert t["observations"]["state"].dtype == np.float32
    assert t["observations"]["state"].shape == (1, 19)
    assert t["observations"]["wrist_1"].dtype == np.uint8
    assert t["observations"]["wrist_1"].shape == (1, H, W, 3)
    assert t["actions"].shape == (6,) and t["actions"].dtype == np.float32
    assert isinstance(t["rewards"], float) and isinstance(t["masks"], float)
    assert isinstance(t["dones"], bool)


def test_multiple_trials_and_sessions_accumulate(tmp_path):
    a = write_session(tmp_path / "a.h5", steps=4, trials=2)
    b = write_session(tmp_path / "b.h5", steps=3, trials=1)
    demos, stats = export_demos([a, b], ResidualObservationSpec())
    assert stats.trials_exported == 3
    assert stats.transitions == len(demos) == 4 + 4 + 3
    assert sum(t["dones"] for t in demos) == 3  # one terminal per trial


def test_directory_input_ignores_partial_files(tmp_path):
    write_session(tmp_path / "good.h5", steps=3)
    (tmp_path / "open.partial.h5").write_bytes(b"not hdf5")
    demos, stats = export_demos([tmp_path], ResidualObservationSpec())
    assert stats.sessions == 1 and len(demos) == 3
