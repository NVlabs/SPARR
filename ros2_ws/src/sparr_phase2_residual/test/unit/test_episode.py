# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA License.
# See the LICENSE file in the repository root for the full license text.

"""Episode assembly: backfill, truncation, and online/offline equivalence.

The live actor and the offline converter share this implementation, so the same
trajectory fed through both paths must produce identical transitions.
"""

import numpy as np
import pytest

from sparr_phase2_residual.episode import (
    DEFAULT_STATIONARY_THRESHOLD_M,
    EpisodeAccumulator,
    EpisodeError,
    build_transitions,
    terminal_run_start,
)
from sparr_phase2_residual.observation_spec import ResidualObservationSpec

H = W = 4
SPEC = ResidualObservationSpec()


def trajectory(n_actions, rng, *, tcp=None):
    states = [rng.normal(size=(1, 19)).astype(np.float32) for _ in range(n_actions + 1)]
    images = [
        {"wrist_1": rng.integers(0, 255, (1, H, W, 3), dtype=np.uint8)}
        for _ in range(n_actions + 1)
    ]
    actions = [rng.uniform(-1, 1, 6).astype(np.float32) for _ in range(n_actions)]
    poses = (
        np.asarray(tcp, np.float64)
        if tcp is not None
        else np.arange(n_actions + 1, dtype=np.float64)[:, None] * np.ones((1, 3)) * 0.01
    )
    return states, images, actions, poses


def build(n_actions=5, *, accepted=True, rng=None, tcp=None, **kwargs):
    rng = rng or np.random.default_rng(0)
    states, images, actions, poses = trajectory(n_actions, rng, tcp=tcp)
    return build_transitions(
        states=states, images=images, actions=actions, tcp_positions=poses,
        accepted=accepted, spec=SPEC, **kwargs
    )


def test_terminal_run_start_finds_the_first_stationary_index():
    poses = np.array([[0.0, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [0.02, 0, 0], [0.02, 0, 0]])
    assert terminal_run_start(poses, 0.001) == 2
    moving = np.arange(5, dtype=float)[:, None] * np.ones((1, 3)) * 0.01
    assert terminal_run_start(moving, 0.001) == 4


def test_an_empty_pose_sequence_is_rejected():
    with pytest.raises(EpisodeError, match="empty"):
        terminal_run_start(np.zeros((0, 3)), 0.001)


def test_a_stationary_tail_is_truncated(tmp_path):
    # moves for 4 steps, then holds for the last 3 observations
    poses = np.array([[0.0, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [0.03, 0, 0],
                      [0.04, 0, 0], [0.04, 0, 0], [0.04, 0, 0]])
    transitions = build(6, tcp=poses)
    assert len(transitions) < 6
    assert transitions[-1]["dones"] is True
    assert transitions[-1]["rewards"] == 1.0


def test_an_entirely_stationary_episode_is_rejected():
    """Refused rather than silently kept as two transitions."""
    with pytest.raises(EpisodeError, match="stationary"):
        build(5, tcp=np.zeros((6, 3)))


def test_only_the_last_kept_step_is_terminal_and_rewarded():
    transitions = build(6)
    assert sum(t["dones"] for t in transitions) == 1
    assert transitions[-1]["dones"] and transitions[-1]["masks"] == 0.0
    assert all(t["masks"] == 1.0 for t in transitions[:-1])
    assert all(t["rewards"] == 0.0 for t in transitions[:-1])


def test_a_failed_episode_carries_no_reward():
    assert all(t["rewards"] == 0.0 for t in build(5, accepted=False))


def test_next_observations_chain():
    transitions = build(6)
    for i in range(len(transitions) - 1):
        assert np.array_equal(
            transitions[i]["next_observations"]["state"],
            transitions[i + 1]["observations"]["state"],
        )
        assert np.array_equal(
            transitions[i]["next_observations"]["wrist_1"],
            transitions[i + 1]["observations"]["wrist_1"],
        )


def test_mismatched_observation_and_action_counts_are_rejected():
    rng = np.random.default_rng(0)
    states, images, actions, poses = trajectory(5, rng)
    with pytest.raises(EpisodeError, match="expected 6 observations"):
        build_transitions(
            states=states[:-1], images=images, actions=actions,
            tcp_positions=poses, accepted=True, spec=SPEC,
        )


def test_an_episode_with_no_actions_is_rejected():
    rng = np.random.default_rng(0)
    states, images, _, poses = trajectory(0, rng)
    with pytest.raises(EpisodeError, match="no actions"):
        build_transitions(
            states=states, images=images, actions=[], tcp_positions=poses,
            accepted=True, spec=SPEC,
        )


def record(rng, tcp):
    return {
        "robot": {
            "flange_pose_world_xyzw": rng.normal(size=7),
            "ext_wrench_tcp": rng.normal(size=6),
            "tcp_pose_world_xyzw": np.concatenate([tcp, np.zeros(4)]),
        },
        "task": {"socket_root_exact_world_xyzw": rng.normal(size=7)},
        "camera": {"wrist_1": {"rgb": rng.integers(0, 255, (H, W, 3), dtype=np.uint8)}},
    }


def test_the_accumulator_builds_the_same_shapes_as_the_converter():
    rng = np.random.default_rng(0)
    accumulator = EpisodeAccumulator(SPEC)
    for step in range(6):
        accumulator.observe(record(rng, np.array([step * 0.01, 0.0, 0.0])), np.zeros(6))
        accumulator.act(rng.uniform(-1, 1, 6))
    accumulator.observe(record(rng, np.array([0.05, 0.0, 0.0])), np.zeros(6))

    transitions = accumulator.finish(accepted=True)
    assert transitions
    assert transitions[0]["observations"]["state"].shape == (1, 19)
    assert transitions[0]["observations"]["wrist_1"].shape == (1, H, W, 3)
    assert transitions[0]["actions"].shape == (6,)
    assert transitions[-1]["dones"] is True


def test_the_accumulator_puts_the_base_mean_in_the_right_slot():
    rng = np.random.default_rng(0)
    accumulator = EpisodeAccumulator(SPEC)
    mean = np.arange(6, dtype=np.float64) * 0.1
    accumulator.observe(record(rng, np.zeros(3)), mean)
    assert np.allclose(accumulator.states[0][0, 13:19], mean)


def test_a_trailing_action_without_an_observation_is_dropped():
    """A trial cut short leaves N == M; fabricating an observation would be worse."""
    rng = np.random.default_rng(0)
    accumulator = EpisodeAccumulator(SPEC)
    for step in range(5):
        accumulator.observe(record(rng, np.array([step * 0.01, 0.0, 0.0])), np.zeros(6))
        accumulator.act(rng.uniform(-1, 1, 6))
    assert len(accumulator.states) == len(accumulator.actions)
    transitions = accumulator.finish(accepted=True)
    assert len(transitions) <= 4


def test_reset_clears_everything():
    rng = np.random.default_rng(0)
    accumulator = EpisodeAccumulator(SPEC)
    accumulator.observe(record(rng, np.zeros(3)), np.zeros(6))
    accumulator.act(np.zeros(6))
    accumulator.reset()
    assert accumulator.steps == 0
    assert not accumulator.states and not accumulator.images


def test_online_and_offline_paths_produce_identical_transitions(tmp_path):
    """One trajectory, both code paths, byte-identical output.

    Divergence would train the policy on one distribution and run it on another.
    """
    from sparr_phase2_residual.demo_export import export_demos

    rng = np.random.default_rng(7)
    steps = 8
    flange = rng.normal(size=(steps + 1, 7))
    wrench = rng.normal(size=(steps + 1, 6))
    socket = rng.normal(size=(steps + 1, 7))
    frames = rng.integers(0, 255, (steps + 1, H, W, 3), dtype=np.uint8)
    tcp = np.arange(steps + 1, dtype=np.float64)[:, None] * np.ones((1, 3)) * 0.01
    means = rng.normal(size=(steps, 6)).astype(np.float32)
    labels = rng.normal(size=(steps, 6)).astype(np.float32)

    # --- online path
    accumulator = EpisodeAccumulator(SPEC)
    for index in range(steps):
        accumulator.observe(
            {
                "robot": {
                    "flange_pose_world_xyzw": flange[index],
                    "ext_wrench_tcp": wrench[index],
                    "tcp_pose_world_xyzw": np.concatenate([tcp[index], np.zeros(4)]),
                },
                "task": {"socket_root_exact_world_xyzw": socket[index]},
                "camera": {"wrist_1": {"rgb": frames[index]}},
            },
            means[index],
        )
        accumulator.act(labels[index])
    accumulator.observe(
        {
            "robot": {
                "flange_pose_world_xyzw": flange[steps],
                "ext_wrench_tcp": wrench[steps],
                "tcp_pose_world_xyzw": np.concatenate([tcp[steps], np.zeros(4)]),
            },
            "task": {"socket_root_exact_world_xyzw": socket[steps]},
            "camera": {"wrist_1": {"rgb": frames[steps]}},
        },
        means[-1],  # carried forward; only ever read as a masked next_observation
    )
    online = accumulator.finish(accepted=True)

    # --- offline path, from an HDF5 file holding the same trajectory
    import h5py

    path = tmp_path / "s.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["closed_cleanly"] = True
        handle.create_group("metadata")
        trial = handle.create_group("trials").create_group("trial_000000")
        for flag in ("accepted", "complete", "export_eligible"):
            trial.attrs[flag] = True
        obs = trial.create_group("observations")
        robot = obs.create_group("robot")
        robot.create_dataset("flange_pose_world_xyzw", data=flange)
        robot.create_dataset("ext_wrench_tcp", data=wrench)
        robot.create_dataset(
            "tcp_pose_world_xyzw",
            data=np.concatenate([tcp, np.zeros((steps + 1, 4))], axis=1),
        )
        obs.create_group("task").create_dataset(
            "socket_root_exact_world_xyzw", data=socket
        )
        obs.create_group("camera").create_group("wrist_1").create_dataset(
            "rgb", data=frames
        )
        acts = trial.create_group("actions")
        acts.create_group("policy").create_dataset("mu_raw_normalized", data=means)
        acts.create_group("mapping").create_dataset(
            "observation_index", data=np.arange(steps, dtype=np.int64)
        )
        lab = trial.create_group("labels")
        lab.create_dataset(
            "residual_effective_software_action_minus_raw_mean", data=labels
        )
        lab.create_dataset("command_valid", data=np.ones(steps, bool))

    offline, _ = export_demos([path], SPEC)

    assert len(online) == len(offline), "the two paths kept different step counts"
    for index, (live, saved) in enumerate(zip(online, offline)):
        for field in ("rewards", "masks", "dones"):
            assert live[field] == saved[field], f"step {index}: {field}"
        assert np.array_equal(live["actions"], saved["actions"]), f"step {index}: action"
        for side in ("observations", "next_observations"):
            assert np.array_equal(
                live[side]["state"], saved[side]["state"]
            ), f"step {index}: {side}.state"
            assert np.array_equal(
                live[side]["wrist_1"], saved[side]["wrist_1"]
            ), f"step {index}: {side}.wrist_1"
