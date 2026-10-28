# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA OneWay Noncommercial License (March 22, 2022).
# See the LICENSE file in the repository root for the full license text:
# https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf

"""The learner against the real ``serl_launcher`` SAC agent: training and resume.

Catches checkpoints that restore nothing and batch shapes the real network
rejects, which the fake agent in the unit tests cannot. Skipped where JAX is
unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("gymnasium")

from sparr_phase2_residual.buffers import populate  # noqa: E402
from sparr_phase2_residual.learner import LearnerConfig, ResidualLearner  # noqa: E402
from sparr_phase2_residual.observation_spec import ResidualObservationSpec  # noqa: E402

IMG = 32  # smaller than production; the ResNet is fully convolutional
SPEC = ResidualObservationSpec()


def episode(rng, length, succeeded=True):
    def observation():
        return {
            "state": rng.normal(size=(1, SPEC.state_dim())).astype(np.float32),
            "wrist_1": rng.integers(0, 255, (1, IMG, IMG, 3), dtype=np.uint8),
        }

    return [
        {
            "observations": observation(),
            "next_observations": observation(),
            "actions": rng.uniform(-1, 1, 6).astype(np.float32),
            "rewards": float(succeeded and i == length - 1),
            "masks": float(i != length - 1),
            "dones": i == length - 1,
            "infos": {},
        }
        for i in range(length)
    ]


def demo_file(tmp_path, rng, lengths=(8, 10, 12)):
    import pickle

    path = tmp_path / "demos.pkl"
    path.write_bytes(pickle.dumps([t for n in lengths for t in episode(rng, n)]))
    return path


def make_learner(tmp_path, **kwargs):
    defaults = dict(image_size=IMG, batch_size=8, training_starts=1, seed=0)
    defaults.update(kwargs)
    return ResidualLearner(SPEC, LearnerConfig(**defaults), checkpoint_dir=tmp_path)


def ready(tmp_path, rng, **kwargs):
    learner = make_learner(tmp_path, **kwargs)
    learner.load_demonstrations([demo_file(tmp_path, rng)])
    populate(learner.online_store, episode(rng, 30), SPEC)
    return learner


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """One trained learner, reused: building the agent is the slow part."""
    tmp_path = tmp_path_factory.mktemp("learner")
    rng = np.random.default_rng(0)
    learner = ready(tmp_path, rng, log_period=2)
    learner.run(steps=6)
    learner.save_checkpoint()
    return learner, tmp_path


def test_real_updates_produce_finite_losses(trained):
    learner, _ = trained
    assert learner.state.step == 6
    metrics = learner.metrics
    assert metrics, "no metrics were captured"
    assert all(np.isfinite(v) for v in metrics.values()), metrics
    assert "critic/critic_loss" in metrics


def test_resume_restores_the_trained_weights_not_a_fresh_initialisation(trained, tmp_path):
    """Guards against a checkpoint that silently restores nothing."""
    learner, checkpoint_dir = trained
    saved = jax.device_get(learner.agent.state.params)

    rng = np.random.default_rng(0)
    fresh = ready(tmp_path, rng)
    fresh.checkpoint_dir = checkpoint_dir
    before = jax.device_get(fresh.agent.state.params)

    # A fresh agent with a different seed must not already match, or this test
    # would pass without restoring anything.
    leaves_before = jax.tree.leaves(before)
    leaves_saved = jax.tree.leaves(saved)
    assert any(
        not np.allclose(a, b) for a, b in zip(leaves_before, leaves_saved)
    ), "fresh and trained params are identical; the test cannot detect a restore"

    step = fresh.restore_checkpoint()
    assert step == 6
    restored = jax.tree.leaves(jax.device_get(fresh.agent.state.params))
    for a, b in zip(restored, leaves_saved):
        assert np.allclose(a, b), "restored params differ from the saved ones"


def test_a_resumed_learner_continues_from_the_saved_step(trained, tmp_path):
    learner, checkpoint_dir = trained
    rng = np.random.default_rng(0)
    fresh = ready(tmp_path, rng)
    fresh.checkpoint_dir = checkpoint_dir
    fresh.restore_checkpoint()
    assert fresh.run(steps=2) == 8


def test_the_batch_the_sampler_produces_is_accepted_by_the_real_network(tmp_path):
    """Catches a shape or dtype the fake agent would happily ignore."""
    rng = np.random.default_rng(2)
    learner = ready(tmp_path, rng)
    batch = learner.sampler.sample()
    assert np.asarray(batch["observations"]["state"]).shape == (8, 1, SPEC.state_dim())
    assert np.asarray(batch["actions"]).shape == (8, 6)
    agent, info = learner.agent.update(
        jax.device_put(batch), networks_to_update=frozenset({"critic"})
    )
    jax.block_until_ready(info)


def test_replayed_episodes_reach_the_buffers_only_when_scheduled(tmp_path):
    """In schedule mode, at step N the buffer holds step N's data."""
    from sparr_phase2_residual.offline_replay import (
        ArrivalSchedule,
        ReplayDriver,
        split_episodes,
    )

    rng = np.random.default_rng(0)
    learner = make_learner(tmp_path)
    learner.load_demonstrations([demo_file(tmp_path, rng)])

    flat = [t for n in (6, 6, 6, 6) for t in episode(rng, n)]
    episodes = split_episodes(flat)
    driver = ReplayDriver(
        episodes=episodes,
        schedule=ArrivalSchedule(steps=[0, 10, 20, 30]),
        insert=lambda ep, promoted: learner.add_replayed_episode(ep, promoted=promoted),
    )

    # serl's buffer keeps one extra frame per episode (the final
    # next_observation it deduplicates against), so a 6-step episode reports 7.
    driver.release_due(0)
    assert len(learner.online_store) == 7
    driver.release_due(15)
    assert len(learner.online_store) == 14
    driver.release_due(100)
    assert len(learner.online_store) == 28
    assert driver.exhausted


def test_a_replayed_promotion_grows_the_demo_buffer(tmp_path):
    from sparr_phase2_residual.offline_replay import ArrivalSchedule, ReplayDriver, split_episodes

    rng = np.random.default_rng(1)
    learner = make_learner(tmp_path)
    learner.load_demonstrations([demo_file(tmp_path, rng)])
    before = len(learner.demo_store)

    episodes = split_episodes([t for t in episode(rng, 6, succeeded=True)])
    ReplayDriver(
        episodes=episodes,
        schedule=ArrivalSchedule(steps=[0], promoted=[True]),
        insert=lambda ep, promoted: learner.add_replayed_episode(ep, promoted=promoted),
    ).release_due(0)

    assert len(learner.demo_store) == before + 7
    assert learner.state.promoted == 1


def test_training_runs_with_episodes_arriving_mid_run(tmp_path):
    """run(release=...) must train through arrivals without shape or RNG trouble."""
    from sparr_phase2_residual.offline_replay import ArrivalSchedule, ReplayDriver, split_episodes

    rng = np.random.default_rng(2)
    learner = make_learner(tmp_path)
    learner.load_demonstrations([demo_file(tmp_path, rng)])

    episodes = split_episodes([t for n in (6, 6, 6) for t in episode(rng, n)])
    driver = ReplayDriver(
        episodes=episodes,
        schedule=ArrivalSchedule(steps=[0, 3, 6]),
        insert=lambda ep, promoted: learner.add_replayed_episode(ep, promoted=promoted),
    )
    driver.release_due(0)

    reached = learner.run(steps=10, release=driver.release_due)
    assert reached == 10
    assert driver.exhausted
    assert len(learner.online_store) == 21


