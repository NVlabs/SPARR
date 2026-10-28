# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA License.
# See the LICENSE file in the repository root for the full license text.

"""The residual learner: config, promotion wiring, loop mechanics, resume.

Uses a fake agent; the SAC update rule belongs to ``serl_launcher`` and is
covered by the integration test.
"""

import json
from pathlib import Path

import flax.struct
import numpy as np
import pytest

from sparr_phase2_residual.buffers import populate
from sparr_phase2_residual.learner import (
    ALL_NETWORKS,
    CRITIC_ONLY,
    LearnerConfig,
    LearnerError,
    LearnerState,
    ResidualLearner,
    scalar_metrics,
)
from sparr_phase2_residual.observation_spec import ResidualObservationSpec

pytest.importorskip("gymnasium")

IMG = 8
SPEC = ResidualObservationSpec()


def transition(rng, done=False, succeeded=True):
    def observation():
        return {
            "state": rng.normal(size=(1, 19)).astype(np.float32),
            "wrist_1": rng.integers(0, 255, (1, IMG, IMG, 3), dtype=np.uint8),
        }

    return {
        "observations": observation(),
        "next_observations": observation(),
        "actions": rng.uniform(-1, 1, 6).astype(np.float32),
        "rewards": float(succeeded and done),
        "masks": float(not done),
        "dones": done,
        "infos": {},
    }


def episode(rng, length, succeeded=True):
    return [
        transition(rng, done=(i == length - 1), succeeded=succeeded)
        for i in range(length)
    ]


class FakeState(flax.struct.PyTreeNode):
    """A real pytree, so device_get and checkpointing behave as in production."""

    params: dict


class FakeAgent:
    """Records which networks were updated, and carries a checkpointable state."""

    def __init__(self):
        self.updates = []
        self.state = FakeState(params={"w": np.zeros(3, np.float32)})

    def update(self, batch, networks_to_update):
        self.updates.append(networks_to_update)
        return self, {"critic": {"critic_loss": 0.5}, "actor_lr": 3e-4}

    def replace(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self


def make_learner(tmp_path=None, **config_kwargs):
    config = LearnerConfig(image_size=IMG, **config_kwargs)
    return ResidualLearner(
        SPEC, config, agent=FakeAgent(), checkpoint_dir=tmp_path
    )


def demo_file(tmp_path, rng, lengths=(4, 6, 8), name="demos.pkl"):
    import pickle

    transitions = [t for length in lengths for t in episode(rng, length)]
    path = tmp_path / name
    path.write_bytes(pickle.dumps(transitions))
    return path


def test_defaults_match_the_reference_implementation():
    config = LearnerConfig()
    assert config.batch_size == 256
    assert config.cta_ratio == 2
    assert config.discount == 0.97
    assert config.training_starts == 100
    assert config.steps_per_update == 50
    assert config.demo_fraction == 0.5
    assert config.target_entropy is None  # serl resolves to -action_dim/2 = -3.0


@pytest.mark.parametrize(
    "kwargs", [{"cta_ratio": 0}, {"batch_size": 1}, {"max_steps": 0}, {"training_starts": 0}]
)
def test_a_nonsense_config_is_rejected(kwargs):
    with pytest.raises(LearnerError):
        LearnerConfig(**kwargs)


def test_networks_to_update_are_frozensets():
    """serl marks this a jit static arg: unhashable fails, reordered retraces."""
    assert isinstance(CRITIC_ONLY, frozenset)
    assert isinstance(ALL_NETWORKS, frozenset)
    assert CRITIC_ONLY == {"critic"}
    assert ALL_NETWORKS == {"critic", "actor", "temperature"}


def test_loading_demonstrations_seeds_the_promotion_median(tmp_path):
    rng = np.random.default_rng(0)
    learner = make_learner()
    inserted = learner.load_demonstrations([demo_file(tmp_path, rng, (4, 6, 8))])
    assert inserted == 18
    assert learner.promoter.lengths == [4, 6, 8]
    assert learner.promoter.median == 6


def test_demonstrations_without_a_terminal_transition_are_rejected(tmp_path):
    """Otherwise the promotion median would be seeded from nothing."""
    import pickle

    rng = np.random.default_rng(0)
    path = tmp_path / "bad.pkl"
    path.write_bytes(pickle.dumps([transition(rng, done=False) for _ in range(5)]))
    with pytest.raises(LearnerError, match="no complete episode"):
        make_learner().load_demonstrations([path])


def ready_learner(tmp_path, rng, **kwargs):
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    learner = make_learner(tmp_path, training_starts=1, batch_size=4, **kwargs)
    learner.load_demonstrations([demo_file(tmp_path, rng, (10, 10))])
    populate(learner.online_store, episode(rng, 20), SPEC)
    return learner


def test_each_iteration_runs_cta_ratio_gradient_steps(tmp_path):
    rng = np.random.default_rng(0)
    learner = ready_learner(tmp_path, rng, cta_ratio=3)
    learner.run(steps=4)
    updates = learner.agent.updates
    assert len(updates) == 4 * 3
    # Two critic-only updates then one full update, repeating.
    assert updates[:3] == [CRITIC_ONLY, CRITIC_ONLY, ALL_NETWORKS]
    assert updates.count(ALL_NETWORKS) == 4


def test_cta_ratio_one_updates_every_network_every_step(tmp_path):
    rng = np.random.default_rng(0)
    learner = ready_learner(tmp_path, rng, cta_ratio=1)
    learner.run(steps=3)
    assert learner.agent.updates == [ALL_NETWORKS] * 3


def test_the_step_counter_advances_and_is_resumable(tmp_path):
    rng = np.random.default_rng(0)
    learner = ready_learner(tmp_path, rng)
    assert learner.run(steps=5) == 5
    assert learner.run(steps=5) == 10


def test_parameters_are_published_initially_and_on_cadence(tmp_path):
    """The actor must never be left running a random initialisation."""
    rng = np.random.default_rng(0)
    learner = ready_learner(tmp_path, rng, steps_per_update=3)
    published = []
    learner.run(steps=9, on_publish=published.append)
    # One publication before the first update, then at steps 3, 6, 9.
    assert len(published) == 4


def test_no_publish_callback_is_tolerated(tmp_path):
    rng = np.random.default_rng(0)
    ready_learner(tmp_path, rng).run(steps=3)


def test_logging_happens_on_the_configured_period(tmp_path):
    rng = np.random.default_rng(0)
    learner = ready_learner(tmp_path, rng, log_period=2)
    seen = []
    learner.run(steps=6, on_log=lambda step, metrics: seen.append(step))
    assert seen == [2, 4, 6]
    assert "critic/critic_loss" in learner.metrics


def test_running_before_the_buffers_are_ready_fails_loudly():
    learner = make_learner(training_starts=1, batch_size=4)
    with pytest.raises(LearnerError):
        learner.run(steps=1)


def test_ready_requires_both_online_volume_and_a_full_batch(tmp_path):
    rng = np.random.default_rng(0)
    learner = make_learner(tmp_path, training_starts=100, batch_size=4)
    learner.load_demonstrations([demo_file(tmp_path, rng, (10, 10))])
    populate(learner.online_store, episode(rng, 20), SPEC)
    assert learner.sampler.ready, "the batch halves are satisfiable"
    assert not learner.ready, "but training_starts is not met"


def test_wait_until_ready_times_out_rather_than_hanging(tmp_path):
    rng = np.random.default_rng(0)
    learner = make_learner(tmp_path, training_starts=10_000, batch_size=4)
    learner.load_demonstrations([demo_file(tmp_path, rng, (10, 10))])
    with pytest.raises(LearnerError, match="timed out"):
        learner.wait_until_ready(poll_s=0.01, timeout_s=0.05)


def test_learner_state_round_trips_through_json():
    state = LearnerState(step=17, promoted=3, promotion_lengths=[4, 5, 6])
    restored = LearnerState.from_json(state.to_json())
    assert restored.step == 17
    assert restored.promoted == 3
    assert restored.promotion_lengths == [4, 5, 6]


def test_the_step_is_recovered_from_a_sidecar_not_a_filename(tmp_path):
    """Slicing the step out of a filename would break on rename."""
    rng = np.random.default_rng(0)
    learner = ready_learner(tmp_path, rng)
    learner.run(steps=7)
    learner.save_checkpoint()

    sidecar = tmp_path / "learner_state.json"
    assert sidecar.is_file()
    assert json.loads(sidecar.read_text())["step"] == 7


def test_checkpointing_without_a_directory_is_an_error():
    with pytest.raises(LearnerError, match="no checkpoint directory"):
        make_learner().save_checkpoint()


def test_restoring_from_a_missing_directory_is_an_error(tmp_path):
    learner = make_learner(tmp_path / "absent")
    with pytest.raises(LearnerError, match="no checkpoint directory"):
        learner.restore_checkpoint()


def test_promotion_state_survives_a_resume(tmp_path):
    rng = np.random.default_rng(0)
    learner = ready_learner(tmp_path, rng)
    from sparr_phase2_residual.offline_replay import Episode

    learner.add_replayed_episode(Episode(episode(rng, 3), succeeded=True), promoted=True)
    assert learner.state.promoted == 1
    before = list(learner.promoter.lengths)
    learner.run(steps=2)
    learner.save_checkpoint()

    state = LearnerState.from_json((tmp_path / "learner_state.json").read_text())
    assert state.promoted == 1
    assert state.promotion_lengths == before


def test_scalar_metrics_flattens_the_nested_tree_and_skips_arrays():
    flat = scalar_metrics(
        {
            "critic": {"critic_loss": 0.5, "q": np.zeros(4)},
            "actor_lr": 3e-4,
            "grid": np.ones((2, 2)),
        }
    )
    assert flat == {"critic/critic_loss": 0.5, "actor_lr": 3e-4}


def test_chunked_runs_consume_the_same_batches_as_one_long_run(tmp_path):
    """run(10) twice must equal run(20).

    get_iterator prefetches queue_size batches, so rebuilding it per run() would
    discard them and advance the buffers' RNG.
    """
    rng = np.random.default_rng(0)
    chunked = ready_learner(tmp_path / "a", rng)
    chunked.run(steps=3)
    first = chunked._batches
    chunked.run(steps=3)
    assert chunked._batches is first, "the iterator must be reused across run() calls"
    assert chunked.state.step == 6
