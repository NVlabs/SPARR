# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Replay buffers and the RLPD demo/online split."""

import numpy as np
import pytest

from sparr_phase2_residual.buffers import (
    ACTION_DIM,
    BufferError,
    RLPDSampler,
    load_demo_transitions,
    make_data_store,
    populate,
    residual_spaces,
    validate_transition,
)
from sparr_phase2_residual.observation_spec import ResidualObservationSpec

pytest.importorskip("gymnasium")

IMG = 16  # small frames keep the fixtures fast; the code is size-agnostic
SPEC = ResidualObservationSpec()


def transition(rng, *, done=False, spec=SPEC, image_size=IMG):
    def observation():
        out = {"state": rng.normal(size=(1, spec.state_dim())).astype(np.float32)}
        for key in spec.image_keys:
            out[key] = rng.integers(
                0, 255, (1, image_size, image_size, 3), dtype=np.uint8
            )
        return out

    return {
        "observations": observation(),
        "next_observations": observation(),
        "actions": rng.uniform(-1, 1, ACTION_DIM).astype(np.float32),
        "rewards": float(done),
        "masks": float(not done),
        "dones": done,
        "infos": {},
    }


def episode(rng, length=5, **kwargs):
    return [transition(rng, done=(i == length - 1), **kwargs) for i in range(length)]


def test_spaces_carry_the_frame_stack_axis():
    """serl reads the stack depth off shape[0]; it must be 1, not the batch."""
    observation_space, action_space = residual_spaces(SPEC, image_size=IMG)
    assert observation_space["state"].shape == (1, 19)
    assert observation_space["state"].dtype == np.float32
    assert observation_space["wrist_1"].shape == (1, IMG, IMG, 3)
    assert observation_space["wrist_1"].dtype == np.uint8
    assert action_space.shape == (ACTION_DIM,)
    assert (action_space.low == -1).all() and (action_space.high == 1).all()


@pytest.mark.parametrize(
    "kwargs,dim", [({}, 19), ({"wrench": False}, 13), ({"socket_pose": True}, 26)]
)
def test_spaces_follow_the_configured_layout(kwargs, dim):
    spec = ResidualObservationSpec(**kwargs)
    observation_space, _ = residual_spaces(spec, image_size=IMG)
    assert observation_space["state"].shape == (1, dim)


def test_two_cameras_each_get_a_space():
    spec = ResidualObservationSpec(image_keys=("wrist_1", "wrist_2"))
    observation_space, _ = residual_spaces(spec, image_size=IMG)
    assert {"state", "wrist_1", "wrist_2"} == set(observation_space.spaces)


def test_transitions_round_trip_into_a_store():
    rng = np.random.default_rng(0)
    store = make_data_store(SPEC, capacity=200, image_size=IMG, seed=0)
    inserted = populate(store, episode(rng, 10), SPEC)
    assert inserted == 10
    # The memory-efficient buffer pads each episode start with frame-stack rows
    # that it marks unsamplable, so len() exceeds the count inserted.
    assert len(store) >= inserted

    batch = store.sample(4)
    assert np.asarray(batch["observations"]["state"]).shape == (4, 1, 19)
    assert np.asarray(batch["observations"]["wrist_1"]).shape == (4, 1, IMG, IMG, 3)
    assert np.asarray(batch["actions"]).shape == (4, ACTION_DIM)


def test_a_zero_capacity_store_is_rejected():
    with pytest.raises(BufferError):
        make_data_store(SPEC, capacity=0, image_size=IMG)


def test_a_well_formed_transition_validates():
    validate_transition(transition(np.random.default_rng(0)), SPEC)


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda t: t.pop("masks"), "masks"),
        (lambda t: t["observations"].__setitem__("state", np.zeros((1, 18), np.float32)), "state"),
        (lambda t: t["observations"].__setitem__("state", np.zeros((1, 19), np.float64)), "float32"),
        (lambda t: t["observations"].pop("wrist_1"), "wrist_1"),
        (lambda t: t["observations"].__setitem__("wrist_1", np.zeros((IMG, IMG, 3), np.uint8)), "wrist_1"),
        (lambda t: t["observations"].__setitem__("wrist_1", np.zeros((1, IMG, IMG, 3), np.float32)), "uint8"),
        (lambda t: t.__setitem__("actions", np.zeros(7, np.float32)), "actions"),
        (lambda t: t.__setitem__("actions", np.full(6, np.nan, np.float32)), "finite"),
        (lambda t: t["next_observations"].__setitem__("state", np.zeros((1, 5), np.float32)), "next_observations"),
    ],
)
def test_a_malformed_transition_is_rejected_before_it_reaches_the_buffer(mutate, message):
    """The buffer copies into preallocated arrays, so a bad shape would either
    raise thousands of inserts later or be silently truncated."""
    item = transition(np.random.default_rng(0))
    mutate(item)
    with pytest.raises(BufferError, match=message):
        validate_transition(item, SPEC)


def test_populate_validates_the_first_transition_of_every_episode():
    rng = np.random.default_rng(0)
    store = make_data_store(SPEC, capacity=200, image_size=IMG)
    good = episode(rng, 3)
    bad = episode(rng, 3)
    bad[0]["actions"] = np.zeros(7, np.float32)  # first of the second episode
    with pytest.raises(BufferError, match="actions"):
        populate(store, good + bad, SPEC)


def test_validation_can_be_skipped_for_bulk_loads():
    rng = np.random.default_rng(0)
    store = make_data_store(SPEC, capacity=200, image_size=IMG)
    assert populate(store, episode(rng, 4), SPEC, validate=False) == 4


def test_demo_files_load_and_concatenate(tmp_path):
    import pickle

    rng = np.random.default_rng(0)
    for name, length in (("a.pkl", 3), ("b.pkl", 4)):
        (tmp_path / name).write_bytes(pickle.dumps(episode(rng, length)))
    loaded = load_demo_transitions([tmp_path / "a.pkl", tmp_path / "b.pkl"])
    assert len(loaded) == 7


def test_a_missing_demo_file_is_reported_with_its_path(tmp_path):
    with pytest.raises(BufferError, match="absent.pkl"):
        load_demo_transitions([tmp_path / "absent.pkl"])


def test_an_empty_demo_file_is_rejected(tmp_path):
    import pickle

    (tmp_path / "empty.pkl").write_bytes(pickle.dumps([]))
    with pytest.raises(BufferError, match="non-empty"):
        load_demo_transitions([tmp_path / "empty.pkl"])


def stores(rng, demo_steps=40, online_steps=40):
    demo = make_data_store(SPEC, capacity=500, image_size=IMG, seed=0)
    online = make_data_store(SPEC, capacity=500, image_size=IMG, seed=1)
    populate(demo, episode(rng, demo_steps), SPEC)
    populate(online, episode(rng, online_steps), SPEC)
    return demo, online


def test_each_batch_is_exactly_half_demonstrations():
    """Exact per batch, not merely in expectation, as RLPD specifies."""
    rng = np.random.default_rng(0)
    sampler = RLPDSampler(*stores(rng), batch_size=32)
    assert sampler.demo_batch == 16 and sampler.online_batch == 16
    batch = sampler.sample()
    assert np.asarray(batch["actions"]).shape == (32, ACTION_DIM)
    assert np.asarray(batch["observations"]["state"]).shape == (32, 1, 19)
    assert np.asarray(batch["observations"]["wrist_1"]).shape == (32, 1, IMG, IMG, 3)


def test_the_demo_fraction_is_configurable():
    rng = np.random.default_rng(0)
    sampler = RLPDSampler(
        *stores(rng, demo_steps=40, online_steps=100), batch_size=100, demo_fraction=0.25
    )
    assert sampler.demo_batch == 25 and sampler.online_batch == 75
    assert np.asarray(sampler.sample()["actions"]).shape == (100, ACTION_DIM)


def test_an_odd_batch_size_still_sums_correctly():
    rng = np.random.default_rng(0)
    sampler = RLPDSampler(*stores(rng), batch_size=33)
    assert sampler.demo_batch + sampler.online_batch == 33


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_a_degenerate_demo_fraction_is_rejected(fraction):
    """0 or 1 would silently turn RLPD into plain SAC or pure offline BC."""
    rng = np.random.default_rng(0)
    with pytest.raises(BufferError):
        RLPDSampler(*stores(rng), batch_size=32, demo_fraction=fraction)


def test_a_batch_too_small_to_split_is_rejected():
    rng = np.random.default_rng(0)
    with pytest.raises(BufferError):
        RLPDSampler(*stores(rng), batch_size=1)


def test_sampling_before_either_buffer_is_ready_fails_loudly():
    rng = np.random.default_rng(0)
    demo = make_data_store(SPEC, capacity=500, image_size=IMG, seed=0)
    online = make_data_store(SPEC, capacity=500, image_size=IMG, seed=1)
    populate(demo, episode(rng, 40), SPEC)  # online stays empty
    sampler = RLPDSampler(demo, online, batch_size=32)
    assert sampler.ready is False
    with pytest.raises(BufferError, match="not enough data"):
        sampler.sample()


def test_ready_flips_once_both_sides_have_enough():
    rng = np.random.default_rng(0)
    demo, online = stores(rng, demo_steps=40, online_steps=40)
    assert RLPDSampler(demo, online, batch_size=32).ready is True


def test_the_online_half_comes_first_in_the_batch():
    """Batch position is load-bearing, so the half order is pinned.

    ``batched_random_crop`` splits its key per batch index and the actor's
    action noise is one (B, A) draw, so both are bound to position rather than
    content; reversing the halves changes every loss.
    """
    rng = np.random.default_rng(0)
    demo, online = stores(rng)
    sampler = RLPDSampler(demo, online, batch_size=8)

    # Mark the two buffers apart so the halves are identifiable in the batch.
    demo_marked = make_data_store(SPEC, capacity=500, image_size=IMG, seed=0)
    online_marked = make_data_store(SPEC, capacity=500, image_size=IMG, seed=1)
    for store, value in ((demo_marked, 1.0), (online_marked, -1.0)):
        marked = episode(np.random.default_rng(7), 40)
        for item in marked:
            item["observations"]["state"] = np.full((1, 19), value, np.float32)
            item["next_observations"]["state"] = np.full((1, 19), value, np.float32)
        populate(store, marked, SPEC)

    batch = RLPDSampler(demo_marked, online_marked, batch_size=8).sample()
    state = np.asarray(batch["observations"]["state"])[:, 0, 0]
    assert (state[:4] == -1.0).all(), "the first half must be online data"
    assert (state[4:] == 1.0).all(), "the second half must be demonstrations"
