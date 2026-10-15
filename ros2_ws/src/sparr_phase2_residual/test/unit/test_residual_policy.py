"""The residual actor wrapper: shape, safety, and train/inference consistency."""

import numpy as np
import pytest

from sparr_phase2_residual.observation_spec import ResidualObservationSpec
from sparr_phase2_residual.residual_policy import (
    ResidualOutputStats,
    ResidualPolicy,
    ResidualPolicyError,
)

H = W = 8


def terms(spec=None, **overrides):
    """Well-formed observation terms for the default 19-D layout."""
    values = {
        "flange_pose": np.arange(7, dtype=np.float64) * 0.1,
        "wrench": np.arange(6, dtype=np.float64) * -0.2,
        "base_mean": np.linspace(-0.5, 0.5, 6),
        "socket_pose": np.arange(7, dtype=np.float64) * 0.01,
    }
    values.update(overrides)
    return values


def images(**overrides):
    frames = {"wrist_1": np.full((H, W, 3), 7, np.uint8)}
    frames.update(overrides)
    return frames


class FakeAgent:
    """Stands in for a serl SACAgent, recording what it was asked."""

    def __init__(self, action=None, raises=None):
        self._action = np.zeros(6) if action is None else np.asarray(action)
        self._raises = raises
        self.calls = []

    def sample_actions(self, *, observations, seed, argmax=False):
        self.calls.append({"observations": observations, "seed": seed, "argmax": argmax})
        if self._raises is not None:
            raise self._raises
        return self._action

    # serl agents are flax structs; mimic just enough for update_params
    def replace(self, **kwargs):
        clone = FakeAgent(self._action, self._raises)
        clone.calls = self.calls
        clone.__dict__.update(kwargs)
        return clone


def make_policy(agent=None, **kwargs):
    return ResidualPolicy(
        agent if agent is not None else FakeAgent(),
        ResidualObservationSpec(),
        **kwargs,
    )


def test_evaluate_returns_a_six_dimensional_residual():
    action = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
    decision = make_policy(FakeAgent(action)).evaluate(terms=terms(), images=images())
    assert decision.ok
    assert decision.residual.shape == (6,)
    assert np.allclose(decision.residual, action)
    assert decision.latency_ns > 0
    assert decision.error == ""


def test_observation_uses_the_declared_layout_and_batched_image():
    agent = FakeAgent()
    make_policy(agent).evaluate(terms=terms(), images=images())
    observation = agent.calls[0]["observations"]
    assert set(observation) == {"state", "wrist_1"}
    state = np.asarray(observation["state"])
    assert state.shape == (1, 19)
    assert np.allclose(state[0, 0:7], terms()["flange_pose"])
    assert np.allclose(state[0, 7:13], terms()["wrench"])
    assert np.allclose(state[0, 13:19], terms()["base_mean"])
    # serl's EncodingWrapper runs with enable_stacking=True: dim 0 is the stack
    assert np.asarray(observation["wrist_1"]).shape == (1, H, W, 3)


@pytest.mark.parametrize(
    "bad_action,reason",
    [
        (np.zeros(7), "wrong width"),
        (np.zeros((2, 6)), "wrong rank"),
        (np.array([np.nan, 0, 0, 0, 0, 0]), "non-finite"),
        (np.array([np.inf, 0, 0, 0, 0, 0]), "non-finite"),
        (np.array([1.5, 0, 0, 0, 0, 0]), "outside tanh range"),
    ],
)
def test_a_malformed_action_degrades_instead_of_raising(bad_action, reason):
    """A bad tick must not abort a trial mid-motion; the base policy is safe."""
    decision = make_policy(FakeAgent(bad_action)).evaluate(terms=terms(), images=images())
    assert not decision.ok, reason
    assert decision.residual is None
    assert decision.error


def test_an_exception_inside_the_agent_is_reported_not_raised():
    decision = make_policy(FakeAgent(raises=RuntimeError("XLA blew up"))).evaluate(
        terms=terms(), images=images()
    )
    assert not decision.ok
    assert "RuntimeError" in decision.error and "XLA blew up" in decision.error


def test_a_missing_residual_is_distinguishable_from_a_zero_residual():
    """Zero is a legitimate output; substituting it for failure would hide faults."""
    zero = make_policy(FakeAgent(np.zeros(6))).evaluate(terms=terms(), images=images())
    failed = make_policy(FakeAgent(np.zeros(7))).evaluate(terms=terms(), images=images())
    assert zero.ok and np.array_equal(zero.residual, np.zeros(6))
    assert not failed.ok and failed.residual is None


@pytest.mark.parametrize("missing", ["flange_pose", "wrench", "base_mean"])
def test_a_missing_state_term_degrades(missing):
    incomplete = {k: v for k, v in terms().items() if k != missing}
    decision = make_policy().evaluate(terms=incomplete, images=images())
    assert not decision.ok
    assert missing in decision.error


def test_a_wrong_width_state_term_degrades_rather_than_shifting_the_layout():
    decision = make_policy().evaluate(
        terms=terms(wrench=np.zeros(5)), images=images()
    )
    assert not decision.ok
    assert "wrench" in decision.error


@pytest.mark.parametrize(
    "frames",
    [
        {},                                              # missing entirely
        {"wrist_1": np.zeros((H, W, 3), np.float32)},    # wrong dtype
        {"wrist_1": np.zeros((H, W), np.uint8)},         # wrong rank
        {"wrist_1": np.zeros((H, W, 4), np.uint8)},      # wrong channels
    ],
)
def test_a_malformed_image_degrades(frames):
    decision = make_policy().evaluate(terms=terms(), images=frames)
    assert not decision.ok
    assert "wrist_1" in decision.error


@pytest.mark.parametrize("scale", [0.0, -0.1, 1.5, float("nan")])
def test_an_out_of_range_action_scale_is_rejected_at_construction(scale):
    """A scale > 1 could saturate the normalized action without the base policy."""
    with pytest.raises(ResidualPolicyError):
        make_policy(action_scale=scale)


def test_action_scale_attenuates_the_residual():
    action = np.array([1.0, -1.0, 0.5, 0.0, 0.25, -0.75])
    decision = make_policy(FakeAgent(action), action_scale=0.1).evaluate(
        terms=terms(), images=images()
    )
    assert np.allclose(decision.residual, action * 0.1)


def test_deterministic_mode_is_forwarded_and_reported():
    agent = FakeAgent()
    decision = make_policy(agent, deterministic=True).evaluate(
        terms=terms(), images=images()
    )
    assert agent.calls[0]["argmax"] is True
    assert decision.deterministic is True


def test_each_step_draws_a_fresh_key():
    """Reusing one key would freeze exploration noise while looking healthy."""
    agent = FakeAgent()
    policy = make_policy(agent)
    for _ in range(3):
        policy.evaluate(terms=terms(), images=images())
    seeds = [np.asarray(call["seed"]).tobytes() for call in agent.calls]
    assert len(set(seeds)) == 3


def test_update_params_bumps_the_version_reported_with_each_decision():
    class Stateful(FakeAgent):
        def __init__(self):
            super().__init__()
            self.state = self

        def replace(self, **kwargs):
            return self

    policy = ResidualPolicy(Stateful(), ResidualObservationSpec())
    assert policy.evaluate(terms=terms(), images=images()).params_version == 0
    assert policy.update_params({"w": 1}) == 1
    assert policy.update_params({"w": 2}) == 2
    assert policy.evaluate(terms=terms(), images=images()).params_version == 2


def test_a_null_agent_is_rejected():
    with pytest.raises(ResidualPolicyError):
        ResidualPolicy(None, ResidualObservationSpec())


@pytest.mark.parametrize(
    "kwargs", [{}, {"wrench": False}, {"socket_pose": True}, {"base_mean": False}]
)
def test_runtime_state_matches_the_converter_byte_for_byte(tmp_path, kwargs):
    """The guard against silent train/inference divergence.

    Offline (HDF5) and online (ROS) observations built from the same values must
    have identical layout and dtype.
    """
    from sparr_phase2_residual.demo_export import export_demos
    from test_demo_export import write_session
    import h5py

    spec = ResidualObservationSpec(**kwargs)
    session = write_session(tmp_path / "s.h5", steps=5)
    demos, stats = export_demos([session], spec)
    assert stats.trials_exported == 1

    with h5py.File(session, "r") as handle:
        trial = handle["trials/trial_000000"]
        recorded = {
            "flange_pose": trial["observations/robot/flange_pose_world_xyzw"][:],
            "wrench": trial["observations/robot/ext_wrench_tcp"][:],
            "base_mean": trial["actions/policy/mu_raw_normalized"][:],
            "socket_pose": trial["observations/task/socket_root_exact_world_xyzw"][:],
        }

    agent = FakeAgent()
    policy = ResidualPolicy(agent, spec)
    for index, demo in enumerate(demos):
        policy.evaluate(
            terms={name: recorded[name][index] for name, _ in spec.terms()},
            images=images(),
        )
        online = np.asarray(agent.calls[index]["observations"]["state"])
        offline = demo["observations"]["state"]
        assert online.shape == offline.shape == (1, spec.state_dim())
        assert online.dtype == offline.dtype == np.float32
        assert np.array_equal(online, offline), f"layout diverged at step {index}"


def test_residual_stats_report_magnitude_direction_and_saturation():
    stats = ResidualOutputStats()
    # Two steps whose signed means cancel on dx but not on dy, so the signed and
    # absolute summaries have to disagree for the metric to be worth logging.
    stats.observe(np.array([0.4, 0.2, 0.0, 0.0, 0.0, 0.0]))
    stats.observe(np.array([-0.4, 0.6, 0.0, 0.0, 0.0, 0.0]))

    summary = stats.summary(action_scale=0.5)

    assert summary["residual_steps"] == 2
    assert summary["residual_mean_axes"][0] == pytest.approx(0.0)
    assert summary["residual_abs_mean_axes"][0] == pytest.approx(0.4)
    assert summary["residual_mean_axes"][1] == pytest.approx(0.4)
    # mean over all 12 components: (0.4+0.2 + 0.4+0.6) / 12
    assert summary["residual_abs_mean"] == pytest.approx(1.6 / 12)
    assert summary["residual_abs_mean_scaled"] == pytest.approx(0.5 * 1.6 / 12)
    assert summary["residual_max_abs"] == pytest.approx(0.6)
    assert summary["residual_saturation"] == pytest.approx(0.0)


def test_residual_stats_count_saturated_components():
    stats = ResidualOutputStats()
    stats.observe(np.array([0.995, -1.0, 0.0, 0.0, 0.0, 0.0]))

    # 2 of 6 components are at the tanh rail.
    assert stats.summary(action_scale=1.0)["residual_saturation"] == pytest.approx(1 / 3)


def test_residual_stats_are_empty_until_a_residual_arrives():
    """An empty summary keeps base-only episodes off the curves.

    Zeros would be indistinguishable from a residual that ran and output nothing.
    """
    stats = ResidualOutputStats()
    assert stats.summary(action_scale=0.5) == {}
    assert stats.steps == 0


def test_residual_stats_ignore_non_finite_and_misshaped_output():
    stats = ResidualOutputStats()
    stats.observe(np.array([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0]))
    stats.observe(np.zeros(3))
    assert stats.summary(action_scale=1.0) == {}

    stats.observe(np.full(6, 0.25))
    assert stats.summary(action_scale=1.0)["residual_steps"] == 1


def test_residual_stats_reset_between_episodes():
    """Per-episode, or a rising trend is swamped by everything before it."""
    stats = ResidualOutputStats()
    stats.observe(np.full(6, 0.9))
    stats.reset()
    stats.observe(np.full(6, 0.1))

    summary = stats.summary(action_scale=1.0)
    assert summary["residual_steps"] == 1
    assert summary["residual_abs_mean"] == pytest.approx(0.1)
    assert summary["residual_max_abs"] == pytest.approx(0.1)


def test_decision_carries_the_unscaled_output_alongside_the_scaled_one():
    """Stats must measure the network, not the authority it was given.

    ``residual`` is post-action_scale, so saturation is measured on
    ``raw_residual``.
    """
    action = np.array([1.0, -1.0, 0.5, 0.0, 0.25, -0.75])
    decision = make_policy(FakeAgent(action), action_scale=0.1).evaluate(
        terms=terms(), images=images()
    )
    assert np.allclose(decision.raw_residual, action)
    assert np.allclose(decision.residual, action * 0.1)

    saturated = ResidualOutputStats()
    saturated.observe(decision.raw_residual)
    # 2 of 6 components are railed in the raw output.
    assert saturated.summary(action_scale=0.1)["residual_saturation"] == pytest.approx(1 / 3)

    hidden = ResidualOutputStats()
    hidden.observe(decision.residual)
    assert hidden.summary(action_scale=0.1)["residual_saturation"] == 0.0


def test_a_failed_decision_carries_no_raw_output_and_is_ignored_by_stats():
    decision = make_policy(FakeAgent(np.zeros(7))).evaluate(
        terms=terms(), images=images()
    )
    assert not decision.ok and decision.raw_residual is None

    stats = ResidualOutputStats()
    stats.observe(decision.raw_residual)
    assert stats.summary(action_scale=0.5) == {}
