# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Phase 2 configuration: an overlay that cannot silently mean the wrong thing."""

import pytest

from sparr_phase2_residual.config import (
    DEFAULT_ACTION_SCALE,
    Phase2Config,
    Phase2ConfigError,
    ResidualRuntimeConfig,
    TRAINING_CONFIG_FILENAME,
    evaluation_config,
)

REPO_CONFIG = "phase2_live.yaml"


def test_defaults_are_shadow_mode_and_a_conservative_scale():
    """A fresh residual must not be able to drive the robot by default."""
    config = Phase2Config()
    assert config.runtime.mode == "shadow"
    assert config.runtime.commands_residual is False
    assert config.runtime.action_scale == DEFAULT_ACTION_SCALE <= 0.1
    assert config.observation.state_dim() == 19
    assert config.observation.socket_pose is False


def test_active_mode_is_the_only_mode_that_commands_the_residual():
    assert ResidualRuntimeConfig(mode="active").commands_residual is True
    assert ResidualRuntimeConfig(mode="shadow").commands_residual is False


@pytest.mark.parametrize("mode", ["", "on", "live", "Shadow", "ACTIVE"])
def test_an_unrecognised_mode_is_rejected(mode):
    """An unknown mode must be refused, not treated as 'shadow' or 'active'."""
    with pytest.raises(Phase2ConfigError, match="mode"):
        ResidualRuntimeConfig(mode=mode)


@pytest.mark.parametrize("scale", [0.0, -0.5, 1.01, 2.0])
def test_an_out_of_range_action_scale_is_rejected(scale):
    with pytest.raises(Phase2ConfigError, match="action_scale"):
        ResidualRuntimeConfig(action_scale=scale)


@pytest.mark.parametrize("value", [0, -1])
def test_max_consecutive_failures_must_allow_at_least_one(value):
    with pytest.raises(Phase2ConfigError):
        ResidualRuntimeConfig(max_consecutive_failures=value)


def test_state_terms_select_the_layout():
    config = Phase2Config.from_mapping(
        {"residual": {"state_terms": {"wrench": False, "socket_pose": True}}}
    )
    assert config.observation.state_dim() == 7 + 6 + 7
    assert config.observation.describe() == (
        "[0:7] flange_pose, [7:13] base_mean, [13:20] socket_pose"
    )


def test_an_unknown_state_term_is_rejected_rather_than_ignored():
    """A typo'd term must not silently leave the layout at its default."""
    with pytest.raises(Phase2ConfigError, match="wrenches"):
        Phase2Config.from_mapping({"residual": {"state_terms": {"wrenches": True}}})


def test_an_unknown_residual_key_is_rejected():
    with pytest.raises(Phase2ConfigError, match="image_key"):
        Phase2Config.from_mapping({"residual": {"image_key": ["wrist_1"]}})


def test_an_unknown_runtime_key_is_rejected():
    with pytest.raises(Phase2ConfigError, match="scale"):
        Phase2Config.from_mapping({"residual": {"runtime": {"scale": 0.5}}})


def test_a_string_image_keys_is_rejected_not_iterated_as_characters():
    with pytest.raises(Phase2ConfigError, match="list"):
        Phase2Config.from_mapping({"residual": {"image_keys": "wrist_1"}})


def test_an_empty_mapping_yields_the_defaults():
    assert Phase2Config.from_mapping({}) == Phase2Config()
    assert Phase2Config.from_mapping({"residual": None}) == Phase2Config()


def test_a_missing_file_is_reported_with_its_path(tmp_path):
    with pytest.raises(Phase2ConfigError, match="absent.yaml"):
        Phase2Config.load(tmp_path / "absent.yaml")


def test_the_shipped_repo_config_is_ready_to_train():
    """The shipped config must parse and train safely.

    Active with no params is safe only when the actor waits for a learner;
    otherwise a random residual would command the robot.
    """
    from pathlib import Path

    path = Path(__file__).resolve().parents[5] / REPO_CONFIG
    if not path.is_file():
        pytest.skip(f"{path} not present")
    config = Phase2Config.load(path)
    assert config.runtime.mode == "active"
    assert config.runtime.params_path is None
    assert config.runtime.learner_host and config.runtime.wait_for_learner
    assert config.runtime.deterministic is False
    assert config.observation.state_dim() == 19
    # Demos, the learner encoder and this file must agree on cameras; a
    # mismatch only surfaces as a shape error at the first buffer insert.
    assert config.observation.image_keys == ("wrist_1", "wrist_2")
    assert "19-D" in config.describe() or "19" in config.describe()


def test_no_learner_is_configured_by_default():
    """Recording-only must be the default: a missing learner cannot hang a run."""
    runtime = ResidualRuntimeConfig()
    assert runtime.learner_host is None
    assert runtime.wait_for_learner is False
    assert runtime.learner_port == 5588
    assert runtime.learner_broadcast_port == 5589


def test_promotion_is_on_by_default_with_an_unseeded_median():
    runtime = ResidualRuntimeConfig()
    assert runtime.promote_episodes is True
    assert runtime.promotion_seed_lengths == ()


@pytest.mark.parametrize("port", [0, -1, 70000])
def test_an_invalid_learner_port_is_rejected(port):
    with pytest.raises(Phase2ConfigError, match="port"):
        ResidualRuntimeConfig(learner_port=port)


def test_the_two_learner_ports_must_differ():
    """agentlace binds both; the same number twice fails obscurely at runtime."""
    with pytest.raises(Phase2ConfigError, match="must differ"):
        ResidualRuntimeConfig(learner_port=5588, learner_broadcast_port=5588)


def test_a_zero_actor_queue_is_rejected():
    with pytest.raises(Phase2ConfigError, match="actor_queue_capacity"):
        ResidualRuntimeConfig(actor_queue_capacity=0)


@pytest.mark.parametrize("bad", [0, -3])
def test_a_nonsense_promotion_seed_length_is_rejected(bad):
    with pytest.raises(Phase2ConfigError, match="promotion_seed_lengths"):
        ResidualRuntimeConfig(promotion_seed_lengths=(10, bad))


def test_the_learner_link_can_be_configured_from_yaml():
    config = Phase2Config.from_mapping(
        {
            "residual": {
                "runtime": {
                    "learner_host": "10.0.0.5",
                    "learner_port": 6000,
                    "learner_broadcast_port": 6001,
                    "promotion_seed_lengths": [40, 50, 60],
                }
            }
        }
    )
    assert config.runtime.learner_host == "10.0.0.5"
    assert config.runtime.learner_port == 6000
    assert config.runtime.promotion_seed_lengths == (40, 50, 60)


def test_the_shipped_config_does_not_point_at_someone_elses_machine():
    """A checked-in remote learner host would make a fresh checkout connect to it.

    localhost, the normal single-machine setup, is allowed.
    """
    from pathlib import Path

    path = Path(__file__).resolve().parents[5] / REPO_CONFIG
    if not path.is_file():
        pytest.skip(f"{path} not present")
    host = Phase2Config.load(path).runtime.learner_host
    assert host in (None, "localhost", "127.0.0.1"), (
        f"learner_host={host!r} points off this machine; unset it before committing"
    )


def test_the_collector_warns_when_a_learner_cannot_receive_anything():
    """Phase 1 shadow mode and residual shadow mode are different things.

    Phase 1 ``execution_mode='shadow'`` publishes no command, so every action is
    non-training-eligible and the online loop ships nothing; a learner configured
    in that mode is flagged at startup.
    """
    import inspect

    from sparr_phase2_residual import collector_node

    init = inspect.getsource(collector_node.Phase2CollectorNode.__init__)
    assert "execution_mode is 'shadow'" in init
    assert "learner_host" in init

    ship = inspect.getsource(collector_node.Phase2CollectorNode._ship_episode)
    assert "non-training-eligible" in ship, (
        "an empty episode must say why, not return silently"
    )


def test_residual_shadow_mode_is_not_the_same_as_execution_mode_shadow():
    """The safe combination is: collect live, residual mode shadow."""
    runtime = ResidualRuntimeConfig(mode="shadow")
    assert runtime.commands_residual is False
    # Only Phase 1's execution_mode controls whether the robot is commanded.
    assert not any("execution" in f for f in ResidualRuntimeConfig.__dataclass_fields__)


class TestRetryConfig:
    def test_retry_defaults_to_off(self) -> None:
        config = Phase2Config.from_mapping({"residual": {}})
        assert config.retry.enabled is False
        assert config.retry.success_rate_threshold == 0.8
        assert config.retry.max_repeats == 5
        assert config.retry.promote_on_success is True

    def test_retry_block_is_parsed(self) -> None:
        config = Phase2Config.from_mapping(
            {
                "residual": {
                    "retry": {
                        "enabled": True,
                        "success_rate_threshold": 0.9,
                        "window": 30,
                        "max_repeats": 2,
                        "promote_on_success": False,
                    }
                }
            }
        )
        assert config.retry.enabled is True
        assert config.retry.success_rate_threshold == 0.9
        assert config.retry.window == 30
        assert config.retry.max_repeats == 2
        assert config.retry.promote_on_success is False

    def test_an_unknown_retry_key_is_rejected(self) -> None:
        with pytest.raises(Phase2ConfigError):
            Phase2Config.from_mapping(
                {"residual": {"retry": {"max_retries": 5}}}
            )

    @pytest.mark.parametrize(
        "retry",
        [
            {"success_rate_threshold": 1.5},
            {"success_rate_threshold": -0.5},
            {"window": 0},
            {"max_repeats": 0},
        ],
    )
    def test_out_of_range_retry_values_are_rejected(self, retry) -> None:
        with pytest.raises(Phase2ConfigError):
            Phase2Config.from_mapping({"residual": {"retry": retry}})


def _params(tmp_path, *, sidecar=None):
    run = tmp_path / "run"
    run.mkdir()
    params = run / "actor_params_step100.pkl"
    params.write_bytes(b"not read by evaluation_config")
    if sidecar is not None:
        (run / TRAINING_CONFIG_FILENAME).write_text(sidecar)
    return params


def test_evaluation_forces_a_fixed_deterministic_checkpoint(tmp_path):
    base = Phase2Config.from_mapping({"residual": {
        "retry": {"enabled": True},
        "runtime": {"mode": "shadow", "learner_host": "localhost",
                    "wait_for_learner": True, "params_poll_ticks": 5},
    }})
    config, sidecar = evaluation_config(base, _params(tmp_path))

    assert sidecar is None
    runtime = config.runtime
    assert runtime.mode == "active"
    assert runtime.params_path.endswith("actor_params_step100.pkl")
    assert runtime.deterministic is True
    # A connected learner or a poll would swap weights mid-measurement.
    assert runtime.learner_host is None and not runtime.wait_for_learner
    assert runtime.params_poll_ticks == 0
    assert runtime.promote_episodes is False
    assert config.retry.enabled is False


def test_evaluation_can_sample(tmp_path):
    config, _ = evaluation_config(Phase2Config(), _params(tmp_path), stochastic=True)
    assert config.runtime.deterministic is False


def test_evaluation_takes_layout_and_scale_from_the_run(tmp_path):
    # Base config: 19-D two-camera at 0.5; the trained run: 13-D at 0.3.
    base = Phase2Config.from_mapping({"residual": {
        "image_keys": ["wrist_1", "wrist_2"],
        "runtime": {"action_scale": 0.5},
    }})
    trained = (
        "residual:\n"
        "  image_keys: [wrist_1]\n"
        "  state_terms: {wrench: false}\n"
        "  runtime: {action_scale: 0.3}\n"
    )
    config, sidecar = evaluation_config(base, _params(tmp_path, sidecar=trained))

    assert sidecar is not None
    assert config.observation.image_keys == ("wrist_1",)
    assert config.observation.state_dim() == 13
    assert config.runtime.action_scale == 0.3


def test_evaluation_refuses_a_missing_params_file(tmp_path):
    with pytest.raises(Phase2ConfigError, match="no actor params"):
        evaluation_config(Phase2Config(), tmp_path / "absent.pkl")
