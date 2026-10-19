# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Safe loading and validation of SPARR Phase 1 policy metadata.

ROS-independent. Extracts only the deployment contract needed by the stochastic
policy and action pipeline; the ``rsl_rl`` model is built by the runtime adapter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


#: Encodable observation layouts, keyed by the checkpoint's declared term order.
#: Both are 18-D (tool point and socket keypoint, each position + 6-D rotation)
#: but differ in order and tool point, so they must never be guessed.
#: ``taskspace_physx`` observes the flange shifted by the checkpoint tool offset;
#: ``newton`` observes the flange itself (zero offset) and puts the socket first.
PHASE1_OBSERVATION_ORDER = (
    "eef_pos",
    "eef_rot_6d",
    "socket_kp_pos",
    "socket_kp_rot_6d",
)
NEWTON_OBSERVATION_ORDER = (
    "socket_pos",
    "tool_pos",
    "tool_rot_6d",
    "socket_rot_6d",
)
SUPPORTED_OBSERVATION_ORDERS = (
    PHASE1_OBSERVATION_ORDER,
    NEWTON_OBSERVATION_ORDER,
)
PHASE1_OBSERVATION_DIM = 18
PHASE1_ACTION_DIM = 6

_OBSERVATION_META_KEYS = {
    "concatenate_terms",
    "concatenate_dim",
    "enable_corruption",
    "history_length",
    "flatten_history_dim",
}
_OBSERVATION_TERM_SIZES = {
    "eef_pos": 3,
    "eef_rot_6d": 6,
    "socket_kp_pos": 3,
    "socket_kp_rot_6d": 6,
    # Newton's names for the same quantities.
    "tool_pos": 3,
    "tool_rot_6d": 6,
    "socket_pos": 3,
    "socket_rot_6d": 6,
}


class CheckpointConfigError(ValueError):
    """Raised when checkpoint metadata cannot satisfy the Phase 1 contract."""


class _IsaacSafeLoader(yaml.SafeLoader):
    """SafeLoader extended for the two Python tags emitted by Isaac Lab.

    Isaac Lab serializes tuples and ``slice`` selectors into ``env.yaml``.
    Using ``yaml.UnsafeLoader`` merely to recover these two value types would
    allow arbitrary Python object construction, so support them explicitly.
    """


def _construct_python_tuple(loader: _IsaacSafeLoader, node: yaml.Node) -> tuple[Any, ...]:
    return tuple(loader.construct_sequence(node, deep=True))


def _construct_python_slice(loader: _IsaacSafeLoader, node: yaml.Node) -> slice:
    values = loader.construct_sequence(node, deep=True)
    if len(values) > 3:
        raise CheckpointConfigError(f"slice tag has {len(values)} values; expected at most 3")
    return slice(*values)


_IsaacSafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", _construct_python_tuple
)
_IsaacSafeLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice",
    _construct_python_slice,
)


@dataclass(frozen=True)
class Phase1PolicySpec:
    """Immutable policy/deployment contract derived from checkpoint configs."""

    observation_order: tuple[str, ...]
    observation_dim: int
    action_dim: int
    eef_offset: tuple[float, float, float]
    socket_keypoint_offset: tuple[float, float, float]
    position_scale: tuple[float, float, float]
    orientation_scale: tuple[float, float, float]
    training_action_clip: float
    ee_body_frame: str
    quaternion_order: str
    control_period_s: float
    recurrent_type: str
    recurrent_hidden_dim: int
    recurrent_num_layers: int
    distribution_class: str

    @property
    def control_rate_hz(self) -> float:
        return 1.0 / self.control_period_s


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Load an Isaac Lab YAML file without enabling arbitrary constructors."""

    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            value = yaml.load(stream, Loader=_IsaacSafeLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise CheckpointConfigError(f"failed to load YAML config {config_path}: {exc}") from exc

    if not isinstance(value, Mapping):
        raise CheckpointConfigError(
            f"YAML config {config_path} must contain a mapping, got {type(value).__name__}"
        )
    return dict(value)


def resolve_config_paths(source: str | Path) -> tuple[Path, Path]:
    """Resolve ``params/agent.yaml`` and ``params/env.yaml`` from a run or checkpoint.

    ``source`` may be a checkpoint file, its run directory, the ``params``
    directory, or either YAML file itself.
    """

    path = Path(source).expanduser()
    if path.is_file() and path.name in {"agent.yaml", "env.yaml"}:
        candidates = [path.parent]
    elif path.is_file():
        candidates = [path.parent / "params", path.parent]
    else:
        candidates = [path / "params", path]

    for params_dir in candidates:
        agent_path = params_dir / "agent.yaml"
        env_path = params_dir / "env.yaml"
        if agent_path.is_file() and env_path.is_file():
            return agent_path, env_path

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise CheckpointConfigError(
        f"could not find agent.yaml and env.yaml for {path}; searched: {searched}"
    )


def load_agent_env_configs(source: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the paired agent and environment configs for a checkpoint run."""

    agent_path, env_path = resolve_config_paths(source)
    return load_yaml_mapping(agent_path), load_yaml_mapping(env_path)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointConfigError(f"{name} must be a mapping")
    return value


def _vector3(
    value: Any,
    name: str,
    *,
    default: Sequence[float] | None = None,
) -> tuple[float, float, float]:
    if value is None:
        if default is None:
            raise CheckpointConfigError(f"{name} is required")
        value = default
    if isinstance(value, (int, float)):
        values = [float(value)] * 3
    else:
        try:
            values = [float(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise CheckpointConfigError(f"{name} must be a scalar or 3-vector") from exc
    if len(values) != 3 or not all(math.isfinite(item) for item in values):
        raise CheckpointConfigError(f"{name} must contain three finite values, got {values}")
    return values[0], values[1], values[2]


def _observation_order(env_config: Mapping[str, Any]) -> tuple[str, ...]:
    explicit = env_config.get("obs_order")
    if explicit:
        return tuple(str(term) for term in explicit)

    observations = _mapping(env_config.get("observations"), "env.observations")
    policy = _mapping(observations.get("policy"), "env.observations.policy")
    # A term set to None is declared but disabled: Isaac Lab keeps the key but
    # excludes it from the observation vector.
    return tuple(
        str(term)
        for term, spec in policy.items()
        if term not in _OBSERVATION_META_KEYS and spec is not None
    )


def _term_offset(
    policy_observations: Mapping[str, Any],
    term_names: Sequence[str],
    name: str,
) -> tuple[float, float, float]:
    for term_name in term_names:
        term = policy_observations.get(term_name)
        if not isinstance(term, Mapping):
            continue
        params = term.get("params") or {}
        if not isinstance(params, Mapping):
            raise CheckpointConfigError(f"parameters for observation term {term_name!r} must map")
        return _vector3(params.get("offset"), name, default=(0.0, 0.0, 0.0))
    raise CheckpointConfigError(
        f"could not find any of observation terms {tuple(term_names)} needed for {name}"
    )


def _task_space_scales(
    action_config: Mapping[str, Any],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if "position_scale" in action_config and "orientation_scale" in action_config:
        return (
            _vector3(action_config["position_scale"], "position_scale"),
            _vector3(action_config["orientation_scale"], "orientation_scale"),
        )

    pose_objectives = [
        objective
        for objective in action_config.get("objectives", ())
        if isinstance(objective, Mapping) and "scale" in objective
    ]
    if len(pose_objectives) != 1:
        raise CheckpointConfigError(
            "could not resolve task-space scales: expected direct position/orientation "
            "scales or one scaled pose objective"
        )
    scale = pose_objectives[0]["scale"]
    if isinstance(scale, (int, float)):
        vector = _vector3(scale, "pose objective scale")
        return vector, vector
    values = [float(item) for item in scale]
    if len(values) != 6:
        raise CheckpointConfigError(
            f"pose objective scale must be scalar or 6-vector, got {values}"
        )
    return _vector3(values[:3], "position scale"), _vector3(values[3:], "orientation scale")


def _symmetric_action_clip(
    agent_config: Mapping[str, Any], action_config: Mapping[str, Any]
) -> float:
    runner_clip = agent_config.get("clip_actions", 1.0)
    limit = 1.0 if runner_clip is None else abs(float(runner_clip))
    if not math.isfinite(limit) or limit <= 0.0:
        raise CheckpointConfigError(f"clip_actions must be finite and positive, got {runner_clip}")

    action_clip = action_config.get("clip")
    if not action_clip:
        return limit
    if not isinstance(action_clip, Mapping):
        raise CheckpointConfigError("arm_action.clip must be a mapping of patterns to bounds")
    for pattern, bounds in action_clip.items():
        if not isinstance(bounds, Sequence) or len(bounds) != 2:
            raise CheckpointConfigError(f"clip bounds for {pattern!r} must be [low, high]")
        low, high = float(bounds[0]), float(bounds[1])
        if not math.isclose(low, -high, rel_tol=0.0, abs_tol=1.0e-12) or high <= 0.0:
            raise CheckpointConfigError(
                f"clip bounds for {pattern!r} must be symmetric about zero, got {(low, high)}"
            )
        limit = min(limit, high)
    return limit


def _quaternion_order(
    policy_observations: Mapping[str, Any],
    observation_order: tuple[str, ...] = PHASE1_OBSERVATION_ORDER,
) -> str:
    """Which quaternion convention the observation is expressed in.

    Scans the terms the checkpoint declares, so other term names do not fall
    through to 'wxyz'. When no observed term is a quaternion (e.g. 6-D-only
    layouts) the convention cannot affect the encoding and 'xyzw', the internal
    convention, is reported.
    """
    for term_name in observation_order:
        term = policy_observations.get(term_name)
        if not isinstance(term, Mapping):
            continue
        if "contrib.deploy" in str(term.get("func", "")):
            return "xyzw"
    if not any(name.endswith("_quat") for name in observation_order):
        return "xyzw"
    return "wxyz"


def build_phase1_policy_spec(
    agent_config: Mapping[str, Any], env_config: Mapping[str, Any]
) -> Phase1PolicySpec:
    """Validate configs and extract the exact 18-D/6-D Phase 1 contract."""

    observation_order = _observation_order(env_config)
    if observation_order not in SUPPORTED_OBSERVATION_ORDERS:
        raise CheckpointConfigError(
            f"Phase 1 supports observation orders {SUPPORTED_OBSERVATION_ORDERS}, "
            f"got {observation_order}"
        )

    observations = _mapping(env_config.get("observations"), "env.observations")
    policy_observations = _mapping(
        observations.get("policy"), "env.observations.policy"
    )
    derived_observation_dim = sum(_OBSERVATION_TERM_SIZES[term] for term in observation_order)
    observation_dim = int(env_config.get("observation_space", derived_observation_dim))
    if observation_dim != PHASE1_OBSERVATION_DIM or derived_observation_dim != observation_dim:
        raise CheckpointConfigError(
            f"Phase 1 observation dimension must be {PHASE1_OBSERVATION_DIM}, "
            f"got metadata={observation_dim}, derived={derived_observation_dim}"
        )

    actions = _mapping(env_config.get("actions"), "env.actions")
    arm_action = _mapping(actions.get("arm_action"), "env.actions.arm_action")
    action_dim = int(env_config.get("action_space", PHASE1_ACTION_DIM))
    if action_dim != PHASE1_ACTION_DIM:
        raise CheckpointConfigError(
            f"Phase 1 action dimension must be {PHASE1_ACTION_DIM}, got {action_dim}"
        )
    action_space = str(env_config.get("policy_action_space", "task"))
    if action_space != "task":
        raise CheckpointConfigError(f"Phase 1 requires task-space actions, got {action_space!r}")

    position_scale, orientation_scale = _task_space_scales(arm_action)
    if any(value <= 0.0 for value in (*position_scale, *orientation_scale)):
        raise CheckpointConfigError("task-space action scales must all be positive")

    sim = _mapping(env_config.get("sim"), "env.sim")
    try:
        control_period_s = float(sim["dt"]) * int(env_config["decimation"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointConfigError("env.sim.dt and env.decimation are required") from exc
    if not math.isfinite(control_period_s) or control_period_s <= 0.0:
        raise CheckpointConfigError(f"control period must be positive, got {control_period_s}")

    actor = agent_config.get("actor") or agent_config.get("policy") or {}
    actor = _mapping(actor, "agent.actor")
    distribution = actor.get("distribution_cfg") or {}
    distribution = _mapping(distribution, "agent.actor.distribution_cfg")

    ee_body_frame = str(arm_action.get("body_name", ""))
    quaternion_order = _quaternion_order(policy_observations, observation_order)
    recurrent_type = str(actor.get("rnn_type", "")).lower()
    distribution_class = str(distribution.get("class_name", ""))
    if ee_body_frame != "flange":
        raise CheckpointConfigError(
            f"SPARR Phase 1 requires arm_action.body_name='flange', got {ee_body_frame!r}"
        )
    if quaternion_order != "xyzw":
        raise CheckpointConfigError(
            f"SPARR Phase 1 requires xyzw policy observations, got {quaternion_order!r}"
        )
    if recurrent_type != "lstm":
        raise CheckpointConfigError(
            f"SPARR Phase 1 requires an LSTM actor, got {recurrent_type!r}"
        )
    if "HeteroscedasticGaussianDistribution" not in distribution_class:
        raise CheckpointConfigError(
            "SPARR Phase 1 requires HeteroscedasticGaussianDistribution, "
            f"got {distribution_class!r}"
        )

    return Phase1PolicySpec(
        observation_order=observation_order,
        observation_dim=observation_dim,
        action_dim=action_dim,
        # "tool_pos" is Newton's name; its zero offset means the policy observes
        # the flange itself, which the encoder must honour.
        eef_offset=_term_offset(
            policy_observations,
            ("eef_pos", "tool_pos", "tcp_pose", "flange_pose"),
            "EEF offset",
        ),
        socket_keypoint_offset=_term_offset(
            policy_observations,
            ("socket_kp_pos", "socket_pos"),
            "socket keypoint offset",
        ),
        position_scale=position_scale,
        orientation_scale=orientation_scale,
        training_action_clip=_symmetric_action_clip(agent_config, arm_action),
        ee_body_frame=ee_body_frame,
        quaternion_order=quaternion_order,
        control_period_s=control_period_s,
        recurrent_type=recurrent_type,
        recurrent_hidden_dim=int(actor.get("rnn_hidden_dim", 0)),
        recurrent_num_layers=int(actor.get("rnn_num_layers", 0)),
        distribution_class=distribution_class,
    )


