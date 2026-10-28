# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA OneWay Noncommercial License (March 22, 2022).
# See the LICENSE file in the repository root for the full license text:
# https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf

"""Concrete RSL-RL checkpoint loader for one-pass stochastic inference.

This is the only module that knows how the supplied RSL-RL checkpoint is
constructed. The rest of the collector depends on the small policy/sample
contract and remains testable without Torch or RSL-RL.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .checkpoint import (
    Phase1PolicySpec,
    build_phase1_policy_spec,
    load_yaml_mapping,
    resolve_config_paths,
)
from .stochastic_policy import PolicySample, RecurrentStochasticPolicy


class PolicyLoadError(RuntimeError):
    """Raised when the checkpoint cannot be instantiated exactly."""


@dataclass(frozen=True)
class RuntimePolicySample:
    raw_mean: np.ndarray
    raw_std: np.ndarray
    raw_sample: np.ndarray
    recurrent_state_in: np.ndarray
    recurrent_state_out: np.ndarray


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - target image supplies Torch
        raise PolicyLoadError("Torch is not installed in this runtime") from exc
    return torch


def _rsl_rl_classes() -> tuple[Any, Any]:
    try:
        from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
    except ImportError as exc:  # pragma: no cover - target image supplies RSL-RL
        raise PolicyLoadError(f"cannot import rsl_rl.modules: {exc}") from exc
    return ActorCritic, ActorCriticRecurrent


def _actor_config(agent_config: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    ActorCritic, ActorCriticRecurrent = _rsl_rl_classes()
    legacy_policy = dict(agent_config.get("policy") or {})
    legacy_name = legacy_policy.pop("class_name", None)
    if legacy_name:
        classes = {
            "ActorCritic": ActorCritic,
            "ActorCriticRecurrent": ActorCriticRecurrent,
        }
        try:
            return classes[str(legacy_name)], legacy_policy
        except KeyError as exc:
            raise PolicyLoadError(f"unsupported RSL-RL policy class {legacy_name!r}") from exc

    actor = agent_config.get("actor")
    if not isinstance(actor, Mapping):
        raise PolicyLoadError("agent.yaml is missing the actor mapping")
    critic = agent_config.get("critic") or actor
    if not isinstance(critic, Mapping):
        raise PolicyLoadError("agent.yaml critic must be a mapping")
    distribution = actor.get("distribution_cfg") or {}
    if not isinstance(distribution, Mapping):
        raise PolicyLoadError("actor.distribution_cfg must be a mapping")
    distribution_name = str(distribution.get("class_name", ""))
    kwargs = {
        "actor_hidden_dims": list(actor["hidden_dims"]),
        "critic_hidden_dims": list(critic["hidden_dims"]),
        "activation": str(actor.get("activation", "elu")),
        "actor_obs_normalization": bool(actor.get("obs_normalization", False)),
        "critic_obs_normalization": bool(critic.get("obs_normalization", False)),
        "rnn_type": str(actor.get("rnn_type", "lstm")),
        "rnn_hidden_dim": int(actor.get("rnn_hidden_dim", 256)),
        "rnn_num_layers": int(actor.get("rnn_num_layers", 1)),
        "init_noise_std": float(distribution.get("init_std", 1.0)),
        "noise_std_type": str(distribution.get("std_type", "scalar")),
        "state_dependent_std": "Heteroscedastic" in distribution_name,
    }
    return ActorCriticRecurrent, kwargs


def _observation_groups(agent_config: Mapping[str, Any]) -> dict[str, list[str]]:
    groups = dict(agent_config.get("obs_groups") or {})
    if "policy" not in groups:
        groups["policy"] = groups.pop("actor", ["policy"])
    groups.setdefault("critic", ["critic"])
    return {str(name): list(values) for name, values in groups.items()}


def _safe_torch_load(path: Path, device: Any) -> Mapping[str, Any]:
    torch = _torch()
    try:
        loaded = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # older Torch lacks weights_only
        loaded = torch.load(path, map_location=device)
    except Exception as exc:
        raise PolicyLoadError(f"failed to load checkpoint {path}: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise PolicyLoadError("checkpoint root must be a mapping")
    return loaded


def _load_weights(model: Any, checkpoint: Mapping[str, Any]) -> None:
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return
    if "actor_state_dict" in checkpoint:
        key_map = {
            "obs_normalizer.": "actor_obs_normalizer.",
            "mlp.": "actor.",
            "rnn.": "memory_a.",
        }
        actor_weights = {}
        for original_key, value in checkpoint["actor_state_dict"].items():
            key = original_key
            for prefix, replacement in key_map.items():
                if key.startswith(prefix):
                    key = replacement + key[len(prefix) :]
                    break
            actor_weights[key] = value
        load_result = model.load_state_dict(actor_weights, strict=False)
        if hasattr(load_result, "missing_keys") and hasattr(
            load_result, "unexpected_keys"
        ):
            missing = list(load_result.missing_keys)
            unexpected = list(load_result.unexpected_keys)
        elif isinstance(load_result, (tuple, list)) and len(load_result) == 2:
            missing = list(load_result[0])
            unexpected = list(load_result[1])
        elif isinstance(load_result, bool):
            # RSL-RL 3.x overrides ``load_state_dict`` and returns a resume-
            # training boolean after delegating to ``torch.nn.Module``.  It
            # discards PyTorch's ``_IncompatibleKeys`` result, so reconstruct
            # the same key-level contract from the model and supplied state.
            if not load_result:
                raise PolicyLoadError("RSL-RL rejected the actor checkpoint")
            model_keys = set(model.state_dict())
            supplied_keys = set(actor_weights)
            missing = sorted(model_keys - supplied_keys)
            unexpected = sorted(supplied_keys - model_keys)
        else:
            raise PolicyLoadError(
                "unsupported load_state_dict result from RSL-RL: "
                f"{type(load_result).__name__}"
            )
        if unexpected:
            raise PolicyLoadError(f"unexpected actor checkpoint keys: {unexpected}")
        # Critic-only keys may be absent from an actor-only export. Actor and
        # recurrent actor parameters may not be absent.
        required_prefixes = ("actor.", "memory_a.")
        required_missing = [key for key in missing if key.startswith(required_prefixes)]
        if required_missing:
            raise PolicyLoadError(f"missing actor checkpoint keys: {required_missing}")
        return
    raise PolicyLoadError(
        f"checkpoint has neither model_state_dict nor actor_state_dict; keys={list(checkpoint)}"
    )


def _numpy(tensor: Any) -> np.ndarray:
    return np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32).copy()


def _flatten_recurrent_state(model: Any, *, expected_size: int) -> np.ndarray:
    if expected_size < 1:
        raise PolicyLoadError("expected recurrent state size must be positive")
    memory = getattr(model, "memory_a", None)
    if memory is None:
        raise PolicyLoadError("recurrent policy does not expose memory_a")
    state = getattr(memory, "hidden_state", None)
    if state is None:
        state = getattr(memory, "hidden_states", None)
    if state is None:
        # RSL-RL commonly represents the post-reset state as ``None`` and
        # allocates zero h/c tensors on the first forward. Record the explicit
        # mathematical zero so HDF5 has one stable trailing shape from row 0.
        return np.zeros(expected_size, dtype=np.float32)
    tensors = state if isinstance(state, (tuple, list)) else (state,)
    values = [_numpy(tensor).reshape(-1) for tensor in tensors if tensor is not None]
    flattened = (
        np.concatenate(values) if values else np.zeros(expected_size, dtype=np.float32)
    )
    if flattened.size != expected_size:
        raise PolicyLoadError(
            f"recurrent state has {flattened.size} values; expected {expected_size}"
        )
    return flattened


def _clear_recurrent_state(model: Any) -> None:
    """Return recurrent memory to its post-construction, pre-forward state.

    ``ActorCritic*.reset(dones=None)`` masks by ``dones`` in some RSL-RL
    versions and is then a no-op, so hidden state produced by a forward pass can
    survive it.  Dropping the tensors restores the ``None`` that the runtime
    records as explicit zeros; if the attribute cannot be rebound, the tensors
    are zeroed in place instead (inference-mode tensors may only be mutated
    inside ``inference_mode``).
    """

    memory = getattr(model, "memory_a", None)
    if memory is None:
        return
    for attribute in ("hidden_state", "hidden_states"):
        state = getattr(memory, attribute, None)
        if state is None:
            continue
        try:
            setattr(memory, attribute, None)
        except AttributeError:
            torch = _torch()
            tensors = state if isinstance(state, (tuple, list)) else (state,)
            with torch.inference_mode():
                for tensor in tensors:
                    if tensor is not None:
                        tensor.zero_()


class RslRlPolicyRuntime:
    """Loaded policy with explicit reset and exactly-one-forward sampling."""

    def __init__(
        self,
        *,
        model: Any,
        device: Any,
        spec: Phase1PolicySpec,
    ) -> None:
        self._model = model
        self._device = device
        self.spec = spec
        self._stochastic = RecurrentStochasticPolicy(model)
        recurrent_multiplier = 2 if spec.recurrent_type == "lstm" else 1
        self._recurrent_state_size = (
            recurrent_multiplier
            * spec.recurrent_num_layers
            * spec.recurrent_hidden_dim
        )
        if self._recurrent_state_size < 1:
            raise PolicyLoadError("checkpoint recurrent state dimensions are invalid")
        self._reset_count = 0
        self._sample_count = 0

    @property
    def reset_count(self) -> int:
        return self._reset_count

    @property
    def sample_count(self) -> int:
        return self._sample_count

    def reset(self) -> None:
        self._stochastic.reset()
        _clear_recurrent_state(self._model)
        self._reset_count += 1

    def warm_up(self, passes: int = 2) -> None:
        """Compile CUDA kernels before the robot is ever commanded.

        The first forward pays lazy CUDA init and autotuning, far longer than a
        control period. Recurrent state is cleared afterwards so the first
        recorded sample starts from zeros; counters are left untouched.
        """

        torch = _torch()
        zeros = torch.zeros(
            (1, self.spec.observation_dim),
            dtype=torch.float32,
            device=self._device,
        )
        for _ in range(max(1, int(passes))):
            self._stochastic.sample_once({"policy": zeros})
        self._stochastic.reset()
        _clear_recurrent_state(self._model)

    def sample_once(self, observation: np.ndarray) -> RuntimePolicySample:
        torch = _torch()
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (self.spec.observation_dim,):
            raise ValueError(
                f"policy observation must have shape ({self.spec.observation_dim},), "
                f"got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("policy observation contains a non-finite value")

        recurrent_in = _flatten_recurrent_state(
            self._model,
            expected_size=self._recurrent_state_size,
        )
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self._device).unsqueeze(0)
        sample: PolicySample = self._stochastic.sample_once({"policy": tensor})
        recurrent_out = _flatten_recurrent_state(
            self._model,
            expected_size=self._recurrent_state_size,
        )
        self._sample_count += 1
        return RuntimePolicySample(
            raw_mean=_numpy(sample.raw_mean).reshape(-1),
            raw_std=_numpy(sample.raw_std).reshape(-1),
            raw_sample=_numpy(sample.raw_sample).reshape(-1),
            recurrent_state_in=recurrent_in,
            recurrent_state_out=recurrent_out,
        )


def _critic_width(checkpoint: Mapping[str, Any], env_config: Mapping[str, Any]) -> int:
    """Width of the critic observation, preferring the checkpoint's own weights.

    ActorCriticRecurrent constructs the (unused) critic, and ``env.state_space``
    is absent from some configs, so the width is read from the critic's first
    LSTM weight, shape ``(4 * hidden, input_size)``.
    """
    for container in ("critic_state_dict", "model_state_dict"):
        state = checkpoint.get(container)
        if not hasattr(state, "get"):
            continue
        for key, value in state.items():
            if "weight_ih_l0" not in key:
                continue
            if container == "model_state_dict" and "critic" not in key:
                continue
            shape = getattr(value, "shape", None)
            if shape is not None and len(shape) == 2 and int(shape[1]) > 0:
                return int(shape[1])

    declared = env_config.get("state_space")
    if declared:
        return int(declared)
    raise PolicyLoadError(
        "cannot determine the critic observation width: the checkpoint has no "
        "critic LSTM weights and env.state_space is unset"
    )


def load_rsl_rl_policy(
    checkpoint_path: str | Path,
    *,
    agent_yaml: str | Path | None = None,
    env_yaml: str | Path | None = None,
    device: str = "cuda:0",
    seed: int = 0,
) -> RslRlPolicyRuntime:
    """Instantiate the supplied checkpoint as a one-pass stochastic policy."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=True)
    if agent_yaml is None or env_yaml is None:
        resolved_agent, resolved_env = resolve_config_paths(checkpoint_path)
        agent_yaml = agent_yaml or resolved_agent
        env_yaml = env_yaml or resolved_env
    agent_config = load_yaml_mapping(agent_yaml)
    env_config = load_yaml_mapping(env_yaml)
    spec = build_phase1_policy_spec(agent_config, env_config)

    torch = _torch()
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise PolicyLoadError(f"configured device {device!r} requires CUDA")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    policy_class, policy_kwargs = _actor_config(agent_config)
    checkpoint = _safe_torch_load(checkpoint_path, selected_device)
    dummy = {
        "policy": torch.zeros((1, spec.observation_dim), dtype=torch.float32),
        "critic": torch.zeros(
            (1, _critic_width(checkpoint, env_config)), dtype=torch.float32
        ),
    }
    try:
        model = policy_class(
            dummy,
            _observation_groups(agent_config),
            spec.action_dim,
            **policy_kwargs,
        )
        _load_weights(model, checkpoint)
        model.to(selected_device)
        model.eval()
    except PolicyLoadError:
        raise
    except Exception as exc:
        raise PolicyLoadError(f"failed to construct RSL-RL policy: {exc}") from exc
    runtime = RslRlPolicyRuntime(model=model, device=selected_device, spec=spec)
    try:
        runtime.warm_up()
    except Exception as exc:
        raise PolicyLoadError(f"policy warm-up failed: {exc}") from exc
    return runtime


__all__ = [
    "PolicyLoadError",
    "RslRlPolicyRuntime",
    "RuntimePolicySample",
    "load_rsl_rl_policy",
]
