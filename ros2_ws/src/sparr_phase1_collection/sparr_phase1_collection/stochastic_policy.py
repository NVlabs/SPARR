# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA OneWay Noncommercial License (March 22, 2022).
# See the LICENSE file in the repository root for the full license text:
# https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf

"""One-pass stochastic adapter for recurrent ``rsl_rl`` actor policies.

``model.act(...)`` runs exactly once per control step, advancing recurrent memory
and sampling the distribution; mean and std are read from that distribution, since
``act_inference`` would advance the LSTM a second time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class StochasticPolicyError(RuntimeError):
    """Raised when a policy cannot expose a one-pass stochastic trace."""


@dataclass(frozen=True)
class PolicySample:
    """Unclipped distribution statistics and the sample from one actor step."""

    raw_mean: Any
    raw_std: Any
    raw_sample: Any


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - deployment image supplies Torch
        raise StochasticPolicyError(
            "Torch is required to execute the stochastic base policy"
        ) from exc
    return torch


def _read_distribution_stats(model: Any) -> tuple[Any, Any]:
    """Read statistics without performing another actor forward."""

    mean = getattr(model, "action_mean", None)
    std = getattr(model, "action_std", None)

    distribution = getattr(model, "distribution", None)
    if mean is None and distribution is not None:
        mean = getattr(distribution, "mean", None)
    if std is None and distribution is not None:
        std = getattr(distribution, "stddev", None)
        if std is None:
            std = getattr(distribution, "scale", None)

    if mean is None or std is None:
        raise StochasticPolicyError(
            "model.act() completed, but the model exposes neither "
            "action_mean/action_std nor distribution mean/stddev"
        )
    if callable(mean) or callable(std):
        raise StochasticPolicyError(
            "distribution statistics must be cached tensors, not callable forwards"
        )
    return mean, std


class RecurrentStochasticPolicy:
    """Adapter that exposes a stochastic policy trace with one recurrent advance.

    The wrapped model must be the complete actor/actor-critic object, not its
    bound ``act_inference`` method.  Its ``act`` implementation is expected to
    update the distribution and return the unbounded stochastic action, as in
    ``rsl_rl`` ActorCritic and ActorCriticRecurrent.
    """

    def __init__(self, model: Any):
        act = getattr(model, "act", None)
        if not callable(act):
            raise StochasticPolicyError(
                "stochastic policy adapter requires a model with callable act()"
            )
        self._model = model

    @property
    def model(self) -> Any:
        return self._model

    def sample_once(self, observation: Any, **act_kwargs: Any) -> PolicySample:
        """Advance recurrent state once and return unclipped mean/std/sample."""

        torch = _require_torch()
        with torch.inference_mode():
            raw_sample = self._model.act(observation, **act_kwargs)
            raw_mean, raw_std = _read_distribution_stats(self._model)

            if not isinstance(raw_sample, torch.Tensor):
                raise StochasticPolicyError(
                    f"model.act() must return a Tensor, got {type(raw_sample).__name__}"
                )
            if not isinstance(raw_mean, torch.Tensor) or not isinstance(raw_std, torch.Tensor):
                raise StochasticPolicyError("raw mean and std must be Torch tensors")
            if raw_mean.shape != raw_sample.shape:
                raise StochasticPolicyError(
                    f"mean shape {tuple(raw_mean.shape)} does not match sample "
                    f"shape {tuple(raw_sample.shape)}"
                )
            try:
                expanded_std = torch.broadcast_to(raw_std, raw_sample.shape)
            except RuntimeError as exc:
                raise StochasticPolicyError(
                    f"std shape {tuple(raw_std.shape)} cannot broadcast to sample "
                    f"shape {tuple(raw_sample.shape)}"
                ) from exc
            if not torch.isfinite(raw_mean).all():
                raise StochasticPolicyError("policy produced a non-finite raw mean")
            if not torch.isfinite(expanded_std).all() or not (expanded_std > 0.0).all():
                raise StochasticPolicyError("policy produced a non-finite or non-positive std")
            if not torch.isfinite(raw_sample).all():
                raise StochasticPolicyError("policy produced a non-finite raw sample")

            return PolicySample(
                raw_mean=raw_mean.detach().clone(),
                raw_std=expanded_std.detach().clone(),
                raw_sample=raw_sample.detach().clone(),
            )

    def reset(self, dones: Any | None = None) -> None:
        """Reset recurrent memory at an episode boundary."""

        reset = getattr(self._model, "reset", None)
        if not callable(reset):
            raise StochasticPolicyError("wrapped policy does not expose reset()")
        if dones is None:
            reset()
        else:
            reset(dones)
