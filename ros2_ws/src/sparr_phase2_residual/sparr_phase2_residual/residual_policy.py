"""The Phase 2 residual actor: a 6-D pose delta evaluated at the control rate.

Wraps the ``serl_launcher`` SAC actor for once-per-tick calls from the
collector. ``evaluate`` never raises: failures are reported through
:class:`ResidualDecision` so the caller can fall back to the base policy.
Parameter swaps happen under a lock.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .observation_spec import ResidualObservationSpec, assemble_state

ACTION_DIM = 6


class ResidualPolicyError(RuntimeError):
    """Raised for configuration faults that must stop startup, not a tick."""


@dataclass(frozen=True)
class ResidualDecision:
    """One residual evaluation, including the failure case.

    ``residual`` is ``None`` when it could not be produced. Do not substitute
    zeros: a zero residual is a legitimate output.
    """

    residual: Optional[np.ndarray]
    latency_ns: int
    params_version: int
    deterministic: bool
    error: str = ""
    #: The actor's output before ``action_scale``; used to measure saturation.
    raw_residual: Optional[np.ndarray] = None

    @property
    def ok(self) -> bool:
        return self.residual is not None


class ResidualOutputStats:
    """Running summary of one episode's residual outputs.

    Feed it ``ResidualDecision.raw_residual`` (unscaled), or saturation reads
    zero; :meth:`summary` derives the scaled figure.
    """

    #: |output| at or above which a tanh-squashed dimension counts as saturated.
    SATURATION_THRESHOLD = 0.99

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._sum = np.zeros(ACTION_DIM, np.float64)
        self._abs_sum = np.zeros(ACTION_DIM, np.float64)
        self._max_abs = 0.0
        self._saturated = 0
        self._steps = 0

    @property
    def steps(self) -> int:
        return self._steps

    def observe(self, residual: np.ndarray) -> None:
        """Record one residual. Non-finite values are ignored, not propagated."""
        if residual is None:
            return
        values = np.asarray(residual, np.float64).reshape(-1)
        if values.size != ACTION_DIM or not np.all(np.isfinite(values)):
            return
        magnitude = np.abs(values)
        self._sum += values
        self._abs_sum += magnitude
        self._max_abs = max(self._max_abs, float(magnitude.max()))
        self._saturated += int(np.count_nonzero(magnitude >= self.SATURATION_THRESHOLD))
        self._steps += 1

    def summary(self, *, action_scale: float) -> dict[str, Any]:
        """Episode aggregates, empty when no residual was produced at all."""
        if self._steps == 0:
            return {}
        abs_mean_axes = self._abs_sum / self._steps
        abs_mean = float(abs_mean_axes.mean())
        return {
            "residual_steps": self._steps,
            "residual_abs_mean": abs_mean,
            "residual_abs_mean_scaled": abs_mean * float(action_scale),
            "residual_mean_axes": (self._sum / self._steps).tolist(),
            "residual_abs_mean_axes": abs_mean_axes.tolist(),
            "residual_max_abs": self._max_abs,
            "residual_saturation": self._saturated / (self._steps * ACTION_DIM),
        }


class ResidualPolicy:
    """Evaluates the residual actor, or reports why it could not."""

    def __init__(
        self,
        agent: Any,
        spec: ResidualObservationSpec,
        *,
        seed: int = 0,
        deterministic: bool = False,
        action_scale: float = 1.0,
    ) -> None:
        if agent is None:
            raise ResidualPolicyError("a residual agent is required")
        if not np.isfinite(action_scale) or not 0.0 < action_scale <= 1.0:
            # Above 1.0 the residual alone could saturate the normalized action.
            raise ResidualPolicyError(
                f"residual action_scale must be in (0, 1], got {action_scale}"
            )
        self._agent = agent
        self._spec = spec
        self._deterministic = bool(deterministic)
        self._action_scale = float(action_scale)
        self._lock = threading.Lock()
        self._params_version = 0
        self._rng_seed = int(seed)
        self._step = 0

    @property
    def params_version(self) -> int:
        with self._lock:
            return self._params_version

    def update_params(self, params: Any) -> int:
        """Swap in learner-published actor parameters between ticks."""
        with self._lock:
            self._agent = self._agent.replace(
                state=self._agent.state.replace(params=params)
            )
            self._params_version += 1
            return self._params_version

    def evaluate(
        self,
        *,
        terms: Mapping[str, Sequence[float] | np.ndarray],
        images: Mapping[str, np.ndarray],
    ) -> ResidualDecision:
        """Produce one 6-D residual. Never raises."""
        started = time.monotonic_ns()
        with self._lock:
            agent = self._agent
            version = self._params_version
        try:
            observation = self._observation(terms, images)
            action = self._sample(agent, observation)
            raw = self._validate(action)
            residual = raw * self._action_scale
        except Exception as exc:  # noqa: BLE001 - a bad tick must not kill the loop
            return ResidualDecision(
                residual=None,
                latency_ns=time.monotonic_ns() - started,
                params_version=version,
                deterministic=self._deterministic,
                error=f"{type(exc).__name__}: {exc}",
            )
        return ResidualDecision(
            residual=residual,
            raw_residual=raw,
            latency_ns=time.monotonic_ns() - started,
            params_version=version,
            deterministic=self._deterministic,
        )

    def _observation(
        self,
        terms: Mapping[str, Sequence[float] | np.ndarray],
        images: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        observation: dict[str, np.ndarray] = {
            "state": assemble_state(self._spec, terms)
        }
        for key in self._spec.image_keys:
            if key not in images:
                raise ResidualPolicyError(f"missing residual image {key!r}")
            frame = np.asarray(images[key])
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                raise ResidualPolicyError(
                    f"residual image {key!r} must be uint8 HxWx3, got "
                    f"{frame.dtype} {frame.shape}"
                )
            observation[key] = frame[None, ...]
        return observation

    def _sample(self, agent: Any, observation: Mapping[str, np.ndarray]) -> Any:
        import jax  # imported lazily so the collector need not depend on JAX

        # A fresh key per step, or exploration noise repeats every tick.
        self._step += 1
        key = jax.random.PRNGKey(self._rng_seed + self._step)
        return agent.sample_actions(
            observations=jax.device_put(observation),
            seed=key,
            argmax=self._deterministic,
        )

    @staticmethod
    def _validate(action: Any) -> np.ndarray:
        value = np.asarray(action, dtype=np.float64).reshape(-1)
        if value.shape != (ACTION_DIM,):
            raise ResidualPolicyError(
                f"residual must have shape ({ACTION_DIM},), got {value.shape}"
            )
        if not np.all(np.isfinite(value)):
            raise ResidualPolicyError("residual contains non-finite values")
        # The actor is tanh-squashed; anything outside [-1, 1] means wrong params.
        if np.any(np.abs(value) > 1.0 + 1e-6):
            raise ResidualPolicyError(
                f"residual outside the tanh range: max|a|={np.abs(value).max():.4f}"
            )
        value.setflags(write=False)
        return value
