# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA OneWay Noncommercial License (March 22, 2022).
# See the LICENSE file in the repository root for the full license text:
# https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf

"""Phase 2 configuration, layered on top of an unchanged Phase 1 config.

Base policy and robot settings live in the Phase 1 file; residual settings here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional

from .observation_spec import DEFAULT_IMAGE_KEYS, ResidualObservationSpec

#: Small so early exploration cannot overpower the base policy: a saturated
#: residual moves the normalized action by at most this fraction of its range.
DEFAULT_ACTION_SCALE = 0.1


#: The training-time Phase 2 config, written beside the learner's checkpoints.
#: Evaluation reads the observation layout and action scale from it.
TRAINING_CONFIG_FILENAME = "phase2_config.yaml"


class Phase2ConfigError(ValueError):
    """Raised when a Phase 2 configuration file cannot be used as written."""


@dataclass(frozen=True)
class ResidualRuntimeConfig:
    """How the residual behaves inside the live control loop."""

    #: ``shadow``  - compute and record the residual, command the base policy
    #:               alone (safe default).
    #: ``active``  - compose the residual into the commanded action.
    mode: str = "shadow"
    #: Scales the residual before composition. See DEFAULT_ACTION_SCALE.
    action_scale: float = DEFAULT_ACTION_SCALE
    #: Use the mean action instead of sampling.
    deterministic: bool = False
    seed: int = 0
    #: Actor parameters published by the learner. Absent means a fresh actor.
    params_path: Optional[str] = None
    #: How often the runtime re-reads params_path, in control ticks. 0 disables.
    params_poll_ticks: int = 0
    #: Where the learner is. Unset means no learner: episodes are recorded to
    #: HDF5 but not sent anywhere.
    learner_host: Optional[str] = None
    learner_port: int = 5588
    learner_broadcast_port: int = 5589
    #: Block at startup until the learner answers. When False, a missing
    #: learner degrades to recording-only instead of hanging.
    wait_for_learner: bool = False
    learner_timeout_ms: int = 3000
    #: Transitions the actor will hold if the learner is slow or absent.
    actor_queue_capacity: int = 50_000
    #: Promote successful episodes no longer than the running median of
    #: demonstration lengths, seeded from ``promotion_seed_lengths``.
    promote_episodes: bool = True
    promotion_seed_lengths: tuple[int, ...] = ()
    #: Consecutive residual failures tolerated before the trial is aborted;
    #: each one falls back to the base policy for that tick.
    max_consecutive_failures: int = 30

    def __post_init__(self) -> None:
        if self.mode not in {"shadow", "active"}:
            raise Phase2ConfigError(
                f"residual mode must be 'shadow' or 'active', got {self.mode!r}"
            )
        if not 0.0 < float(self.action_scale) <= 1.0:
            raise Phase2ConfigError(
                f"residual action_scale must be in (0, 1], got {self.action_scale}"
            )
        if int(self.params_poll_ticks) < 0:
            raise Phase2ConfigError("params_poll_ticks must not be negative")
        if int(self.max_consecutive_failures) < 1:
            raise Phase2ConfigError("max_consecutive_failures must be at least 1")
        for name in ("learner_port", "learner_broadcast_port"):
            port = int(getattr(self, name))
            if not 1 <= port <= 65535:
                raise Phase2ConfigError(f"{name} must be a valid port, got {port}")
        if self.learner_port == self.learner_broadcast_port:
            raise Phase2ConfigError(
                "learner_port and learner_broadcast_port must differ"
            )
        if int(self.actor_queue_capacity) < 1:
            raise Phase2ConfigError("actor_queue_capacity must be positive")
        for length in self.promotion_seed_lengths:
            if int(length) < 1:
                raise Phase2ConfigError(
                    f"promotion_seed_lengths must all be >= 1, got {length}"
                )

    @property
    def commands_residual(self) -> bool:
        """Whether the residual actually reaches the robot."""
        return self.mode == "active"


@dataclass(frozen=True)
class ResidualRetryConfig:
    """Replaying failed attempts once the windowed success rate is high enough."""

    #: Off by default: replays cost robot time.
    enabled: bool = False
    #: Windowed success rate at or above which replaying begins.
    success_rate_threshold: float = 0.8
    #: Attempts the rate is measured over, and the minimum sample size.
    window: int = 20
    #: Replays of one attempt before giving up and moving on.
    max_repeats: int = 5
    #: Promote the success that ends a replay sequence into the demo buffer,
    #: bypassing the running-median length criterion.
    promote_on_success: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.success_rate_threshold) <= 1.0:
            raise Phase2ConfigError(
                "retry.success_rate_threshold must be in [0, 1], got "
                f"{self.success_rate_threshold}"
            )
        if int(self.window) < 1:
            raise Phase2ConfigError(f"retry.window must be >= 1, got {self.window}")
        if int(self.max_repeats) < 1:
            raise Phase2ConfigError(
                f"retry.max_repeats must be >= 1, got {self.max_repeats}"
            )


@dataclass(frozen=True)
class Phase2Config:
    observation: ResidualObservationSpec = field(
        default_factory=ResidualObservationSpec
    )
    runtime: ResidualRuntimeConfig = field(default_factory=ResidualRuntimeConfig)
    retry: ResidualRetryConfig = field(default_factory=ResidualRetryConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Phase2Config":
        residual = dict(data.get("residual") or {})
        unknown = set(residual) - {"image_keys", "state_terms", "runtime", "retry"}
        if unknown:
            raise Phase2ConfigError(
                f"unknown key(s) under 'residual': {sorted(unknown)}"
            )

        terms = dict(residual.get("state_terms") or {})
        allowed = {"flange_pose", "wrench", "base_mean", "socket_pose"}
        unknown_terms = set(terms) - allowed
        if unknown_terms:
            raise Phase2ConfigError(
                f"unknown state term(s): {sorted(unknown_terms)}; "
                f"expected any of {sorted(allowed)}"
            )
        image_keys = residual.get("image_keys") or list(DEFAULT_IMAGE_KEYS)
        if isinstance(image_keys, str):
            raise Phase2ConfigError("residual.image_keys must be a list, not a string")

        spec = ResidualObservationSpec(image_keys=tuple(image_keys), **terms)

        runtime = dict(residual.get("runtime") or {})
        if "promotion_seed_lengths" in runtime:
            runtime["promotion_seed_lengths"] = tuple(
                runtime["promotion_seed_lengths"] or ()
            )
        unknown_runtime = set(runtime) - set(
            ResidualRuntimeConfig.__dataclass_fields__
        )
        if unknown_runtime:
            raise Phase2ConfigError(
                f"unknown key(s) under 'residual.runtime': {sorted(unknown_runtime)}"
            )
        retry = dict(residual.get("retry") or {})
        unknown_retry = set(retry) - set(ResidualRetryConfig.__dataclass_fields__)
        if unknown_retry:
            raise Phase2ConfigError(
                f"unknown key(s) under 'residual.retry': {sorted(unknown_retry)}"
            )
        return cls(
            observation=spec,
            runtime=ResidualRuntimeConfig(**runtime),
            retry=ResidualRetryConfig(**retry),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Phase2Config":
        import yaml

        source = Path(path)
        if not source.is_file():
            raise Phase2ConfigError(f"no Phase 2 configuration at {source}")
        with source.open() as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, Mapping):
            raise Phase2ConfigError(f"{source} must contain a mapping at the top level")
        try:
            return cls.from_mapping(data)
        except ValueError as exc:
            raise Phase2ConfigError(f"{source}: {exc}") from exc


    def describe(self) -> str:
        return (
            f"residual: mode={self.runtime.mode} "
            f"scale={self.runtime.action_scale} "
            f"{'deterministic' if self.runtime.deterministic else 'stochastic'}\n"
            f"  state ({self.observation.state_dim()}-D): {self.observation.describe()}\n"
            f"  images: {', '.join(self.observation.image_keys)}"
        )


def evaluation_config(
    base: Phase2Config, params_path: str | Path, *, stochastic: bool = False
) -> tuple[Phase2Config, Optional[Path]]:
    """``base`` turned into a fixed-checkpoint evaluation of ``params_path``.

    The residual is active, its weights come only from the file (no learner, no
    polling), it uses the mean unless ``stochastic``, and nothing is promoted
    or replayed. Observation layout and ``action_scale`` come from the
    ``phase2_config.yaml`` beside the params file when present.

    Returns ``(config, sidecar)``; ``sidecar`` is None if that file is absent.
    """
    params = Path(params_path).expanduser()
    if not params.is_file():
        raise Phase2ConfigError(f"no actor params at {params}")
    sidecar = params.parent / TRAINING_CONFIG_FILENAME
    trained = Phase2Config.load(sidecar) if sidecar.is_file() else None
    source = trained or base
    runtime = replace(
        base.runtime,
        mode="active",
        params_path=str(params),
        params_poll_ticks=0,
        deterministic=not stochastic,
        action_scale=source.runtime.action_scale,
        learner_host=None,
        wait_for_learner=False,
        promote_episodes=False,
    )
    config = replace(
        base,
        observation=source.observation,
        runtime=runtime,
        retry=replace(base.retry, enabled=False),
    )
    return config, (sidecar if trained is not None else None)

