# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA License.
# See the LICENSE file in the repository root for the full license text.

"""The residual learner: RLPD updates, demo promotion, checkpoint/resume.

Needs neither ROS nor the robot; observation shapes come from
:func:`buffers.residual_spaces`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .buffers import (
    RLPDSampler,
    load_demo_transitions,
    make_data_store,
    populate,
)
from .observation_spec import ResidualObservationSpec
from .promotion import DemoPromoter, episode_lengths

#: Networks updated on a critic-only and a full step. Frozensets because serl's
#: ``networks_to_update`` is a jit static argument and must be hashable.
CRITIC_ONLY = frozenset({"critic"})
ALL_NETWORKS = frozenset({"critic", "actor", "temperature"})


#: Pickled actor parameters written beside each checkpoint, for ``params_path``.
ACTOR_PARAMS_FILENAME = "actor_params.pkl"


class LearnerError(RuntimeError):
    """Raised when the learner cannot start or continue safely."""


@dataclass(frozen=True)
class LearnerConfig:
    """Training hyperparameters."""

    batch_size: int = 256
    #: Critic-to-actor ratio. Per step: ``cta_ratio - 1`` critic-only updates
    #: then one full update, so 2 means two gradient steps per iteration.
    cta_ratio: int = 2
    #: Online transitions required before the first gradient step. The demo
    #: buffer is filled up front, so this gates only online data.
    training_starts: int = 100
    #: Learner steps between parameter publications to the actor.
    steps_per_update: int = 50
    log_period: int = 10
    checkpoint_period: int = 5000
    max_steps: int = 1_000_000
    replay_buffer_capacity: int = 200_000
    discount: float = 0.97
    reward_bias: float = 0.0
    #: None resolves to -action_dim/2 = -3.0 inside serl.
    target_entropy: Optional[float] = None
    demo_fraction: float = 0.5
    seed: int = 42
    #: Square encoder input; must match the recorded image resolution.
    image_size: int = 256
    #: Cap on promoted episodes, bounding dilution of operator demonstrations.
    max_promotions: Optional[int] = None

    def __post_init__(self) -> None:
        if self.cta_ratio < 1:
            raise LearnerError(f"cta_ratio must be at least 1, got {self.cta_ratio}")
        if self.batch_size < 2:
            raise LearnerError("batch_size must be at least 2")
        if self.max_steps < 1:
            raise LearnerError("max_steps must be positive")
        if self.training_starts < 1:
            raise LearnerError("training_starts must be positive")


@dataclass
class LearnerState:
    """Everything a resumed run needs that is not in the agent checkpoint."""

    step: int = 0
    promoted: int = 0
    promotion_lengths: list[int] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "step": self.step,
                "promoted": self.promoted,
                "promotion_lengths": self.promotion_lengths,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> "LearnerState":
        data = json.loads(text)
        return cls(
            step=int(data.get("step", 0)),
            promoted=int(data.get("promoted", 0)),
            promotion_lengths=list(data.get("promotion_lengths", [])),
        )


class ResidualLearner:
    """Owns the buffers, the agent, and the RLPD update loop."""

    def __init__(
        self,
        spec: ResidualObservationSpec,
        config: Optional[LearnerConfig] = None,
        *,
        agent: Any = None,
        checkpoint_dir: Optional[str | Path] = None,
    ) -> None:
        self.spec = spec
        self.config = config or LearnerConfig()
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.state = LearnerState()

        self.demo_store = make_data_store(
            spec,
            self.config.replay_buffer_capacity,
            image_size=self.config.image_size,
            seed=self.config.seed,
        )
        self.online_store = make_data_store(
            spec,
            self.config.replay_buffer_capacity,
            image_size=self.config.image_size,
            seed=self.config.seed + 1,
        )
        self.sampler = RLPDSampler(
            self.demo_store,
            self.online_store,
            batch_size=self.config.batch_size,
            demo_fraction=self.config.demo_fraction,
        )
        self.promoter = DemoPromoter(max_promotions=self.config.max_promotions)
        self.agent = agent if agent is not None else self._build_agent()
        #: Called with the step after each checkpoint is saved.
        self.on_checkpoint: Optional[Callable[[int], None]] = None
        self._metrics: dict[str, float] = {}
        self._batches: Any = None

    def _build_agent(self) -> Any:
        from .residual_agent import make_residual_agent

        return make_residual_agent(
            self.spec,
            seed=self.config.seed,
            image_size=self.config.image_size,
            discount=self.config.discount,
            reward_bias=self.config.reward_bias,
            target_entropy=self.config.target_entropy,
        )

    def load_demonstrations(self, paths: Sequence[str | Path]) -> int:
        """Fill the demonstration buffer and seed the promotion median."""
        transitions = load_demo_transitions(paths)
        inserted = populate(self.demo_store, transitions, self.spec)
        lengths = episode_lengths(transitions)
        if not lengths:
            raise LearnerError(
                "demonstrations contain no complete episode (no terminal "
                "transition); the promotion median cannot be seeded"
            )
        self.promoter.lengths.extend(lengths)
        return inserted


    @property
    def ready(self) -> bool:
        return (
            len(self.online_store) >= self.config.training_starts and self.sampler.ready
        )

    def add_replayed_episode(self, episode: Any, *, promoted: bool) -> None:
        """Insert one recorded episode during offline replay.

        ``promoted`` is the actor's recorded decision; it is not recomputed here.
        """
        from .offline_replay import Episode

        if not isinstance(episode, Episode):
            raise LearnerError("add_replayed_episode expects an offline_replay.Episode")
        if not episode.transitions:
            raise LearnerError("cannot add an empty episode")
        populate(self.online_store, episode.transitions, self.spec)
        if promoted:
            populate(self.demo_store, episode.transitions, self.spec, validate=False)
            self.state.promoted += 1
            self.promoter.lengths.append(len(episode.transitions))

    def waiting_on(self) -> str:
        """Why :attr:`ready` is False, in the caller's words."""
        if len(self.online_store) < self.config.training_starts:
            return (f"online transitions {len(self.online_store)}/"
                    f"{self.config.training_starts}")
        return self.sampler.describe()

    def wait_until_ready(
        self,
        *,
        poll_s: float = 1.0,
        timeout_s: Optional[float] = None,
        on_wait: Optional[Callable[[str], None]] = None,
        report_every_s: float = 30.0,
    ):
        """Block until enough online data has arrived to start training.

        ``on_wait`` is called every ``report_every_s`` with :meth:`waiting_on`.
        """
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        next_report = time.monotonic() + report_every_s
        while not self.ready:
            if deadline is not None and time.monotonic() > deadline:
                raise LearnerError(f"timed out waiting for data: {self.waiting_on()}")
            now = time.monotonic()
            if on_wait is not None and now >= next_report:
                on_wait(self.waiting_on())
                next_report = now + report_every_s
            time.sleep(poll_s)

    def train_step(self, batches: Any) -> dict[str, float]:
        """One learner iteration: ``cta_ratio - 1`` critic updates, then a full one."""
        for _ in range(self.config.cta_ratio - 1):
            self.agent, _ = self.agent.update(
                next(batches), networks_to_update=CRITIC_ONLY
            )
        self.agent, info = self.agent.update(
            next(batches), networks_to_update=ALL_NETWORKS
        )
        self.state.step += 1
        return info

    def run(
        self,
        *,
        steps: Optional[int] = None,
        on_publish: Optional[Callable[[Any], None]] = None,
        on_log: Optional[Callable[[int, dict[str, float]], None]] = None,
        device: Any = None,
        release: Optional[Callable[[int], int]] = None,
    ) -> int:
        """The training loop. Returns the step reached.

        ``release`` is the offline-replay hook, called with the current step
        before each iteration to admit episodes due by then.
        """
        import jax

        if not self.sampler.ready:
            raise LearnerError(f"cannot start: {self.sampler.describe()}")

        target = self.config.max_steps if steps is None else self.state.step + steps

        # Reuse the prefetching iterator across calls so split runs sample the
        # same batches as one long run.
        if self._batches is None:
            self._batches = self.sampler.iterator(device=device)
        batches = self._batches

        if on_publish is not None:
            # Publish the starting parameters before the first update.
            on_publish(jax.device_get(self.agent.state.params))

        while self.state.step < target:
            if release is not None:
                release(self.state.step)
            info = self.train_step(batches)

            if on_publish and self.state.step % self.config.steps_per_update == 0:
                self.agent = jax.block_until_ready(self.agent)
                on_publish(jax.device_get(self.agent.state.params))

            if self.state.step % self.config.log_period == 0:
                self._metrics = scalar_metrics(jax.device_get(info))
                if on_log:
                    on_log(self.state.step, self._metrics)

            if (
                self.checkpoint_dir
                and self.config.checkpoint_period
                and self.state.step % self.config.checkpoint_period == 0
            ):
                self.save_checkpoint()

        return self.state.step

    @property
    def metrics(self) -> dict[str, float]:
        return dict(self._metrics)

    def save_checkpoint(self, directory: Optional[str | Path] = None) -> Path:
        """Persist agent state, plus step and promotion state in ``learner_state.json``."""
        import jax
        from flax.training import checkpoints

        target = Path(directory) if directory else self.checkpoint_dir
        if target is None:
            raise LearnerError("no checkpoint directory configured")
        target.mkdir(parents=True, exist_ok=True)

        checkpoints.save_checkpoint(
            str(target.absolute()),
            jax.device_get(self.agent.state),
            step=self.state.step,
            keep=100,
            overwrite=True,
        )
        self.state.promotion_lengths = list(self.promoter.lengths)
        (target / "learner_state.json").write_text(self.state.to_json())
        self.save_actor_params(target / ACTOR_PARAMS_FILENAME)
        if self.on_checkpoint is not None:
            self.on_checkpoint(self.state.step)
        return target

    def save_actor_params(self, path: Optional[str | Path] = None) -> Path:
        """Write the single actor-params pickle the robot loads via ``params_path``.

        Written atomically, so a reader never sees a truncated file.
        """
        from .residual_agent import save_actor_params

        if path is None:
            if self.checkpoint_dir is None:
                raise LearnerError("no checkpoint directory configured")
            path = Path(self.checkpoint_dir) / ACTOR_PARAMS_FILENAME
        return save_actor_params(self.agent, path)

    def restore_checkpoint(
        self,
        directory: Optional[str | Path] = None,
        *,
        step: Optional[int] = None,
    ) -> int:
        """Restore agent and learner state. Returns the step resumed from.

        ``step`` selects one checkpoint; the default takes the newest.
        """
        from flax.training import checkpoints

        target = Path(directory) if directory else self.checkpoint_dir
        if target is None:
            raise LearnerError("no checkpoint directory configured")
        if not target.is_dir():
            raise LearnerError(f"no checkpoint directory at {target}")

        restored = checkpoints.restore_checkpoint(
            str(target.absolute()), self.agent.state, step=step
        )
        if restored is None:
            raise LearnerError(
                f"no checkpoint found under {target}"
                + (f" for step {step}" if step is not None else "")
            )
        self.agent = self.agent.replace(state=restored)

        sidecar = target / "learner_state.json"
        if sidecar.is_file() and step is None:
            # The sidecar describes the newest checkpoint only.
            self.state = LearnerState.from_json(sidecar.read_text())
            self.promoter = DemoPromoter(
                lengths=list(self.state.promotion_lengths),
                max_promotions=self.config.max_promotions,
                promoted_count=self.state.promoted,
            )
        return self.state.step

    def describe(self) -> str:
        return (
            f"learner step {self.state.step}/{self.config.max_steps} | "
            f"{self.sampler.describe()} | {self.promoter.describe()}"
        )


def scalar_metrics(info: Any, prefix: str = "") -> dict[str, float]:
    """Flatten serl's nested metric tree to scalars, skipping arrays."""
    out: dict[str, float] = {}
    for key, value in info.items():
        if hasattr(value, "items"):
            out.update(scalar_metrics(value, f"{prefix}{key}/"))
        elif np.ndim(value) == 0:
            out[prefix + key] = float(value)
    return out
