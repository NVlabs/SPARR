# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA OneWay Noncommercial License (March 22, 2022).
# See the LICENSE file in the repository root for the full license text:
# https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf

"""Replay buffers for the residual learner, and the RLPD demo/online split.

Builds ``serl_launcher`` buffers from a :class:`ResidualObservationSpec`, loads
demonstration pickles, and mixes demo and online batches. Does not import ROS.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

import numpy as np

from .observation_spec import ResidualObservationSpec

DEFAULT_IMAGE_SIZE = 256
ACTION_DIM = 6


class BufferError(RuntimeError):
    """Raised when demonstrations or buffers cannot be prepared."""


def residual_spaces(
    spec: ResidualObservationSpec, *, image_size: int = DEFAULT_IMAGE_SIZE
) -> tuple[Any, Any]:
    """The ``gymnasium`` observation and action spaces for one transition.

    The leading ``1`` is serl's frame-stack axis, not a batch axis:
    ``MemoryEfficientReplayBuffer`` reads the stack depth from ``shape[0]``, so
    it must match the ``(1, ...)`` rows the demo converter writes.
    """
    from gymnasium.spaces import Box, Dict as DictSpace

    entries: dict[str, Any] = {
        "state": Box(
            low=-np.inf,
            high=np.inf,
            shape=(1, spec.state_dim()),
            dtype=np.float32,
        )
    }
    for key in spec.image_keys:
        entries[key] = Box(
            low=0, high=255, shape=(1, image_size, image_size, 3), dtype=np.uint8
        )
    observation_space = DictSpace(entries)
    action_space = Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
    return observation_space, action_space


def make_data_store(
    spec: ResidualObservationSpec,
    capacity: int,
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    seed: Optional[int] = None,
) -> Any:
    """A thread-safe, image-deduplicating replay buffer for residual data.

    Transitions must be inserted in episode order with ``dones`` marking the
    boundaries, since only one frame is stored per transition.
    """
    if capacity < 1:
        raise BufferError(f"buffer capacity must be positive, got {capacity}")
    from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

    observation_space, action_space = residual_spaces(spec, image_size=image_size)
    store = MemoryEfficientReplayBufferDataStore(
        observation_space,
        action_space,
        capacity=capacity,
        image_keys=tuple(spec.image_keys),
    )
    if seed is not None:
        store.seed(seed)
    return store


def load_demo_transitions(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Read the transition lists written by ``sparr_phase2_make_demos``."""
    transitions: list[dict[str, Any]] = []
    for path in paths:
        source = Path(path)
        if not source.is_file():
            raise BufferError(f"no demonstration file at {source}")
        with source.open("rb") as handle:
            loaded = pickle.load(handle)
        if not isinstance(loaded, list) or not loaded:
            raise BufferError(f"{source} does not contain a non-empty transition list")
        transitions.extend(loaded)
    return transitions


def validate_transition(
    transition: Any, spec: ResidualObservationSpec, *, where: str = "transition"
) -> None:
    """Reject a transition whose shapes or dtypes would silently corrupt the buffer."""
    required = {
        "observations",
        "next_observations",
        "actions",
        "rewards",
        "masks",
        "dones",
    }
    missing = required - set(transition)
    if missing:
        raise BufferError(f"{where} is missing {sorted(missing)}")

    for field in ("observations", "next_observations"):
        observation = transition[field]
        state = np.asarray(observation["state"])
        if state.shape != (1, spec.state_dim()):
            raise BufferError(
                f"{where}.{field}.state has shape {state.shape}, "
                f"expected (1, {spec.state_dim()})"
            )
        if state.dtype != np.float32:
            raise BufferError(f"{where}.{field}.state must be float32, got {state.dtype}")
        for key in spec.image_keys:
            if key not in observation:
                raise BufferError(f"{where}.{field} is missing image {key!r}")
            image = np.asarray(observation[key])
            if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] != 3:
                raise BufferError(
                    f"{where}.{field}.{key} has shape {image.shape}, expected (1,H,W,3)"
                )
            if image.dtype != np.uint8:
                raise BufferError(
                    f"{where}.{field}.{key} must be uint8, got {image.dtype}"
                )

    action = np.asarray(transition["actions"])
    if action.shape != (ACTION_DIM,):
        raise BufferError(f"{where}.actions has shape {action.shape}, expected (6,)")
    if not np.all(np.isfinite(action)):
        raise BufferError(f"{where}.actions is not finite")


def populate(
    store: Any,
    transitions: Iterable[dict[str, Any]],
    spec: ResidualObservationSpec,
    *,
    validate: bool = True,
) -> int:
    """Insert transitions in order, validating the first of each episode."""
    inserted = 0
    check_next = True
    for index, transition in enumerate(transitions):
        if validate and check_next:
            validate_transition(transition, spec, where=f"transition[{index}]")
            check_next = False
        store.insert(transition)
        inserted += 1
        if bool(transition.get("dones", False)):
            check_next = True
    return inserted


@dataclass
class RLPDSampler:
    """Draws every training batch from demonstrations and online data (RLPD).

    The demo fraction is exact per batch, not in expectation.
    """

    demo_store: Any
    online_store: Any
    batch_size: int = 256
    demo_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.batch_size < 2:
            raise BufferError("batch_size must be at least 2 to split")
        if not 0.0 < self.demo_fraction < 1.0:
            raise BufferError(
                f"demo_fraction must be strictly between 0 and 1, "
                f"got {self.demo_fraction}"
            )
        self.demo_batch = int(round(self.batch_size * self.demo_fraction))
        self.online_batch = self.batch_size - self.demo_batch
        if self.demo_batch < 1 or self.online_batch < 1:
            raise BufferError(
                f"batch_size {self.batch_size} at demo_fraction "
                f"{self.demo_fraction} leaves one side empty"
            )

    @property
    def ready(self) -> bool:
        """Whether both buffers can supply their half of a batch."""
        return (
            len(self.demo_store) >= self.demo_batch
            and len(self.online_store) >= self.online_batch
        )

    def sample(self) -> Any:
        """One combined batch, as the FrozenDict serl_launcher expects."""
        if not self.ready:
            raise BufferError(
                f"not enough data: demos={len(self.demo_store)}/{self.demo_batch}, "
                f"online={len(self.online_store)}/{self.online_batch}"
            )
        from serl_launcher.utils.train_utils import concat_batches

        demo_batch = self.demo_store.sample(self.demo_batch)
        online_batch = self.online_store.sample(self.online_batch)
        # Online first, then demos: augmentation and action-noise RNGs are
        # indexed by batch position, so the order changes every loss.
        # axis=0 is the batch axis; serl's default axis=1 gives a wrong shape.
        return concat_batches(online_batch, demo_batch, axis=0)

    def iterator(self, *, queue_size: int = 2, device: Any = None) -> Iterator[Any]:
        """Prefetching :meth:`sample`, keeping ``queue_size`` batches on ``device``."""
        from serl_launcher.utils.train_utils import concat_batches

        demo = self.demo_store.get_iterator(
            queue_size=queue_size,
            sample_args={"batch_size": self.demo_batch},
            device=device,
        )
        online = self.online_store.get_iterator(
            queue_size=queue_size,
            sample_args={"batch_size": self.online_batch},
            device=device,
        )
        while True:
            # Same online-then-demo order as sample().
            yield concat_batches(next(online), next(demo), axis=0)

    def describe(self) -> str:
        return (
            f"batch {self.batch_size} = {self.demo_batch} demo "
            f"+ {self.online_batch} online "
            f"(demos={len(self.demo_store)}, online={len(self.online_store)})"
        )
