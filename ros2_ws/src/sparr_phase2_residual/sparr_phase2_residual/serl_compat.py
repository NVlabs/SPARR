# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA OneWay Noncommercial License (March 22, 2022).
# See the LICENSE file in the repository root for the full license text:
# https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf

"""Compatibility shims for running vendored ``serl_launcher`` (JAX 0.4.x) on newer JAX.

Kept here, narrowly scoped, so the vendored tree stays unmodified.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

# ``resnet10_params.pkl`` carries the removed ``ShapedArray.named_shape`` field,
# which newer JAX rejects. It is empty for ordinary arrays, so dropping it is exact.
_DROPPED_AVAL_KWARGS = ("named_shape",)


@contextlib.contextmanager
def resnet10_pickle_compat() -> Iterator[None]:
    """Allow JAX 0.4-era pickles to load, only for the duration of the block."""
    from jax._src import core

    original = core.ShapedArray.update

    def update(self, **kwargs: Any):
        dropped = {k: kwargs.pop(k) for k in _DROPPED_AVAL_KWARGS if k in kwargs}
        for name, value in dropped.items():
            # Only an *empty* named_shape is safely droppable.
            if value:
                raise ValueError(
                    f"checkpoint array carries a non-empty {name}={value!r}; "
                    "these weights cannot be loaded on this JAX version"
                )
        return original(self, **kwargs)

    core.ShapedArray.update = update
    try:
        yield
    finally:
        core.ShapedArray.update = original


# serl_launcher calls ``batch.copy(add_or_replace=...)``, which needs ``FrozenDict``.


def as_serl_batch(batch: Any) -> Any:
    """Deep-convert a batch of plain dicts into the ``FrozenDict`` serl expects."""
    from flax.core import FrozenDict

    def freeze(value: Any) -> Any:
        if isinstance(value, dict):
            return FrozenDict({k: freeze(v) for k, v in value.items()})
        return value

    if not isinstance(batch, dict):
        raise TypeError(f"expected a batch dict, got {type(batch).__name__}")
    for required in ("observations", "next_observations", "actions"):
        if required not in batch:
            raise KeyError(f"batch is missing {required!r}")
    missing = [k for k in batch["observations"] if k not in batch["next_observations"]]
    if missing:
        # serl's ``_unpack`` treats keys absent from next_observations as a frame stack.
        raise ValueError(
            f"observation keys {sorted(missing)} are absent from next_observations; "
            "serl would reinterpret them as a frame stack to split"
        )
    return freeze(batch)
