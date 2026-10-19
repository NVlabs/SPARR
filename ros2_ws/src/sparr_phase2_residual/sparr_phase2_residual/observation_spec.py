# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""The residual policy's observation layout.

The offline converter and the online runtime both build state vectors here, so
field order is fixed in one place. Socket pose is an optional ablation term.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

DEFAULT_IMAGE_KEYS: tuple[str, ...] = ("wrist_1",)

FLANGE_POSE_WIDTH = 7   # xyz + quaternion xyzw
WRENCH_WIDTH = 6        # force xyz + torque xyz
BASE_MEAN_WIDTH = 6     # the base policy's raw pre-clamp mean
SOCKET_POSE_WIDTH = 7   # xyz + quaternion xyzw


class ResidualObservationError(ValueError):
    """Raised when a residual observation cannot be assembled."""


@dataclass(frozen=True)
class ResidualObservationSpec:
    """Which terms make up the residual's state vector, and in what order."""

    flange_pose: bool = True
    wrench: bool = True
    base_mean: bool = True
    socket_pose: bool = False
    image_keys: tuple[str, ...] = DEFAULT_IMAGE_KEYS

    def __post_init__(self) -> None:
        if not self.image_keys:
            raise ResidualObservationError("at least one camera is required")
        if len(set(self.image_keys)) != len(self.image_keys):
            raise ResidualObservationError(f"duplicate image keys: {self.image_keys}")
        if self.state_dim() == 0:
            raise ResidualObservationError("at least one state term is required")

    def terms(self) -> tuple[tuple[str, int], ...]:
        """Enabled terms in layout order, as ``(name, width)``."""
        return tuple(
            (name, width)
            for name, width, enabled in (
                ("flange_pose", FLANGE_POSE_WIDTH, self.flange_pose),
                ("wrench", WRENCH_WIDTH, self.wrench),
                ("base_mean", BASE_MEAN_WIDTH, self.base_mean),
                ("socket_pose", SOCKET_POSE_WIDTH, self.socket_pose),
            )
            if enabled
        )

    def state_dim(self) -> int:
        return sum(width for _, width in self.terms())

    def slices(self) -> dict[str, slice]:
        out: dict[str, slice] = {}
        offset = 0
        for name, width in self.terms():
            out[name] = slice(offset, offset + width)
            offset += width
        return out

    def describe(self) -> str:
        return ", ".join(
            f"[{s.start}:{s.stop}] {name}" for name, s in self.slices().items()
        )


def assemble_state(
    spec: ResidualObservationSpec,
    terms: Mapping[str, Sequence[float] | np.ndarray],
    *,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Concatenate the enabled terms into a ``(1, state_dim)`` row.

    Every enabled term must have exactly its declared width and be finite.
    """
    parts: list[np.ndarray] = []
    for name, width in spec.terms():
        if name not in terms:
            raise ResidualObservationError(f"missing enabled residual term {name!r}")
        value = np.asarray(terms[name], dtype=np.float64).reshape(-1)
        if value.shape != (width,):
            raise ResidualObservationError(
                f"residual term {name!r} must have width {width}, got {value.shape[0]}"
            )
        if not np.all(np.isfinite(value)):
            raise ResidualObservationError(f"residual term {name!r} is not finite")
        parts.append(value)
    state = np.concatenate(parts, axis=0).astype(dtype, copy=False)
    if state.shape != (spec.state_dim(),):
        raise ResidualObservationError(
            f"assembled state width {state.shape[0]} != {spec.state_dim()}"
        )
    return state[None, :]
