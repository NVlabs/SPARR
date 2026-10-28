# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA OneWay Noncommercial License (March 22, 2022).
# See the LICENSE file in the repository root for the full license text:
# https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf

"""Deterministic Phase 1 action composition and inverse labeling.

The pipeline is stateless (no EMA): it clips the sample to the checkpoint's
training bound, applies checkpoint scales, composes the delta onto the flange
pose from the same observation snapshot, and converts the flange target to TCP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, TYPE_CHECKING

import numpy as np

from .pose_math import (
    Pose,
    apply_world_pose_delta,
    flange_to_tcp_pose,
)

if TYPE_CHECKING:
    from .checkpoint import Phase1PolicySpec


ACTION_DIM = 6


def _numpy_value(value: Any) -> np.ndarray:
    """Convert arrays or detached CPU/GPU Torch tensors without importing Torch."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def _action6(value: Any, name: str) -> np.ndarray:
    action = _numpy_value(value)
    # A base policy normally returns [batch=1, action=6].  Accept that without
    # making the pure action path carry a batch dimension.
    if action.shape == (1, ACTION_DIM):
        action = action[0]
    if action.shape != (ACTION_DIM,):
        raise ValueError(f"{name} must have shape ({ACTION_DIM},), got {action.shape}")
    if not np.all(np.isfinite(action)):
        raise ValueError(f"{name} must contain only finite values")
    result = action.copy()
    result.setflags(write=False)
    return result


def _scale3(value: float | Sequence[float], name: str) -> tuple[float, float, float]:
    if isinstance(value, (int, float)):
        scale = np.full(3, float(value), dtype=np.float64)
    else:
        scale = np.asarray(value, dtype=np.float64)
    if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError(f"{name} must be a positive finite scalar or 3-vector")
    return float(scale[0]), float(scale[1]), float(scale[2])


@dataclass(frozen=True)
class ActionPipelineConfig:
    """Checkpoint-derived task-space scales and symmetric action bound."""

    position_scale: tuple[float, float, float]
    orientation_scale: tuple[float, float, float]
    action_clip: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "position_scale", _scale3(self.position_scale, "position_scale")
        )
        object.__setattr__(
            self,
            "orientation_scale",
            _scale3(self.orientation_scale, "orientation_scale"),
        )
        action_clip = float(self.action_clip)
        if not np.isfinite(action_clip) or action_clip <= 0.0:
            raise ValueError("action_clip must be positive and finite")
        object.__setattr__(self, "action_clip", action_clip)

    @classmethod
    def from_policy_spec(cls, spec: "Phase1PolicySpec") -> "ActionPipelineConfig":
        return cls(
            position_scale=spec.position_scale,
            orientation_scale=spec.orientation_scale,
            action_clip=spec.training_action_clip,
        )

    @property
    def position_scale_array(self) -> np.ndarray:
        return np.asarray(self.position_scale, dtype=np.float64)

    @property
    def orientation_scale_array(self) -> np.ndarray:
        return np.asarray(self.orientation_scale, dtype=np.float64)


@dataclass(frozen=True)
class ActionTrace:
    """Complete pure-math trace needed for Phase 1 pseudo-labeling."""

    raw_mean: np.ndarray
    raw_sample: np.ndarray
    clipped_sample: np.ndarray
    effective_software_action: np.ndarray
    scaled_translation_world: np.ndarray
    scaled_rotation_vector_world: np.ndarray
    observed_flange_pose: Pose
    effective_flange_target: Pose
    submitted_controller_tcp_target: Pose
    effective_action_minus_raw_mean: np.ndarray
    deterministic_inference: bool = False
    # Phase 2 only; None in Phase 1.
    requested_residual: Optional[np.ndarray] = None
    # Whether ``requested_residual`` reached the command; shadow mode records a
    # residual without applying it.
    residual_applied: bool = False
    #: Which learner broadcast produced ``requested_residual``. -1 when there
    #: was no residual.
    residual_params_version: int = -1

    def __post_init__(self) -> None:
        for field_name in (
            "raw_mean",
            "raw_sample",
            "clipped_sample",
            "effective_software_action",
            "effective_action_minus_raw_mean",
        ):
            object.__setattr__(
                self, field_name, _action6(getattr(self, field_name), field_name)
            )
        for field_name in ("scaled_translation_world", "scaled_rotation_vector_world"):
            value = _numpy_value(getattr(self, field_name))
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{field_name} must be a finite 3-vector")
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, field_name, value)


def clip_normalized_action(
    raw_sample: Any, action_clip: float, *, name: str = "raw_sample"
) -> np.ndarray:
    """Clip a raw normalized action to a symmetric normalized-action bound."""

    sample = _action6(raw_sample, name)
    clip = float(action_clip)
    if not np.isfinite(clip) or clip <= 0.0:
        raise ValueError("action_clip must be positive and finite")
    clipped = np.clip(sample, -clip, clip)
    clipped.setflags(write=False)
    return clipped


def build_action_trace(
    *,
    raw_mean: Any,
    raw_sample: Any,
    observed_flange_pose: Pose,
    flange_to_tcp: Pose,
    config: ActionPipelineConfig,
    deterministic: bool = False,
    residual: Any = None,
    residual_params_version: int = -1,
) -> ActionTrace:
    """Run clip -> scale -> pose_rel -> flange-to-TCP and create the label trace.

    With ``deterministic=True`` the clipped mean drives the command; the sample is
    still recorded. Callers must mark such trials debug-only, since their residual
    label is ~zero by construction.

    ``residual`` (Phase 2; ``None`` in Phase 1) is a 6-D delta composed as
    ``clip(mean + residual)``, so it can steer a base action saturated at the
    bound. It supersedes ``deterministic``: exploration comes from the residual
    policy, not the base policy's sigma.
    """

    mean = _action6(raw_mean, "raw_mean")
    sample = _action6(raw_sample, "raw_sample")
    clipped_sample = clip_normalized_action(sample, config.action_clip)
    if residual is not None:
        residual_action = _action6(residual, "residual")
        effective_action = clip_normalized_action(
            mean + residual_action, config.action_clip, name="mean+residual"
        )
    elif deterministic:
        effective_action = clip_normalized_action(
            mean, config.action_clip, name="raw_mean"
        )
    else:
        effective_action = clipped_sample
    translation = effective_action[:3] * config.position_scale_array
    rotation_vector = effective_action[3:] * config.orientation_scale_array
    effective_flange_target = apply_world_pose_delta(
        observed_flange_pose, translation, rotation_vector
    )
    submitted_tcp_target = flange_to_tcp_pose(effective_flange_target, flange_to_tcp)
    label = effective_action - mean

    return ActionTrace(
        raw_mean=mean,
        raw_sample=sample,
        clipped_sample=clipped_sample,
        effective_software_action=effective_action,
        scaled_translation_world=translation,
        scaled_rotation_vector_world=rotation_vector,
        observed_flange_pose=observed_flange_pose,
        effective_flange_target=effective_flange_target,
        submitted_controller_tcp_target=submitted_tcp_target,
        effective_action_minus_raw_mean=label,
        deterministic_inference=bool(deterministic),
        requested_residual=(None if residual is None else residual_action),
        residual_applied=residual is not None,
        residual_params_version=int(residual_params_version),
    )
