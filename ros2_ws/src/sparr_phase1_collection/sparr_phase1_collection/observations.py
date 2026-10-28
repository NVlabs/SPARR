# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA License.
# See the LICENSE file in the repository root for the full license text.

"""Exact ROS-independent 18-D base-policy observation encoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TYPE_CHECKING

import numpy as np

from .pose_math import Pose, quaternion_to_matrix_xyzw, rotation_6d_rows_xyzw

if TYPE_CHECKING:
    from .checkpoint import Phase1PolicySpec


OBSERVATION_ORDER = (
    "eef_pos",
    "eef_rot_6d",
    "socket_kp_pos",
    "socket_kp_rot_6d",
)
#: Newton layout: same four quantities, socket first; its tool offset is zero.
NEWTON_OBSERVATION_ORDER = (
    "socket_pos",
    "tool_pos",
    "tool_rot_6d",
    "socket_rot_6d",
)
#: Orders this encoder can produce. Membership is checked because a wrong
#: order is silent: the policy still gets 18 finite numbers.
SUPPORTED_OBSERVATION_ORDERS = (OBSERVATION_ORDER, NEWTON_OBSERVATION_ORDER)
OBSERVATION_DIM = 18


def _offset3(value: Sequence[float], name: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain three finite values")
    return float(array[0]), float(array[1]), float(array[2])


@dataclass(frozen=True)
class ObservationConfig:
    """Local-frame keypoint offsets captured in the policy checkpoint."""

    eef_offset_flange: tuple[float, float, float]
    socket_keypoint_offset: tuple[float, float, float]
    observation_order: tuple[str, ...] = OBSERVATION_ORDER

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "eef_offset_flange",
            _offset3(self.eef_offset_flange, "eef_offset_flange"),
        )
        object.__setattr__(
            self,
            "socket_keypoint_offset",
            _offset3(self.socket_keypoint_offset, "socket_keypoint_offset"),
        )
        order = tuple(self.observation_order)
        if order not in SUPPORTED_OBSERVATION_ORDERS:
            raise ValueError(
                f"observation order must be one of {SUPPORTED_OBSERVATION_ORDERS}, "
                f"got {order}"
            )
        object.__setattr__(self, "observation_order", order)

    @classmethod
    def from_policy_spec(cls, spec: "Phase1PolicySpec") -> "ObservationConfig":
        return cls(
            eef_offset_flange=spec.eef_offset,
            socket_keypoint_offset=spec.socket_keypoint_offset,
            observation_order=spec.observation_order,
        )


def offset_point_world(pose_world: Pose, offset_local: Sequence[float]) -> np.ndarray:
    """Apply a body-local point offset and return its position in world."""

    offset = np.asarray(_offset3(offset_local, "offset_local"), dtype=np.float64)
    return (
        pose_world.position
        + quaternion_to_matrix_xyzw(pose_world.quaternion_xyzw) @ offset
    )


def socket_root_from_opening(
    opening_pose_world: Pose, socket_keypoint_offset: Sequence[float]
) -> Pose:
    """Socket root pose whose keypoint (root + R * offset) sits at the opening.

    Inverse of ``offset_point_world`` for the socket: orientation is shared, and
    the keypoint offset is removed from the position.
    """

    offset = np.asarray(
        _offset3(socket_keypoint_offset, "socket_keypoint_offset"), dtype=np.float64
    )
    rotation = quaternion_to_matrix_xyzw(opening_pose_world.quaternion_xyzw)
    return Pose(
        position=opening_pose_world.position - rotation @ offset,
        quaternion_xyzw=opening_pose_world.quaternion_xyzw,
    )


def encode_phase1_observation(
    flange_pose_world: Pose,
    socket_pose_world: Pose,
    config: ObservationConfig,
    *,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Encode the 18-D policy observation in the order the checkpoint declares.

    Supported orders are ``eef_pos, eef_rot_6d, socket_kp_pos, socket_kp_rot_6d``
    and ``socket_pos, tool_pos, tool_rot_6d, socket_rot_6d``. Positions apply the
    checkpoint's body-local keypoint offsets; 6-D rotations are the first two rows
    of the world rotation matrices.
    """

    tool_point = offset_point_world(flange_pose_world, config.eef_offset_flange)
    tool_rot = rotation_6d_rows_xyzw(flange_pose_world.quaternion_xyzw)
    socket_point = offset_point_world(socket_pose_world, config.socket_keypoint_offset)
    socket_rot = rotation_6d_rows_xyzw(socket_pose_world.quaternion_xyzw)

    terms = {
        "eef_pos": tool_point,
        "eef_rot_6d": tool_rot,
        "socket_kp_pos": socket_point,
        "socket_kp_rot_6d": socket_rot,
        # Newton's names for the same four quantities.
        "tool_pos": tool_point,
        "tool_rot_6d": tool_rot,
        "socket_pos": socket_point,
        "socket_rot_6d": socket_rot,
    }
    missing = [name for name in config.observation_order if name not in terms]
    if missing:
        raise ValueError(
            f"observation order {config.observation_order} names terms this "
            f"encoder cannot produce: {missing}"
        )
    values = np.concatenate([terms[name] for name in config.observation_order])
    if values.shape != (OBSERVATION_DIM,):
        raise AssertionError(
            f"internal observation encoder produced {values.shape}, expected ({OBSERVATION_DIM},)"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("encoded observation contains non-finite values")
    return values.astype(dtype, copy=False)
