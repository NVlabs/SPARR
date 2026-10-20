# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""ROS-independent pose and quaternion math for the Phase 1 action path.

Quaternions use ROS component order ``[x, y, z, w]`` throughout.  Policy
translation deltas are expressed in the world frame and rotation-vector deltas
are left-multiplied, matching Isaac Lab ``apply_delta_pose`` for ``pose_rel``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


ArrayLike = Sequence[float] | np.ndarray
_EPS = 1.0e-12


def _finite_vector(value: ArrayLike, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def normalize_quaternion_xyzw(quaternion: ArrayLike) -> np.ndarray:
    """Return a unit ``xyzw`` quaternion, rejecting a zero quaternion."""

    quat = _finite_vector(quaternion, 4, "quaternion")
    norm = float(np.linalg.norm(quat))
    if norm < _EPS:
        raise ValueError("quaternion norm is zero")
    return quat / norm


@dataclass(frozen=True)
class Pose:
    """A position and unit quaternion expressed in one parent frame."""

    position: np.ndarray
    quaternion_xyzw: np.ndarray

    def __post_init__(self) -> None:
        position = _finite_vector(self.position, 3, "position")
        quaternion = normalize_quaternion_xyzw(self.quaternion_xyzw)
        position.setflags(write=False)
        quaternion.setflags(write=False)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "quaternion_xyzw", quaternion)

    @classmethod
    def from_vector(cls, value: ArrayLike) -> "Pose":
        vector = _finite_vector(value, 7, "pose")
        return cls(vector[:3], vector[3:])

    def as_vector(self, *, dtype: np.dtype | type | None = None) -> np.ndarray:
        vector = np.concatenate((self.position, self.quaternion_xyzw))
        return vector.astype(dtype, copy=False) if dtype is not None else vector


def quaternion_conjugate_xyzw(quaternion: ArrayLike) -> np.ndarray:
    quat = normalize_quaternion_xyzw(quaternion)
    return np.array((-quat[0], -quat[1], -quat[2], quat[3]), dtype=np.float64)


def quaternion_multiply_xyzw(left: ArrayLike, right: ArrayLike) -> np.ndarray:
    """Hamilton product ``left * right`` in ROS ``xyzw`` order."""

    x1, y1, z1, w1 = normalize_quaternion_xyzw(left)
    x2, y2, z2, w2 = normalize_quaternion_xyzw(right)
    product = np.array(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dtype=np.float64,
    )
    return normalize_quaternion_xyzw(product)


def quaternion_to_matrix_xyzw(quaternion: ArrayLike) -> np.ndarray:
    """Convert a unit ``xyzw`` quaternion to a 3-by-3 rotation matrix."""

    x, y, z, w = normalize_quaternion_xyzw(quaternion)
    return np.array(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
            (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
            (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def rotation_6d_rows_xyzw(quaternion: ArrayLike) -> np.ndarray:
    """Return the first two rows of ``R`` in row-major order.

    This is the 6-D rotation encoding the policy was trained with, yielding
    six values ``R[0, :]`` then ``R[1, :]``.
    """

    rotation = quaternion_to_matrix_xyzw(quaternion)
    return rotation[:2, :].reshape(6).copy()


def rotvec_to_quaternion_xyzw(rotation_vector: ArrayLike) -> np.ndarray:
    """Convert an axis-angle rotation vector to a unit ``xyzw`` quaternion."""

    rotvec = _finite_vector(rotation_vector, 3, "rotation_vector")
    angle = float(np.linalg.norm(rotvec))
    if angle < _EPS:
        return np.array((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    half_angle = 0.5 * angle
    xyz = rotvec * (math.sin(half_angle) / angle)
    return normalize_quaternion_xyzw((*xyz, math.cos(half_angle)))


def quaternion_to_rotvec_xyzw(quaternion: ArrayLike) -> np.ndarray:
    """Convert ``xyzw`` to the principal axis-angle rotation vector."""

    quat = normalize_quaternion_xyzw(quaternion)
    # q and -q encode the same orientation.  Choose w >= 0 for the principal
    # rotation in [-pi, pi], which makes small action round-trips unambiguous.
    if quat[3] < 0.0:
        quat = -quat
    vector_norm = float(np.linalg.norm(quat[:3]))
    if vector_norm < _EPS:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, float(quat[3]))
    return quat[:3] * (angle / vector_norm)


def compose_poses(parent_to_child: Pose, child_to_grandchild: Pose) -> Pose:
    """Compose two rigid transforms represented as poses."""

    position = (
        parent_to_child.position
        + quaternion_to_matrix_xyzw(parent_to_child.quaternion_xyzw)
        @ child_to_grandchild.position
    )
    quaternion = quaternion_multiply_xyzw(
        parent_to_child.quaternion_xyzw, child_to_grandchild.quaternion_xyzw
    )
    return Pose(position, quaternion)


def invert_pose(pose: Pose) -> Pose:
    inverse_quaternion = quaternion_conjugate_xyzw(pose.quaternion_xyzw)
    inverse_position = -quaternion_to_matrix_xyzw(inverse_quaternion) @ pose.position
    return Pose(inverse_position, inverse_quaternion)


def apply_world_pose_delta(
    base_pose: Pose,
    translation_world: ArrayLike,
    rotation_vector_world: ArrayLike,
) -> Pose:
    """Apply the trained task-space ``pose_rel`` convention to ``base_pose``.

    Translation is added directly in the world frame.  Rotation is a world-frame
    rotation vector whose quaternion is left-multiplied onto the base orientation.
    """

    translation = _finite_vector(translation_world, 3, "translation_world")
    delta_quaternion = rotvec_to_quaternion_xyzw(rotation_vector_world)
    return Pose(
        base_pose.position + translation,
        quaternion_multiply_xyzw(delta_quaternion, base_pose.quaternion_xyzw),
    )


def world_pose_delta(base_pose: Pose, target_pose: Pose) -> tuple[np.ndarray, np.ndarray]:
    """Invert :func:`apply_world_pose_delta` for the principal rotation."""

    translation = target_pose.position - base_pose.position
    delta_quaternion = quaternion_multiply_xyzw(
        target_pose.quaternion_xyzw,
        quaternion_conjugate_xyzw(base_pose.quaternion_xyzw),
    )
    return translation, quaternion_to_rotvec_xyzw(delta_quaternion)


def flange_to_tcp_pose(flange_pose_world: Pose, flange_to_tcp: Pose) -> Pose:
    """Restate a desired flange pose as the corresponding TCP pose."""

    return compose_poses(flange_pose_world, flange_to_tcp)


