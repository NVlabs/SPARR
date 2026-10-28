# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA OneWay Noncommercial License (March 22, 2022).
# See the LICENSE file in the repository root for the full license text:
# https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf

"""Pure reset-trajectory helpers; policy actions are never smoothed here."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .pose_math import Pose, normalize_quaternion_xyzw


def quaternion_slerp_xyzw(start: np.ndarray, goal: np.ndarray, fraction: float) -> np.ndarray:
    """Shortest-path spherical interpolation in ROS xyzw order."""

    if not math.isfinite(fraction):
        raise ValueError("fraction must be finite")
    alpha = min(1.0, max(0.0, float(fraction)))
    first = normalize_quaternion_xyzw(start)
    second = normalize_quaternion_xyzw(goal)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return normalize_quaternion_xyzw(first + alpha * (second - first))
    angle = math.acos(dot)
    denominator = math.sin(angle)
    return normalize_quaternion_xyzw(
        math.sin((1.0 - alpha) * angle) / denominator * first
        + math.sin(alpha * angle) / denominator * second
    )


@dataclass(frozen=True)
class TimedPoseTrajectory:
    """A finite reset/reposition trajectory between two flange poses."""

    start: Pose
    goal: Pose
    start_time_ns: int
    duration_ns: int

    def __post_init__(self) -> None:
        if self.start_time_ns < 0:
            raise ValueError("start_time_ns must be non-negative")
        if self.duration_ns <= 0:
            raise ValueError("duration_ns must be positive")

    def fraction(self, now_ns: int) -> float:
        return min(1.0, max(0.0, (int(now_ns) - self.start_time_ns) / self.duration_ns))

    def sample(self, now_ns: int) -> Pose:
        alpha = self.fraction(now_ns)
        return Pose(
            self.start.position + alpha * (self.goal.position - self.start.position),
            quaternion_slerp_xyzw(
                self.start.quaternion_xyzw,
                self.goal.quaternion_xyzw,
                alpha,
            ),
        )

    def complete(self, now_ns: int) -> bool:
        return int(now_ns) >= self.start_time_ns + self.duration_ns


def pose_error(current: Pose, goal: Pose) -> tuple[float, float]:
    """Return translation metres and shortest orientation radians."""

    translation = float(np.linalg.norm(goal.position - current.position))
    dot = abs(float(np.dot(current.quaternion_xyzw, goal.quaternion_xyzw)))
    rotation = 2.0 * math.acos(min(1.0, max(-1.0, dot)))
    return translation, rotation


def offset_position_world(pose: Pose, offset_xyz_m: tuple[float, float, float]) -> Pose:
    offset = np.asarray(offset_xyz_m, dtype=np.float64)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError("offset_xyz_m must be a finite 3-vector")
    return Pose(pose.position + offset, pose.quaternion_xyzw)


__all__ = [
    "TimedPoseTrajectory",
    "offset_position_world",
    "pose_error",
    "quaternion_slerp_xyzw",
]
