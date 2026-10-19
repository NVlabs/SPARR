# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Pure validation of live CameraInfo identity and intrinsic calibration."""

from __future__ import annotations

from typing import Any

import numpy as np

from .configuration import CameraConfig


def camera_info_errors(message: Any, config: CameraConfig) -> tuple[str, ...]:
    """Return mismatches that make camera calibration provenance ambiguous."""

    errors: list[str] = []
    header = getattr(message, "header", None)
    frame_id = "" if header is None else str(getattr(header, "frame_id", ""))
    if frame_id != config.frame_id:
        errors.append(
            f"frame_id {frame_id!r} does not match configured {config.frame_id!r}"
        )

    width = int(getattr(message, "width", 0))
    height = int(getattr(message, "height", 0))
    if config.input_width is None or width != config.input_width:
        errors.append(f"width {width} does not match configured {config.input_width}")
    if config.input_height is None or height != config.input_height:
        errors.append(f"height {height} does not match configured {config.input_height}")

    try:
        intrinsics = np.asarray(getattr(message, "k", ()), dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        intrinsics = np.empty((0,), dtype=np.float64)
    if (
        intrinsics.shape != (9,)
        or not np.all(np.isfinite(intrinsics))
        or intrinsics[0] <= 0.0
        or intrinsics[4] <= 0.0
    ):
        errors.append("K must be a finite calibrated 3x3 matrix with positive fx and fy")
    return tuple(errors)


__all__ = ["camera_info_errors"]
