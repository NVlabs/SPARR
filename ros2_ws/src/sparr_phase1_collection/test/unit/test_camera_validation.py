# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA License.
# See the LICENSE file in the repository root for the full license text.

from types import SimpleNamespace

from sparr_phase1_collection.camera_validation import camera_info_errors
from sparr_phase1_collection.configuration import CameraConfig


def _config() -> CameraConfig:
    return CameraConfig(
        name="wrist_1",
        frame_id="wrist_1_optical",
        input_width=640,
        input_height=480,
    )


def _message(*, frame_id="wrist_1_optical", width=640, height=480, k=None):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id=frame_id),
        width=width,
        height=height,
        k=k or [500.0, 0.0, 320.0, 0.0, 501.0, 240.0, 0.0, 0.0, 1.0],
    )


def test_camera_info_matches_configured_identity_and_calibration():
    assert camera_info_errors(_message(), _config()) == ()


def test_camera_info_rejects_frame_dimensions_and_blank_intrinsics():
    errors = camera_info_errors(
        _message(frame_id="wrong", width=320, height=240, k=[0.0] * 9),
        _config(),
    )
    assert any("frame_id" in error for error in errors)
    assert any("width" in error for error in errors)
    assert any("height" in error for error in errors)
    assert any("positive fx and fy" in error for error in errors)
