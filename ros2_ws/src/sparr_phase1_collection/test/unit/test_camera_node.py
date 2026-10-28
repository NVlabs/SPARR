# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA OneWay Noncommercial License (March 22, 2022).
# See the LICENSE file in the repository root for the full license text:
# https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf

"""Unit tests for the pyrealsense2-backed camera node.

These run in an image with neither ``pyrealsense2`` nor ``sensor_msgs``
installed, so every message type is a stub built by an injected factory.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from sparr_phase1_collection.camera_node import (
    CameraNodeError,
    ProfileNotSupportedError,
    StreamProfileSpec,
    build_camera_info,
    build_frame_messages,
    build_image,
    build_stamp,
    resolve_frame_stamp_ns,
    ros_distortion_model,
    rs_format_for_encoding,
    select_color_profile,
)
from sparr_phase1_collection.camera_validation import camera_info_errors
from sparr_phase1_collection.configuration import CameraConfig


FRAME_ID = "camera_color_optical_frame"
WIDTH = 848
HEIGHT = 480


def _stamp_stub():
    return SimpleNamespace(sec=0, nanosec=0)


def _image_stub():
    return SimpleNamespace(
        header=SimpleNamespace(stamp=None, frame_id=""),
        height=0,
        width=0,
        encoding="",
        is_bigendian=0,
        step=0,
        data=b"",
    )


def _camera_info_stub():
    return SimpleNamespace(
        header=SimpleNamespace(stamp=None, frame_id=""),
        height=0,
        width=0,
        distortion_model="",
        d=[],
        k=[0.0] * 9,
        r=[0.0] * 9,
        p=[0.0] * 12,
        binning_x=0,
        binning_y=0,
    )


def _intrinsics(width=WIDTH, height=HEIGHT, fx=615.5, fy=616.25, ppx=423.5, ppy=239.75):
    return SimpleNamespace(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        ppx=ppx,
        ppy=ppy,
        model=SimpleNamespace(name="brown_conrady"),
        coeffs=[0.1, -0.2, 0.001, 0.002, 0.0],
    )


def _frame(width=WIDTH, height=HEIGHT, channels=3):
    return np.arange(width * height * channels, dtype=np.uint8).reshape(
        height, width, channels
    )


def _collector_config():
    return CameraConfig(
        name="wrist_1",
        frame_id=FRAME_ID,
        encoding="rgb8",
        input_width=WIDTH,
        input_height=HEIGHT,
    )


def test_camera_info_uses_real_intrinsics_in_row_major_k_layout():
    info = build_camera_info(
        _intrinsics(),
        frame_id=FRAME_ID,
        stamp=build_stamp(0, stamp_factory=_stamp_stub),
        message_factory=_camera_info_stub,
    )
    assert list(info.k) == [615.5, 0.0, 423.5, 0.0, 616.25, 239.75, 0.0, 0.0, 1.0]
    assert (info.width, info.height) == (WIDTH, HEIGHT)
    assert info.header.frame_id == FRAME_ID
    assert info.distortion_model == "plumb_bob"
    assert list(info.d) == [0.1, -0.2, 0.001, 0.002, 0.0]
    assert list(info.r) == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert list(info.p) == [615.5, 0.0, 423.5, 0.0, 0.0, 616.25, 239.75, 0.0, 0.0, 0.0, 1.0, 0.0]


def test_camera_info_satisfies_collector_validation():
    info = build_camera_info(
        _intrinsics(),
        frame_id=FRAME_ID,
        stamp=build_stamp(1, stamp_factory=_stamp_stub),
        message_factory=_camera_info_stub,
    )
    assert camera_info_errors(info, _collector_config()) == ()


def test_camera_info_rejects_uncalibrated_focal_lengths():
    with pytest.raises(CameraNodeError, match="non-positive focal"):
        build_camera_info(
            _intrinsics(fx=0.0),
            frame_id=FRAME_ID,
            stamp=build_stamp(0, stamp_factory=_stamp_stub),
            message_factory=_camera_info_stub,
        )


def test_unmapped_distortion_keeps_the_librealsense_name():
    assert ros_distortion_model("modified_brown_conrady") == "plumb_bob"
    assert ros_distortion_model(SimpleNamespace(name="distortion.ftheta")) == "equidistant"
    assert ros_distortion_model("kannala_brandt4") == "kannala_brandt4"


def test_image_matches_the_encoding_size_and_frame_the_collector_requires():
    config = _collector_config()
    image = build_image(
        _frame(),
        frame_id=FRAME_ID,
        stamp=build_stamp(0, stamp_factory=_stamp_stub),
        encoding="rgb8",
        message_factory=_image_stub,
    )
    assert image.encoding == config.encoding
    assert image.width == config.input_width
    assert image.height == config.input_height
    assert image.header.frame_id == config.frame_id
    assert image.step == WIDTH * 3
    assert image.is_bigendian == 0
    assert len(image.data) == WIDTH * HEIGHT * 3


def test_image_data_is_the_untouched_rgb_buffer():
    frame = _frame(width=4, height=2)
    image = build_image(
        frame,
        frame_id=FRAME_ID,
        stamp=build_stamp(0, stamp_factory=_stamp_stub),
        message_factory=_image_stub,
    )
    restored = np.frombuffer(image.data, dtype=np.uint8).reshape(2, 4, 3)
    assert np.array_equal(restored, frame)


def test_image_rejects_a_channel_count_that_does_not_match_the_encoding():
    with pytest.raises(CameraNodeError, match="channel"):
        build_image(
            _frame(width=4, height=2, channels=1),
            frame_id=FRAME_ID,
            stamp=build_stamp(0, stamp_factory=_stamp_stub),
            encoding="rgb8",
            message_factory=_image_stub,
        )


def test_unknown_encoding_is_rejected_with_the_supported_list():
    with pytest.raises(CameraNodeError, match="not supported"):
        rs_format_for_encoding("yuyv")
    assert rs_format_for_encoding("rgb8") == ("rgb8", 3)


def test_image_and_camera_info_carry_the_identical_stamp():
    stamp = build_stamp(1_234_000_005_678, stamp_factory=_stamp_stub)
    image, info = build_frame_messages(
        _frame(width=8, height=4),
        _intrinsics(width=8, height=4),
        frame_id=FRAME_ID,
        stamp=stamp,
        image_factory=_image_stub,
        camera_info_factory=_camera_info_stub,
    )
    assert image.header.stamp is info.header.stamp
    assert (image.header.stamp.sec, image.header.stamp.nanosec) == (1234, 5678)


def test_global_time_domain_is_preferred_over_arrival_time():
    ros_now_ns = 1_700_000_000_000_000_000
    hardware_ms = (ros_now_ns - 4_000_000) / 1_000_000.0  # 4 ms before arrival
    stamp_ns, source = resolve_frame_stamp_ns(
        domain="global_time",
        frame_timestamp_ms=hardware_ms,
        ros_now_ns=ros_now_ns,
    )
    assert source == "global_time"
    assert stamp_ns == ros_now_ns - 4_000_000


def test_hardware_clock_domain_falls_back_to_the_ros_clock():
    stamp_ns, source = resolve_frame_stamp_ns(
        domain="timestamp_domain.hardware_clock",
        frame_timestamp_ms=12345.678,  # device uptime, unknown epoch
        ros_now_ns=1_700_000_000_000_000_000,
    )
    assert source == "ros_clock"
    assert stamp_ns == 1_700_000_000_000_000_000


def test_wildly_skewed_epoch_stamp_is_distrusted():
    stamp_ns, source = resolve_frame_stamp_ns(
        domain="system_time",
        frame_timestamp_ms=1.0,
        ros_now_ns=1_700_000_000_000_000_000,
    )
    assert source == "ros_clock"
    assert stamp_ns == 1_700_000_000_000_000_000


AVAILABLE = (
    StreamProfileSpec(848, 480, 30, "rgb8"),
    StreamProfileSpec(848, 480, 60, "rgb8"),
    StreamProfileSpec(640, 480, 30, "bgr8"),
)


def test_supported_profile_is_selected_exactly():
    profile = select_color_profile(
        AVAILABLE, width=848, height=480, fps=30, format_name="rgb8"
    )
    assert profile == StreamProfileSpec(848, 480, 30, "rgb8")


@pytest.mark.parametrize(
    "width,height,fps,format_name",
    [
        (1280, 720, 30, "rgb8"),
        (848, 480, 15, "rgb8"),
        (848, 480, 30, "yuyv"),
    ],
)
def test_unsupported_profile_raises_and_lists_what_the_device_offers(
    width, height, fps, format_name
):
    with pytest.raises(ProfileNotSupportedError) as excinfo:
        select_color_profile(
            AVAILABLE, width=width, height=height, fps=fps, format_name=format_name
        )
    message = str(excinfo.value)
    assert "is not supported" in message
    assert "848x480 @ 30Hz rgb8" in message
    assert "640x480 @ 30Hz bgr8" in message


def test_profile_error_is_clear_when_the_device_advertises_nothing():
    with pytest.raises(ProfileNotSupportedError, match="no colour profiles"):
        select_color_profile((), width=848, height=480, fps=30, format_name="rgb8")


def _write_config(tmp_path, *, wrist_2_enabled=False):
    """A collection config with one or two cameras enabled.

    Field values come from phase1_live.yaml, but the enabled flags are set
    explicitly so the test does not depend on which cameras are configured there.
    """
    import shutil
    from pathlib import Path

    import yaml

    source = Path(__file__).resolve().parents[5] / "phase1_live.yaml"
    if not source.is_file():
        import pytest

        pytest.skip(f"{source} not present")
    data = yaml.safe_load(source.read_text())
    data["cameras"]["wrist_1"]["enabled"] = True
    data["cameras"].setdefault("wrist_2", {})["enabled"] = False
    if wrist_2_enabled:
        second = dict(data["cameras"]["wrist_1"])
        second.update(
            {
                "serial": "999888777666",
                "image_topic": "/camera/wrist_2/color/image_rect_raw",
                "camera_info_topic": "/camera/wrist_2/color/camera_info",
                "frame_id": "wrist_2_color_optical_frame",
                "calibration_id": "realsense-camera-info-999888777666-848x480-rgb8-v1",
                "enabled": True,
            }
        )
        data["cameras"]["wrist_2"] = second
        data.setdefault("synchronization", {})["max_camera_pair_skew_ms"] = 20.0
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_settings_come_from_the_named_cameras_block(tmp_path):
    """Guards against the driver ignoring the config and running on its defaults."""
    from sparr_phase1_collection.camera_node import RealSenseCameraNode

    path = _write_config(tmp_path, wrist_2_enabled=True)
    first = RealSenseCameraNode._defaults_from_config(str(path), "wrist_1")
    second = RealSenseCameraNode._defaults_from_config(str(path), "wrist_2")

    assert first["serial"] != second["serial"], "each camera must claim its own device"
    assert first["image_topic"] != second["image_topic"]
    assert second["serial"] == "999888777666"
    assert second["frame_id"] == "wrist_2_color_optical_frame"
    assert first["encoding"] == second["encoding"] == "rgb8"


def test_a_single_enabled_camera_needs_no_name(tmp_path):
    from sparr_phase1_collection.camera_node import RealSenseCameraNode

    path = _write_config(tmp_path)
    settings = RealSenseCameraNode._defaults_from_config(str(path), "")
    assert settings["serial"]
    assert settings["image_topic"]


def test_two_enabled_cameras_require_a_name(tmp_path):
    """Otherwise both instances would silently claim the same device."""
    import pytest

    from sparr_phase1_collection.camera_node import CameraNodeError, RealSenseCameraNode

    path = _write_config(tmp_path, wrist_2_enabled=True)
    with pytest.raises(CameraNodeError, match="pass -p camera"):
        RealSenseCameraNode._defaults_from_config(str(path), "")


def test_an_unknown_camera_name_is_rejected(tmp_path):
    import pytest

    from sparr_phase1_collection.camera_node import CameraNodeError, RealSenseCameraNode

    path = _write_config(tmp_path)
    with pytest.raises(CameraNodeError, match="unknown camera"):
        RealSenseCameraNode._defaults_from_config(str(path), "wrist_3")


def test_starting_a_driver_for_a_disabled_camera_is_refused(tmp_path):
    """The collector would never subscribe to it, so the stream would go nowhere."""
    import pytest

    from sparr_phase1_collection.camera_node import CameraNodeError, RealSenseCameraNode

    path = _write_config(tmp_path)  # wrist_2 disabled
    with pytest.raises(CameraNodeError, match="disabled"):
        RealSenseCameraNode._defaults_from_config(str(path), "wrist_2")


def test_naming_a_camera_without_a_config_is_refused():
    import pytest

    from sparr_phase1_collection.camera_node import CameraNodeError, RealSenseCameraNode

    with pytest.raises(CameraNodeError, match="config_path is unset"):
        RealSenseCameraNode._defaults_from_config("", "wrist_1")


def test_no_config_keeps_the_built_in_defaults():
    from sparr_phase1_collection.camera_node import RealSenseCameraNode

    assert RealSenseCameraNode._defaults_from_config("", "") == {}
