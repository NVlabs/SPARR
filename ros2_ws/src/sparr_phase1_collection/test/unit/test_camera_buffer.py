# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA License.
# See the LICENSE file in the repository root for the full license text.

import numpy as np
import pytest

from sparr_phase1_collection.camera_buffer import (
    CameraBuffer,
    CameraBufferError,
    CropRegion,
    TwoCameraBuffer,
)


def test_single_camera_frame_set_has_no_pair_skew():
    buffer = CameraBuffer(("wrist_1",), (2, 2))
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    buffer.add_frame("wrist_1", 95, image)

    frames = buffer.closest_frames(
        100,
        max_delta_ns=10,
        # Pair-skew limits do not apply to a single camera.
        max_camera_skew_ns=0,
    )

    assert frames is not None
    assert frames.camera_ids == ("wrist_1",)
    assert frames.by_camera("wrist_1").timestamp_ns == 95
    assert frames.per_camera_delta_ns == {"wrist_1": 5}
    assert frames.camera_skew_ns is None


def test_frame_is_cropped_resized_and_detached_before_storage():
    source = np.arange(8 * 10 * 3, dtype=np.uint16).reshape(8, 10, 3)
    buffer = TwoCameraBuffer(
        ("wrist", "side"),
        (2, 2),
        crops={"wrist": CropRegion(x=2, y=2, width=4, height=4)},
    )

    frame = buffer.add_frame("wrist", 100, source)
    stored_before_mutation = frame.image.copy()
    source[...] = 0

    assert frame.source_shape == (8, 10, 3)
    assert frame.image.shape == (2, 2, 3)
    assert frame.image.nbytes < source.nbytes
    assert not np.shares_memory(frame.image, source)
    assert np.array_equal(frame.image, stored_before_mutation)
    assert frame.image.flags.writeable is False


def test_crop_bounds_are_rejected_instead_of_clamped():
    buffer = TwoCameraBuffer(
        ("wrist", "side"),
        (2, 2),
        crops={"wrist": CropRegion(x=8, y=0, width=4, height=4)},
    )

    with pytest.raises(CameraBufferError, match="exceeds image bounds"):
        buffer.add_frame("wrist", 100, np.zeros((8, 10, 3), dtype=np.uint8))


def test_closest_pair_uses_older_frame_on_tie_and_enforces_both_windows():
    buffer = TwoCameraBuffer(("wrist", "side"), (2, 2))
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    buffer.add_frame("wrist", 90, image)
    buffer.add_frame("wrist", 110, image)
    buffer.add_frame("side", 95, image)
    buffer.add_frame("side", 130, image)

    pair = buffer.closest_pair(100, max_delta_ns=10, max_camera_skew_ns=5)

    assert pair is not None
    assert pair.by_camera("wrist").timestamp_ns == 90
    assert pair.by_camera("side").timestamp_ns == 95
    assert pair.camera_skew_ns == 5
    assert pair.per_camera_delta_ns == {"wrist": 10, "side": 5}
    assert buffer.closest_pair(100, max_delta_ns=4) is None
    assert buffer.closest_pair(100, max_delta_ns=10, max_camera_skew_ns=4) is None


def test_buffer_is_bounded_per_camera_even_with_out_of_order_arrival():
    buffer = TwoCameraBuffer(("wrist", "side"), (1, 1), capacity=2)
    image = np.zeros((2, 2), dtype=np.uint8)
    buffer.add_frame("wrist", 30, image)
    buffer.add_frame("wrist", 10, image)
    buffer.add_frame("wrist", 20, image)

    assert buffer.count("wrist") == 2
    assert buffer.closest_frame("wrist", 10) is not None
    # Capacity retains the two newest timestamps, so timestamp 10 was evicted.
    assert buffer.closest_frame("wrist", 10).timestamp_ns == 20
