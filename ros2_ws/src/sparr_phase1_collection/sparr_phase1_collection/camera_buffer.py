"""Bounded, timestamped storage for one or two processed camera streams.

Frames are cropped and resized before entering the buffer and the caller's image
is never retained. With ``retain_source`` an owning copy of the captured frame is
also kept so it can be recorded.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .robot_state import timestamp_to_ns


class CameraBufferError(ValueError):
    """Raised when a camera frame cannot be stored without changing policy."""


@dataclass(frozen=True, slots=True)
class CropRegion:
    """An exact pixel crop; out-of-bounds regions are rejected, never clipped."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise CameraBufferError("crop values must be integers")
        if self.x < 0 or self.y < 0:
            raise CameraBufferError("crop origin must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise CameraBufferError("crop width and height must be positive")

    def validate_for(self, image_shape: Sequence[int]) -> None:
        if len(image_shape) < 2:
            raise CameraBufferError("image must have height and width dimensions")
        image_height, image_width = int(image_shape[0]), int(image_shape[1])
        if self.x + self.width > image_width or self.y + self.height > image_height:
            raise CameraBufferError(
                f"crop {self} exceeds image bounds {image_width}x{image_height}"
            )


@dataclass(frozen=True, slots=True)
class BufferedCameraFrame:
    camera_id: str
    timestamp_ns: int
    image: np.ndarray
    source_shape: tuple[int, ...]
    metadata: Mapping[str, Any]
    #: The frame before crop and resize; None unless built with ``retain_source``.
    source_image: Any = None


@dataclass(frozen=True, slots=True)
class CameraFrameSet:
    """Closest processed frame from every enabled camera for one robot sample."""

    target_timestamp_ns: int
    frames: tuple[BufferedCameraFrame, ...]

    def __post_init__(self) -> None:
        if len(self.frames) not in (1, 2):
            raise CameraBufferError("a camera frame set must contain one or two frames")
        camera_ids = tuple(frame.camera_id for frame in self.frames)
        if len(set(camera_ids)) != len(camera_ids):
            raise CameraBufferError("camera frame IDs must be unique")

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(frame.camera_id for frame in self.frames)

    @property
    def first(self) -> BufferedCameraFrame:
        """First enabled camera frame."""

        return self.frames[0]

    @property
    def second(self) -> BufferedCameraFrame:
        """Second enabled camera frame."""

        if len(self.frames) < 2:
            raise IndexError("the camera frame set contains only one frame")
        return self.frames[1]

    @property
    def per_camera_delta_ns(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                frame.camera_id: abs(frame.timestamp_ns - self.target_timestamp_ns)
                for frame in self.frames
            }
        )

    @property
    def camera_skew_ns(self) -> int | None:
        if len(self.frames) < 2:
            return None
        timestamps = [frame.timestamp_ns for frame in self.frames]
        return max(timestamps) - min(timestamps)

    def by_camera(self, camera_id: str) -> BufferedCameraFrame:
        for frame in self.frames:
            if frame.camera_id == camera_id:
                return frame
        raise KeyError(camera_id)


class CameraBuffer:
    """Thread-safe closest-frame buffer for one or two enabled cameras.

    Args:
        camera_ids: One or two unique, non-empty identifiers.
        output_size: ``(height, width)`` stored for every frame.
        crops: Optional exact per-camera :class:`CropRegion` definitions.
        capacity: Maximum processed frames retained per camera.
        retain_source: Also keep an owning copy of each frame as captured.

    Equal-distance ties choose the older frame, never a future one.
    """

    def __init__(
        self,
        camera_ids: Sequence[str],
        output_size: tuple[int, int],
        *,
        crops: Mapping[str, CropRegion | tuple[int, int, int, int]] | None = None,
        capacity: int = 120,
        retain_source: bool = False,
    ) -> None:
        if len(camera_ids) not in (1, 2):
            raise CameraBufferError("one or two camera IDs are required")
        normalized_ids = tuple(str(camera_id) for camera_id in camera_ids)
        if any(not camera_id for camera_id in normalized_ids):
            raise CameraBufferError("camera IDs must be non-empty")
        if len(set(normalized_ids)) != len(normalized_ids):
            raise CameraBufferError("camera IDs must be unique")
        if (
            len(output_size) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in output_size)
            or output_size[0] <= 0
            or output_size[1] <= 0
        ):
            raise CameraBufferError("output_size must be positive integer (height, width)")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise CameraBufferError("capacity must be a positive integer")

        unknown_crops = set(crops or ()) - set(normalized_ids)
        if unknown_crops:
            raise CameraBufferError(f"crop configured for unknown cameras: {unknown_crops}")

        self._camera_ids = normalized_ids
        self._output_size = (int(output_size[0]), int(output_size[1]))
        self._capacity = capacity
        self._crops: dict[str, CropRegion | None] = {}
        for camera_id in normalized_ids:
            crop = None if crops is None else crops.get(camera_id)
            if crop is not None and not isinstance(crop, CropRegion):
                crop = CropRegion(*crop)
            self._crops[camera_id] = crop
        self._retain_source = bool(retain_source)
        self._frames: dict[str, list[BufferedCameraFrame]] = {
            camera_id: [] for camera_id in normalized_ids
        }
        self._lock = RLock()

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return self._camera_ids

    @property
    def output_size(self) -> tuple[int, int]:
        return self._output_size

    def __len__(self) -> int:
        with self._lock:
            return sum(len(frames) for frames in self._frames.values())

    def count(self, camera_id: str) -> int:
        with self._lock:
            return len(self._camera_frames(camera_id))

    def clear(self) -> None:
        with self._lock:
            for frames in self._frames.values():
                frames.clear()

    def add_frame(
        self,
        camera_id: str,
        timestamp: Any,
        image: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> BufferedCameraFrame:
        """Process and retain one frame, returning the retained object."""

        if camera_id not in self._frames:
            raise CameraBufferError(f"unknown camera ID {camera_id!r}")
        timestamp_ns = timestamp_to_ns(timestamp)
        if timestamp_ns < 0:
            raise CameraBufferError("camera timestamp must be non-negative")

        source = np.asarray(image)
        if source.ndim not in (2, 3):
            raise CameraBufferError("image must be HxW or HxWxC")
        if source.shape[0] <= 0 or source.shape[1] <= 0:
            raise CameraBufferError("image dimensions must be non-empty")

        source_shape = tuple(int(value) for value in source.shape)
        crop = self._crops[camera_id]
        if crop is not None:
            crop.validate_for(source_shape)
            processed_source = source[
                crop.y : crop.y + crop.height,
                crop.x : crop.x + crop.width,
                ...,
            ]
        else:
            processed_source = source

        # Skip the resize when sizes already match: an identity cv2.resize still
        # copies and filters every pixel.
        if tuple(processed_source.shape[:2]) == tuple(self._output_size):
            processed = processed_source
        else:
            processed = _resize_linear(processed_source, self._output_size)
        # A contiguous owning copy prevents a view from keeping the source image
        # alive and prevents later caller mutation from altering the sample.
        processed = np.array(processed, copy=True, order="C")
        processed.setflags(write=False)
        retained = None
        if self._retain_source:
            # An owning copy: the driver reuses its buffer for the next frame.
            retained = np.array(source, copy=True, order="C")
            retained.setflags(write=False)

        frame = BufferedCameraFrame(
            camera_id=camera_id,
            timestamp_ns=timestamp_ns,
            image=processed,
            source_image=retained,
            source_shape=source_shape,
            metadata=_freeze_metadata(metadata or {}),
        )

        with self._lock:
            frames = self._frames[camera_id]
            position = bisect_right(
                [candidate.timestamp_ns for candidate in frames], timestamp_ns
            )
            frames.insert(position, frame)
            if len(frames) > self._capacity:
                del frames[: len(frames) - self._capacity]
        return frame

    # Alias for callback adapters.
    push = add_frame

    def closest_frame(
        self,
        camera_id: str,
        target_timestamp: Any,
        *,
        max_delta_ns: int | None = None,
    ) -> BufferedCameraFrame | None:
        target_ns = timestamp_to_ns(target_timestamp)
        if max_delta_ns is not None and max_delta_ns < 0:
            raise CameraBufferError("max_delta_ns must be non-negative")
        with self._lock:
            frames = self._camera_frames(camera_id)
            frame = _closest(frames, target_ns)
        if frame is None:
            return None
        if max_delta_ns is not None and abs(frame.timestamp_ns - target_ns) > max_delta_ns:
            return None
        return frame

    def closest_frames(
        self,
        target_timestamp: Any,
        *,
        max_delta_ns: int,
        max_camera_skew_ns: int | None = None,
    ) -> CameraFrameSet | None:
        """Select the closest acceptable frame from each enabled camera.

        Every frame must be within ``max_delta_ns`` of the target. When two
        cameras are enabled, ``max_camera_skew_ns`` additionally limits their
        mutual timestamp skew. No fallback to an out-of-window frame occurs.
        """

        if max_delta_ns < 0:
            raise CameraBufferError("max_delta_ns must be non-negative")
        if max_camera_skew_ns is not None and max_camera_skew_ns < 0:
            raise CameraBufferError("max_camera_skew_ns must be non-negative")
        target_ns = timestamp_to_ns(target_timestamp)
        with self._lock:
            selected = tuple(
                _closest(self._frames[camera_id], target_ns)
                for camera_id in self._camera_ids
            )
        if any(frame is None for frame in selected):
            return None
        frames = tuple(frame for frame in selected if frame is not None)
        if any(abs(frame.timestamp_ns - target_ns) > max_delta_ns for frame in frames):
            return None
        frame_set = CameraFrameSet(target_ns, frames)
        skew_ns = frame_set.camera_skew_ns
        if (
            max_camera_skew_ns is not None
            and skew_ns is not None
            and skew_ns > max_camera_skew_ns
        ):
            return None
        return frame_set

    closest_pair = closest_frames


    def _camera_frames(self, camera_id: str) -> list[BufferedCameraFrame]:
        try:
            return self._frames[camera_id]
        except KeyError as exc:
            raise CameraBufferError(f"unknown camera ID {camera_id!r}") from exc


# Aliases used by two-camera callers.
CameraFramePair = CameraFrameSet
TwoCameraBuffer = CameraBuffer
TwoCameraFrameBuffer = CameraBuffer


def _closest(
    frames: Sequence[BufferedCameraFrame], target_timestamp_ns: int
) -> BufferedCameraFrame | None:
    if not frames:
        return None
    timestamps = [frame.timestamp_ns for frame in frames]
    insertion = bisect_right(timestamps, target_timestamp_ns)
    candidates: list[BufferedCameraFrame] = []
    if insertion > 0:
        candidates.append(frames[insertion - 1])
    if insertion < len(frames):
        candidates.append(frames[insertion])
    # The timestamp secondary key chooses the older frame for an exact tie.
    return min(
        candidates,
        key=lambda frame: (abs(frame.timestamp_ns - target_timestamp_ns), frame.timestamp_ns),
    )


def _resize_linear(image: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    """OpenCV ``INTER_LINEAR`` resize; the source is not retained."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - declared runtime dependency
        raise CameraBufferError("OpenCV is required to resize RGB frames") from exc
    output_height, output_width = output_size
    return cv2.resize(
        image,
        (output_width, output_height),
        interpolation=cv2.INTER_LINEAR,
    )


def _freeze_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(metadata, Mapping):
        raise CameraBufferError("metadata must be a mapping")

    def freeze(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            return value
        if isinstance(value, Mapping):
            return MappingProxyType({str(key): freeze(item) for key, item in value.items()})
        if isinstance(value, (tuple, list)):
            return tuple(freeze(item) for item in value)
        # Metadata must not become an accidental path for retaining an image.
        if isinstance(value, np.ndarray):
            raise CameraBufferError("array-valued metadata is not retained")
        raise CameraBufferError(
            f"unsupported metadata value type {type(value).__name__}"
        )

    return MappingProxyType({str(key): freeze(value) for key, value in metadata.items()})


__all__ = [
    "BufferedCameraFrame",
    "CameraBufferError",
    "CameraBuffer",
    "CameraFrameSet",
    "CameraFramePair",
    "CropRegion",
    "TwoCameraBuffer",
    "TwoCameraFrameBuffer",
]
