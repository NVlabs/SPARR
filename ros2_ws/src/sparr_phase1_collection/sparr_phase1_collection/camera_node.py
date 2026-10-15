"""RealSense colour publisher built directly on the librealsense Python API.

Publishes the single colour stream Phase 1 needs without ``realsense2_camera``.
Third-party imports are guarded and message/profile building is pure, so the
logic is unit-testable without hardware. The requested profile is validated up
front with no fallback, because the collector rejects frames whose
encoding/size/frame_id differ from its config.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

try:  # pragma: no cover - requires librealsense
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - the unit-test image has no librealsense
    rs = None

try:  # pragma: no cover - exercised only in a sourced ROS environment
    import rclpy
    from rclpy.node import Node as _RclpyNode
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
        qos_profile_sensor_data,
    )
except ImportError:  # pragma: no cover
    rclpy = None
    _RclpyNode = object
    DurabilityPolicy = HistoryPolicy = QoSProfile = ReliabilityPolicy = None
    qos_profile_sensor_data = None

try:  # pragma: no cover - exercised only in a sourced ROS environment
    from builtin_interfaces.msg import Time as TimeMessage
    from sensor_msgs.msg import CameraInfo, Image
except ImportError:  # pragma: no cover
    TimeMessage = None
    CameraInfo = None
    Image = None


DEFAULT_IMAGE_TOPIC = "/camera/camera/color/image_rect_raw"
DEFAULT_CAMERA_INFO_TOPIC = "/camera/camera/color/camera_info"
DEFAULT_FRAME_ID = "camera_color_optical_frame"
DEFAULT_WIDTH = 848
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30
DEFAULT_ENCODING = "rgb8"
DEFAULT_QOS = "reliable"

#: ROS ``sensor_msgs/Image`` encoding -> (librealsense format name, bytes/pixel).
ENCODING_TO_RS_FORMAT: dict[str, tuple[str, int]] = {
    "rgb8": ("rgb8", 3),
    "bgr8": ("bgr8", 3),
    "rgba8": ("rgba8", 4),
    "bgra8": ("bgra8", 4),
    "mono8": ("y8", 1),
}

#: librealsense ``rs.distortion`` enum name -> ROS ``distortion_model`` name.
#: Unmapped models (e.g. ``kannala_brandt4``) are published under their
#: librealsense name rather than mislabelled.
RS_TO_ROS_DISTORTION: dict[str, str] = {
    "none": "plumb_bob",
    "brown_conrady": "plumb_bob",
    "modified_brown_conrady": "plumb_bob",
    "inverse_brown_conrady": "plumb_bob",
    "ftheta": "equidistant",
}

#: Frame timestamp domains that are already expressed on the host epoch and can
#: therefore be compared against the ROS clock.
EPOCH_TIMESTAMP_DOMAINS = frozenset({"global_time", "system_time"})

#: If a hardware-derived stamp disagrees with the arrival time by more than
#: this, distrust the domain and use the ROS clock instead.
MAX_TIMESTAMP_SKEW_NS = 5_000_000_000


class CameraNodeError(RuntimeError):
    """Raised for configuration problems that must not be silently absorbed."""


class ProfileNotSupportedError(CameraNodeError):
    """Raised when the device cannot deliver the requested colour profile."""


@dataclass(frozen=True)
class StreamProfileSpec:
    """A colour stream profile the device advertises."""

    width: int
    height: int
    fps: int
    format: str

    def describe(self) -> str:
        return f"{self.width}x{self.height} @ {self.fps}Hz {self.format}"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def rs_format_for_encoding(encoding: str) -> tuple[str, int]:
    """Return the librealsense format name and bytes-per-pixel for *encoding*."""

    key = str(encoding).strip().lower()
    if key not in ENCODING_TO_RS_FORMAT:
        supported = ", ".join(sorted(ENCODING_TO_RS_FORMAT))
        raise CameraNodeError(
            f"encoding {encoding!r} is not supported by this node; supported: {supported}"
        )
    return ENCODING_TO_RS_FORMAT[key]


def select_color_profile(
    available: Iterable[StreamProfileSpec],
    *,
    width: int,
    height: int,
    fps: int,
    format_name: str,
) -> StreamProfileSpec:
    """Return the exact matching profile or raise listing what is available.

    No nearest-match fallback: the collector drops frames whose size or encoding
    differs from its config.
    """

    profiles = list(available)
    wanted = StreamProfileSpec(
        int(width), int(height), int(fps), str(format_name).strip().lower()
    )
    for profile in profiles:
        if (
            int(profile.width) == wanted.width
            and int(profile.height) == wanted.height
            and int(profile.fps) == wanted.fps
            and str(profile.format).strip().lower() == wanted.format
        ):
            return profile
    if profiles:
        listing = "; ".join(
            sorted({profile.describe() for profile in profiles})
        )
    else:
        listing = "<device advertised no colour profiles>"
    raise ProfileNotSupportedError(
        f"RealSense colour profile {wanted.describe()} is not supported. "
        f"Available colour profiles: {listing}"
    )


def ros_distortion_model(model_name: Any) -> str:
    """Map a librealsense distortion enum (or its name) to a ROS model name."""

    name = getattr(model_name, "name", model_name)
    key = str(name).strip().lower()
    if key.startswith("distortion."):
        key = key.split(".", 1)[1]
    # Unmapped librealsense models keep their own name; see RS_TO_ROS_DISTORTION.
    return RS_TO_ROS_DISTORTION.get(key, key)


def resolve_frame_stamp_ns(
    *,
    domain: Any,
    frame_timestamp_ms: Optional[float],
    ros_now_ns: int,
) -> tuple[int, str]:
    """Choose the stamp for a frame and report which clock it came from.

    The collector matches frames to robot state by ``header.stamp``, so the stamp
    should be capture time. ``global_time`` (device clock fitted to the host) and
    ``system_time`` (host epoch at arrival) are used directly; ``hardware_clock``
    has an unknown epoch, so the ROS clock at arrival is used instead. A stamp
    off from arrival by more than :data:`MAX_TIMESTAMP_SKEW_NS` is distrusted.
    """

    name = getattr(domain, "name", domain)
    key = str(name).strip().lower()
    if key.startswith("timestamp_domain."):
        key = key.split(".", 1)[1]
    if key in EPOCH_TIMESTAMP_DOMAINS and frame_timestamp_ms is not None:
        try:
            candidate = int(round(float(frame_timestamp_ms) * 1_000_000.0))
        except (TypeError, ValueError, OverflowError):
            return int(ros_now_ns), "ros_clock"
        if candidate > 0 and abs(candidate - int(ros_now_ns)) <= MAX_TIMESTAMP_SKEW_NS:
            return candidate, key
    return int(ros_now_ns), "ros_clock"


def build_stamp(nanoseconds: int, *, stamp_factory: Optional[Callable[[], Any]] = None) -> Any:
    """Build a ``builtin_interfaces/Time`` (or stub) from nanoseconds."""

    factory = stamp_factory or TimeMessage
    if factory is None:  # pragma: no cover - ROS-less environments must pass one
        raise CameraNodeError("builtin_interfaces is unavailable; pass stamp_factory")
    total = int(nanoseconds)
    stamp = factory()
    stamp.sec = total // 1_000_000_000
    stamp.nanosec = total % 1_000_000_000
    return stamp


def build_camera_info(
    intrinsics: Any,
    *,
    frame_id: str,
    stamp: Any,
    message_factory: Optional[Callable[[], Any]] = None,
) -> Any:
    """Build a ``CameraInfo`` from real librealsense intrinsics.

    ``intrinsics`` is duck-typed on ``rs.intrinsics``: ``width``, ``height``,
    ``fx``, ``fy``, ``ppx``, ``ppy``, ``model``, ``coeffs``. The values are
    recorded as calibration, so identity placeholders are never acceptable.
    """

    factory = message_factory or CameraInfo
    if factory is None:  # pragma: no cover
        raise CameraNodeError("sensor_msgs is unavailable; pass message_factory")

    fx = float(intrinsics.fx)
    fy = float(intrinsics.fy)
    cx = float(intrinsics.ppx)
    cy = float(intrinsics.ppy)
    if not (fx > 0.0 and fy > 0.0):
        raise CameraNodeError(
            f"RealSense reported non-positive focal lengths fx={fx}, fy={fy}; "
            "the device calibration is unusable"
        )

    message = factory()
    message.header.stamp = stamp
    message.header.frame_id = str(frame_id)
    message.width = int(intrinsics.width)
    message.height = int(intrinsics.height)
    message.distortion_model = ros_distortion_model(getattr(intrinsics, "model", "none"))
    message.d = [float(value) for value in getattr(intrinsics, "coeffs", ()) or ()]
    message.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    # The monocular colour stream is published unrectified, so R is
    # identity and P is K with a zero translation column.
    message.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    message.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    message.binning_x = 0
    message.binning_y = 0
    return message


def build_image(
    array: Any,
    *,
    frame_id: str,
    stamp: Any,
    encoding: str = DEFAULT_ENCODING,
    message_factory: Optional[Callable[[], Any]] = None,
) -> Any:
    """Build a ``sensor_msgs/Image`` directly from an ``rgb8`` frame buffer.

    The frame already arrives in the requested layout: shape check plus copy.
    """

    factory = message_factory or Image
    if factory is None:  # pragma: no cover
        raise CameraNodeError("sensor_msgs is unavailable; pass message_factory")

    _, bytes_per_pixel = rs_format_for_encoding(encoding)
    frame = np.asanyarray(array)
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8, copy=False)
    if frame.ndim == 2:
        channels = 1
        height, width = frame.shape
    elif frame.ndim == 3:
        height, width, channels = frame.shape
    else:
        raise CameraNodeError(f"frame has unsupported shape {frame.shape!r}")
    if channels != bytes_per_pixel:
        raise CameraNodeError(
            f"frame has {channels} channel(s) but encoding {encoding!r} needs "
            f"{bytes_per_pixel}"
        )

    message = factory()
    message.header.stamp = stamp
    message.header.frame_id = str(frame_id)
    message.height = int(height)
    message.width = int(width)
    message.encoding = str(encoding)
    message.is_bigendian = 0
    message.step = int(width) * bytes_per_pixel
    message.data = np.ascontiguousarray(frame).tobytes()
    return message


def build_frame_messages(
    array: Any,
    intrinsics: Any,
    *,
    frame_id: str,
    stamp: Any,
    encoding: str = DEFAULT_ENCODING,
    image_factory: Optional[Callable[[], Any]] = None,
    camera_info_factory: Optional[Callable[[], Any]] = None,
) -> tuple[Any, Any]:
    """Build the Image/CameraInfo pair for one frame with an identical stamp."""

    image = build_image(
        array,
        frame_id=frame_id,
        stamp=stamp,
        encoding=encoding,
        message_factory=image_factory,
    )
    info = build_camera_info(
        intrinsics,
        frame_id=frame_id,
        stamp=stamp,
        message_factory=camera_info_factory,
    )
    return image, info


def qos_profile_for(name: str) -> Any:
    """Mirror ``collector_node._camera_qos`` so publisher and subscriber match."""

    if QoSProfile is None:  # pragma: no cover
        raise CameraNodeError("rclpy is unavailable; cannot build a QoS profile")
    normalized = str(name).strip().lower()
    if normalized == "sensor_data":
        return qos_profile_sensor_data
    reliability = (
        ReliabilityPolicy.BEST_EFFORT
        if normalized == "best_effort"
        else ReliabilityPolicy.RELIABLE
    )
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
        reliability=reliability,
        durability=DurabilityPolicy.VOLATILE,
    )


# ---------------------------------------------------------------------------
# Thin pyrealsense2 wrapper
# ---------------------------------------------------------------------------


def enumerate_color_profiles(device: Any) -> list[StreamProfileSpec]:
    """List the colour profiles a librealsense device advertises."""

    profiles: list[StreamProfileSpec] = []
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() != rs.stream.color:
                continue
            if not profile.is_video_stream_profile():
                continue
            video = profile.as_video_stream_profile()
            profiles.append(
                StreamProfileSpec(
                    width=int(video.width()),
                    height=int(video.height()),
                    fps=int(profile.fps()),
                    format=str(profile.format()).split(".")[-1].lower(),
                )
            )
    return profiles


class RealSenseColorStream:
    """Minimal ``rs.pipeline`` wrapper: start, blocking read, stop."""

    def __init__(
        self,
        *,
        serial: str = "",
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fps: int = DEFAULT_FPS,
        encoding: str = DEFAULT_ENCODING,
    ) -> None:
        if rs is None:  # pragma: no cover
            raise CameraNodeError(
                "pyrealsense2 is not installed; install librealsense Python bindings"
            )
        self.serial = str(serial or "")
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.encoding = str(encoding)
        self._format_name, _ = rs_format_for_encoding(encoding)
        self._pipeline: Any = None
        self._profile: Any = None
        self._intrinsics: Any = None

    # -- lifecycle ------------------------------------------------------

    def _resolve_device(self) -> Any:
        devices = list(rs.context().devices)
        if not devices:
            raise CameraNodeError("no RealSense device is connected")
        if not self.serial:
            return devices[0]
        for device in devices:
            if str(device.get_info(rs.camera_info.serial_number)) == self.serial:
                return device
        found = ", ".join(
            str(device.get_info(rs.camera_info.serial_number)) for device in devices
        )
        raise CameraNodeError(
            f"RealSense serial {self.serial!r} not found; connected serials: {found}"
        )

    def start(self) -> Any:
        device = self._resolve_device()
        serial = str(device.get_info(rs.camera_info.serial_number))
        # Fail loudly before pipeline.start(); librealsense would otherwise
        # raise a generic "couldn't resolve requests" with no diagnostics.
        select_color_profile(
            enumerate_color_profiles(device),
            width=self.width,
            height=self.height,
            fps=self.fps,
            format_name=self._format_name,
        )
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            getattr(rs.format, self._format_name),
            self.fps,
        )
        self._pipeline = rs.pipeline()
        self._profile = self._pipeline.start(config)
        self.serial = serial
        color_stream = self._profile.get_stream(rs.stream.color)
        self._intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        return self._intrinsics

    @property
    def intrinsics(self) -> Any:
        if self._intrinsics is None:
            raise CameraNodeError("stream has not been started")
        return self._intrinsics

    def read(self, timeout_ms: int = 5000) -> Optional[Any]:
        """Block for the next colour frame; ``None`` if the frameset is empty."""

        frames = self._pipeline.wait_for_frames(timeout_ms)
        color = frames.get_color_frame()
        if not color:
            return None
        return color

    def stop(self) -> None:
        pipeline, self._pipeline = self._pipeline, None
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:  # pragma: no cover - already stopped/unplugged
                pass


def frame_timestamp_domain_name(frame: Any) -> str:
    """Return the frame's timestamp domain as a lower-case name."""

    try:
        domain = frame.get_frame_timestamp_domain()
    except Exception:  # pragma: no cover - older librealsense builds
        return "unknown"
    return str(getattr(domain, "name", domain)).split(".")[-1].lower()


# ---------------------------------------------------------------------------
# ROS node
# ---------------------------------------------------------------------------


class RealSenseCameraNode(_RclpyNode):
    """Publish RealSense colour frames as Image + CameraInfo.

    One instance drives one camera (``-p camera:=wrist_1`` / ``wrist_2``), reading
    its serial, topics, resolution and encoding from the collection config.
    """

    @staticmethod
    def _defaults_from_config(config_path: str, camera_name: str) -> dict[str, Any]:
        """Read one camera's settings out of the collection config.

        The serial and topics must come from the config: with two cameras,
        built-in defaults would make both instances claim the same device.
        """
        if not config_path:
            if camera_name:
                raise CameraNodeError(
                    f"camera:={camera_name!r} was requested but config_path is "
                    "unset; the camera's settings can only come from the config"
                )
            return {}

        from .configuration import CollectionConfig

        config = CollectionConfig.load(config_path)
        enabled = config.enabled_cameras
        if not enabled:
            raise CameraNodeError(f"no camera is enabled in {config_path}")

        if camera_name:
            matches = [c for c in config.cameras if c.name == camera_name]
            if not matches:
                known = ", ".join(c.name for c in config.cameras)
                raise CameraNodeError(
                    f"unknown camera {camera_name!r}; the config defines: {known}"
                )
            camera = matches[0]
            if not camera.enabled:
                raise CameraNodeError(
                    f"camera {camera_name!r} is disabled in {config_path}; enable it "
                    "there rather than starting a driver the collector will ignore"
                )
        else:
            # No camera named: only unambiguous with exactly one enabled.
            if len(enabled) != 1:
                names = ", ".join(c.name for c in enabled)
                raise CameraNodeError(
                    f"{len(enabled)} cameras are enabled ({names}); pass "
                    "-p camera:=<name> to say which one this driver is"
                )
            camera = enabled[0]

        settings: dict[str, Any] = {
            # Shared by every camera, so the pair cannot drift into different
            # clock domains -- see SynchronizationConfig.camera_timestamp_source.
            "timestamp_source": str(
                config.synchronization.camera_timestamp_source or "auto"
            ),
        }
        for key, attribute in (
            ("serial", "serial"),
            ("image_topic", "image_topic"),
            ("camera_info_topic", "camera_info_topic"),
            ("frame_id", "frame_id"),
            ("encoding", "encoding"),
            ("qos", "qos"),
        ):
            value = getattr(camera, attribute, None)
            if value:
                settings[key] = str(value)
        if camera.input_width:
            settings["width"] = int(camera.input_width)
        if camera.input_height:
            settings["height"] = int(camera.input_height)
        if camera.fps:
            settings["fps"] = int(camera.fps)
        return settings

    def __init__(self, stream_factory: Optional[Callable[..., Any]] = None) -> None:
        super().__init__("sparr_phase1_camera")

        # One pipeline per process, so one camera failing or being re-plugged
        # cannot take the other down.
        self.declare_parameter("config_path", "")
        self.declare_parameter("camera", "")

        self._camera_name = str(self.get_parameter("camera").value).strip()
        defaults = self._defaults_from_config(
            str(self.get_parameter("config_path").value).strip(),
            self._camera_name,
        )

        # Declared after the config is read; precedence is --ros-args -p, then
        # the config block, then the built-in default.
        self.declare_parameter("serial", defaults.get("serial", ""))
        self.declare_parameter("image_topic", defaults.get("image_topic", DEFAULT_IMAGE_TOPIC))
        self.declare_parameter(
            "camera_info_topic",
            defaults.get("camera_info_topic", DEFAULT_CAMERA_INFO_TOPIC),
        )
        self.declare_parameter("frame_id", defaults.get("frame_id", DEFAULT_FRAME_ID))
        self.declare_parameter("width", defaults.get("width", DEFAULT_WIDTH))
        self.declare_parameter("height", defaults.get("height", DEFAULT_HEIGHT))
        self.declare_parameter("fps", defaults.get("fps", DEFAULT_FPS))
        self.declare_parameter("encoding", defaults.get("encoding", DEFAULT_ENCODING))
        self.declare_parameter("qos", defaults.get("qos", DEFAULT_QOS))
        self.declare_parameter("frame_timeout_ms", 5000)
        self.declare_parameter(
            "timestamp_source", defaults.get("timestamp_source", "auto")
        )

        def _get(name: str) -> Any:
            return self.get_parameter(name).value

        self._frame_id = str(_get("frame_id"))
        self._encoding = str(_get("encoding"))
        self._frame_timeout_ms = int(_get("frame_timeout_ms"))

        qos = qos_profile_for(str(_get("qos")))
        self._image_publisher = self.create_publisher(Image, str(_get("image_topic")), qos)
        self._info_publisher = self.create_publisher(
            CameraInfo, str(_get("camera_info_topic")), qos
        )

        factory = stream_factory or RealSenseColorStream
        self._stream = factory(
            serial=str(_get("serial")),
            width=int(_get("width")),
            height=int(_get("height")),
            fps=int(_get("fps")),
            encoding=self._encoding,
        )
        self._intrinsics = self._stream.start()
        self.get_logger().info(
            f"[{self._camera_name or 'unnamed'}] RealSense "
            f"{self._stream.serial or '<first device>'} streaming "
            f"{_get('width')}x{_get('height')} @ {_get('fps')}Hz {self._encoding} "
            f"as {_get('image_topic')}"
        )

        self._stop_event = threading.Event()
        self._timestamp_source = ""
        self._required_timestamp_source = str(
            _get("timestamp_source") or "auto"
        ).strip().lower()
        if self._required_timestamp_source not in {
            "auto", "global_time", "system_time", "ros_clock"
        }:
            raise CameraNodeError(
                "synchronization.camera_timestamp_source must be one of "
                "auto, global_time, system_time, ros_clock; got "
                f"{self._required_timestamp_source!r}"
            )
        # ``wait_for_frames`` blocks for up to a frame period, so capture runs off
        # the executor; rclpy publishers are thread-safe.
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="realsense_capture", daemon=True
        )
        self._capture_thread.start()

    # -- capture --------------------------------------------------------

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._stream.read(self._frame_timeout_ms)
            except Exception as exc:  # pragma: no cover - hardware failure path
                if self._stop_event.is_set():
                    return
                self.get_logger().warning(f"RealSense frame read failed: {exc}")
                continue
            if frame is None:
                continue
            try:
                self._publish_frame(frame)
            except Exception as exc:  # pragma: no cover - hardware failure path
                self.get_logger().warning(f"Dropping RealSense frame: {exc}")

    def _publish_frame(self, frame: Any) -> None:
        ros_now_ns = self.get_clock().now().nanoseconds
        try:
            timestamp_ms = float(frame.get_timestamp())
        except Exception:  # pragma: no cover - older librealsense builds
            timestamp_ms = None
        stamp_ns, source = resolve_frame_stamp_ns(
            domain=frame_timestamp_domain_name(frame),
            frame_timestamp_ms=timestamp_ms,
            ros_now_ns=ros_now_ns,
        )
        if self._required_timestamp_source != "auto" and source != self._required_timestamp_source:
            # A per-camera fallback could leave two cameras on different clocks,
            # making the pair-skew check meaningless.
            raise CameraNodeError(
                f"camera {self._camera_name or self._frame_id!r} could not stamp frames "
                f"with the configured {self._required_timestamp_source!r} domain "
                f"(librealsense offered {source!r}). Either fix the device/driver or set "
                "synchronization.camera_timestamp_source to a domain every camera can "
                "deliver; do not leave the cameras on different clocks."
            )
        if source != self._timestamp_source:
            previous = self._timestamp_source
            self._timestamp_source = source
            if previous:
                # A mid-run change silently reinterprets every later stamp.
                self.get_logger().warning(
                    f"RealSense frame stamp domain changed {previous!r} -> {source!r}; "
                    "camera pair skew before and after this point are not comparable"
                )
            else:
                self.get_logger().info(f"RealSense frame stamps sourced from {source!r}")

        stamp = build_stamp(stamp_ns)
        array = np.asanyarray(frame.get_data())
        # One stamp object is shared by both messages, so the collector's
        # stamp-keyed matching always sees an exact pair.
        image, info = build_frame_messages(
            array,
            self._intrinsics,
            frame_id=self._frame_id,
            stamp=stamp,
            encoding=self._encoding,
        )
        self._image_publisher.publish(image)
        self._info_publisher.publish(info)

    # -- shutdown -------------------------------------------------------

    def stop(self) -> None:
        # rclpy calls destroy_node() even when __init__ raised part-way, so
        # every attribute here is optional; otherwise an AttributeError would
        # hide the real startup failure.
        stop_event = getattr(self, "_stop_event", None)
        if stop_event is not None:
            stop_event.set()
        thread = getattr(self, "_capture_thread", None)
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        stream = getattr(self, "_stream", None)
        if stream is not None:
            stream.stop()

    def destroy_node(self) -> bool:
        self.stop()
        return super().destroy_node()


def main(argv: Optional[Sequence[str]] = None) -> int:
    if rclpy is None:  # pragma: no cover
        raise CameraNodeError("rclpy is unavailable; source a ROS 2 environment")
    rclpy.init(args=list(argv) if argv is not None else None)
    node: Optional[RealSenseCameraNode] = None
    try:
        node = RealSenseCameraNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


__all__ = [
    "CameraNodeError",
    "ProfileNotSupportedError",
    "RealSenseCameraNode",
    "RealSenseColorStream",
    "StreamProfileSpec",
    "build_camera_info",
    "build_frame_messages",
    "build_image",
    "build_stamp",
    "main",
    "qos_profile_for",
    "resolve_frame_stamp_ns",
    "ros_distortion_model",
    "rs_format_for_encoding",
    "select_color_profile",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
