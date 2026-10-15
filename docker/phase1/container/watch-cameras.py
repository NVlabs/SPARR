#!/usr/bin/env python3
"""Live view of the camera topics, safe to run during a collection session.

Subscribes to the drivers' image topics instead of opening the devices.

Usage:

    docker/phase1/session.sh watch                 # every enabled camera
    docker/phase1/session.sh watch wrist_1         # just one

Press q or Esc to quit. Closing the window also quits.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import time

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    sys.exit("OpenCV is not available; run this inside the Phase 1 image")

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
except ImportError:  # pragma: no cover
    sys.exit("rclpy/sensor_msgs unavailable; run this inside a sourced ROS env")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "ros2_ws" / "src" / "sparr_phase1_collection"))
from sparr_phase1_collection.configuration import CollectionConfig  # noqa: E402

WINDOW = "sparr camera topics"


def _to_bgr(msg: Image) -> np.ndarray:
    """Decode a sensor_msgs/Image without cv_bridge."""
    encoding = (msg.encoding or "").lower()
    buffer = np.frombuffer(msg.data, dtype=np.uint8)
    if encoding in ("rgb8", "bgr8"):
        frame = buffer.reshape(msg.height, msg.width, 3)
        return frame[..., ::-1] if encoding == "rgb8" else frame
    if encoding in ("rgba8", "bgra8"):
        frame = buffer.reshape(msg.height, msg.width, 4)[..., :3]
        return frame[..., ::-1] if encoding == "rgba8" else frame
    if encoding == "mono8":
        return cv2.cvtColor(buffer.reshape(msg.height, msg.width), cv2.COLOR_GRAY2BGR)
    raise ValueError(f"unsupported encoding {msg.encoding!r}")


class Watcher(Node):
    def __init__(self, cameras, panel_px: int):
        super().__init__("sparr_phase1_watch")
        self._panel = panel_px
        self._latest: dict[str, np.ndarray] = {}
        self._stamp_ns: dict[str, int] = {}
        self._count: dict[str, int] = {name: 0 for name, _ in cameras}
        self._error: dict[str, str] = {}
        for name, camera in cameras:
            # Best-effort, depth 1: no backpressure on the driver, no stale queue.
            qos = QoSProfile(
                depth=1,
                history=HistoryPolicy.KEEP_LAST,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            )
            self.create_subscription(
                Image, camera.image_topic,
                lambda msg, n=name: self._on_image(n, msg), qos,
            )
            self.get_logger().info(f"watching {name} on {camera.image_topic}")
        self._names = [name for name, _ in cameras]

    def _on_image(self, name: str, msg: Image) -> None:
        try:
            self._latest[name] = _to_bgr(msg)
            self._stamp_ns[name] = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            self._count[name] += 1
            self._error.pop(name, None)
        except Exception as exc:  # noqa: BLE001 - a bad frame must not kill the view
            self._error[name] = f"{type(exc).__name__}: {exc}"

    def compose(self) -> np.ndarray:
        panels = []
        now_ns = time.time_ns()
        for name in self._names:
            frame = self._latest.get(name)
            if frame is None:
                panel = np.full((self._panel, self._panel, 3), 40, np.uint8)
                note = self._error.get(name, "waiting for frames...")
            else:
                h, w = frame.shape[:2]
                scale = self._panel / max(h, w)
                panel = cv2.resize(frame, (int(w * scale), int(h * scale)))
                pad = np.full((self._panel, self._panel, 3), 20, np.uint8)
                y = (self._panel - panel.shape[0]) // 2
                x = (self._panel - panel.shape[1]) // 2
                pad[y:y + panel.shape[0], x:x + panel.shape[1]] = panel
                panel = pad
                age_ms = (now_ns - self._stamp_ns.get(name, now_ns)) / 1e6
                note = f"{w}x{h}  n={self._count[name]}  age {age_ms:.0f} ms"
            cv2.rectangle(panel, (0, 0), (self._panel, 44), (20, 20, 20), -1)
            cv2.putText(panel, name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(panel, note, (10, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (185, 185, 185), 1, cv2.LINE_AA)
            panels.append(panel)
        return np.hstack(panels) if panels else np.zeros((240, 320, 3), np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cameras", nargs="*", help="camera names (default: all enabled)")
    parser.add_argument("--config", default=os.environ.get(
        "SPARR_CONFIG", str(REPO / "phase1_live.yaml")))
    parser.add_argument("--panel", type=int, default=480, help="panel size in pixels")
    args = parser.parse_args()

    config = CollectionConfig.load(args.config)
    enabled = {camera.name: camera for camera in config.enabled_cameras}
    if not enabled:
        sys.exit(f"no camera is enabled in {args.config}")
    names = args.cameras or list(enabled)
    unknown = [n for n in names if n not in enabled]
    if unknown:
        sys.exit(f"not enabled in {args.config}: {', '.join(unknown)}  "
                 f"(enabled: {', '.join(enabled)})")

    rclpy.init()
    node = Watcher([(n, enabled[n]) for n in names], args.panel)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    print("q or Esc to quit", flush=True)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            cv2.imshow(WINDOW, node.compose())
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
