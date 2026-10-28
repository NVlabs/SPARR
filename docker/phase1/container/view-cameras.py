#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA OneWay Noncommercial License (March 22, 2022).
# See the LICENSE file in the repository root for the full license text:
# https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf

"""Live side-by-side view of every connected RealSense colour stream.

Opens the devices directly; a camera that never delivers shows its last error
instead of hanging the window.

Usage (from the repo root, with a display available):

    docker/phase1/session.sh view

or directly:

    python3 docker/phase1/container/view-cameras.py [--width 640] [--height 480]
                                                    [--fps 30] [--serial SN ...]

Press q or Esc to quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Optional

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit("pyrealsense2 is not available; run this inside the Phase 1 image")

try:
    import cv2
except ImportError:
    sys.exit("OpenCV is not available; run this inside the Phase 1 image")

PANEL_LABEL_HEIGHT = 52


class Camera:
    """One device's colour stream, with its own failure state."""

    def __init__(self, device: Any, width: int, height: int, fps: int) -> None:
        self.serial = device.get_info(rs.camera_info.serial_number)
        try:
            self.usb = device.get_info(rs.camera_info.usb_type_descriptor)
        except Exception:
            self.usb = "?"
        self.width, self.height = width, height
        self.frames = 0
        self.error: Optional[str] = None
        self.last_stamp = 0.0
        self.rate = 0.0
        self._pipeline: Any = None
        self._started = 0.0

        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        try:
            self._pipeline = rs.pipeline()
            self._pipeline.start(config)
            self._started = time.monotonic()
        except Exception as exc:
            self.error = f"start failed: {exc}"
            self._pipeline = None

    def read(self) -> Optional[np.ndarray]:
        """Latest frame as BGR, or None if this camera is not delivering."""
        if self._pipeline is None:
            return None
        try:
            # Short poll so one dead camera does not stall the other panels.
            ok, frames = self._pipeline.try_wait_for_frames(60)
            if not ok:
                return None
            frame = frames.get_color_frame()
            if not frame:
                return None
            self.frames += 1
            now = time.monotonic()
            elapsed = now - self._started
            self.rate = self.frames / elapsed if elapsed > 0 else 0.0
            self.error = None
            rgb = np.asanyarray(frame.get_data())
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            self.error = str(exc)[:60]
            return None

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass

    def panel(self, image: Optional[np.ndarray]) -> np.ndarray:
        """A labelled panel, even when there is no image to show."""
        if image is None:
            image = np.zeros((self.height, self.width, 3), np.uint8)
            message = self.error or "no frames"
            cv2.putText(image, message[:48], (12, self.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
        canvas = np.zeros((self.height + PANEL_LABEL_HEIGHT, self.width, 3), np.uint8)
        canvas[PANEL_LABEL_HEIGHT:] = image
        healthy = self.frames > 0 and self.error is None
        colour = (0, 220, 0) if healthy else (0, 0, 255)
        cv2.putText(canvas, f"{self.serial}  USB {self.usb}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{self.frames} frames   {self.rate:.1f} fps", (10, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serial", action="append", default=None,
                        help="limit to these serials (repeatable)")
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="exit after N seconds; 0 means run until q/Esc")
    args = parser.parse_args()

    devices = list(rs.context().devices)
    if args.serial:
        devices = [
            d for d in devices
            if d.get_info(rs.camera_info.serial_number) in set(args.serial)
        ]
    if not devices:
        return int(bool(print("no RealSense devices found")))

    cameras = [Camera(d, args.width, args.height, args.fps) for d in devices]
    for camera in cameras:
        state = camera.error or "started"
        print(f"{camera.serial}  USB {camera.usb}  {args.width}x{args.height}"
              f"@{args.fps}  {state}")
    print("\nq or Esc to quit\n")

    window = "SPARR cameras"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    deadline = time.monotonic() + args.seconds if args.seconds else None
    latest: dict[str, Optional[np.ndarray]] = {c.serial: None for c in cameras}
    try:
        while True:
            for camera in cameras:
                image = camera.read()
                if image is not None:
                    latest[camera.serial] = image
            cv2.imshow(window, np.hstack([c.panel(latest[c.serial]) for c in cameras]))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            if deadline and time.monotonic() > deadline:
                break
    except KeyboardInterrupt:
        pass
    finally:
        for camera in cameras:
            camera.stop()
        cv2.destroyAllWindows()

    print()
    for camera in cameras:
        verdict = "OK" if camera.frames else f"NO FRAMES ({camera.error or 'none'})"
        print(f"{camera.serial}: {camera.frames} frames, {camera.rate:.1f} fps -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
