# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA License.
# See the LICENSE file in the repository root for the full license text.

"""Console entry point for the Phase 2 residual collector (ROS parameters, Ctrl-C to stop)."""

from __future__ import annotations

from typing import Optional, Sequence

import rclpy

from .collector_node import Phase2CollectorNode


def collector_main(argv: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=argv)
    node = None
    try:
        node = Phase2CollectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            # Close the session cleanly so the HDF5 file is not left partial.
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    collector_main()
