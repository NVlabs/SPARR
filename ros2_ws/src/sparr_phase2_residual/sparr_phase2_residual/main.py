# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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
