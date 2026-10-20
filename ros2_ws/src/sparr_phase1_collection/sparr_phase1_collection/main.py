# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Console entry point for the Phase 1 collector node."""

from __future__ import annotations

from typing import Optional


def collector_main(argv: Optional[list[str]] = None) -> int:
    import rclpy

    from .collector_node import Phase1CollectorNode

    rclpy.init(args=argv)
    node = None
    try:
        node = Phase1CollectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.shutdown()
            finally:
                node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(collector_main())
