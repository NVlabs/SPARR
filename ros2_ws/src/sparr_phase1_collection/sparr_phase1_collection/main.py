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
