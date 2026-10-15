#!/usr/bin/env python3
"""Check that the residual's per-tick compute fits the control period.

Times residual evaluation plus action composition at the runtime image size and
network, and compares against Phase 1 timing recorded in data/phase1 when present.
Measures compute only, not scheduling under live DDS load.

Usage:  python3 /opt/sparr/verify-control-budget.py [--iterations N]
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

for candidate in (
    "/opt/sparr/ros2_ws/src/sparr_phase2_residual",
    "ros2_ws/src/sparr_phase2_residual",
    "ros2_ws/src/sparr_phase1_collection",
):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

CONTROL_RATE_HZ = 30.0
TICK_MS = 1000.0 / CONTROL_RATE_HZ
IMAGE_SIZE = 128
#: The residual should be a small share of the tick, not merely fit inside it.
MAX_SHARE_OF_TICK = 0.25

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{f': {detail}' if detail else ''}")
    if not ok:
        failures.append(name)


def measured_phase1_budget() -> dict[str, float] | None:
    """p99 inference and composition times from Phase 1 recordings, or None."""
    try:
        import h5py
    except ImportError:
        return None
    recordings = sorted(Path("data/phase1").glob("*.h5"))
    inference, composition = [], []
    for path in recordings:
        try:
            handle = h5py.File(path, "r")
        except OSError:
            continue
        with handle:
            for name in handle.get("trials", {}):
                trial = handle[f"trials/{name}"]
                if "actions/timing/inference_ns" not in trial:
                    continue
                inference.extend(np.asarray(trial["actions/timing/inference_ns"][:]))
                composition.extend(
                    np.asarray(trial["actions/timing/composition_ns"][:])
                )
    if not inference:
        return None
    inference = np.asarray(inference, np.float64)
    composition = np.asarray(composition, np.float64)
    return {
        "trials": len(recordings),
        "inference_p99_ms": float(np.percentile(inference[inference >= 0], 99)) / 1e6,
        "composition_p99_ms": float(
            np.percentile(composition[composition >= 0], 99)
        ) / 1e6,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    print(f"Control tick: {TICK_MS:.1f} ms at {CONTROL_RATE_HZ:.0f} Hz\n")

    print("== 1. what Phase 1 costs today (from real recordings)")
    baseline = measured_phase1_budget()
    if baseline is None:
        print("  no Phase 1 recordings found; skipping the baseline comparison")
    else:
        print(f"  measured over {baseline['trials']} session(s)")
        print(f"  base policy inference p99 : {baseline['inference_p99_ms']:.2f} ms")
        print(f"  action composition   p99 : {baseline['composition_p99_ms']:.2f} ms")

    print("\n== 2. the residual, at the real image size")
    from sparr_phase1_collection.action_pipeline import (
        ActionPipelineConfig,
        build_action_trace,
    )
    from sparr_phase1_collection.pose_math import Pose
    from sparr_phase2_residual.config import Phase2Config
    from sparr_phase2_residual.residual_agent import make_residual_agent
    from sparr_phase2_residual.residual_policy import ResidualPolicy

    config_path = Path("phase2_live.yaml")
    config = (
        Phase2Config.load(config_path) if config_path.is_file() else Phase2Config()
    )
    spec = config.observation
    print(f"  state {spec.state_dim()}-D, images {', '.join(spec.image_keys)}")

    agent = make_residual_agent(spec, seed=0, image_size=IMAGE_SIZE)
    policy = ResidualPolicy(
        agent, spec, seed=0, action_scale=config.runtime.action_scale
    )

    rng = np.random.default_rng(0)
    pipeline = ActionPipelineConfig(
        position_scale=(0.004, 0.004, 0.004),
        orientation_scale=(0.02, 0.02, 0.02),
        action_clip=1.0,
    )
    flange = Pose(np.array([0.47, 0.12, 0.34]), np.array([0.707, 0.707, 0.0, 0.0]))
    flange_to_tcp = Pose(np.array([0.0, 0.0, 0.15]), np.array([0.0, 0.0, 0.0, 1.0]))

    def tick() -> float:
        """Time one tick of residual evaluation plus action composition, in ms."""
        terms = {
            name: rng.normal(size=width)
            for name, width in spec.terms()
        }
        images = {
            key: rng.integers(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            for key in spec.image_keys
        }
        mean = rng.uniform(-1, 1, 6)
        sample = rng.uniform(-1, 1, 6)

        start = time.monotonic_ns()
        decision = policy.evaluate(terms=terms, images=images)
        if not decision.ok:
            raise RuntimeError(f"residual failed: {decision.error}")
        build_action_trace(
            raw_mean=mean,
            raw_sample=sample,
            observed_flange_pose=flange,
            flange_to_tcp=flange_to_tcp,
            config=pipeline,
            residual=decision.residual,
        )
        return (time.monotonic_ns() - start) / 1e6

    for _ in range(5):  # absorb tracing and compilation
        tick()
    samples = [tick() for _ in range(args.iterations)]

    median = statistics.median(samples)
    p99 = float(np.percentile(samples, 99))
    worst = max(samples)
    print(
        f"  residual + composition over {args.iterations} ticks: "
        f"median {median:.2f} ms, p99 {p99:.2f} ms, max {worst:.2f} ms"
    )

    print("\n== 3. verdict")
    check(
        "every tick fits inside the control period",
        worst < TICK_MS,
        f"worst {worst:.2f} ms of {TICK_MS:.1f} ms",
    )
    check(
        f"residual stays under {MAX_SHARE_OF_TICK:.0%} of the tick",
        p99 < MAX_SHARE_OF_TICK * TICK_MS,
        f"p99 {p99:.2f} ms = {p99 / TICK_MS:.1%} of the tick",
    )
    if baseline is not None:
        # Phase 2 replaces Phase 1's composition step, so the added cost is the difference.
        added = p99 - baseline["composition_p99_ms"]
        total = baseline["inference_p99_ms"] + p99
        print(
            f"  projected loop p99: {baseline['inference_p99_ms']:.2f} (base) + "
            f"{p99:.2f} (residual+composition) = {total:.2f} ms "
            f"of {TICK_MS:.1f} ms"
        )
        check(
            "projected loop leaves headroom",
            total < 0.5 * TICK_MS,
            f"{total:.2f} ms = {total / TICK_MS:.0%} of the tick "
            f"(residual adds {added:+.2f} ms)",
        )

    print()
    print("  note: this measures added compute, not scheduling under live DDS load.")
    print("        Only a shadow-mode session on the robot can show that.")
    print()
    if failures:
        print(f"CONTROL BUDGET: FAIL ({len(failures)}): {', '.join(failures)}")
        return 1
    print("CONTROL BUDGET: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
