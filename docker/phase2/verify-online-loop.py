#!/usr/bin/env python3
"""End-to-end check of the actor/learner loop without a robot.

Runs episode accumulation -> agentlace -> learner buffers -> RLPD updates ->
published parameters -> residual actor, and asserts the residual's output changes
once new parameters arrive. Standalone rather than pytest: agentlace never
terminates its ZeroMQ contexts, so the process must exit via os._exit.

Usage:  python3 /opt/sparr/verify-online-loop.py [--steps N]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

for candidate in (
    "/opt/sparr/ros2_ws/src/sparr_phase2_residual",
    "ros2_ws/src/sparr_phase2_residual",
):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from sparr_phase2_residual.episode import EpisodeAccumulator  # noqa: E402
from sparr_phase2_residual.learner import LearnerConfig, ResidualLearner  # noqa: E402
from sparr_phase2_residual.observation_spec import ResidualObservationSpec  # noqa: E402
from sparr_phase2_residual.promotion import DemoPromoter  # noqa: E402
from sparr_phase2_residual.residual_policy import ResidualPolicy  # noqa: E402
from sparr_phase2_residual.transport import (  # noqa: E402
    connect_actor_client,
    start_learner_server,
    stop_quietly,
)

IMAGE_SIZE = 32          # smaller than production; the ResNet is convolutional
SPEC = ResidualObservationSpec()
PORT, BROADCAST_PORT = 15590, 15591
SETTLE_S = 15.0

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{f': {detail}' if detail else ''}")
    if not ok:
        failures.append(name)


def wait_for(predicate, timeout_s: float = SETTLE_S, poll_s: float = 0.1) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return False


def observation_record(rng, step: int):
    """A random observation record in the Phase 1 recorder's layout."""
    return {
        "robot": {
            "flange_pose_world_xyzw": rng.normal(size=7),
            "ext_wrench_tcp": rng.normal(size=6),
            "tcp_pose_world_xyzw": np.concatenate(
                [np.array([step * 0.01, 0.0, 0.0]), np.zeros(4)]
            ),
        },
        "task": {"socket_root_exact_world_xyzw": rng.normal(size=7)},
        "camera": {
            "wrist_1": {
                "rgb": rng.integers(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            }
        },
    }


def run_episode(rng, length: int) -> list:
    """Accumulate one random accepted episode into transitions."""
    accumulator = EpisodeAccumulator(SPEC)
    for step in range(length):
        accumulator.observe(observation_record(rng, step), rng.normal(size=6))
        accumulator.act(rng.uniform(-1, 1, 6))
    accumulator.observe(observation_record(rng, length), rng.normal(size=6))
    return accumulator.finish(accepted=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6, help="learner steps to run")
    args = parser.parse_args()

    import jax
    from agentlace.data.data_store import QueuedDataStore

    rng = np.random.default_rng(0)

    print("== 1. learner, seeded with demonstrations")
    learner = ResidualLearner(
        SPEC,
        LearnerConfig(
            image_size=IMAGE_SIZE, batch_size=8, training_starts=1, seed=0,
            log_period=2, steps_per_update=2,
        ),
    )
    import pickle
    import tempfile

    scratch = Path(tempfile.mkdtemp(prefix="sparr_online_"))
    demo_path = scratch / "demos.pkl"
    demo_path.write_bytes(
        pickle.dumps([t for _ in range(3) for t in run_episode(rng, 10)])
    )
    learner.load_demonstrations([demo_path])
    learner.checkpoint_dir = scratch / "ckpt"
    print(f"  demonstrations: {len(learner.demo_store)} rows, "
          f"promotion median {learner.promoter.median}")

    server = start_learner_server(
        learner.online_store, learner.demo_store, port=PORT, broadcast_port=BROADCAST_PORT
    )
    print(f"  serving on :{PORT} (broadcast :{BROADCAST_PORT})")

    print("\n== 2. actor, with a live residual")
    from sparr_phase2_residual.residual_agent import make_residual_agent

    actor_agent = make_residual_agent(SPEC, seed=1, image_size=IMAGE_SIZE)
    residual = ResidualPolicy(actor_agent, SPEC, seed=1, action_scale=1.0)

    received: list = []

    def on_params(params):
        residual.update_params(params)
        received.append(params)

    actor_online, actor_demo = QueuedDataStore(5000), QueuedDataStore(5000)
    client = connect_actor_client(
        actor_online, actor_demo, host="localhost", port=PORT,
        broadcast_port=BROADCAST_PORT, on_params=on_params, timeout_ms=5000,
    )
    check("actor connected to the learner", True)

    # Fixed probe observation, so any output change comes from the parameters.
    probe_terms = {name: rng.normal(size=width) for name, width in SPEC.terms()}
    probe_images = {
        "wrist_1": rng.integers(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    }
    before = residual.evaluate(terms=probe_terms, images=probe_images)
    check("residual evaluates before training", before.ok, before.error or "")

    print("\n== 3. actor -> learner: a completed episode")
    online_before = len(learner.online_store)
    demo_before = len(learner.demo_store)

    promoter = DemoPromoter(lengths=list(learner.promoter.lengths))
    episode = run_episode(rng, 5)          # shorter than the median -> promoted
    decision = promoter.consider([{"rewards": 1.0, "dones": True}] * len(episode),
                                 succeeded=True)
    for transition in episode:
        actor_online.insert(transition)
        if decision.promoted:
            actor_demo.insert(transition)
    client.update()

    check(
        "transitions reached the learner's online buffer",
        wait_for(lambda: len(learner.online_store) > online_before),
        f"{online_before} -> {len(learner.online_store)}",
    )
    check(
        "a promoted episode reached the demonstration buffer",
        decision.promoted
        and wait_for(lambda: len(learner.demo_store) > demo_before),
        f"{decision} | {demo_before} -> {len(learner.demo_store)}",
    )

    print("\n== 4. the learner trains on it")
    check("learner has enough data to start", learner.ready, learner.sampler.describe())
    published: list = []

    def publish(params):
        """Publish to the actor and record the push."""
        server.publish_network(params)
        published.append(params)

    # Warm-up steps so the timing below excludes XLA compilation.
    learner.run(steps=2, on_publish=publish)
    start = time.monotonic()
    learner.run(steps=args.steps, on_publish=publish)
    elapsed = time.monotonic() - start
    rate = args.steps / elapsed if elapsed else float("inf")
    metrics = learner.metrics
    check("learner produced finite losses",
          bool(metrics) and all(np.isfinite(v) for v in metrics.values()),
          ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if "loss" in k))
    print(f"        throughput: {rate:.1f} learner steps/s over {elapsed:.1f} s")
    check("learner published parameters", bool(published), f"{len(published)} push(es)")

    print("\n== 5. learner -> actor: parameters take effect")
    check(
        "actor received published parameters",
        wait_for(lambda: len(received) > 0),
        f"{len(received)} received",
    )
    if received:
        after = residual.evaluate(terms=probe_terms, images=probe_images)
        check("residual still evaluates after the swap", after.ok, after.error or "")
        check(
            "parameter version advanced",
            residual.params_version > 0,
            f"version {residual.params_version}",
        )
        if before.ok and after.ok:
            delta = float(np.abs(after.residual - before.residual).max())
            check(
                "the residual's output actually changed",
                delta > 1e-9,
                f"max|after - before| = {delta:.3e}",
            )

    print("\n== 6. shutdown")
    for problem in stop_quietly(client, server, timeout_s=10.0):
        print(f"  note: {problem}")
    check("shutdown returned rather than hanging", True)

    print()
    if failures:
        print(f"ONLINE LOOP: FAIL ({len(failures)}): {', '.join(failures)}")
        return 1
    print("ONLINE LOOP: PASS -- actor and learner are genuinely connected")
    return 0


if __name__ == "__main__":
    status = main()
    sys.stdout.flush()
    os._exit(status)  # agentlace leaves its ZeroMQ contexts un-terminated
