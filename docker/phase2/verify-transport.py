#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Socket-level check of the actor/learner link over loopback.

Exchanges transitions, stats and a parameter broadcast between a real agentlace
server and client. Standalone rather than pytest: agentlace never terminates its
ZeroMQ contexts, so the process must exit via os._exit.

Usage:  python3 /opt/sparr/verify-transport.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, "/opt/sparr/ros2_ws/src/sparr_phase2_residual")
sys.path.insert(0, "ros2_ws/src/sparr_phase2_residual")

from sparr_phase2_residual.buffers import make_data_store  # noqa: E402
from sparr_phase2_residual.observation_spec import ResidualObservationSpec  # noqa: E402
from sparr_phase2_residual.transport import (  # noqa: E402
    connect_actor_client,
    start_learner_server,
    stop_quietly,
)

IMAGE_SIZE = 16
SPEC = ResidualObservationSpec()
#: Non-default ports, so a running learner cannot interfere.
PORT, BROADCAST_PORT = 15588, 15589
SETTLE_S = 10.0

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


def transition(rng, done: bool = False) -> dict:
    def observation():
        return {
            "state": rng.normal(size=(1, SPEC.state_dim())).astype(np.float32),
            "wrist_1": rng.integers(
                0, 255, (1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8
            ),
        }

    return {
        "observations": observation(),
        "next_observations": observation(),
        "actions": rng.uniform(-1, 1, 6).astype(np.float32),
        "rewards": float(done),
        "masks": float(not done),
        "dones": done,
        "infos": {},
    }


def main() -> int:
    from agentlace.data.data_store import QueuedDataStore

    rng = np.random.default_rng(0)
    online = make_data_store(SPEC, 500, image_size=IMAGE_SIZE)
    demo = make_data_store(SPEC, 500, image_size=IMAGE_SIZE)
    stats: list[dict] = []
    params: list = []

    print(f"== learner server on :{PORT} (broadcast :{BROADCAST_PORT})")
    server = start_learner_server(
        online, demo, port=PORT, broadcast_port=BROADCAST_PORT, on_stats=stats.append
    )
    actor_online, actor_demo = QueuedDataStore(200), QueuedDataStore(200)
    client = connect_actor_client(
        actor_online,
        actor_demo,
        host="localhost",
        port=PORT,
        broadcast_port=BROADCAST_PORT,
        on_params=params.append,
        timeout_ms=5000,
    )
    check("actor connected to learner", True)

    print("\n== actor -> learner")
    for index in range(10):
        item = transition(rng, done=(index == 9))
        actor_online.insert(item)
        if index >= 5:
            # Promoted episodes go on both streams.
            actor_demo.insert(item)
    client.update()
    check(
        "transitions reached the online buffer",
        wait_for(lambda: len(online) > 0),
        f"{len(online)} rows",
    )
    check(
        "transitions reached the demonstration buffer",
        wait_for(lambda: len(demo) > 0),
        f"{len(demo)} rows",
    )

    print("\n== actor stats -> learner")
    client.request("send-stats", {"episode_return": 1.0, "length": 42})
    check("stats reached the learner", wait_for(lambda: len(stats) > 0), str(stats[-1:]))

    print("\n== learner params -> actor")
    published = {"w": np.arange(4, dtype=np.float32)}
    server.publish_network(published)
    received = wait_for(lambda: len(params) > 0)
    check("parameters reached the actor", received)
    if received:
        check(
            "parameters arrived intact",
            bool(np.allclose(np.asarray(params[-1]["w"]), published["w"])),
            str(np.asarray(params[-1]["w"]).tolist()),
        )

    print("\n== shutdown")
    problems = stop_quietly(client, server, timeout_s=10.0)
    for problem in problems:
        print(f"  note: {problem}")
    # Leftover contexts are expected; the check is that stop_quietly returned.
    check("shutdown returned rather than hanging", True)

    print()
    if failures:
        print(f"TRANSPORT CHECK: FAIL ({len(failures)}): {', '.join(failures)}")
        return 1
    print("TRANSPORT CHECK: PASS")
    return 0


if __name__ == "__main__":
    status = main()
    sys.stdout.flush()
    # agentlace leaves its ZeroMQ contexts un-terminated; a normal exit would block.
    os._exit(status)
