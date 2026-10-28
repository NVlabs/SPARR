#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA License.
# See the LICENSE file in the repository root for the full license text.

"""Phase 2 environment gate for the JAX stack; exits non-zero if any check fails.

Checks imports, GPU kernel throughput, the pretrained ResNet-10 encoder, actor
latency against the control tick, one SAC update, and JAX/Torch sharing one GPU.

Usage:  python3 /opt/sparr/verify-phase2-env.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Match the image's setting so VRAM behaviour is representative.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

IMAGE_SIZE = 128
STATE_DIM = 19
ACTION_DIM = 6
BATCH = 256
IMAGE_KEYS = ("wrist_1",)

# A jaxlib without native kernels for this GPU falls back to PTX JIT and runs far below this.
MIN_TFLOPS = 10.0
# p99 must be a small share of the tick; a single call need only fit the tick (allows jitter).
TICK_MS = 1000.0 / 30.0
MAX_ACTOR_P99_MS = 10.0
MAX_ACTOR_SINGLE_MS = TICK_MS

_failures: list[str] = []
_skipped: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}{f': {detail}' if detail else ''}")
    if not ok:
        _failures.append(name)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="skip the SAC update (slowest check, ~30 s of compilation)",
    )
    args = parser.parse_args()

    print("== 1. imports and versions")
    import jax
    import chex, distrax, flax, optax  # noqa: F401

    print(f"  python  {sys.version.split()[0]}")
    for module in (jax, flax, optax, distrax, chex):
        print(f"  {module.__name__:<9} {module.__version__}")
    import cv2

    check(
        "apt OpenCV was not shadowed by a pip opencv-python",
        cv2.__version__.startswith("4."),
        f"cv2 {cv2.__version__} (cv_bridge needs the apt build)",
    )

    print("\n== 2. GPU visible and at full speed")
    devices = jax.devices()
    gpu = check("JAX sees a GPU", jax.default_backend() == "gpu", str(devices))
    if gpu:
        import jax.numpy as jnp

        a = jnp.ones((4096, 4096))
        for _ in range(3):  # warm-up excludes compilation
            a = (a @ a) / 4096.0
        a.block_until_ready()
        start = time.monotonic()
        for _ in range(10):
            a = (a @ a) / 4096.0
        a.block_until_ready()
        tflops = 10 * 2 * 4096**3 / (time.monotonic() - start) / 1e12
        check(
            "JAX reaches real GPU kernels",
            tflops >= MIN_TFLOPS,
            f"{tflops:.1f} TFLOP/s (floor {MIN_TFLOPS})",
        )
    else:
        _skipped.append("GPU-dependent checks")

    print("\n== 3. agent construction and pretrained encoder")
    sys.path.insert(0, "/opt/sparr/ros2_ws/src/sparr_phase2_residual")
    sys.path.insert(0, "ros2_ws/src/sparr_phase2_residual")
    from sparr_phase2_residual.serl_compat import as_serl_batch, resnet10_pickle_compat
    from sparr_phase2_residual.residual_agent import ensure_resnet10_cached

    ensure_resnet10_cached()
    from serl_launcher.utils.launcher import make_sac_pixel_agent

    sample_obs = {
        "state": np.zeros((1, STATE_DIM), np.float32),
        "wrist_1": np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8),
    }
    with resnet10_pickle_compat():
        agent = make_sac_pixel_agent(
            0,
            sample_obs,
            np.zeros((ACTION_DIM,), np.float32),
            image_keys=IMAGE_KEYS,
            encoder_type="resnet-pretrained",
        )
    check("SACAgent built with the pretrained ResNet-10 encoder", True)

    print("\n== 4. actor latency inside the control tick")
    action = np.asarray(
        jax.device_get(
            agent.sample_actions(
                observations=jax.device_put(sample_obs), seed=jax.random.PRNGKey(0)
            )
        )
    ).reshape(-1)
    check("actor returns a 6-D action", action.shape == (ACTION_DIM,), str(action.shape))
    check(
        "actor output is tanh-squashed into [-1, 1]",
        bool(np.all(np.abs(action) <= 1.0 + 1e-6)),
        f"max|a|={np.abs(action).max():.4f}",
    )
    latencies = []
    for i in range(30):
        start = time.monotonic_ns()
        jax.device_get(
            agent.sample_actions(
                observations=jax.device_put(sample_obs), seed=jax.random.PRNGKey(i)
            )
        )
        latencies.append((time.monotonic_ns() - start) / 1e6)
    warm = latencies[1:]
    p99 = float(np.percentile(warm, 99))
    detail = (
        f"median {np.median(warm):.2f} ms, p99 {p99:.2f} ms, max {max(warm):.2f} ms "
        f"(first call {latencies[0]:.1f} ms)"
    )
    check("actor p99 is a small share of the tick", p99 < MAX_ACTOR_P99_MS, detail)
    check(
        "no single actor call blows the tick",
        max(warm) < MAX_ACTOR_SINGLE_MS,
        f"worst {max(warm):.2f} ms of {MAX_ACTOR_SINGLE_MS:.1f} ms",
    )

    if args.quick:
        print("\n== 5. SAC update  [skipped: --quick]")
        _skipped.append("SAC update")
    else:
        print("\n== 5. one SAC update on synthetic data")
        rng = np.random.default_rng(0)

        def observations():
            return {
                "state": rng.normal(size=(BATCH, 1, STATE_DIM)).astype(np.float32),
                "wrist_1": rng.integers(
                    0, 255, (BATCH, 1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8
                ),
            }

        batch = as_serl_batch(
            {
                "observations": observations(),
                "next_observations": observations(),
                "actions": rng.uniform(-1, 1, (BATCH, ACTION_DIM)).astype(np.float32),
                "rewards": rng.integers(0, 2, BATCH).astype(np.float32),
                "masks": np.ones(BATCH, np.float32),
                "dones": np.zeros(BATCH, bool),
            }
        )
        batch = jax.device_put(batch)
        networks = frozenset({"actor", "critic", "temperature"})
        agent, info = agent.update(batch, networks_to_update=networks)
        jax.block_until_ready(info)

        def scalars(tree, prefix=""):
            for key, value in tree.items():
                if hasattr(value, "items"):
                    yield from scalars(value, f"{prefix}{key}/")
                elif np.ndim(value) == 0:
                    yield prefix + key, float(value)

        metrics = dict(scalars(jax.device_get(info)))
        check("update produced metrics", bool(metrics), f"{len(metrics)} scalars")
        bad = {k: v for k, v in metrics.items() if not np.isfinite(v)}
        check("all losses are finite", not bad, str(bad) if bad else "")
        for key in ("actor/actor_loss", "critic/critic_loss"):
            if key in metrics:
                print(f"        {key} = {metrics[key]:.4f}")

        for _ in range(3):
            agent, info = agent.update(batch, networks_to_update=networks)
        jax.block_until_ready((agent, info))
        start = time.monotonic()
        for _ in range(10):
            agent, info = agent.update(batch, networks_to_update=networks)
        jax.block_until_ready((agent, info))
        per_step = (time.monotonic() - start) / 10 * 1000
        print(f"        steady-state: {per_step:.1f} ms/update ({1000/per_step:.0f}/s)")

    print("\n== 6. JAX and Torch on one GPU in one process")
    import torch

    if torch.cuda.is_available():
        product = torch.ones(512, 512, device="cuda") @ torch.ones(512, 512, device="cuda")
        torch.cuda.synchronize()
        check(
            "Torch still reaches CUDA after JAX initialised",
            float(product.sum()) == 512**3,
            f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}",
        )
        check("JAX still holds its device", jax.default_backend() == "gpu")
    else:
        check("Torch reaches CUDA", False, "torch.cuda.is_available() is False")

    print()
    if _skipped:
        print(f"skipped: {', '.join(_skipped)}")
    if _failures:
        print(f"PHASE 2 ENV GATE: FAIL ({len(_failures)}): {', '.join(_failures)}")
        return 1
    print("PHASE 2 ENV GATE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
