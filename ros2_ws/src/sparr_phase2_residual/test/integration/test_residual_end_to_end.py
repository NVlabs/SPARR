# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA OneWay Noncommercial License (March 22, 2022).
# See the LICENSE file in the repository root for the full license text:
# https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf

"""The shipped ``phase2_live.yaml`` driving the real JAX SAC agent.

Runs :class:`ResidualPolicy` end to end to catch mismatches between the
configured observation layout and the network it builds. Skipped where JAX is
unavailable.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from sparr_phase2_residual.config import Phase2Config  # noqa: E402
from sparr_phase2_residual.residual_agent import make_residual_agent  # noqa: E402
from sparr_phase2_residual.residual_policy import ResidualPolicy  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[5]
IMAGE_SIZE = 128
#: Residual inference budget per tick of the 30 Hz control loop.
TICK_BUDGET_MS = 10.0


@pytest.fixture(scope="module")
def config() -> Phase2Config:
    path = REPO_ROOT / "phase2_live.yaml"
    if not path.is_file():
        pytest.skip(f"{path} not present")
    return Phase2Config.load(path)


@pytest.fixture(scope="module")
def policy(config: Phase2Config) -> ResidualPolicy:
    agent = make_residual_agent(config.observation, seed=config.runtime.seed)
    return ResidualPolicy(
        agent,
        config.observation,
        seed=config.runtime.seed,
        deterministic=config.runtime.deterministic,
        action_scale=config.runtime.action_scale,
    )


def observation(config: Phase2Config, rng: np.random.Generator):
    widths = dict(config.observation.terms())
    terms = {
        name: rng.normal(size=width).astype(np.float64)
        for name, width in widths.items()
    }
    images = {
        key: rng.integers(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        for key in config.observation.image_keys
    }
    return terms, images


def test_the_shipped_config_produces_a_usable_residual(config, policy):
    terms, images = observation(config, np.random.default_rng(0))
    decision = policy.evaluate(terms=terms, images=images)
    assert decision.ok, decision.error
    assert decision.residual.shape == (6,)
    assert np.all(np.isfinite(decision.residual))
    # The actor is tanh-squashed, then attenuated by the configured scale.
    assert np.all(np.abs(decision.residual) <= config.runtime.action_scale + 1e-6)


def test_the_residual_fits_inside_the_control_tick(config, policy):
    rng = np.random.default_rng(1)
    terms, images = observation(config, rng)
    policy.evaluate(terms=terms, images=images)  # absorb tracing/compilation

    latencies = []
    for _ in range(20):
        terms, images = observation(config, rng)
        start = time.monotonic_ns()
        decision = policy.evaluate(terms=terms, images=images)
        latencies.append((time.monotonic_ns() - start) / 1e6)
        assert decision.ok, decision.error
    median = float(np.median(latencies))
    worst = max(latencies)
    assert worst < TICK_BUDGET_MS, (
        f"residual took up to {worst:.2f} ms (median {median:.2f} ms), "
        f"budget {TICK_BUDGET_MS} ms of a 33 ms tick"
    )


def test_different_observations_produce_different_residuals(config, policy):
    """Guards against an encoder that ignores its input."""
    rng = np.random.default_rng(2)
    residuals = []
    for _ in range(5):
        terms, images = observation(config, rng)
        decision = policy.evaluate(terms=terms, images=images)
        assert decision.ok, decision.error
        residuals.append(decision.residual)
    spread = np.abs(np.stack(residuals) - residuals[0]).max()
    assert spread > 1e-6, "the residual did not respond to its observation at all"


def test_a_shape_mismatch_against_the_built_network_is_caught(config, policy):
    """Guards against the config and the network disagreeing."""
    terms, images = observation(config, np.random.default_rng(3))
    terms["flange_pose"] = np.zeros(6)  # declared width is 7
    decision = policy.evaluate(terms=terms, images=images)
    assert not decision.ok
    assert "flange_pose" in decision.error
