# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Pure collector decisions shared by the ROS node and unit tests."""

from __future__ import annotations

from enum import Enum

from .safety import GuardStatus


class RolloutInputDisposition(str, Enum):
    """Required action for one synchronized rollout input."""

    ACCEPT = "accept"
    HOLD = "hold"
    REJECT = "reject"


class TerminalObservationPlan(str, Enum):
    """How to satisfy the canonical ``N == M + 1`` trial invariant."""

    ALREADY_COMPLETE = "already_complete"
    APPEND = "append"


def classify_rollout_input(
    *,
    guard_status: GuardStatus,
    camera_pair_valid: bool,
    policy_input_valid: bool,
) -> RolloutInputDisposition:
    """Classify inputs without collapsing a hard rejection into a stale hold.

    Camera age/skew failures are transient and hold; an invalid robot payload or
    failed policy encoding rejects the attempt before another command is sent.
    """

    if guard_status is GuardStatus.REJECT:
        return RolloutInputDisposition.REJECT
    if guard_status is GuardStatus.HOLD or not camera_pair_valid:
        return RolloutInputDisposition.HOLD
    if not policy_input_valid:
        return RolloutInputDisposition.REJECT
    return RolloutInputDisposition.ACCEPT


def plan_terminal_observation(
    observation_count: int,
    action_count: int,
) -> TerminalObservationPlan:
    """Return the only safe next step for a trial's observation/action counts."""

    if observation_count < 0 or action_count < 0:
        raise ValueError("observation and action counts must be non-negative")
    if observation_count == action_count + 1:
        return TerminalObservationPlan.ALREADY_COMPLETE
    if observation_count == action_count:
        return TerminalObservationPlan.APPEND
    raise RuntimeError(
        "collector sequence is inconsistent: expected N == M or N == M + 1, "
        f"got N={observation_count}, M={action_count}"
    )


def trial_export_eligible(
    execution_mode: str, deterministic_inference: bool = False
) -> bool:
    """Only physical live stochastic collection may become training data.

    Deterministic trials are excluded like shadow: their residual label
    ``effective_software_action - mu_raw`` is ~zero by construction.
    """

    if execution_mode not in {"live", "shadow"}:
        raise ValueError(f"unknown execution mode {execution_mode!r}")
    if deterministic_inference:
        return False
    return execution_mode == "live"


def policy_goal_publish_enabled(execution_mode: str, configured: bool) -> bool:
    """Suppress the goal side channel in shadow mode as well as commands."""

    if execution_mode not in {"live", "shadow"}:
        raise ValueError(f"unknown execution mode {execution_mode!r}")
    return bool(configured) and execution_mode != "shadow"


__all__ = [
    "RolloutInputDisposition",
    "TerminalObservationPlan",
    "classify_rollout_input",
    "plan_terminal_observation",
    "policy_goal_publish_enabled",
    "trial_export_eligible",
]
