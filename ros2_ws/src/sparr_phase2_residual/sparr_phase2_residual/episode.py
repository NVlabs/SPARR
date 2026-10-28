# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA License.
# See the LICENSE file in the repository root for the full license text.

"""Turning a trajectory into RLPD transitions.

Shared by the offline converter and the live actor so both produce identical
transitions. Everything after the first step of the trailing run where the TCP
stays within a threshold of its final position is discarded, so the terminal
reward lands where the arm arrived.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .observation_spec import ResidualObservationSpec, assemble_state

#: TCP position tolerance (metres) for the terminal stationary run.
DEFAULT_STATIONARY_THRESHOLD_M = 0.001


class EpisodeError(ValueError):
    """Raised when a trajectory cannot be turned into transitions."""


def terminal_run_start(poses: np.ndarray, threshold_m: float) -> int:
    """First index of the trailing run that never leaves ``threshold_m`` of the end.

    Returns 0 when the whole trajectory is stationary, so the caller can reject
    the episode instead of training on a degenerate one.
    """
    if len(poses) == 0:
        raise EpisodeError("cannot analyse an empty pose sequence")
    terminal = poses[-1]
    start = 0
    for index in range(len(poses) - 1, -1, -1):
        if np.any(np.abs(poses[index] - terminal) > threshold_m):
            start = index + 1
            break
    return start


def build_transitions(
    *,
    states: Sequence[np.ndarray],
    images: Sequence[Mapping[str, np.ndarray]],
    actions: Sequence[np.ndarray],
    tcp_positions: Sequence[np.ndarray],
    accepted: bool,
    spec: ResidualObservationSpec,
    stationary_threshold_m: float = DEFAULT_STATIONARY_THRESHOLD_M,
    info: Optional[Mapping[str, Any]] = None,
    truncated: Optional[bool] = None,
) -> list[dict[str, Any]]:
    """Assemble one episode's transitions, with backfill and truncation applied.

    ``states`` and ``images`` hold N observations; ``actions`` holds N-1. Every
    array is already in serl's frame-stack layout: state ``(1, D)``, image
    ``(1, H, W, 3)``.
    """
    n_actions = len(actions)
    if n_actions == 0:
        raise EpisodeError("episode contains no actions")
    if len(states) != n_actions + 1:
        raise EpisodeError(
            f"expected {n_actions + 1} observations for {n_actions} actions, "
            f"got {len(states)}"
        )
    if len(images) != len(states):
        raise EpisodeError("images and states disagree in length")
    if len(tcp_positions) < n_actions + 1:
        raise EpisodeError(
            f"expected at least {n_actions + 1} TCP poses, got {len(tcp_positions)}"
        )

    poses = np.asarray(tcp_positions[: n_actions + 1], dtype=np.float64)
    run_start = terminal_run_start(poses, stationary_threshold_m)
    if run_start == 0:
        raise EpisodeError(
            f"the whole episode is stationary within {stationary_threshold_m} m; "
            "there is nothing to learn from it"
        )
    keep = min(run_start + 1, n_actions)

    # A timed-out trial is not a terminal state; masking it would bias bootstrapping.
    if truncated is None:
        truncated = not accepted
    if truncated and keep == n_actions:
        # The final next_observation carries a copied base mean; do not bootstrap from it.
        keep -= 1
        if keep == 0:
            raise EpisodeError(
                "truncated episode has no transition with a real terminal "
                "next_observation"
            )

    reward = 1.0 if accepted else 0.0
    extra = dict(info or {})
    transitions: list[dict[str, Any]] = []
    for index in range(keep):
        terminal = index == keep - 1
        observation: dict[str, np.ndarray] = {"state": states[index]}
        next_observation: dict[str, np.ndarray] = {"state": states[index + 1]}
        for key in spec.image_keys:
            observation[key] = images[index][key]
            next_observation[key] = images[index + 1][key]
        transitions.append(
            {
                "observations": observation,
                "actions": np.asarray(actions[index], np.float32),
                "next_observations": next_observation,
                "rewards": reward if terminal else 0.0,
                # dones marks the episode boundary for the buffer; masks gates
                # bootstrapping.
                "masks": 0.0 if (terminal and not truncated) else 1.0,
                "dones": bool(terminal),
                "infos": {**extra, "succeed": bool(accepted and terminal),
                          "truncated": bool(terminal and truncated),
                          "source_step": int(index)},
            }
        )

    # The copied base mean in the terminal next_observation is safe only if masked.
    if not truncated and transitions[-1]["masks"] != 0.0:
        raise EpisodeError("the terminal transition must be masked")
    return transitions


@dataclass
class EpisodeAccumulator:
    """Collects one live episode, step by step, into RLPD transitions.

    Fed the same records the HDF5 writer receives, so the result matches what
    ``demo_export`` reads back from the file.
    """

    spec: ResidualObservationSpec
    stationary_threshold_m: float = DEFAULT_STATIONARY_THRESHOLD_M
    states: list[np.ndarray] = field(default_factory=list)
    images: list[dict[str, np.ndarray]] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    tcp_positions: list[np.ndarray] = field(default_factory=list)

    #: Record paths of each state term; must match the HDF5 paths demo_export reads.
    TERM_SOURCES = {
        "flange_pose": ("robot", "flange_pose_world_xyzw"),
        "wrench": ("robot", "ext_wrench_tcp"),
        "socket_pose": ("task", "socket_root_exact_world_xyzw"),
    }

    def reset(self) -> None:
        self.states.clear()
        self.images.clear()
        self.actions.clear()
        self.tcp_positions.clear()

    @property
    def steps(self) -> int:
        return len(self.actions)

    def observe(self, record: Mapping[str, Any], base_mean: np.ndarray) -> None:
        """Record one observation.

        ``base_mean`` is the base policy's mean for this step; for the terminal
        observation the caller passes the previous step's mean.
        """
        terms: dict[str, np.ndarray] = {}
        for name, _ in self.spec.terms():
            if name == "base_mean":
                terms[name] = np.asarray(base_mean, np.float64).reshape(-1)
                continue
            group, key = self.TERM_SOURCES[name]
            terms[name] = np.asarray(record[group][key], np.float64)
        self.states.append(assemble_state(self.spec, terms))

        frames: dict[str, np.ndarray] = {}
        for key in self.spec.image_keys:
            frames[key] = np.asarray(record["camera"][key]["rgb"])[None, ...]
        self.images.append(frames)

        self.tcp_positions.append(
            np.asarray(record["robot"]["tcp_pose_world_xyzw"], np.float64)[:3]
        )

    def act(self, action: np.ndarray) -> None:
        """Record the realized residual for the step just taken."""
        self.actions.append(np.asarray(action, np.float32).reshape(-1))

    def finish(
        self, *, accepted: bool, info: Optional[Mapping[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """Close the episode and return its transitions.

        Raises :class:`EpisodeError` if the episode is too short or entirely
        stationary.
        """
        if len(self.states) == len(self.actions):
            # No terminal observation: drop the unpaired last action.
            self.actions = self.actions[:-1]
        return build_transitions(
            states=self.states,
            images=self.images,
            actions=self.actions,
            tcp_positions=self.tcp_positions,
            accepted=accepted,
            spec=self.spec,
            stationary_threshold_m=self.stationary_threshold_m,
            info=info,
        )
