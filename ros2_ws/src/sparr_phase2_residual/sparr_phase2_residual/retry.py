# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA OneWay Noncommercial License (March 22, 2022).
# See the LICENSE file in the repository root for the full license text:
# https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf

"""Replay a failed attempt under identical conditions once the policy is good.

When the windowed success rate reaches a threshold, a failed attempt is replayed
(same socket offset and start pose) up to ``max_repeats`` times; the eventual
success can be promoted into the demonstration buffer.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Optional


class RetryError(ValueError):
    """Raised when a retry policy cannot be configured as written."""


@dataclass(frozen=True)
class RetryDecision:
    """What to do after one completed attempt."""

    #: Replay the same attempt: same socket offset, same randomized start.
    repeat: bool
    #: Promote this episode into the demonstration buffer regardless of length.
    force_promote: bool
    reason: str
    success_rate: Optional[float]
    repeats_used: int

    def __str__(self) -> str:
        rate = "n/a" if self.success_rate is None else f"{self.success_rate:.0%}"
        return f"{self.reason} (success rate {rate}, repeats used {self.repeats_used})"


#: A decision that changes nothing.
def _inert(reason: str, rate: Optional[float], used: int) -> RetryDecision:
    return RetryDecision(
        repeat=False, force_promote=False, reason=reason, success_rate=rate, repeats_used=used
    )


@dataclass
class RetryPolicy:
    """Decides when a failed attempt is replayed and whether to force-promote.

    Feed every completed attempt to :meth:`consider`.
    """

    enabled: bool = False
    #: Windowed success rate at or above which replaying begins.
    success_rate_threshold: float = 0.8
    #: Attempts the rate is measured over; no rate until the window is full.
    window: int = 20
    #: Replays of a single attempt before giving up and moving on.
    max_repeats: int = 5
    #: Promote the success that ends a replay sequence into the demo buffer.
    promote_on_success: bool = True

    _outcomes: Deque[bool] = field(default_factory=deque, init=False)
    _repeats_used: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.success_rate_threshold) <= 1.0:
            raise RetryError(
                "success_rate_threshold must be in [0, 1], got "
                f"{self.success_rate_threshold}"
            )
        if int(self.window) < 1:
            raise RetryError(f"window must be at least 1, got {self.window}")
        if int(self.max_repeats) < 1:
            raise RetryError(f"max_repeats must be at least 1, got {self.max_repeats}")
        self._outcomes = deque(maxlen=int(self.window))

    @property
    def success_rate(self) -> Optional[float]:
        """Windowed success rate, or None until the window has filled."""
        if len(self._outcomes) < self._outcomes.maxlen:
            return None
        return sum(self._outcomes) / len(self._outcomes)

    @property
    def gate_open(self) -> bool:
        rate = self.success_rate
        return rate is not None and rate >= self.success_rate_threshold

    @property
    def repeats_used(self) -> int:
        """Replays spent on the attempt currently in progress."""
        return self._repeats_used

    @property
    def replaying(self) -> bool:
        return self._repeats_used > 0

    def consider(self, *, succeeded: bool) -> RetryDecision:
        """Judge one completed attempt and advance the policy's state.

        Call exactly once per attempt, including replays.
        """
        replaying = self.replaying
        # Replays are excluded from the window, or they would close the gate
        # that opened them.
        if not replaying:
            self._outcomes.append(bool(succeeded))
        rate = self.success_rate
        used = self._repeats_used

        if not self.enabled:
            return _inert("retry disabled", rate, used)

        if succeeded:
            if replaying:
                self._repeats_used = 0
                return RetryDecision(
                    repeat=False,
                    force_promote=bool(self.promote_on_success),
                    reason=f"replay succeeded after {used} repeat(s)",
                    success_rate=rate,
                    repeats_used=used,
                )
            return _inert("attempt succeeded", rate, used)

        if not replaying and not self.gate_open:
            if rate is None:
                return _inert(
                    f"success rate not established yet "
                    f"({len(self._outcomes)}/{self._outcomes.maxlen} attempts)",
                    rate,
                    used,
                )
            return _inert(
                f"success rate {rate:.0%} below the {self.success_rate_threshold:.0%} "
                "threshold",
                rate,
                used,
            )

        if used >= self.max_repeats:
            self._repeats_used = 0
            return _inert(f"gave up after {self.max_repeats} repeat(s)", rate, used)

        self._repeats_used = used + 1
        return RetryDecision(
            repeat=True,
            force_promote=False,
            reason=f"replaying failed attempt ({self._repeats_used}/{self.max_repeats})",
            success_rate=rate,
            repeats_used=self._repeats_used,
        )

    def abandon(self) -> None:
        """Drop any replay in progress, e.g. when a trial is discarded."""
        self._repeats_used = 0

    def state(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "success_rate_threshold": self.success_rate_threshold,
            "window": self.window,
            "max_repeats": self.max_repeats,
            "promote_on_success": self.promote_on_success,
            "outcomes": list(self._outcomes),
            "repeats_used": self._repeats_used,
        }

    def describe(self) -> str:
        if not self.enabled:
            return "retry: disabled"
        rate = self.success_rate
        shown = "n/a" if rate is None else f"{rate:.0%}"
        gate = "open" if self.gate_open else "closed"
        return (
            f"retry: gate {gate} (success rate {shown} over "
            f"{len(self._outcomes)}/{self._outcomes.maxlen}, threshold "
            f"{self.success_rate_threshold:.0%}), max {self.max_repeats} repeats"
        )


__all__ = ["RetryPolicy", "RetryDecision", "RetryError"]
