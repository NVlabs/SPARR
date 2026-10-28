# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA OneWay Noncommercial License (March 22, 2022).
# See the LICENSE file in the repository root for the full license text:
# https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf

"""Demonstration promotion: successful online episodes become demonstrations.

A successful episode is promoted when it is no longer than the running median
of demonstration lengths, which pulls the policy toward shorter cycle times.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


class PromotionError(ValueError):
    """Raised when an episode cannot be considered for promotion."""


@dataclass(frozen=True)
class PromotionDecision:
    """Why one episode was or was not promoted."""

    promoted: bool
    reason: str
    length: int
    median: Optional[float]

    def __str__(self) -> str:
        verdict = "promoted" if self.promoted else "kept out"
        median = "n/a" if self.median is None else f"{self.median:.1f}"
        return f"{verdict}: {self.reason} (length {self.length}, median {median})"


@dataclass
class DemoPromoter:
    """Decides which successful online episodes join the demonstration buffer.

    ``lengths`` seeds the running median from the demonstrations already loaded.
    """

    lengths: list[int] = field(default_factory=list)
    #: Cap on promotions, bounding dilution of the original demonstrations.
    max_promotions: Optional[int] = None
    promoted_count: int = 0

    def __post_init__(self) -> None:
        for length in self.lengths:
            if length < 1:
                raise PromotionError(f"demonstration lengths must be >= 1, got {length}")
        if self.max_promotions is not None and self.max_promotions < 0:
            raise PromotionError("max_promotions must not be negative")

    @property
    def median(self) -> Optional[float]:
        """Running median demonstration length, or None before any exist."""
        return statistics.median(self.lengths) if self.lengths else None

    def consider(
        self,
        episode: Sequence[Any],
        *,
        succeeded: Optional[bool] = None,
        force: bool = False,
    ) -> PromotionDecision:
        """Judge one completed online episode.

        ``succeeded`` defaults to the terminal transition's reward being positive.
        ``force`` admits a successful episode whatever its length; it counts
        against ``max_promotions`` but its length is not added to the median.
        """
        if not episode:
            raise PromotionError("cannot consider an empty episode")
        length = len(episode)

        if succeeded is None:
            succeeded = float(episode[-1].get("rewards", 0.0)) > 0.0
        if not succeeded:
            return PromotionDecision(False, "episode did not succeed", length, self.median)

        if self.max_promotions is not None and self.promoted_count >= self.max_promotions:
            return PromotionDecision(
                False,
                f"promotion cap reached ({self.max_promotions})",
                length,
                self.median,
            )

        if force:
            self.promoted_count += 1
            return PromotionDecision(
                True, "forced (retry succeeded)", length, self.median
            )

        median = self.median
        if median is None:
            self._record(length)
            return PromotionDecision(True, "first demonstration", length, None)

        if length <= median:
            self._record(length)
            return PromotionDecision(
                True, f"at least as efficient as the median ({median:.1f})", length, median
            )
        return PromotionDecision(
            False, f"longer than the median ({median:.1f})", length, median
        )

    def _record(self, length: int) -> None:
        self.lengths.append(length)
        self.promoted_count += 1

    def state(self) -> dict[str, Any]:
        """Serialisable state for resuming with the same median."""
        return {
            "lengths": list(self.lengths),
            "promoted_count": self.promoted_count,
            "max_promotions": self.max_promotions,
        }


    def describe(self) -> str:
        median = "n/a" if self.median is None else f"{self.median:.1f}"
        return (
            f"promotion: {self.promoted_count} promoted, "
            f"{len(self.lengths)} demo episodes, median length {median}"
        )


def episode_lengths(transitions: Sequence[Any]) -> list[int]:
    """Episode lengths in a flat transition list, split on ``dones``.

    A trailing run without a terminal transition is not counted.
    """
    lengths: list[int] = []
    current = 0
    for transition in transitions:
        current += 1
        if bool(transition.get("dones", False)):
            lengths.append(current)
            current = 0
    return lengths
