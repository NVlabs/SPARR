"""Train from a finished Phase 2 session as if the data were arriving live.

Replays recorded episodes into the learner's buffers in one of three modes:

``all``
    Every episode present from step 0; skips the early-data regime.

``schedule``
    Episodes released at the learner steps they arrived at in the recorded
    run, read from its ``episode/succeeded`` and ``episode/promoted`` scalars.

``interval``
    Episodes released every N steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence


class ReplayError(RuntimeError):
    """Raised when a session cannot be replayed as asked."""


@dataclass(frozen=True)
class Episode:
    """One recorded episode, with the verdict the learner would have seen."""

    transitions: list[dict[str, Any]]
    succeeded: bool

    def __len__(self) -> int:
        return len(self.transitions)


def split_episodes(transitions: Sequence[dict[str, Any]]) -> list[Episode]:
    """Split a flat transition list on ``dones`` into episodes.

    A trailing run with no terminal transition is dropped, since the buffer
    needs ``dones`` to reconstruct image pairs.
    """
    episodes: list[Episode] = []
    current: list[dict[str, Any]] = []
    for transition in transitions:
        current.append(transition)
        if bool(transition.get("dones", False)):
            episodes.append(
                Episode(transitions=current, succeeded=_succeeded(current[-1]))
            )
            current = []
    return episodes


def _succeeded(terminal: dict[str, Any]) -> bool:
    """Whether the episode ending in this transition was a success.

    ``rewards`` decides; ``infos.succeed`` must agree with it.
    """
    reward = float(terminal.get("rewards", 0.0)) > 0.0
    infos = terminal.get("infos") or {}
    flagged = bool(infos.get("succeed", reward))
    if reward != flagged:
        raise ReplayError(
            "episode terminal transition disagrees with itself: "
            f"rewards={terminal.get('rewards')!r} but infos.succeed={flagged!r}"
        )
    return reward



@dataclass
class ArrivalSchedule:
    """When each episode enters the buffers, in learner steps.

    ``steps[i]`` is the step episode ``i`` becomes visible; ``promoted[i]`` is
    the actor's recorded decision to also put it in the demonstration buffer.
    """

    steps: list[int]
    promoted: list[bool] = field(default_factory=list)

    def __post_init__(self) -> None:
        if any(b < a for a, b in zip(self.steps, self.steps[1:])):
            raise ReplayError("arrival steps must be non-decreasing")
        if not self.promoted:
            self.promoted = [False] * len(self.steps)
        if len(self.promoted) != len(self.steps):
            raise ReplayError(
                f"promoted has {len(self.promoted)} entries for "
                f"{len(self.steps)} arrivals"
            )

    def __len__(self) -> int:
        return len(self.steps)

    @classmethod
    def all_at_once(cls, count: int) -> "ArrivalSchedule":
        return cls(steps=[0] * count)

    @classmethod
    def every(cls, count: int, interval: int, *, start: int = 0) -> "ArrivalSchedule":
        if interval < 1:
            raise ReplayError(f"interval must be at least 1, got {interval}")
        return cls(steps=[start + i * interval for i in range(count)])

    @classmethod
    def from_tensorboard(cls, logdir: str | Path) -> "ArrivalSchedule":
        """Recover the real schedule from a previous run's TensorBoard log."""
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        path = Path(logdir)
        if not path.exists():
            raise ReplayError(f"no TensorBoard directory at {path}")
        # size_guidance 0 keeps every point; the default downsampling drops arrivals.
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        tags = accumulator.Tags().get("scalars", [])
        if "episode/succeeded" not in tags:
            raise ReplayError(
                f"{path} has no 'episode/succeeded' scalar, so it records no "
                "episode arrivals. It is probably a run that trained offline, "
                "or one started with --no-tensorboard."
            )
        arrivals = accumulator.Scalars("episode/succeeded")
        steps = [int(event.step) for event in arrivals]
        promoted: list[bool] = []
        if "episode/promoted" in tags:
            marks = accumulator.Scalars("episode/promoted")
            if len(marks) == len(arrivals):
                promoted = [bool(event.value) for event in marks]
        return cls(steps=steps, promoted=promoted)


def build_schedule(
    mode: str,
    count: int,
    *,
    logdir: Optional[str | Path] = None,
    interval: Optional[int] = None,
) -> ArrivalSchedule:
    """The schedule for ``count`` episodes under one of the three modes."""
    if mode == "all":
        return ArrivalSchedule.all_at_once(count)
    if mode == "interval":
        if interval is None:
            raise ReplayError("mode 'interval' needs --replay-every")
        return ArrivalSchedule.every(count, interval)
    if mode == "schedule":
        if logdir is None:
            raise ReplayError("mode 'schedule' needs --replay-schedule-tb")
        schedule = ArrivalSchedule.from_tensorboard(logdir)
        if len(schedule) != count:
            raise ReplayError(
                f"the recorded run has {len(schedule)} episode arrivals but the "
                f"replay data has {count} episodes. They must be the same "
                "session: export the online data from the SAME recording the "
                "TensorBoard log came from, with --include-unaccepted so "
                "failures are present too."
            )
        return schedule
    raise ReplayError(
        f"unknown replay mode {mode!r}; expected 'all', 'schedule' or 'interval'"
    )



@dataclass
class ReplayDriver:
    """Releases episodes into a learner as its step counter advances.

    Call :meth:`release_due` once per learner iteration; being pull-based keeps
    a replay reproducible.
    """

    episodes: list[Episode]
    schedule: ArrivalSchedule
    #: ``(episode, promoted) -> None``. Supplied by the learner.
    insert: Any
    released: int = 0

    def __post_init__(self) -> None:
        if len(self.episodes) != len(self.schedule):
            raise ReplayError(
                f"{len(self.episodes)} episodes but {len(self.schedule)} "
                "scheduled arrivals"
            )

    @property
    def exhausted(self) -> bool:
        return self.released >= len(self.episodes)

    @property
    def pending(self) -> int:
        return len(self.episodes) - self.released

    def release_due(self, step: int) -> int:
        """Release every episode scheduled at or before ``step``. Returns how many."""
        count = 0
        while (
            not self.exhausted and self.schedule.steps[self.released] <= step
        ):
            index = self.released
            self.insert(self.episodes[index], self.schedule.promoted[index])
            self.released += 1
            count += 1
        return count

    def release_one(self) -> bool:
        """Release the next episode ahead of schedule, to reach learner readiness."""
        if self.exhausted:
            return False
        index = self.released
        self.insert(self.episodes[index], self.schedule.promoted[index])
        self.released += 1
        return True

    def describe(self) -> str:
        successes = sum(1 for e in self.episodes if e.succeeded)
        transitions = sum(len(e) for e in self.episodes)
        span = self.schedule.steps[-1] if self.schedule.steps else 0
        return (
            f"replay: {len(self.episodes)} episodes "
            f"({successes} success / {len(self.episodes) - successes} failure), "
            f"{transitions} transitions, arrivals spanning steps "
            f"0..{span}, {sum(self.schedule.promoted)} promoted"
        )


__all__ = [
    "ArrivalSchedule",
    "Episode",
    "ReplayDriver",
    "ReplayError",
    "build_schedule",
    "split_episodes",
]
