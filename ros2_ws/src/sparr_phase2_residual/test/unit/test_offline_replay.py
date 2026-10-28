# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA OneWay Noncommercial License (March 22, 2022).
# See the LICENSE file in the repository root for the full license text:
# https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf

"""Replaying a recorded session: episode splitting, schedules, release order."""

from __future__ import annotations

import pytest

from sparr_phase2_residual.offline_replay import (
    ArrivalSchedule,
    Episode,
    ReplayDriver,
    ReplayError,
    build_schedule,
    split_episodes,
)


def transition(*, done=False, reward=0.0, succeed=None):
    return {
        "actions": [0.0] * 6,
        "rewards": reward,
        "dones": done,
        "masks": 0.0 if done else 1.0,
        "infos": {"succeed": reward > 0.0 if succeed is None else succeed},
    }


def episode_transitions(length: int, *, succeeded: bool):
    rows = [transition() for _ in range(length - 1)]
    rows.append(transition(done=True, reward=1.0 if succeeded else 0.0))
    return rows


class TestSplittingEpisodes:
    def test_splits_on_dones_and_reads_the_verdict(self):
        flat = episode_transitions(3, succeeded=True) + episode_transitions(
            2, succeeded=False
        )
        episodes = split_episodes(flat)
        assert [len(e) for e in episodes] == [3, 2]
        assert [e.succeeded for e in episodes] == [True, False]

    def test_a_trailing_fragment_without_dones_is_dropped(self):
        # Inserting a boundary-less fragment would corrupt a buffer that
        # reconstructs image pairs from `dones`.
        flat = episode_transitions(3, succeeded=True) + [transition(), transition()]
        episodes = split_episodes(flat)
        assert [len(e) for e in episodes] == [3]

    def test_a_terminal_transition_that_contradicts_itself_is_rejected(self):
        bad = transition(done=True, reward=1.0, succeed=False)
        with pytest.raises(ReplayError):
            split_episodes([bad])

    def test_no_episodes_from_an_empty_list(self):
        assert split_episodes([]) == []


class TestSchedules:
    def test_all_at_once_puts_everything_at_step_zero(self):
        assert ArrivalSchedule.all_at_once(4).steps == [0, 0, 0, 0]

    def test_interval_spaces_arrivals_evenly(self):
        assert ArrivalSchedule.every(4, 50).steps == [0, 50, 100, 150]

    def test_decreasing_arrival_steps_are_rejected(self):
        with pytest.raises(ReplayError):
            ArrivalSchedule(steps=[0, 100, 50])

    def test_promoted_defaults_to_none_promoted(self):
        schedule = ArrivalSchedule(steps=[0, 10])
        assert schedule.promoted == [False, False]

    def test_promoted_length_must_match(self):
        with pytest.raises(ReplayError):
            ArrivalSchedule(steps=[0, 10], promoted=[True])

    def test_build_schedule_rejects_an_unknown_mode(self):
        with pytest.raises(ReplayError):
            build_schedule("sometimes", 3)

    def test_interval_mode_needs_an_interval(self):
        with pytest.raises(ReplayError):
            build_schedule("interval", 3)

    def test_schedule_mode_needs_a_logdir(self):
        with pytest.raises(ReplayError):
            build_schedule("schedule", 3)


class TestReleasing:
    @staticmethod
    def _driver(schedule_steps, *, lengths=None, promoted=None):
        lengths = lengths or [2] * len(schedule_steps)
        episodes = [
            Episode(transitions=episode_transitions(n, succeeded=True), succeeded=True)
            for n in lengths
        ]
        released = []
        driver = ReplayDriver(
            episodes=episodes,
            schedule=ArrivalSchedule(steps=list(schedule_steps), promoted=promoted),
            insert=lambda ep, promo: released.append((len(ep), promo)),
        )
        return driver, released

    def test_nothing_is_released_before_its_step(self):
        driver, released = self._driver([0, 100, 200])
        driver.release_due(0)
        assert len(released) == 1
        driver.release_due(99)
        assert len(released) == 1
        driver.release_due(100)
        assert len(released) == 2

    def test_a_jump_releases_every_episode_it_passed(self):
        # The learner only consults the schedule once per iteration, so a
        # schedule with several arrivals inside one step must not lose any.
        driver, released = self._driver([0, 10, 20, 30])
        driver.release_due(25)
        assert len(released) == 3

    def test_simultaneous_arrivals_all_release(self):
        driver, released = self._driver([0, 0, 0])
        driver.release_due(0)
        assert len(released) == 3
        assert driver.exhausted

    def test_the_promotion_flag_is_replayed_not_recomputed(self):
        driver, released = self._driver([0, 0], promoted=[False, True])
        driver.release_due(0)
        assert [p for _, p in released] == [False, True]

    def test_release_one_forces_an_early_arrival(self):
        driver, released = self._driver([0, 5000])
        driver.release_due(0)
        assert len(released) == 1
        assert driver.release_one() is True
        assert len(released) == 2
        assert driver.release_one() is False

    def test_counts_mismatch_is_rejected_at_construction(self):
        episodes = [Episode(transitions=episode_transitions(2, succeeded=True), succeeded=True)]
        with pytest.raises(ReplayError):
            ReplayDriver(
                episodes=episodes,
                schedule=ArrivalSchedule(steps=[0, 10]),
                insert=lambda ep, promo: None,
            )
