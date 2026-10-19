# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Demonstration promotion by running median of demonstration lengths."""

import pytest

from sparr_phase2_residual.promotion import (
    DemoPromoter,
    PromotionError,
    episode_lengths,
)


def episode(length, *, succeeded=True):
    return [
        {"rewards": float(succeeded and i == length - 1), "dones": i == length - 1}
        for i in range(length)
    ]


def test_a_failed_episode_is_never_promoted():
    """Failures are useful online data but must not seed the demo buffer."""
    promoter = DemoPromoter(lengths=[100])
    decision = promoter.consider(episode(5, succeeded=False))
    assert not decision.promoted
    assert "did not succeed" in decision.reason
    assert promoter.lengths == [100]


def test_a_successful_episode_below_the_median_is_promoted():
    promoter = DemoPromoter(lengths=[10, 20, 30])  # median 20
    assert promoter.median == 20
    assert promoter.consider(episode(15)).promoted


def test_an_episode_exactly_at_the_median_is_promoted():
    """"At least as efficient" is inclusive, so equality qualifies."""
    promoter = DemoPromoter(lengths=[10, 20, 30])
    assert promoter.consider(episode(20)).promoted


def test_each_promotion_immediately_tightens_the_bar():
    """The ratchet: promoting 15 moves the median to 17.5, so 20 no longer passes."""
    promoter = DemoPromoter(lengths=[10, 20, 30])
    assert promoter.consider(episode(15)).promoted
    assert promoter.median == 17.5
    assert not promoter.consider(episode(20)).promoted


def test_a_successful_but_longer_episode_is_rejected():
    promoter = DemoPromoter(lengths=[10, 20, 30])
    decision = promoter.consider(episode(21))
    assert not decision.promoted
    assert "longer than the median" in decision.reason
    assert promoter.lengths == [10, 20, 30], "a rejected episode must not shift the median"


def test_promotion_ratchets_the_median_downward():
    """The cycle-time mechanism: each promotion can only lower or hold it."""
    promoter = DemoPromoter(lengths=[100, 100, 100])
    medians = [promoter.median]
    for length in (90, 80, 70, 60, 50):
        assert promoter.consider(episode(length)).promoted
        medians.append(promoter.median)
    assert medians == sorted(medians, reverse=True), medians
    assert promoter.median < 100


def test_the_median_never_rises_even_across_many_promotions():
    promoter = DemoPromoter(lengths=[50])
    previous = promoter.median
    for length in (50, 49, 50, 30, 40, 20):
        promoter.consider(episode(length))
        assert promoter.median <= previous
        previous = promoter.median


def test_the_first_episode_is_promoted_when_there_are_no_demonstrations():
    """Otherwise a run started without operator demos could never fill the buffer."""
    promoter = DemoPromoter()
    assert promoter.median is None
    decision = promoter.consider(episode(42))
    assert decision.promoted and decision.reason == "first demonstration"
    assert promoter.median == 42


def test_a_failed_first_episode_still_is_not_promoted():
    promoter = DemoPromoter()
    assert not promoter.consider(episode(5, succeeded=False)).promoted
    assert promoter.median is None


def test_success_can_be_supplied_explicitly():
    promoter = DemoPromoter(lengths=[10])
    # Reward says failure, caller says success: the caller wins.
    assert promoter.consider(episode(5, succeeded=False), succeeded=True).promoted
    assert not promoter.consider(episode(5), succeeded=False).promoted


def test_the_promotion_cap_bounds_dilution_of_operator_demonstrations():
    promoter = DemoPromoter(lengths=[100], max_promotions=2)
    assert promoter.consider(episode(10)).promoted
    assert promoter.consider(episode(10)).promoted
    decision = promoter.consider(episode(10))
    assert not decision.promoted and "cap reached" in decision.reason


def test_an_empty_episode_is_rejected_loudly():
    with pytest.raises(PromotionError, match="empty"):
        DemoPromoter().consider([])


@pytest.mark.parametrize("bad", [0, -1])
def test_a_nonsense_seed_length_is_rejected(bad):
    with pytest.raises(PromotionError):
        DemoPromoter(lengths=[bad])


def test_a_negative_cap_is_rejected():
    with pytest.raises(PromotionError):
        DemoPromoter(max_promotions=-1)


def test_episode_lengths_splits_a_flat_transition_list_on_dones():
    transitions = episode(3) + episode(5) + episode(2)
    assert episode_lengths(transitions) == [3, 5, 2]


def test_a_truncated_trailing_episode_is_not_counted():
    """Counting it would understate the median and make promotion too easy."""
    transitions = episode(4) + [{"rewards": 0.0, "dones": False}] * 2
    assert episode_lengths(transitions) == [4]


def test_seeding_from_demonstrations_sets_the_first_median():
    transitions = episode(10) + episode(20) + episode(30)
    promoter = DemoPromoter(lengths=episode_lengths(transitions))
    assert promoter.median == 20
    assert not promoter.consider(episode(25)).promoted


class TestForcedPromotion:
    """The retry policy's path: promote a hard case without moving the bar."""

    def test_force_admits_an_episode_far_longer_than_the_median(self) -> None:
        promoter = DemoPromoter(lengths=[40, 42, 44])
        episode = [{"rewards": 1.0, "dones": True}] * 200
        assert not promoter.consider(episode, succeeded=True).promoted
        decision = promoter.consider(episode, succeeded=True, force=True)
        assert decision.promoted
        assert "forced" in decision.reason

    def test_a_forced_promotion_does_not_raise_the_median(self) -> None:
        # The median stays a ratchet: a long replay must not loosen the bar for
        # later promotions.
        promoter = DemoPromoter(lengths=[40, 42, 44])
        before = promoter.median
        promoter.consider([{"rewards": 1.0, "dones": True}] * 200, succeeded=True, force=True)
        assert promoter.median == before
        assert promoter.lengths == [40, 42, 44]

    def test_a_forced_promotion_still_counts_against_the_cap(self) -> None:
        promoter = DemoPromoter(lengths=[40], max_promotions=1)
        episode = [{"rewards": 1.0, "dones": True}] * 200
        assert promoter.consider(episode, succeeded=True, force=True).promoted
        second = promoter.consider(episode, succeeded=True, force=True)
        assert not second.promoted
        assert "cap reached" in second.reason

    def test_force_does_not_promote_a_failure(self) -> None:
        promoter = DemoPromoter(lengths=[40])
        decision = promoter.consider(
            [{"rewards": 0.0, "dones": True}] * 10, succeeded=False, force=True
        )
        assert not decision.promoted
