"""Resolve a trial's success verdict, handling recordings that stored ``accepted and export_eligible``.

In such recordings every deterministic (non-export-eligible) success reads as a failure.
The signature is whole-session: no trial accepted and none export-eligible; a lone
``success_candidate`` with ``accepted == False`` is an operator override and is left
alone. ``--trust-auto-outcome`` opts in to scoring by the geometric check instead.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

#: The auto_outcome the geometric success check writes when it fires.
SUCCESS_CANDIDATE = "success_candidate"


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def trial_accepted(attrs: Mapping[str, Any], *, trust_auto: bool = False) -> bool:
    """One trial's success verdict.

    With ``trust_auto`` a fired geometric check counts as success regardless of ``accepted``.
    """
    if bool(attrs.get("accepted", False)):
        return True
    if trust_auto and _text(attrs.get("auto_outcome", "")) == SUCCESS_CANDIDATE:
        return True
    return False


def mislabelled_success_count(trial_attrs: Sequence[Mapping[str, Any]]) -> int:
    """Number of likely mislabelled successes; 0 unless the whole session has the signature."""
    if not trial_attrs:
        return 0
    if any(bool(a.get("accepted", False)) for a in trial_attrs):
        return 0
    if any(bool(a.get("export_eligible", True)) for a in trial_attrs):
        return 0
    return sum(
        1
        for a in trial_attrs
        if _text(a.get("auto_outcome", "")) == SUCCESS_CANDIDATE
    )


def warn_if_mislabelled(
    trial_attrs: Sequence[Mapping[str, Any]], *, trust_auto: bool
) -> str:
    """A one-paragraph notice for the console, or "" when there is nothing to say."""
    count = mislabelled_success_count(trial_attrs)
    if not count:
        return ""
    if trust_auto:
        return (
            f"--trust-auto-outcome: treating {count} trial(s) with "
            f"auto_outcome={SUCCESS_CANDIDATE!r} as successes; the stored "
            "'accepted' flag says failure for all of them."
        )
    return (
        f"NOTE: this session has {count} trial(s) where the automated success "
        f"check fired (auto_outcome={SUCCESS_CANDIDATE!r}) but 'accepted' is "
        "False, and no trial in the file is accepted or export-eligible.\n"
        "      That is the signature of older recordings, which "
        "recorded `accepted and export_eligible` and so stored every\n"
        "      deterministic-mode success as a failure. Success rates below "
        "will read 0%.\n"
        "      Pass --trust-auto-outcome to score these by the automated check "
        "instead."
    )


def add_trust_argument(parser: Any) -> None:
    parser.add_argument(
        "--trust-auto-outcome",
        action="store_true",
        help="score trials by the automated success check rather than the "
        "stored 'accepted' flag, for older recordings that stored "
        "deterministic-mode successes as failures.",
    )


__all__ = [
    "SUCCESS_CANDIDATE",
    "add_trust_argument",
    "mislabelled_success_count",
    "trial_accepted",
    "warn_if_mislabelled",
]
