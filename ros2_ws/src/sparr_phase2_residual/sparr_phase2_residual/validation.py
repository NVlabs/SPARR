# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA OneWay Noncommercial License (March 22, 2022).
# See the LICENSE file in the repository root for the full license text:
# https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf

"""Offline validation of Phase 1/Phase 2 HDF5 sessions.

Checks control rate and jitter, base-policy and composition latency (the
residual runs inside the composition timer), executed-action fidelity
(``clip(mean + residual)``), shadow-mode integrity, and reports success rate
and cycle time per mode.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import numpy as np

#: The control loop's nominal rate.
NOMINAL_RATE_HZ = 30.0
#: Jitter tolerance on the mean period, as a fraction of nominal.
RATE_TOLERANCE = 0.10
#: A single step may overrun by this much before it counts as a miss.
DEADLINE_FACTOR = 1.5
#: Fraction of steps allowed to miss the deadline.
MAX_MISS_FRACTION = 0.02
#: Recomposition must be exact to float32 rounding.
FIDELITY_TOLERANCE = 1e-6


class ValidationError(RuntimeError):
    """Raised when a session cannot be analysed at all."""


@dataclass
class Check:
    name: str
    passed: Optional[bool]  # None = not measurable from a recording
    detail: str = ""

    def __str__(self) -> str:
        mark = {True: "PASS", False: "FAIL", None: "n/a "}[self.passed]
        return f"  [{mark}] {self.name}{f': {self.detail}' if self.detail else ''}"


@dataclass
class TrialReport:
    where: str
    steps: int
    accepted: bool
    residual_steps: int
    residual_applied_steps: int
    duration_s: float
    checks: list[Check] = field(default_factory=list)

    @property
    def mode(self) -> str:
        if self.residual_steps == 0:
            return "phase1"
        return "active" if self.residual_applied_steps else "shadow"


def _period_stats(step_start_ns: np.ndarray) -> dict[str, float]:
    """Loop period statistics, in milliseconds."""
    if len(step_start_ns) < 2:
        return {}
    periods = np.diff(np.asarray(step_start_ns, np.float64)) / 1e6
    periods = periods[np.isfinite(periods) & (periods > 0)]
    if not len(periods):
        return {}
    return {
        "mean_ms": float(np.mean(periods)),
        "median_ms": float(np.median(periods)),
        "p99_ms": float(np.percentile(periods, 99)),
        "max_ms": float(np.max(periods)),
        "rate_hz": 1000.0 / float(np.mean(periods)),
        "miss_fraction": float(
            np.mean(periods > DEADLINE_FACTOR * 1000.0 / NOMINAL_RATE_HZ)
        ),
        "count": int(len(periods)),
    }


def _ns_stats(values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, np.float64)
    data = data[data >= 0]
    if not len(data):
        return {}
    return {
        "median_ms": float(np.median(data)) / 1e6,
        "p99_ms": float(np.percentile(data, 99)) / 1e6,
        "max_ms": float(np.max(data)) / 1e6,
    }


def check_control_rate(trial: Any) -> tuple[Check, Check]:
    stats = _period_stats(trial["actions/timing/step_start_monotonic_ns"][:])
    if not stats:
        return (
            Check("control rate", None, "too few steps to measure"),
            Check("deadline misses", None, ""),
        )
    low = NOMINAL_RATE_HZ * (1 - RATE_TOLERANCE)
    high = NOMINAL_RATE_HZ * (1 + RATE_TOLERANCE)
    rate_ok = low <= stats["rate_hz"] <= high
    miss_ok = stats["miss_fraction"] <= MAX_MISS_FRACTION
    return (
        Check(
            "control rate",
            rate_ok,
            f"{stats['rate_hz']:.2f} Hz (want {low:.1f}-{high:.1f}), "
            f"period median {stats['median_ms']:.1f} ms, p99 {stats['p99_ms']:.1f} ms",
        ),
        Check(
            "deadline misses",
            miss_ok,
            f"{stats['miss_fraction']:.2%} of steps over "
            f"{DEADLINE_FACTOR * 1000 / NOMINAL_RATE_HZ:.0f} ms "
            f"(allow {MAX_MISS_FRACTION:.0%}), worst {stats['max_ms']:.1f} ms",
        ),
    )


def check_composition_latency(trial: Any) -> Check:
    """Residual evaluation is inside Phase 1's composition timer."""
    stats = _ns_stats(trial["actions/timing/composition_ns"][:])
    if not stats:
        return Check("composition latency", None, "not recorded")
    budget_ms = 1000.0 / NOMINAL_RATE_HZ
    ok = stats["p99_ms"] < budget_ms
    return Check(
        "composition latency (includes the residual)",
        ok,
        f"median {stats['median_ms']:.2f} ms, p99 {stats['p99_ms']:.2f} ms, "
        f"max {stats['max_ms']:.2f} ms (tick is {budget_ms:.1f} ms)",
    )


def check_inference_latency(trial: Any) -> Check:
    stats = _ns_stats(trial["actions/timing/inference_ns"][:])
    if not stats:
        return Check("base policy inference", None, "not recorded")
    return Check(
        "base policy inference",
        stats["p99_ms"] < 1000.0 / NOMINAL_RATE_HZ,
        f"median {stats['median_ms']:.2f} ms, p99 {stats['p99_ms']:.2f} ms",
    )


def check_action_fidelity(trial: Any, action_clip: float) -> Check:
    """The commanded action must be exactly clip(mean + residual), as recorded."""
    mean = np.asarray(trial["actions/policy/mu_raw_normalized"][:], np.float64)
    executed = np.asarray(
        trial["actions/action/effective_software_action_normalized"][:], np.float64
    )
    residual_path = "labels/extra/requested_residual"
    applied_path = "labels/extra/residual_applied"

    if residual_path not in trial:
        return Check("executed-action fidelity", None, "no residual recorded (Phase 1)")

    residual = np.asarray(trial[residual_path][:], np.float64)
    applied = (
        np.asarray(trial[applied_path][:], bool)
        if applied_path in trial
        else np.ones(len(residual), bool)
    )
    if not applied.any():
        return Check(
            "executed-action fidelity",
            None,
            "shadow mode, no residual was applied",
        )

    rows = applied
    expected = np.clip(mean[rows] + residual[rows], -action_clip, action_clip)
    deviation = np.abs(expected - executed[rows]).max() if rows.any() else 0.0
    return Check(
        "executed-action fidelity",
        bool(deviation <= FIDELITY_TOLERANCE),
        f"max|clip(mu+r) - executed| = {deviation:.3e} over {int(rows.sum())} "
        f"applied steps (tol {FIDELITY_TOLERANCE:.0e})",
    )


def check_shadow_integrity(trial: Any, action_clip: float) -> Check:
    """In shadow mode the command must be what Phase 1 alone would have sent."""
    applied_path = "labels/extra/residual_applied"
    if "labels/extra/requested_residual" not in trial:
        return Check("shadow-mode integrity", None, "no residual recorded")
    applied = (
        np.asarray(trial[applied_path][:], bool)
        if applied_path in trial
        else np.ones(len(trial["labels/extra/requested_residual"]), bool)
    )
    rows = ~applied
    if not rows.any():
        return Check("shadow-mode integrity", None, "no shadow steps in this trial")

    mean = np.asarray(trial["actions/policy/mu_raw_normalized"][:], np.float64)
    clipped_sample = np.asarray(
        trial["actions/policy/sample_clipped_normalized"][:], np.float64
    )
    executed = np.asarray(
        trial["actions/action/effective_software_action_normalized"][:], np.float64
    )
    # Phase 1 commands either the clipped sample (stochastic) or clip(mean)
    # (deterministic). A shadow step must match one of them exactly.
    deterministic = np.clip(mean, -action_clip, action_clip)
    matches_sample = np.abs(executed[rows] - clipped_sample[rows]).max(axis=1)
    matches_mean = np.abs(executed[rows] - deterministic[rows]).max(axis=1)
    worst = float(np.minimum(matches_sample, matches_mean).max())
    return Check(
        "shadow-mode integrity",
        worst <= FIDELITY_TOLERANCE,
        f"max deviation from the Phase 1 action = {worst:.3e} over "
        f"{int(rows.sum())} shadow steps",
    )


def check_command_validity(trial: Any) -> Check:
    valid = np.asarray(trial["labels/command_valid"][:], bool)
    published = (
        np.asarray(trial["actions/action/command_published"][:], bool)
        if "actions/action/command_published" in trial
        else None
    )
    detail = f"{int(valid.sum())}/{len(valid)} steps training-eligible"
    if published is not None:
        detail += f", {int(published.sum())} published"
    return Check("command validity", bool(valid.any()), detail)


def analyse_trial(trial: Any, where: str, action_clip: float) -> TrialReport:
    steps = len(trial["actions/policy/mu_raw_normalized"])
    residual_steps = (
        len(trial["labels/extra/requested_residual"])
        if "labels/extra/requested_residual" in trial
        else 0
    )
    applied = 0
    if "labels/extra/residual_applied" in trial:
        applied = int(np.asarray(trial["labels/extra/residual_applied"][:], bool).sum())
    elif residual_steps:
        applied = residual_steps

    stats = _period_stats(trial["actions/timing/step_start_monotonic_ns"][:])
    duration = steps * stats.get("mean_ms", 1000.0 / NOMINAL_RATE_HZ) / 1000.0

    report = TrialReport(
        where=where,
        steps=steps,
        accepted=bool(trial.attrs.get("accepted", False)),
        residual_steps=residual_steps,
        residual_applied_steps=applied,
        duration_s=duration,
    )
    rate, misses = check_control_rate(trial)
    report.checks = [
        rate,
        misses,
        check_inference_latency(trial),
        check_composition_latency(trial),
        check_action_fidelity(trial, action_clip),
        check_shadow_integrity(trial, action_clip),
        check_command_validity(trial),
    ]
    return report


def _sessions(paths: Sequence[str | Path]) -> Iterator[Path]:
    for entry in paths:
        path = Path(entry)
        if path.is_dir():
            yield from sorted(
                p for p in path.rglob("*.h5") if ".partial." not in p.name
            )
        elif path.is_file():
            yield path
        else:
            raise ValidationError(f"no such file or directory: {path}")


def analyse(paths: Sequence[str | Path]) -> list[TrialReport]:
    import h5py

    reports: list[TrialReport] = []
    for session in _sessions(paths):
        try:
            handle = h5py.File(session, "r")
        except OSError as exc:
            raise ValidationError(f"{session}: {exc}") from exc
        with handle:
            if "trials" not in handle:
                continue
            action_clip = float(handle.attrs.get("training_action_clip", 1.0))
            for name in sorted(handle["trials"]):
                trial = handle[f"trials/{name}"]
                if "actions/policy/mu_raw_normalized" not in trial:
                    continue
                reports.append(
                    analyse_trial(trial, f"{session.name}/{name}", action_clip)
                )
    return reports


def summarise(reports: Sequence[TrialReport]) -> dict[str, Any]:
    """KPIs, split by mode."""
    summary: dict[str, Any] = {"trials": len(reports)}
    by_mode: dict[str, list[TrialReport]] = {}
    for report in reports:
        by_mode.setdefault(report.mode, []).append(report)

    for mode, group in sorted(by_mode.items()):
        accepted = [r for r in group if r.accepted]
        entry: dict[str, Any] = {
            "trials": len(group),
            "successes": len(accepted),
            "success_rate": len(accepted) / len(group) if group else 0.0,
        }
        if accepted:
            durations = [r.duration_s for r in accepted]
            entry["cycle_time_s"] = {
                "mean": float(np.mean(durations)),
                "median": float(np.median(durations)),
                "min": float(np.min(durations)),
                "max": float(np.max(durations)),
            }
        summary[mode] = entry
    return summary


def format_report(reports: Sequence[TrialReport], *, verbose: bool = False) -> str:
    lines: list[str] = []
    failures = 0
    for report in reports:
        header = (
            f"{report.where}  [{report.mode}]  {report.steps} steps, "
            f"{report.duration_s:.1f} s, "
            f"{'accepted' if report.accepted else 'not accepted'}"
        )
        shown = [c for c in report.checks if verbose or c.passed is False]
        failures += sum(1 for c in report.checks if c.passed is False)
        if shown:
            lines.append(header)
            lines.extend(str(c) for c in shown)
        elif verbose:
            lines.append(header)

    summary = summarise(reports)
    lines.append("")
    lines.append(f"== summary over {summary['trials']} trial(s)")
    for mode in ("phase1", "shadow", "active"):
        if mode not in summary:
            continue
        entry = summary[mode]
        line = (
            f"  {mode:7s} {entry['successes']}/{entry['trials']} succeeded "
            f"({entry['success_rate']:.0%})"
        )
        if "cycle_time_s" in entry:
            cycle = entry["cycle_time_s"]
            line += (
                f", cycle time mean {cycle['mean']:.1f} s "
                f"(min {cycle['min']:.1f}, max {cycle['max']:.1f})"
            )
        lines.append(line)

    lines.append("")
    lines.append("== not measurable from a recording")
    lines.append("  reset reliability          -- needs a live session")
    lines.append("  sample efficiency (REQ-14) -- needs wall-clock training time")
    lines.append("  learner gradient steps/s   -- measured by the learner, not here")

    lines.append("")
    lines.append(
        f"VALIDATION: {'FAIL' if failures else 'PASS'}"
        f"{f' ({failures} failed check(s))' if failures else ''}"
    )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sparr_phase2_validate",
        description="Validate Phase 2 recordings: control rate, action fidelity, KPIs",
    )
    parser.add_argument("paths", nargs="+", help="HDF5 files or directories")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="show every check, not only failures")
    parser.add_argument("--json", action="store_true", help="emit the summary as JSON")
    args = parser.parse_args(argv)

    try:
        reports = analyse(args.paths)
    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not reports:
        print("no analysable trials found", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(summarise(reports), indent=2))
        return 0

    print(format_report(reports, verbose=args.verbose))
    return 0 if "VALIDATION: PASS" in format_report(reports) else 1


if __name__ == "__main__":
    sys.exit(main())
