#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA License.
# See the LICENSE file in the repository root for the full license text.

"""Session-level failure-pattern analysis for a Phase 1/2 recording.

Writes one dashboard PNG plus a per-trial CSV: outcome timeline, socket-noise
and start-pose scatter by outcome, duration vs difficulty (trend fitted on
successes), contact force over normalized progress, and final |dz| by outcome.

Needs matplotlib, which is in the Phase 2 image only. Run it as:

  ./docker/phase2/session.sh shell -c \
    'python3 "$SPARR_REPO"/tools/sparr_failure_analysis.py <file.h5>'
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - environment guard, not logic
    sys.exit(
        "matplotlib is required and is NOT in the Phase 1 image.\n"
        "Run this in the Phase 2 container instead:\n"
        "  ./docker/phase2/session.sh shell -c "
        "'python3 tools/sparr_failure_analysis.py <file.h5>'"
    )

# Outcome colours are CVD-safe, and each outcome also has its own marker shape.
OK_COLOR = "#1baf7a"
FAIL_COLOR = "#eb6834"
OK_MARK = "o"
FAIL_MARK = "X"
INK = "#0b0b0b"
INK_SOFT = "#4e5152"
GRID = "#dee2e6"
SURFACE = "#ffffff"

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_eef_offset(checkpoint, env_yaml, override: Path | None) -> np.ndarray:
    """Tool-point offset from the flange, read from the policy's env.yaml.

    Uses ``--env-yaml`` when given, else the env.yaml (or checkpoint) the session
    was recorded with. Fails loudly rather than guessing a default: a wrong offset
    silently shifts every plotted distance.
    """
    from sparr_phase1_collection.checkpoint import (
        build_phase1_policy_spec,
        load_agent_env_configs,
    )

    candidates = [override] if override else [env_yaml, checkpoint]
    tried = []
    for raw in candidates:
        if not raw:
            continue
        path = Path(str(raw)).expanduser()
        for option in ([path] if path.is_absolute() else [REPO_ROOT / path, path]):
            tried.append(str(option))
            try:
                agent, env = load_agent_env_configs(option)
                return np.asarray(build_phase1_policy_spec(agent, env).eef_offset, float)
            except Exception:  # noqa: BLE001 - try the next candidate
                continue
    sys.exit(
        "cannot read the tool offset from the policy's env.yaml "
        f"(tried: {', '.join(tried) or 'nothing recorded in the file'}).\n"
        "Pass --env-yaml <path/to/params/env.yaml> (or the checkpoint's folder)."
    )


def _quat_to_matrix_xyzw(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _corr(x, y):
    """(pearson, spearman). NaN when a constant input makes them undefined."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), float("nan")
    pearson = float(np.corrcoef(x, y)[0, 1])
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    spearman = float(np.corrcoef(rx, ry)[0, 1])
    return pearson, spearman


sys.path.insert(0, str(Path(__file__).resolve().parent))
from sparr_trial_outcome import (  # noqa: E402
    add_trust_argument,
    trial_accepted,
    warn_if_mislabelled,
)


def collect(path: Path, env_yaml: Path | None = None) -> tuple[list[dict], dict]:
    """Per-trial rows plus session metadata. Trials that never ran are skipped."""
    rows, skipped = [], []
    with h5py.File(path, "r", locking=False) as f:
        cfg = f.get("metadata/collection_config")

        def conf(key, default=None):
            if cfg is None or key not in cfg:
                return default
            v = cfg[key][()]
            if isinstance(v, bytes):
                return v.decode()
            a = np.asarray(v).ravel()
            return a[0] if a.size == 1 else a.tolist()

        eef_offset = resolve_eef_offset(
            conf("policy/checkpoint"), conf("policy/env_yaml"), env_yaml)

        meta = {
            "file": path.name,
            "closed_cleanly": bool(f.attrs.get("closed_cleanly", False)),
            "stochastic": bool(conf("policy/stochastic", False)),
            "control_rate_hz": float(conf("policy/control_rate_hz", 0.0) or 0.0),
            "socket_noise_mm": np.asarray(
                conf("randomization/socket_policy_observation/xy_half_range_m",
                     [0.0, 0.0]), float).ravel()[:2] * 1000.0,
            "start_noise_mm": np.asarray(
                conf("randomization/start/xy_half_range_m", [0.0, 0.0]),
                float).ravel()[:2] * 1000.0,
            "residual_mode": f.attrs.get("residual_mode", "base only"),
            "residual_params": Path(
                str(f.attrs.get("residual_params_path", ""))).name or "-",
            "residual_scale": f.attrs.get("residual_action_scale", "-"),
        }
        if isinstance(meta["residual_mode"], bytes):
            meta["residual_mode"] = meta["residual_mode"].decode()

        trials = f.get("trials")
        if trials is None:
            return [], meta

        for name in sorted(trials):
            t = trials[name]
            a = t.attrs
            n = int(a.get("observation_count", 0))
            if "actions" not in t or "observations" not in t or n < 3:
                skipped.append(name)
                continue

            obs = t["observations"]
            socket = np.asarray(a.get("socket_policy_offset_xyz_rpy",
                                      [np.nan] * 6), float)
            start = np.asarray(a.get("start_offset_xyz_rpy", [np.nan] * 6), float)

            stamps = np.asarray(obs["timing/ros_stamp_ns"][:], np.int64)
            duration_s = float(stamps[-1] - stamps[0]) / 1e9

            fz = np.asarray(obs["robot/ext_wrench_world"][:, 2], float)
            flange = np.asarray(obs["robot/flange_pose_world_xyzw"][:], float)
            sk = np.asarray(obs["task/socket_keypoint_exact_world_xyzw"][:], float)
            eef = np.array([q[:3] + _quat_to_matrix_xyzw(q[3:]) @ eef_offset
                            for q in flange])
            d_mm = (eef - sk[:, :3]) * 1000.0
            closest = int(np.argmin(np.abs(d_mm[:, 2])))

            mu = np.asarray(t["actions/policy/mu_raw_normalized"][:], float)
            residual = t.get("labels/extra/requested_residual")

            rows.append({
                "trial": name,
                "attempt_index": int(a.get("attempt_index", -1)),
                # Re-resolved in process() according to --trust-auto-outcome.
                "accepted": bool(a.get("accepted", False)),
                "auto_outcome": str(a.get("auto_outcome", "")),
                "export_eligible": bool(a.get("export_eligible", True)),
                "operator_label": str(a.get("operator_label", "")),
                "termination_reason": str(a.get("termination_reason", "")),
                "steps": n,
                "duration_s": duration_s,
                "socket_dx_mm": socket[0] * 1000.0,
                "socket_dy_mm": socket[1] * 1000.0,
                "socket_mag_mm": float(np.hypot(socket[0], socket[1]) * 1000.0),
                "start_dx_mm": start[0] * 1000.0,
                "start_dy_mm": start[1] * 1000.0,
                "start_mag_mm": float(np.hypot(start[0], start[1]) * 1000.0),
                "final_dx_mm": d_mm[closest, 0],
                "final_dy_mm": d_mm[closest, 1],
                "final_dz_mm": d_mm[closest, 2],
                "peak_lateral_mm": float(
                    np.max(np.linalg.norm(d_mm[:, :2], axis=1))),
                # Contact drives Fz positive; the negative excursion is a separate event.
                "fz_max_n": float(np.max(fz)),
                "fz_min_n": float(np.min(fz)),
                "fz_final_n": float(fz[-1]),
                "fz_mean_n": float(np.mean(fz)),
                "saturation": float(np.mean(np.abs(mu) >= 1.0)),
                "held_ticks": int(a.get("held_tick_count", 0)),
                "residual_abs_mean": (
                    float(np.abs(np.asarray(residual[:], float)).mean())
                    if residual is not None else float("nan")
                ),
                "_fz_trace": fz,
            })
    meta["skipped_trials"] = skipped
    return rows, meta


def _scatter_outcome(ax, rows, xkey, ykey, title, xlabel, ylabel, limit_mm=None):
    ok = [r for r in rows if r["accepted"]]
    bad = [r for r in rows if not r["accepted"]]
    for group, color, mark, label in (
        (ok, OK_COLOR, OK_MARK, f"success ({len(ok)})"),
        (bad, FAIL_COLOR, FAIL_MARK, f"failure ({len(bad)})"),
    ):
        if not group:
            continue
        ax.scatter([r[xkey] for r in group], [r[ykey] for r in group],
                   c=color, marker=mark, s=52, linewidths=0.8,
                   edgecolors="white", label=label, zorder=3)
    if limit_mm:
        ax.add_patch(plt.Rectangle(
            (-limit_mm, -limit_mm), 2 * limit_mm, 2 * limit_mm,
            fill=False, edgecolor=GRID, linestyle="--", linewidth=1.2, zorder=1))
    ax.axhline(0, color=GRID, lw=1, zorder=1)
    ax.axvline(0, color=GRID, lw=1, zorder=1)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=8, color=INK_SOFT)
    ax.set_ylabel(ylabel, fontsize=8, color=INK_SOFT)
    # Shared symmetric range and square box so axis scaling cannot fake a directional cluster.
    span = limit_mm * 1.25 if limit_mm else max(
        [abs(r[k]) for r in rows for k in (xkey, ykey)] + [1.0]) * 1.2
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=7, frameon=False, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)


def _panel_timeline(ax, rows):
    idx = [r["attempt_index"] for r in rows]
    mag = [r["socket_mag_mm"] for r in rows]
    for r in rows:
        color = OK_COLOR if r["accepted"] else FAIL_COLOR
        ax.vlines(r["attempt_index"], 0, r["socket_mag_mm"],
                  color=color, lw=2.4, alpha=0.85, zorder=2)
    ok = [r for r in rows if r["accepted"]]
    bad = [r for r in rows if not r["accepted"]]
    for group, color, mark, label in (
        (ok, OK_COLOR, OK_MARK, "success"), (bad, FAIL_COLOR, FAIL_MARK, "failure"),
    ):
        if group:
            ax.scatter([r["attempt_index"] for r in group],
                       [r["socket_mag_mm"] for r in group],
                       c=color, marker=mark, s=40, edgecolors="white",
                       linewidths=0.7, label=label, zorder=3)
    ax.set_title("session timeline: outcome and socket-noise magnitude per attempt",
                 fontsize=10, color=INK, loc="left")
    ax.set_xlabel("attempt index", fontsize=8, color=INK_SOFT)
    ax.set_ylabel("|socket noise| (mm)", fontsize=8, color=INK_SOFT)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
    if idx and mag:
        ax.set_xlim(min(idx) - 0.8, max(idx) + 0.8)


def _panel_time_vs_difficulty(ax, rows, xkey, xlabel):
    ok = [r for r in rows if r["accepted"]]
    bad = [r for r in rows if not r["accepted"]]
    for group, color, mark, label in (
        (ok, OK_COLOR, OK_MARK, f"success ({len(ok)})"),
        (bad, FAIL_COLOR, FAIL_MARK, f"failure ({len(bad)})"),
    ):
        if group:
            ax.scatter([r[xkey] for r in group], [r["duration_s"] for r in group],
                       c=color, marker=mark, s=52, edgecolors="white",
                       linewidths=0.8, label=label, zorder=3)
    note = "not enough successes to fit"
    if len(ok) >= 3:
        x = np.array([r[xkey] for r in ok], float)
        y = np.array([r["duration_s"] for r in ok], float)
        if np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 32)
            ax.plot(xs, slope * xs + intercept, color=OK_COLOR, lw=1.6,
                    ls="--", zorder=2)
            pear, spear = _corr(x, y)
            note = (f"successes only: {slope:+.2f} s/mm, "
                    f"r={pear:+.2f}  rho={spear:+.2f}")
    ax.set_title("time to finish vs difficulty", fontsize=10, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=8, color=INK_SOFT)
    ax.set_ylabel("trial duration (s)", fontsize=8, color=INK_SOFT)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.annotate(note, xy=(0.02, 0.02), xycoords="axes fraction",
                fontsize=7.5, color=INK_SOFT)


def _panel_force(ax, rows, samples=60):
    """Median Fz with an IQR band, on normalized progress.

    Normalized because failures run to the timeout, so wall-clock x would not align phases.
    """
    grid = np.linspace(0.0, 1.0, samples)
    drew = False
    for group, color, label in (
        ([r for r in rows if r["accepted"]], OK_COLOR, "success"),
        ([r for r in rows if not r["accepted"]], FAIL_COLOR, "failure"),
    ):
        if not group:
            continue
        stack = []
        for r in group:
            fz = r["_fz_trace"]
            if len(fz) < 2:
                continue
            stack.append(np.interp(grid, np.linspace(0, 1, len(fz)), fz))
        if not stack:
            continue
        stack = np.vstack(stack)
        med = np.median(stack, axis=0)
        lo, hi = np.percentile(stack, [25, 75], axis=0)
        ax.fill_between(grid, lo, hi, color=color, alpha=0.18, lw=0, zorder=2)
        ax.plot(grid, med, color=color, lw=2.0, label=f"{label} (n={len(stack)})",
                zorder=3)
        drew = True
    ax.axhline(0, color=GRID, lw=1, zorder=1)
    ax.set_title("contact force Fz: median and IQR over normalized progress",
                 fontsize=10, color=INK, loc="left")
    ax.set_xlabel("progress through trial (0 = start, 1 = end)",
                  fontsize=8, color=INK_SOFT)
    ax.set_ylabel("Fz world (N)", fontsize=8, color=INK_SOFT)
    if drew:
        ax.legend(fontsize=7, frameon=False, loc="lower left")


def _panel_strip(ax, rows, key, title, ylabel):
    """One-dimensional distribution by outcome; every point drawn, plus median."""
    rng = np.random.default_rng(0)   # reproducible jitter
    for i, (group, color, label) in enumerate((
        ([r for r in rows if r["accepted"]], OK_COLOR, "success"),
        ([r for r in rows if not r["accepted"]], FAIL_COLOR, "failure"),
    )):
        if not group:
            continue
        vals = np.array([r[key] for r in group], float)
        vals = vals[np.isfinite(vals)]
        if not len(vals):
            continue
        ax.scatter(np.full(len(vals), i) + rng.uniform(-0.09, 0.09, len(vals)),
                   vals, c=color, s=34, alpha=0.75, edgecolors="white",
                   linewidths=0.6, zorder=3)
        med = float(np.median(vals))
        ax.hlines(med, i - 0.26, i + 0.26, color=color, lw=2.4, zorder=4)
        ax.annotate(f"{med:.2f}", xy=(i + 0.30, med), fontsize=7.5, color=INK,
                    va="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["success", "failure"], fontsize=8, color=INK_SOFT)
    ax.set_xlim(-0.5, 1.7)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_ylabel(ylabel, fontsize=8, color=INK_SOFT)


def render(rows, meta, out_png: Path) -> None:
    has_start = any(abs(r["start_mag_mm"]) > 1e-9 for r in rows)
    has_socket = any(abs(r["socket_mag_mm"]) > 1e-9 for r in rows)
    # Difficulty is the socket-noise axis if varied, otherwise start randomization.
    diff_key, diff_label = (
        ("socket_mag_mm", "|socket observation noise| (mm)") if has_socket
        else ("start_mag_mm", "|start-pose randomization| (mm)")
    )

    fig = plt.figure(figsize=(15, 13.5), facecolor=SURFACE)
    gs = fig.add_gridspec(4, 2, hspace=0.42, wspace=0.22,
                          top=0.925, bottom=0.05, left=0.07, right=0.97)

    ok = sum(r["accepted"] for r in rows)
    fig.suptitle(
        f"{meta['file']}   |   {ok}/{len(rows)} = {100 * ok / max(len(rows), 1):.1f}% success"
        f"   |   base {'stochastic' if meta['stochastic'] else 'deterministic'}"
        f"   |   residual: {meta['residual_mode']}"
        + (f" ({meta['residual_params']}, scale {meta['residual_scale']})"
           if meta["residual_mode"] != "base only" else "")
        + f"\nsocket noise +/-{meta['socket_noise_mm'][0]:.2f} mm"
          f"   start randomization +/-{meta['start_noise_mm'][0]:.2f} mm",
        fontsize=11, color=INK, y=0.985,
    )

    _panel_timeline(fig.add_subplot(gs[0, :]), rows)
    _scatter_outcome(
        fig.add_subplot(gs[1, 0]), rows, "socket_dx_mm", "socket_dy_mm",
        "socket observation noise: where do failures sit?",
        "socket noise dx (mm)", "socket noise dy (mm)",
        limit_mm=float(meta["socket_noise_mm"][0]) or None,
    )
    ax_start = fig.add_subplot(gs[1, 1])
    if has_start:
        _scatter_outcome(
            ax_start, rows, "start_dx_mm", "start_dy_mm",
            "start-pose randomization: where do failures sit?",
            "start offset dx (mm)", "start offset dy (mm)",
            limit_mm=float(meta["start_noise_mm"][0]) or None,
        )
    else:
        ax_start.text(0.5, 0.5, "start randomization disabled\n"
                                "(all start offsets are exactly zero)",
                      ha="center", va="center", fontsize=9, color=INK_SOFT)
        ax_start.set_xticks([])
        ax_start.set_yticks([])
        for s in ax_start.spines.values():
            s.set_visible(False)

    _panel_time_vs_difficulty(fig.add_subplot(gs[2, 0]), rows, diff_key, diff_label)
    _panel_force(fig.add_subplot(gs[2, 1]), rows)
    _panel_strip(fig.add_subplot(gs[3, 0]), rows, "final_dz_mm",
                 "final dz at closest approach (seated vs rim stall)", "dz (mm)")
    _panel_strip(fig.add_subplot(gs[3, 1]), rows, "fz_min_n",
                 "most-negative Fz during trial", "min Fz (N)")

    for ax in fig.axes:
        ax.tick_params(labelsize=7.5, colors=INK_SOFT)
        ax.grid(True, color=GRID, lw=0.7, alpha=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)

    fig.savefig(out_png, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def summarize(rows, meta) -> None:
    ok = [r for r in rows if r["accepted"]]
    bad = [r for r in rows if not r["accepted"]]
    print(f"\n{meta['file']}")
    print(f"  {len(ok)}/{len(rows)} = {100 * len(ok) / max(len(rows), 1):.1f}% success"
          f"   (skipped {len(meta['skipped_trials'])} trial group(s) that never ran)")
    from collections import Counter
    print(f"  termination: {dict(Counter(r['termination_reason'] for r in rows))}")

    def cmp(key, unit=""):
        a = np.array([r[key] for r in ok], float) if ok else np.array([])
        b = np.array([r[key] for r in bad], float) if bad else np.array([])
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        sa = f"{a.mean():+7.3f}+-{a.std():5.3f}" if len(a) else "       -      "
        sb = f"{b.mean():+7.3f}+-{b.std():5.3f}" if len(b) else "       -      "
        print(f"    {key:<20} success {sa}  |  failure {sb} {unit}")

    print("\n  success vs failure:")
    for key, unit in (("socket_mag_mm", "mm"), ("start_mag_mm", "mm"),
                      ("duration_s", "s"), ("steps", ""), ("final_dz_mm", "mm"),
                      ("final_dy_mm", "mm"), ("peak_lateral_mm", "mm"),
                      ("fz_max_n", "N"), ("fz_min_n", "N"), ("saturation", "")):
        cmp(key, unit)

    if len(ok) >= 3:
        print("\n  among SUCCESSES -- does difficulty cost time?")
        for key in ("socket_mag_mm", "start_mag_mm"):
            x = np.array([r[key] for r in ok], float)
            y = np.array([r["duration_s"] for r in ok], float)
            if np.ptp(x) == 0:
                continue
            pear, spear = _corr(x, y)
            slope = np.polyfit(x, y, 1)[0]
            print(f"    duration vs |{key[:-3]}|: {slope:+.3f} s/mm   "
                  f"pearson {pear:+.3f}   spearman {spear:+.3f}")

    if bad and ok:
        mags = np.array([r["socket_mag_mm"] for r in rows], float)
        acc = np.array([r["accepted"] for r in rows], bool)
        med = float(np.median(mags))
        lo, hi = mags <= med, mags > med
        if lo.sum() and hi.sum():
            print(f"\n  split at median |socket noise| ({med:.2f} mm): "
                  f"below {acc[lo].sum()}/{lo.sum()} = {100 * acc[lo].mean():.0f}%"
                  f"   above {acc[hi].sum()}/{hi.sum()} = {100 * acc[hi].mean():.0f}%")


def write_csv(rows, out_csv: Path) -> None:
    fields = [k for k in rows[0] if not k.startswith("_")]
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def process(path: Path, outdir: Path, args) -> int:
    rows, meta = collect(path, args.env_yaml)
    if not rows:
        print(f"{path.name}: no trials with recorded actions; nothing to analyse")
        return 1
    notice = warn_if_mislabelled(rows, trust_auto=args.trust_auto_outcome)
    if notice:
        print(f"{path.name}: {notice}")
    if args.trust_auto_outcome:
        for r in rows:
            r["accepted"] = trial_accepted(r, trust_auto=True)
    if meta["skipped_trials"]:
        print(f"{path.name}: skipped {len(meta['skipped_trials'])} trial group(s) "
              f"that never ran: {', '.join(meta['skipped_trials'][:6])}"
              + (" ..." if len(meta["skipped_trials"]) > 6 else ""))
    summarize(rows, meta)
    if not args.no_csv:
        write_csv(rows, outdir / f"{path.stem}_trials.csv")
    if not args.no_plot:
        render(rows, meta, outdir / f"{path.stem}_analysis.png")
    print(f"  -> {outdir}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path, help="an .h5 session file, or a directory of them")
    p.add_argument("--out", type=Path, default=None,
                   help="output directory (default: alongside the file, in analysis/)")
    p.add_argument("--env-yaml", type=Path, default=None,
                   help="policy params/env.yaml (or checkpoint folder) to read the tool "
                        "offset from; default: the one the session was recorded with")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--no-csv", action="store_true")
    add_trust_argument(p)
    args = p.parse_args()

    src = args.path.expanduser().resolve()
    if src.is_dir():
        files = sorted(f for f in src.glob("*.h5")
                       if not f.name.endswith(".partial.h5"))
        outdir = args.out or src / "analysis"
    else:
        files = [src]
        outdir = args.out or src.parent / "analysis"
    if not files:
        print(f"no .h5 files under {src}")
        return 1
    outdir.mkdir(parents=True, exist_ok=True)

    status = 0
    for f in files:
        status |= process(f, outdir, args)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
