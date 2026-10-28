#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA License.
# See the LICENSE file in the repository root for the full license text.

"""Render matching trials from 2-3 Phase 1/2 sessions side by side.

Writes one video (and optionally JSON) per matched trial: one camera feed per
run, and signed per-axis strips with one line per run (distance to socket,
commanded action, base-policy mean, residual and sample deviation when present,
contact Fz). Trials are matched by attempt_index, which selects the same
randomization in sessions that share a seed; use --match-by name otherwise.

Run inside the container:

  ./docker/phase1/session.sh shell -c \
    'python3 "$SPARR_REPO"/tools/sparr_compare.py \
       A.h5 B.h5 [C.h5] --labels "base only" "base+residual"'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sparr_visualize import (  # noqa: E402
    C1, C2, C3, INK_SOFT, SURFACE, TRUST_AUTO,
    _cam_panel, _plot, _text, _trial_arrays, load_run_arrays,
)
from sparr_trial_outcome import add_trust_argument  # noqa: E402

MAX_RUNS = 3
RUN_COLORS = (C1, C2, C3)
#: The six action dimensions, in pipeline order.
AXIS_NAMES = ("dx", "dy", "dz", "rx", "ry", "rz")
CAM_PX = 300
INFO_H = 96
STRIP_H = 84
STRIP_GAP = 30


def _session_policy_config(handle: h5py.File) -> dict:
    """Session-level settings that decide which extra strips this run needs."""
    cfg = handle.get("metadata/collection_config")
    stochastic = False
    if cfg is not None and "policy/stochastic" in cfg:
        stochastic = bool(np.asarray(cfg["policy/stochastic"][()]).ravel()[0])
    return {"stochastic": stochastic}


def _matchable_trials(handle: h5py.File, match_by: str) -> dict:
    """key -> trial name, for every trial that actually ran (has recorded actions)."""
    trials = handle.get("trials")
    if trials is None:
        return {}
    out = {}
    for name in trials:
        t = trials[name]
        if "observations" not in t or "actions" not in t:
            continue
        if int(t.attrs.get("observation_count", 0)) < 3:
            continue
        key = int(t.attrs["attempt_index"]) if match_by == "attempt_index" else name
        out[key] = name
    return out


def _residual_provenance(handle: h5py.File) -> str:
    mode = handle.attrs.get("residual_mode")
    if mode is None:
        return "base only"
    params = handle.attrs.get("residual_params_path", "")
    scale = handle.attrs.get("residual_action_scale", "?")
    det = handle.attrs.get("residual_deterministic", "?")
    tag = Path(str(params)).name if params else "(uninitialised)"
    return f"{mode}: {tag}  scale={scale}  det={det}"


def _magnitude(offset_xyz_rpy: np.ndarray) -> tuple[float, float, float]:
    dx, dy = offset_xyz_rpy[0] * 1000.0, offset_xyz_rpy[1] * 1000.0
    return float(np.hypot(dx, dy)), dx, dy


def render_comparison(
    handles: list[h5py.File],
    names: list[str],
    labels: list[str],
    cams: list[str],
    out_mp4: Path,
    fps: float,
    codec: str,
    slowdown: int = 1,
    axes: str = "trans",
) -> bool:
    n_runs = len(handles)
    trials = [handles[r]["trials"][names[r]] for r in range(n_runs)]
    tarr = [_trial_arrays(trials[r], [cams[r]]) for r in range(n_runs)]
    rarr = [load_run_arrays(trials[r]) for r in range(n_runs)]
    idx = [tarr[r]["obs_index_for_action"] for r in range(n_runs)]
    n_steps = [len(idx[r]) for r in range(n_runs)]
    if max(n_steps) < 2:
        return False
    n_total = max(n_steps)

    session_cfg = [_session_policy_config(handles[r]) for r in range(n_runs)]
    has_residual = [rarr[r]["residual"] is not None for r in range(n_runs)]
    has_stochastic = [session_cfg[r]["stochastic"] for r in range(n_runs)]
    show_residual = any(has_residual)
    show_stochastic = any(has_stochastic)
    show_translation = axes in ("trans", "both")
    show_rotation = axes in ("rot", "both")

    show_effective = any(
        rarr[r]["effective"] is not None for r in range(n_runs)
    )
    # Each action group costs 3 strips (6 with rotation).
    per_group = 3 * (int(show_translation) + int(show_rotation))
    n_strips = (1                                   # distance
                + per_group * int(show_effective)   # commanded
                + per_group                         # base mean
                + per_group * int(show_residual)    # residual
                + 1                                 # Fz
                + int(show_stochastic))
    width = 32 + n_runs * CAM_PX + (n_runs - 1) * 10
    top_h = CAM_PX + INFO_H
    height = 18 + top_h + 20 + n_strips * (STRIP_H + STRIP_GAP) + 10
    plot_x = 60
    plot_w = width - plot_x - 92

    writer = cv2.VideoWriter(
        str(out_mp4), cv2.VideoWriter_fourcc(*codec), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cv2 could not open a {codec!r} writer for {out_mp4}")

    # Every series is resampled onto action steps.
    dist_mm = [tarr[r]["dist"][idx[r]] * 1000.0 for r in range(n_runs)]
    fz = [tarr[r]["fz"][idx[r]] for r in range(n_runs)]
    # Signed per axis, not L2 magnitude. commanded = post-clip with residual;
    # base mean = pre-clip policy output without residual.
    effective = [rarr[r]["effective"] for r in range(n_runs)]
    base_mean = [tarr[r]["mu"] for r in range(n_runs)]
    residual = [rarr[r]["residual"] if has_residual[r] else None
                for r in range(n_runs)]
    sample_dev = [
        np.linalg.norm(rarr[r]["sample"] - tarr[r]["mu"], axis=1)
        if rarr[r]["sample"] is not None else None
        for r in range(n_runs)
    ]

    d_lo = min(float(np.min(x)) for x in dist_mm) * 0.95
    d_hi = max(float(np.max(x)) for x in dist_mm) * 1.05
    f_lim = max(10.0, max(float(np.abs(x).max()) for x in fz) * 1.1)
    def sym_limit(arrays, cols, floor=1.0):
        """Symmetric +-limit shared by the axes of one group.

        Per group because post-clip commanded actions span a far smaller range
        than the pre-clip mean.
        """
        vals = [float(np.abs(a[:, cols]).max()) for a in arrays if a is not None]
        return max(floor, (max(vals) if vals else floor) * 1.15)

    eff_t_lim = sym_limit(effective, slice(0, 3))
    eff_r_lim = sym_limit(effective, slice(3, 6))
    mu_t_lim = sym_limit(base_mean, slice(0, 3))
    mu_r_lim = sym_limit(base_mean, slice(3, 6))
    res_t_lim = sym_limit(residual, slice(0, 3), floor=0.1)
    res_r_lim = sym_limit(residual, slice(3, 6), floor=0.1)
    dev_hi = (
        max(0.1, max(float(x.max()) for x in sample_dev if x is not None) * 1.15)
        if show_stochastic else 1.0
    )

    outcome = ["success" if rarr[r]["accepted"] else rarr[r]["operator_label"]
              for r in range(n_runs)]
    provenance = [_residual_provenance(handles[r]) for r in range(n_runs)]

    for k in range(n_total):
        canvas = np.full((height, width, 3), SURFACE, np.uint8)

        x = 16
        for r in range(n_runs):
            local = min(k, n_steps[r] - 1)
            i = int(idx[r][local])
            frozen = k >= n_steps[r]
            panel = _cam_panel(
                tarr[r]["rgb"][cams[r]][i], CAM_PX,
                f"{labels[r]}" + ("  (ended)" if frozen else f"  {local + 1}/{n_steps[r]}"),
                border=RUN_COLORS[r], border_thick=2,
            )
            canvas[18:18 + CAM_PX, x:x + CAM_PX] = panel

            sock_m, sock_dx, sock_dy = _magnitude(rarr[r]["socket_offset_xyz_rpy"])
            start_m, start_dx, start_dy = _magnitude(rarr[r]["start_offset_xyz_rpy"])
            info = [
                f"noise {sock_m:.2f} mm (dx{sock_dx:+.2f} dy{sock_dy:+.2f})",
                f"start {start_m:.2f} mm (dx{start_dx:+.2f} dy{start_dy:+.2f})",
                provenance[r],
                f"outcome: {outcome[r]}",
            ]
            iy = 18 + CAM_PX + 16
            for line in info:
                _text(canvas, line, (x + 4, iy), 0.38, INK_SOFT)
                iy += 18
            x += CAM_PX + 10

        y = 18 + top_h + 20

        def strip(series, ylim, title, hlines=()):
            nonlocal y
            _plot(canvas, (plot_x, y, plot_w, STRIP_H), series, ylim, k, title,
                  hlines=hlines, n_steps=n_total)
            y += STRIP_H + STRIP_GAP

        strip(
            [(dist_mm[r], RUN_COLORS[r], labels[r]) for r in range(n_runs)],
            (d_lo, d_hi), "distance to socket keypoint (mm)",
        )

        def axis_strips(arrays, cols, limit, title_fmt):
            """One signed strip per axis, one line per run on each."""
            for offset, col in enumerate(cols):
                series = [(arrays[r][:, col], RUN_COLORS[r], labels[r])
                          for r in range(n_runs) if arrays[r] is not None]
                if not series:
                    continue
                strip(series, (-limit, limit),
                      title_fmt.format(axis=AXIS_NAMES[col]),
                      hlines=(-1.0, 1.0))

        cols_t, cols_r = range(0, 3), range(3, 6)
        if show_effective:
            if show_translation:
                axis_strips(effective, cols_t, eff_t_lim,
                            "COMMANDED action {axis} (normalized, post-clip, "
                            "residual included)")
            if show_rotation:
                axis_strips(effective, cols_r, eff_r_lim,
                            "COMMANDED action {axis} (normalized, post-clip, "
                            "residual included)")
        if show_translation:
            axis_strips(base_mean, cols_t, mu_t_lim,
                        "base policy mean {axis} (pre-clip, residual NOT included)")
        if show_rotation:
            axis_strips(base_mean, cols_r, mu_r_lim,
                        "base policy mean {axis} (pre-clip, residual NOT included)")
        if show_residual:
            if show_translation:
                axis_strips(residual, cols_t, res_t_lim,
                            "requested residual {axis} (normalized, post action_scale)")
            if show_rotation:
                axis_strips(residual, cols_r, res_r_lim,
                            "requested residual {axis} (normalized, post action_scale)")
        strip(
            [(fz[r], RUN_COLORS[r], labels[r]) for r in range(n_runs)],
            (-f_lim, f_lim), "contact force Fz world (N)",
        )
        if show_stochastic:
            strip(
                [(sample_dev[r], RUN_COLORS[r], labels[r]) for r in range(n_runs)
                 if sample_dev[r] is not None],
                (0.0, dev_hi), "sampled action deviation from mean (normalized)",
            )

        writer.write(canvas)
    writer.release()
    return True


def dump_json(handles, names, labels, cams, out_json: Path) -> None:
    payload = {"runs": []}
    for r, handle in enumerate(handles):
        t = handle["trials"][names[r]]
        d = _trial_arrays(t, [cams[r]])
        e = load_run_arrays(t)
        payload["runs"].append({
            "label": labels[r],
            "trial": names[r],
            "attempt_index": e["attempt_index"],
            "accepted": e["accepted"],
            "operator_label": e["operator_label"],
            "socket_offset_xyz_rpy": e["socket_offset_xyz_rpy"].tolist(),
            "start_offset_xyz_rpy": e["start_offset_xyz_rpy"].tolist(),
            "residual_provenance": _residual_provenance(handle),
            "eef_to_socket_distance_m": d["dist"].tolist(),
            "fz_world_n": d["fz"].tolist(),
            "mu_raw_normalized": d["mu"].tolist(),
            "effective_software_action_normalized": (
                e["effective"].tolist() if e["effective"] is not None else None
            ),
            "sample_raw_normalized": (
                e["sample"].tolist() if e["sample"] is not None else None
            ),
            "requested_residual": (
                e["residual"].tolist() if e["residual"] is not None else None
            ),
        })
    out_json.write_text(json.dumps(payload, indent=1))


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("files", nargs="+", type=Path,
                   help=f"2-{MAX_RUNS} .h5 session files to compare")
    p.add_argument("--labels", nargs="+", default=None,
                   help="one label per file (default: the file's own name, since "
                        "the experiment directory name is usually too long for the "
                        "legend); always specify these explicitly for anything "
                        "you intend to read closely")
    p.add_argument("--match-by", choices=("attempt_index", "name"),
                   default="attempt_index",
                   help="how a trial in one file is paired with a trial in another")
    p.add_argument("--attempts", nargs="*", type=int, default=None,
                   help="explicit keys to render (attempt_index or trial-name-derived "
                        "int, per --match-by); default renders every match found")
    p.add_argument("--limit", type=int, default=12,
                   help="cap on how many matched trials to render (0 = no cap); "
                        "dropped matches are reported, never silently truncated")
    p.add_argument("--camera", default="wrist_1",
                   help="which camera to show per run (default wrist_1); a run "
                        "missing it falls back to its first available camera")
    p.add_argument("--out", type=Path, default=None,
                   help="output directory (default: <first file>/../compare)")
    p.add_argument("--fps", type=float, default=15.0,
                   help="video frame rate; 15 is real time at the 15 Hz control rate")
    p.add_argument("--axes", choices=("trans", "rot", "both"), default="trans",
                   help="which action dimensions to plot, one signed strip per "
                        "axis: trans = dx/dy/dz (default), rot = rx/ry/rz, "
                        "both = all six. 'both' triples the strip count, so the "
                        "video gets tall.")
    p.add_argument("--slowdown", type=int, default=1, metavar="N",
                   help="hold every step for N video frames, so playback is N x "
                        "slower while the frame rate stays player-friendly. "
                        "Prefer this to a very low --fps: some players handle "
                        "sub-10 fps badly and scrubbing gets chunky.")
    p.add_argument("--codec", default="avc1")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--no-json", action="store_true")
    add_trust_argument(p)
    args = p.parse_args()
    # Shared with sparr_visualize.load_run_arrays.
    TRUST_AUTO[0] = bool(args.trust_auto_outcome)

    if not 2 <= len(args.files) <= MAX_RUNS:
        p.error(f"give 2-{MAX_RUNS} files, got {len(args.files)}")
    labels = args.labels or [f.stem for f in args.files]
    if len(labels) != len(args.files):
        p.error("--labels must have exactly one entry per file")

    outdir = args.out or (args.files[0].parent / "compare")
    outdir.mkdir(parents=True, exist_ok=True)

    handles = [h5py.File(f, "r", locking=False) for f in args.files]
    try:
        maps = [_matchable_trials(h, args.match_by) for h in handles]
        common = set(maps[0])
        for m in maps[1:]:
            common &= set(m)
        if not common:
            print("no trials matched across all files"
                  f" (by {args.match_by}); nothing to render")
            return 1

        keys = sorted(common)
        if args.attempts is not None:
            missing = [k for k in args.attempts if k not in common]
            if missing:
                print(f"not matched in every file, skipped: {missing}")
            keys = [k for k in args.attempts if k in common]
        elif args.limit and len(keys) > args.limit:
            print(f"{len(keys)} matched trials, rendering the first {args.limit} "
                  f"(--limit 0 for all, or --attempts to pick specific ones): "
                  f"dropped {keys[args.limit:]}")
            keys = keys[:args.limit]
        if not keys:
            print("none of the requested keys matched in every file; nothing to render")
            return 1

        # Per-run fallback when --camera is absent, from the first matched trial.
        cams = []
        for r, handle in enumerate(handles):
            available = sorted(handle["trials"][maps[r][keys[0]]]["observations/camera"].keys())
            if args.camera in available:
                cams.append(args.camera)
            else:
                cams.append(available[0])
                print(f"{labels[r]}: no {args.camera!r} camera, using {available[0]!r}")

        for key in keys:
            names = [maps[r][key] for r in range(len(handles))]
            tag = f"attempt_{key:04d}" if args.match_by == "attempt_index" else str(key)
            print(f"{tag}  " + " | ".join(f"{labels[r]}={names[r]}"
                                          for r in range(len(handles))) + " ... ",
                  end="", flush=True)
            if not args.no_video:
                ok = render_comparison(
                    handles, names, labels, cams, outdir / f"{tag}.mp4",
                    args.fps, args.codec, args.slowdown, args.axes,
                )
                if not ok:
                    print("too short, skipped")
                    continue
            if not args.no_json:
                dump_json(handles, names, labels, cams, outdir / f"{tag}.json")
            print("done")
    finally:
        for h in handles:
            h.close()

    print(f"\nwrote to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
