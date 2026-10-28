#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA OneWay Noncommercial License (March 22, 2022).
# See the LICENSE file in the repository root for the full license text:
# https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf

"""Render Phase 1 sessions to synchronised video + JSON.

Writes per trial ``<trial>.mp4`` (cameras plus plots up to the current step) and
``<trial>.json`` (observations, actions, timing, attributes). Action k is paired
with observation ``actions/mapping/observation_index[k]``; a trial has one more
observation than actions, so index-for-index pairing would be off by one.

Run it inside the container:

  ./docker/phase1/session.sh shell -c \
    'python3 "$SPARR_REPO"/tools/sparr_visualize.py <path>'
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
from sparr_trial_outcome import (  # noqa: E402
    add_trust_argument,
    trial_accepted,
)

#: Set once from --trust-auto-outcome; read by helpers that never see the parsed arguments.
TRUST_AUTO = [False]


# Categorical series colours (BGR). Aqua is low-contrast on the light surface, so
# every series also carries a direct label.
C1 = (214, 120, 42)    # blue   #2a78d6, BGR
C2 = (52, 104, 235)    # orange #eb6834
C3 = (122, 175, 27)    # aqua   #1baf7a
INK = (11, 11, 11)
INK_SOFT = (78, 81, 82)
GRID = (222, 226, 230)
RULE = (136, 136, 138)
SURFACE = (251, 252, 252)

CAM_PX = 360           # on-screen size of each camera panel
WIDTH = 1180
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale=0.42, color=INK_SOFT, thick=1):
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def _plot(img, rect, series, ylim, upto, title, hlines=(), n_steps=None):
    """Draw one strip: series = [(values, color, label), ...] drawn to `upto`.

    ``n_steps`` fixes the x-axis scale, so runs of different length
    (sparr_compare.py) share one timeline. A series shorter than n_steps
    freezes at its last point once upto exceeds it.
    """
    x0, y0, w, h = rect
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (245, 246, 247), -1)
    lo, hi = ylim
    span = (hi - lo) or 1.0
    n = max(n_steps or len(series[0][0]), 2)

    def px(i):
        return int(x0 + (i / (n - 1)) * w)

    def py(v):
        v = float(np.clip(v, lo, hi))
        return int(y0 + h - (v - lo) / span * h)

    if lo < 0.0 < hi:
        cv2.line(img, (x0, py(0.0)), (x0 + w, py(0.0)), GRID, 1, cv2.LINE_AA)
    for hv in hlines:
        if lo <= hv <= hi:
            y = py(hv)
            for xs in range(x0, x0 + w, 9):      # dashed
                cv2.line(img, (xs, y), (min(xs + 5, x0 + w), y), RULE, 1, cv2.LINE_AA)

    for values, color, label in series:
        k = min(upto, len(values) - 1)
        if k < 1:
            continue
        pts = np.array([[px(i), py(values[i])] for i in range(k + 1)], np.int32)
        cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)
        cv2.circle(img, tuple(pts[-1]), 4, color, -1, cv2.LINE_AA)

    if upto >= 1:
        cv2.line(img, (px(upto), y0), (px(upto), y0 + h), (215, 217, 219), 1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (208, 211, 214), 1)
    _text(img, title, (x0, y0 - 6), 0.44, INK)
    _text(img, f"{hi:.3g}", (x0 + w + 8, y0 + 10), 0.36)
    _text(img, f"{lo:.3g}", (x0 + w + 8, y0 + h), 0.36)

    # Direct labels, so identity is never by colour alone.
    lx = x0 + 8
    for _v, color, label in series:
        (tw, _), _ = cv2.getTextSize(label, FONT, 0.38, 1)
        cv2.circle(img, (lx + 4, y0 + 12), 4, color, -1, cv2.LINE_AA)
        _text(img, label, (lx + 12, y0 + 16), 0.38, INK)
        lx += tw + 26


def _cam_panel(rgb, size, label, border=(208, 211, 214), border_thick=1):
    """One camera panel, always ``size`` x ``size`` on the canvas.

    The frame is letterboxed, not stretched, to keep the geometry the policy saw.
    """
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    fitted = cv2.resize(img, (max(1, int(round(w * scale))),
                              max(1, int(round(h * scale)))),
                        interpolation=cv2.INTER_NEAREST)
    panel = np.full((size, size, 3), SURFACE, np.uint8)
    y0 = (size - fitted.shape[0]) // 2
    x0 = (size - fitted.shape[1]) // 2
    panel[y0:y0 + fitted.shape[0], x0:x0 + fitted.shape[1]] = fitted
    cv2.rectangle(panel, (0, 0), (size - 1, size - 1), border, border_thick)
    cv2.rectangle(panel, (0, 0), (size, 22), (255, 255, 255), -1)
    _text(panel, label, (8, 16), 0.46, INK)
    return panel


def _trial_arrays(trial, cams):
    obs = trial["observations"]
    act = trial["actions"]
    idx = act["mapping/observation_index"][:].astype(int)

    data = {
        "obs_index_for_action": idx,
        "dist": obs["task/eef_keypoint_to_socket_keypoint_distance_m"][:],
        "fz": obs["robot/ext_wrench_world"][:, 2],
        "policy_input": obs["policy/input"][:],
        "flange": obs["robot/flange_pose_world_xyzw"][:],
        "tcp": obs["robot/tcp_pose_world_xyzw"][:],
        "mu": act["policy/mu_raw_normalized"][:],
        "clipped": act["policy/sample_clipped_normalized"][:],
        "delta": act["action/delta_scaled_pose_rel"][:],
        "rgb": {c: obs[f"camera/{c}/rgb"] for c in cams},
        # Present only when recorded with recording.record_source_frames.
        "source_rgb": {
            c: obs[f"extra/{c}_source_rgb"]
            for c in cams if f"extra/{c}_source_rgb" in obs
        },
        "tick_ns": obs["timing/control_tick_monotonic_ns"][:],
    }
    for c in cams:
        data[f"{c}_stamp"] = obs[f"timing/{c}_stamp_ns"][:]
        data[f"{c}_age_ms"] = obs[f"timing/{c}_age_ns"][:] / 1e6
    if "timing/camera_skew_ns" in obs:
        data["skew_ms"] = obs["timing/camera_skew_ns"][:] / 1e6
    return data


def load_run_arrays(trial):
    """Per-trial arrays for sparr_compare.py beyond ``_trial_arrays``.

    Returns the sampled action, commanded action, requested residual and
    noise/start attributes; fields absent from the trial come back as None.
    """
    act = trial["actions"]
    attrs = trial.attrs
    sample = act["policy/sample_raw_normalized"][:] \
        if "policy/sample_raw_normalized" in act else None
    # Commanded action, post-clip and including any residual (unlike mu_raw_normalized).
    effective = act["action/effective_software_action_normalized"][:] \
        if "action/effective_software_action_normalized" in act else None
    residual = trial.get("labels/extra/requested_residual")
    residual = residual[:] if residual is not None else None
    residual_applied = trial.get("labels/extra/residual_applied")
    residual_applied = residual_applied[:] if residual_applied is not None else None
    return {
        "sample": sample,
        "effective": effective,
        "residual": residual,
        "residual_applied": residual_applied,
        "socket_offset_xyz_rpy": np.asarray(
            attrs.get("socket_policy_offset_xyz_rpy", [np.nan] * 6), float
        ),
        "start_offset_xyz_rpy": np.asarray(
            attrs.get("start_offset_xyz_rpy", [np.nan] * 6), float
        ),
        "accepted": trial_accepted(attrs, trust_auto=TRUST_AUTO[0]),
        "accepted_recorded": bool(attrs.get("accepted", False)),
        "auto_outcome": str(attrs.get("auto_outcome", "")),
        "operator_label": str(attrs.get("operator_label", "")),
        "attempt_index": int(attrs.get("attempt_index", -1)),
    }


def render_trial(trial, name, cams, out_mp4, fps, threshold, codec="avc1",
                 slowdown=1, source_frames=False):
    d = _trial_arrays(trial, cams)
    idx = d["obs_index_for_action"]
    n_steps = len(idx)
    if n_steps < 2:
        return False

    ncam = len(cams)
    cam_w = ncam * CAM_PX + (ncam - 1) * 10
    top_h = CAM_PX
    plot_x = 60
    plot_w = WIDTH - plot_x - 92
    strips = 4
    strip_h = 84
    height = 18 + top_h + 30 + strips * (strip_h + 30) + 10

    # avc1 (H.264) by default: some players refuse mp4v's MPEG-4 Part 2 stream.
    writer = cv2.VideoWriter(
        str(out_mp4), cv2.VideoWriter_fourcc(*codec), fps, (WIDTH, height)
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"cv2 could not open a {codec!r} writer for {out_mp4}; "
            "try --codec mp4v"
        )

    mu, dist, fz = d["mu"], d["dist"], d["fz"]
    # Elapsed time from recorded tick timestamps, independent of --fps / --slowdown.
    tick_s = (d["tick_ns"][idx] - d["tick_ns"][idx[0]]) / 1e9
    elapsed_s = np.asarray(tick_s, np.float64)
    # Resample per-observation traces onto action steps to match the shown frame.
    dist_s = dist[idx]
    fz_s = fz[idx]
    a_lim = float(max(2.0, np.abs(mu).max() * 1.05))
    d_lo, d_hi = float(dist_s.min()) * 0.95, float(dist_s.max()) * 1.05
    f_lim = float(max(10.0, np.abs(fz_s).max() * 1.1))

    for k in range(n_steps):
        i = int(idx[k])
        canvas = np.full((height, WIDTH, 3), SURFACE, np.uint8)

        x = 16
        for c in cams:
            frames = d["source_rgb"] if source_frames else d["rgb"]
            tag = " raw" if source_frames else ""
            panel = _cam_panel(frames[c][i], CAM_PX,
                               f"{c}{tag}   age {d[f'{c}_age_ms'][i]:.0f} ms")
            canvas[18:18 + CAM_PX, x:x + CAM_PX] = panel
            x += CAM_PX + 10

        ix = 16 + cam_w + 22
        lines = [
            (f"{name}", 0.6, INK),
            (f"step {k + 1}/{n_steps}   obs #{i}   t = {elapsed_s[k]:.2f} s",
             0.46, INK_SOFT),
            ("", 0.4, INK_SOFT),
            (f"eef->socket   {dist_s[k] * 1000:7.1f} mm", 0.5, INK),
            (f"Fz            {fz_s[k]:7.1f} N", 0.5, INK),
            ("", 0.4, INK_SOFT),
            (f"|mu|max       {np.abs(mu[k]).max():7.2f}", 0.46, INK_SOFT),
            (f"clipped dims  {int((np.abs(mu[k]) >= 1.0).sum())}/6", 0.46, INK_SOFT),
        ]
        if "skew_ms" in d:
            lines.append((f"cam pair skew {d['skew_ms'][i]:7.1f} ms", 0.46, INK_SOFT))
        yy = 46
        for s, sc, col in lines:
            if s:
                _text(canvas, s, (ix, yy), sc, col, 1)
            yy += 26

        y = 18 + top_h + 34
        _plot(canvas, (plot_x, y, plot_w, strip_h),
              [(dist_s, C1, "eef->socket distance (m)")],
              (d_lo, d_hi), k, "distance to socket keypoint",
              hlines=(threshold,) if threshold else ())
        y += strip_h + 30
        _plot(canvas, (plot_x, y, plot_w, strip_h),
              [(mu[:, 0], C1, "dx"), (mu[:, 1], C2, "dy"), (mu[:, 2], C3, "dz")],
              (-a_lim, a_lim), k, "action mean, translation (normalized)",
              hlines=(-1.0, 1.0))
        y += strip_h + 30
        _plot(canvas, (plot_x, y, plot_w, strip_h),
              [(mu[:, 3], C1, "rx"), (mu[:, 4], C2, "ry"), (mu[:, 5], C3, "rz")],
              (-a_lim, a_lim), k, "action mean, rotation (normalized)",
              hlines=(-1.0, 1.0))
        y += strip_h + 30
        _plot(canvas, (plot_x, y, plot_w, strip_h),
              [(fz_s, C1, "Fz world (N)")], (-f_lim, f_lim), k, "contact force")

        # Repeat frames rather than lowering fps; some players mishandle very low rates.
        for _ in range(slowdown):
            writer.write(canvas)
    writer.release()
    return True


def dump_json(trial, name, cams, out_json):
    d = _trial_arrays(trial, cams)
    attrs = {}
    for k, v in trial.attrs.items():
        attrs[k] = v.tolist() if isinstance(v, np.ndarray) else (
            v.decode() if isinstance(v, bytes) else
            (bool(v) if isinstance(v, np.bool_) else
             (v.item() if isinstance(v, np.generic) else v))
        )
    payload = {
        "trial": name,
        "attrs": attrs,
        "sync": {
            "note": "action k pairs with observation obs_index_for_action[k]; "
                    "the camera frame shown is that observation's frame",
            "obs_index_for_action": d["obs_index_for_action"].tolist(),
        },
        "observations": {
            "policy_input_18d": d["policy_input"].tolist(),
            "eef_to_socket_distance_m": d["dist"].tolist(),
            "fz_world_n": d["fz"].tolist(),
            "flange_pose_world_xyzw": d["flange"].tolist(),
            "tcp_pose_world_xyzw": d["tcp"].tolist(),
        },
        "actions": {
            "mu_raw_normalized": d["mu"].tolist(),
            "sample_clipped_normalized": d["clipped"].tolist(),
            "delta_scaled_pose_rel": d["delta"].tolist(),
        },
        "timing": {
            "control_tick_monotonic_ns": d["tick_ns"].tolist(),
            **{f"{c}_stamp_ns": d[f"{c}_stamp"].tolist() for c in cams},
            **{f"{c}_age_ms": d[f"{c}_age_ms"].tolist() for c in cams},
            **({"camera_pair_skew_ms": d["skew_ms"].tolist()} if "skew_ms" in d else {}),
        },
    }
    out_json.write_text(json.dumps(payload, indent=1))


def process(path: Path, outdir: Path, args) -> None:
    with h5py.File(path, "r", locking=False) as f:
        trials = f.get("trials")
        if trials is None:
            print(f"  {path.name}: no trials group, skipped")
            return
        try:
            threshold = float(
                np.asarray(f["metadata/collection_config/success/"
                             "distance_threshold_m"][()]).ravel()[0]
            )
        except Exception:
            threshold = 0.0
        names = sorted(trials.keys())
        dest = outdir / path.stem
        dest.mkdir(parents=True, exist_ok=True)
        for name in names:
            t = trials[name]
            accepted = trial_accepted(t.attrs, trust_auto=TRUST_AUTO[0])
            if args.accepted_only and not accepted:
                continue
            if args.trials and name not in args.trials:
                continue
            # A trial discarded before it ran has no actions group; skip it.
            missing = [g for g in ("observations", "actions") if g not in t]
            if missing:
                print(f"  {name} [{t.attrs.get('operator_label', '?')}] "
                      f"skipped: no {'/'.join(missing)} (never ran)")
                continue
            cams = sorted(t["observations/camera"].keys())
            tag = "accepted" if accepted else str(t.attrs.get("operator_label", "?"))
            if accepted and not bool(t.attrs.get("accepted", False)):
                tag = "accepted*"   # by the automated check, not the stored flag
            print(f"  {name} [{tag}] cams={','.join(cams)} ... ", end="", flush=True)
            if not args.no_json:
                dump_json(t, name, cams, dest / f"{name}.json")
            if not args.no_video:
                if args.source_frames:
                    missing = [c for c in cams
                               if f"extra/{c}_source_rgb" not in t["observations"]]
                    if missing:
                        print(f"no raw frames for {','.join(missing)} "
                              "(session not recorded with record_source_frames)")
                        continue
                ok = render_trial(t, name, cams, dest / f"{name}.mp4",
                                  args.fps, threshold, args.codec,
                                  args.slowdown, args.source_frames)
                if not ok:
                    print("too short, skipped")
                    continue
            print("done")
        print(f"  -> {dest}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="an .h5 file, or a directory of them")
    p.add_argument("--out", default=None,
                   help="output directory (default: <path>/viz or alongside the file)")
    p.add_argument("--fps", type=float, default=15.0,
                   help="video frame rate; 15 is real time at the 15 Hz control rate")
    p.add_argument("--source-frames", action="store_true",
                   help="render the RAW frames as captured (848x480) instead of "
                        "the cropped/resized ones the policy saw. Only available "
                        "for sessions recorded with recording.record_source_frames; "
                        "letterboxed into the square panel so the aspect ratio is "
                        "preserved.")
    p.add_argument("--slowdown", type=int, default=1, metavar="N",
                   help="hold every step for N video frames, so playback is N x "
                        "slower while the frame rate stays player-friendly. "
                        "Prefer this to a very low --fps: some players handle "
                        "sub-10 fps badly and scrubbing gets chunky. The on-screen "
                        "elapsed time comes from the recording either way.")
    p.add_argument("--accepted-only", action="store_true",
                   help="only render trials the operator accepted")
    p.add_argument("--trials", nargs="*", default=None,
                   help="explicit trial names, e.g. trial_000003")
    p.add_argument("--codec", default="avc1",
                   help="fourcc for the video writer (default avc1 = H.264; "
                        "use mp4v if avc1 is unavailable)")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--no-json", action="store_true")
    add_trust_argument(p)
    args = p.parse_args()
    TRUST_AUTO[0] = bool(args.trust_auto_outcome)

    src = Path(args.path).expanduser().resolve()
    if src.is_dir():
        files = sorted(f for f in src.glob("*.h5") if not f.name.endswith(".partial.h5"))
        outdir = Path(args.out) if args.out else src / "viz"
    else:
        files = [src]
        outdir = Path(args.out) if args.out else src.parent / "viz"
    if not files:
        print(f"no .h5 files under {src}")
        return 1
    outdir.mkdir(parents=True, exist_ok=True)
    for f in files:
        print(f"{f.name}")
        process(f, outdir, args)
    print(f"\nwrote to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
