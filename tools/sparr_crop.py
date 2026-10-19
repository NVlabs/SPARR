#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Horizontally crop the camera streams of a Phase 1 recording.

Two modes:

  --preview TRIAL   render one trial's cropped streams to mp4, change nothing
  (default)         write a NEW .h5 with the cropped streams, everything else
                    byte-identical

Only ``observations/camera/*/rgb`` (and the matching calibration/shape
attributes) change; everything else is copied verbatim. ``--fraction`` is taken
off each side.

Run inside the container:

  ./docker/phase1/session.sh shell -c \
    'python3 "$SPARR_REPO"/tools/sparr_crop.py <src.h5> --preview trial_000000'
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np

CAMERA_RGB_SUFFIX = "/rgb"
CAMERA_PREFIX = "observations/camera/"


def crop_bounds(width: int, left: float, right: float) -> tuple[int, int]:
    """(x0, new_width) for a horizontal crop, fractions off each side."""
    cut_l = int(round(width * left))
    cut_r = int(round(width * right))
    new_width = width - cut_l - cut_r
    if new_width <= 0:
        raise ValueError(
            f"crop {left}/{right} removes the whole frame ({width} px wide)"
        )
    return cut_l, new_width


def parse_crop_specs(specs, default_fraction: float) -> dict:
    """``--crop wrist_1=0.30,0.20`` -> {"wrist_1": (0.30, 0.20)}.

    Cameras without an entry use the symmetric default.
    """
    table: dict[str, tuple[float, float]] = {}
    for spec in specs or ():
        if "=" not in spec:
            raise ValueError(f"--crop needs NAME=LEFT,RIGHT, got {spec!r}")
        name, _, pair = spec.partition("=")
        parts = pair.split(",")
        if len(parts) != 2:
            raise ValueError(f"--crop needs NAME=LEFT,RIGHT, got {spec!r}")
        left, right = (float(v) for v in parts)
        if left < 0 or right < 0:
            raise ValueError(f"--crop fractions must be >= 0, got {spec!r}")
        table[name.strip()] = (left, right)
    table.setdefault("__default__", (default_fraction, default_fraction))
    return table


def crop_for(table: dict, camera: str) -> tuple[float, float]:
    return table.get(camera, table["__default__"])


def _camera_names(trial: h5py.Group) -> list[str]:
    return sorted(trial["observations/camera"].keys())


def render_preview(src: Path, trial_name: str, table: dict, out_mp4: Path,
                   fps: float, codec: str) -> tuple[int, list]:
    with h5py.File(src, "r", locking=False) as f:
        trial = f["trials"][trial_name]
        cams = _camera_names(trial)
        streams = [trial[f"observations/camera/{c}/rgb"] for c in cams]
        n, h, w, _ = streams[0].shape
        geom = []
        for c in cams:
            left, right = crop_for(table, c)
            x0, cw = crop_bounds(w, left, right)
            geom.append((c, x0, cw, left, right))

        header = 54
        frame_w = sum(g[2] for g in geom) + (len(cams) - 1) * 8
        writer = cv2.VideoWriter(
            str(out_mp4), cv2.VideoWriter_fourcc(*codec), fps,
            (frame_w, h + header),
        )
        if not writer.isOpened():
            raise RuntimeError(f"cv2 could not open a {codec!r} writer for {out_mp4}")

        for i in range(n):
            canvas = np.full((h + header, frame_w, 3), 245, np.uint8)
            x = 0
            for (name, x0, cw, left, right), ds in zip(geom, streams):
                patch = np.asarray(ds[i, :, x0:x0 + cw, :])
                canvas[header:, x:x + cw] = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                cv2.putText(canvas,
                            f"{name}  L{int(left*100)}% R{int(right*100)}%  {cw}x{h}",
                            (x + 8, 44), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (11, 11, 11), 1, cv2.LINE_AA)
                x += cw + 8
            label = f"{trial_name}   frame {i + 1}/{n}   source {w}x{h}"
            cv2.putText(canvas, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (78, 81, 82), 1, cv2.LINE_AA)
            writer.write(canvas)
        writer.release()
        return n, geom



#: (label, width, height). "sq" distorts; "ar" keeps a 424:480 cropped aspect.
DEFAULT_RESIZES = [
    ("native", 0, 0),
    ("128x128 sq", 128, 128),
    ("113x128 ar", 113, 128),
    ("128x145 ar", 128, 145),
    ("160x181 ar", 160, 181),
    ("192x217 ar", 192, 217),
    ("226x256 ar", 226, 256),
    ("256x290 ar", 256, 290),
    ("256x256 sq", 256, 256),
]


def render_resize_compare(src: Path, trial_name: str, table: dict, out_mp4: Path,
                          fps: float, codec: str, panel_h: int = 240) -> tuple:
    """One row per camera, one column per candidate resize.

    Each option is resized, then upscaled with INTER_NEAREST at its own aspect,
    so detail loss and aspect distortion stay visible.
    """
    with h5py.File(src, "r", locking=False) as f:
        trial = f["trials"][trial_name]
        cams = _camera_names(trial)
        streams = [trial[f"observations/camera/{c}/rgb"] for c in cams]
        n, h, w, _ = streams[0].shape
        geom = []
        for c in cams:
            left, right = crop_for(table, c)
            x0, cw = crop_bounds(w, left, right)
            geom.append((c, x0, cw))
        _, _, cw0 = geom[0]

        opts = []
        for label, ow, oh in DEFAULT_RESIZES:
            if ow == 0:
                ow, oh = cw0, h
                label = f"native {ow}x{oh}"
            disp_w = max(1, int(round(ow * panel_h / oh)))
            opts.append((label, ow, oh, disp_w))

        gap, head, lab = 10, 30, 26
        frame_w = sum(o[3] for o in opts) + gap * (len(opts) - 1)
        row_h = panel_h + lab
        frame_h = head + len(cams) * (row_h + gap)
        writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*codec),
                                 fps, (frame_w, frame_h))
        if not writer.isOpened():
            raise RuntimeError(f"cv2 could not open a {codec!r} writer for {out_mp4}")

        for i in range(n):
            canvas = np.full((frame_h, frame_w, 3), 245, np.uint8)
            cv2.putText(canvas, f"{trial_name}  frame {i+1}/{n}  "
                        f"cropped source {cw0}x{h}  (panels upscaled to a common "
                        "height, own aspect kept)",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (78, 81, 82), 1, cv2.LINE_AA)
            y = head
            for (name, x0, cw), ds in zip(geom, streams):
                patch = cv2.cvtColor(np.asarray(ds[i, :, x0:x0 + cw, :]),
                                     cv2.COLOR_RGB2BGR)
                x = 0
                for label, ow, oh, disp_w in opts:
                    small = cv2.resize(patch, (ow, oh), interpolation=cv2.INTER_AREA)
                    shown = cv2.resize(small, (disp_w, panel_h),
                                       interpolation=cv2.INTER_NEAREST)
                    canvas[y + lab:y + lab + panel_h, x:x + disp_w] = shown
                    cv2.putText(canvas, f"{name}  {label}", (x + 2, y + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (11, 11, 11), 1,
                                cv2.LINE_AA)
                    x += disp_w + gap
                y += row_h + gap
            writer.write(canvas)
        writer.release()
        return n, opts, (frame_w, frame_h)


def _copy_attrs(src_obj, dst_obj) -> None:
    for key, value in src_obj.attrs.items():
        dst_obj.attrs[key] = value


def crop_file(src: Path, dst: Path, table: dict, resize: tuple | None = None,
              chunk: int = 32) -> dict:
    """Crop (and optionally resize) every camera stream into a new file.

    ``resize`` is (width, height) applied after the crop with INTER_LINEAR, the
    same filter camera_buffer uses during collection.
    """
    stats = {"cropped": 0, "copied": 0, "frames": 0}
    with h5py.File(src, "r", locking=False) as fin, h5py.File(dst, "w") as fout:
        _copy_attrs(fin, fout)

        def visit(name, obj):
            if isinstance(obj, h5py.Group):
                g = fout.require_group(name)
                _copy_attrs(obj, g)
                return
            is_camera = name.startswith("trials/") and CAMERA_PREFIX in name \
                and name.endswith(CAMERA_RGB_SUFFIX)
            if not is_camera:
                fout.create_dataset(
                    name, data=obj[()], dtype=obj.dtype,
                    compression=obj.compression, compression_opts=obj.compression_opts,
                )
                _copy_attrs(obj, fout[name])
                stats["copied"] += 1
                return
            camera = name.split(CAMERA_PREFIX, 1)[1].split("/", 1)[0]
            left, right = crop_for(table, camera)
            n, h, w, c = obj.shape
            x0, cw = crop_bounds(w, left, right)
            ow, oh = (cw, h) if resize is None else resize
            out = fout.create_dataset(
                name, shape=(n, oh, ow, c), dtype=obj.dtype,
                chunks=(1, oh, ow, c), compression=obj.compression,
                compression_opts=obj.compression_opts,
            )
            # Chunked so a full session never lands in memory at once.
            for start in range(0, n, chunk):
                stop = min(start + chunk, n)
                block = obj[start:stop, :, x0:x0 + cw, :]
                if resize is not None:
                    block = np.stack([
                        cv2.resize(f, (ow, oh), interpolation=cv2.INTER_LINEAR)
                        for f in block
                    ])
                out[start:stop] = block
            _copy_attrs(obj, out)
            stats["cropped"] += 1
            stats["frames"] += n

        fin.visititems(visit)

        # stored_size and crop_xywh must describe the post-processed frame.
        if "calibration" in fout:
            for cam in fout["calibration"]:
                node = fout[f"calibration/{cam}"]
                if "stored_size" not in node:
                    continue
                left, right = crop_for(table, cam)
                stored = np.asarray(node["stored_size"][()]).ravel()
                x0, cw = crop_bounds(int(stored[0]), left, right)
                final = (cw, int(stored[1])) if resize is None else resize
                node["stored_size"][...] = np.array(final, stored.dtype)
                if "crop_xywh" in node:
                    prev = np.asarray(node["crop_xywh"][()]).ravel()
                    node["crop_xywh"][...] = np.array(
                        [prev[0] + x0, prev[1], cw, prev[3]], prev.dtype
                    )
        fout.attrs["postprocess_horizontal_crop"] = ";".join(
            f"{k}={v[0]},{v[1]}" for k, v in sorted(table.items())
        )
        # The validator checks camera datasets against the root image-shape attributes,
        # which describe every camera, so without a resize all crops must give one width.
        if "image_width" in fout.attrs and "image_height" in fout.attrs:
            src_w = int(fout.attrs["image_width"])
            src_h = int(fout.attrs["image_height"])
            if resize is not None:
                final_w, final_h = resize
            else:
                cameras = [n for n in table if n != "__default__"] or ["__default__"]
                widths = {
                    crop_bounds(src_w, *crop_for(table, name))[1] for name in cameras
                }
                if len(widths) > 1:
                    raise ValueError(
                        f"cameras crop to different widths {sorted(widths)}; one "
                        "pair of root image_width/height attributes cannot "
                        "describe both. Use crops with equal totals, or --resize."
                    )
                final_w, final_h = widths.pop(), src_h
            fout.attrs["image_width"] = final_w
            fout.attrs["image_height"] = final_h
        fout.attrs["postprocess_source_file"] = str(src.name)
        if resize is not None:
            fout.attrs["postprocess_resize"] = f"{resize[0]}x{resize[1]}"
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path)
    p.add_argument("--fraction", type=float, default=0.20,
                   help="symmetric fallback fraction per side (default 0.20)")
    p.add_argument("--crop", action="append", metavar="NAME=LEFT,RIGHT",
                   help="per-camera asymmetric crop, e.g. --crop wrist_1=0.30,0.20; "
                        "repeatable. Cameras not named use --fraction.")
    p.add_argument("--preview", metavar="TRIAL",
                   help="render this trial to mp4 and exit; writes no .h5")
    p.add_argument("--compare-resizes", metavar="TRIAL",
                   help="render candidate resizes side by side and exit")
    p.add_argument("--out", type=Path, help="output .h5 or .mp4 path")
    p.add_argument("--resize", metavar="WxH",
                   help="resize after cropping, e.g. --resize 256x256")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--codec", default="avc1")
    args = p.parse_args()

    src = args.source.expanduser().resolve()
    table = parse_crop_specs(args.crop, args.fraction)
    if args.compare_resizes:
        out = args.out or src.parent / f"{src.stem}.{args.compare_resizes}.resizes.mp4"
        n, opts, size = render_resize_compare(src, args.compare_resizes, table, out,
                                              args.fps, args.codec)
        print(f"{src.name} [{args.compare_resizes}]: {n} frames, canvas {size[0]}x{size[1]}")
        for label, ow, oh, _ in opts:
            print(f"   {label:<16} {ow}x{oh}  aspect {ow/oh:.4f}")
        print(f"wrote {out}")
        return 0

    if args.preview:
        out = args.out or src.parent / f"{src.stem}.{args.preview}.crop.mp4"
        n, geom = render_preview(src, args.preview, table, out, args.fps, args.codec)
        print(f"{src.name} [{args.preview}]: {n} frames")
        for name, x0, cw, left, right in geom:
            print(f"   {name}: L{left:.2f} R{right:.2f} -> x[{x0}:{x0+cw}] width {cw}")
        print(f"wrote {out}")
        return 0

    out = args.out or src.parent / f"{src.stem}.crop.h5"
    if out.exists():
        print(f"refusing to overwrite {out}")
        return 1
    resize = None
    if args.resize:
        rw, _, rh = args.resize.lower().partition("x")
        resize = (int(rw), int(rh))
    stats = crop_file(src, out, table, resize)
    print(f"{src.name}: cropped {stats['cropped']} camera datasets "
          f"({stats['frames']} frames), copied {stats['copied']} others")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
