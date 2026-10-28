#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA OneWay Noncommercial License (March 22, 2022).
# See the LICENSE file in the repository root for the full license text:
# https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf

"""Combine a resumed Phase 1 session with its base session into one .h5, for analysis.

Copies the base session, then appends the continuation's trials with renumbered
group names; recorded trial_index values are kept. Source files are not modified.

Usage:
    sparr_combine_sessions.py BASE.h5 CONTINUATION.h5 OUT.h5 [--keep-stubs] [--force]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np


def _text(v):
    return v.decode() if isinstance(v, bytes) else v


#: Config fields that must agree for the combined randomization stream to be continuous.
_MUST_MATCH = (
    "randomization/numpy_seed",
    "randomization/socket_noise_stream",
    "randomization/start/xy_half_range_m",
    "randomization/start/z_half_range_m",
    "randomization/socket_policy_observation/xy_half_range_m",
    "randomization/socket_policy_observation/z_half_range_m",
    "policy/stochastic",
    "policy/torch_seed",
    "socket/opening_pose_world_xyzw",
    # Sessions recorded before the opening-pose key stored the root instead.
    "socket/root_pose_world_xyzw",
)


def _cfg(f: h5py.File, key: str):
    c = f["metadata/collection_config"]
    if key not in c:
        return None
    v = c[key][()]
    if key == "randomization/socket_noise_stream":
        # "per_trial" is the same stream as the unset default.
        return None if _text(v) == "per_trial" else _text(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return _text(v)


def _check_compatible(a: h5py.File, b: h5py.File) -> list[str]:
    problems = []
    for key in _MUST_MATCH:
        va, vb = _cfg(a, key), _cfg(b, key)
        if va != vb:
            problems.append(f"{key}: base={va!r} continuation={vb!r}")
    return problems


def _trial_index(group: h5py.Group) -> int | None:
    if "observations" not in group:
        return None
    ds = group["observations/extra/trial_index"]
    if ds.shape[0] == 0:
        return None
    return int(ds[0])


def _is_stub(group: h5py.Group) -> bool:
    """True for a trial group with no rollout (the trailing abort/discard record).

    Matches recover.py's complete_trials_only default.
    """
    if "actions" in group:
        return False
    if "observations" not in group:
        return True
    return int(group.attrs.get("observation_count", 0)) <= 1


def combine(
    base: Path,
    continuation: Path,
    out: Path,
    *,
    keep_stubs: bool = False,
    force: bool = False,
) -> None:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {out}")
    if out.resolve() in (base.resolve(), continuation.resolve()):
        raise ValueError("output must differ from both source files")

    with h5py.File(base, "r") as fa, h5py.File(continuation, "r") as fb:
        problems = _check_compatible(fa, fb)
        if problems and not force:
            raise ValueError(
                "base and continuation sessions do not look like the same "
                "seed/config -- refusing to combine them into one "
                "randomization stream:\n  " + "\n  ".join(problems) +
                "\nPass --force to combine anyway."
            )
        if problems:
            print("--force: combining despite mismatches:")
            for p in problems:
                print(f"  {p}")

        base_trials = sorted(fa["trials"].keys())
        base_indices = {n: _trial_index(fa["trials"][n]) for n in base_trials}
        known = {i for i in base_indices.values() if i is not None}
        max_base_index = max(known) if known else -1

        cont_trials = sorted(fb["trials"].keys())
        to_copy = []
        skipped_overlap = []
        skipped_stub = []
        for n in cont_trials:
            group = fb["trials"][n]
            idx = _trial_index(group)
            if idx is not None and idx in known:
                skipped_overlap.append((n, idx))
                continue
            if not keep_stubs and _is_stub(group):
                skipped_stub.append((n, idx))
                continue
            to_copy.append((n, idx))

        if skipped_overlap:
            print(f"skipping {len(skipped_overlap)} trial(s) already present "
                  f"in the base session (by trial_index): "
                  f"{[f'{n}(idx {i})' for n, i in skipped_overlap]}")
        if skipped_stub:
            print(f"skipping {len(skipped_stub)} trial(s) with no rollout "
                  f"(the trailing abort/discard artifact): "
                  f"{[f'{n}(idx {i})' for n, i in skipped_stub]}")
        if not to_copy:
            raise ValueError("nothing new to add -- every continuation trial "
                              "is either already present or a stub")

        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"copying base session ({len(base_trials)} trials) -> {out}")
        shutil.copy2(base, out)

    with h5py.File(out, "r+") as fout, h5py.File(continuation, "r") as fb:
        trials = fout["trials"]
        next_index = len(trials)
        added = []
        for name, idx in to_copy:
            new_name = f"trial_{next_index:06d}"
            fb.copy(f"trials/{name}", trials, name=new_name)
            added.append((new_name, idx))
            next_index += 1
        trials.attrs.modify("count", next_index)
        fout.attrs["closed_cleanly"] = True
        fout.attrs["session_complete"] = True
        fout.attrs["combined_from_base"] = str(base.resolve())
        fout.attrs["combined_from_continuation"] = str(continuation.resolve())
        fout.attrs["combined_utc"] = np.bytes_(
            __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat()
        )
        fout.flush()

    print(f"added {len(added)} trial(s):")
    for name, idx in added:
        print(f"  {name}  (trial_index={idx})")
    print(f"combined file: {out}  ({next_index} trials total)")
    print("both source files are unchanged.")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("base", type=Path, help="the earlier session (trials 0..N)")
    p.add_argument("continuation", type=Path,
                   help="the resumed session (first_attempt_index N+1..)")
    p.add_argument("out", type=Path, help="path for the new combined .h5")
    p.add_argument("--keep-stubs", action="store_true",
                   help="also copy trailing trials with no rollout (aborts/"
                        "discards). Off by default, matching sparr_phase1_recover.")
    p.add_argument("--force", action="store_true",
                   help="combine even if base/continuation config disagrees")
    args = p.parse_args()

    try:
        combine(args.base, args.continuation, args.out,
                keep_stubs=args.keep_stubs, force=args.force)
    except (FileExistsError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
