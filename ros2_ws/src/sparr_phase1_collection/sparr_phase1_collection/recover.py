# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Non-destructive recovery CLI for interrupted Phase 1 HDF5 sessions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .validation import recover_partial_file


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Source .partial.h5 file")
    parser.add_argument("--destination", type=Path)
    parser.add_argument(
        "--keep-incomplete",
        action="store_true",
        help="Retain structurally incomplete trials for forensic inspection.",
    )
    args = parser.parse_args(argv)
    result = recover_partial_file(
        args.path,
        destination=args.destination,
        complete_trials_only=not args.keep_incomplete,
    )
    print(f"Recovered copy: {result.destination_path}")
    print(f"Kept trials: {len(result.kept_trials)}")
    print(f"Dropped trials: {len(result.dropped_trials)}")
    print(result.report.summary())
    for issue in result.report.issues:
        print(issue)
    return 0 if result.report.is_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
