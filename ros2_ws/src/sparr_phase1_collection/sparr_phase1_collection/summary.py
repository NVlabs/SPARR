# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA License.
# See the LICENSE file in the repository root for the full license text.

"""Produce a compact quality summary for a Phase 1 HDF5 session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import h5py
import numpy as np

from .hdf5_schema import SUPPORTED_CAMERA_NAMES, decode_enabled_cameras
from .validation import validate_file


def _python(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _dataset(trial: h5py.Group, path: str) -> np.ndarray | None:
    item = trial.get(path)
    return None if not isinstance(item, h5py.Dataset) else np.asarray(item)


def _concatenate(trials: Iterable[h5py.Group], path: str) -> np.ndarray:
    values = [value for trial in trials if (value := _dataset(trial, path)) is not None]
    if not values:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate(values, axis=0)


def _boolean_rate(values: np.ndarray) -> Optional[float]:
    return None if values.size == 0 else float(np.asarray(values, dtype=bool).mean())


def _percentiles(values: np.ndarray, *, scale: float = 1.0) -> dict[str, Optional[float]]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)] * scale
    if finite.size == 0:
        return {"p50": None, "p95": None, "max": None}
    return {
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def _enabled_cameras(file: h5py.File, trials: Iterable[h5py.Group]) -> tuple[str, ...]:
    if "enabled_cameras" in file.attrs:
        try:
            return decode_enabled_cameras(file.attrs["enabled_cameras"])
        except (TypeError, ValueError, UnicodeError):
            pass
    # Without camera identity the schema-v1 two-camera contract applies; a missing
    # wrist_2 is a validation error, not a single-camera session.
    return SUPPORTED_CAMERA_NAMES


def build_summary(path: str | Path, *, allow_partial: bool = False) -> dict[str, Any]:
    file_path = Path(path)
    validation = validate_file(file_path, allow_partial=allow_partial)
    result: dict[str, Any] = {
        "path": str(file_path.resolve()),
        "valid": validation.is_valid,
        "validation_errors": len(validation.errors),
        "validation_warnings": len(validation.warnings),
        "trial_count": validation.trial_count,
        "complete_trial_count": validation.complete_trial_count,
        "accepted_trial_count": validation.accepted_trial_count,
        "observation_count": validation.observation_count,
        "action_count": validation.action_count,
    }
    if not file_path.exists():
        return result

    try:
        file = h5py.File(file_path, "r")
    except OSError:
        return result
    with file:
        result["schema_name"] = _python(file.attrs.get("schema_name", ""))
        result["schema_version"] = _python(file.attrs.get("schema_version", ""))
        result["session_uuid"] = _python(file.attrs.get("session_uuid", ""))
        result["closed_cleanly"] = bool(_python(file.attrs.get("closed_cleanly", False)))
        trials_group = file.get("trials")
        if not isinstance(trials_group, h5py.Group):
            return result
        trials = [trials_group[name] for name in sorted(trials_group)]
        enabled_cameras = _enabled_cameras(file, trials)
        result["enabled_cameras"] = list(enabled_cameras)
        result["camera_count"] = len(enabled_cameras)
        result["outcomes"] = {}
        for trial in trials:
            outcome = str(_python(trial.attrs.get("operator_label", "unknown")))
            result["outcomes"][outcome] = result["outcomes"].get(outcome, 0) + 1

        for key, path_name in {
            "robot_state_valid_rate": "observations/validity/robot_state",
            "policy_input_valid_rate": "observations/validity/policy_input",
            "label_valid_rate": "labels/command_valid",
            "command_published_rate": "actions/action/command_published",
        }.items():
            result[key] = _boolean_rate(_concatenate(trials, path_name))

        for camera_name in enabled_cameras:
            result[f"{camera_name}_valid_rate"] = _boolean_rate(
                _concatenate(trials, f"observations/validity/{camera_name}")
            )
            result[f"{camera_name}_age_ms"] = _percentiles(
                _concatenate(trials, f"observations/timing/{camera_name}_age_ns"),
                scale=1.0e-6,
            )
        if len(enabled_cameras) == 2:
            result["camera_skew_ms"] = _percentiles(
                _concatenate(trials, "observations/timing/camera_skew_ns"),
                scale=1.0e-6,
            )
        result["inference_ms"] = _percentiles(
            _concatenate(trials, "actions/timing/inference_ns"), scale=1.0e-6
        )
    return result


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    summary = build_summary(args.path, allow_partial=args.allow_partial)
    if args.as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        for key, value in summary.items():
            print(f"{key}: {value}")
    return 0 if summary["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
