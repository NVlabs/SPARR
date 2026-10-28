# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA OneWay Noncommercial License (March 22, 2022).
# See the LICENSE file in the repository root for the full license text:
# https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf

"""Read-only, offline replay and inspection of canonical Phase 1 sessions.

This module never imports ROS and never publishes robot commands.  It opens the
source HDF5 file read-only, validates it, and exposes recorded transitions in
their canonical observation/action mapping order.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterator, Sequence

import h5py
import numpy as np

from .validation import validate_file


def _attribute(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _array(group: h5py.Group, path: str, index: int) -> np.ndarray:
    dataset = group.get(path)
    if not isinstance(dataset, h5py.Dataset):
        raise KeyError(f"{group.name} is missing canonical dataset {path!r}")
    if index < 0 or index >= dataset.shape[0]:
        raise IndexError(f"Row {index} is outside {group.name}/{path}")
    return np.asarray(dataset[index]).copy()


def _scalar(group: h5py.Group, path: str, index: int) -> Any:
    value = _array(group, path, index)
    return value.item() if value.ndim == 0 else value


def _optional_array(group: h5py.Group, path: str, index: int) -> np.ndarray | None:
    dataset = group.get(path)
    if dataset is None:
        return None
    if not isinstance(dataset, h5py.Dataset):
        raise TypeError(f"{group.name}/{path} is not a dataset")
    return _array(group, path, index)


def _leading_length(group: h5py.Group, path: str) -> int:
    dataset = group.get(path)
    return int(dataset.shape[0]) if isinstance(dataset, h5py.Dataset) else 0


@dataclass(frozen=True)
class TrialDescriptor:
    """Small immutable description of one recorded attempt."""

    name: str
    trial_uuid: str
    attempt_index: int
    complete: bool
    accepted: bool
    export_eligible: bool
    operator_label: str
    auto_outcome: str
    termination_reason: str
    observation_count: int
    action_count: int

    def as_dict(self) -> dict[str, Any]:
        return dict(vars(self))


@dataclass(frozen=True)
class ReplayStep:
    """One recorded action and the observations it maps between."""

    trial_name: str
    action_index: int
    observation_index: int
    next_observation_index: int | None
    current_ros_stamp_ns: int
    next_ros_stamp_ns: int | None
    step_start_monotonic_ns: int
    current_flange_pose_world_xyzw: np.ndarray
    next_flange_pose_world_xyzw: np.ndarray | None
    socket_root_exact_world_xyzw: np.ndarray
    raw_mean_normalized: np.ndarray
    raw_std_normalized: np.ndarray | None
    raw_sample_normalized: np.ndarray
    effective_software_action_normalized: np.ndarray
    residual_effective_software_action_minus_raw_mean: np.ndarray
    submitted_controller_tcp_target_world_xyzw: np.ndarray
    command_published: bool
    command_valid: bool
    guard_bitmask: int
    wrist_1_rgb: np.ndarray | None = None
    wrist_2_rgb: np.ndarray | None = None

    def as_dict(self, *, include_image_statistics: bool = False) -> dict[str, Any]:
        """Return a JSON-serializable view without embedding raw image pixels."""

        result: dict[str, Any] = {
            "trial_name": self.trial_name,
            "action_index": self.action_index,
            "observation_index": self.observation_index,
            "next_observation_index": self.next_observation_index,
            "current_ros_stamp_ns": self.current_ros_stamp_ns,
            "next_ros_stamp_ns": self.next_ros_stamp_ns,
            "step_start_monotonic_ns": self.step_start_monotonic_ns,
            "current_flange_pose_world_xyzw": self.current_flange_pose_world_xyzw.tolist(),
            "next_flange_pose_world_xyzw": (
                None
                if self.next_flange_pose_world_xyzw is None
                else self.next_flange_pose_world_xyzw.tolist()
            ),
            "socket_root_exact_world_xyzw": self.socket_root_exact_world_xyzw.tolist(),
            "raw_mean_normalized": self.raw_mean_normalized.tolist(),
            "raw_std_normalized": (
                None
                if self.raw_std_normalized is None
                else self.raw_std_normalized.tolist()
            ),
            "raw_sample_normalized": self.raw_sample_normalized.tolist(),
            "effective_software_action_normalized": (
                self.effective_software_action_normalized.tolist()
            ),
            "residual_effective_software_action_minus_raw_mean": (
                self.residual_effective_software_action_minus_raw_mean.tolist()
            ),
            "submitted_controller_tcp_target_world_xyzw": (
                self.submitted_controller_tcp_target_world_xyzw.tolist()
            ),
            "command_published": self.command_published,
            "command_valid": self.command_valid,
            "guard_bitmask": self.guard_bitmask,
        }
        if include_image_statistics:
            result["images"] = {
                "wrist_1": _image_statistics(self.wrist_1_rgb),
                "wrist_2": _image_statistics(self.wrist_2_rgb),
            }
        return result


def _image_statistics(image: np.ndarray | None) -> dict[str, Any] | None:
    if image is None:
        return None
    contiguous = np.ascontiguousarray(image)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "mean_rgb": np.mean(contiguous, axis=(0, 1)).tolist(),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


class ReadOnlyReplay:
    """Validated read-only view over one canonical SPARR HDF5 session."""

    def __init__(self, path: str | Path, *, allow_partial: bool = False) -> None:
        self.path = Path(path)
        report = validate_file(self.path, allow_partial=allow_partial)
        if not report.is_valid:
            details = "; ".join(str(issue) for issue in report.errors[:10])
            raise ValueError(f"Cannot replay invalid HDF5 file: {details}")
        self.validation_report = report
        self._file = h5py.File(self.path, "r")

    @property
    def is_closed(self) -> bool:
        return not bool(self._file.id.valid)

    def close(self) -> None:
        if not self.is_closed:
            self._file.close()

    def __enter__(self) -> "ReadOnlyReplay":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def trials(self) -> tuple[TrialDescriptor, ...]:
        trials = self._file.get("trials")
        if not isinstance(trials, h5py.Group):
            return ()
        result: list[TrialDescriptor] = []
        for name in sorted(trials.keys()):
            trial = trials[name]
            if not isinstance(trial, h5py.Group):
                continue
            result.append(
                TrialDescriptor(
                    name=name,
                    trial_uuid=str(_attribute(trial.attrs.get("trial_uuid", ""))),
                    attempt_index=int(_attribute(trial.attrs.get("attempt_index", -1))),
                    complete=bool(_attribute(trial.attrs.get("complete", False))),
                    accepted=bool(_attribute(trial.attrs.get("accepted", False))),
                    export_eligible=bool(
                        _attribute(trial.attrs.get("export_eligible", True))
                    ),
                    operator_label=str(
                        _attribute(trial.attrs.get("operator_label", ""))
                    ),
                    auto_outcome=str(_attribute(trial.attrs.get("auto_outcome", ""))),
                    termination_reason=str(
                        _attribute(trial.attrs.get("termination_reason", ""))
                    ),
                    observation_count=_leading_length(
                        trial, "observations/timing/ros_stamp_ns"
                    ),
                    action_count=_leading_length(
                        trial, "actions/mapping/observation_index"
                    ),
                )
            )
        return tuple(result)

    def select_trials(
        self,
        selectors: Sequence[str] | None = None,
        *,
        include_incomplete: bool = False,
    ) -> tuple[TrialDescriptor, ...]:
        """Select groups by name or integer attempt index.

        With no selectors, every complete trial is returned.  Incomplete trials
        are excluded unless ``include_incomplete`` is explicit.
        """

        available = self.trials()
        if selectors:
            selected: list[TrialDescriptor] = []
            for selector in selectors:
                selector_text = str(selector)
                matches = [trial for trial in available if trial.name == selector_text]
                if not matches:
                    try:
                        attempt_index = int(selector_text)
                    except ValueError:
                        attempt_index = None
                    if attempt_index is not None:
                        matches = [
                            trial
                            for trial in available
                            if trial.attempt_index == attempt_index
                        ]
                if len(matches) != 1:
                    raise KeyError(
                        f"Trial selector {selector_text!r} matched {len(matches)} trials"
                    )
                if matches[0] not in selected:
                    selected.append(matches[0])
        else:
            selected = list(available)
        if include_incomplete:
            return tuple(selected)
        return tuple(trial for trial in selected if trial.complete)

    def iter_steps(
        self,
        trial_name: str,
        *,
        start: int = 0,
        limit: int | None = None,
        include_images: bool = False,
    ) -> Iterator[ReplayStep]:
        if start < 0:
            raise ValueError("start must be non-negative")
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        trials = self._file.get("trials")
        if not isinstance(trials, h5py.Group) or trial_name not in trials:
            raise KeyError(f"Unknown trial group {trial_name!r}")
        trial = trials[trial_name]
        if not isinstance(trial, h5py.Group):
            raise TypeError(f"{trial_name!r} is not a trial group")
        action_count = _leading_length(trial, "actions/mapping/observation_index")
        stop = action_count if limit is None else min(action_count, start + limit)
        for action_index in range(min(start, action_count), stop):
            observation_index = int(
                _scalar(trial, "actions/mapping/observation_index", action_index)
            )
            raw_next_index = int(
                _scalar(trial, "actions/mapping/next_observation_index", action_index)
            )
            next_index = raw_next_index if raw_next_index >= 0 else None
            yield ReplayStep(
                trial_name=trial_name,
                action_index=action_index,
                observation_index=observation_index,
                next_observation_index=next_index,
                current_ros_stamp_ns=int(
                    _scalar(trial, "observations/timing/ros_stamp_ns", observation_index)
                ),
                next_ros_stamp_ns=(
                    None
                    if next_index is None
                    else int(
                        _scalar(trial, "observations/timing/ros_stamp_ns", next_index)
                    )
                ),
                step_start_monotonic_ns=int(
                    _scalar(trial, "actions/timing/step_start_monotonic_ns", action_index)
                ),
                current_flange_pose_world_xyzw=_array(
                    trial,
                    "observations/robot/flange_pose_world_xyzw",
                    observation_index,
                ),
                next_flange_pose_world_xyzw=(
                    None
                    if next_index is None
                    else _array(
                        trial,
                        "observations/robot/flange_pose_world_xyzw",
                        next_index,
                    )
                ),
                socket_root_exact_world_xyzw=_array(
                    trial,
                    "observations/task/socket_root_exact_world_xyzw",
                    observation_index,
                ),
                raw_mean_normalized=_array(
                    trial, "actions/policy/mu_raw_normalized", action_index
                ),
                raw_std_normalized=_optional_array(
                    trial, "actions/policy/std_raw_normalized", action_index
                ),
                raw_sample_normalized=_array(
                    trial, "actions/policy/sample_raw_normalized", action_index
                ),
                effective_software_action_normalized=_array(
                    trial,
                    "actions/action/effective_software_action_normalized",
                    action_index,
                ),
                residual_effective_software_action_minus_raw_mean=_array(
                    trial,
                    "labels/residual_effective_software_action_minus_raw_mean",
                    action_index,
                ),
                submitted_controller_tcp_target_world_xyzw=_array(
                    trial,
                    "actions/action/submitted_controller_tcp_target_world_xyzw",
                    action_index,
                ),
                command_published=bool(
                    _scalar(trial, "actions/action/command_published", action_index)
                ),
                command_valid=bool(
                    _scalar(trial, "labels/command_valid", action_index)
                ),
                guard_bitmask=int(
                    _scalar(trial, "actions/action/guard_bitmask", action_index)
                ),
                wrist_1_rgb=(
                    _array(
                        trial,
                        "observations/camera/wrist_1/rgb",
                        observation_index,
                    )
                    if include_images
                    else None
                ),
                wrist_2_rgb=(
                    _optional_array(
                        trial,
                        "observations/camera/wrist_2/rgb",
                        observation_index,
                    )
                    if include_images
                    else None
                ),
            )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _emit_json(value: Any) -> None:
    print(json.dumps(_json_safe(value), sort_keys=True, allow_nan=False))


def _sleep_for_recorded_timing(previous_ns: int | None, current_ns: int, speed: float) -> None:
    if speed <= 0.0 or previous_ns is None:
        return
    delay_s = max(0.0, (current_ns - previous_ns) * 1.0e-9 / speed)
    if delay_s:
        time.sleep(delay_s)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Canonical .h5 or crash .partial.h5 file")
    parser.add_argument(
        "--trial",
        action="append",
        help="Trial group name or attempt index; repeat to select multiple trials.",
    )
    parser.add_argument("--list-trials", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include incomplete groups when inspecting a partial file.",
    )
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help="Recorded-timing playback multiplier; 0 prints immediately (default).",
    )
    parser.add_argument(
        "--image-statistics",
        action="store_true",
        help="Read each pair of RGB crops and print shape, mean RGB, and SHA-256.",
    )
    args = parser.parse_args(argv)
    if args.speed < 0.0 or not np.isfinite(args.speed):
        parser.error("--speed must be finite and non-negative")
    if args.start_step < 0:
        parser.error("--start-step must be non-negative")
    if args.max_steps is not None and args.max_steps < 0:
        parser.error("--max-steps must be non-negative")
    if args.include_incomplete and not args.allow_partial:
        parser.error("--include-incomplete requires --allow-partial")

    try:
        with ReadOnlyReplay(args.path, allow_partial=args.allow_partial) as replay:
            selected = replay.select_trials(
                args.trial,
                include_incomplete=args.include_incomplete,
            )
            if args.list_trials:
                for trial in selected:
                    _emit_json({"record_type": "trial", **trial.as_dict()})
                return 0
            if not selected:
                raise ValueError("No matching complete trials to replay")
            for descriptor in selected:
                _emit_json({"record_type": "trial", **descriptor.as_dict()})
                previous_ns: int | None = None
                for step in replay.iter_steps(
                    descriptor.name,
                    start=args.start_step,
                    limit=args.max_steps,
                    include_images=args.image_statistics,
                ):
                    _sleep_for_recorded_timing(
                        previous_ns, step.step_start_monotonic_ns, args.speed
                    )
                    previous_ns = step.step_start_monotonic_ns
                    _emit_json(
                        {
                            "record_type": "step",
                            **step.as_dict(
                                include_image_statistics=args.image_statistics
                            ),
                        }
                    )
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"Replay failed: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ReadOnlyReplay", "ReplayStep", "TrialDescriptor", "main"]
