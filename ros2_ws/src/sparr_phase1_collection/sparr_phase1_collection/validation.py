# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA License.
# See the LICENSE file in the repository root for the full license text.

"""Structural and semantic validation for SPARR Phase 1 HDF5 files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
from typing import Any, Iterable

import h5py
import numpy as np

from .hdf5_schema import (
    ALL_DATASET_SPECS,
    LEGACY_TWO_CAMERA_REQUIRED_OBSERVATION_PATHS,
    REQUIRED_ACTION_PATHS,
    ROOT_REQUIRED_ATTRIBUTES,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    SUPPORTED_CAMERA_NAMES,
    decode_enabled_cameras,
    is_extension_path,
    required_observation_paths,
)


_REQUIRED_ROOT_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "session_uuid",
        "created_utc",
        "closed_cleanly",
        "pose_convention",
        "action_convention",
        "image_storage",
        "image_height",
        "image_width",
        "image_channels",
        "training_action_clip",
    }
)
_FORBIDDEN_PATH_TOKENS = frozenset(
    {"full_res", "full_resolution", "invented_gripper"}
)
_CAMERA_DATASET_PATHS = {
    camera_name: frozenset(
        {
            f"observations/timing/{camera_name}_stamp_ns",
            f"observations/timing/{camera_name}_age_ns",
            f"observations/validity/{camera_name}",
            f"observations/camera/{camera_name}/rgb",
        }
    )
    for camera_name in SUPPORTED_CAMERA_NAMES
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    location: str
    message: str

    def __str__(self) -> str:
        return f"{self.severity.upper()} {self.code} at {self.location}: {self.message}"


@dataclass
class ValidationReport:
    path: Path
    allow_partial: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    trial_count: int = 0
    complete_trial_count: int = 0
    accepted_trial_count: int = 0
    observation_count: int = 0
    action_count: int = 0

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def add_error(self, code: str, location: str, message: str) -> None:
        self.issues.append(ValidationIssue("error", code, location, message))

    def add_warning(self, code: str, location: str, message: str) -> None:
        self.issues.append(ValidationIssue("warning", code, location, message))

    def summary(self) -> str:
        status = "valid" if self.is_valid else "invalid"
        return (
            f"{self.path}: {status}; {self.trial_count} trials, "
            f"{self.observation_count} observations, {self.action_count} actions, "
            f"{len(self.errors)} errors, {len(self.warnings)} warnings"
        )


@dataclass(frozen=True)
class RecoveryResult:
    source_path: Path
    destination_path: Path
    kept_trials: tuple[str, ...]
    dropped_trials: tuple[str, ...]
    report: ValidationReport


def _dataset_map(group: h5py.Group) -> dict[str, h5py.Dataset]:
    result: dict[str, h5py.Dataset] = {}

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            result[name] = obj

    group.visititems(visitor)
    return result


def _leading_count(
    datasets: dict[str, h5py.Dataset],
    prefixes: tuple[str, ...],
) -> int:
    lengths = [
        int(ds.shape[0])
        for path, ds in datasets.items()
        if path.startswith(prefixes) and ds.ndim >= 1
    ]
    return max(lengths, default=0)


def _is_bool_attr(value: Any) -> bool:
    return isinstance(_attr(value), (bool, np.bool_))


def _training_action_clip(file: h5py.File) -> float:
    try:
        return float(file.attrs["training_action_clip"])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _declared_enabled_cameras(file: h5py.File) -> tuple[str, ...] | None:
    """Return a valid explicit camera declaration, or None if the file has none."""

    if "enabled_cameras" not in file.attrs or "camera_count" not in file.attrs:
        return None
    try:
        return decode_enabled_cameras(file.attrs["enabled_cameras"])
    except (TypeError, ValueError, UnicodeError):
        return None


def _is_complete_trial(group: h5py.Group) -> bool:
    if not bool(_attr(group.attrs.get("complete", False))):
        return False
    datasets = _dataset_map(group)
    n_obs = _leading_count(datasets, ("observations/",))
    n_actions = _leading_count(datasets, ("actions/", "labels/"))
    if n_obs != n_actions + 1:
        return False
    obs_map = datasets.get("actions/mapping/observation_index")
    next_map = datasets.get("actions/mapping/next_observation_index")
    if obs_map is None or next_map is None:
        return n_actions == 0
    expected = np.arange(n_actions, dtype=np.int64)
    return bool(
        np.array_equal(np.asarray(obs_map), expected)
        and np.array_equal(np.asarray(next_map), expected + 1)
    )


def validate_file(
    path: str | os.PathLike[str],
    *,
    allow_partial: bool = False,
) -> ValidationReport:
    """Validate one canonical HDF5 file without modifying it."""

    file_path = Path(path)
    report = ValidationReport(path=file_path, allow_partial=allow_partial)
    if not file_path.exists():
        report.add_error("file_missing", "/", "File does not exist")
        return report
    try:
        file = h5py.File(file_path, "r")
    except (OSError, ValueError) as exc:
        report.add_error("hdf5_open_failed", "/", str(exc))
        return report

    with file:
        _validate_root(file, report)
        trials_obj = file.get("trials")
        if not isinstance(trials_obj, h5py.Group):
            report.add_error("trials_group_missing", "/trials", "Required trials group is missing")
            return report
        trial_names = sorted(trials_obj.keys())
        report.trial_count = len(trial_names)
        declared_count = trials_obj.attrs.get("count")
        if declared_count is not None:
            try:
                declared_count_value = int(declared_count)
            except (TypeError, ValueError):
                report.add_error("trial_count_type", "/trials", "count must be an integer")
            else:
                if declared_count_value != len(trial_names):
                    report.add_error(
                        "trial_count_mismatch",
                        "/trials",
                        f"count attribute is {declared_count_value}, found "
                        f"{len(trial_names)} groups",
                    )
        for trial_name in trial_names:
            trial = trials_obj[trial_name]
            location = f"/trials/{trial_name}"
            if not isinstance(trial, h5py.Group):
                report.add_error("trial_not_group", location, "Trial entry is not a group")
                continue
            _validate_trial(trial, location, report)
    return report


def _validate_root(file: h5py.File, report: ValidationReport) -> None:
    for name in sorted(_REQUIRED_ROOT_FIELDS):
        if name not in file.attrs:
            report.add_error("root_attribute_missing", "/", f"Missing root attribute {name!r}")
    for name, expected in ROOT_REQUIRED_ATTRIBUTES.items():
        if name in file.attrs and str(_attr(file.attrs[name])) != str(expected):
            report.add_error(
                "root_attribute_value",
                "/",
                f"{name!r} is {_attr(file.attrs[name])!r}, expected {expected!r}",
            )
    if "schema_name" in file.attrs and str(_attr(file.attrs["schema_name"])) != SCHEMA_NAME:
        report.add_error("schema_name", "/", f"Expected schema {SCHEMA_NAME!r}")
    if (
        "schema_version" in file.attrs
        and str(_attr(file.attrs["schema_version"])) != SCHEMA_VERSION
    ):
        report.add_error("schema_version", "/", f"Expected version {SCHEMA_VERSION!r}")
    if "closed_cleanly" in file.attrs:
        if not _is_bool_attr(file.attrs["closed_cleanly"]):
            report.add_error("closed_cleanly_type", "/", "closed_cleanly must be boolean")
        elif not bool(_attr(file.attrs["closed_cleanly"])):
            if report.allow_partial:
                report.add_warning("partial_file", "/", "File was not closed cleanly")
            else:
                report.add_error(
                    "partial_file",
                    "/",
                    "File was not closed cleanly; pass allow_partial=True for crash inspection",
                )
    if report.path.name.endswith(".partial.h5"):
        if report.allow_partial:
            report.add_warning("partial_suffix", "/", "File retains the in-progress suffix")
        else:
            report.add_error("partial_suffix", "/", "Partial file requires allow_partial=True")
    for group_name in ("metadata", "calibration"):
        if not isinstance(file.get(group_name), h5py.Group):
            report.add_error(
                "named_group_missing", f"/{group_name}", f"Required {group_name} group is missing"
            )
    dimensions: list[int] = []
    for name in ("image_height", "image_width", "image_channels"):
        if name in file.attrs:
            try:
                dimensions.append(int(file.attrs[name]))
            except (TypeError, ValueError):
                report.add_error("image_dimension_type", "/", f"{name} must be an integer")
    if len(dimensions) == 3:
        if dimensions[0] <= 0 or dimensions[1] <= 0 or dimensions[2] != 3:
            report.add_error(
                "image_dimensions",
                "/",
                f"Expected positive cropped RGB dimensions with three channels, got {dimensions}",
            )
    if "training_action_clip" in file.attrs:
        try:
            training_action_clip = float(file.attrs["training_action_clip"])
        except (TypeError, ValueError):
            report.add_error(
                "training_action_clip_type",
                "/",
                "training_action_clip must be a positive finite float",
            )
        else:
            if not np.isfinite(training_action_clip) or training_action_clip <= 0.0:
                report.add_error(
                    "training_action_clip_value",
                    "/",
                    "training_action_clip must be a positive finite float",
                )

    has_enabled_cameras = "enabled_cameras" in file.attrs
    has_camera_count = "camera_count" in file.attrs
    if has_enabled_cameras != has_camera_count:
        report.add_error(
            "camera_identity_incomplete",
            "/",
            "enabled_cameras and camera_count must either both be present or both be absent",
        )
    elif has_enabled_cameras:
        try:
            enabled_cameras = decode_enabled_cameras(file.attrs["enabled_cameras"])
        except (TypeError, ValueError, UnicodeError) as exc:
            report.add_error(
                "enabled_cameras_value",
                "/",
                str(exc),
            )
        else:
            try:
                camera_count = int(file.attrs["camera_count"])
            except (TypeError, ValueError):
                report.add_error(
                    "camera_count_type",
                    "/",
                    "camera_count must be an integer",
                )
            else:
                if camera_count != len(enabled_cameras):
                    report.add_error(
                        "camera_count_mismatch",
                        "/",
                        f"camera_count is {camera_count}, enabled_cameras has "
                        f"{len(enabled_cameras)} entries",
                    )


def _validate_trial(
    trial: h5py.Group,
    location: str,
    report: ValidationReport,
) -> None:
    required_attrs = {
        "trial_uuid",
        "attempt_index",
        "complete",
        "accepted",
        "operator_label",
        "auto_outcome",
        "termination_reason",
        "start_utc",
    }
    for name in sorted(required_attrs):
        if name not in trial.attrs:
            report.add_error(
                "trial_attribute_missing",
                location,
                f"Missing trial attribute {name!r}",
            )
    for name in ("complete", "accepted"):
        if name in trial.attrs and not _is_bool_attr(trial.attrs[name]):
            report.add_error(
                "trial_attribute_type",
                location,
                f"{name} must be boolean",
            )
    if "export_eligible" in trial.attrs and not _is_bool_attr(
        trial.attrs["export_eligible"]
    ):
        report.add_error(
            "trial_attribute_type",
            location,
            "export_eligible must be boolean when present",
        )

    datasets = _dataset_map(trial)
    declared_cameras = _declared_enabled_cameras(trial.file)
    _validate_camera_contract(
        datasets,
        location,
        declared_cameras=declared_cameras,
        report=report,
    )
    n_obs = _leading_count(datasets, ("observations/",))
    n_actions = _leading_count(datasets, ("actions/", "labels/"))
    report.observation_count += n_obs
    report.action_count += n_actions
    complete = bool(_attr(trial.attrs.get("complete", False)))
    accepted = bool(_attr(trial.attrs.get("accepted", False)))
    if complete:
        report.complete_trial_count += 1
    if accepted:
        report.accepted_trial_count += 1

    for path, dataset in datasets.items():
        dataset_location = f"{location}/{path}"
        if not path.startswith(("observations/", "actions/", "labels/")):
            report.add_error(
                "dataset_namespace",
                dataset_location,
                "Unknown trial dataset namespace",
            )
            continue
        if any(part.lower() in _FORBIDDEN_PATH_TOKENS for part in path.split("/")):
            report.add_error("forbidden_dataset", dataset_location, "Forbidden dataset name")
        record_kind = "observation" if path.startswith("observations/") else "action"
        spec = ALL_DATASET_SPECS.get(path)
        if spec is None and not is_extension_path(path, record_kind=record_kind):
            report.add_error(
                "unknown_dataset",
                dataset_location,
                "Custom datasets must be placed below an approved extra namespace",
            )
        expected_leading = n_obs if record_kind == "observation" else n_actions
        if dataset.ndim < 1:
            report.add_error(
                "dataset_rank",
                dataset_location,
                "Trial datasets must be row-oriented",
            )
            continue
        if int(dataset.shape[0]) != expected_leading:
            report.add_error(
                "dataset_length",
                dataset_location,
                f"Leading dimension is {dataset.shape[0]}, expected {expected_leading}",
            )
        if spec is not None:
            if np.dtype(dataset.dtype) != spec.numpy_dtype():
                report.add_error(
                    "dataset_dtype",
                    dataset_location,
                    f"dtype is {dataset.dtype}, expected {spec.numpy_dtype()}",
                )
            if spec.image:
                expected_tail = _root_image_shape(trial.file)
            else:
                expected_tail = spec.tail_shape
            if expected_tail is not None and tuple(dataset.shape[1:]) != tuple(expected_tail):
                report.add_error(
                    "dataset_shape",
                    dataset_location,
                    f"tail shape is {dataset.shape[1:]}, expected {expected_tail}",
                )

    if n_obs > 0 or complete:
        required_observations = (
            required_observation_paths(declared_cameras)
            if declared_cameras is not None
            else LEGACY_TWO_CAMERA_REQUIRED_OBSERVATION_PATHS
        )
    else:
        required_observations = frozenset()
    required_actions = REQUIRED_ACTION_PATHS if n_actions > 0 else frozenset()
    for path in sorted(required_observations - set(datasets)):
        report.add_error("required_dataset_missing", location, f"Missing required dataset {path!r}")
    for path in sorted(required_actions - set(datasets)):
        report.add_error("required_dataset_missing", location, f"Missing required dataset {path!r}")

    for attr_name, expected_count, code in (
        ("observation_count", n_obs, "observation_count_mismatch"),
        ("action_count", n_actions, "action_count_mismatch"),
    ):
        declared = trial.attrs.get(attr_name)
        if declared is None:
            continue
        try:
            declared_value = int(declared)
        except (TypeError, ValueError):
            report.add_error(
                "trial_count_type",
                location,
                f"{attr_name} must be an integer",
            )
        else:
            if declared_value != expected_count:
                report.add_error(
                    code,
                    location,
                    f"Attribute says {declared_value}, datasets contain {expected_count}",
                )

    if complete:
        if n_obs != n_actions + 1:
            report.add_error(
                "row_invariant",
                location,
                f"Complete trial requires N == M + 1, got N={n_obs}, M={n_actions}",
            )
    else:
        message = f"Trial is incomplete with N={n_obs}, M={n_actions}"
        if report.allow_partial and n_obs in {n_actions, n_actions + 1}:
            report.add_warning("incomplete_trial", location, message)
        else:
            report.add_error("incomplete_trial", location, message)
    if accepted and not complete:
        report.add_error("accepted_incomplete", location, "An accepted trial must be complete")
    operator_label = str(_attr(trial.attrs.get("operator_label", ""))).strip().lower()
    if accepted and operator_label in {"", "unlabeled", "unknown", "pending"}:
        report.add_error(
            "accepted_without_confirmation",
            location,
            "Accepted success requires a non-placeholder operator label",
        )
    if "accepted_success_index" in trial.attrs:
        try:
            success_index = int(trial.attrs["accepted_success_index"])
        except (TypeError, ValueError):
            report.add_error(
                "accepted_success_index_type",
                location,
                "accepted_success_index must be an integer",
            )
        else:
            if not accepted:
                report.add_error(
                    "success_index_without_acceptance",
                    location,
                    "accepted_success_index is only valid for accepted trials",
                )
            if success_index < 0 or success_index >= n_actions:
                report.add_error(
                    "accepted_success_index_range",
                    location,
                    f"accepted_success_index {success_index} is outside [0, {n_actions})",
                )
    if "reset_randomization_xyz_yaw" in trial.attrs:
        reset_randomization = np.asarray(trial.attrs["reset_randomization_xyz_yaw"])
        if reset_randomization.shape != (4,) or not np.isfinite(reset_randomization).all():
            report.add_error(
                "reset_randomization_shape",
                location,
                "reset_randomization_xyz_yaw must be a finite four-vector",
            )
    if complete and "end_utc" not in trial.attrs:
        report.add_error("end_time_missing", location, "Complete trial is missing end_utc")

    _validate_mapping(datasets, location, n_obs, n_actions, complete, report)
    _validate_action_semantics(
        datasets,
        location,
        n_actions,
        _training_action_clip(trial.file),
        report,
    )
    _validate_task_semantics(datasets, location, n_obs, report)
    _validate_time_order(datasets, location, report)


def _validate_camera_contract(
    datasets: dict[str, h5py.Dataset],
    location: str,
    *,
    declared_cameras: tuple[str, ...] | None,
    report: ValidationReport,
) -> None:
    """Validate explicit camera identity and prevent placeholder camera rows."""

    enabled = (
        declared_cameras
        if declared_cameras is not None
        else SUPPORTED_CAMERA_NAMES
    )
    disabled = set(SUPPORTED_CAMERA_NAMES) - set(enabled)
    for camera_name in sorted(disabled):
        unexpected = sorted(_CAMERA_DATASET_PATHS[camera_name] & set(datasets))
        unexpected.extend(
            sorted(
                path
                for path in datasets
                if path.startswith(f"observations/extra/{camera_name}_")
            )
        )
        for path in unexpected:
            report.add_error(
                "disabled_camera_dataset",
                f"{location}/{path}",
                f"{camera_name} is not enabled; its datasets must be absent",
            )

    skew_path = "observations/timing/camera_skew_ns"
    if len(enabled) == 1 and skew_path in datasets:
        report.add_error(
            "single_camera_skew_dataset",
            f"{location}/{skew_path}",
            "camera_skew_ns must be absent from one-camera sessions",
        )


def _root_image_shape(file: h5py.File) -> tuple[int, int, int] | None:
    try:
        return (
            int(file.attrs["image_height"]),
            int(file.attrs["image_width"]),
            int(file.attrs["image_channels"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _validate_mapping(
    datasets: dict[str, h5py.Dataset],
    location: str,
    n_obs: int,
    n_actions: int,
    complete: bool,
    report: ValidationReport,
) -> None:
    if n_actions == 0:
        return
    obs_dataset = datasets.get("actions/mapping/observation_index")
    next_dataset = datasets.get("actions/mapping/next_observation_index")
    if obs_dataset is None or next_dataset is None:
        return
    observation_indexes = np.asarray(obs_dataset)
    next_indexes = np.asarray(next_dataset)
    if observation_indexes.shape != (n_actions,) or next_indexes.shape != (n_actions,):
        return
    expected = np.arange(n_actions, dtype=np.int64)
    if not np.array_equal(observation_indexes, expected):
        report.add_error(
            "observation_mapping",
            location,
            "observation_index must equal the action row index",
        )
    if complete or n_obs == n_actions + 1:
        expected_next = expected + 1
    else:
        expected_next = expected + 1
        expected_next[-1] = -1
    if not np.array_equal(next_indexes, expected_next):
        report.add_error(
            "next_observation_mapping",
            location,
            f"next_observation_index is invalid; expected {expected_next.tolist()}",
        )


def _validate_action_semantics(
    datasets: dict[str, h5py.Dataset],
    location: str,
    n_actions: int,
    training_action_clip: float,
    report: ValidationReport,
) -> None:
    if n_actions == 0:
        return
    raw = datasets.get("actions/policy/sample_raw_normalized")
    clipped = datasets.get("actions/policy/sample_clipped_normalized")
    if raw is not None and clipped is not None:
        raw_values = np.asarray(raw)
        clipped_values = np.asarray(clipped)
        if raw_values.shape == clipped_values.shape and not np.allclose(
            np.clip(raw_values, -training_action_clip, training_action_clip),
            clipped_values,
            rtol=1e-6,
            atol=1e-6,
            equal_nan=False,
        ):
            report.add_error(
                "policy_clip",
                location,
                "sample_clipped_normalized must use the recorded training_action_clip",
            )
    std = datasets.get("actions/policy/std_raw_normalized")
    if std is not None and np.any(np.asarray(std) < 0):
        report.add_error("policy_std", location, "Policy standard deviation cannot be negative")

    mean = datasets.get("actions/policy/mu_raw_normalized")
    effective = datasets.get("actions/action/effective_software_action_normalized")
    residual = datasets.get("labels/residual_effective_software_action_minus_raw_mean")
    valid = datasets.get("labels/command_valid")
    published = datasets.get("actions/action/command_published")
    if mean is not None and effective is not None and residual is not None and valid is not None:
        mean_values = np.asarray(mean)
        effective_values = np.asarray(effective)
        residual_values = np.asarray(residual)
        valid_values = np.asarray(valid, dtype=bool)
        if (
            mean_values.shape == (n_actions, 6)
            and effective_values.shape == (n_actions, 6)
            and residual_values.shape == (n_actions, 6)
            and valid_values.shape == (n_actions,)
        ):
            expected = effective_values - mean_values
            if np.any(valid_values) and not np.allclose(
                residual_values[valid_values],
                expected[valid_values],
                rtol=1e-6,
                atol=1e-6,
                equal_nan=False,
            ):
                report.add_error(
                    "canonical_label",
                    location,
                    "Valid residual labels must equal effective software action minus "
                    "same-step raw mean",
                )
            invalid_populated = ~valid_values & ~np.isnan(residual_values).all(axis=1)
            if np.any(invalid_populated):
                report.add_error(
                    "invalid_label_not_nan",
                    location,
                    "Invalid or shadow-mode residual labels must be all-NaN",
                )
    if valid is not None and published is not None:
        valid_values = np.asarray(valid, dtype=bool)
        published_values = np.asarray(published, dtype=bool)
        if valid_values.shape == published_values.shape and np.any(
            valid_values & ~published_values
        ):
            report.add_error(
                "label_without_command",
                location,
                "A command label cannot be valid when no command was published",
            )


def _validate_task_semantics(
    datasets: dict[str, h5py.Dataset],
    location: str,
    n_obs: int,
    report: ValidationReport,
) -> None:
    flange = datasets.get("observations/robot/flange_pose_world_xyzw")
    exact_root = datasets.get("observations/task/socket_root_exact_world_xyzw")
    distance = datasets.get("observations/task/flange_to_socket_root_exact_distance_m")
    if flange is not None and exact_root is not None and distance is not None:
        flange_values = np.asarray(flange)
        exact_values = np.asarray(exact_root)
        distance_values = np.asarray(distance)
        if (
            flange_values.shape == (n_obs, 7)
            and exact_values.shape == (n_obs, 7)
            and distance_values.shape == (n_obs,)
        ):
            expected = np.linalg.norm(flange_values[:, :3] - exact_values[:, :3], axis=1)
            if not np.allclose(distance_values, expected, rtol=1e-6, atol=1e-7):
                report.add_error(
                    "exact_root_distance",
                    location,
                    "flange_to_socket_root_exact_distance_m does not match "
                    "flange-to-exact-root L2 distance",
                )
    # Optional keypoint metric, recomputable because policy/input[:, :3] is the
    # world-frame EEF keypoint (the EEF offset itself is not recorded).
    exact_keypoint = datasets.get("observations/task/socket_keypoint_exact_world_xyzw")
    policy_input = datasets.get("observations/policy/input")
    keypoint_distance = datasets.get(
        "observations/task/eef_keypoint_to_socket_keypoint_distance_m"
    )
    if (
        exact_keypoint is not None
        and policy_input is not None
        and keypoint_distance is not None
    ):
        keypoint_values = np.asarray(exact_keypoint)
        input_values = np.asarray(policy_input, dtype=np.float64)
        distance_values = np.asarray(keypoint_distance)
        if (
            keypoint_values.shape == (n_obs, 7)
            and input_values.shape == (n_obs, 18)
            and distance_values.shape == (n_obs,)
        ):
            expected = np.linalg.norm(
                input_values[:, :3] - keypoint_values[:, :3], axis=1
            )
            # Rows whose policy input could not be encoded are stored as NaN and
            # carry no keypoint information to check against.
            comparable = np.isfinite(expected) & np.isfinite(distance_values)
            if np.any(comparable) and not np.allclose(
                distance_values[comparable],
                expected[comparable],
                # policy/input is float32, so allow single-precision slack.
                rtol=1e-5,
                atol=1e-6,
            ):
                report.add_error(
                    "eef_keypoint_distance",
                    location,
                    "eef_keypoint_to_socket_keypoint_distance_m does not match the "
                    "EEF-keypoint-to-exact-socket-keypoint L2 distance",
                )
    randomized_start = datasets.get("observations/task/randomized_start_flange_world_xyzw")
    if randomized_start is not None and randomized_start.shape[0] > 1:
        values = np.asarray(randomized_start)
        if not np.allclose(values, values[0], rtol=0.0, atol=1e-12):
            report.add_warning(
                "randomized_start_changed",
                location,
                "randomized_start_flange_world_xyzw should remain constant within a trial",
            )


def _validate_time_order(
    datasets: dict[str, h5py.Dataset],
    location: str,
    report: ValidationReport,
) -> None:
    for path in (
        "observations/timing/ros_stamp_ns",
        "observations/timing/robot_stamp_ns",
        "observations/timing/monotonic_receive_ns",
        "actions/timing/step_start_monotonic_ns",
    ):
        dataset = datasets.get(path)
        if dataset is None or dataset.shape[0] < 2:
            continue
        values = np.asarray(dataset, dtype=np.int64)
        if np.any(np.diff(values) < 0):
            report.add_warning(
                "nonmonotonic_time",
                f"{location}/{path}",
                "Timestamps move backwards",
            )


def recover_partial_file(
    path: str | os.PathLike[str],
    *,
    destination: str | os.PathLike[str] | None = None,
    complete_trials_only: bool = True,
) -> RecoveryResult:
    """Make a non-destructive recovered copy of a partial collection file.

    The source is never changed.  By default, any in-progress or structurally
    incomplete trailing trials are removed from the recovered copy.
    """

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if destination is None:
        text = str(source)
        if text.endswith(".partial.h5"):
            target = Path(text[: -len(".partial.h5")] + ".recovered.h5")
        elif text.endswith(".h5"):
            target = Path(text[: -len(".h5")] + ".recovered.h5")
        else:
            target = Path(text + ".recovered.h5")
    else:
        target = Path(destination)
    if target.exists():
        raise FileExistsError(target)
    if target.resolve() == source.resolve():
        raise ValueError("Recovery destination must differ from the source")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

    kept: list[str] = []
    dropped: list[str] = []
    try:
        with h5py.File(target, "r+") as file:
            trials = file.get("trials")
            if not isinstance(trials, h5py.Group):
                raise ValueError("Source has no trials group")
            for name in sorted(trials.keys()):
                trial = trials[name]
                structurally_complete = isinstance(trial, h5py.Group) and _is_complete_trial(trial)
                if complete_trials_only and not structurally_complete:
                    del trials[name]
                    dropped.append(name)
                else:
                    kept.append(name)
            trials.attrs["count"] = len(trials)
            file.attrs["closed_cleanly"] = True
            file.attrs["session_complete"] = bool(complete_trials_only)
            file.attrs["recovered_utc"] = _utc_now()
            file.attrs["recovered_from"] = str(source.resolve())
            file.flush()
    except Exception:
        try:
            target.unlink()
        except OSError:
            pass
        raise
    report = validate_file(target, allow_partial=not complete_trials_only)
    return RecoveryResult(source, target, tuple(kept), tuple(dropped), report)


def _format_issues(issues: Iterable[ValidationIssue]) -> str:
    return "\n".join(str(issue) for issue in issues)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    exit_code = 0
    for file_path in args.paths:
        report = validate_file(file_path, allow_partial=args.allow_partial)
        print(report.summary())
        if report.issues:
            print(_format_issues(report.issues))
        if not report.is_valid:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RecoveryResult",
    "ValidationIssue",
    "ValidationReport",
    "recover_partial_file",
    "validate_file",
]
