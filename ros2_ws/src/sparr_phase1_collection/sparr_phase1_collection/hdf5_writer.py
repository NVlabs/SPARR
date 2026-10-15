"""Incremental, crash-tolerant HDF5 writer for Phase 1 collection.

Files are always written through a ``.partial.h5`` path.  A clean close flushes
the file, marks it closed, and atomically renames it to ``.h5``.  A process
failure leaves the partial file in place for validation or recovery.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
from typing import Any, Mapping, Sequence
import uuid

import h5py
import numpy as np

from .hdf5_schema import (
    ACTION_CONVENTION,
    ALL_DATASET_SPECS,
    IMAGE_STORAGE_CONVENTION,
    POSE_CONVENTION,
    REQUIRED_ACTION_PATHS,
    ROOT_REQUIRED_ATTRIBUTES,
    ROOT_REQUIRED_RUNTIME_ATTRIBUTES,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    SUPPORTED_CAMERA_NAMES,
    encode_enabled_cameras,
    is_extension_path,
    normalize_enabled_cameras,
    normalize_record_paths,
    required_observation_paths,
)


_FORBIDDEN_PATH_TOKENS = frozenset(
    {"full_res", "full_resolution", "invented_gripper"}
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _paths(path: str | os.PathLike[str]) -> tuple[Path, Path]:
    requested = Path(path)
    text = str(requested)
    if text.endswith(".partial.h5"):
        partial = requested
        final = Path(text[: -len(".partial.h5")] + ".h5")
    elif text.endswith(".h5"):
        final = requested
        partial = Path(text[: -len(".h5")] + ".partial.h5")
    else:
        final = Path(text + ".h5")
        partial = Path(text + ".partial.h5")
    return final, partial


def _safe_attr_value(value: Any) -> Any:
    if value is None:
        return "null"
    if isinstance(value, (str, bytes, bool, int, float, np.generic)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind != "O":
            return value
        return json.dumps(value.tolist(), sort_keys=True, default=str)
    if isinstance(value, (list, tuple)):
        array = np.asarray(value)
        if array.dtype.kind in "biufS":
            return array
    return json.dumps(value, sort_keys=True, default=str)


def _write_named_value(parent: h5py.Group, key: str, value: Any) -> None:
    """Write arbitrary named metadata without interpreting its semantics."""

    clean_key = str(key).strip("/")
    if not clean_key or "/" in clean_key or clean_key in {".", ".."}:
        raise ValueError(f"Invalid metadata key: {key!r}")
    if isinstance(value, Mapping):
        group = parent.create_group(clean_key)
        for child_key, child_value in value.items():
            _write_named_value(group, str(child_key), child_value)
        return
    if value is None:
        parent.create_dataset(clean_key, data="null", dtype=h5py.string_dtype("utf-8"))
        return
    if isinstance(value, (str, Path)):
        parent.create_dataset(
            clean_key,
            data=str(value),
            dtype=h5py.string_dtype("utf-8"),
        )
        return
    if isinstance(value, bytes):
        parent.create_dataset(clean_key, data=np.void(value))
        return
    array = np.asarray(value)
    if array.dtype.kind == "U":
        string_dtype = h5py.string_dtype("utf-8")
        parent.create_dataset(clean_key, data=array.astype(object), dtype=string_dtype)
        return
    if array.dtype.kind == "O":
        parent.create_dataset(
            clean_key,
            data=json.dumps(value, sort_keys=True, default=str),
            dtype=h5py.string_dtype("utf-8"),
        )
        return
    parent.create_dataset(clean_key, data=array)


def _has_forbidden_path_token(path: str) -> bool:
    tokens = {part.lower() for part in path.split("/")}
    return bool(tokens & _FORBIDDEN_PATH_TOKENS) or any(
        "full_res" in token for token in tokens
    )


def _extension_dtype_and_shape(value: Any) -> tuple[Any, tuple[int, ...], Any]:
    if isinstance(value, (str, Path)):
        return h5py.string_dtype("utf-8"), (), ""
    array = np.asarray(value)
    if array.dtype.kind == "U":
        return h5py.string_dtype("utf-8"), array.shape, ""
    if array.dtype.kind == "O":
        raise TypeError("Extension datasets must contain numeric, boolean, or UTF-8 values")
    if array.dtype.kind == "f":
        fill = np.nan
    elif array.dtype.kind == "b":
        fill = False
    elif array.dtype.kind == "u":
        fill = 0
    elif array.dtype.kind == "i":
        fill = -1
    elif array.dtype.kind in "SU":
        fill = ""
    else:
        raise TypeError(f"Unsupported extension dtype: {array.dtype}")
    return array.dtype, array.shape, fill


class HDF5SessionWriter:
    """Write a collection session incrementally to a recoverable HDF5 file."""

    def __init__(
        self,
        final_path: str | os.PathLike[str],
        *,
        metadata: Mapping[str, Any] | None = None,
        calibration: Mapping[str, Any] | None = None,
        root_attrs: Mapping[str, Any] | None = None,
        image_shape: tuple[int, int, int] = (128, 128, 3),
        enabled_cameras: Sequence[str] = SUPPORTED_CAMERA_NAMES,
        training_action_clip: float | None = None,
        flush_every: int = 30,
        overwrite: bool = False,
    ) -> None:
        if len(image_shape) != 3 or any(int(v) <= 0 for v in image_shape):
            raise ValueError("image_shape must be a positive (height, width, channels) tuple")
        if int(image_shape[2]) != 3:
            raise ValueError("Only cropped three-channel RGB images are supported")
        if flush_every <= 0:
            raise ValueError("flush_every must be positive")
        self.enabled_cameras = normalize_enabled_cameras(enabled_cameras)
        supplied_root_attrs = dict(root_attrs or {})
        if training_action_clip is None:
            training_action_clip = supplied_root_attrs.pop("training_action_clip", None)
        elif "training_action_clip" in supplied_root_attrs:
            raise ValueError("Provide training_action_clip directly or in root_attrs, not both")
        if training_action_clip is None:
            raise ValueError(
                "training_action_clip is required and must come from the resolved training YAML"
            )
        training_action_clip = float(training_action_clip)
        if not np.isfinite(training_action_clip) or training_action_clip <= 0.0:
            raise ValueError("training_action_clip must be a positive finite value")
        protected_root_attrs = set(ROOT_REQUIRED_ATTRIBUTES) | set(
            ROOT_REQUIRED_RUNTIME_ATTRIBUTES
        ) | {
            "session_uuid",
            "created_utc",
            "closed_cleanly",
            "session_complete",
            "image_height",
            "image_width",
            "image_channels",
            "enabled_cameras",
            "camera_count",
        }
        overlap = protected_root_attrs & set(supplied_root_attrs)
        if overlap:
            raise ValueError(
                f"root_attrs may not override protected attributes: {sorted(overlap)}"
            )

        self.final_path, self.partial_path = _paths(final_path)
        self.image_shape = tuple(int(v) for v in image_shape)
        self.training_action_clip = training_action_clip
        self.flush_every = int(flush_every)
        self._append_since_flush = 0
        self._closed = False
        self._current_trial: TrialWriter | None = None
        self._lock = threading.RLock()

        self.partial_path.parent.mkdir(parents=True, exist_ok=True)
        if not overwrite:
            collisions = [path for path in (self.final_path, self.partial_path) if path.exists()]
            if collisions:
                joined = ", ".join(str(path) for path in collisions)
                raise FileExistsError(f"Refusing to overwrite existing collection file: {joined}")

        self._file = h5py.File(self.partial_path, "w", libver="latest")
        attrs = {
            **ROOT_REQUIRED_ATTRIBUTES,
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "session_uuid": str(uuid.uuid4()),
            "created_utc": _utc_now(),
            "closed_cleanly": False,
            "session_complete": False,
            "pose_convention": POSE_CONVENTION,
            "action_convention": ACTION_CONVENTION,
            "image_storage": IMAGE_STORAGE_CONVENTION,
            "image_height": self.image_shape[0],
            "image_width": self.image_shape[1],
            "image_channels": self.image_shape[2],
            "enabled_cameras": encode_enabled_cameras(self.enabled_cameras),
            "camera_count": len(self.enabled_cameras),
            "training_action_clip": self.training_action_clip,
        }
        if supplied_root_attrs:
            attrs.update(supplied_root_attrs)
        for key, value in attrs.items():
            self._file.attrs[str(key)] = _safe_attr_value(value)

        metadata_group = self._file.create_group("metadata")
        calibration_group = self._file.create_group("calibration")
        trials_group = self._file.create_group("trials")
        trials_group.attrs["count"] = 0
        for key, value in (metadata or {}).items():
            _write_named_value(metadata_group, str(key), value)
        for key, value in (calibration or {}).items():
            _write_named_value(calibration_group, str(key), value)
        self._file.flush()

    @property
    def session_uuid(self) -> str:
        return str(self._file.attrs["session_uuid"])

    @property
    def is_closed(self) -> bool:
        return self._closed

    def start_trial(
        self,
        *,
        attempt_index: int,
        attrs: Mapping[str, Any] | None = None,
        trial_id: str | None = None,
    ) -> "TrialWriter":
        """Start the next sequential trial.

        Only one live trial is allowed.  ``trial_id`` changes the UUID attribute,
        not the deterministic group name.
        """

        with self._lock:
            self._ensure_open()
            if self._current_trial is not None and not self._current_trial.is_finished:
                raise RuntimeError("Finish the current trial before starting another")
            trials = self._file["trials"]
            index = len(trials)
            group_name = f"trial_{index:06d}"
            group = trials.create_group(group_name)
            trial = TrialWriter(
                session=self,
                group=group,
                attempt_index=int(attempt_index),
                trial_id=trial_id,
                attrs=attrs,
            )
            self._current_trial = trial
            trials.attrs.modify("count", index + 1)
            self._file.flush()
            return trial

    def flush(self) -> None:
        with self._lock:
            self._ensure_open()
            self._file.flush()
            self._append_since_flush = 0

    def close(self, *, clean: bool = True) -> Path:
        """Close the file, atomically promoting it only when ``clean`` is true."""

        with self._lock:
            if self._closed:
                return self.final_path if clean and self.final_path.exists() else self.partial_path
            if clean and self._current_trial is not None and not self._current_trial.is_finished:
                raise RuntimeError("Cannot cleanly close while a trial is unfinished")
            if clean:
                self._file.attrs["closed_cleanly"] = True
                self._file.attrs["session_complete"] = True
                self._file.attrs["completed_utc"] = _utc_now()
            self._file.flush()
            self._file.close()
            self._closed = True
            if clean:
                os.replace(self.partial_path, self.final_path)
                return self.final_path
            return self.partial_path

    def _note_append(self) -> None:
        self._append_since_flush += 1
        if self._append_since_flush >= self.flush_every:
            self._file.flush()
            self._append_since_flush = 0

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("HDF5SessionWriter is closed")

    def __enter__(self) -> "HDF5SessionWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is not None:
            self.close(clean=False)
            return
        try:
            self.close(clean=True)
        except Exception:
            self.close(clean=False)
            raise


class TrialWriter:
    """Streaming writer for one trial group."""

    def __init__(
        self,
        *,
        session: HDF5SessionWriter,
        group: h5py.Group,
        attempt_index: int,
        trial_id: str | None,
        attrs: Mapping[str, Any] | None,
    ) -> None:
        self._session = session
        self._group = group
        self._datasets: dict[str, h5py.Dataset] = {}
        self._observation_count = 0
        self._action_count = 0
        self._pending_action_index: int | None = None
        self._finished = False
        initial_attrs = {
            "trial_uuid": str(trial_id or uuid.uuid4()),
            "attempt_index": int(attempt_index),
            "complete": False,
            "accepted": False,
            "operator_label": "unlabeled",
            "auto_outcome": "unknown",
            "termination_reason": "in_progress",
            "start_utc": _utc_now(),
            "observation_count": 0,
            "action_count": 0,
        }
        if attrs:
            protected = set(initial_attrs)
            overlap = protected & set(attrs)
            if overlap:
                raise ValueError(
                    f"trial attrs may not override protected attributes: {sorted(overlap)}"
                )
            initial_attrs.update(attrs)
        for key, value in initial_attrs.items():
            self._group.attrs[str(key)] = _safe_attr_value(value)

    @property
    def name(self) -> str:
        return self._group.name.rsplit("/", 1)[-1]

    @property
    def trial_uuid(self) -> str:
        return str(self._group.attrs["trial_uuid"])

    @property
    def observation_count(self) -> int:
        return self._observation_count

    @property
    def action_count(self) -> int:
        return self._action_count

    @property
    def is_finished(self) -> bool:
        return self._finished

    def append_observation(self, record: Mapping[str, Any]) -> int:
        """Append one observation and resolve any immediately preceding action."""

        with self._session._lock:
            self._ensure_live()
            normalized = normalize_record_paths(record, record_kind="observation")
            if "wrist_2" not in self._session.enabled_cameras:
                disabled_paths = [
                    path
                    for path in normalized
                    if path
                    in {
                        "observations/timing/wrist_2_stamp_ns",
                        "observations/timing/wrist_2_age_ns",
                        "observations/validity/wrist_2",
                        "observations/camera/wrist_2/rgb",
                        "observations/timing/camera_skew_ns",
                    }
                    or path.startswith("observations/extra/wrist_2_")
                ]
                if disabled_paths:
                    raise ValueError(
                        "Disabled wrist_2 datasets must be absent: "
                        f"{sorted(disabled_paths)}"
                    )
            index = self._observation_count
            self._append_row(
                normalized,
                record_kind="observation",
                row_index=index,
                required=required_observation_paths(self._session.enabled_cameras),
            )
            self._observation_count += 1
            self._group.attrs.modify("observation_count", self._observation_count)
            if self._pending_action_index is not None:
                mapping = self._datasets["actions/mapping/next_observation_index"]
                mapping[self._pending_action_index] = index
                self._pending_action_index = None
            self._session._note_append()
            return index

    def append_action(self, record: Mapping[str, Any]) -> int:
        """Append an action generated from the latest observation.

        Mapping fields are assigned by the writer.  The next-observation index
        remains -1 until the subsequent observation is appended.
        """

        with self._session._lock:
            self._ensure_live()
            if self._observation_count != self._action_count + 1:
                raise RuntimeError("An action requires one unmatched latest observation")
            normalized = normalize_record_paths(record, record_kind="action")
            observation_index = self._observation_count - 1
            supplied_observation = normalized.get("actions/mapping/observation_index")
            if supplied_observation is not None and int(supplied_observation) != observation_index:
                raise ValueError("Supplied observation_index does not match the writer sequence")
            supplied_next = normalized.get("actions/mapping/next_observation_index")
            if supplied_next is not None and int(supplied_next) != -1:
                raise ValueError("next_observation_index is writer-managed and must start at -1")
            normalized["actions/mapping/observation_index"] = observation_index
            normalized["actions/mapping/next_observation_index"] = -1
            self._validate_action_semantics(normalized)
            index = self._action_count
            self._append_row(
                normalized,
                record_kind="action",
                row_index=index,
                required=REQUIRED_ACTION_PATHS,
            )
            self._action_count += 1
            self._pending_action_index = index
            self._group.attrs.modify("action_count", self._action_count)
            self._session._note_append()
            return index

    def invalidate_pending_action(self, *, guard_bitmask: int = 0) -> bool:
        """Mark the unmatched last action ineligible after an input/sync failure.

        Used when an action was submitted but the next synchronized observation
        is stale or invalid; the submitted command trace is unchanged.
        """

        with self._session._lock:
            self._ensure_live()
            if guard_bitmask < 0 or guard_bitmask > np.iinfo(np.uint32).max:
                raise ValueError("guard_bitmask must fit uint32")
            if self._pending_action_index is None:
                return False
            index = self._pending_action_index
            self._datasets[
                "labels/residual_effective_software_action_minus_raw_mean"
            ][index] = np.full(6, np.nan, dtype=np.float32)
            self._datasets["labels/command_valid"][index] = False
            measured_valid = self._datasets.get("labels/measured_valid")
            if measured_valid is not None:
                measured_valid[index] = False
            measured_residual = self._datasets.get(
                "labels/residual_measured_minus_raw_mean"
            )
            if measured_residual is not None:
                measured_residual[index] = np.full(6, np.nan, dtype=np.float32)
            current_mask = int(self._datasets["actions/action/guard_bitmask"][index])
            self._datasets["actions/action/guard_bitmask"][index] = np.uint32(
                current_mask | int(guard_bitmask)
            )
            self._session._file.flush()
            return True

    def finish(
        self,
        *,
        accepted: bool,
        operator_label: str,
        auto_outcome: str,
        termination_reason: str,
        accepted_success_index: int | None = None,
        reset_randomization_xyz_yaw: Any | None = None,
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        """Finalize a structurally complete trial and record its outcome."""

        with self._session._lock:
            self._ensure_live()
            if self._pending_action_index is not None:
                raise RuntimeError(
                    "Cannot finish until the final post-action observation is recorded"
                )
            if self._observation_count != self._action_count + 1:
                raise RuntimeError(
                    "Complete trials require N observations and M actions with N == M + 1"
                )
            if not operator_label:
                raise ValueError("operator_label must be non-empty")
            if not auto_outcome:
                raise ValueError("auto_outcome must be non-empty")
            if not termination_reason:
                raise ValueError("termination_reason must be non-empty")
            updates: dict[str, Any] = {
                "complete": True,
                "accepted": bool(accepted),
                "operator_label": str(operator_label),
                "auto_outcome": str(auto_outcome),
                "termination_reason": str(termination_reason),
                "end_utc": _utc_now(),
            }
            if accepted_success_index is not None:
                if not accepted:
                    raise ValueError("accepted_success_index is only valid for accepted trials")
                if accepted_success_index < 0 or accepted_success_index >= self._action_count:
                    raise ValueError("accepted_success_index must identify a recorded action")
                updates["accepted_success_index"] = int(accepted_success_index)
            if reset_randomization_xyz_yaw is not None:
                randomization = np.asarray(reset_randomization_xyz_yaw, dtype=np.float64)
                if randomization.shape != (4,):
                    raise ValueError("reset_randomization_xyz_yaw must have shape (4,)")
                updates["reset_randomization_xyz_yaw"] = randomization
            if attrs:
                protected = set(updates) | {
                    "trial_uuid",
                    "attempt_index",
                    "observation_count",
                    "action_count",
                    "start_utc",
                }
                overlap = protected & set(attrs)
                if overlap:
                    raise ValueError(
                        f"finish attrs may not override protected attributes: {sorted(overlap)}"
                    )
                updates.update(attrs)
            for key, value in updates.items():
                self._group.attrs[str(key)] = _safe_attr_value(value)
            self._finished = True
            self._session._file.flush()

    def _append_row(
        self,
        record: Mapping[str, Any],
        *,
        record_kind: str,
        row_index: int,
        required: frozenset[str],
    ) -> None:
        missing = required - set(record)
        if missing:
            raise KeyError(f"Missing required {record_kind} datasets: {sorted(missing)}")
        for path in record:
            if _has_forbidden_path_token(path):
                raise ValueError(f"Forbidden dataset path: {path}")
            if path not in ALL_DATASET_SPECS and not is_extension_path(
                path, record_kind=record_kind
            ):
                raise KeyError(
                    f"Unknown {record_kind} dataset {path!r}; custom fields must use "
                    "an extra namespace"
                )
            if record_kind == "observation" and not path.startswith("observations/"):
                raise KeyError(f"Observation record cannot write {path!r}")
            if record_kind == "action" and not path.startswith(("actions/", "labels/")):
                raise KeyError(f"Action record cannot write {path!r}")

        for path, value in record.items():
            if path not in self._datasets:
                self._datasets[path] = self._create_dataset(
                    path,
                    value,
                    prior_rows=row_index,
                    record_kind=record_kind,
                )

        if record_kind == "observation":
            datasets = {
                path: dataset
                for path, dataset in self._datasets.items()
                if path.startswith("observations/")
            }
        else:
            datasets = {
                path: dataset
                for path, dataset in self._datasets.items()
                if path.startswith(("actions/", "labels/"))
            }
        prepared = {
            path: self._coerce_value(path, record[path], dataset)
            for path, dataset in datasets.items()
            if path in record
        }
        for path, dataset in datasets.items():
            dataset.resize(row_index + 1, axis=0)
            if path in record:
                dataset[row_index] = prepared[path]
            else:
                dataset[row_index] = dataset.fillvalue

    def _create_dataset(
        self,
        path: str,
        value: Any,
        *,
        prior_rows: int,
        record_kind: str,
    ) -> h5py.Dataset:
        schema_spec = ALL_DATASET_SPECS.get(path)
        if schema_spec is not None:
            dtype = schema_spec.numpy_dtype()
            if schema_spec.image:
                tail_shape = self._session.image_shape
            elif schema_spec.tail_shape is None:
                tail_shape = np.asarray(value).shape
            else:
                tail_shape = schema_spec.tail_shape
            fill_value = schema_spec.fill_value
        else:
            if not is_extension_path(path, record_kind=record_kind):
                raise KeyError(path)
            dtype, tail_shape, fill_value = _extension_dtype_and_shape(value)

        value_shape = np.asarray(value).shape
        if value_shape != tail_shape:
            raise ValueError(
                f"Dataset {path!r} expects per-row shape {tail_shape}, got {value_shape}"
            )
        if schema_spec is not None and schema_spec.image:
            array = np.asarray(value)
            if array.dtype != np.dtype(np.uint8):
                raise TypeError(f"Dataset {path!r} requires uint8 cropped RGB data")

        parent_path, leaf = path.rsplit("/", 1)
        parent = self._group.require_group(parent_path)
        chunks = (1, *tail_shape) if tail_shape else (min(256, max(1, self._session.flush_every)),)
        use_compression = bool(tail_shape) and h5py.check_string_dtype(np.dtype(dtype)) is None
        dataset = parent.create_dataset(
            leaf,
            shape=(prior_rows, *tail_shape),
            maxshape=(None, *tail_shape),
            chunks=chunks,
            dtype=dtype,
            fillvalue=fill_value,
            compression="lzf" if use_compression else None,
        )
        if schema_spec is not None:
            dataset.attrs["schema_defined"] = True
            if schema_spec.unit:
                dataset.attrs["unit"] = schema_spec.unit
            if schema_spec.description:
                dataset.attrs["description"] = schema_spec.description
        else:
            dataset.attrs["schema_defined"] = False
        return dataset

    def _coerce_value(
        self,
        path: str,
        value: Any,
        dataset: h5py.Dataset,
    ) -> Any:
        if h5py.check_string_dtype(dataset.dtype) is not None:
            array = np.asarray(value, dtype=object)
        else:
            array = np.asarray(value)
            if path in ALL_DATASET_SPECS:
                expected = ALL_DATASET_SPECS[path].numpy_dtype()
                if array.dtype != expected:
                    if expected.kind == "b" and array.dtype.kind != "b":
                        raise TypeError(
                            f"Dataset {path!r} requires {expected}, got {array.dtype}"
                        )
                    if expected.kind in "iu":
                        if array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)):
                            raise TypeError(
                                f"Dataset {path!r} requires finite integer values"
                            )
                        if np.any(array != np.floor(array)):
                            raise TypeError(f"Dataset {path!r} requires integer values")
                        limits = np.iinfo(expected)
                        if np.any(array < limits.min) or np.any(array > limits.max):
                            raise OverflowError(f"Dataset {path!r} exceeds {expected} range")
                    elif expected.kind == "f" and array.dtype.kind not in "biuf":
                        raise TypeError(
                            f"Dataset {path!r} requires numeric values, got {array.dtype}"
                        )
                    array = array.astype(expected, copy=False)
                else:
                    array = array.astype(expected, copy=False)
            else:
                if not np.can_cast(array.dtype, dataset.dtype, casting="safe"):
                    raise TypeError(
                        f"Extension dataset {path!r} was fixed as {dataset.dtype}, "
                        f"got {array.dtype}"
                    )
                array = array.astype(dataset.dtype, copy=False)
        if array.shape != dataset.shape[1:]:
            raise ValueError(
                f"Dataset {path!r} expects per-row shape {dataset.shape[1:]}, got {array.shape}"
            )
        return array

    def _validate_action_semantics(self, record: Mapping[str, Any]) -> None:
        raw = np.asarray(record["actions/policy/sample_raw_normalized"], dtype=np.float32)
        clipped = np.asarray(
            record["actions/policy/sample_clipped_normalized"], dtype=np.float32
        )
        expected_clipped = np.clip(
            raw,
            -self._session.training_action_clip,
            self._session.training_action_clip,
        )
        if not np.allclose(clipped, expected_clipped, rtol=1e-6, atol=1e-6):
            raise ValueError(
                "sample_clipped_normalized must use the recorded training_action_clip"
            )
        mean = np.asarray(record["actions/policy/mu_raw_normalized"], dtype=np.float32)
        effective = np.asarray(
            record["actions/action/effective_software_action_normalized"],
            dtype=np.float32,
        )
        residual = np.asarray(
            record["labels/residual_effective_software_action_minus_raw_mean"],
            dtype=np.float32,
        )
        label_valid = bool(record["labels/command_valid"])
        published = bool(record["actions/action/command_published"])
        if label_valid:
            if not published:
                raise ValueError("A command label cannot be valid when no command was published")
            if not np.allclose(residual, effective - mean, rtol=1e-6, atol=1e-6):
                raise ValueError(
                    "Canonical label must equal effective software action minus same-step raw mean"
                )
        elif not np.isnan(residual).all():
            raise ValueError("Invalid or shadow-mode command labels must be all-NaN")

    def _ensure_live(self) -> None:
        self._session._ensure_open()
        if self._finished:
            raise RuntimeError("TrialWriter is already finished")


__all__ = ["HDF5SessionWriter", "TrialWriter"]
