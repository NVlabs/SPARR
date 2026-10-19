# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Canonical HDF5 schema definitions for SPARR Phase 1.

The live collector writes observations and actions incrementally.  A complete
trial has N observations and M actions where N == M + 1.  Dataset paths in this
module are relative to a trial group.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


SCHEMA_NAME = "sparr_phase1"
SCHEMA_VERSION = "1.0.0"
POSE_CONVENTION = "world pose [x,y,z,qx,qy,qz,qw]"
ACTION_CONVENTION = (
    "normalized [dx,dy,dz,rx,ry,rz]; world translation; "
    "world-frame rotvec left-multiplied onto the observed flange orientation"
)
IMAGE_STORAGE_CONVENTION = "cropped_rgb_uint8_only"
SUPPORTED_CAMERA_NAMES = ("wrist_1", "wrist_2")


class SchemaContractError(ValueError):
    """Raised when the checked-in declarative schema drifts from Python."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: yaml.Loader,
    node: yaml.Node,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise SchemaContractError(f"duplicate declarative schema key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class DatasetSpec:
    """Description of one extensible, row-oriented dataset."""

    path: str
    dtype: Any
    tail_shape: tuple[int, ...] | None
    required: bool = False
    fill_value: Any = None
    unit: str = ""
    description: str = ""
    image: bool = False

    def numpy_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)


def _spec(
    path: str,
    dtype: Any,
    tail_shape: tuple[int, ...] | None = (),
    *,
    required: bool = False,
    fill_value: Any = None,
    unit: str = "",
    description: str = "",
    image: bool = False,
) -> DatasetSpec:
    if fill_value is None:
        kind = np.dtype(dtype).kind
        if kind == "f":
            fill_value = np.nan
        elif kind in "iu":
            fill_value = 0 if kind == "u" else -1
        elif kind == "b":
            fill_value = False
    return DatasetSpec(
        path=path,
        dtype=dtype,
        tail_shape=tail_shape,
        required=required,
        fill_value=fill_value,
        unit=unit,
        description=description,
        image=image,
    )


OBSERVATION_SPECS = {
    spec.path: spec
    for spec in (
        _spec(
            "observations/timing/ros_stamp_ns",
            np.int64,
            required=True,
            unit="ns",
            description="Authoritative ROS stamp for the latched robot state.",
        ),
        _spec(
            "observations/timing/robot_stamp_ns",
            np.int64,
            required=True,
            unit="ns",
            description="Flexiv robot timestamp converted to nanoseconds.",
        ),
        _spec(
            "observations/timing/monotonic_receive_ns",
            np.int64,
            required=True,
            unit="ns",
            description="Local monotonic time when the state was received.",
        ),
        _spec("observations/timing/control_tick_monotonic_ns", np.int64, unit="ns"),
        _spec("observations/timing/wrist_1_stamp_ns", np.int64, unit="ns"),
        _spec("observations/timing/wrist_2_stamp_ns", np.int64, unit="ns"),
        _spec("observations/timing/wrist_1_age_ns", np.int64, unit="ns"),
        _spec("observations/timing/wrist_2_age_ns", np.int64, unit="ns"),
        _spec("observations/timing/camera_skew_ns", np.int64, unit="ns"),
        _spec("observations/validity/robot_state", np.bool_, required=True),
        _spec("observations/validity/wrist_1", np.bool_, required=True),
        # wrist_2 is required only when named in the enabled_cameras root
        # attribute, so one-camera sessions remain schema v1.
        _spec("observations/validity/wrist_2", np.bool_),
        _spec("observations/validity/policy_input", np.bool_, required=True),
        _spec(
            "observations/camera/wrist_1/rgb",
            np.uint8,
            None,
            required=True,
            fill_value=0,
            description="Cropped RGB wrist image. Full-resolution frames are forbidden.",
            image=True,
        ),
        _spec(
            "observations/camera/wrist_2/rgb",
            np.uint8,
            None,
            required=False,
            fill_value=0,
            description="Cropped RGB wrist image. Full-resolution frames are forbidden.",
            image=True,
        ),
        _spec(
            "observations/robot/flange_pose_world_xyzw",
            np.float64,
            (7,),
            required=True,
            unit="m, quaternion",
        ),
        _spec(
            "observations/robot/tcp_pose_world_xyzw",
            np.float64,
            (7,),
            required=True,
            unit="m, quaternion",
        ),
        _spec("observations/robot/q", np.float64, (7,), required=True, unit="rad"),
        _spec("observations/robot/theta", np.float64, (7,), unit="rad"),
        _spec("observations/robot/dq", np.float64, (7,), required=True, unit="rad/s"),
        _spec("observations/robot/dtheta", np.float64, (7,), unit="rad/s"),
        _spec("observations/robot/tau", np.float64, (7,), unit="Nm"),
        _spec("observations/robot/tau_des", np.float64, (7,), unit="Nm"),
        _spec("observations/robot/tau_dot", np.float64, (7,), unit="Nm/s"),
        _spec("observations/robot/tau_ext", np.float64, (7,), unit="Nm"),
        _spec("observations/robot/tau_interact", np.float64, (7,), unit="Nm"),
        _spec("observations/robot/temperature", np.float64, (7,), unit="degC"),
        _spec("observations/robot/tcp_velocity_world", np.float64, (6,), unit="m/s, rad/s"),
        _spec(
            "observations/robot/ft_sensor_raw_flange",
            np.float64,
            (6,),
            unit="N, Nm",
        ),
        _spec(
            "observations/robot/ext_wrench_tcp",
            np.float64,
            (6,),
            required=True,
            unit="N, Nm",
        ),
        _spec("observations/robot/ext_wrench_world", np.float64, (6,), unit="N, Nm"),
        _spec("observations/robot/ext_wrench_tcp_raw", np.float64, (6,), unit="N, Nm"),
        _spec("observations/robot/ext_wrench_world_raw", np.float64, (6,), unit="N, Nm"),
        _spec(
            "observations/task/socket_pose_supplied_world_xyzw",
            np.float64,
            (7,),
            unit="m, quaternion",
            description="Optional unmodified socket-pose provenance supplied to the run.",
        ),
        _spec(
            "observations/task/socket_root_exact_world_xyzw",
            np.float64,
            (7,),
            required=True,
            unit="m, quaternion",
        ),
        _spec(
            "observations/task/socket_root_policy_world_xyzw",
            np.float64,
            (7,),
            required=True,
            unit="m, quaternion",
        ),
        _spec(
            "observations/task/socket_keypoint_exact_world_xyzw",
            np.float64,
            (7,),
            required=True,
            unit="m, quaternion",
        ),
        _spec(
            "observations/task/socket_keypoint_policy_world_xyzw",
            np.float64,
            (7,),
            required=True,
            unit="m, quaternion",
        ),
        _spec(
            "observations/task/randomized_start_flange_world_xyzw",
            np.float64,
            (7,),
            required=True,
            unit="m, quaternion",
        ),
        _spec(
            "observations/task/flange_to_socket_root_exact_distance_m",
            np.float64,
            required=True,
            unit="m",
            description="L2 distance from the observed flange origin to the exact socket root.",
        ),
        _spec(
            "observations/task/eef_keypoint_to_socket_keypoint_distance_m",
            np.float64,
            unit="m",
            description=(
                "Optional L2 distance from the flange-local EEF keypoint to the "
                "exact socket keypoint, i.e. the two points the policy observes."
            ),
        ),
        _spec(
            "observations/policy/input",
            np.float32,
            (18,),
            required=True,
            description="Exact ordered base-policy input.",
        ),
        _spec("observations/policy/mu_raw_normalized", np.float32, (6,)),
    )
}


ACTION_SPECS = {
    spec.path: spec
    for spec in (
        _spec(
            "actions/mapping/observation_index",
            np.int64,
            required=True,
            description="Observation used to produce this action.",
        ),
        _spec(
            "actions/mapping/next_observation_index",
            np.int64,
            required=True,
            fill_value=-1,
            description="Subsequent observation; -1 while an action is pending.",
        ),
        _spec(
            "actions/policy/mu_raw_normalized",
            np.float32,
            (6,),
            required=True,
            description="Distribution mean before clip, scale, smoothing, or guards.",
        ),
        _spec("actions/policy/std_raw_normalized", np.float32, (6,)),
        _spec(
            "actions/policy/sample_raw_normalized",
            np.float32,
            (6,),
            required=True,
            description="Stochastic sample before clipping.",
        ),
        _spec(
            "actions/policy/sample_clipped_normalized",
            np.float32,
            (6,),
            required=True,
            description="Raw sample clipped symmetrically by the recorded training_action_clip.",
        ),
        _spec("actions/policy/sample_standard_normal", np.float32, (6,)),
        _spec("actions/policy/recurrent_hidden_in", np.float32, None),
        _spec("actions/policy/recurrent_cell_in", np.float32, None),
        _spec("actions/policy/recurrent_hidden_out", np.float32, None),
        _spec("actions/policy/recurrent_cell_out", np.float32, None),
        _spec(
            "actions/action/delta_scaled_pose_rel",
            np.float64,
            (6,),
            required=True,
            unit="m, rad",
        ),
        _spec(
            "actions/action/requested_flange_target_world_xyzw",
            np.float64,
            (7,),
            required=True,
            unit="m, quaternion",
        ),
        _spec(
            "actions/action/effective_software_flange_target_world_xyzw",
            np.float64,
            (7,),
            required=True,
            unit="m, quaternion",
        ),
        _spec(
            "actions/action/submitted_controller_tcp_target_world_xyzw",
            np.float64,
            (7,),
            required=True,
            unit="m, quaternion",
        ),
        _spec(
            "actions/action/effective_software_action_normalized",
            np.float32,
            (6,),
            required=True,
            description=(
                "Action represented by the effective software flange target; "
                "not controller acceptance or measured execution."
            ),
        ),
        _spec("actions/action/measured_next_normalized", np.float32, (6,)),
        _spec("actions/action/command_wrench", np.float64, (6,), unit="N, Nm"),
        _spec("actions/action/command_velocity", np.float64, (6,), unit="m/s, rad/s"),
        _spec(
            "actions/action/command_acceleration",
            np.float64,
            (6,),
            unit="m/s^2, rad/s^2",
        ),
        _spec("actions/action/command_published", np.bool_, required=True),
        _spec("actions/action/guard_bitmask", np.uint32, required=True, fill_value=0),
        _spec("actions/timing/step_start_monotonic_ns", np.int64, required=True, unit="ns"),
        _spec("actions/timing/command_ros_stamp_ns", np.int64, unit="ns"),
        _spec("actions/timing/inference_ns", np.int64, required=True, unit="ns"),
        _spec("actions/timing/composition_ns", np.int64, unit="ns"),
        _spec("actions/timing/publish_ns", np.int64, required=True, unit="ns"),
        _spec("actions/timing/recorder_enqueue_ns", np.int64, unit="ns"),
        _spec("actions/timing/writer_queue_depth", np.int32),
        _spec(
            "labels/residual_effective_software_action_minus_raw_mean",
            np.float32,
            (6,),
            required=True,
            description="effective_software_action_normalized minus same-step raw mean.",
        ),
        _spec("labels/residual_measured_minus_raw_mean", np.float32, (6,)),
        _spec("labels/command_valid", np.bool_, required=True),
        _spec("labels/measured_valid", np.bool_),
    )
}


ALL_DATASET_SPECS = {**OBSERVATION_SPECS, **ACTION_SPECS}
REQUIRED_OBSERVATION_PATHS = frozenset(
    path for path, spec in OBSERVATION_SPECS.items() if spec.required
)
REQUIRED_ACTION_PATHS = frozenset(
    path for path, spec in ACTION_SPECS.items() if spec.required
)
LEGACY_TWO_CAMERA_REQUIRED_OBSERVATION_PATHS = REQUIRED_OBSERVATION_PATHS | frozenset(
    {
        "observations/validity/wrist_2",
        "observations/camera/wrist_2/rgb",
    }
)

_CAMERA_CONDITIONAL_PATHS = {
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

ROOT_REQUIRED_ATTRIBUTES = {
    "schema_name": SCHEMA_NAME,
    "schema_version": SCHEMA_VERSION,
    "pose_convention": POSE_CONVENTION,
    "action_convention": ACTION_CONVENTION,
    "image_storage": IMAGE_STORAGE_CONVENTION,
}

ROOT_REQUIRED_RUNTIME_ATTRIBUTES = frozenset({"training_action_clip"})


def normalize_enabled_cameras(camera_names: Sequence[str]) -> tuple[str, ...]:
    """Return the canonical one-or-two-camera session identity."""

    if isinstance(camera_names, (str, bytes)):
        raise ValueError("enabled_cameras must be a sequence of camera names")
    normalized = tuple(str(name).strip() for name in camera_names)
    if normalized not in (("wrist_1",), SUPPORTED_CAMERA_NAMES):
        raise ValueError(
            "enabled_cameras must be ['wrist_1'] or ['wrist_1', 'wrist_2']"
        )
    return normalized


def encode_enabled_cameras(camera_names: Sequence[str]) -> str:
    """Encode enabled cameras as a portable UTF-8 JSON root attribute."""

    return json.dumps(
        list(normalize_enabled_cameras(camera_names)),
        separators=(",", ":"),
    )


def decode_enabled_cameras(value: Any) -> tuple[str, ...]:
    """Decode the JSON root attribute or a native HDF5 string array."""

    if isinstance(value, np.ndarray):
        raw_values = value.reshape(-1).tolist()
    elif isinstance(value, (tuple, list)):
        raw_values = list(value)
    else:
        if isinstance(value, (bytes, np.bytes_)):
            value = bytes(value).decode("utf-8", errors="strict")
        if not isinstance(value, str):
            raise ValueError("enabled_cameras must be a JSON list or string array")
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("enabled_cameras must contain a JSON list") from exc
        if not isinstance(decoded, list):
            raise ValueError("enabled_cameras must contain a JSON list")
        raw_values = decoded
    names = tuple(
        bytes(name).decode("utf-8", errors="strict")
        if isinstance(name, (bytes, np.bytes_))
        else str(name)
        for name in raw_values
    )
    return normalize_enabled_cameras(names)


def required_observation_paths(camera_names: Sequence[str]) -> frozenset[str]:
    """Resolve unconditional and per-enabled-camera observation datasets."""

    enabled = normalize_enabled_cameras(camera_names)
    paths = set(REQUIRED_OBSERVATION_PATHS)
    for camera_name in enabled:
        paths.update(_CAMERA_CONDITIONAL_PATHS[camera_name])
    if len(enabled) == 2:
        paths.add("observations/timing/camera_skew_ns")
    return frozenset(paths)


def flatten_mapping(mapping: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested mappings into slash-delimited HDF5 paths."""

    flattened: dict[str, Any] = {}
    for raw_key, value in mapping.items():
        key = str(raw_key).strip("/")
        if not key or key in {".", ".."} or "/../" in f"/{key}/":
            raise ValueError(f"Invalid HDF5 record key: {raw_key!r}")
        path = f"{prefix}/{key}".strip("/")
        if isinstance(value, Mapping):
            nested = flatten_mapping(value, path)
            overlap = flattened.keys() & nested.keys()
            if overlap:
                raise ValueError(f"Duplicate flattened paths: {sorted(overlap)}")
            flattened.update(nested)
        else:
            if path in flattened:
                raise ValueError(f"Duplicate flattened path: {path}")
            flattened[path] = value
    return flattened


def normalize_record_paths(
    record: Mapping[str, Any],
    *,
    record_kind: str,
) -> dict[str, Any]:
    """Normalize convenience records to canonical trial-relative paths.

    Observation records may either begin with the observations key or omit it.
    Action records may begin with actions and labels.  Unprefixed action keys
    are placed below actions.
    """

    flattened = flatten_mapping(record)
    if record_kind == "observation":
        return {
            path if path.startswith("observations/") else f"observations/{path}": value
            for path, value in flattened.items()
        }
    if record_kind == "action":
        return {
            path if path.startswith(("actions/", "labels/")) else f"actions/{path}": value
            for path, value in flattened.items()
        }
    raise ValueError(f"Unsupported record kind: {record_kind!r}")


def is_extension_path(path: str, *, record_kind: str) -> bool:
    """Return whether a non-schema path is an explicitly namespaced extension."""

    if record_kind == "observation":
        return path.startswith("observations/extra/")
    return path.startswith(("actions/extra/", "labels/extra/"))


def image_paths(
    camera_names: Sequence[str] = SUPPORTED_CAMERA_NAMES,
) -> tuple[str, ...]:
    return tuple(
        f"observations/camera/{camera_name}/rgb"
        for camera_name in normalize_enabled_cameras(camera_names)
    )


def _declarative_dataset_entries(
    schema: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    datasets = schema.get("datasets")
    if not isinstance(datasets, Mapping):
        raise SchemaContractError("declarative schema must contain a datasets mapping")
    entries: dict[str, Mapping[str, Any]] = {}

    def walk(value: Any, prefix: str) -> None:
        if not isinstance(value, Mapping):
            raise SchemaContractError(f"dataset node {prefix!r} must be a mapping")
        if "dtype" in value or "shape" in value:
            if "dtype" not in value or "shape" not in value:
                raise SchemaContractError(
                    f"dataset {prefix!r} must declare both dtype and shape"
                )
            if prefix in entries:
                raise SchemaContractError(f"duplicate declarative dataset path {prefix!r}")
            entries[prefix] = value
            return
        for raw_key, child in value.items():
            key = str(raw_key).strip("/")
            if not key or key in {".", ".."} or "/../" in f"/{key}/":
                raise SchemaContractError(f"invalid declarative dataset key {raw_key!r}")
            walk(child, f"{prefix}/{key}".strip("/"))

    expected_sections = {"observations", "actions", "labels"}
    unknown_sections = set(datasets) - expected_sections
    if unknown_sections:
        raise SchemaContractError(
            f"unknown declarative dataset sections: {sorted(unknown_sections)}"
        )
    for section in sorted(expected_sections):
        if section not in datasets:
            raise SchemaContractError(f"declarative schema is missing {section!r}")
        walk(datasets[section], section)
    return entries


def load_declarative_schema(path: str | Path) -> Mapping[str, Any]:
    """Strictly load a schema YAML, rejecting duplicate keys at every level."""

    schema_path = Path(path)
    try:
        with schema_path.open("r", encoding="utf-8") as stream:
            loaded = yaml.load(stream, Loader=_UniqueKeyLoader)
    except OSError as exc:
        raise SchemaContractError(f"cannot read declarative schema {schema_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise SchemaContractError(f"invalid declarative schema YAML: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise SchemaContractError("declarative schema root must be a mapping")
    return loaded


def validate_declarative_schema(path: str | Path) -> None:
    """Verify exact dataset, dtype, shape, and required-flag parity with the YAML."""

    schema = load_declarative_schema(path)
    identity = schema.get("schema")
    if not isinstance(identity, Mapping):
        raise SchemaContractError("declarative schema is missing schema identity")
    if str(identity.get("name", "")) != SCHEMA_NAME:
        raise SchemaContractError(
            f"schema name is {identity.get('name')!r}, expected {SCHEMA_NAME!r}"
        )
    if str(identity.get("version", "")) != SCHEMA_VERSION:
        raise SchemaContractError(
            f"schema version is {identity.get('version')!r}, expected {SCHEMA_VERSION!r}"
        )

    entries = _declarative_dataset_entries(schema)
    expected_paths = set(ALL_DATASET_SPECS)
    actual_paths = set(entries)
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        raise SchemaContractError(
            f"declarative dataset paths differ; missing={missing}, extra={extra}"
        )

    errors: list[str] = []
    for dataset_path in sorted(expected_paths):
        spec = ALL_DATASET_SPECS[dataset_path]
        entry = entries[dataset_path]
        try:
            declared_dtype = np.dtype(str(entry["dtype"]))
        except (TypeError, ValueError) as exc:
            errors.append(f"{dataset_path}: invalid dtype {entry.get('dtype')!r}: {exc}")
            continue
        if declared_dtype != spec.numpy_dtype():
            errors.append(
                f"{dataset_path}: dtype {declared_dtype} != {spec.numpy_dtype()}"
            )
        declared_required = bool(entry.get("required", False))
        if declared_required != spec.required:
            errors.append(
                f"{dataset_path}: required={declared_required} != {spec.required}"
            )

        shape = entry.get("shape")
        if not isinstance(shape, (list, tuple)) or not shape:
            errors.append(f"{dataset_path}: shape must be a non-empty sequence")
            continue
        expected_rows = "N" if dataset_path.startswith("observations/") else "M"
        if str(shape[0]) != expected_rows:
            errors.append(
                f"{dataset_path}: leading shape {shape[0]!r} != {expected_rows!r}"
            )
            continue
        tail = tuple(shape[1:])
        if spec.image:
            if len(tail) != 3 or str(tail[0]) != "H" or str(tail[1]) != "W" or tail[2] != 3:
                errors.append(f"{dataset_path}: image tail must be [H, W, 3]")
        elif spec.tail_shape is not None:
            try:
                numeric_tail = tuple(int(value) for value in tail)
            except (TypeError, ValueError):
                errors.append(f"{dataset_path}: fixed tail shape must be numeric, got {tail}")
            else:
                if numeric_tail != spec.tail_shape:
                    errors.append(
                        f"{dataset_path}: tail shape {numeric_tail} != {spec.tail_shape}"
                    )

    extensions = schema.get("extensions")
    allowed = extensions.get("allowed_prefixes") if isinstance(extensions, Mapping) else None
    expected_extensions = [
        "observations/extra/",
        "actions/extra/",
        "labels/extra/",
    ]
    if list(allowed or ()) != expected_extensions:
        errors.append(
            f"extension prefixes {allowed!r} != {expected_extensions!r}"
        )
    if errors:
        raise SchemaContractError("declarative schema drift:\n- " + "\n- ".join(errors))


__all__ = [
    "ACTION_CONVENTION",
    "ACTION_SPECS",
    "ALL_DATASET_SPECS",
    "DatasetSpec",
    "IMAGE_STORAGE_CONVENTION",
    "LEGACY_TWO_CAMERA_REQUIRED_OBSERVATION_PATHS",
    "OBSERVATION_SPECS",
    "POSE_CONVENTION",
    "REQUIRED_ACTION_PATHS",
    "REQUIRED_OBSERVATION_PATHS",
    "ROOT_REQUIRED_ATTRIBUTES",
    "ROOT_REQUIRED_RUNTIME_ATTRIBUTES",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "SUPPORTED_CAMERA_NAMES",
    "SchemaContractError",
    "decode_enabled_cameras",
    "encode_enabled_cameras",
    "flatten_mapping",
    "image_paths",
    "is_extension_path",
    "load_declarative_schema",
    "normalize_enabled_cameras",
    "normalize_record_paths",
    "required_observation_paths",
    "validate_declarative_schema",
]
