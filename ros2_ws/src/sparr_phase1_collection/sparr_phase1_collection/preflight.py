# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA OneWay Noncommercial License (March 22, 2022).
# See the LICENSE file in the repository root for the full license text:
# https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf

"""Static and ROS-graph preflight for SPARR Phase 1 collection."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from enum import Enum
import importlib.util
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable, Optional

from .configuration import CollectionConfig, ConfigurationError
from .hdf5_schema import SchemaContractError, validate_declarative_schema
from .provenance import (
    artifact_provenance,
    find_repository_root,
    resolve_artifact_path,
    resolve_output_directory,
)
from .trial_state_machine import (
    AXIS_TOLERANCE_SUCCESS_METRICS,
    SUPPORTED_SUCCESS_METRICS,
    TrialConfigurationError,
    validate_axis_tolerances,
)


class CheckLevel(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass(frozen=True)
class CheckResult:
    name: str
    level: CheckLevel
    detail: str


@dataclass(frozen=True)
class PreflightReport:
    mode: str
    config_path: str
    checks: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        return all(check.level is not CheckLevel.FAIL for check in self.checks)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "mode": self.mode,
            "config_path": self.config_path,
            "checks": [
                {"name": item.name, "level": item.level.value, "detail": item.detail}
                for item in self.checks
            ],
        }


_OBSOLETE_KEYS = frozenset({"tcp_z_safety_min", "insertion_success_z"})
_CONTROLLER_COMMAND_TYPE = "flexiv_msgs/msg/CartesianMotionForceCommand"


def _result(name: str, condition: bool, good: str, bad: str) -> CheckResult:
    return CheckResult(
        name,
        CheckLevel.PASS if condition else CheckLevel.FAIL,
        good if condition else bad,
    )


def _subscriber_type_check(
    topic: str,
    endpoints: Iterable[object],
    *,
    expected_type: str,
) -> CheckResult:
    """Require at least one subscriber endpoint with the exact ROS interface type."""

    endpoint_types = [str(getattr(endpoint, "topic_type", "")) for endpoint in endpoints]
    matching_count = sum(topic_type == expected_type for topic_type in endpoint_types)
    reported_types = sorted({topic_type for topic_type in endpoint_types if topic_type})
    return _result(
        f"ros_subscriber:{topic}",
        matching_count > 0,
        f"{matching_count} subscriber(s) using {expected_type}",
        f"expected {expected_type}; graph reports {reported_types or 'no subscribers'}",
    )


def _module_checks(names: Iterable[str]) -> list[CheckResult]:
    checks = []
    for name in names:
        available = importlib.util.find_spec(name) is not None
        checks.append(
            _result(
                f"python_module:{name}",
                available,
                "available",
                "not importable in this environment",
            )
        )
    return checks


def _artifact_checks(
    config: CollectionConfig,
    config_path: Path,
) -> tuple[list[CheckResult], dict[str, Path]]:
    configured = {
        "checkpoint": config.policy.checkpoint,
        "agent_yaml": config.policy.agent_yaml or "",
        "env_yaml": config.policy.env_yaml or "",
    }
    checks: list[CheckResult] = []
    resolved: dict[str, Path] = {}
    for name, value in configured.items():
        if not value:
            checks.append(CheckResult(name, CheckLevel.FAIL, "path is not configured"))
            continue
        path = resolve_artifact_path(value, config_path=config_path)
        resolved[name] = path
        if not path.is_file():
            checks.append(CheckResult(name, CheckLevel.FAIL, f"missing file: {path}"))
            continue
        provenance = artifact_provenance(path)
        checks.append(
            CheckResult(
                name,
                CheckLevel.PASS,
                f"{path} ({provenance.size_bytes} bytes, sha256={provenance.sha256})",
            )
        )
    return checks, resolved


def _policy_contract_check(
    resolved: dict[str, Path],
    configured_control_rate_hz: float,
    control_rate_mismatch_reason: str = "",
) -> CheckResult:
    if not {"checkpoint", "agent_yaml", "env_yaml"}.issubset(resolved):
        return CheckResult("checkpoint_contract", CheckLevel.FAIL, "required artifact is missing")
    try:
        from .checkpoint import (
            build_phase1_policy_spec,
            load_yaml_mapping,
        )

        spec = build_phase1_policy_spec(
            load_yaml_mapping(resolved["agent_yaml"]),
            load_yaml_mapping(resolved["env_yaml"]),
        )
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        return CheckResult("checkpoint_contract", CheckLevel.FAIL, str(exc))

    observation_size = spec.observation_dim
    action_size = spec.action_dim
    detail = f"observation_size={observation_size}, action_size={action_size}, spec={spec!r}"
    rate_matches = abs(spec.control_rate_hz - configured_control_rate_hz) <= 1.0e-6
    reason = str(control_rate_mismatch_reason or "").strip()

    if observation_size != 18 or action_size != 6:
        return CheckResult("checkpoint_contract", CheckLevel.FAIL, detail)

    if not rate_matches:
        mismatch = (
            f"configured_control_rate_hz={configured_control_rate_hz} does not match "
            f"checkpoint={spec.control_rate_hz} "
            f"(from env.sim.dt * env.decimation)"
        )
        if not reason:
            return CheckResult(
                "checkpoint_contract",
                CheckLevel.FAIL,
                f"{detail}, {mismatch}; set policy.control_rate_mismatch_reason "
                "to run deliberately off the trained rate",
            )
        # Downgraded, never silenced: a recurrent policy off its trained rate
        # behaves differently in ways action rescaling cannot fix.
        return CheckResult(
            "checkpoint_contract",
            CheckLevel.WARNING,
            f"{detail}, {mismatch}; RUNNING OFF THE TRAINED RATE by operator "
            f"opt-in: {reason!r}",
        )

    return CheckResult("checkpoint_contract", CheckLevel.PASS, detail)


def _policy_runtime_smoke_check(
    config: CollectionConfig,
    resolved: dict[str, Path],
    *,
    loader: Any = None,
) -> CheckResult:
    """Load and sample the recurrent policy without touching ROS."""

    required = ("checkpoint", "agent_yaml", "env_yaml")
    if not all(name in resolved and resolved[name].is_file() for name in required):
        return CheckResult(
            "policy_runtime_smoke",
            CheckLevel.FAIL,
            "required policy artifact is missing",
        )
    try:
        import numpy as np

        if loader is None:
            from .rsl_rl_runtime import load_rsl_rl_policy

            loader = load_rsl_rl_policy
        runtime = loader(
            resolved["checkpoint"],
            agent_yaml=resolved["agent_yaml"],
            env_yaml=resolved["env_yaml"],
            device=config.policy.device,
            seed=config.policy.torch_seed,
        )
        runtime.reset()
        first = runtime.sample_once(np.zeros(runtime.spec.observation_dim, dtype=np.float32))
        second = runtime.sample_once(np.zeros(runtime.spec.observation_dim, dtype=np.float32))
        expected_state_size = (
            2
            * runtime.spec.recurrent_num_layers
            * runtime.spec.recurrent_hidden_dim
        )
        vector_fields = {
            "raw_mean": first.raw_mean,
            "raw_std": first.raw_std,
            "raw_sample": first.raw_sample,
        }
        if any(
            np.asarray(value).shape != (runtime.spec.action_dim,)
            for value in vector_fields.values()
        ):
            raise ValueError(
                "policy sample shapes are not all "
                f"({runtime.spec.action_dim},): "
                + ", ".join(
                    f"{name}={np.asarray(value).shape}"
                    for name, value in vector_fields.items()
                )
            )
        if not np.all(np.isfinite(first.raw_mean)):
            raise ValueError("policy raw mean is non-finite")
        if not np.all(np.isfinite(first.raw_sample)):
            raise ValueError("policy raw sample is non-finite")
        if not np.all(np.isfinite(first.raw_std)) or not np.all(first.raw_std > 0.0):
            raise ValueError("policy raw standard deviation is not positive and finite")
        if first.recurrent_state_in.shape != (expected_state_size,):
            raise ValueError(
                f"first recurrent input shape is {first.recurrent_state_in.shape}; "
                f"expected ({expected_state_size},)"
            )
        if not np.array_equal(
            first.recurrent_state_in,
            np.zeros(expected_state_size, dtype=np.float32),
        ):
            raise ValueError("first post-reset recurrent input is not all zeros")
        if not np.array_equal(second.recurrent_state_in, first.recurrent_state_out):
            raise ValueError("recurrent state continuity failed between two samples")
    except Exception as exc:
        return CheckResult("policy_runtime_smoke", CheckLevel.FAIL, str(exc))
    return CheckResult(
        "policy_runtime_smoke",
        CheckLevel.PASS,
        (
            f"loaded on {config.policy.device}; two stochastic samples passed; "
            f"recurrent_state_size={expected_state_size}"
        ),
    )


def _schema_contract_check() -> CheckResult:
    candidates = [
        Path(__file__).resolve().parents[1] / "schema" / "sparr_phase1_v1.yaml",
    ]
    try:
        from ament_index_python.packages import get_package_share_directory

        candidates.append(
            Path(get_package_share_directory("sparr_phase1_collection"))
            / "schema"
            / "sparr_phase1_v1.yaml"
        )
    except (ImportError, LookupError):
        pass
    schema_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if schema_path is None:
        return CheckResult(
            "hdf5_schema_contract",
            CheckLevel.FAIL,
            f"schema file not found; checked {[str(path) for path in candidates]}",
        )
    try:
        validate_declarative_schema(schema_path)
    except (OSError, SchemaContractError, ValueError) as exc:
        return CheckResult("hdf5_schema_contract", CheckLevel.FAIL, str(exc))
    return CheckResult(
        "hdf5_schema_contract",
        CheckLevel.PASS,
        f"Python and declarative schema agree: {schema_path}",
    )


def _camera_extrinsics_checks(
    config: CollectionConfig,
    config_path: Path,
    mode: str,
) -> list[CheckResult]:
    """Report optional extrinsics provenance without requiring calibration.

    Phase 1 never projects camera data into a robot frame, so a missing
    flange-to-camera transform is not a failure.
    """

    checks: list[CheckResult] = []
    for camera in config.enabled_cameras:
        configured = camera.extrinsics_file
        value = "" if configured is None else str(configured).strip()
        name = f"camera:{camera.name}:extrinsics"
        if configured is None:
            checks.append(
                CheckResult(
                    name,
                    CheckLevel.PASS,
                    "not provided (optional for Phase 1)",
                )
            )
            continue
        if not value or value.upper().startswith("TODO"):
            level = CheckLevel.FAIL if mode == "live" else CheckLevel.WARNING
            checks.append(
                CheckResult(
                    name,
                    level,
                    "placeholder is ambiguous; use null for intentionally absent extrinsics",
                )
            )
            continue
        path = resolve_artifact_path(value, config_path=config_path)
        if not path.is_file():
            level = CheckLevel.FAIL if mode == "live" else CheckLevel.WARNING
            checks.append(
                CheckResult(
                    name,
                    level,
                    f"optional configured file is missing: {path}",
                )
            )
            continue
        provenance = artifact_provenance(path)
        checks.append(
            CheckResult(
                name,
                CheckLevel.PASS,
                f"{path} ({provenance.size_bytes} bytes, sha256={provenance.sha256})",
            )
        )
    return checks


def _config_checks(config: CollectionConfig, mode: str, raw_text: str) -> list[CheckResult]:
    checks: list[CheckResult] = []
    enabled_cameras = config.enabled_cameras
    try:
        config.validate(live=mode == "live")
    except ConfigurationError as exc:
        checks.append(CheckResult("configuration", CheckLevel.FAIL, str(exc)))
    else:
        checks.append(CheckResult("configuration", CheckLevel.PASS, f"valid for {mode} mode"))

    metric = config.success.metric
    threshold_m = config.success.distance_threshold_m
    axis_tolerances_m = config.success.axis_tolerances_m
    if metric not in SUPPORTED_SUCCESS_METRICS:
        success_level = CheckLevel.FAIL
        success_detail = (
            f"unsupported success.metric {metric!r}; expected one of "
            f"{', '.join(SUPPORTED_SUCCESS_METRICS)}"
        )
    elif metric in AXIS_TOLERANCE_SUCCESS_METRICS:
        # World-axis box; correct only for an axis-vertical socket mount.  See
        # the frame caveat in trial_state_machine.evaluate_success.
        try:
            tolerances = validate_axis_tolerances(
                axis_tolerances_m, "success.axis_tolerances_m"
            )
        except TrialConfigurationError as exc:
            success_level = CheckLevel.FAIL
            success_detail = f"{metric} requires a valid axis_tolerances_m: {exc}"
        else:
            success_level = CheckLevel.PASS
            success_detail = (
                f"{metric} with axis_tolerances_m (world x, y, z) = "
                f"({tolerances[0]:g}, {tolerances[1]:g}, {tolerances[2]:g}) m"
            )
    else:
        success_level = CheckLevel.PASS
        success_detail = f"{metric} with distance_threshold_m={threshold_m}"
    checks.append(CheckResult("success_metric", success_level, success_detail))

    if config.policy.deterministic_inference:
        checks.append(
            CheckResult(
                "policy_inference_mode",
                CheckLevel.WARNING,
                "deterministic (policy.stochastic=false): the clipped distribution "
                "mean drives the action, the SPARR residual label collapses to "
                "~zero, and every trial is recorded as debug-only and NOT "
                "export-eligible; set policy.stochastic=true to collect training data",
            )
        )
    else:
        checks.append(
            CheckResult(
                "policy_inference_mode",
                CheckLevel.PASS,
                "stochastic sampling (policy.stochastic=true); trials are export-eligible",
            )
        )

    found_obsolete = sorted(key for key in _OBSOLETE_KEYS if key in raw_text)
    checks.append(
        _result(
            "obsolete_z_thresholds_absent",
            not found_obsolete,
            "no obsolete TCP-Z success/safety parameters",
            f"remove obsolete keys: {', '.join(found_obsolete)}",
        )
    )
    # The Phase 2 encoder defaults to 256x256 (learner image_size). Other sizes
    # are allowed but warn: they must match Phase 2's image_size and cannot be
    # mixed with other sizes in one training set.
    stored_sizes = sorted(
        {(camera.output_width, camera.output_height) for camera in enabled_cameras}
    )
    if stored_sizes == [(256, 256)]:
        checks.append(
            _result("camera_output_shape", True, "all enabled stored RGB crops are 256x256", "")
        )
    else:
        described = ", ".join(f"{w}x{h}" for w, h in stored_sizes)
        checks.append(
            CheckResult(
                "camera_output_shape",
                CheckLevel.WARNING,
                f"stored RGB crops are {described}, not 256x256: Phase 2 defaults to "
                "256x256 images, so set its image_size to match, and do not mix "
                "sizes in one training set",
            )
        )
    for camera in enabled_cameras:
        if (
            camera.crop_xywh is None
            or camera.input_width is None
            or camera.input_height is None
        ):
            level = CheckLevel.FAIL if mode == "live" else CheckLevel.WARNING
            checks.append(
                CheckResult(
                    f"camera:{camera.name}:crop",
                    level,
                    "crop/input size is TODO",
                )
            )
            continue
        x, y, width, height = camera.crop_xywh
        valid = x >= 0 and y >= 0 and width > 0 and height > 0
        valid = valid and x + width <= camera.input_width and y + height <= camera.input_height
        checks.append(
            _result(
                f"camera:{camera.name}:crop",
                valid,
                "crop lies inside the configured input frame",
                f"invalid crop {camera.crop_xywh} for {camera.input_width}x{camera.input_height}",
            )
        )
    return checks


def _ros_graph_checks(config: CollectionConfig, timeout_s: float) -> list[CheckResult]:
    """Wait for required publishers and the controller command subscriber."""

    try:
        import rclpy
    except ImportError as exc:
        return [CheckResult("ros_graph", CheckLevel.FAIL, f"cannot import rclpy: {exc}")]

    required_publishers = {
        config.robot.state_topic: "flexiv_msgs/msg/RobotStates",
    }
    for camera in config.enabled_cameras:
        required_publishers[camera.image_topic] = "sensor_msgs/msg/Image"
        required_publishers[camera.camera_info_topic] = "sensor_msgs/msg/CameraInfo"

    rclpy.init(args=None)
    node = rclpy.create_node("sparr_phase1_preflight")
    deadline = time.monotonic() + timeout_s
    available: dict[str, list[str]] = {}
    controller_subscribers = []
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            available = dict(node.get_topic_names_and_types())
            controller_subscribers = node.get_subscriptions_info_by_topic(
                config.robot.command_topic
            )
            controller_type_ready = any(
                getattr(endpoint, "topic_type", "") == _CONTROLLER_COMMAND_TYPE
                for endpoint in controller_subscribers
            )
            if all(topic in available for topic in required_publishers) and controller_type_ready:
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()

    checks: list[CheckResult] = []
    for topic, expected_type in required_publishers.items():
        actual = available.get(topic, [])
        checks.append(
            _result(
                f"ros_publisher:{topic}",
                expected_type in actual,
                f"published as {expected_type}",
                f"expected {expected_type}; graph reports {actual or 'no publisher'}",
            )
        )
    checks.append(
        _subscriber_type_check(
            config.robot.command_topic,
            controller_subscribers,
            expected_type=_CONTROLLER_COMMAND_TYPE,
        )
    )
    return checks


def run_preflight(
    config_path: str | Path,
    *,
    mode: Optional[str] = None,
    wait_for_ros: bool = False,
    timeout_s: float = 10.0,
) -> PreflightReport:
    """Run all non-mutating preflight checks and return a structured report."""

    path = Path(config_path).expanduser().resolve()
    checks: list[CheckResult] = []
    try:
        raw_text = path.read_text(encoding="utf-8")
        config = CollectionConfig.load(path)
    except (OSError, ConfigurationError) as exc:
        return PreflightReport(
            mode=mode or "unknown",
            config_path=str(path),
            checks=(CheckResult("configuration", CheckLevel.FAIL, str(exc)),),
        )

    selected_mode = mode or config.runtime.execution_mode
    config = replace(config, runtime=replace(config.runtime, execution_mode=selected_mode))
    checks.extend(_config_checks(config, selected_mode, raw_text))
    module_names = ["yaml", "numpy", "h5py", "cv2", "torch", "rsl_rl"]
    checks.extend(_module_checks(module_names))
    checks.append(_schema_contract_check())
    checks.extend(_camera_extrinsics_checks(config, path, selected_mode))
    artifact_checks, resolved = _artifact_checks(config, path)
    checks.extend(artifact_checks)
    checks.append(
        _policy_contract_check(
            resolved,
            config.policy.control_rate_hz,
            config.policy.control_rate_mismatch_reason,
        )
    )
    checks.append(_policy_runtime_smoke_check(config, resolved))

    # The writer creates missing folders, so the nearest existing ancestor must
    # be a writable directory.
    output = resolve_output_directory(
        config.recording.output_directory, find_repository_root(path)
    )
    existing = output
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    checks.append(
        _result(
            "recording_output_parent",
            existing.is_dir() and os.access(existing, os.W_OK),
            f"{output} (creating under {existing})" if existing != output else str(output),
            f"cannot create {output}: nearest existing parent {existing} is not "
            "a writable directory",
        )
    )

    if wait_for_ros:
        checks.extend(_ros_graph_checks(config, timeout_s))
    elif selected_mode == "live":
        checks.append(
            CheckResult(
                "ros_graph",
                CheckLevel.WARNING,
                "not checked; rerun with --wait-for-ros before motion",
            )
        )
    return PreflightReport(selected_mode, str(path), tuple(checks))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Phase 1 YAML configuration")
    parser.add_argument("--mode", choices=("shadow", "live"))
    parser.add_argument("--wait-for-ros", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_preflight(
        args.config,
        mode=args.mode,
        wait_for_ros=args.wait_for_ros,
        timeout_s=args.timeout_s,
    )
    if args.as_json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        for check in report.checks:
            print(f"[{check.level.value:7}] {check.name}: {check.detail}")
        print("PASS" if report.ok else "FAIL")
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
