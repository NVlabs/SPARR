from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sparr_phase1_collection.configuration import (
    CameraConfig,
    CollectionConfig,
    ConfigurationError,
)
from sparr_phase1_collection.preflight import (
    CheckLevel,
    _camera_extrinsics_checks,
    _config_checks,
    _policy_runtime_smoke_check,
    _subscriber_type_check,
    run_preflight,
)


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "phase1_config.yaml"


def test_controller_subscriber_requires_exact_flexiv_command_type():
    topic = "/cartesian_motion_controller/command"
    expected = "flexiv_msgs/msg/CartesianMotionForceCommand"

    wrong = _subscriber_type_check(
        topic,
        [SimpleNamespace(topic_type="geometry_msgs/msg/PoseStamped")],
        expected_type=expected,
    )
    assert wrong.level is CheckLevel.FAIL
    assert expected in wrong.detail
    assert "geometry_msgs/msg/PoseStamped" in wrong.detail

    mixed = _subscriber_type_check(
        topic,
        [
            SimpleNamespace(topic_type="geometry_msgs/msg/PoseStamped"),
            SimpleNamespace(topic_type=expected),
        ],
        expected_type=expected,
    )
    assert mixed.level is CheckLevel.PASS
    assert "1 subscriber(s)" in mixed.detail


def test_packaged_default_config_keeps_the_flange_success_metric():
    for config in (CollectionConfig.load(FIXTURE),):
        assert config.success.metric == "flange_to_socket_root_position_l2"


@pytest.mark.parametrize(
    "metric",
    (
        "flange_to_socket_root_position_l2",
        "eef_keypoint_to_socket_keypoint_position_l2",
    ),
)
def test_both_success_metrics_are_accepted(metric):
    config = CollectionConfig.load(FIXTURE)
    changed = replace(config, success=replace(config.success, metric=metric))

    changed.validate(live=False)
    assert changed.success.metric == metric


def test_unknown_success_metric_is_rejected():
    config = CollectionConfig.load(FIXTURE)
    changed = replace(
        config,
        success=replace(config.success, metric="tcp_z_below_socket"),
    )

    with pytest.raises(ConfigurationError, match="unsupported success metric"):
        changed.validate(live=False)


AXIS_TOLERANCE_METRIC = "eef_keypoint_to_socket_keypoint_axis_tolerances"
MEASURED_TOLERANCES = (0.005, 0.005, 0.045)


def _axis_tolerance_config(*, live: bool = False, **success_overrides):
    config = CollectionConfig.load(FIXTURE)
    overrides = {
        "metric": AXIS_TOLERANCE_METRIC,
        "distance_threshold_m": None,
        "axis_tolerances_m": MEASURED_TOLERANCES,
        **success_overrides,
    }
    success = replace(config.success, **overrides)
    if not live:
        return replace(config, success=success)
    return replace(
        config,
        success=success,
        robot=replace(
            config.robot,
            state_topic="/Rizon4s_063459/flexiv_robot_states",
            command_topic="/cartesian_motion_controller/command",
        ),
        runtime=replace(
            config.runtime, execution_mode="live", goal_pose_topic="/goal_pose"
        ),
    )


@pytest.mark.parametrize("live", (False, True))
def test_axis_tolerance_metric_does_not_require_distance_threshold(live):
    config = _axis_tolerance_config(live=live)

    config.validate(live=live)
    assert config.success.axis_tolerances_m == MEASURED_TOLERANCES
    assert config.success.distance_threshold_m is None


def test_axis_tolerance_metric_requires_axis_tolerances():
    config = _axis_tolerance_config(axis_tolerances_m=None)

    with pytest.raises(ConfigurationError, match="axis_tolerances_m is required"):
        config.validate(live=False)


def test_axis_tolerances_and_distance_threshold_are_mutually_exclusive():
    both = _axis_tolerance_config(distance_threshold_m=0.01)
    with pytest.raises(
        ConfigurationError, match="distance_threshold_m is meaningless"
    ):
        both.validate(live=False)

    config = CollectionConfig.load(FIXTURE)
    scalar = replace(
        config,
        success=replace(config.success, axis_tolerances_m=MEASURED_TOLERANCES),
    )
    with pytest.raises(ConfigurationError, match="axis_tolerances_m is meaningless"):
        scalar.validate(live=False)


@pytest.mark.parametrize(
    "tolerances",
    ((0.005, 0.005), (0.005, 0.005, 0.045, 0.1), (0.005, 0.0, 0.045),
     (0.005, -0.005, 0.045), (0.005, 0.005, float("inf"))),
)
def test_malformed_axis_tolerances_are_rejected(tolerances):
    config = _axis_tolerance_config(axis_tolerances_m=tolerances)

    with pytest.raises(ConfigurationError, match="axis_tolerances_m"):
        config.validate(live=False)


def test_malformed_axis_tolerances_are_rejected_while_loading_yaml(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        f"success:\n  metric: {AXIS_TOLERANCE_METRIC}\n"
        "  axis_tolerances_m: 0.005\n",
        encoding="utf-8",
    )

    report = run_preflight(path, mode="shadow")

    assert not report.ok
    assert report.checks[0].name == "configuration"
    assert "axis_tolerances_m" in report.checks[0].detail


def test_preflight_reports_axis_tolerances_and_fails_without_them():
    checks = {
        check.name: check
        for check in _config_checks(_axis_tolerance_config(), "shadow", "")
    }
    assert checks["success_metric"].level is CheckLevel.PASS
    assert AXIS_TOLERANCE_METRIC in checks["success_metric"].detail
    assert "0.005, 0.005, 0.045" in checks["success_metric"].detail

    missing = {
        check.name: check
        for check in _config_checks(
            _axis_tolerance_config(axis_tolerances_m=None), "shadow", ""
        )
    }
    assert missing["success_metric"].level is CheckLevel.FAIL
    assert "axis_tolerances_m" in missing["success_metric"].detail


def test_packaged_default_configs_keep_scalar_success_thresholds():
    for config in (CollectionConfig.load(FIXTURE),):
        assert config.success.axis_tolerances_m is None
        assert config.success.distance_threshold_m == 0.01


def test_packaged_default_configs_stay_stochastic():
    for config in (CollectionConfig.load(FIXTURE),):
        assert config.policy.stochastic is True
        assert config.policy.deterministic_inference is False


@pytest.mark.parametrize("live", (False, True))
def test_deterministic_inference_is_accepted_by_validation(live):
    config = CollectionConfig.load(FIXTURE)
    changed = replace(config, policy=replace(config.policy, stochastic=False))

    changed.validate(live=live)
    assert changed.policy.deterministic_inference is True
    # The value stays in the HDF5 provenance snapshot.
    assert changed.resolved_mapping()["policy"]["stochastic"] is False


def test_preflight_warns_but_does_not_fail_on_deterministic_inference():
    config = CollectionConfig.load(FIXTURE)

    stochastic_checks = {
        check.name: check for check in _config_checks(config, "shadow", "")
    }
    assert stochastic_checks["policy_inference_mode"].level is CheckLevel.PASS

    deterministic = replace(config, policy=replace(config.policy, stochastic=False))
    checks = {check.name: check for check in _config_checks(deterministic, "live", "")}
    mode_check = checks["policy_inference_mode"]

    assert mode_check.level is CheckLevel.WARNING
    assert "deterministic" in mode_check.detail
    assert "NOT export-eligible" in mode_check.detail
    assert checks["configuration"].level is CheckLevel.PASS


def test_camera_string_fields_are_normalized_from_yaml_scalars():
    config = CollectionConfig.from_mapping(
        {
            "cameras": {
                "wrist_1": {
                    "image_topic": " /camera/image_raw ",
                    "serial": 335122270518,
                }
            }
        }
    )

    wrist_1 = config.cameras[0]
    assert wrist_1.image_topic == "/camera/image_raw"
    assert wrist_1.serial == "335122270518"


def test_wrist_2_can_be_explicitly_disabled_without_resolving_its_fields():
    config = CollectionConfig.from_mapping(
        {"cameras": {"wrist_2": {"enabled": False}}}
    )

    assert tuple(camera.name for camera in config.enabled_cameras) == ("wrist_1",)
    assert config.cameras[1].has_placeholders is False


def test_wrist_1_cannot_be_disabled():
    with pytest.raises(ConfigurationError, match="wrist_1.*must be enabled"):
        CollectionConfig.from_mapping(
            {"cameras": {"wrist_1": {"enabled": False}}}
        )


def test_camera_enabled_must_be_a_yaml_boolean():
    with pytest.raises(ConfigurationError, match="enabled must be a boolean"):
        CollectionConfig.from_mapping(
            {"cameras": {"wrist_2": {"enabled": "false"}}}
        )


@pytest.mark.parametrize("camera_mapping", ({}, {"extrinsics_file": None}))
def test_camera_extrinsics_may_be_omitted_explicitly_or_implicitly(camera_mapping):
    config = CollectionConfig.from_mapping(
        {"cameras": {"wrist_1": camera_mapping}}
    )

    assert config.cameras[0].extrinsics_file is None


def test_optional_extrinsics_do_not_make_a_resolved_camera_a_placeholder():
    config = CollectionConfig.load(
        FIXTURE
    )
    without_extrinsics = replace(config.cameras[0], extrinsics_file=None)
    unresolved_extrinsics = replace(config.cameras[0], extrinsics_file="TODO")

    assert not without_extrinsics.has_placeholders
    assert unresolved_extrinsics.has_placeholders


def test_live_configuration_accepts_intentionally_absent_extrinsics():
    config = CollectionConfig.load(
        FIXTURE
    )
    changed = replace(
        config,
        robot=replace(
            config.robot,
            state_topic="/Rizon4s_063459/flexiv_robot_states",
            command_topic="/cartesian_motion_controller/command",
        ),
        cameras=tuple(
            replace(camera, extrinsics_file=None) for camera in config.cameras
        ),
        runtime=replace(
            config.runtime,
            execution_mode="live",
            goal_pose_topic="/goal_pose",
        ),
    )

    changed.validate(live=True)


def test_live_single_camera_configuration_does_not_require_pair_skew():
    config = CollectionConfig.load(
        FIXTURE
    )
    changed = replace(
        config,
        robot=replace(
            config.robot,
            state_topic="/Rizon4s_063459/flexiv_robot_states",
            command_topic="/cartesian_motion_controller/command",
        ),
        cameras=(config.cameras[0], replace(config.cameras[1], enabled=False)),
        synchronization=replace(
            config.synchronization,
            max_camera_pair_skew_ms=None,
        ),
        runtime=replace(
            config.runtime,
            execution_mode="live",
            goal_pose_topic="/goal_pose",
        ),
    )

    changed.validate(live=True)


def test_live_preflight_checks_torch_and_rsl_rl_runtime_dependencies():
    # Uses the frozen fixture so the live YAML cannot affect this test.
    report = run_preflight(
        FIXTURE,
        mode="live",
    )
    check_names = {check.name for check in report.checks}

    assert "python_module:torch" in check_names
    assert "python_module:rsl_rl" in check_names


def test_policy_runtime_smoke_checks_reset_shapes_and_state_continuity(
    tmp_path: Path,
):
    resolved = {}
    for name in ("checkpoint", "agent_yaml", "env_yaml"):
        path = tmp_path / name
        path.touch()
        resolved[name] = path

    class FakeRuntime:
        spec = SimpleNamespace(
            observation_dim=18,
            action_dim=6,
            recurrent_num_layers=2,
            recurrent_hidden_dim=2,
        )

        def __init__(self):
            self.state = np.zeros(8, dtype=np.float32)

        def reset(self):
            self.state.fill(0.0)

        def sample_once(self, observation):
            assert observation.shape == (18,)
            state_in = self.state.copy()
            self.state += 1.0
            return SimpleNamespace(
                raw_mean=np.zeros(6, dtype=np.float32),
                raw_std=np.ones(6, dtype=np.float32),
                raw_sample=np.full(6, 0.5, dtype=np.float32),
                recurrent_state_in=state_in,
                recurrent_state_out=self.state.copy(),
            )

    config = CollectionConfig()
    result = _policy_runtime_smoke_check(
        config,
        resolved,
        loader=lambda *args, **kwargs: FakeRuntime(),
    )

    assert result.level is CheckLevel.PASS
    assert "recurrent_state_size=8" in result.detail


def test_socket_opening_is_parameterized_and_validated():
    config = CollectionConfig()
    changed = replace(
        config,
        socket=replace(
            config.socket,
            opening_pose_world_xyzw=(0.476, 0.128, 0.07, 0.5, 0.5, 0.5, -0.5),
        ),
    )
    changed.validate(live=False)

    invalid = replace(
        config,
        socket=replace(
            config.socket,
            opening_pose_world_xyzw=(0.473, 0.121, 0.071, 0.0, 0.0, 0.0, 0.0),
        ),
    )
    with pytest.raises(ConfigurationError, match="zero quaternion"):
        invalid.validate(live=False)


def test_camera_empty_string_counts_as_live_placeholder():
    config = CollectionConfig()
    camera = replace(config.cameras[0], image_topic="")
    changed = replace(config, cameras=(camera, config.cameras[1]))

    with pytest.raises(ConfigurationError, match="still contains TODO"):
        changed.validate(live=True)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("encoding", "mono8", "encoding must be 'rgb8' or 'bgr8'"),
        ("qos", "unknown", "qos must be sensor_data, best_effort, or reliable"),
    ),
)
def test_camera_encoding_and_qos_are_explicit_enumerations(field, value, message):
    config = CollectionConfig.load(
        FIXTURE
    )
    camera = replace(config.cameras[0], **{field: value})
    changed = replace(config, cameras=(camera, config.cameras[1]))

    with pytest.raises(ConfigurationError, match=message):
        changed.validate(live=False)


def test_live_preflight_accepts_intentionally_absent_camera_extrinsics(
    tmp_path: Path,
):
    config = CollectionConfig(
        cameras=(
            CameraConfig(name="wrist_1", extrinsics_file=None),
            CameraConfig(name="wrist_2", extrinsics_file=None),
        )
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")

    checks = _camera_extrinsics_checks(config, config_path, "live")

    assert [check.level for check in checks] == [CheckLevel.PASS, CheckLevel.PASS]
    assert all("optional for Phase 1" in check.detail for check in checks)


def test_preflight_omits_disabled_camera_extrinsics_checks(tmp_path: Path):
    config = CollectionConfig(
        cameras=(
            CameraConfig(name="wrist_1", extrinsics_file=None),
            CameraConfig(
                name="wrist_2",
                enabled=False,
                extrinsics_file="TODO_DISABLED_CAMERA_MUST_NOT_BE_CHECKED",
            ),
        )
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")

    checks = _camera_extrinsics_checks(config, config_path, "live")

    assert [check.name for check in checks] == ["camera:wrist_1:extrinsics"]
    assert checks[0].level is CheckLevel.PASS


def test_live_preflight_rejects_missing_configured_extrinsics_files(tmp_path: Path):
    config = CollectionConfig(
        cameras=(
            CameraConfig(name="wrist_1", extrinsics_file="cal/wrist_1.yaml"),
            CameraConfig(name="wrist_2", extrinsics_file="cal/wrist_2.yaml"),
        )
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")

    checks = _camera_extrinsics_checks(config, config_path, "live")

    assert [check.level for check in checks] == [CheckLevel.FAIL, CheckLevel.FAIL]
    assert all("is missing" in check.detail for check in checks)


def test_live_preflight_rejects_ambiguous_extrinsics_placeholder(tmp_path: Path):
    config = CollectionConfig(
        cameras=(
            CameraConfig(name="wrist_1", extrinsics_file="TODO"),
            CameraConfig(name="wrist_2", extrinsics_file=""),
        )
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")

    checks = _camera_extrinsics_checks(config, config_path, "live")

    assert [check.level for check in checks] == [CheckLevel.FAIL, CheckLevel.FAIL]
    assert all("use null" in check.detail for check in checks)


def test_extrinsics_are_resolved_relative_to_config(tmp_path: Path):
    calibration_dir = tmp_path / "cal"
    calibration_dir.mkdir()
    for name in ("wrist_1.yaml", "wrist_2.yaml"):
        (calibration_dir / name).write_text("transform: identity\n", encoding="utf-8")
    config = CollectionConfig(
        cameras=(
            CameraConfig(name="wrist_1", extrinsics_file="cal/wrist_1.yaml"),
            CameraConfig(name="wrist_2", extrinsics_file="cal/wrist_2.yaml"),
        )
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")

    checks = _camera_extrinsics_checks(config, config_path, "live")

    assert [check.level for check in checks] == [CheckLevel.PASS, CheckLevel.PASS]
    assert all("sha256=" in check.detail for check in checks)


@pytest.mark.parametrize(
    "text",
    (
        "robot:\n  unknown_typo: true\n",
        "robot:\n  max_state_age_ms: not-a-number\n",
        "robot: [not, a, mapping]\n",
        "robot:\n  serial: one\n  serial: two\n",
    ),
)
def test_malformed_configuration_fails_preflight_without_crashing(
    tmp_path: Path, text: str
):
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")

    report = run_preflight(path, mode="shadow")

    assert not report.ok
    assert len(report.checks) == 1
    assert report.checks[0].name == "configuration"
    assert report.checks[0].level is CheckLevel.FAIL


def _contract(rate_hz, reason=""):
    from pathlib import Path

    from sparr_phase1_collection.preflight import CheckLevel, _policy_contract_check

    root = Path(__file__).resolve().parents[5] / "taskspace_physx/uncalib_none_g4_m950"
    resolved = {
        "checkpoint": root / "model_950.pt",
        "agent_yaml": root / "params/agent.yaml",
        "env_yaml": root / "params/env.yaml",
    }
    if not all(p.is_file() for p in resolved.values()):
        import pytest

        pytest.skip("checkpoint assets not present")
    return _policy_contract_check(resolved, rate_hz, reason), CheckLevel


def test_the_trained_rate_passes_without_any_opt_in():
    result, CheckLevel = _contract(30.0)
    assert result.level is CheckLevel.PASS


def test_an_off_rate_run_fails_without_a_stated_reason():
    """Silently accepting it would let a recurrent policy run at the wrong rate."""
    result, CheckLevel = _contract(15.0)
    assert result.level is CheckLevel.FAIL
    assert "does not match checkpoint" in result.detail
    assert "control_rate_mismatch_reason" in result.detail


def test_a_stated_reason_downgrades_to_a_warning_and_is_quoted():
    result, CheckLevel = _contract(15.0, "deliberate 15 Hz experiment")
    assert result.level is CheckLevel.WARNING
    assert "RUNNING OFF THE TRAINED RATE" in result.detail
    assert "deliberate 15 Hz experiment" in result.detail


def test_whitespace_is_not_a_reason():
    result, CheckLevel = _contract(15.0, "   ")
    assert result.level is CheckLevel.FAIL


def test_a_reason_does_not_excuse_a_wrong_observation_or_action_size():
    """The opt-in covers the rate only; a shape mismatch is still fatal."""
    from pathlib import Path

    from sparr_phase1_collection.preflight import CheckLevel, _policy_contract_check

    result = _policy_contract_check({"checkpoint": Path("/nonexistent")}, 15.0, "any reason")
    assert result.level is CheckLevel.FAIL


def test_the_reason_is_recorded_in_session_metadata():
    """Data collected off-rate must identify itself after the fact."""
    from sparr_phase1_collection.configuration import CollectionConfig, PolicyConfig
    from dataclasses import replace

    config = CollectionConfig()
    config = replace(
        config, policy=replace(config.policy, control_rate_mismatch_reason="because X")
    )
    assert config.resolved_mapping()["policy"]["control_rate_mismatch_reason"] == "because X"


def test_the_collector_node_enforces_the_same_opt_in_as_preflight():
    """The node can be started without preflight, so it must guard too."""
    import inspect

    from sparr_phase1_collection import collector_node

    source = inspect.getsource(collector_node.Phase1CollectorNode.__init__)
    guard = source[source.index("configured policy rate does not match checkpoint")
                   - 900:]
    assert "control_rate_mismatch_reason" in guard, (
        "the node's rate guard must honour the same opt-in as preflight"
    )
    assert "RUNNING OFF THE TRAINED CONTROL RATE" in guard, (
        "an off-rate run must be logged loudly by the node, not silently allowed"
    )


def test_a_timed_out_trial_can_be_relabelled_a_success():
    """The keypoint check cannot tell a seated plug from one on the socket rim."""
    from sparr_phase1_collection.collector_node import Phase1CollectorNode
    from sparr_phase1_collection.trial_state_machine import TrialState

    allowed = Phase1CollectorNode._OPERATOR_SUCCESS_STATES
    assert TrialState.TIMED_OUT in allowed
    assert TrialState.SUCCEEDED in allowed


def test_a_rejected_or_aborted_trial_cannot_be_relabelled_a_success():
    """Those stopped on a guard, so their terminal observation is invalid."""
    from sparr_phase1_collection.collector_node import Phase1CollectorNode
    from sparr_phase1_collection.trial_state_machine import TrialState

    allowed = Phase1CollectorNode._OPERATOR_SUCCESS_STATES
    assert TrialState.REJECTED not in allowed
    assert TrialState.ABORTED not in allowed
    assert TrialState.IDLE not in allowed
    for running in (TrialState.RUNNING, TrialState.HOLDING, TrialState.SETTLING):
        assert running not in allowed


def test_an_override_is_recorded_distinctly_from_a_detected_success():
    """Phase 2 trains on both; they are not the same evidence."""
    import inspect

    from sparr_phase1_collection.collector_node import Phase1CollectorNode

    source = inspect.getsource(Phase1CollectorNode._confirm_success_service)
    assert "operator_override_success_after_timeout" in source
    assert "operator_confirmed_success" in source
    # The override must also be logged.
    assert "warning" in source


def test_phase1_and_phase2_launch_the_same_robot_driver():
    """Both session scripts must launch the same cartesian robot driver.

    Without rizon_type the launch spawns joint-trajectory and gpio controllers,
    so /cartesian_motion_controller/command never exists.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[5] / "docker"
    scripts = {p: (root / p / "session.sh") for p in ("phase1", "phase2")}
    if not all(v.is_file() for v in scripts.values()):
        import pytest

        pytest.skip("session scripts not present")

    launches = {}
    for phase, path in scripts.items():
        text = path.read_text()
        match = re.search(r"ros2 launch flexiv_bringup (\S+)", text)
        assert match, f"{phase}: no flexiv_bringup launch found"
        # The launch arguments span a backslash continuation; scan a window.
        args = text[match.end() : match.end() + 200]
        launches[phase] = (match.group(1), "rizon_type" in args)

    assert launches["phase1"] == launches["phase2"], (
        f"robot launch differs: {launches}"
    )
    assert launches["phase1"][0] == "rizon_cartesian.launch.py"
    assert launches["phase1"][1], "rizon_type must be passed or 'joints' resolves empty"


def test_retraction_clearance_is_optional_and_not_a_placeholder():
    """Unset keeps the default reset path."""
    config = CollectionConfig.from_mapping({})
    assert config.trial_control.retraction_clearance_m is None
    assert "retraction_clearance_m" in config.trial_control.OPTIONAL_FIELDS


def test_retraction_clearance_must_be_positive():
    with pytest.raises(ConfigurationError, match="retraction_clearance_m"):
        CollectionConfig.from_mapping(
            {"trial_control": {"retraction_clearance_m": -0.04}}
        ).validate(live=True)


def _output_check(tmp_path, output_directory):
    import yaml

    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    data = yaml.safe_load(FIXTURE.read_text())
    data["recording"]["output_directory"] = str(output_directory)
    path = repo / "phase1_live.yaml"
    path.write_text(yaml.safe_dump(data))
    report = run_preflight(path, mode="shadow")
    return {check.name: check for check in report.checks}["recording_output_parent"]


def test_a_relative_output_directory_that_does_not_exist_yet_passes(tmp_path):
    # The writer creates it; preflight resolves it against the repository.
    check = _output_check(tmp_path, "data/phase1")
    assert check.level is CheckLevel.PASS
    assert str(tmp_path / "repo" / "data" / "phase1") in check.detail


def test_an_output_directory_that_cannot_be_created_fails(tmp_path):
    # A file where a directory should be; permission tests are unreliable as root.
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")
    check = _output_check(tmp_path, blocker / "phase1")
    assert check.level is CheckLevel.FAIL


def _validation_errors(config):
    try:
        config.validate(live=False)
    except ConfigurationError as exc:
        return str(exc)
    return ""


def test_state_topic_is_derived_from_the_robot_serial():
    config = CollectionConfig.from_mapping({"robot": {"serial": "Rizon4s-012345"}})

    assert config.robot.state_topic == "/Rizon4s_012345/flexiv_robot_states"
    errors = _validation_errors(config)
    assert "robot.serial" not in errors
    assert "state_topic" not in errors


def test_an_explicit_state_topic_is_kept():
    config = CollectionConfig.from_mapping(
        {"robot": {"serial": "Rizon4s-012345", "state_topic": "/custom/states"}}
    )

    assert config.robot.state_topic == "/custom/states"


def test_a_placeholder_robot_serial_is_rejected():
    with pytest.raises(ConfigurationError, match="robot.serial must be set"):
        CollectionConfig.from_mapping({"robot": {"serial": "TODO"}})


def test_calibration_id_is_derived_from_serial_and_stream_profile():
    config = CollectionConfig.from_mapping(
        {
            "cameras": {
                "wrist_1": {
                    "serial": 335122270518,
                    "input_width": 848,
                    "input_height": 480,
                    "encoding": "rgb8",
                }
            }
        }
    )

    assert (
        config.cameras[0].calibration_id
        == "realsense-camera-info-335122270518-848x480-rgb8-v1"
    )


def test_an_explicit_calibration_id_is_kept():
    config = CollectionConfig.from_mapping(
        {"cameras": {"wrist_1": {"serial": 1, "calibration_id": "mine-v2"}}}
    )

    assert config.cameras[0].calibration_id == "mine-v2"


def test_calibration_id_stays_a_placeholder_until_its_inputs_are_set():
    config = CollectionConfig.from_mapping({"cameras": {"wrist_1": {}}})

    assert config.cameras[0].calibration_id == "TODO"


def test_socket_may_be_given_as_an_opening_pose():
    opening = [0.474, 0.127, 0.071, 0.5, 0.5, 0.5, -0.5]
    config = CollectionConfig.from_mapping({"socket": {"opening_pose_world_xyzw": opening}})

    assert config.socket.opening_pose_world_xyzw == tuple(opening)


def test_the_legacy_root_pose_key_is_rejected_with_a_pointer_to_the_new_one():
    with pytest.raises(ConfigurationError, match="opening_pose_world_xyzw"):
        CollectionConfig.from_mapping(
            {"socket": {"root_pose_world_xyzw": [0.47, 0.12, 0.0335, 0.5, 0.5, 0.5, -0.5]}}
        )


def test_opening_pose_converts_to_the_root_by_removing_the_keypoint_offset():
    from sparr_phase1_collection.observations import (
        offset_point_world,
        socket_root_from_opening,
    )
    from sparr_phase1_collection.pose_math import Pose

    opening = Pose.from_vector([0.47625, 0.122, 0.071, 0.5, 0.5, 0.5, -0.5])
    offset = (0.0375, 0.0, 0.0)

    root = socket_root_from_opening(opening, offset)

    # +X in the socket frame is world +Z for this orientation.
    assert root.position == pytest.approx([0.47625, 0.122, 0.0335])
    assert root.quaternion_xyzw == pytest.approx(opening.quaternion_xyzw)
    # The policy's keypoint is back on the opening pose that was entered.
    assert offset_point_world(root, offset) == pytest.approx(opening.position)


def test_action_scale_multiplier_may_differ_from_one_but_must_be_positive():
    base = CollectionConfig.from_mapping({})

    half = replace(base, policy=replace(base.policy, action_scale_multiplier=0.5))
    assert "action_scale_multiplier" not in _validation_errors(half)

    for bad in (0.0, -1.0, float("nan")):
        broken = replace(base, policy=replace(base.policy, action_scale_multiplier=bad))
        assert "action_scale_multiplier" in _validation_errors(broken)
