# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from pathlib import Path

import pytest

from sparr_phase1_collection.checkpoint import (
    PHASE1_OBSERVATION_ORDER,
    build_phase1_policy_spec,
    load_agent_env_configs,
    resolve_config_paths,
)


def _example_run() -> Path:
    workspace = Path(__file__).resolve().parents[5]
    run = workspace / "taskspace_physx" / "uncalib_none_g4_m950"
    # The policy is not in git and not baked into the image; skip where it is absent.
    if not run.is_dir():
        pytest.skip(f"example policy run not present: {run}")
    return run


def test_loads_real_isaac_yaml_and_extracts_exact_phase1_contract():
    run = _example_run()

    # env.yaml contains both !!python/tuple and !!python/object/apply:slice.
    agent, env = load_agent_env_configs(run)
    assert agent["actor"]["rnn_type"] == "lstm"
    assert isinstance(env["scene"]["robot"]["init_state"]["pos"], tuple)
    assert isinstance(
        env["observations"]["policy"]["eef_pos"]["params"]["asset_cfg"]["joint_ids"],
        slice,
    )

    spec = build_phase1_policy_spec(*load_agent_env_configs(run / "model_950.pt"))
    assert spec.observation_order == PHASE1_OBSERVATION_ORDER
    assert spec.observation_dim == 18
    assert spec.action_dim == 6
    assert spec.eef_offset == pytest.approx((0.0, 0.0, 0.1925))
    assert spec.socket_keypoint_offset == pytest.approx((0.0375, 0.0, 0.0))
    # Canary: fails if the checkpoint's action scales change.
    assert spec.position_scale == pytest.approx((0.002, 0.002, 0.002))
    assert spec.orientation_scale == pytest.approx((0.002, 0.002, 0.002))
    # The checkpoint's trained control rate; running off-rate is a config opt-in.
    assert spec.control_rate_hz == pytest.approx(30.0)
    assert spec.training_action_clip == pytest.approx(1.0)
    assert spec.ee_body_frame == "flange"
    assert spec.quaternion_order == "xyzw"
    assert spec.control_period_s == pytest.approx(1.0 / 30.0)
    assert spec.control_rate_hz == pytest.approx(30.0)
    assert spec.recurrent_type == "lstm"
    assert spec.recurrent_hidden_dim == 256
    assert spec.recurrent_num_layers == 2
    assert spec.distribution_class == "HeteroscedasticGaussianDistribution"


def test_config_path_resolution_accepts_run_params_yaml_and_checkpoint():
    run = _example_run()
    expected = (run / "params" / "agent.yaml", run / "params" / "env.yaml")
    assert resolve_config_paths(run) == expected
    assert resolve_config_paths(run / "params") == expected
    assert resolve_config_paths(run / "params" / "agent.yaml") == expected
    assert resolve_config_paths(run / "model_950.pt") == expected


def _newton_root():
    from pathlib import Path

    root = Path(__file__).resolve().parents[5] / "newton_sparr"
    if not (root / "params" / "env.yaml").is_file():
        import pytest

        pytest.skip("newton_sparr assets not present")
    return root


def test_disabled_observation_terms_are_not_counted():
    """A term set to None is declared-but-disabled and is not in the vector."""
    from sparr_phase1_collection.checkpoint import (
        NEWTON_OBSERVATION_ORDER,
        build_phase1_policy_spec,
        load_yaml_mapping,
    )

    root = _newton_root()
    spec = build_phase1_policy_spec(
        load_yaml_mapping(root / "params" / "agent.yaml"),
        load_yaml_mapping(root / "params" / "env.yaml"),
    )
    assert spec.observation_order == NEWTON_OBSERVATION_ORDER
    assert spec.observation_dim == 18
    for disabled in ("joint_pos", "joint_vel", "socket_quat"):
        assert disabled not in spec.observation_order


def test_newton_observes_the_flange_not_a_tool_keypoint():
    """Its body_offset is (0,0,0); the approved checkpoint's is 0.1925 m."""
    from sparr_phase1_collection.checkpoint import build_phase1_policy_spec, load_yaml_mapping

    root = _newton_root()
    spec = build_phase1_policy_spec(
        load_yaml_mapping(root / "params" / "agent.yaml"),
        load_yaml_mapping(root / "params" / "env.yaml"),
    )
    assert spec.eef_offset == pytest.approx((0.0, 0.0, 0.0))
    assert spec.socket_keypoint_offset == pytest.approx((0.0375, 0.0, 0.0))
    assert spec.training_action_clip == pytest.approx(0.5)
    assert spec.control_rate_hz == pytest.approx(1.0 / 0.03)


def test_a_rotation_only_observation_needs_no_quaternion_convention():
    """Newton's rotations are 6-D, so wxyz-vs-xyzw cannot affect the encoding."""
    from sparr_phase1_collection.checkpoint import build_phase1_policy_spec, load_yaml_mapping

    root = _newton_root()
    spec = build_phase1_policy_spec(
        load_yaml_mapping(root / "params" / "agent.yaml"),
        load_yaml_mapping(root / "params" / "env.yaml"),
    )
    assert spec.quaternion_order == "xyzw"
    assert not any(term.endswith("_quat") for term in spec.observation_order)


def test_the_two_layouts_encode_the_same_data_in_different_orders():
    """Same quantities, different vector: the layouts are not interchangeable."""
    import numpy as np

    from sparr_phase1_collection.observations import (
        NEWTON_OBSERVATION_ORDER,
        OBSERVATION_ORDER as PHASE1_OBSERVATION_ORDER,
        ObservationConfig,
        encode_phase1_observation,
    )
    from sparr_phase1_collection.pose_math import Pose

    flange = Pose(np.array([0.47, 0.12, 0.34]), np.array([0.707, 0.707, 0.0, 0.0]))
    socket = Pose(np.array([0.473, 0.125, 0.0335]), np.array([0.5, 0.5, 0.5, -0.5]))

    # Same offsets on both, so only the order differs.
    common = dict(eef_offset_flange=(0.0, 0.0, 0.0), socket_keypoint_offset=(0.0375, 0.0, 0.0))
    approved = encode_phase1_observation(
        flange, socket,
        ObservationConfig(**common, observation_order=PHASE1_OBSERVATION_ORDER),
    )
    newton = encode_phase1_observation(
        flange, socket,
        ObservationConfig(**common, observation_order=NEWTON_OBSERVATION_ORDER),
    )
    assert approved.shape == newton.shape == (18,)
    assert not np.array_equal(approved, newton), "the orders must produce different vectors"
    # approved = [tool_pos, tool_rot, socket_pos, socket_rot]
    # newton   = [socket_pos, tool_pos, tool_rot, socket_rot]
    assert np.allclose(newton[0:3], approved[9:12])    # socket_pos
    assert np.allclose(newton[3:6], approved[0:3])     # tool_pos
    assert np.allclose(newton[6:12], approved[3:9])    # tool_rot_6d
    assert np.allclose(newton[12:18], approved[12:18])  # socket_rot_6d


def test_an_unknown_observation_order_is_refused_by_the_encoder():
    """A wrong order is silent otherwise: 18 finite numbers, wrong meaning."""
    import numpy as np

    from sparr_phase1_collection.observations import (
        ObservationConfig,
        encode_phase1_observation,
    )
    from sparr_phase1_collection.pose_math import Pose

    pose = Pose(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="observation order must be one of"):
        ObservationConfig(
            eef_offset_flange=(0.0, 0.0, 0.0),
            socket_keypoint_offset=(0.0, 0.0, 0.0),
            observation_order=("joint_pos", "joint_vel"),
        )
    config = ObservationConfig(
        eef_offset_flange=(0.0, 0.0, 0.0),
        socket_keypoint_offset=(0.0, 0.0, 0.0),
    )
    # A valid config still encodes, so the rejection above is the order check
    # and not a broken fixture.
    assert encode_phase1_observation(pose, pose, config).shape == (18,)
