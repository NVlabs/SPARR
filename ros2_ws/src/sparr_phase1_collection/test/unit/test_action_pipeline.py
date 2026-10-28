# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA License.
# See the LICENSE file in the repository root for the full license text.

import math
from types import SimpleNamespace

import numpy as np

from sparr_phase1_collection.action_pipeline import (
    ActionPipelineConfig,
    build_action_trace,
)
from sparr_phase1_collection.pose_math import (
    Pose,
    apply_world_pose_delta,
    quaternion_to_matrix_xyzw,
    rotvec_to_quaternion_xyzw,
    world_pose_delta,
)


def action_to_flange_target(observed_flange_pose, normalized_action, config):
    """Independent reference: scale an action and apply it to the flange pose."""
    action = np.asarray(normalized_action, dtype=np.float64)
    return apply_world_pose_delta(
        observed_flange_pose,
        action[:3] * config.position_scale_array,
        action[3:] * config.orientation_scale_array,
    )


def inverse_normalized_action(observed_flange_pose, final_flange_pose, config):
    """Independent reference: recover the normalized action from a flange target."""
    translation, rotation_vector = world_pose_delta(observed_flange_pose, final_flange_pose)
    return np.concatenate(
        (
            translation / config.position_scale_array,
            rotation_vector / config.orientation_scale_array,
        )
    )


def test_checkpoint_bound_scale_world_pose_rel_tcp_conversion_and_effective_label():
    config = ActionPipelineConfig(
        position_scale=(0.01, 0.02, 0.03),
        orientation_scale=(0.1, 0.2, 0.3),
        action_clip=0.75,
    )
    observed_flange = Pose(
        np.array([0.5, -0.1, 0.3]),
        rotvec_to_quaternion_xyzw([0.0, 0.0, math.pi / 2.0]),
    )
    flange_to_tcp = Pose(
        np.array([0.0, 0.0, 0.2]), np.array([0.0, 0.0, 0.0, 1.0])
    )
    raw_mean = np.array([0.25, -0.5, 1.2, 0.0, 0.2, -0.4])
    raw_sample = np.array([2.0, -2.0, 0.5, 1.5, -1.5, 0.25])

    trace = build_action_trace(
        raw_mean=raw_mean,
        raw_sample=raw_sample,
        observed_flange_pose=observed_flange,
        flange_to_tcp=flange_to_tcp,
        config=config,
    )

    effective_action = np.array([0.75, -0.75, 0.5, 0.75, -0.75, 0.25])
    np.testing.assert_allclose(trace.raw_mean, raw_mean)
    np.testing.assert_allclose(trace.raw_sample, raw_sample)
    np.testing.assert_allclose(trace.effective_software_action, effective_action)
    np.testing.assert_allclose(
        trace.scaled_translation_world, [0.0075, -0.015, 0.015]
    )
    np.testing.assert_allclose(
        trace.scaled_rotation_vector_world, [0.075, -0.15, 0.075]
    )

    # Translation remains world-relative even though the flange has 90-degree yaw.
    np.testing.assert_allclose(
        trace.effective_flange_target.position,
        observed_flange.position + np.array([0.0075, -0.015, 0.015]),
        atol=1e-12,
    )
    expected_rotation = (
        quaternion_to_matrix_xyzw(
            rotvec_to_quaternion_xyzw(trace.scaled_rotation_vector_world)
        )
        @ quaternion_to_matrix_xyzw(observed_flange.quaternion_xyzw)
    )
    np.testing.assert_allclose(
        quaternion_to_matrix_xyzw(trace.effective_flange_target.quaternion_xyzw),
        expected_rotation,
        atol=1e-12,
    )

    # TCP conversion applies the rigid offset at the effective flange orientation.
    expected_tcp_position = (
        trace.effective_flange_target.position
        + expected_rotation @ np.array([0.0, 0.0, 0.2])
    )
    np.testing.assert_allclose(
        trace.submitted_controller_tcp_target.position, expected_tcp_position
    )

    recovered = inverse_normalized_action(
        observed_flange, trace.effective_flange_target, config
    )
    np.testing.assert_allclose(recovered, effective_action, atol=1e-12)
    # The pseudo-label uses the raw, pre-clamp mean.
    np.testing.assert_array_equal(
        trace.effective_action_minus_raw_mean,
        trace.effective_software_action - trace.raw_mean,
    )


def _deterministic_case():
    config = ActionPipelineConfig(
        position_scale=(0.01, 0.02, 0.03),
        orientation_scale=(0.1, 0.2, 0.3),
        action_clip=0.75,
    )
    observed_flange = Pose(
        np.array([0.5, -0.1, 0.3]),
        rotvec_to_quaternion_xyzw([0.0, 0.0, math.pi / 2.0]),
    )
    flange_to_tcp = Pose(np.array([0.0, 0.0, 0.2]), np.array([0.0, 0.0, 0.0, 1.0]))
    raw_mean = np.array([0.25, -0.5, 1.2, 0.0, 0.2, -0.4])
    raw_sample = np.array([2.0, -2.0, 0.5, 1.5, -1.5, 0.25])
    return config, observed_flange, flange_to_tcp, raw_mean, raw_sample


def test_deterministic_mode_drives_the_pipeline_with_the_clipped_mean():
    config, observed_flange, flange_to_tcp, raw_mean, raw_sample = _deterministic_case()

    stochastic = build_action_trace(
        raw_mean=raw_mean,
        raw_sample=raw_sample,
        observed_flange_pose=observed_flange,
        flange_to_tcp=flange_to_tcp,
        config=config,
    )
    deterministic = build_action_trace(
        raw_mean=raw_mean,
        raw_sample=raw_sample,
        observed_flange_pose=observed_flange,
        flange_to_tcp=flange_to_tcp,
        config=config,
        deterministic=True,
    )

    assert stochastic.deterministic_inference is False
    assert deterministic.deterministic_inference is True

    # Stochastic mode keeps using the clipped sample.
    np.testing.assert_allclose(
        stochastic.effective_software_action, np.clip(raw_sample, -0.75, 0.75)
    )
    # Deterministic mode uses the clipped mean, and the clip is not bypassed.
    np.testing.assert_allclose(
        deterministic.effective_software_action, np.clip(raw_mean, -0.75, 0.75)
    )
    assert deterministic.effective_software_action[2] == 0.75  # mean 1.2 was clipped

    # Both modes still expose the untouched mean, sample, and clipped sample.
    for trace in (stochastic, deterministic):
        np.testing.assert_allclose(trace.raw_mean, raw_mean)
        np.testing.assert_allclose(trace.raw_sample, raw_sample)
        np.testing.assert_allclose(trace.clipped_sample, np.clip(raw_sample, -0.75, 0.75))

    # Everything downstream follows the effective action.
    np.testing.assert_allclose(
        deterministic.scaled_translation_world,
        deterministic.effective_software_action[:3] * np.array([0.01, 0.02, 0.03]),
    )
    np.testing.assert_allclose(
        inverse_normalized_action(
            observed_flange, deterministic.effective_flange_target, config
        ),
        deterministic.effective_software_action,
        atol=1e-12,
    )
    assert not np.allclose(
        deterministic.submitted_controller_tcp_target.position,
        stochastic.submitted_controller_tcp_target.position,
    )


def test_deterministic_residual_label_collapses_to_zero_inside_the_clip():
    config, observed_flange, flange_to_tcp, _, raw_sample = _deterministic_case()
    inside_clip_mean = np.array([0.25, -0.5, 0.1, 0.0, 0.2, -0.4])

    trace = build_action_trace(
        raw_mean=inside_clip_mean,
        raw_sample=raw_sample,
        observed_flange_pose=observed_flange,
        flange_to_tcp=flange_to_tcp,
        config=config,
        deterministic=True,
    )

    # Exactly zero wherever |mu| <= action_clip, so not a trainable residual.
    np.testing.assert_array_equal(
        trace.effective_action_minus_raw_mean, np.zeros(6)
    )


def test_from_policy_spec_uses_yaml_derived_training_action_clip():
    spec = SimpleNamespace(
        position_scale=(0.01, 0.01, 0.01),
        orientation_scale=(0.02, 0.02, 0.02),
        training_action_clip=0.8,
    )

    config = ActionPipelineConfig.from_policy_spec(spec)

    assert config.action_clip == 0.8


def _residual_fixture():
    config = ActionPipelineConfig(
        position_scale=(0.004, 0.004, 0.004),
        orientation_scale=(0.004, 0.004, 0.004),
        action_clip=1.0,
    )
    observed_flange = Pose(
        np.array([0.47, 0.12, 0.30]),
        rotvec_to_quaternion_xyzw([0.0, 0.0, 0.3]),
    )
    flange_to_tcp = Pose(np.array([0.0, 0.0, 0.15]), np.array([0.0, 0.0, 0.0, 1.0]))
    return config, observed_flange, flange_to_tcp


def test_residual_none_is_bit_identical_to_phase_one():
    """Phase 1 must be untouched by the Phase 2 code path."""
    config, flange, flange_to_tcp = _residual_fixture()
    mean = np.array([0.3, -0.4, -1.7, 0.2, 0.1, -0.05])
    sample = np.array([0.35, -0.42, -1.9, 0.25, 0.05, -0.10])

    for deterministic in (False, True):
        without = build_action_trace(
            raw_mean=mean, raw_sample=sample, observed_flange_pose=flange,
            flange_to_tcp=flange_to_tcp, config=config, deterministic=deterministic,
        )
        explicit_none = build_action_trace(
            raw_mean=mean, raw_sample=sample, observed_flange_pose=flange,
            flange_to_tcp=flange_to_tcp, config=config, deterministic=deterministic,
            residual=None,
        )
        for field in (
            "effective_software_action", "scaled_translation_world",
            "scaled_rotation_vector_world", "effective_action_minus_raw_mean",
            "clipped_sample",
        ):
            assert np.array_equal(
                getattr(without, field), getattr(explicit_none, field)
            ), field
        assert np.array_equal(
            without.submitted_controller_tcp_target.as_vector(),
            explicit_none.submitted_controller_tcp_target.as_vector(),
        )
        assert without.requested_residual is None
        assert explicit_none.requested_residual is None


def test_residual_is_added_before_the_clip_not_after():
    """clip(mu + r) != clip(mu) + r whenever the mean is saturated.

    A residual must be able to steer a base action that is already at the bound.
    """
    config, flange, flange_to_tcp = _residual_fixture()
    mean = np.array([2.5, -3.0, 1.8, 0.0, 0.0, 0.0])  # saturated on 3 axes
    residual = np.array([-0.4, 0.6, -0.3, 0.1, -0.2, 0.05])
    sample = np.zeros(6)

    trace = build_action_trace(
        raw_mean=mean, raw_sample=sample, observed_flange_pose=flange,
        flange_to_tcp=flange_to_tcp, config=config, residual=residual,
    )
    expected = np.clip(mean + residual, -config.action_clip, config.action_clip)
    assert np.allclose(trace.effective_software_action, expected)

    # The orderings only differ on saturated axes; check that they differ here.
    wrong = np.clip(mean, -config.action_clip, config.action_clip) + residual
    assert not np.allclose(expected, wrong)


def test_residual_supersedes_deterministic_and_composes_onto_the_mean():
    config, flange, flange_to_tcp = _residual_fixture()
    mean = np.array([0.2, -0.3, 0.4, 0.0, 0.1, -0.1])
    sample = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    residual = np.array([0.05, 0.05, -0.05, 0.0, 0.0, 0.0])
    expected = np.clip(mean + residual, -config.action_clip, config.action_clip)

    for deterministic in (False, True):
        trace = build_action_trace(
            raw_mean=mean, raw_sample=sample, observed_flange_pose=flange,
            flange_to_tcp=flange_to_tcp, config=config,
            deterministic=deterministic, residual=residual,
        )
        # The sample must not leak into the executed action in Phase 2.
        assert np.allclose(trace.effective_software_action, expected)
        assert np.array_equal(trace.requested_residual, residual)


def test_realized_residual_label_still_equals_executed_minus_mean():
    """The SPARR label convention must hold on the Phase 2 path too."""
    config, flange, flange_to_tcp = _residual_fixture()
    mean = np.array([0.9, -0.95, 0.5, 0.0, 0.0, 0.0])
    residual = np.array([0.4, -0.4, 0.1, 0.0, 0.0, 0.0])  # drives 2 axes past the bound

    trace = build_action_trace(
        raw_mean=mean, raw_sample=np.zeros(6), observed_flange_pose=flange,
        flange_to_tcp=flange_to_tcp, config=config, residual=residual,
    )
    assert np.allclose(
        trace.effective_action_minus_raw_mean,
        trace.effective_software_action - trace.raw_mean,
    )
    # Realized, not requested: clipping means they differ once saturated.
    assert not np.allclose(trace.effective_action_minus_raw_mean, residual)


def test_residual_must_be_a_finite_six_vector():
    config, flange, flange_to_tcp = _residual_fixture()
    kwargs = dict(
        raw_mean=np.zeros(6), raw_sample=np.zeros(6), observed_flange_pose=flange,
        flange_to_tcp=flange_to_tcp, config=config,
    )
    for bad in (np.zeros(5), np.zeros((2, 6)), np.array([np.nan] * 6),
                np.array([np.inf, 0, 0, 0, 0, 0])):
        try:
            build_action_trace(residual=bad, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"residual {bad.shape} {bad} should have been rejected")


def test_residual_scaling_and_pose_composition_match_the_effective_action():
    config, flange, flange_to_tcp = _residual_fixture()
    mean = np.array([0.1, 0.2, -0.3, 0.05, -0.05, 0.0])
    residual = np.array([0.2, -0.1, 0.15, 0.0, 0.1, -0.02])

    trace = build_action_trace(
        raw_mean=mean, raw_sample=np.zeros(6), observed_flange_pose=flange,
        flange_to_tcp=flange_to_tcp, config=config, residual=residual,
    )
    effective = trace.effective_software_action
    assert np.allclose(
        trace.scaled_translation_world, effective[:3] * np.asarray(config.position_scale)
    )
    assert np.allclose(
        trace.scaled_rotation_vector_world,
        effective[3:] * np.asarray(config.orientation_scale),
    )
    # pose_rel is composed onto the same-step observed flange pose.
    assert np.array_equal(
        trace.observed_flange_pose.as_vector(), flange.as_vector()
    )
    assert np.allclose(
        trace.effective_flange_target.as_vector(),
        action_to_flange_target(flange, effective, config).as_vector(),
    )
