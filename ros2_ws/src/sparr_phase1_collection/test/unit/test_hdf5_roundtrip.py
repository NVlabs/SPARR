# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA OneWay Noncommercial License (March 22, 2022).
# See the LICENSE file in the repository root for the full license text:
# https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")

from sparr_phase1_collection.hdf5_writer import HDF5SessionWriter  # noqa: E402
from sparr_phase1_collection.validation import (  # noqa: E402
    recover_partial_file,
    validate_file,
)

from recording_fixtures import (  # noqa: E402
    IMAGE_SHAPE,
    TRAINING_ACTION_CLIP,
    make_action,
    make_observation,
    make_single_camera_observation,
    write_complete_trial,
)


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def test_incremental_roundtrip_promotes_clean_file_and_preserves_mapping(tmp_path: Path):
    final_path = tmp_path / "phase1_session.h5"
    writer = HDF5SessionWriter(
        final_path,
        image_shape=IMAGE_SHAPE,
        training_action_clip=TRAINING_ACTION_CLIP,
        flush_every=2,
        metadata={"checkpoint_sha256": "abc123", "random_seeds": {"policy": 7}},
        calibration={
            "flange_to_controller_tcp_xyzw": [0, 0, -0.01, 0, 0, 0, 1],
            "wrist_1": {
                "extrinsics_file": None,
                "extrinsics_available": False,
                "extrinsics_used_by_phase1": False,
                "extrinsics_status": "not_provided",
            },
        },
    )
    trial = writer.start_trial(attempt_index=4)
    assert trial.append_observation(make_observation(0)) == 0
    assert trial.append_action(make_action(0)) == 0
    assert trial.append_observation(make_observation(1, include_extra=True)) == 1
    assert trial.append_action(make_action(1, include_extra=True)) == 1
    assert trial.append_observation(make_observation(2)) == 2
    trial.finish(
        accepted=True,
        operator_label="confirmed_success",
        auto_outcome="success",
        termination_reason="exact_root_distance_threshold",
        accepted_success_index=1,
        reset_randomization_xyz_yaw=np.asarray([0.002, -0.003, 0.0, 0.02]),
    )
    completed = writer.close(clean=True)

    assert completed == final_path
    assert final_path.exists()
    assert not (tmp_path / "phase1_session.partial.h5").exists()
    report = validate_file(final_path)
    assert report.is_valid, [str(issue) for issue in report.errors]
    assert (report.observation_count, report.action_count) == (3, 2)

    with h5py.File(final_path, "r") as file:
        assert bool(file.attrs["closed_cleanly"])
        assert _text(file.attrs["enabled_cameras"]) == '["wrist_1","wrist_2"]'
        assert int(file.attrs["camera_count"]) == 2
        assert float(file.attrs["training_action_clip"]) == TRAINING_ACTION_CLIP
        assert _text(file["metadata/checkpoint_sha256"][()]) == "abc123"
        camera_calibration = file["calibration/wrist_1"]
        assert _text(camera_calibration["extrinsics_file"][()]) == "null"
        assert not bool(camera_calibration["extrinsics_available"][()])
        assert not bool(camera_calibration["extrinsics_used_by_phase1"][()])
        assert _text(camera_calibration["extrinsics_status"][()]) == "not_provided"
        assert "extrinsics_provenance" not in camera_calibration
        stored = file["trials/trial_000000"]
        np.testing.assert_array_equal(
            stored["actions/mapping/observation_index"][:], [0, 1]
        )
        np.testing.assert_array_equal(
            stored["actions/mapping/next_observation_index"][:], [1, 2]
        )
        assert stored["observations/camera/wrist_1/rgb"].shape == (3, *IMAGE_SHAPE)
        temperatures = stored["observations/extra/ambient_temperature_c"][:]
        assert np.isnan(temperatures[0]) and temperatures[1] == np.float32(25.0)
        assert np.isnan(temperatures[2])
        np.testing.assert_array_equal(
            stored["actions/extra/controller_sequence"][:], [-1, 101]
        )
        assert stored["actions/policy/recurrent_hidden_in"].shape == (2, 2, 4)
        assert stored["actions/policy/recurrent_cell_in"].shape == (2, 2, 4)
        assert stored["actions/policy/recurrent_hidden_out"].shape == (2, 2, 4)
        assert stored["actions/policy/recurrent_cell_out"].shape == (2, 2, 4)
        np.testing.assert_array_equal(
            stored["actions/policy/recurrent_hidden_in"][0],
            np.arange(8, dtype=np.float32).reshape(2, 4),
        )
        expected = (
            stored["actions/action/effective_software_action_normalized"][:]
            - stored["actions/policy/mu_raw_normalized"][:]
        )
        np.testing.assert_allclose(
            stored["labels/residual_effective_software_action_minus_raw_mean"][:],
            expected,
        )


def test_single_camera_roundtrip_omits_second_camera_and_skew(tmp_path: Path):
    final_path = tmp_path / "single_camera.h5"
    writer = HDF5SessionWriter(
        final_path,
        image_shape=IMAGE_SHAPE,
        enabled_cameras=("wrist_1",),
        training_action_clip=TRAINING_ACTION_CLIP,
    )
    trial = writer.start_trial(attempt_index=0)
    trial.append_observation(make_single_camera_observation(0))
    trial.append_action(make_action(0))
    trial.append_observation(make_single_camera_observation(1))
    trial.finish(
        accepted=True,
        operator_label="confirmed_success",
        auto_outcome="success",
        termination_reason="success",
        accepted_success_index=0,
    )
    writer.close(clean=True)

    report = validate_file(final_path)
    assert report.is_valid, [str(issue) for issue in report.errors]
    with h5py.File(final_path, "r") as file:
        assert _text(file.attrs["enabled_cameras"]) == '["wrist_1"]'
        assert int(file.attrs["camera_count"]) == 1
        observations = file["trials/trial_000000/observations"]
        assert "wrist_2" not in observations["camera"]
        assert "wrist_2" not in observations["validity"]
        assert "wrist_2_stamp_ns" not in observations["timing"]
        assert "wrist_2_age_ns" not in observations["timing"]
        assert "camera_skew_ns" not in observations["timing"]


def test_single_camera_writer_rejects_disabled_camera_placeholders(tmp_path: Path):
    writer = HDF5SessionWriter(
        tmp_path / "disabled_camera.h5",
        image_shape=IMAGE_SHAPE,
        enabled_cameras=("wrist_1",),
        training_action_clip=TRAINING_ACTION_CLIP,
    )
    trial = writer.start_trial(attempt_index=0)
    with pytest.raises(ValueError, match="Disabled wrist_2 datasets"):
        trial.append_observation(make_observation(0))
    writer.close(clean=False)


def test_missing_camera_identity_uses_legacy_two_camera_contract(tmp_path: Path):
    path = tmp_path / "single_without_identity.h5"
    writer = HDF5SessionWriter(
        path,
        image_shape=IMAGE_SHAPE,
        enabled_cameras=("wrist_1",),
        training_action_clip=TRAINING_ACTION_CLIP,
    )
    trial = writer.start_trial(attempt_index=0)
    trial.append_observation(make_single_camera_observation(0))
    trial.finish(
        accepted=False,
        operator_label="confirmed_failure",
        auto_outcome="timeout",
        termination_reason="timeout",
    )
    writer.close(clean=True)
    with h5py.File(path, "r+") as file:
        del file.attrs["enabled_cameras"]
        del file.attrs["camera_count"]

    report = validate_file(path)
    assert not report.is_valid
    assert any(
        issue.code == "required_dataset_missing" and "wrist_2" in issue.message
        for issue in report.errors
    )


def test_legacy_two_camera_file_keeps_original_optional_timing_contract(tmp_path: Path):
    path = tmp_path / "legacy_two_camera.h5"
    writer = HDF5SessionWriter(
        path,
        image_shape=IMAGE_SHAPE,
        training_action_clip=TRAINING_ACTION_CLIP,
    )
    write_complete_trial(writer, attempt_index=0, accepted=False, action_count=1)
    writer.close(clean=True)
    with h5py.File(path, "r+") as file:
        del file.attrs["enabled_cameras"]
        del file.attrs["camera_count"]
        timing = file["trials/trial_000000/observations/timing"]
        for name in (
            "wrist_1_stamp_ns",
            "wrist_2_stamp_ns",
            "wrist_1_age_ns",
            "wrist_2_age_ns",
            "camera_skew_ns",
        ):
            del timing[name]

    report = validate_file(path)
    assert report.is_valid, [str(issue) for issue in report.errors]


def test_partial_validation_and_non_destructive_recovery_drop_pending_trial(tmp_path: Path):
    final_path = tmp_path / "crash_session.h5"
    writer = HDF5SessionWriter(
        final_path,
        image_shape=IMAGE_SHAPE,
        training_action_clip=TRAINING_ACTION_CLIP,
    )
    write_complete_trial(writer, attempt_index=0, accepted=True, action_count=2)
    pending = writer.start_trial(attempt_index=1)
    pending.append_observation(make_observation(0))
    pending.append_action(make_action(0))
    partial_path = writer.close(clean=False)

    assert partial_path.name.endswith(".partial.h5")
    assert partial_path.exists()
    strict_report = validate_file(partial_path)
    assert not strict_report.is_valid
    crash_report = validate_file(partial_path, allow_partial=True)
    assert crash_report.is_valid, [str(issue) for issue in crash_report.errors]
    assert any(issue.code == "incomplete_trial" for issue in crash_report.warnings)

    recovery = recover_partial_file(partial_path)
    assert partial_path.exists(), "Recovery must preserve the crash artifact"
    assert recovery.kept_trials == ("trial_000000",)
    assert recovery.dropped_trials == ("trial_000001",)
    assert recovery.report.is_valid, [str(issue) for issue in recovery.report.errors]
    with h5py.File(recovery.destination_path, "r") as file:
        assert sorted(file["trials"].keys()) == ["trial_000000"]
        assert bool(file.attrs["closed_cleanly"])
        assert bool(file.attrs["session_complete"])


def test_writer_rejects_noncanonical_image_and_shadow_label(tmp_path: Path):
    writer = HDF5SessionWriter(
        tmp_path / "rejected.h5",
        image_shape=IMAGE_SHAPE,
        training_action_clip=TRAINING_ACTION_CLIP,
    )
    trial = writer.start_trial(attempt_index=0)
    wrong_image = make_observation(0)
    wrong_image["camera"]["wrist_1"]["rgb"] = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="expects per-row shape"):
        trial.append_observation(wrong_image)

    trial.append_observation(make_observation(0))
    invalid = make_action(0, valid=False)
    invalid["labels"]["residual_effective_software_action_minus_raw_mean"][:] = 0.0
    with pytest.raises(ValueError, match="all-NaN"):
        trial.append_action(invalid)
    writer.close(clean=False)


def _session_writer(path: Path) -> HDF5SessionWriter:
    return HDF5SessionWriter(
        path,
        image_shape=IMAGE_SHAPE,
        training_action_clip=TRAINING_ACTION_CLIP,
        flush_every=2,
        metadata={"checkpoint_sha256": "abc123"},
        calibration={"flange_to_controller_tcp_xyzw": [0, 0, -0.01, 0, 0, 0, 1]},
    )


def test_eef_keypoint_distance_roundtrips_and_is_recomputed_by_the_validator(
    tmp_path: Path,
):
    path = tmp_path / "keypoint_session.h5"
    writer = _session_writer(path)
    write_complete_trial(writer, attempt_index=0, accepted=True)
    writer.close(clean=True)

    report = validate_file(path)
    assert report.is_valid, [str(issue) for issue in report.errors]

    dataset_path = (
        "trials/trial_000000/observations/task/"
        "eef_keypoint_to_socket_keypoint_distance_m"
    )
    with h5py.File(path, "r") as file:
        stored = file[dataset_path]
        assert stored.dtype == np.dtype(np.float64)
        assert stored.shape == (3,)
        assert _text(stored.attrs["unit"]) == "m"
        assert bool(stored.attrs["schema_defined"])
        eef_keypoint = file[
            "trials/trial_000000/observations/policy/input"
        ][:, :3].astype(np.float64)
        socket_keypoint = file[
            "trials/trial_000000/observations/task/socket_keypoint_exact_world_xyzw"
        ][:, :3]
        np.testing.assert_allclose(
            stored[:],
            np.linalg.norm(eef_keypoint - socket_keypoint, axis=1),
            rtol=1e-6,
            atol=1e-9,
        )

    # Corrupting the stored distance must be caught by the recomputation check.
    with h5py.File(path, "r+") as file:
        file[dataset_path][0] = float(file[dataset_path][0]) + 0.05
    corrupted = validate_file(path)
    assert not corrupted.is_valid
    assert any(issue.code == "eef_keypoint_distance" for issue in corrupted.errors)


def test_files_without_the_optional_keypoint_distance_still_validate(tmp_path: Path):
    path = tmp_path / "legacy_session.h5"
    writer = _session_writer(path)
    write_complete_trial(writer, attempt_index=0, accepted=True)
    writer.close(clean=True)

    dataset_path = (
        "trials/trial_000000/observations/task/"
        "eef_keypoint_to_socket_keypoint_distance_m"
    )
    with h5py.File(path, "r+") as file:
        del file[dataset_path]

    report = validate_file(path)
    assert report.is_valid, [str(issue) for issue in report.errors]
