# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA License.
# See the LICENSE file in the repository root for the full license text.

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("h5py")

from sparr_phase1_collection.hdf5_writer import HDF5SessionWriter  # noqa: E402
from sparr_phase1_collection.replay import ReadOnlyReplay, main  # noqa: E402

from recording_fixtures import (  # noqa: E402
    IMAGE_SHAPE,
    TRAINING_ACTION_CLIP,
    make_action,
    make_observation,
    make_single_camera_observation,
    write_complete_trial,
)


def _collection(tmp_path: Path) -> Path:
    path = tmp_path / "replay_source.h5"
    writer = HDF5SessionWriter(
        path,
        image_shape=IMAGE_SHAPE,
        training_action_clip=TRAINING_ACTION_CLIP,
    )
    write_complete_trial(writer, attempt_index=7, accepted=True, action_count=2)
    write_complete_trial(writer, attempt_index=8, accepted=False, action_count=1)
    writer.close(clean=True)
    return path


def test_replay_iterates_canonical_mapping_without_writing_source(tmp_path: Path):
    source = _collection(tmp_path)
    before = source.stat().st_mtime_ns

    with ReadOnlyReplay(source) as replay:
        trials = replay.trials()
        assert [(trial.name, trial.attempt_index) for trial in trials] == [
            ("trial_000000", 7),
            ("trial_000001", 8),
        ]
        assert replay.select_trials(["7"]) == (trials[0],)
        steps = list(replay.iter_steps("trial_000000", include_images=True))
        assert len(steps) == 2
        assert (steps[0].observation_index, steps[0].next_observation_index) == (0, 1)
        np.testing.assert_allclose(
            steps[0].raw_mean_normalized,
            make_action(0)["policy"]["mu_raw_normalized"],
        )
        np.testing.assert_allclose(
            steps[0].submitted_controller_tcp_target_world_xyzw,
            make_action(0)["action"][
                "submitted_controller_tcp_target_world_xyzw"
            ],
        )
        assert steps[0].wrist_1_rgb.shape == IMAGE_SHAPE
        data = steps[0].as_dict(include_image_statistics=True)
        assert data["images"]["wrist_1"]["shape"] == list(IMAGE_SHAPE)
        assert len(data["images"]["wrist_1"]["sha256"]) == 64

    assert source.stat().st_mtime_ns == before


def test_replay_includes_available_image_for_single_camera_session(tmp_path: Path):
    path = tmp_path / "single_camera_replay.h5"
    writer = HDF5SessionWriter(
        path,
        image_shape=IMAGE_SHAPE,
        enabled_cameras=("wrist_1",),
        training_action_clip=TRAINING_ACTION_CLIP,
    )
    trial = writer.start_trial(attempt_index=0)
    trial.append_observation(make_single_camera_observation(0))
    trial.append_action(make_action(0))
    trial.append_observation(make_single_camera_observation(1))
    trial.finish(
        accepted=False,
        operator_label="confirmed_failure",
        auto_outcome="failure",
        termination_reason="test",
    )
    writer.close(clean=True)

    with ReadOnlyReplay(path) as replay:
        step = next(replay.iter_steps("trial_000000", include_images=True))

    assert step.wrist_1_rgb is not None
    assert step.wrist_1_rgb.shape == IMAGE_SHAPE
    assert step.wrist_2_rgb is None


def test_replay_cli_lists_trials_and_emits_bounded_json_steps(tmp_path: Path, capsys):
    source = _collection(tmp_path)
    assert main([str(source), "--list-trials", "--trial", "8"]) == 0
    listed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(listed) == 1
    assert listed[0]["record_type"] == "trial"
    assert listed[0]["attempt_index"] == 8

    assert main(
        [
            str(source),
            "--trial",
            "trial_000000",
            "--start-step",
            "1",
            "--max-steps",
            "1",
        ]
    ) == 0
    replayed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["record_type"] for row in replayed] == ["trial", "step"]
    assert replayed[1]["action_index"] == 1
    assert "submitted_controller_tcp_target_world_xyzw" in replayed[1]


def test_replay_rejects_incomplete_source_without_explicit_partial_mode(tmp_path: Path):
    writer = HDF5SessionWriter(
        tmp_path / "interrupted.h5",
        image_shape=IMAGE_SHAPE,
        training_action_clip=TRAINING_ACTION_CLIP,
    )
    trial = writer.start_trial(attempt_index=0)
    trial.append_observation(make_observation(0))
    trial.append_action(make_action(0))
    partial = writer.close(clean=False)

    with pytest.raises(ValueError, match="Cannot replay invalid"):
        ReadOnlyReplay(partial)
    with ReadOnlyReplay(partial, allow_partial=True) as replay:
        assert replay.select_trials() == ()
        incomplete = replay.select_trials(include_incomplete=True)
        assert len(incomplete) == 1 and not incomplete[0].complete
        pending = list(replay.iter_steps(incomplete[0].name))
        assert pending[0].next_observation_index is None
        assert pending[0].next_flange_pose_world_xyzw is None


def test_replay_supports_optional_std_and_emits_strict_json_for_invalid_label(
    tmp_path: Path, capsys
):
    path = tmp_path / "optional_std.h5"
    writer = HDF5SessionWriter(
        path,
        image_shape=IMAGE_SHAPE,
        training_action_clip=TRAINING_ACTION_CLIP,
    )
    trial = writer.start_trial(attempt_index=0)
    trial.append_observation(make_observation(0))
    action = make_action(0, valid=False)
    del action["policy"]["std_raw_normalized"]
    trial.append_action(action)
    trial.append_observation(make_observation(1))
    trial.finish(
        accepted=False,
        operator_label="safety_reject",
        auto_outcome="rejected",
        termination_reason="invalid_input",
    )
    writer.close(clean=True)

    with ReadOnlyReplay(path) as replay:
        step = next(replay.iter_steps("trial_000000"))
        assert step.raw_std_normalized is None

    assert main([str(path), "--trial", "trial_000000"]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rows[1]["raw_std_normalized"] is None
    assert rows[1]["residual_effective_software_action_minus_raw_mean"] == [None] * 6
