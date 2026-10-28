# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA OneWay Noncommercial License (March 22, 2022).
# See the LICENSE file in the repository root for the full license text:
# https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf

from pathlib import Path

from sparr_phase1_collection.hdf5_writer import HDF5SessionWriter
from sparr_phase1_collection.summary import _boolean_rate, _percentiles, build_summary

import numpy as np

from recording_fixtures import (
    IMAGE_SHAPE,
    TRAINING_ACTION_CLIP,
    make_action,
    make_single_camera_observation,
)


def test_quality_helpers_handle_empty_and_scale():
    assert _boolean_rate(np.array([], dtype=bool)) is None
    assert _boolean_rate(np.array([True, False])) == 0.5
    values = _percentiles(np.array([1_000_000, 2_000_000]), scale=1.0e-6)
    assert values["p50"] == 1.5
    assert values["max"] == 2.0


def test_single_camera_summary_only_reports_enabled_camera_metrics(tmp_path: Path):
    path = tmp_path / "single.h5"
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
        auto_outcome="timeout",
        termination_reason="timeout",
    )
    writer.close(clean=True)

    summary = build_summary(path)
    assert summary["valid"]
    assert summary["enabled_cameras"] == ["wrist_1"]
    assert summary["camera_count"] == 1
    assert summary["wrist_1_valid_rate"] == 1.0
    assert summary["wrist_1_age_ms"]["max"] == 1.0
    assert "wrist_2_valid_rate" not in summary
    assert "wrist_2_age_ms" not in summary
    assert "camera_skew_ms" not in summary
