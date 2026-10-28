<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

Licensed under the NVIDIA License.
See the LICENSE file in the repository root for the full license text.
-->

# Phase 1 HDF5 schema

The canonical format is schema name `sparr_phase1`, version `1.0.0`. The
machine-readable field inventory is installed from
`schema/sparr_phase1_v1.yaml`; this document describes the storage model and
the semantics that downstream tools must preserve.

## File and row model

Collection always begins in a `.partial.h5` file. A clean session close sets
`closed_cleanly=true` and `session_complete=true`, adds `completed_utc`, closes
the file, and atomically renames it to `.h5`.

Every attempted trial has a deterministic group name and an independent UUID:

```text
/
  attrs
    schema_name = sparr_phase1
    schema_version = 1.0.0
    session_uuid, created_utc, closed_cleanly, session_complete
    pose_convention, action_convention, image_storage
    image_height, image_width, image_channels
    training_action_clip
    camera_count, enabled_cameras
  metadata/
  calibration/
  trials/
    attrs: count
    trial_000000/
      attrs
        trial_uuid, attempt_index, start_utc, end_utc
        complete, accepted, export_eligible
        operator_label, auto_outcome, termination_reason
        observation_count, action_count
      observations/...
      actions/...
      labels/...
```

For a complete trial, `N` observations and `M` actions satisfy `N = M + 1`.
Action row `i` records `observation_index=i` and
`next_observation_index=i+1`. While an action is pending in a partial file, its
next index is `-1`. Observation and action rows are appended independently so
this relationship survives incremental flushes.

The integer root attribute `camera_count` and JSON root attribute
`enabled_cameras` declare the session camera layout. `metadata/` holds the
resolved collection configuration, checkpoint policy contract, source and
asset provenance, environment, label semantics, and command semantics.
`calibration/` holds the captured camera information and the
availability/status of optional extrinsics. When a real extrinsics file is
provided, its artifact provenance is included; otherwise the metadata records
that no transform was provided or used by Phase 1. Named metadata values are
datasets, not root attributes. The precise contents may grow without changing
canonical trial rows.

## Observation rows

Timing preserves ROS, robot, local monotonic, control-tick, and selected camera
timestamps. Camera age is stored for every enabled stream. Camera-pair skew is
stored only when both wrist cameras are enabled; the schema does not imply
hardware synchronization.

The robot group contains the numeric fields available from the single
`flexiv_msgs/RobotStates` sample, including:

- link- and motor-side joint position and velocity;
- measured, desired, derivative, external, and interaction torque;
- temperature, flange pose, TCP pose, and TCP velocity as published;
- raw flange FT wrench and external wrench variants in their source frames.

`observations/extra/raw_robot_states_json` preserves the full message field
structure and header/frame metadata for provenance. Numeric canonical datasets
remain the source for training and validation.

Task rows keep the supplied socket pose, exact and policy-noisy socket roots,
exact and policy-noisy socket keypoints, randomized physical start flange pose,
and the 3-D flange-to-exact-root distance. Success is based only on that exact
root distance. Socket noise is policy-observation noise and does not move the
success reference.

`observations/policy/input` is the exact 18-D base-policy input in this order:

```text
[eef_position(3), eef_rotation_6d(6), socket_keypoint_position(3),
 socket_keypoint_rotation_6d(6)]
```

`wrist_1` is required. `wrist_2` is session-optional. Each enabled image
dataset contains only configured cropped/resized RGB `uint8` frames with
canonical shape `[N, 128, 128, 3]`. A one-camera session omits `wrist_2` image,
timestamp, age, validity, and camera-skew datasets. It never writes zero-filled
stand-ins for a camera that was disabled. Full-resolution frames are
forbidden.

Quality summaries report `camera_count`, `enabled_cameras`, and validity/age
metrics only for enabled streams. They report camera skew only for a two-camera
session.

## Action rows and labels

Each action row comes from one recurrent stochastic policy forward and stores:

- raw normalized mean, standard deviation, and stochastic sample;
- the sample clipped by the root `training_action_clip` value;
- recurrent input/output state when exposed by the runtime; for the approved
  LSTM, hidden and cell are separate `[M, layers, hidden]` datasets in
  layer-major order;
- scaled world-frame translation and rotation-vector delta;
- requested/effective flange target;
- `submitted_controller_tcp_target_world_xyzw`;
- command publish flag, guard bitmask, and timing.

The submitted target records only what the collector published to the
controller topic. It is never described as accepted or executed, because this
command path has no acknowledgement.

The canonical pseudo-label dataset is:

```text
labels/residual_effective_software_action_minus_raw_mean
  = actions/action/effective_software_action_normalized
  - actions/policy/mu_raw_normalized
```

`labels/command_valid` is false when the action is not training-eligible,
including shadow-mode or invalid/stale-input rows. Invalid residuals are stored
as all-NaN. A valid command label requires `command_published=true`.

## Extensions

Extra raw fields are permitted only below `observations/extra/`,
`actions/extra/`, or `labels/extra/`. The first value fixes an extension's
dtype and trailing shape; missing rows are backfilled with the dtype-specific
fill value. Names that imply full-resolution images or invented gripper data
are rejected.

## Validation and crash recovery

`sparr_phase1_validate` checks root identity, enabled-camera metadata against
the camera datasets, dtype/shape consistency, required fields, row mappings,
image constraints, action clipping, label arithmetic, invalid-label NaNs,
exact-root distance, and trial completion. Use
`--allow-partial` only for explicit crash inspection.

`sparr_phase1_recover` copies the source first and never modifies it. By
default it drops each structurally incomplete trial group from the recovered
copy, updates the trial count, records recovery provenance, and validates the
result. `--keep-incomplete` retains those groups for forensic inspection and
therefore produces a file that is still partial in semantics.
