# SPARR: Simulation-based Policies with Asymmetric Real-world Residuals for Assembly

Yijie Guo, Iretiayo Akinola, Lars Johannsmeier, Hugo Hadfield, Abhishek Gupta, Yashraj Narang

[[Paper]](https://arxiv.org/abs/2602.23253) [[Website]](https://research.nvidia.com/labs/srl/projects/sparr/)

SPARR combines a **base policy trained in simulation** (low-dimensional state,
dense rewards) with a **residual policy learned in the real world** from wrist
camera images and proprioception, using only a sparse success reward. The
residual's action is added to the base policy's action, so real-world learning
only has to correct the sim-to-real gap rather than learn the task from
scratch.

This repository is a real-robot implementation of SPARR for **DisplayPort plug
insertion on a Flexiv Rizon 4s** with wrist-mounted Intel RealSense cameras (the
reference rig uses two D405s; one or two cameras are supported).

## Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Verify the Socket Pose](#verify-the-socket-pose)
- [Phase 1: Collect Demonstrations](#phase-1-collect-demonstrations)
- [Running Trials](#running-trials)
- [Phase 2: Train the Residual](#phase-2-train-the-residual)
- [Inference: Run a Trained Residual](#inference-run-a-trained-residual)
- [Offline Training from a Recorded Session](#offline-training-from-a-recorded-session)
- [Monitoring and Analysis](#monitoring-and-analysis)
- [Configuration Reference](#configuration-reference)
- [Code Structure](#code-structure)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Overview

```
 Phase 1                      Phase 2                          Inference
 collect demos       ──►      train the residual      ──►      evaluate a trained
 (base policy only)           on the robot (RLPD)              residual
 data/phase1/*.h5             data/runs/<run>/                 data/eval/<run>/
```

**Phase 1** runs the frozen base policy (an RSL-RL checkpoint trained in Isaac
Lab) on the robot with stochastic actions and records every trial — both
cameras, robot state, wrench and the commanded action — to HDF5. Successful
trials become the demonstrations Phase 2 starts from.

**Phase 2** trains the residual with RLPD (SAC with every batch drawn half
from demonstrations, half from online data). It is split into an **actor**,
which runs on the robot at 15 Hz and adds the residual to the base policy's
mean action, and a **learner**, which trains on a GPU and never touches the
robot. They talk over ZeroMQ: finished episodes go to the learner, updated
weights come back between episodes. Two additions on top of plain RLPD:

- **Promotion** — a successful online episode that is no longer than the
  median demonstration is copied into the demonstration buffer.
- **Retry** — once the policy is reliable, a failed attempt is replayed under
  identical conditions (same socket offset and start pose) until it succeeds,
  and that success is promoted.

| | Base policy | Residual policy |
|---|---|---|
| Trained | in simulation (Isaac Lab, PPO) | on the real robot (RLPD) |
| Observes | end-effector and socket keypoint poses (18-D) | 1–2 wrist RGB images (256×256) + flange pose, wrench, base action (19-D) |
| Reward | dense | sparse: 1 on success |
| Output | 6-D end-effector delta | 6-D correction, scaled by `action_scale` and added to the base mean |

Everything runs in two Docker images, one long-lived container per phase.
**Run only one phase at a time** — both claim the robot and the cameras.

## System Requirements

| | Requirement |
|---|---|
| Architecture | **x86_64 only.** SPARR runs on an x86 workstation (the robot and cameras are driven from it). The images are not built or tested for `aarch64`, so it does **not** run on Jetson (including Jetson AGX Thor), even though the Isaac for Manipulation reference deployment targets Thor. |
| Host OS | Linux, with Docker and the NVIDIA Container Toolkit. The Isaac for Manipulation x86 setup targets Ubuntu 24.04; the SPARR containers themselves are Ubuntu 26.04 (ROS 2 Lyrical) and bring their own userland. |
| GPU | An NVIDIA GPU with a driver new enough for CUDA 13 (the Phase 2 image installs JAX with the `cuda13` extra; `session.sh verify` checks GPU/JAX). Phase 1 inference and the Phase 2 learner both use it. |
| Robot | Flexiv Rizon 4s on the same network, firmware compatible with the pinned RDK v1.9.4 (see the note under [Installation](#installation)). |
| Cameras | One or two Intel RealSense cameras with a colour stream, each on its own USB 3 port. The D405 is the reference model, not a requirement: the driver opens any RealSense by serial through `pyrealsense2`. Other camera brands have no driver here. More than two cameras are not supported (the config and recording format name exactly `wrist_1` and optional `wrist_2`). |
| Display | A desktop session (`DISPLAY`) if you want the camera viewers. |

Workstation sizing (GPU memory, CPU RAM) follows the Isaac for Manipulation
release's "System Requirements" page (`manipulator_docs/getting_started/requirements.rst`
in the Isaac ROS docs); note that page's
Thor entry is for the Isaac ROS physical-robot workflow, not SPARR.

### Scene and Background

The residual policy learns from the wrist-camera images, so it is sensitive to
what those cameras see. For best results, make sure the **background visible in
both wrist-camera views is static and constant**: nothing moving behind the
workspace (people, other equipment, screens), no changes in lighting, and the
same layout and objects across every session — Phase 1 collection, Phase 2
training and inference. Changes between sessions look like a distribution shift
to the policy and degrade the success rate.

**Set up the cameras with the livestream before starting SPARR**, and check it
again before every session (see [Live camera view](#monitoring-and-analysis)):

```bash
./docker/phase1/session.sh watch    # camera topics; safe during a session
./docker/phase1/session.sh view     # straight from the devices; stop the camera drivers first
```

1. **Camera angles.** Adjust each wrist camera's mount until the plug and the
   socket are both clearly in view from the start pose all the way to full
   insertion. For best results, make sure the **tip of the plug and the top of
   the socket (its rim)** are clearly visible, so that any contact between them
   during insertion can be seen in the images — that contact is what the
   residual learns to correct.
2. **Crop.** The livestream shows the full frame, but the policy only sees the
   region set by `cameras.wrist_*.crop_xywh` (`[x, y, width, height]` in pixels,
   resized to 256×256). Set it so the plug and socket sit near the centre of
   that region throughout the motion.
3. **Background.** Clear and fix what is visible behind the workspace, as
   described above.

Once training has started, keep the camera mounts, crop and scene unchanged: the
policy is trained on these exact views.

### Grasp and Fingertips

SPARR needs a **stable, constant grasp** of the plug: the base policy and the
socket pose both assume the plug sits in the same place in the gripper on every
trial, and the residual cannot correct for a plug that slips or rotates between
or during trials. We provide **custom fingertips** that hold the plug in a
repeatable position: [`hardware/custom_fingers.step`](hardware/custom_fingers.step)
(STEP, millimetres; both fingertips) — use it to **3D print** the fingertips and
mount them on your gripper. They place the plug slightly differently from the grasp the
measured socket pose assumes, so after fitting them the ground-truth socket pose
needs a small correction — see [Verify the Socket Pose](#verify-the-socket-pose).
Re-check the pose whenever the fingertips or the plug are changed or re-seated.

## Installation

**Requirements:** an **x86_64** Linux machine with Docker and the NVIDIA
Container Toolkit, an NVIDIA GPU, a Flexiv Rizon 4s on the network, and one or
two RealSense cameras on USB 3. See [System Requirements](#system-requirements)
for details.

**1. Create a workspace and clone the repository.** The location is up to you;
every command in this README runs from the repository root, and the scripts
find the repository relative to themselves, so nothing depends on the path.

```bash
mkdir -p ~/workspaces && cd ~/workspaces
git clone https://github.com/NVIDIA/SPARR.git sparr
cd sparr
git checkout release-r1-validation
```

**2. Build the images.** Phase 2 is built on top of Phase 1.

```bash
cd ~/workspaces/sparr                  # the clone from step 1, if you are in a new shell
./docker/phase1/build-phase1.sh        # sparr-phase1:latest — ROS 2 Lyrical, Flexiv driver, Torch
./docker/phase2/build-phase2.sh        # sparr-phase2:latest — adds JAX / serl_launcher
```

Rebuild Phase 2 whenever you rebuild Phase 1, and restart running containers
(`session.sh stop` / `start`) to pick up a new image. Python edits never need a
rebuild — the repository is mounted into the containers. The same goes for
the base policy and `phase1_live.yaml`: they are read from the repository at run
time, not baked into the image, so changing them never needs a rebuild.

> **NOTE:** The Flexiv RDK must match the robot's firmware (Flexiv's own
> compatibility rule). The image is built from NVIDIA's
> [flexiv_ros2](https://github.com/NVIDIA-ISAAC-ROS/flexiv_ros2) fork, pinned to a
> commit and patched for ROS 2 Lyrical (`FLEXIV_ROS2_COMMIT` in
> `docker/phase1/source-lock-phase1.env`), with the RDK pinned separately to
> **v1.9.4** (`FLEXIV_RDK_COMMIT`). **Only RDK v1.9.4 has been built and tested
> here.** If your firmware needs a different RDK, you can try pointing
> `FLEXIV_RDK_COMMIT` at that
> [flexiv_rdk](https://github.com/flexivrobotics/flexiv_rdk) tag's commit
> (`git ls-remote https://github.com/flexivrobotics/flexiv_rdk.git 'refs/tags/v1.*'`)
> and rebuilding both images, but it is not guaranteed to work: the fork and the
> Lyrical patches target a specific RDK API, so the build may fail or need
> changes to the fork pin or patches, and the result is untested.

**3. Add the base policy.** It is not in git. The base policy is an RSL-RL
policy trained in simulation with Isaac Lab, published on NGC in the
[Isaac ROS Manipulation assets](https://catalog.ngc.nvidia.com/orgs/nvidia/isaac/resources/isaac_ros_manipulation_assets/-/file-browser)
(version **5.0.5**, the default latest version). Inside the asset it is under:

```
isaac_ros_manipulation/displayport_insertion/flexiv_policy_torch/model_950.pt
```

How it was trained is described in the
[Isaac ROS](https://nvidia-isaac-ros.github.io/) manipulation documentation.

Download it, together with the `agent.yaml` and `env.yaml` of the same training
run, and place them so these paths exist (or point `policy:` in
`phase1_live.yaml` at wherever you put them, relative to the repository root or
absolute — no image rebuild needed):

```
taskspace_physx/uncalib_none_g4_m950/model_950.pt
taskspace_physx/uncalib_none_g4_m950/params/agent.yaml     # or next to model_950.pt
taskspace_physx/uncalib_none_g4_m950/params/env.yaml
```

The checkpoint's SHA-256 is recorded in every recording's metadata, so each
dataset stays traceable to the policy that produced it. The tool offset and
socket keypoint offset are read from the run's `env.yaml`.

Optionally check the Phase 1 image now. It needs the policy from this step
(the build itself does not), and skips the policy test if it is missing:

```bash
./docker/phase1/test-image-phase1.sh
```

**4. Set up `phase1_live.yaml` for your rig.** Everything else can stay as
shipped.

| Setting | What to set |
|---|---|
| `robot.serial` | your Rizon's serial, e.g. `Rizon4s-012345`. The state topic (`/Rizon4s_012345/flexiv_robot_states`) is derived from it, and the session scripts use it for the Flexiv driver and for `SPARR_ALLOW_LIVE`, which must equal it |
| `socket.opening_pose_world_xyzw` | the pose of the socket opening (where the plug enters) in the world frame, as you measured it. Re-measure whenever the socket moves — success is measured against it |
| `cameras.wrist_*.serial` | each RealSense's serial (`calibration_id` is derived from it) |
| `cameras.wrist_*.crop_xywh` | crop that centres plug and socket in the image |
| `recording.output_directory` | where Phase 1 sessions are written; one folder per experiment |

**Other values you may want to change for your setup.** The shipped
`phase1_live.yaml` matches the reference rig; check each of these against yours.

- **Camera topics, frame IDs and crop.** `cameras.wrist_*.image_topic`,
  `camera_info_topic`, `frame_id` and `crop_xywh` are specific to the reference
  rig (topic namespaces, mounting, field of view). Adjust them, and the crop,
  until the plug and socket are centred as in the reference images. `calibration_id`
  is derived as `realsense-camera-info-<serial>-<W>x<H>-<encoding>-v1`; set it
  explicitly only if you keep your own calibration versions.
- **Socket pose.** `socket.opening_pose_world_xyzw` is the ground-truth pose of
  the socket *opening*, where the plug enters, in the world frame (shipped file:
  z 0.071; the Isaac for Manipulation `manual_socket_pose` uses z ≈ 0.07). The
  policy observes it and success is checked against it. Start from the pose you
  measured, then tune it as described in
  [Verify the Socket Pose](#verify-the-socket-pose) — the fingertips shift where
  the plug sits, so the measured pose usually needs a small correction.
  Re-measure and re-verify whenever the socket moves.
- **Success tolerances.** `success.axis_tolerances_m` (default `[0.008, 0.008,
  0.041]`) assumes the tool keypoint sits about 4 cm behind the plug tip, so a
  seated plug reads |dz| ≈ 40 mm. Tune it for your plug, tool offset and how
  tight you want "inserted" to be.
- **Control rate and action scale.** `policy.control_rate_hz` must match the
  checkpoint's training rate unless you give `control_rate_mismatch_reason`.
  `policy.action_scale_multiplier` accepts any positive value.
- **ROS domain ID.** The session scripts set `ROS_DOMAIN_ID=221` in every
  container. If another machine on your network uses the same ID, or you run
  several SPARR setups side by side, pick a different one with
  `SPARR_ROS_DOMAIN_ID=<0-232>` (exported in each terminal you start
  `session.sh` from; restart the container with `session.sh stop` / `start`).
  All SPARR terminals on one robot must use the same ID.

## Verify the Socket Pose

Do this **before starting SPARR**, and again whenever the socket, plug or
fingertips change. SPARR learns a correction on top of the base policy; if the
ground-truth socket pose is off, the base policy misses consistently and that
error ends up baked into every demonstration.

Set up the cameras first (see [Scene and Background](#scene-and-background)).
The check is the base policy alone, with no randomness, inserting repeatedly
into the socket pose you configured:

1. In `phase1_live.yaml`, set `policy.stochastic: false` (the base policy uses
   its mean action) and set both randomization ranges to zero:

   ```yaml
   randomization:
     start:
       xy_half_range_m: [0.0, 0.0]
     socket_policy_observation:
       xy_half_range_m: [0.0, 0.0]
   ```

   Point `recording.output_directory` at a separate folder (e.g.
   `data/socket_check`) so these trials stay out of your demonstrations.

2. Start Phase 1 as usual (see [Phase 1](#phase-1-collect-demonstrations)) and
   run a batch of identical trials:

   ```bash
   ./docker/phase1/session.sh teach
   ./docker/phase1/session.sh run 20 --yes
   ```

3. **Aim for 90–100% success.** If it is lower, adjust
   `socket.opening_pose_world_xyzw` (usually x/y in 0.5–1 mm steps), restart
   `collect`, and repeat until it is. For trials that did insert, the per-trial
   CSV from `tools/sparr_failure_analysis.py` lists `final_dx_mm`/`final_dy_mm`:
   where the plug actually seated relative to the configured socket. If those are
   consistent across trials, add their average to the x/y of
   `opening_pose_world_xyzw` (converted to metres).

4. When the base policy reliably inserts, set `policy.stochastic: true` again,
   set the randomization you want for training (see below), and start SPARR.

## Phase 1: Collect Demonstrations

**Set the task variation first.** Two settings in `phase1_live.yaml` decide how
much each trial varies; set them for your application before collecting:

```yaml
randomization:
  start:
    xy_half_range_m: [0.01, 0.01]               # robot start pose: +/- 10 mm in x and y
  socket_policy_observation:
    xy_half_range_m: [0.0015, 0.0015]           # socket pose given to the policy: +/- 1.5 mm noise in x and y
```

We suggest **±1.5 mm socket observation noise** and **±10 mm start-pose
offset** in x and y, which is what our experiments used. Keep these values the
**same in Phase 1 and Phase 2** (and in evaluation): the residual learns to
correct the base policy under exactly these conditions, and demonstrations
collected under different ones teach it a different task.

**Aim for at least 30 successful trials** with the base policy. Successful
trials become the Phase 2 demonstrations; fewer gives the residual too little to
start from.

Start the container and the robot-side processes, one terminal each:

```bash
./docker/phase1/session.sh start              # once

./docker/phase1/session.sh robot              # terminal 1: Flexiv driver
./docker/phase1/session.sh camera wrist_1     # terminal 2
./docker/phase1/session.sh camera wrist_2     # terminal 3
SPARR_ALLOW_LIVE=<robot.serial> ./docker/phase1/session.sh collect live   # terminal 4
```

Jog the plug a few cm above the socket, then in a fifth terminal:

```bash
./docker/phase1/session.sh teach      # save this pose as the reset pose (once per session)
./docker/phase1/session.sh run        # run trials back to back
```

`run` drives the trials and asks you to confirm each one; `run 30 --yes` runs
30 trials unattended. See [Running Trials](#running-trials) for both modes.
Keep running until you have **at least 30 successes** (the running tally shows
the count); `check` reports the total for a recording.

- `collect shadow` instead of `collect live` computes and records but never
  moves the robot — use it to check a new setup.
- `collect live 31` continues an earlier session at trial index 31 instead of
  repeating its randomization; it writes a new file. See
  [Stopping, Aborting and Resuming](#stopping-aborting-and-resuming).
- `./docker/phase1/session.sh abort` ends the session and closes the recording.
- `./docker/phase1/session.sh check` validates and summarises the newest recording.
- `./docker/phase1/session.sh stop` when done.

> **NOTE:** Demonstrations must be collected with `policy.stochastic: true`
> (the shipped value). Phase 2 only learns from accepted, stochastic, live
> trials. To measure the base policy on its own, set `policy.stochastic: false`
> and collect as above — the base then outputs its mean action.

## Running Trials

`session.sh run` is the same in Phase 1, Phase 2 training and evaluation. It
starts each trial, waits for it to finish, records its outcome, and prints a
running tally:

```
--- trial 21 of 30 ---   successes 13/20 (65.0%)   |   last 20: 13/20 (65.0%)   |   completed 20/30
```

The first rate is cumulative for this `run`; `last 20` is over the 20 most
recent completed trials (`--window N` to change it, `0` to hide it).

**Interactive** (default) — you confirm every step:

```bash
./docker/phase1/session.sh run          # until recording.target_successes is reached
./docker/phase1/session.sh run 30       # until 30 trials are completed
```

1. **Enter** resets: retract, move to the randomized start pose, settle.
2. **Enter** runs the policy (or `s` to skip this trial).
3. Label the outcome: **Enter** accepts the suggestion, or `s` success /
   `f` failure / `d` discard.

**Automated** — `--yes` skips every prompt and accepts the automatic verdict:

```bash
./docker/phase1/session.sh run 30 --yes
```

| Option | Default | Meaning |
|---|---|---|
| `N` (number) | until `recording.target_successes` | number of **completed** trials to run |
| `--yes` | off | no prompts; label each trial with the automatic success check |
| `--max-consecutive-rejects K` | 10 | with `--yes`, stop after K rejected attempts in a row (`0` = never) |
| `--window N` | 20 | also show the success rate over the last N completed trials (`0` hides it) |
| `--trial-timeout-s S` | 180 | give up if one attempt has not finished after S seconds |

What counts:
- A trial is **completed** when it is labelled success or failure. `run N`
  stops after N completed trials, or earlier if `recording.target_successes`
  is reached.
- A **success** is the geometric check in `success:` of `phase1_live.yaml`
  firing; with `--yes` everything else is recorded as a failure.
- **Not counted:** discarded trials, and trials whose reset fails before the
  policy starts (e.g. the start pose is not reached within
  `trial_control.reposition_timeout_s`). The latter stay in the recording as
  discarded attempts and the **same trial** — same start pose and socket
  offset — runs again.
- A trial stopped by the lateral-drift guard (`max_lateral_offset_m`) or by
  pressing **`x`** during the rollout is always a failure.

### Stopping, Aborting and Resuming

**Stopping `run`.** Press **`x`** to end the current rollout as a failure. `q`
at any prompt (interactive), or **Ctrl-C** (with `--yes`), stops `run`. `run`
also stops on its own after `--max-consecutive-rejects` failed resets in a row,
or when an attempt exceeds `--trial-timeout-s` — fix the cause (robot in Manual
mode, start pose unreachable, a camera down) first. In every case the session
and its recording stay open: run `session.sh run` again to carry on in the same
file. If `run` stopped in the middle of a trial, label that trial first with
`session.sh failure` or `session.sh discard`.

**Ending the session.** When you are done — or to give up on a session that
went wrong — close the recording cleanly:

```bash
./docker/phase1/session.sh abort        # or ./docker/phase2/session.sh abort
```

This ends any trial in progress (recorded as aborted, never as a success),
closes the file and renames it from `.partial.h5` to `.h5`. Ctrl-C in the
`collect` terminal does the same. Always end the session this way before
`session.sh stop` or switching to Phase 2.

**After a crash.** If the collector or the machine died, the recording is left
as `.partial.h5`. Repair a copy (the original is not modified):

```bash
./docker/phase1/session.sh shell -c 'sparr_phase1_recover "$SPARR_REPO"/data/phase1/<file>.partial.h5'
```

**Continuing later.** A new `collect` session always writes a **new file**; it
never appends to an old one. To continue the same randomization sequence
instead of repeating it, pass the next trial index to `collect` — the previous
session's last `attempt_index` + 1, with `numpy_seed` and `randomization:`
unchanged:

```bash
SPARR_ALLOW_LIVE=<robot.serial> ./docker/phase1/session.sh collect live 31
```

The same works for Phase 2 `collect` and for `eval` (`--resume 31`).

**Combining Phase 1 files before Phase 2.** A Phase 1 data set collected over
several sessions ends up in several files. Combine them into one before moving
to Phase 2 (two at a time; chain them for more):

```bash
./docker/phase2/session.sh shell -c 'cd "$SPARR_REPO" && python3 tools/sparr_combine_sessions.py \
    data/phase1/<first>.h5 data/phase1/<continuation>.h5 data/phase1/combined.h5'
```

It checks that both sessions used the same seed, randomization, policy and
socket pose, and leaves the originals untouched. (`session.sh demos` can also
read a folder of files directly; combining gives one file to check, share and
analyse.)

## Phase 2: Train the Residual

```bash
./docker/phase1/session.sh stop       # if running
./docker/phase2/session.sh start
```

Keep `randomization:` in `phase1_live.yaml` exactly as it was for Phase 1 —
Phase 2 uses the same file.

**1. Turn Phase 1 recordings into demonstrations** (once per demo set):

```bash
./docker/phase2/session.sh demos data/phase1 -o data/demos.pkl
```

Sources can be any mix of `.h5` files and folders. The command prints a
`promotion_seed_lengths: [...]` line — paste it into `phase2_live.yaml`.

**2. Name the run.** Run these two lines in **every** terminal you use for
Phase 2, before any `session.sh` command — including the learner terminal:

```bash
export SPARR_DEMOS=data/demos.pkl
export SPARR_PHASE2_RUNS=data/runs/my_first_run      # checkpoints, TensorBoard and recordings
```

Each terminal is a separate shell, so an `export` in one does not reach the
others. A terminal without `SPARR_PHASE2_RUNS` **silently uses
`data/runs/latest`** — for example, a learner started without it writes its
checkpoints to `data/runs/latest` while the robot recordings go to
`data/runs/my_first_run/sessions`. Check before training starts: `learner`
prints `checkpoints: <folder>` and `collect` prints `recordings: <folder>/sessions`,
and both must name your run.

**3. Start the learner and the robot side**, one terminal each (each with the
two `export` lines above):

```bash
./docker/phase2/session.sh learner --checkpoint-period 2500   # terminal 1: training (no robot)
./docker/phase2/session.sh robot                              # terminal 2
./docker/phase2/session.sh camera wrist_1                     # terminal 3
./docker/phase2/session.sh camera wrist_2                     # terminal 4
SPARR_ALLOW_LIVE=<robot.serial> ./docker/phase2/session.sh collect live   # terminal 5
```

then teach and run trials as in Phase 1 (see [Running Trials](#running-trials)):

```bash
./docker/phase2/session.sh teach
./docker/phase2/session.sh run 150 --yes
```

**Watching training (TensorBoard).** In another terminal:

```bash
./docker/phase2/session.sh shell -c 'tensorboard --logdir "$SPARR_REPO"/data/runs --bind_all'
```

then open **http://localhost:6006** (or `http://<robot-pc-ip>:6006` from another
machine). Every run under `data/runs` appears by its folder name; each learner
writes its logs to `<run>/tb`. Logged: losses, Q-values and temperature
(`train/`), buffer sizes and promotions (`buffer/`), per-episode success,
rolling success rate and residual statistics (`episode/`), and a mark at each
checkpoint (`checkpoint/`). If port 6006 is already in use, add `--port 6007`
and open that port instead.

**When to stop.** Watch the `last 20` success rate in the running tally. Once
it reaches **about 85%**, the residual has learned the task and you can stop.
In our experience **about 150 trials** are enough, which takes **about 45
minutes** on the robot. To stop:

1. Stop `run` (Ctrl-C, or let it finish its trials), then close the recording
   with `./docker/phase2/session.sh abort`.
2. Press **Ctrl-C once** in the learner terminal: it finishes the current step
   and writes a final checkpoint (a second Ctrl-C aborts without waiting).
3. Export and evaluate checkpoints (below). The last one is not always the
   best, so evaluate a few.

Robot recordings are saved under `data/runs/my_first_run/sessions/`. To resume
a stopped run, keep the same `SPARR_PHASE2_RUNS` and add `--resume` to
`learner`.

**4. Pick checkpoints from the training logs.** Once Phase 2 is done, open
TensorBoard (see [Watching training](#phase-2-train-the-residual) above) and find
the peaks of `episode/success_rate_last_20`, the success rate over the last 20
trials, plotted against learner steps (iterations). You may see an early peak
around 10k iterations; **ignore it**. For best results use checkpoints at peaks
**after 20k iterations**. Checkpoints are written every `--checkpoint-period`
steps (2,500 above), so take the ones at or just after each peak.

**5. Export those checkpoints so the robot can load them:**

```bash
./docker/phase2/session.sh export data/runs/my_first_run 22500,25000  # the steps you picked
./docker/phase2/session.sh export data/runs/my_first_run              # or every checkpoint
```

This writes `data/runs/my_first_run/actor_params_step<N>.pkl`.

> **NOTE:** The graph narrows down the candidates, but the robot decides.
> Evaluate a few (see [Inference](#inference-run-a-trained-residual)) and keep
> the best.

## Inference: Run a Trained Residual

Same container, robot and cameras as Phase 2, but **no learner**, and `eval`
instead of `collect`:

```bash
SPARR_ALLOW_LIVE=<robot.serial> ./docker/phase2/session.sh eval \
    data/runs/my_first_run/actor_params_step25000.pkl

./docker/phase2/session.sh teach
./docker/phase2/session.sh run 45 --yes       # 45 completed trials, unattended
```

No yaml editing is needed. `eval` runs the residual deterministically, with no
learner, no weight updates, no promotion and no retry, and reads the camera
and state layout and `action_scale` the run was trained with from
`data/runs/<run>/phase2_config.yaml`. Recordings go to
`data/eval/<run>/actor_params_step25000/`.

| Option | Effect |
|---|---|
| `--out DIR` | write recordings somewhere else |
| `--stochastic` | sample the residual instead of taking its mean |
| `--shadow` | dry run: compute and record, never move the robot |
| `--resume N` | continue at trial index N |

The evaluation conditions come from `randomization:` in `phase1_live.yaml`
(start-pose range and socket observation noise). With the same `numpy_seed`,
trial *k* gets the same offsets in every evaluation, so checkpoints are
compared on identical conditions.

**Generalization.** In our experiments SPARR generalizes well to settings it was
not trained on. It holds up under **harder** conditions than it was trained for
(for example more socket observation noise), and does **even better** under
easier ones (for example less observation noise). To test this, change only the
`randomization:` ranges for the evaluation session.

Read the success rate with
`./docker/phase2/session.sh check <file.h5>` or `tools/sparr_failure_analysis.py`.

> **NOTE:** Runs trained before `phase2_config.yaml` was introduced have no such
> file; `eval` then uses `phase2_live.yaml` and warns. Its `image_keys`,
> `state_terms` and `action_scale` must match what the run trained with.

## Offline Training from a Recorded Session

Train the learner again on data a past Phase 2 run already collected, with no
robot — useful for trying learner settings without new robot time.

```bash
# the original run's robot recordings, failures included
./docker/phase2/session.sh demos data/runs/<original_run>/sessions -o data/online.pkl --include-unaccepted

SPARR_DEMOS=data/demos.pkl SPARR_PHASE2_RUNS=data/runs/replay_test \
./docker/phase2/session.sh learner --replay-online data/online.pkl \
    --replay-mode schedule --replay-schedule-tb data/runs/<original_run>/tb
```

| `--replay-mode` | Episodes become available… |
|---|---|
| `schedule` | at the learner steps they arrived in the original run (needs `--replay-schedule-tb`); the most faithful |
| `interval` | one every `--replay-every N` steps |
| `all` | all at step 0; fastest, least like a live run |

## Monitoring and Analysis

**Live camera view**

```bash
./docker/phase2/session.sh watch           # camera topics; safe during a session
./docker/phase2/session.sh view            # straight from the devices; stop the camera drivers first
```

Both work from `docker/phase1/session.sh` too. They open a window, so they need
a desktop session (`DISPLAY`).

**Training curves** — see [Watching training](#phase-2-train-the-residual) in
Phase 2 (TensorBoard).

**Data tools**

| Command | What it does |
|---|---|
| `session.sh check [file.h5]` | validate and summarise a recording (trials, success, camera health) |
| `tools/sparr_visualize.py <file\|dir>` | per-trial video (both cameras + plots) and JSON |
| `tools/sparr_compare.py a.h5 b.h5 [c.h5]` | the same trial across 2–3 runs, side by side |
| `tools/sparr_failure_analysis.py <file\|dir>` | success rate, failure patterns, per-trial CSV (Phase 2 container) |
| `tools/sparr_combine_sessions.py base.h5 cont.h5 out.h5` | merge a resumed session into one file |
| `tools/sparr_crop.py <file.h5>` | preview or apply a different camera crop |
| `sparr_phase1_replay <file.h5> --list-trials` | step through a recording in text |
| `sparr_phase1_recover <file.partial.h5>` | repair a session that crashed (non-destructive) |
| `docker/phase2/session.sh verify` | Phase 2 environment checks: GPU/JAX, actor↔learner link, control-loop timing |
| `session.sh test` | unit tests |

Run Python tools inside a container, with `$SPARR_REPO` paths (not `~`).
`sparr_failure_analysis.py` reads the tool offset from the `env.yaml` of the
policy the session was recorded with; pass `--env-yaml <run>/params/env.yaml`
if that file has moved:

```bash
./docker/phase2/session.sh shell -c 'python3 "$SPARR_REPO"/tools/sparr_failure_analysis.py "$SPARR_REPO"/data/eval/<run>/<ckpt>'
```

## Configuration Reference

**`phase1_live.yaml`** — robot, base policy, cameras, success check, randomization, reset.

| Setting | Effect |
|---|---|
| `randomization.start.xy_half_range_m` | how far (±, m) the start pose is randomized per trial; suggested `[0.01, 0.01]` (±10 mm) |
| `randomization.socket_policy_observation.xy_half_range_m` | noise (±, m) on where the policy is *told* the socket is; suggested `[0.0015, 0.0015]` (±1.5 mm) |
| `randomization.numpy_seed` | which sequence of offsets is drawn |
| `policy.stochastic` | `true` to collect demos, `false` for a base-only baseline |
| `success.axis_tolerances_m`, `hold_steps`, `post_success_steps` | what counts as inserted, and how long to keep pushing after |
| `recording.trial_timeout_s` | rollout length before a trial counts as failed |
| `trial_control.max_lateral_offset_m` | abort a trial that drifts this far sideways |

**`phase2_live.yaml`** — residual settings, layered on top of `phase1_live.yaml`.

| Setting | Effect |
|---|---|
| `runtime.action_scale` | how strongly the residual can move the base action |
| `retry.enabled`, `success_rate_threshold`, `window`, `max_repeats` | replay failed attempts until they succeed, once the success rate is high enough |
| `runtime.promote_episodes` | copy short successful episodes into the demonstration buffer |
| `image_keys` | which cameras the residual sees |
| `state_terms` | which state terms the residual sees (e.g. `wrench`) |

Changing `image_keys` or `state_terms` requires re-running `demos` and a new run.

**Using one camera.** Set `cameras.wrist_2.enabled: false` in `phase1_live.yaml`
(`wrist_1` must stay enabled), run only `session.sh camera wrist_1`, and set
`image_keys: [wrist_1]` in `phase2_live.yaml`.

**Learner flags** (`session.sh learner ...`)

| Flag | Default | Effect |
|---|---|---|
| `--checkpoint-period N` | 5000 | learner steps between checkpoints |
| `--steps N` | until stopped | stop after N steps |
| `--resume [--checkpoint-step N]` | | continue a run from its newest (or a given) checkpoint |
| `--batch-size N` | 256 | half from demonstrations, half from online data |
| `--cta-ratio N` | 2 | critic updates per actor update |
| `--training-starts N` | 100 | online transitions required before training starts |
| `--steps-per-update N` | 50 | learner steps between weight broadcasts to the robot |
| `--max-promotions N` | unlimited | cap on promoted episodes |
| `--seed N` | 42 | |
| `--offline` | | dry run on demonstrations only (checks the loop; not a usable policy) |
| `--replay-online ...` | | see [offline training](#offline-training-from-a-recorded-session) |

**Environment variables** (`session.sh`)

| Variable | Default |
|---|---|
| `SPARR_CONFIG` | `phase1_live.yaml` |
| `SPARR_PHASE2_CONFIG` | `phase2_live.yaml` |
| `SPARR_DEMOS` | `data/demos.pkl` |
| `SPARR_PHASE2_RUNS` | `data/runs/latest` |
| `SPARR_PHASE1_DATA` | `data/phase1` (default source for `demos`) |
| `SPARR_ALLOW_LIVE` | unset — must equal the robot serial for anything that moves the robot |
| `SPARR_ROBOT_SN` | `robot.serial` from `phase1_live.yaml` |
| `SPARR_ROS_DOMAIN_ID` | `221` |

## Code Structure

| Path | Contents |
|---|---|
| [`phase1_live.yaml`](phase1_live.yaml), [`phase2_live.yaml`](phase2_live.yaml) | the two configs |
| [`docker/phase1/`](docker/phase1) | Phase 1 image and `session.sh` (robot, cameras, collection, viewers) |
| [`docker/phase2/`](docker/phase2) | Phase 2 image and `session.sh` (learner, demos, collect, eval, export, verify) |
| [`ros2_ws/src/sparr_phase1_collection/`](ros2_ws/src/sparr_phase1_collection) | collector node, base-policy runtime, trial state machine and reset, camera node, HDF5 recording, preflight |
| [`ros2_ws/src/sparr_phase2_residual/`](ros2_ws/src/sparr_phase2_residual) | residual policy and agent, actor-side collector, learner, RLPD buffers, demo export, promotion, retry, offline replay |
| [`serl_launcher/`](serl_launcher) | vendored SAC/RLPD agents and networks ([HIL-SERL](https://github.com/rail-berkeley/hil-serl)) |
| [`agentlace/`](agentlace) | vendored actor–learner transport |
| [`hardware/custom_fingers.step`](hardware/custom_fingers.step) | CAD of the custom fingertips, for 3D printing; they give a stable plug grasp (see [Grasp and Fingertips](#grasp-and-fingertips)) |
| [`tools/`](tools) | visualisation and analysis of recordings |
| `data/` | recordings, demonstrations, runs (not in git) |

The HDF5 recording format is documented in
[`phase1_hdf5_schema.md`](ros2_ws/src/sparr_phase1_collection/docs/phase1_hdf5_schema.md).

## Troubleshooting

- **Phase 2 checkpoints ended up in `data/runs/latest`** (or recordings and
  checkpoints are in different run folders). One terminal did not have
  `SPARR_PHASE2_RUNS` exported; see step 2 of [Phase 2](#phase-2-train-the-residual).
  Nothing is lost: the files are complete where they were written, and you can
  move them into the intended run folder or pass that folder to `export`.

- **A trial never starts or the reset hangs.** The robot may be in Manual mode,
  which blocks enabling by design. Switch it to Auto on the pendant.
- **The driver reports "Version of this client is incompatible with robot"**, or
  connects and drops within a second or two. The RDK does not match the robot
  firmware — see the RDK note under [Installation](#installation).
- **Success rate drops after a scene change.** Check the background seen by the
  wrist cameras with `session.sh watch`; it must match earlier sessions (see
  [Scene and Background](#scene-and-background)).
- **Camera "Input/output error".** A RealSense can only be opened once. Stop any
  viewer or duplicate driver (`docker rm -f sparr-camera-view`).
- **`demos` exports nothing.** Only accepted trials collected live with
  `policy.stochastic: true` are exported.
- **Learner shape error at the first insert.** The demonstrations were exported
  with different `image_keys`/`state_terms` than the run uses — re-run `demos`.
- **Python edits** take effect on the next process start. `session.sh rebuild`
  is only for `package.xml`/entry-point changes; `build-*.sh` only for
  Dockerfile, dependency or RDK changes.

## Citation

```bibtex
@article{guo2026sparr,
  title   = {{SPARR}: Simulation-based Policies with Asymmetric Real-world Residuals for Assembly},
  author  = {Guo, Yijie and Akinola, Iretiayo and Johannsmeier, Lars and Hadfield, Hugo and Gupta, Abhishek and Narang, Yashraj},
  journal = {arXiv preprint arXiv:2602.23253},
  year    = {2026}
}
```

## Acknowledgements

The residual learner builds on [HIL-SERL / SERL](https://github.com/rail-berkeley/hil-serl)
(`serl_launcher`, `agentlace`) and [RLPD](https://github.com/ikostrikov/rlpd).
The base policy runs with [RSL-RL](https://github.com/leggedrobotics/rsl_rl);
the robot interface uses [flexiv_ros2](https://github.com/flexivrobotics/flexiv_ros2)
and the [Flexiv RDK](https://github.com/flexivrobotics/flexiv_rdk).
