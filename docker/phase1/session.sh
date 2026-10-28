#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA OneWay Noncommercial License (March 22, 2022).
# See the LICENSE file in the repository root for the full license text:
# https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf

#
# Phase 1 session: one long-lived container hosts every process, one terminal each via
# `docker exec`. All DDS peers must share an IPC namespace: FastDDS uses shared memory
# between same-host peers and silently delivers nothing across separate namespaces.
# The repo is bind-mounted at its host path.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

IMAGE_TAG=${SPARR_IMAGE_TAG:-sparr-phase1:latest}
CONTAINER=${SPARR_CONTAINER:-sparr-phase1}
VIEW_CONTAINER=${SPARR_VIEW_CONTAINER:-sparr-camera-view}
WATCH_CONTAINER=${SPARR_WATCH_CONTAINER:-sparr-camera-watch}
ROS_DOMAIN_ID_VALUE=${SPARR_ROS_DOMAIN_ID:-221}
CONFIG=${SPARR_CONFIG:-${REPO}/phase1_live.yaml}
# The robot serial comes from SPARR_ROBOT_SN, else robot.serial in the config.
config_robot_serial() {
  awk '/^robot:/{f=1;next} /^[^ #]/{f=0} f&&$1=="serial:"{print $2; exit}' "$1" 2>/dev/null || true
}
ROBOT_SN=${SPARR_ROBOT_SN:-$(config_robot_serial "${CONFIG}")}

LIVE_ENV="${REPO}/docker/phase1/container/live-env.sh"
ENTRYPOINT=/usr/local/lib/sparr-image/entrypoint.sh

fail() { echo "ERROR: $*" >&2; exit 1; }

[[ -n ${ROBOT_SN} ]] || fail "no robot serial: set robot.serial in ${CONFIG} or SPARR_ROBOT_SN"

usage() {
  cat <<EOF
Phase 1 session -- every process in one container, one terminal each.

  session.sh start              start the container (run once)
  session.sh stop               stop and remove it
  session.sh status             what is running

  session.sh robot              TERMINAL 1: Flexiv driver
  session.sh camera [name] [fps]  TERMINAL 2: RealSense driver for one
                                camera (default wrist_1; fps from the config
                                unless given). Run once per enabled camera.
  session.sh collect [mode]     TERMINAL 3: preflight + collector
                                mode = shadow (default) | live
  session.sh shell              extra terminal inside the container

  session.sh teach              teach the reset pose (do this once)
  session.sh collect <mode> [N] resume: start the randomization stream at
                                trial index N instead of 0, to extend an
                                earlier session of the same seed
  session.sh run [N] [--yes]    run trials back to back, confirming each
  session.sh trial              reset: retract, move to the randomized start, settle
  session.sh begin              run the policy from the start pose (after 'trial')
  session.sh success            confirm the last trial succeeded
  session.sh failure            confirm the last trial failed
  session.sh discard            discard the last trial
  session.sh abort              close the session cleanly

  session.sh view [args...]     live side-by-side view of every connected
                                camera (a stand-in for realsense-viewer).
                                Opens the devices, so stop the drivers first.
  session.sh watch [name...]    live view of the camera TOPICS -- safe to run
                                during a collection session
  session.sh check [file.h5]    validate + summarise a recording
  session.sh test               run the unit test suite
  session.sh rebuild            colcon overlay build (only needed after
                                editing launch files or package.xml)

Environment overrides:
  SPARR_IMAGE_TAG   (default ${IMAGE_TAG})
  SPARR_CONFIG      (default ${CONFIG})
  SPARR_ROS_DOMAIN_ID (default ${ROS_DOMAIN_ID_VALUE})
  SPARR_ROBOT_SN    (default: robot.serial from the config; now ${ROBOT_SN})
EOF
}

running() { [[ -n $(docker ps -q --filter "name=^${CONTAINER}$") ]]; }

require_running() {
  running || fail "container '${CONTAINER}' is not running; run: ${0##*/} start"
}

# Run a command in the session container with ROS sourced and the live source first.
in_container() {
  local tty_flags=()
  [[ -t 0 && -t 1 ]] && tty_flags=(-it)
  docker exec "${tty_flags[@]}" \
    --env "SPARR_REPO=${REPO}" \
    "${CONTAINER}" \
    "${ENTRYPOINT}" "${LIVE_ENV}" "$@"
}

service_call() {
  require_running
  in_container ros2 service call "/sparr_phase1/$1" std_srvs/srv/Trigger '{}'
}

cmd_start() {
  running && { echo "already running: ${CONTAINER}"; return 0; }
  docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1 ||
    fail "image not found: ${IMAGE_TAG}; build it with ${SCRIPT_DIR}/build-phase1.sh"
  [[ -f ${CONFIG} ]] || fail "config not found: ${CONFIG}"
  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
  mkdir -p "${REPO}/data"

  # RealSense may appear as USB (major 189) or video4linux (major 81); /dev is mounted
  # whole so re-plugged devices appear without a restart. Group 46 is plugdev.
  # --ipc=host keeps DDS shared memory working with the watch container and host.
  local user_flags=()
  if [[ ${SPARR_RUN_AS_ROOT:-0} != 1 ]]; then
    user_flags=(--user "$(id -u):$(id -g)" --group-add 46)
  fi
  docker run -d --name "${CONTAINER}" \
    --network host \
    --ipc=host \
    --gpus all \
    --cap-add SYS_NICE \
    --ulimit rtprio=99 \
    --ulimit memlock=-1 \
    --volume /dev:/dev \
    --device-cgroup-rule 'c 189:* rmw' \
    --device-cgroup-rule 'c 81:* rmw' \
    --volume /run/udev:/run/udev:ro \
    "${user_flags[@]}" \
    --env HOME=/tmp/sparr_home \
    --volume "${REPO}:${REPO}" \
    --workdir "${REPO}" \
    --env "SPARR_REPO=${REPO}" \
    --env "ROS_DOMAIN_ID=${ROS_DOMAIN_ID_VALUE}" \
    --env "SPARR_RUNTIME_IMAGE_ID=$(docker image inspect "${IMAGE_TAG}" --format '{{.Id}}')" \
    --entrypoint "${ENTRYPOINT}" \
    "${IMAGE_TAG}" \
    sleep infinity >/dev/null

  echo "started ${CONTAINER}  (ROS_DOMAIN_ID=${ROS_DOMAIN_ID_VALUE})"
  echo "repository mounted at ${REPO} -- same path inside and outside"
  echo
  echo "Next, in three separate terminals:"
  echo "  ${0##*/} robot"
  echo "  ${0##*/} camera"
  echo "  ${0##*/} collect shadow"
}

cmd_stop() {
  docker rm -f "${CONTAINER}" >/dev/null 2>&1 && echo "stopped ${CONTAINER}" ||
    echo "not running"
}

cmd_status() {
  if running; then
    echo "container: ${CONTAINER} UP"
  else
    echo "container: ${CONTAINER} DOWN"
    return 0
  fi
  echo
  echo "processes:"
  # The [x] bracket keeps pgrep from matching this bash -lc command line itself.
  docker exec "${CONTAINER}" bash -lc '
    pgrep -af "[r]os2_control_node"     >/dev/null && echo "  robot driver   RUNNING" || echo "  robot driver   -"
    pgrep -af "[r]ealsense2_camera_node|[s]parr_phase1_camera" >/dev/null && echo "  camera driver  RUNNING" || echo "  camera driver  -"
    pgrep -af "[s]parr_phase1_collector">/dev/null && echo "  collector      RUNNING" || echo "  collector      -"' 2>/dev/null || true
  echo
  echo "topics:"
  in_container ros2 topic list 2>/dev/null | grep -E 'flexiv_robot_states|image_rect_raw|sparr_phase1' |
    sed 's/^/  /' || echo "  (none yet)"
}

cmd_robot() {
  require_running
  echo "Flexiv driver for ${ROBOT_SN}.  Ctrl-C to stop."
  in_container ros2 launch flexiv_bringup rizon_cartesian.launch.py \
    "robot_sn:=${ROBOT_SN}" rizon_type:=Rizon4s start_rviz:=false
}

# One camera driver per camera, node renamed per camera so parameter servers do not collide.
cmd_camera() {
  require_running
  # Empty fps leaves the config's value in effect.
  local camera=${1:-wrist_1} fps=${2:-}
  local fps_args=()
  [[ -n ${fps} ]] && fps_args=(-p "fps:=${fps}")
  echo "RealSense driver for '${camera}' @ ${fps:-config} fps, from ${CONFIG##*/}.  Ctrl-C to stop."
  in_container ros2 run sparr_phase1_collection sparr_phase1_camera --ros-args \
    -r "__node:=sparr_phase1_camera_${camera}" \
    -p "config_path:=${CONFIG}" \
    -p "camera:=${camera}" \
    "${fps_args[@]}"
}

cmd_collect() {
  require_running
  local mode=${1:-shadow}
  local first_attempt=${2:-0}
  [[ ${first_attempt} =~ ^[0-9]+$ ]] ||
    fail "the resume index must be a non-negative integer, got '${first_attempt}'"
  [[ ${mode} == shadow || ${mode} == live ]] ||
    fail "mode must be 'shadow' or 'live', got '${mode}'"
  if [[ ${mode} == live && ${SPARR_ALLOW_LIVE:-} != "${ROBOT_SN}" ]]; then
    fail "live mode needs an explicit opt-in: SPARR_ALLOW_LIVE=${ROBOT_SN} ${0##*/} collect live"
  fi
  echo "Preflight (${mode})..."
  in_container ros2 run sparr_phase1_collection sparr_phase1_preflight \
    --config "${CONFIG}" --mode "${mode}" --wait-for-ros ||
    fail "preflight failed -- fix the reported problems before collecting"
  echo
  echo "Collector starting in ${mode} mode.  Ctrl-C for a clean close."
  # `ros2 run`, not the launch file, so the node comes from the live source tree.
  in_container ros2 run sparr_phase1_collection sparr_phase1_collector --ros-args \
    -p "config_path:=${CONFIG}" \
    -p "execution_mode_override:=${mode}" \
    -p "repository_root:=${REPO}" \
    -p "first_attempt_index:=${first_attempt}" \
    -p use_sim_time:=false
}

# Direct camera viewer in its own container (needs X); a RealSense device can be opened only once.
cmd_view() {
  local display=${DISPLAY:-:0}
  [[ -S /tmp/.X11-unix/X${display#:} ]] ||
    fail "no X socket for DISPLAY=${display}; set DISPLAY, or run: xhost +local:"
  if running && docker exec "${CONTAINER}" pgrep -f "sparr_phase1_camera" >/dev/null 2>&1; then
    fail "a camera driver is running and holds the device; stop it first (Ctrl-C in its terminal)"
  fi
  local strays
  strays=$(docker ps -q --filter "ancestor=${IMAGE_TAG}" --filter "name=${VIEW_CONTAINER}")
  [[ -n ${strays} ]] && echo "replacing a previous viewer that is still running"
  # Fixed name so a forgotten viewer is replaced instead of holding the cameras open.
  docker rm -f "${VIEW_CONTAINER}" >/dev/null 2>&1 || true
  docker run --rm -it --name "${VIEW_CONTAINER}" \
    --network host \
    --volume /dev:/dev \
    --device-cgroup-rule 'c 189:* rmw' \
    --device-cgroup-rule 'c 81:* rmw' \
    --volume /run/udev:/run/udev:ro \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --volume "${REPO}:${REPO}" \
    --workdir "${REPO}" \
    --user "$(id -u):$(id -g)" \
    --env "DISPLAY=${display}" \
    --env HOME=/tmp/sparr_home \
    --entrypoint python3 \
    "${IMAGE_TAG}" \
    "${REPO}/docker/phase1/container/view-cameras.py" "$@"
}

# Topic-based camera view, safe during a session; --ipc=host so DDS shared memory reaches it.
cmd_watch() {
  require_running
  local display=${DISPLAY:-:0}
  [[ -S /tmp/.X11-unix/X${display#:} ]] ||
    fail "no X socket for DISPLAY=${display}; set DISPLAY, or run: xhost +local:"
  docker rm -f "${WATCH_CONTAINER}" >/dev/null 2>&1 || true
  docker run --rm -it --name "${WATCH_CONTAINER}" \
    --network host \
    --ipc=host \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --volume "${REPO}:${REPO}" \
    --workdir "${REPO}" \
    --user "$(id -u):$(id -g)" \
    --env "DISPLAY=${display}" \
    --env HOME=/tmp/sparr_home \
    --env "ROS_DOMAIN_ID=${ROS_DOMAIN_ID_VALUE}" \
    --env "SPARR_CONFIG=${CONFIG}" \
    --entrypoint "${ENTRYPOINT}" \
    "${IMAGE_TAG}" \
    python3 "${REPO}/docker/phase1/container/watch-cameras.py" "$@"
}

cmd_check() {
  require_running
  local file=${1:-}
  if [[ -z ${file} ]]; then
    file=$(ls -t "${REPO}"/data/*/*.h5 2>/dev/null | head -1) ||
      fail "no recordings found under ${REPO}/data"
    [[ -n ${file} ]] || fail "no recordings found under ${REPO}/data"
    echo "most recent recording: ${file}"
    echo
  fi
  in_container ros2 run sparr_phase1_collection sparr_phase1_validate "${file}"
  echo
  in_container ros2 run sparr_phase1_collection sparr_phase1_summary "${file}"
}

cmd_run() {
  require_running
  in_container python3 "${REPO}/docker/phase1/container/run_trials.py" "$@"
}

cmd_test() {
  require_running
  # --basetemp: pytest refuses a /tmp/pytest-of-<user> owned by another uid.
  in_container bash -lc "cd '${REPO}/ros2_ws/src/sparr_phase1_collection' && \
    python3 -m pytest -q --basetemp=/tmp/sparr_pytest_\$\$ test/unit"
}

cmd_rebuild() {
  require_running
  echo "colcon overlay build (needed only for launch/share changes)..."
  in_container bash -lc "cd '${REPO}/ros2_ws' && colcon build \
    --packages-select sparr_phase1_collection \
    --build-base '${REPO}/ros2_ws/build' \
    --install-base '${REPO}/ros2_ws/install' --merge-install"
  echo "done -- new terminals will pick it up"
}

case ${1:-} in
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  status)  cmd_status ;;
  robot)   cmd_robot ;;
  camera)  shift; cmd_camera "${1:-wrist_1}" "${2:-}" ;;
  collect) shift; cmd_collect "${1:-shadow}" "${2:-0}" ;;
  shell)   require_running; shift; in_container bash "$@" ;;
  teach)   service_call teach_reset_pose ;;
  trial)   service_call start_trial ;;
  begin)   service_call begin_rollout ;;
  success) service_call confirm_success ;;
  failure) service_call confirm_failure ;;
  discard) service_call discard_trial ;;
  abort)   service_call abort_session ;;
  run)     shift; cmd_run "$@" ;;
  view)    shift; cmd_view "$@" ;;
  watch)   shift; cmd_watch "$@" ;;
  check)   shift; cmd_check "${1:-}" ;;
  test)    cmd_test ;;
  rebuild) cmd_rebuild ;;
  ""|-h|--help|help) usage ;;
  *) usage; exit 2 ;;
esac
