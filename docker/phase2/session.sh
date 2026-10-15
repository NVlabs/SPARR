#!/usr/bin/env bash
#
# Phase 2 session: Phase 1 collection with a learned residual, plus the learner that trains it.
# Same layout as docker/phase1/session.sh; must not run alongside a Phase 1 session.
# The learner needs no robot or ROS and talks to the actor over ZeroMQ.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

IMAGE_TAG=${SPARR_PHASE2_IMAGE_TAG:-sparr-phase2:latest}
CONTAINER=${SPARR_PHASE2_CONTAINER:-sparr-phase2}
VIEW_CONTAINER=${SPARR_VIEW_CONTAINER:-sparr-camera-view}
WATCH_CONTAINER=${SPARR_WATCH_CONTAINER:-sparr-camera-watch}
ROS_DOMAIN_ID_VALUE=${SPARR_ROS_DOMAIN_ID:-221}
CONFIG=${SPARR_CONFIG:-${REPO}/phase1_live.yaml}
# The robot serial comes from SPARR_ROBOT_SN, else robot.serial in the config.
config_robot_serial() {
  awk '/^robot:/{f=1;next} /^[^ #]/{f=0} f&&$1=="serial:"{print $2; exit}' "$1" 2>/dev/null || true
}
ROBOT_SN=${SPARR_ROBOT_SN:-$(config_robot_serial "${CONFIG}")}
PHASE2_CONFIG=${SPARR_PHASE2_CONFIG:-${REPO}/phase2_live.yaml}
DEMOS=${SPARR_DEMOS:-${REPO}/data/demos.pkl}
PHASE1_DATA=${SPARR_PHASE1_DATA:-${REPO}/data/phase1}
RUNS=${SPARR_PHASE2_RUNS:-${REPO}/data/runs/latest}

LIVE_ENV="${REPO}/docker/phase2/container/live-env.sh"
ENTRYPOINT=/usr/local/lib/sparr-image/entrypoint.sh

fail() { echo "ERROR: $*" >&2; exit 1; }

[[ -n ${ROBOT_SN} ]] || fail "no robot serial: set robot.serial in ${CONFIG} or SPARR_ROBOT_SN"

usage() {
  cat <<EOF
Phase 2 session -- Phase 1 collection with a learned residual.

  session.sh start              start the container (run once)
  session.sh stop               stop and remove it
  session.sh status             what is running

  session.sh demos [src ...] [-o out.pkl]
                                convert Phase 1 recordings -> demonstrations.
                                src is any mix of .h5 files and directories of
                                them; it defaults to \$SPARR_PHASE1_DATA.
  session.sh learner [args...]  TERMINAL 1: RLPD learner (no robot, no ROS)

  session.sh robot              TERMINAL 2: Flexiv driver
  session.sh camera [name] [fps]  TERMINAL 3: RealSense driver for one
                                camera (default wrist_1, 30 fps). Run it
                                once per enabled camera.
  session.sh collect [mode] [N] TERMINAL 4: preflight + Phase 2 collector.
                                resume: start the randomization stream at
                                trial index N instead of 0, to extend an
                                earlier session of the same seed/config
                                mode = shadow (default) | live
                                recordings go to \$SPARR_PHASE2_RUNS/sessions

  session.sh export <run-dir> [all|N,N...]
                                turn learner checkpoints into robot-loadable
                                <run-dir>/actor_params_step<N>.pkl (default all)
  session.sh eval <params.pkl> [--out DIR] [--stochastic] [--shadow] [--resume N]
                                TERMINAL 4, instead of 'collect': run a trained
                                residual on the robot. No yaml edits: the
                                residual is forced active + deterministic, no
                                learner, no promotion, no retry. Recordings go
                                to data/eval/<run>/<params>/ unless --out.
                                Live by default (needs SPARR_ALLOW_LIVE);
                                --shadow records without commanding the robot.

  session.sh teach              teach the reset pose (do this once)
  session.sh run [N] [--yes]    run trials back to back
  session.sh trial              start a single trial
  session.sh success            confirm the last trial succeeded
  session.sh failure            confirm the last trial failed
  session.sh discard            discard the last trial
  session.sh abort              close the session cleanly

  session.sh watch [name...]    live view of the camera topics -- safe to run
                                during a session (training or eval)
  session.sh view [args...]     live view straight from the devices; stop the
                                camera drivers first
  session.sh check [file.h5]    validate + summarise a recording (Phase 1 tool)
  session.sh validate [path]    Phase 2 KPIs: control rate, action fidelity,
                                shadow integrity, success rate, cycle time
  session.sh verify             environment, transport, control-budget and
                                online-loop gates
  session.sh test               run the Phase 2 test suite
  session.sh shell              extra terminal inside the container
  session.sh rebuild            colcon overlay build (launch/share changes only)

Two safety gates worth knowing:
  * 'collect live' needs SPARR_ALLOW_LIVE=${ROBOT_SN}. This is Phase 1's gate:
    it decides whether commands reach the robot at all.
  * Whether the RESIDUAL reaches the commanded action is separate, and set by
    'residual.runtime.mode' in ${PHASE2_CONFIG##*/}. It ships as 'shadow',
    which records the residual without ever commanding it. Active mode also
    refuses to start without trained parameters.

Environment overrides:
  SPARR_PHASE2_IMAGE_TAG  (default ${IMAGE_TAG})
  SPARR_CONFIG            (default ${CONFIG})
  SPARR_PHASE2_CONFIG     (default ${PHASE2_CONFIG})
  SPARR_DEMOS             (default ${DEMOS})
  SPARR_PHASE2_RUNS       (default ${RUNS})
  SPARR_ROS_DOMAIN_ID     (default ${ROS_DOMAIN_ID_VALUE})
  SPARR_ROBOT_SN          (default: robot.serial from the config; now ${ROBOT_SN})
EOF
}

running() { [[ -n $(docker ps -q --filter "name=^${CONTAINER}$") ]]; }

require_running() {
  running || fail "container '${CONTAINER}' is not running; run: ${0##*/} start"
}

phase1_running() { [[ -n $(docker ps -q --filter "name=^sparr-phase1$") ]]; }

in_container() {
  local tty_flags=()
  [[ -t 0 && -t 1 ]] && tty_flags=(-it)
  docker exec "${tty_flags[@]}" \
    --env "SPARR_REPO=${REPO}" \
    --env "SPARR_PHASE2_CONFIG=${PHASE2_CONFIG}" \
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
    fail "image not found: ${IMAGE_TAG}; build it with ${SCRIPT_DIR}/build-phase2.sh"
  [[ -f ${CONFIG} ]] || fail "Phase 1 config not found: ${CONFIG}"
  [[ -f ${PHASE2_CONFIG} ]] || fail "Phase 2 config not found: ${PHASE2_CONFIG}"

  phase1_running &&
    fail "the Phase 1 container is running; stop it first: docker/phase1/session.sh stop"

  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
  mkdir -p "${REPO}/data/runs"

  # Device access as in docker/phase1/session.sh.
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
  echo "Next:"
  echo "  ${0##*/} demos                 # once, from Phase 1 recordings"
  echo "  ${0##*/} learner               # terminal 1"
  echo "  ${0##*/} robot                 # terminal 2"
  echo "  ${0##*/} camera                # terminal 3"
  echo "  ${0##*/} collect shadow        # terminal 4"
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
  # The [x] bracket keeps pgrep from matching its own command line.
  local procs=(
    "flexiv driver:[f]lexiv"
    "camera:sparr_phase1_[c]amera"
    "collector:sparr_phase2_[c]ollector"
    "learner:sparr_phase2_[l]earner"
  )
  for entry in "${procs[@]}"; do
    local label=${entry%%:*} pattern=${entry#*:}
    if docker exec "${CONTAINER}" pgrep -af "${pattern}" >/dev/null 2>&1; then
      echo "  ${label}: running"
    else
      echo "  ${label}: not running"
    fi
  done
  echo
  echo "residual mode: $(grep -E '^\s+mode:' "${PHASE2_CONFIG}" | head -1 | awk '{print $2}')"
  [[ -f ${DEMOS} ]] && echo "demonstrations: ${DEMOS}" ||
    echo "demonstrations: MISSING (${DEMOS}) -- run: ${0##*/} demos"
}

cmd_demos() {
  require_running
  local out=${DEMOS}
  local -a sources=() extra=()
  while (( $# )); do
    case $1 in
      -o|--output) [[ -n ${2:-} ]] || fail "${1} needs a path"; out=$2; shift 2 ;;
      -*)          extra+=("$1"); shift ;;   # passed through to demo_export
      *)           sources+=("$1"); shift ;;
    esac
  done
  (( ${#sources[@]} )) || sources=("${PHASE1_DATA}")
  local src
  for src in "${sources[@]}"; do
    [[ -e ${src} ]] || fail "no such recording or directory: ${src}"
  done
  mkdir -p "$(dirname "${out}")"
  echo "Converting Phase 1 recordings -> demonstrations"
  printf '  source: %s\n' "${sources[@]}"
  echo "  output: ${out}"
  echo
  # --config so the exported observation layout matches the runtime's.
  in_container python3 -m sparr_phase2_residual.demo_export \
    "${sources[@]}" --output "${out}" --config "${PHASE2_CONFIG}" "${extra[@]}" || {
      echo
      echo "No demonstrations were exported."
      echo "A Phase 1 trial is only eligible if it was ACCEPTED and was"
      echo "collected with 'stochastic: true' in ${CONFIG##*/}. Deterministic"
      echo "trials are debug-only: their residual label is zero by"
      echo "construction, so training on them would learn nothing."
      return 1
    }
}

cmd_learner() {
  require_running
  [[ -f ${DEMOS} ]] ||
    fail "no demonstrations at ${DEMOS}; run: ${0##*/} demos"
  mkdir -p "${RUNS}"
  echo "Learner starting. No robot, no ROS -- ZeroMQ on 5588/5589."
  echo "  demonstrations: ${DEMOS}"
  echo "  checkpoints:    ${RUNS}"
  echo
  in_container python3 -m sparr_phase2_residual.learner_main \
    --demos "${DEMOS}" \
    --checkpoint-dir "${RUNS}" \
    --config "${PHASE2_CONFIG}" \
    "$@"
}

# rizon_cartesian.launch.py provides the cartesian controller the collector publishes to.
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
  [[ ${mode} == shadow || ${mode} == live ]] ||
    fail "mode must be 'shadow' or 'live', got '${mode}'"
  [[ ${first_attempt} =~ ^[0-9]+$ ]] ||
    fail "the resume index must be a non-negative integer, got '${first_attempt}'"
  if [[ ${mode} == live && ${SPARR_ALLOW_LIVE:-} != "${ROBOT_SN}" ]]; then
    fail "live mode needs an explicit opt-in: SPARR_ALLOW_LIVE=${ROBOT_SN} ${0##*/} collect live"
  fi
  [[ -f ${PHASE2_CONFIG} ]] || fail "Phase 2 config not found: ${PHASE2_CONFIG}"

  echo "Preflight (${mode})..."
  in_container ros2 run sparr_phase1_collection sparr_phase1_preflight \
    --config "${CONFIG}" --mode "${mode}" --wait-for-ros ||
    fail "preflight failed -- fix the reported problems before collecting"
  echo
  local out="${RUNS}/sessions"
  [[ ${out} = /* ]] || out="${REPO}/${out}"
  echo "Phase 2 collector starting in ${mode} mode.  Ctrl-C for a clean close."
  echo "  recordings: ${out}"
  in_container python3 -m sparr_phase2_residual.main --ros-args \
    -p "config_path:=${CONFIG}" \
    -p "phase2_config_path:=${PHASE2_CONFIG}" \
    -p "execution_mode_override:=${mode}" \
    -p "repository_root:=${REPO}" \
    -p "first_attempt_index:=${first_attempt}" \
    -p "output_directory_override:=${out}" \
    -p use_sim_time:=false
}

cmd_export() {
  require_running
  local run=${1:-} steps=${2:-all}
  [[ -n ${run} ]] || fail "usage: ${0##*/} export <run-dir> [all|N,N...]"
  [[ -d ${run} ]] || fail "no such run directory: ${run}"
  in_container python3 -m sparr_phase2_residual.learner_main \
    --checkpoint-dir "${run}" \
    --config "${PHASE2_CONFIG}" \
    --no-tensorboard \
    --export-steps "${steps}"
}

cmd_eval() {
  require_running
  local params="" out="" mode=live stochastic=false first_attempt=0
  while (( $# )); do
    case $1 in
      --out)        [[ -n ${2:-} ]] || fail "--out needs a directory"; out=$2; shift 2 ;;
      --stochastic) stochastic=true; shift ;;
      --shadow)     mode=shadow; shift ;;
      --resume)     [[ ${2:-} =~ ^[0-9]+$ ]] || fail "--resume needs a trial index"
                    first_attempt=$2; shift 2 ;;
      -*)           fail "unknown eval option: $1" ;;
      *)            [[ -z ${params} ]] || fail "only one params file, got '${params}' and '$1'"
                    params=$1; shift ;;
    esac
  done
  [[ -n ${params} ]] ||
    fail "usage: ${0##*/} eval <actor_params.pkl> [--out DIR] [--stochastic] [--shadow] [--resume N]"
  [[ ${params} = /* ]] || params="${REPO}/${params}"
  [[ -f ${params} ]] || fail "no actor params at ${params}; make one with: ${0##*/} export <run-dir>"
  if [[ -z ${out} ]]; then
    local stem=${params##*/}
    out="${REPO}/data/eval/$(basename "$(dirname "${params}")")/${stem%.pkl}"
  fi
  [[ ${out} = /* ]] || out="${REPO}/${out}"
  if [[ ${mode} == live && ${SPARR_ALLOW_LIVE:-} != "${ROBOT_SN}" ]]; then
    fail "eval drives the robot and needs an explicit opt-in: SPARR_ALLOW_LIVE=${ROBOT_SN} ${0##*/} eval ..."
  fi

  echo "Preflight (${mode})..."
  in_container ros2 run sparr_phase1_collection sparr_phase1_preflight \
    --config "${CONFIG}" --mode "${mode}" --wait-for-ros ||
    fail "preflight failed -- fix the reported problems before evaluating"
  echo
  echo "Evaluating ${params}"
  echo "  recordings: ${out}"
  echo "  start/socket randomization: from ${CONFIG##*/} (randomization:)"
  echo
  in_container python3 -m sparr_phase2_residual.main --ros-args \
    -p "config_path:=${CONFIG}" \
    -p "phase2_config_path:=${PHASE2_CONFIG}" \
    -p "execution_mode_override:=${mode}" \
    -p "repository_root:=${REPO}" \
    -p "first_attempt_index:=${first_attempt}" \
    -p "output_directory_override:=${out}" \
    -p "eval_params_path:=${params}" \
    -p "eval_stochastic:=${stochastic}" \
    -p use_sim_time:=false
}

# Camera viewers as in docker/phase1/session.sh.
x_display() {
  local display=${DISPLAY:-:0}
  [[ -S /tmp/.X11-unix/X${display#:} ]] ||
    fail "no X socket for DISPLAY=${display}; set DISPLAY, or run: xhost +local:"
  echo "${display}"
}

cmd_watch() {
  require_running
  local display
  display=$(x_display)
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

cmd_view() {
  local display
  display=$(x_display)
  if running && docker exec "${CONTAINER}" pgrep -f "sparr_phase1_camera" >/dev/null 2>&1; then
    fail "a camera driver is running and holds the device; stop it first (Ctrl-C in its terminal)"
  fi
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

cmd_verify() {
  require_running
  echo "== environment gate"
  in_container python3 /opt/sparr/verify-phase2-env.py "$@"
  echo
  echo "== transport gate"
  in_container python3 /opt/sparr/verify-transport.py
  echo
  echo "== control-budget gate"
  in_container python3 "${REPO}/docker/phase2/verify-control-budget.py"
  echo
  echo "== online-loop gate (actor <-> learner, no robot)"
  in_container python3 "${REPO}/docker/phase2/verify-online-loop.py"
}

cmd_validate() {
  require_running
  local target=${1:-}
  if [[ -z ${target} ]]; then
    target="${PHASE1_DATA}"
    echo "validating everything under ${target}"
    echo
  fi
  in_container python3 -m sparr_phase2_residual.validation "${target}" --verbose
}

cmd_test() {
  require_running
  in_container bash -lc "cd '${REPO}/ros2_ws/src/sparr_phase2_residual' && \
    python3 -m pytest -q --basetemp=/tmp/sparr_pytest_\$\$ test/"
}

cmd_rebuild() {
  require_running
  echo "colcon overlay build (needed only for launch/share changes)..."
  in_container bash -lc "cd '${REPO}/ros2_ws' && colcon build \
    --packages-select sparr_phase1_collection sparr_phase2_residual \
    --build-base '${REPO}/ros2_ws/build' \
    --install-base '${REPO}/ros2_ws/install' --merge-install"
  echo "done -- new terminals will pick it up"
}

case ${1:-} in
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  status)  cmd_status ;;
  demos)   shift; cmd_demos "$@" ;;
  learner) shift; cmd_learner "$@" ;;
  robot)   cmd_robot ;;
  camera)  shift; cmd_camera "${1:-wrist_1}" "${2:-}" ;;
  collect) shift; cmd_collect "${1:-shadow}" "${2:-0}" ;;
  export)  shift; cmd_export "$@" ;;
  eval)    shift; cmd_eval "$@" ;;
  shell)   require_running; shift; in_container bash "$@" ;;
  teach)   service_call teach_reset_pose ;;
  trial)   service_call start_trial ;;
  success) service_call confirm_success ;;
  failure) service_call confirm_failure ;;
  discard) service_call discard_trial ;;
  abort)   service_call abort_session ;;
  run)     require_running; shift
           in_container python3 "${REPO}/docker/phase1/container/run_trials.py" "$@" ;;
  watch)   shift; cmd_watch "$@" ;;
  view)    shift; cmd_view "$@" ;;
  check)   shift; cmd_check "${1:-}" ;;
  validate) shift; cmd_validate "${1:-}" ;;
  verify)  shift; cmd_verify "$@" ;;
  test)    cmd_test ;;
  rebuild) cmd_rebuild ;;
  ""|-h|--help|help) usage ;;
  *) usage; fail "unknown command: $1" ;;
esac
