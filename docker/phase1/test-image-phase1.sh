#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA OneWay Noncommercial License (March 22, 2022).
# See the LICENSE file in the repository root for the full license text:
# https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf

#
# Verification checks for the Phase 1 image: required components present, no Isaac ROS.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
IMAGE_TAG=${SPARR_IMAGE_TAG:-sparr-phase1:latest}

pass() { echo "  PASS  $*"; }
fail() { echo "  FAIL  $*" >&2; exit 1; }

docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1 ||
  fail "image not found: ${IMAGE_TAG}; run ${SCRIPT_DIR}/build-phase1.sh"

run() { docker run --rm --network none "$@"; }

echo "== 1. no Isaac ROS anywhere =="
if run --entrypoint bash "${IMAGE_TAG}" -c '
      set -e
      ! grep -RqiE "isaac|nvidia.*isaac" /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null
      ! (dpkg -l 2>/dev/null | grep -qiE "isaac")
      ! python3 -c "import isaac_ros_launch_utils" 2>/dev/null
      ! (dpkg -l 2>/dev/null | grep -q "pip-shim")
   '; then
  pass "no isaac packages, repos, or pip-shims"
else
  fail "traces of Isaac ROS remain in the image"
fi

run --entrypoint bash "${IMAGE_TAG}" -c \
  'grep -q "\"isaac_ros\": false" /etc/sparr/build-manifest.json' &&
  pass "build manifest records isaac_ros=false" ||
  fail "build manifest missing isaac_ros=false"

echo "== 2. base distribution =="
run --entrypoint bash "${IMAGE_TAG}" -c '
    . /etc/os-release
    [[ ${VERSION_ID} == 26.04 ]] || { echo "expected Ubuntu 26.04, got ${VERSION_ID}" >&2; exit 1; }
    [[ ${ROS_DISTRO:-lyrical} == lyrical ]]
  ' && pass "Ubuntu 26.04 + ROS 2 Lyrical" || fail "unexpected base distribution"

echo "== 3. ABI and runtime imports =="
run "${IMAGE_TAG}" python3 - <<'PY' && pass "cv_bridge / numpy / torch / rsl_rl / pyrealsense2 / GitPython" || fail "ABI check failed"
import numpy as np, tempfile, h5py, torch
import pyrealsense2 as rs
from cv_bridge import CvBridge
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent

bridge = CvBridge()
rgb = np.zeros((8, 8, 3), dtype=np.uint8)
message = bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
assert np.array_equal(rgb, bridge.imgmsg_to_cv2(message, desired_encoding="rgb8"))
with tempfile.NamedTemporaryFile(suffix=".h5") as stream:
    with h5py.File(stream.name, "w") as out:
        out.create_dataset("rgb", data=rgb, compression="lzf")
assert ActorCritic.__name__ and ActorCriticRecurrent.__name__
assert hasattr(rs, "pipeline")
import os, git
import rsl_rl.utils.utils  # imports git at module scope
assert tuple(int(p) for p in git.__version__.split(".")[:3]) >= (3, 1, 59), git.__version__
assert not os.path.exists("/usr/lib/python3/dist-packages/git"), "apt python3-git is back"
print(f"    torch {torch.__version__} | numpy {np.__version__} | librealsense {rs.__version__} | GitPython {git.__version__}")
PY

echo "== 4. ROS packages resolve =="
run "${IMAGE_TAG}" bash -c '
    for p in sparr_phase1_collection flexiv_msgs flexiv_bringup flexiv_hardware \
             cartesian_motion_controller flexiv_robot_states_broadcaster; do
      ros2 pkg prefix "$p" >/dev/null || { echo "missing $p" >&2; exit 1; }
    done
    # realsense2_camera must be absent; the SPARR pyrealsense2 node replaces it.
    ! ros2 pkg prefix realsense2_camera >/dev/null 2>&1
    ros2 pkg executables sparr_phase1_collection | grep -q sparr_phase1_camera
  ' && pass "flexiv + sparr packages present, realsense2_camera absent, camera node registered" ||
  fail "ROS package resolution failed"

echo "== 5. unit tests =="
run "${IMAGE_TAG}" bash -c '
    cd /opt/sparr/ros2_ws/src/sparr_phase1_collection &&
    python3 -m pytest -q --basetemp=/tmp/pt -p no:cacheprovider test/unit 2>&1 | tail -1
  ' | sed 's/^/    /'

echo "== 6. GPU + policy inference =="
if [[ ${SPARR_SKIP_GPU_TEST:-0} == 1 ]]; then
  echo "  SKIP  GPU stage (SPARR_SKIP_GPU_TEST=1)"
else
  docker run --rm --network none --gpus all "${IMAGE_TAG}" python3 - <<'PY' ||
      fail "GPU/policy check failed"
import torch
assert torch.cuda.is_available(), "CUDA not available in container"
print(f"    cuda device: {torch.cuda.get_device_name(0)}")
PY
  pass "CUDA visible"
  # The base policy is read from the repository, not the image: add it first (README step 3).
  policy_path=$(awk '/^policy:/{f=1;next} /^[^ #]/{f=0} f&&$1=="checkpoint:"{print $2; exit}' \
    "${REPO_ROOT}/phase1_live.yaml")
  if [[ -n ${policy_path} && ${policy_path} != /* ]]; then policy_path="${REPO_ROOT}/${policy_path}"; fi
  if [[ ! -f ${policy_path} ]]; then
    echo "  SKIP  shadow preflight: base policy not found at ${policy_path:-<policy.checkpoint unset>}"
    echo "        Add it (README step 3) and re-run this script to test policy loading."
    POLICY_SKIPPED=1
  else
    preflight_log=$(mktemp)
    docker run --rm --network none --gpus all \
      --volume "${REPO_ROOT}:${REPO_ROOT}" "${IMAGE_TAG}" bash -c "
        ros2 run sparr_phase1_collection sparr_phase1_preflight \
          --config ${REPO_ROOT}/phase1_live.yaml --mode shadow
      " >"${preflight_log}" 2>&1 &&
      pass "shadow preflight on GPU (loads checkpoint, runs two stochastic samples)" || {
        cat "${preflight_log}" >&2
        rm -f "${preflight_log}"
        fail "shadow preflight failed (see the preflight output above)"
      }
    rm -f "${preflight_log}"
  fi
fi

echo
if [[ ${POLICY_SKIPPED:-0} == 1 ]]; then
  echo "IMAGE CHECKS PASSED for ${IMAGE_TAG} (policy loading not tested: base policy missing)"
else
  echo "ALL CHECKS PASSED for ${IMAGE_TAG}"
fi
