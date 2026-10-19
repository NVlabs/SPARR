#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#
# Runs inside the Phase 1 container: puts the bind-mounted package ahead of the baked-in copy.
# Launch files and share/ still come from the image or the `session.sh rebuild` overlay.

set -eo pipefail

SPARR_REPO=${SPARR_REPO:?SPARR_REPO must be set by session.sh}
LIVE_SRC="${SPARR_REPO}/ros2_ws/src/sparr_phase1_collection"

if [[ ! -d ${LIVE_SRC} ]]; then
  echo "ERROR: live source not found at ${LIVE_SRC}" >&2
  echo "       is the repository bind-mounted into the container?" >&2
  exit 1
fi

# Source the overlay first; it prepends to PYTHONPATH and would shadow the live source.
OVERLAY="${SPARR_REPO}/ros2_ws/install/setup.bash"
if [[ -f ${OVERLAY} ]]; then
  had_nounset=0
  case $- in *u*) had_nounset=1 ;; esac
  set +u
  # shellcheck disable=SC1090
  source "${OVERLAY}"
  [[ ${had_nounset} == 1 ]] && set -u
fi

export PYTHONPATH="${LIVE_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

exec "$@"
