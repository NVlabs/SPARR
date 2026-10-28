#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA License.
# See the LICENSE file in the repository root for the full license text.

#
# Runs inside the Phase 2 container: puts both bind-mounted packages on PYTHONPATH
# (Phase 2 subclasses the Phase 1 collector). The vendored serl_launcher/agentlace are
# re-appended because the repo-root serl_launcher/ directory would shadow the package.

set -eo pipefail

SPARR_REPO=${SPARR_REPO:?SPARR_REPO must be set by session.sh}
PHASE2_SRC="${SPARR_REPO}/ros2_ws/src/sparr_phase2_residual"
PHASE1_SRC="${SPARR_REPO}/ros2_ws/src/sparr_phase1_collection"

for src in "${PHASE2_SRC}" "${PHASE1_SRC}"; do
  if [[ ! -d ${src} ]]; then
    echo "ERROR: live source not found at ${src}" >&2
    echo "       is the repository bind-mounted into the container?" >&2
    exit 1
  fi
done

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

VENDORED="/opt/sparr/serl_launcher:/opt/sparr/agentlace"

export PYTHONPATH="${PHASE2_SRC}:${PHASE1_SRC}${PYTHONPATH:+:${PYTHONPATH}}:${VENDORED}"

# JAX must not preallocate: the Torch base policy shares this GPU.
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}

exec "$@"
