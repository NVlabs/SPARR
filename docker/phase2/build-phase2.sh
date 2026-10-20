#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#
# Build the Phase 2 (residual RL) image as a layer on the Phase 1 image.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
LOCK_FILE="${SCRIPT_DIR}/source-lock-phase2.env"

IMAGE_TAG=${SPARR_PHASE2_IMAGE_TAG:-sparr-phase2:latest}

fail() { echo "ERROR: $*" >&2; exit 1; }

[[ -f ${LOCK_FILE} ]] || fail "missing ${LOCK_FILE}"
# shellcheck disable=SC1090
source "${LOCK_FILE}"

command -v docker >/dev/null || fail "docker is required"
docker image inspect "${BASE_IMAGE_REF}" >/dev/null 2>&1 \
  || fail "base image ${BASE_IMAGE_REF} not found; build Phase 1 first with docker/phase1/build-phase1.sh"

[[ -f ${REPO_ROOT}/${SERL_LAUNCHER_PATH}/setup.py ]] || fail "missing ${SERL_LAUNCHER_PATH}"
[[ -f ${REPO_ROOT}/${AGENTLACE_PATH}/setup.py ]] || fail "missing ${AGENTLACE_PATH}"

echo "Building ${IMAGE_TAG}"
echo "  base            ${BASE_IMAGE_REF}"
echo "  jax             ${JAX_VERSION} [${JAX_CUDA_EXTRA}]"
echo "  flax            ${FLAX_VERSION}"
echo "  agentlace       ${AGENTLACE_UPSTREAM_COMMIT:0:7} (vendored)"

build_args=()
for name in JAX_VERSION JAX_CUDA_EXTRA FLAX_VERSION DISTRAX_VERSION OPTAX_VERSION \
            CHEX_VERSION ORBAX_CHECKPOINT_VERSION EINOPS_VERSION \
            GYMNASIUM_VERSION MATPLOTLIB_VERSION LZ4_VERSION PYZMQ_VERSION TENSORBOARD_VERSION \
            RESNET10_URL RESNET10_SHA256 BASE_IMAGE_REF; do
  build_args+=(--build-arg "${name}=${!name}")
done

docker build \
  --file "${SCRIPT_DIR}/Dockerfile.phase2" \
  --tag "${IMAGE_TAG}" \
  "${build_args[@]}" \
  "${REPO_ROOT}"

echo
echo "Built ${IMAGE_TAG}. Verify it with:"
echo "  docker run --rm --gpus all -v ${REPO_ROOT}:${REPO_ROOT} -w ${REPO_ROOT} \\"
echo "    ${IMAGE_TAG} python3 /opt/sparr/verify-phase2-env.py"
