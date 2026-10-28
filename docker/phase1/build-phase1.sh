#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA License.
# See the LICENSE file in the repository root for the full license text.

#
# Build the Phase 1 image (Ubuntu 26.04 / ROS 2 Lyrical) from public inputs.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
LOCK_FILE="${SCRIPT_DIR}/source-lock-phase1.env"

IMAGE_TAG=${SPARR_IMAGE_TAG:-sparr-phase1:latest}
BUILD_JOBS=${SPARR_BUILD_JOBS:-8}
BUILD_UTC=${SPARR_BUILD_UTC:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}

PACKAGE_DIR="${REPO_ROOT}/ros2_ws/src/sparr_phase1_collection"

fail() { echo "ERROR: $*" >&2; exit 1; }

[[ -f ${LOCK_FILE} ]] || fail "missing ${LOCK_FILE}"
# shellcheck disable=SC1090
source "${LOCK_FILE}"

command -v docker >/dev/null || fail "docker is required"
docker buildx version >/dev/null 2>&1 || fail "docker buildx is required"
[[ -f ${PACKAGE_DIR}/package.xml ]] || fail "missing package at ${PACKAGE_DIR}"

# Snapshot the package so edits during the build cannot skew the recorded source hash.
STAGING=$(mktemp -d /tmp/sparr-phase1-build.XXXXXX)
trap 'rm -rf "${STAGING}"' EXIT
PACKAGE_CONTEXT="${STAGING}/package"
mkdir -p "${PACKAGE_CONTEXT}"
tar -C "${PACKAGE_DIR}" \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
  --exclude='build' --exclude='install' --exclude='log' \
  --exclude='*.h5' --exclude='*.partial.h5' \
  -cf - . | tar -C "${PACKAGE_CONTEXT}" -xf -
SPARR_SOURCE_SHA256=$(cd "${PACKAGE_CONTEXT}" && find . -type f -print0 | sort -z |
  xargs -0 sha256sum | sha256sum | cut -d' ' -f1)

echo "building ${IMAGE_TAG}"
echo "  base            ${BASE_IMAGE_REF}   (public)"
echo "  ros             ${ROS_DISTRO_LOCK} on ${UBUNTU_CODENAME}"
echo "  torch           ${TORCH_VERSION}"
echo "  rsl_rl          ${RSL_RL_VERSION}"
echo "  GitPython       ${GITPYTHON_VERSION}"
echo "  pyrealsense2    ${PYREALSENSE2_VERSION}"
echo "  package sha256  ${SPARR_SOURCE_SHA256}"

exec docker buildx build --load --platform linux/amd64 \
  --build-context "sparr_package=${PACKAGE_CONTEXT}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE_REF}" \
  --build-arg "UBUNTU_CODENAME=${UBUNTU_CODENAME}" \
  --build-arg "ROS_DISTRO_LOCK=${ROS_DISTRO_LOCK}" \
  --build-arg "RMW_IMPLEMENTATION_LOCK=${RMW_IMPLEMENTATION_LOCK}" \
  --build-arg "BUILD_JOBS=${BUILD_JOBS}" \
  --build-arg "TORCH_INDEX_URL=${TORCH_INDEX_URL}" \
  --build-arg "TORCH_VERSION=${TORCH_VERSION}" \
  --build-arg "RSL_RL_VERSION=${RSL_RL_VERSION}" \
  --build-arg "GITPYTHON_VERSION=${GITPYTHON_VERSION}" \
  --build-arg "PYREALSENSE2_VERSION=${PYREALSENSE2_VERSION}" \
  --build-arg "FLEXIV_ROS2_REPOSITORY=${FLEXIV_ROS2_REPOSITORY}" \
  --build-arg "FLEXIV_ROS2_COMMIT=${FLEXIV_ROS2_COMMIT}" \
  --build-arg "FLEXIV_RDK_COMMIT=${FLEXIV_RDK_COMMIT}" \
  --build-arg "SPARR_SOURCE_SHA256=${SPARR_SOURCE_SHA256}" \
  --build-arg "BUILD_UTC=${BUILD_UTC}" \
  --tag "${IMAGE_TAG}" \
  --file "${SCRIPT_DIR}/Dockerfile.phase1" \
  "${SCRIPT_DIR}"
