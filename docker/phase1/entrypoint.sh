#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#
# Entrypoint for the Phase 1 image: refuse an unexpected ROS distro or RMW,
# then source ROS, the Flexiv workspace, and the SPARR install in order.

set -eo pipefail

SPARR_ROS_DISTRO_LOCK=${SPARR_ROS_DISTRO_LOCK:-lyrical}
if [[ ${SPARR_ROS_DISTRO_LOCK} != lyrical ]]; then
  echo "ERROR: this image recipe only supports ROS 2 Lyrical" >&2
  exit 1
fi
if [[ -n ${ROS_DISTRO:-} && ${ROS_DISTRO} != "${SPARR_ROS_DISTRO_LOCK}" ]]; then
  echo "ERROR: SPARR image requires ROS 2 ${SPARR_ROS_DISTRO_LOCK}, found ${ROS_DISTRO}" >&2
  exit 1
fi

SPARR_RMW_IMPLEMENTATION_LOCK=${SPARR_RMW_IMPLEMENTATION_LOCK:-rmw_fastrtps_cpp}
if [[ -n ${RMW_IMPLEMENTATION:-} &&
      ${RMW_IMPLEMENTATION} != "${SPARR_RMW_IMPLEMENTATION_LOCK}" ]]; then
  echo "ERROR: SPARR image requires ${SPARR_RMW_IMPLEMENTATION_LOCK}, found ${RMW_IMPLEMENTATION}" >&2
  exit 1
fi
export RMW_IMPLEMENTATION=${SPARR_RMW_IMPLEMENTATION_LOCK}
export FLEXIV_RDK_PREFIX=${FLEXIV_RDK_PREFIX:-/opt/sparr/flexiv_rdk}
export HOME=${HOME:-/tmp/sparr_home}
mkdir -p "${HOME}"

had_nounset=0
case $- in
  *u*) had_nounset=1 ;;
esac
set +u
# shellcheck disable=SC1090
source "/opt/ros/${SPARR_ROS_DISTRO_LOCK}/setup.bash"
# shellcheck disable=SC1091
source /opt/sparr/flexiv_ws/install/setup.bash
# shellcheck disable=SC1091
source /opt/sparr/install/setup.bash
export LD_LIBRARY_PATH="${FLEXIV_RDK_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ ${had_nounset} == 1 ]]; then
  set -u
fi
unset had_nounset

exec "$@"
