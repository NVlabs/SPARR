#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA OneWay Noncommercial License (March 22, 2022).
# See the LICENSE file in the repository root for the full license text:
# https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf

"""Adapt the pinned Flexiv fork to current ROS 2 Lyrical.

Each patch is behaviour-preserving and verifies that it applied, so a silent
no-op fails the build.

Usage: patch-flexiv-lyrical.py <workspace-src-dir>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SHIM_INCLUDE = "include(/opt/sparr/cmake/ament_target_dependencies_shim.cmake)"
FIND_AMENT = "find_package(ament_cmake REQUIRED)"

# target_link_libraries(<target>\n  dep...  -- the plain (non-keyword) form.
PLAIN_TLL = re.compile(r"^target_link_libraries\(([^ )]+)[ \t]*$", re.MULTILINE)

BROADCASTER_REL = (
    "flexiv_controllers/flexiv_robot_states_broadcaster/src/"
    "flexiv_robot_states_broadcaster.cpp"
)
COPY_CONSTRUCTION = (
    "std::make_unique<semantic_components::FlexivRobotStates>(\n"
    "            semantic_components::FlexivRobotStates(robot_sn))"
)
IN_PLACE_CONSTRUCTION = "std::make_unique<semantic_components::FlexivRobotStates>(robot_sn)"


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def patch_cmakelists(root: Path) -> int:
    """Include the ament_target_dependencies shim and normalise plain links to PUBLIC.

    The shim uses the keyword link signature, and CMake refuses to mix forms on one target.
    """
    patched = 0
    for cmakelists in sorted(root.rglob("CMakeLists.txt")):
        text = cmakelists.read_text()
        if "ament_target_dependencies" not in text:
            continue
        if FIND_AMENT not in text:
            fail(f"{cmakelists}: no '{FIND_AMENT}' to anchor the shim include")

        text = text.replace(FIND_AMENT, f"{FIND_AMENT}\n{SHIM_INCLUDE}", 1)
        text = PLAIN_TLL.sub(r"target_link_libraries(\1 PUBLIC", text)
        cmakelists.write_text(text)

        verify = cmakelists.read_text()
        if SHIM_INCLUDE not in verify:
            fail(f"{cmakelists}: shim include did not apply")
        if PLAIN_TLL.search(verify):
            fail(f"{cmakelists}: a plain target_link_libraries call survived")
        patched += 1

    if patched == 0:
        fail("no CMakeLists.txt used ament_target_dependencies; has the fork changed?")
    return patched


def patch_broadcaster(root: Path) -> None:
    """Construct FlexivRobotStates in place; SemanticComponentInterface is no longer copyable."""
    source_path = root / "flexiv_ros2" / BROADCASTER_REL
    if not source_path.is_file():
        fail(f"broadcaster source not found at {source_path}")

    source = source_path.read_text()
    if IN_PLACE_CONSTRUCTION in source and COPY_CONSTRUCTION not in source:
        return  # already in place
    if COPY_CONSTRUCTION not in source:
        fail(f"{source_path}: expected copy-construction pattern not found")

    source_path.write_text(source.replace(COPY_CONSTRUCTION, IN_PLACE_CONSTRUCTION))
    if IN_PLACE_CONSTRUCTION not in source_path.read_text():
        fail(f"{source_path}: in-place construction patch did not apply")


YAML_CPP_INSTALLER_REL = "flexiv_hardware/rdk/thirdparty/scripts/install_yaml-cpp.sh"
YAML_CPP_ANCHOR = "# Configure CMake"
YAML_CPP_FIX = """# Added by patch-flexiv-lyrical.py: yaml-cpp 0.7.0 uses uint16_t in
# src/emitterutils.cpp without including <cstdint>. Newer libstdc++ no longer
# pulls that header in transitively, so the declaration of
# EncodeUTF16SurrogatePair fails and its callers then report the confusing
# "request for member 'second' in ... non-class type 'int'".
grep -q '#include <cstdint>' src/emitterutils.cpp || \\
  sed -i '1i #include <cstdint>' src/emitterutils.cpp

# Configure CMake"""


def patch_yaml_cpp_installer(root: Path) -> None:
    """Add the missing <cstdint> include to the vendored yaml-cpp (skipped if not vendored)."""
    installer = root / "flexiv_ros2" / YAML_CPP_INSTALLER_REL
    if not installer.is_file():
        thirdparty = installer.parent
        if thirdparty.is_dir():
            print(
                f"skipping yaml-cpp patch: this RDK does not vendor yaml-cpp "
                f"({thirdparty} has {sorted(p.name for p in thirdparty.iterdir())})"
            )
            return
        fail(
            f"yaml-cpp installer not found at {installer}, and neither is its "
            f"directory -- the RDK layout is not one this script understands"
        )

    text = installer.read_text()
    if "#include <cstdint>" in text:
        return  # already patched
    if YAML_CPP_ANCHOR not in text:
        fail(f"{installer}: no '{YAML_CPP_ANCHOR}' anchor to insert the fix before")

    installer.write_text(text.replace(YAML_CPP_ANCHOR, YAML_CPP_FIX, 1))
    if "#include <cstdint>" not in installer.read_text():
        fail(f"{installer}: yaml-cpp cstdint patch did not apply")


STATIC_DEPS_REL = "scripts/build_and_install_rdk_dependencies_static.sh"
STATIC_DEPS_LOOP = """for dependency_script in "${DEPENDENCY_SCRIPTS[@]}"; do
    bash "${SCRIPT_DIR}/scripts/${dependency_script}"
done"""
STATIC_DEPS_FIX = """for dependency_script in "${DEPENDENCY_SCRIPTS[@]}"; do
    # Added by patch-flexiv-lyrical.py: RDK v1.9.4 stopped vendoring most of
    # these -- thirdparty/scripts holds only install_eigen.sh, the rest now
    # ship inside the prebuilt library. This list was written for v1.9.2, where
    # all of them existed, so a missing installer means "no longer needed",
    # not "broken checkout". Skip it and say so; the RDK's own CMake still
    # fails loudly if a dependency it actually needs is absent.
    if [[ ! -f "${SCRIPT_DIR}/scripts/${dependency_script}" ]]; then
        echo "skipping ${dependency_script}: not vendored by this RDK version"
        continue
    fi
    bash "${SCRIPT_DIR}/scripts/${dependency_script}"
done"""


def patch_static_dependency_list(root: Path) -> None:
    """Skip dependency installers that the pinned RDK no longer vendors."""
    script = root / "flexiv_ros2" / STATIC_DEPS_REL
    if not script.is_file():
        print(f"skipping static-dependency patch: {script} is absent")
        return
    text = script.read_text()
    if "not vendored by this RDK version" in text:
        return  # already patched
    if STATIC_DEPS_LOOP not in text:
        fail(f"{script}: dependency loop not in the expected form")
    script.write_text(text.replace(STATIC_DEPS_LOOP, STATIC_DEPS_FIX, 1))
    if "not vendored by this RDK version" not in script.read_text():
        fail(f"{script}: static-dependency patch did not apply")


CARTESIAN_LAUNCH_REL = "flexiv_bringup/launch/rizon_cartesian.launch.py"
SPAWNER_WITHOUT_PARAMS = (
    'arguments=["cartesian_motion_controller", "--controller-manager", "/controller_manager"],'
)
SPAWNER_WITH_PARAMS = (
    'arguments=["cartesian_motion_controller", "--controller-manager", "/controller_manager"],\n'
    '        # Current ros2_control no longer forwards the controller_manager\'s own\n'
    '        # parameter file down to each controller, so tcp_name arrived unset and\n'
    '        # the controller refused to initialise. Supplying it on the spawner\n'
    '        # mirrors how the states broadcaster above already passes robot_sn --\n'
    '        # the one spawner in this file that kept working. Passing the yaml via\n'
    '        # --param-file does NOT work here: the spawner reads that path itself,\n'
    '        # so $(var robot_sn) is never expanded and the controller then looks\n'
    '        # for a command interface literally named "$(var robot_sn)_tcp".\n'
    '        parameters=[{"tcp_name": ParameterValue([robot_sn, "_tcp"], value_type=str)}],'
)


def patch_cartesian_spawner(root: Path) -> None:
    """Pass tcp_name to the Cartesian controller on its spawner.

    ros2_control no longer forwards the controller_manager parameter file to controllers.
    """
    launch = root / "flexiv_ros2" / CARTESIAN_LAUNCH_REL
    if not launch.is_file():
        fail(f"cartesian launch file not found at {launch}")

    text = launch.read_text()
    if '"tcp_name"' in text:
        return  # already patched
    if SPAWNER_WITHOUT_PARAMS not in text:
        fail(f"{launch}: expected cartesian spawner arguments not found")
    if "ParameterValue" not in text:
        fail(f"{launch}: ParameterValue is not imported; cannot supply tcp_name")

    launch.write_text(text.replace(SPAWNER_WITHOUT_PARAMS, SPAWNER_WITH_PARAMS, 1))
    if '"tcp_name"' not in launch.read_text():
        fail(f"{launch}: cartesian spawner tcp_name patch did not apply")


def main() -> int:
    if len(sys.argv) != 2:
        fail("usage: patch-flexiv-lyrical.py <workspace-src-dir>")
    root = Path(sys.argv[1])
    if not root.is_dir():
        fail(f"not a directory: {root}")

    count = patch_cmakelists(root)
    patch_broadcaster(root)
    patch_yaml_cpp_installer(root)
    patch_static_dependency_list(root)
    patch_cartesian_spawner(root)
    print(
        f"patched {count} CMakeLists.txt files, the states broadcaster, the "
        "vendored yaml-cpp, and the cartesian spawner for Lyrical"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
