# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA License.
# See the LICENSE file in the repository root for the full license text.

# Shim for ament_target_dependencies (removed in ROS 2 Lyrical), still called by the Flexiv fork.
# Links ${pkg_TARGETS}, then pkg::pkg, then the legacy _INCLUDE_DIRS/_LIBRARIES variables.

if(NOT COMMAND ament_target_dependencies)
  macro(ament_target_dependencies _atd_target)
    if(NOT TARGET ${_atd_target})
      message(FATAL_ERROR
        "ament_target_dependencies shim: '${_atd_target}' is not a target")
    endif()

    set(_atd_scope "PUBLIC")
    set(_atd_deps "")
    foreach(_atd_arg ${ARGN})
      if(_atd_arg STREQUAL "PUBLIC" OR _atd_arg STREQUAL "PRIVATE"
         OR _atd_arg STREQUAL "INTERFACE")
        set(_atd_scope "${_atd_arg}")
      elseif(_atd_arg STREQUAL "SYSTEM")
        # SYSTEM is dropped; it only affected warning visibility.
      else()
        list(APPEND _atd_deps "${_atd_arg}")
      endif()
    endforeach()

    foreach(_atd_dep ${_atd_deps})
      if(NOT ${_atd_dep}_FOUND)
        find_package(${_atd_dep} REQUIRED)
      endif()

      if(DEFINED ${_atd_dep}_TARGETS)
        # rosidl-generated interface packages export typesupport libraries here.
        target_link_libraries(${_atd_target} ${_atd_scope} ${${_atd_dep}_TARGETS})
      elseif(TARGET ${_atd_dep}::${_atd_dep})
        target_link_libraries(${_atd_target} ${_atd_scope} ${_atd_dep}::${_atd_dep})
      else()
        if(${_atd_dep}_INCLUDE_DIRS)
          target_include_directories(${_atd_target} ${_atd_scope}
            ${${_atd_dep}_INCLUDE_DIRS})
        endif()
        if(${_atd_dep}_LIBRARIES)
          target_link_libraries(${_atd_target} ${_atd_scope}
            ${${_atd_dep}_LIBRARIES})
        endif()
        if(NOT ${_atd_dep}_INCLUDE_DIRS AND NOT ${_atd_dep}_LIBRARIES)
          message(WARNING
            "ament_target_dependencies shim: '${_atd_dep}' exposed no target "
            "and no legacy variables; nothing was linked for it")
        endif()
      endif()
    endforeach()

    unset(_atd_scope)
    unset(_atd_deps)
  endmacro()
endif()
