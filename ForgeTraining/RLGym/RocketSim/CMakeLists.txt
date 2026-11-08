cmake_minimum_required (VERSION 3.10)

project("RocketSimV2")

# Add all headers and code files
file(GLOB_RECURSE FILES_ROCKETSIM "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/src/*.h")

# Only include bullet headers when using MSVC, otherwise just code files
file(GLOB_RECURSE FILES_BULLET "${PROJECT_SOURCE_DIR}/libsrc/bullet3-3.24/*.cpp" "${PROJECT_SOURCE_DIR}/libsrc/bullet3-3.24/*.h")

add_library(RocketSimV2 ${FILES_ROCKETSIM} ${FILES_BULLET})
set_target_properties(RocketSimV2 PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Bullet is vendored third-party code; silence its warnings rather than edit its numerics.
if (MSVC)
	set_source_files_properties(${FILES_BULLET} PROPERTIES COMPILE_OPTIONS "/wd4244;/wd4267;/wd4305;/wd4715")
endif()

target_link_libraries(RocketSimV2 PUBLIC RocketForgeSim)

set_target_properties(RocketSimV2 PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(RocketSimV2 PROPERTIES CXX_STANDARD 20)