cmake_minimum_required(VERSION 3.10)

project("RocketForgeSim")

file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h" "include/*.h")

# The v3 backend is compiled only when RocketSimV3 found its source and a Rust toolchain.
if(NOT ROCKETFORGE_V3_AVAILABLE)
    list(FILTER FILES_SRC EXCLUDE REGEX "/src/Backend/V3Backend[.](cpp|h)$")
endif()

add_library(RocketForgeSim STATIC ${FILES_SRC})

set_target_properties(RocketForgeSim PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(RocketForgeSim PROPERTIES CXX_STANDARD 20)
set_target_properties(RocketForgeSim PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(RocketForgeSim PUBLIC "include")

# The v2 backend wraps the vendored engine, which in turn uses the value types here; CMake links the two static libraries as a cycle.
target_link_libraries(RocketForgeSim PRIVATE RocketSimV2)

if(ROCKETFORGE_V3_AVAILABLE)
    target_link_libraries(RocketForgeSim PRIVATE RocketSimV3Ffi)
    set(RF_ENGINE_V3_AVAILABLE 1)
    set(RF_ENGINE_V3_REASON "")
else()
    set(RF_ENGINE_V3_AVAILABLE 0)

    # The reason is spliced into a C string literal.
    string(REPLACE "\\" "\\\\" RF_ENGINE_V3_REASON "${ROCKETSIM_V3_REASON}")
    string(REPLACE "\"" "\\\"" RF_ENGINE_V3_REASON "${RF_ENGINE_V3_REASON}")
endif()
set(RF_ENGINE_V3_REVISION "${ROCKETSIM_V3_REVISION}")

# Compiled code reaches the meshes only through RocketSim::CollisionMeshesFolder() and CollisionMeshesRelativePath();
# the submittal cpp.Dockerfile template is the one other place that names this path.
set(RF_COLLISION_MESHES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/collision_meshes")
file(RELATIVE_PATH RF_COLLISION_MESHES_RELATIVE_PATH "${CMAKE_SOURCE_DIR}" "${RF_COLLISION_MESHES_DIR}")

# tools/build/build_and_test.* print this line; the file lets them show it again on a build that skips the configure.
if(ROCKETFORGE_V3_AVAILABLE)
    set(_rf_engines_line "Engines: v2 (built), v3 (built)")
else()
    string(REGEX REPLACE "^RocketSim v3 isn't built: " "" _rf_v3_why "${ROCKETSIM_V3_REASON}")
    set(_rf_engines_line "Engines: v2 (built), v3 (not built: ${_rf_v3_why})")
endif()
message(STATUS "${_rf_engines_line}")
file(WRITE "${CMAKE_BINARY_DIR}/engines.txt" "${_rf_engines_line}\n")
configure_file(EngineConfig.h.in "${CMAKE_CURRENT_BINARY_DIR}/generated/EngineConfig.h" @ONLY)
target_include_directories(RocketForgeSim PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
