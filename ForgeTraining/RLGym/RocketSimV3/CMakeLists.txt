cmake_minimum_required(VERSION 3.20)
project(RocketSimV3 LANGUAGES C CXX)

include("${CMAKE_SOURCE_DIR}/cmake/DetectRocketSimV3.cmake")

set(ROCKETSIM_V3_RUST_TOOLCHAIN "" CACHE STRING "Overrides the Rust toolchain pinned in bindings/rust-toolchain.toml (requires Rust 1.95+)")
RocketForgeDetectRocketSimV3("${CMAKE_CURRENT_SOURCE_DIR}/upstream")

# The Sim layer next door reads these to build its engine description.
set(ROCKETFORGE_V3_AVAILABLE "${ROCKETFORGE_V3_AVAILABLE}" PARENT_SCOPE)
set(ROCKETSIM_V3_REASON "${ROCKETSIM_V3_REASON}" PARENT_SCOPE)
set(ROCKETSIM_V3_REVISION "${ROCKETSIM_V3_REVISION}" PARENT_SCOPE)
if(NOT ROCKETFORGE_V3_AVAILABLE)
    return()
endif()

# Without an override, rustup picks the pin from bindings/rust-toolchain.toml;
# it looks for that file from the working directory, not from --manifest-path.
set(_cargo_toolchain)
if(ROCKETSIM_V3_RUST_TOOLCHAIN)
    set(_cargo_toolchain "+${ROCKETSIM_V3_RUST_TOOLCHAIN}")
endif()

# The submittal export vendors the crates with the same cargo.
set_property(GLOBAL PROPERTY ROCKETSIM_V3_CARGO_EXECUTABLE "${CARGO_EXECUTABLE}")

# Clang target triples name the matching Rust targets, so a cross-compiling toolchain file selects cargo's target too.
set(_rust_target_default "")
if(CMAKE_CROSSCOMPILING AND CMAKE_CXX_COMPILER_TARGET)
    set(_rust_target_default "${CMAKE_CXX_COMPILER_TARGET}")
endif()

set(ROCKETSIM_V3_RUST_TARGET "${_rust_target_default}" CACHE STRING "Rust target triple for the v3 bindings (empty: the host)")

# One cargo build serves every C++ configuration: Rust's release profile is always used, and the C ABI owns its allocations.
set(_cargo_dir "${CMAKE_CURRENT_BINARY_DIR}/cargo")
set(_cargo_target_args)
set(_cargo_output_dir "${_cargo_dir}/release")
if(ROCKETSIM_V3_RUST_TARGET)
    set(_cargo_target_args --target "${ROCKETSIM_V3_RUST_TARGET}")
    set(_cargo_output_dir "${_cargo_dir}/${ROCKETSIM_V3_RUST_TARGET}/release")
endif()

if(WIN32)
    set(_rust_lib "${_cargo_output_dir}/rocketsim_ffi.lib")
else()
    set(_rust_lib "${_cargo_output_dir}/librocketsim_ffi.a")
endif()

add_custom_target(RocketSimV3Cargo
    COMMAND "${CARGO_EXECUTABLE}" ${_cargo_toolchain} build --locked --release ${_cargo_target_args}
        --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/bindings/Cargo.toml" --target-dir "${_cargo_dir}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bindings"
    BYPRODUCTS "${_rust_lib}"
    VERBATIM)

add_library(RocketSimV3Ffi STATIC IMPORTED GLOBAL)
set_target_properties(RocketSimV3Ffi PROPERTIES IMPORTED_LOCATION "${_rust_lib}")
target_include_directories(RocketSimV3Ffi INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/bindings/include")
add_dependencies(RocketSimV3Ffi RocketSimV3Cargo)

if(WIN32)
    target_link_libraries(RocketSimV3Ffi INTERFACE ntdll userenv ws2_32 dbghelp bcrypt)
elseif(APPLE)
    target_link_libraries(RocketSimV3Ffi INTERFACE "-framework CoreFoundation" "-framework Security")
elseif(UNIX)
    target_link_libraries(RocketSimV3Ffi INTERFACE pthread dl m)
endif()

if(ROCKETFORGE_BUILD_TESTS)
    add_library(RocketSimV3CHeaderCheck OBJECT bindings/tests/header_smoke.c)
    target_include_directories(RocketSimV3CHeaderCheck PRIVATE bindings/include)
    set_target_properties(RocketSimV3CHeaderCheck PROPERTIES C_STANDARD 11)
    add_dependencies(RocketSimV3Cargo RocketSimV3CHeaderCheck)
endif()
