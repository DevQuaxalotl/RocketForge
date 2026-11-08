# Decides on every configure whether RocketSim v3 (a Rust crate in a git submodule) can be built.
# Sets in the caller's scope: ROCKETFORGE_V3_AVAILABLE, ROCKETSIM_V3_REASON (why not, when unavailable),
# ROCKETSIM_V3_REVISION, and CARGO_EXECUTABLE (when available).

set(ROCKETFORGE_ENABLE_V3 "AUTO" CACHE STRING "Build RocketSim v3 (AUTO: when its source and Rust are available; ON: required; OFF: never)")
set_property(CACHE ROCKETFORGE_ENABLE_V3 PROPERTY STRINGS AUTO ON OFF)

# Test-only switches for tools/build/test_v3_detection.ps1; they make detection behave as if the dependency were missing.
option(ROCKETFORGE_V3_TEST_HIDE_CARGO "Test-only: act as if cargo is not installed" OFF)
option(ROCKETFORGE_V3_TEST_HIDE_SUBMODULE "Test-only: act as if the v3 submodule is empty and cannot be fetched" OFF)
mark_as_advanced(ROCKETFORGE_V3_TEST_HIDE_CARGO ROCKETFORGE_V3_TEST_HIDE_SUBMODULE)

set(ROCKETFORGE_V3_MIN_RUST_VERSION "1.95")
set(_rf_v3_submodule_path "ForgeTraining/RLGym/RocketSimV3/upstream")

function(RocketForgeDetectRocketSimV3 upstreamDir)
    set(available FALSE)
    set(reason "")
    set(revision "")
    set(cargo "")

    string(TOUPPER "${ROCKETFORGE_ENABLE_V3}" mode)
    if(NOT mode MATCHES "^(AUTO|ON|OFF)$")
        message(FATAL_ERROR "ROCKETFORGE_ENABLE_V3 must be AUTO, ON or OFF (got '${ROCKETFORGE_ENABLE_V3}').")
    endif()

    set(cargoManifest "${upstreamDir}/rocketsim/Cargo.toml")

    if(mode STREQUAL "OFF")
        set(reason "RocketSim v3 is turned off in this build (ROCKETFORGE_ENABLE_V3=OFF).")
    endif()

    # Submodule
    if(NOT reason AND (ROCKETFORGE_V3_TEST_HIDE_SUBMODULE OR NOT EXISTS "${cargoManifest}"))
        if(NOT ROCKETFORGE_V3_TEST_HIDE_SUBMODULE AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
            find_package(Git QUIET)
            if(GIT_FOUND)
                message(STATUS "Fetching the RocketSim v3 submodule...")
                execute_process(COMMAND "${GIT_EXECUTABLE}" submodule update --init "${_rf_v3_submodule_path}"
                    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" OUTPUT_QUIET ERROR_QUIET)
            endif()
        endif()

        if(ROCKETFORGE_V3_TEST_HIDE_SUBMODULE OR NOT EXISTS "${cargoManifest}")
            set(reason "RocketSim v3 isn't built: its source (the ${_rf_v3_submodule_path} submodule) is missing. Run \"git submodule update --init ${_rf_v3_submodule_path}\", then re-run CMake.")
        endif()
    endif()

    # Cargo
    if(NOT reason)
        if(NOT ROCKETFORGE_V3_TEST_HIDE_CARGO)
            find_program(foundCargo cargo NO_CACHE HINTS "$ENV{USERPROFILE}/.cargo/bin" "$ENV{HOME}/.cargo/bin")
        endif()

        set(installHint "Run tools/setup/install_rust.ps1 (Windows) or tools/setup/install_rust.sh (Linux), then rebuild.")
        if(ROCKETFORGE_V3_TEST_HIDE_CARGO OR NOT foundCargo)
            set(reason "RocketSim v3 isn't built: Rust isn't installed. ${installHint}")
        else()
            set(toolchainArg "")
            if(ROCKETSIM_V3_RUST_TOOLCHAIN)
                set(toolchainArg "+${ROCKETSIM_V3_RUST_TOOLCHAIN}")
            endif()

            # Run from the bindings folder so rustup applies the toolchain pinned there.
            execute_process(COMMAND "${foundCargo}" ${toolchainArg} --version
                WORKING_DIRECTORY "${upstreamDir}/../bindings"
                RESULT_VARIABLE versionResult OUTPUT_VARIABLE versionText OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
            if(NOT versionResult EQUAL 0 OR NOT versionText MATCHES "cargo ([0-9]+[.][0-9]+([.][0-9]+)?)")
                set(reason "RocketSim v3 isn't built: Rust isn't installed. ${installHint}")
            elseif(CMAKE_MATCH_1 VERSION_LESS ROCKETFORGE_V3_MIN_RUST_VERSION)
                set(reason "RocketSim v3 isn't built: Rust ${CMAKE_MATCH_1} is older than the required ${ROCKETFORGE_V3_MIN_RUST_VERSION}. ${installHint}")
            else()
                set(cargo "${foundCargo}")
            endif()
        endif()
    endif()

    if(reason AND mode STREQUAL "ON")
        message(FATAL_ERROR "${reason}")
    elseif(reason AND mode STREQUAL "AUTO")
        message(WARNING "${reason}")
    endif()

    # Revision
    if(NOT reason)
        set(available TRUE)

        # A directory that isn't its own repository would report the enclosing repository's commit.
        if(EXISTS "${upstreamDir}/.git")
            find_package(Git QUIET)
            if(GIT_FOUND)
                execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD WORKING_DIRECTORY "${upstreamDir}"
                    RESULT_VARIABLE revisionResult OUTPUT_VARIABLE revisionText OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
                if(revisionResult EQUAL 0)
                    set(revision "${revisionText}")
                endif()
            endif()
        endif()

        if(NOT revision)
            file(STRINGS "${cargoManifest}" versionLines REGEX "^version[ \t]*=")
            if(versionLines MATCHES "version[ \t]*=[ \t]*\"([^\"]+)\"")
                set(revision "crate ${CMAKE_MATCH_1}")
            else()
                set(revision "unknown")
            endif()
        endif()
    endif()

    set(ROCKETFORGE_V3_AVAILABLE ${available} PARENT_SCOPE)
    set(ROCKETSIM_V3_REASON "${reason}" PARENT_SCOPE)
    set(ROCKETSIM_V3_REVISION "${revision}" PARENT_SCOPE)
    set(CARGO_EXECUTABLE "${cargo}" PARENT_SCOPE)
endfunction()
