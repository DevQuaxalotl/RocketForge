cmake_minimum_required (VERSION 3.10)

# libtorch's imported targets force /Z7 on their consumers, so use it everywhere instead of CMake's default /Zi.
if(POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>")
endif()

# The Windows presets build this path from $env{CUDA_PATH_V12_6}, which has backslashes.
# CMake before 4.0 writes it unescaped into the generated CMakeCUDACompiler.cmake, which then fails to parse.
if(DEFINED CMAKE_CUDA_COMPILER)
    file(TO_CMAKE_PATH "${CMAKE_CUDA_COMPILER}" _CUDA_COMPILER_PATH)
    set(CMAKE_CUDA_COMPILER "${_CUDA_COMPILER_PATH}" CACHE FILEPATH "CUDA compiler" FORCE)
endif()

project("RocketForge")

set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

include(cmake/FetchRLBotCppInterface.cmake)
RocketForgeFetchRLBotCppInterface()

option(ROCKETFORGE_BUILD_TESTS "Build the ForgeTraining unit tests" OFF)
option(RLBOT_BUILD_TESTS "Build the RLBot runtime tests (with RLBOT=ON)" OFF)

# Not cached, so a changed QTDIR takes effect on the next configure.
# The fallback is where tools/setup/install_qt.ps1 installs by default.
if(WIN32 AND NOT CMAKE_PREFIX_PATH)
    if(DEFINED ENV{QTDIR})
        set(CMAKE_PREFIX_PATH "$ENV{QTDIR}")
    else()
        set(CMAKE_PREFIX_PATH "C:/Qt/6.10.2/msvc2022_64")
    endif()
endif()

if(ROCKETFORGE_BUILD_TESTS OR RLBOT_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
        URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    # Match the shared MSVC runtime that libtorch and the rest of the build use.
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    include(GoogleTest)
    enable_testing()
endif()

# For RLBot only build what is needed for RLBot (GigaLearnInfer, Agents, and RLBot)
option(RLBOT "Build RLBot" OFF)
if(RLBOT)

    # Include RLGym for GigaLearnInfer
    add_subdirectory(ForgeTraining/RLGym)

    # Include only GigaLearnInfer for RLBot
    add_subdirectory(ForgeTraining/GigaLearn/GigaLearnInfer)

    # Include Bots (creates Agents + TrainingPlans libs; each bot handles RLBOT conditional)
    add_subdirectory(Bots)

    # Add framework-provided component registrations after the Agents target exists.
    add_subdirectory(ForgeTraining/GigaLearn/DefaultAgentComponents)

    # Include RLBot support
    add_subdirectory(RLBot)

else()

    # Include ForgeTraining for all libraries needed for training
    add_subdirectory(ForgeTraining)

    # Include Bots (creates Agents + TrainingPlans libs; each bot handles RLBOT conditional)
    add_subdirectory(Bots)

    # Add framework-provided component registrations after the Agents target exists.
    add_subdirectory(ForgeTraining/GigaLearn/DefaultAgentComponents)

    # Include ForgeConsoleTrainer for training via console
    add_subdirectory(ForgeConsoleTrainer)

    set(BUILD_GUI "AUTO" CACHE STRING "Build ForgeGUI (AUTO: when Qt 6 is found; ON: required; OFF: never)")
    set_property(CACHE BUILD_GUI PROPERTY STRINGS AUTO ON OFF)
    if(BUILD_GUI STREQUAL "AUTO")
        find_package(Qt6 QUIET COMPONENTS Widgets Sql)
        if(Qt6_FOUND)
            set(_rf_build_gui ON)
            message(STATUS "GUI: built (Qt ${Qt6_VERSION} at ${Qt6_DIR})")
        else()
            set(_rf_build_gui OFF)
            message(STATUS "GUI: not built (Qt 6 not found; install it with tools/setup/install_qt.ps1 or set QTDIR)")
        endif()
    elseif(BUILD_GUI)
        set(_rf_build_gui ON)
    else()
        set(_rf_build_gui OFF)
        message(STATUS "GUI: not built (BUILD_GUI=OFF)")
    endif()

    if(_rf_build_gui)
        add_subdirectory(ForgeGUI)
    endif()

endif()

add_subdirectory(tools/rlbot/parity_probe)

if(ROCKETFORGE_BUILD_TESTS)
    add_subdirectory(ForgeTrainingTests)
endif()
