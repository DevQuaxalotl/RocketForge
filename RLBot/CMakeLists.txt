cmake_minimum_required (VERSION 3.10)

project("RLBot") 

option(RLBOT_USE_GPU "Use GPU-enabled libtorch for RLBot inference" ON)

include(cmake/PrepareCppInterface.cmake)
prepare_cpp_interface(
    "${CMAKE_CURRENT_SOURCE_DIR}/inc/cpp-interface"
    "${CMAKE_CURRENT_SOURCE_DIR}/patches/cpp-interface.patch"
    RLBOT_CPP_SOURCE_DIR)
add_subdirectory("${RLBOT_CPP_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/inc/cpp-interface")

# The cpp-interface's sample bots aren't used here; build them only when asked for by name.
set_target_properties(ATBA ExampleBot ExampleBot-Launcher PROPERTIES EXCLUDE_FROM_ALL TRUE)

set(RLBOT_RUNTIME_SOURCES
    src/Runtime/RLBotClient.cpp
    src/Runtime/RLBotCapture.cpp
    src/Runtime/RLBotDevice.cpp
    src/Runtime/RLBotPolicyRuntime.cpp
    src/Runtime/RunRLBot.cpp)
set(RLBOT_DEVELOPMENT_SOURCES
    src/Development/Main.cpp
    src/Development/RLBotAnalysis.cpp
    src/Development/RLBotDevelopmentExtension.cpp
    src/Development/RLBotWallDashDrill.cpp)
add_executable(RLBot ${RLBOT_RUNTIME_SOURCES} ${RLBOT_DEVELOPMENT_SOURCES})
target_include_directories(RLBot PRIVATE src)

# Set C++ version to 20
set_target_properties(RLBot PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(RLBot PROPERTIES CXX_STANDARD 20)
if(RLBOT_USE_GPU)
    target_compile_definitions(RLBot PRIVATE PROJECT_ROOT="${CMAKE_SOURCE_DIR}" RLBOT_USE_GPU=1)
else()
    target_compile_definitions(RLBot PRIVATE PROJECT_ROOT="${CMAKE_SOURCE_DIR}" RLBOT_USE_GPU=0)
endif()

# Add agents you want RLBot to access here
target_link_libraries(RLBot PUBLIC GigaLearnInfer Agents RLBotCPP-static)

function(get_rlbot_agent_name AGENT_DIR OUT_VAR)
    set(AGENT_NAME "")
    file(GLOB_RECURSE AGENT_SOURCES
        "${AGENT_DIR}/src/*.cpp"
        "${AGENT_DIR}/src/*.h"
        "${AGENT_DIR}/src/*.hpp")

    foreach(AGENT_SOURCE ${AGENT_SOURCES})
        file(STRINGS "${AGENT_SOURCE}" REGISTER_AGENT_LINES REGEX "REGISTER_AGENT[ \t]*\\(")
        foreach(REGISTER_AGENT_LINE ${REGISTER_AGENT_LINES})
            if(REGISTER_AGENT_LINE MATCHES "REGISTER_AGENT[ \t]*\\([ \t]*\"([^\"]+)\"")
                set(AGENT_NAME "${CMAKE_MATCH_1}")
                break()
            endif()
        endforeach()

        if(AGENT_NAME)
            break()
        endif()
    endforeach()

    if(NOT AGENT_NAME)
        get_filename_component(AGENT_NAME "${AGENT_DIR}" NAME)
        message(WARNING "Could not find REGISTER_AGENT() name for ${AGENT_DIR}; using folder name '${AGENT_NAME}'")
    endif()

    set(${OUT_VAR} "${AGENT_NAME}" PARENT_SCOPE)
endfunction()

# Built on request (the build-rlbot-all-agents preset) to copy RLBot.exe into every agent's rlbot folder.
add_custom_target(RLBotPackageAllAgents
    DEPENDS RLBot
    COMMENT "Packaging missing RLBot executables for agents")

file(GLOB AGENT_DIRS LIST_DIRECTORIES true "${CMAKE_SOURCE_DIR}/Bots/*/*")
foreach(AGENT_DIR ${AGENT_DIRS})
    if(IS_DIRECTORY "${AGENT_DIR}")
        set(AGENT_RLBOT_DIR "${AGENT_DIR}/rlbot")
    else()
        continue()
    endif()

    if(IS_DIRECTORY "${AGENT_RLBOT_DIR}")
        get_rlbot_agent_name("${AGENT_DIR}" AGENT_NAME)
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/run.bat.in"
            "${AGENT_RLBOT_DIR}/run.bat"
            @ONLY)

        add_custom_command(TARGET RLBotPackageAllAgents POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                -D SOURCE_EXE=$<TARGET_FILE:RLBot>
                -D DEST_EXE="${AGENT_RLBOT_DIR}/RLBot.exe"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/UpdateRlBotExes.cmake")
    endif()
endforeach()

if(RLBOT_BUILD_TESTS)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    add_test(NAME RLBot.CaptureReports COMMAND "${Python3_EXECUTABLE}" -B -m unittest discover
        -s "${CMAKE_SOURCE_DIR}/tools/rlbot/capture/tests" -v)
    set_tests_properties(RLBot.CaptureReports PROPERTIES TIMEOUT 120 LABELS RLBot)

    add_test(NAME RLBot.Device COMMAND "${Python3_EXECUTABLE}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_device.py" "$<TARGET_FILE:RLBot>"
        "$<IF:$<BOOL:${RLBOT_USE_GPU}>,gpu,cpu>")
    set_tests_properties(RLBot.Device PROPERTIES TIMEOUT 180 LABELS RLBot)

    foreach(device CPU CUDA)
        if(device STREQUAL "CUDA" AND NOT WIN32)
            continue()
        endif()

        add_executable(RLBotLauncherTest${device} src/Runtime/RLBotLauncher.cpp)
        target_compile_features(RLBotLauncherTest${device} PRIVATE cxx_std_20)
        target_compile_definitions(RLBotLauncherTest${device} PRIVATE RLBOT_USE_GPU=$<STREQUAL:${device},CUDA>)
        if(WIN32)
            target_link_libraries(RLBotLauncherTest${device} PRIVATE shell32)
        endif()

        add_executable(RLBotLauncherCoreTest${device} tests/LauncherTestCore.cpp)
        target_compile_features(RLBotLauncherCoreTest${device} PRIVATE cxx_std_20)
        target_compile_definitions(RLBotLauncherCoreTest${device} PRIVATE TEST_DEVICE="${device}")

        add_test(NAME RLBotLauncher.${device} COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_launcher.py"
            "$<TARGET_FILE:RLBotLauncherTest${device}>" "$<TARGET_FILE:RLBotLauncherCoreTest${device}>" "${device}")
        set_tests_properties(RLBotLauncher.${device} PROPERTIES TIMEOUT 180 LABELS RLBot)
    endforeach()

    add_test(NAME RLBot.BobBuild COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_bob_build.py")
    set_tests_properties(RLBot.BobBuild PROPERTIES TIMEOUT 120 LABELS RLBot)

    add_executable(RLBotHivemindTests
        tests/Hivemind_tests.cpp
        ${RLBOT_RUNTIME_SOURCES}
        src/Development/RLBotAnalysis.cpp
        src/Development/RLBotDevelopmentExtension.cpp
        src/Development/RLBotWallDashDrill.cpp
        ${CMAKE_SOURCE_DIR}/ForgeTrainingTests/GTestMain.cpp
        ${CMAKE_SOURCE_DIR}/ForgeTrainingTests/TestSupport.cpp)
    target_include_directories(RLBotHivemindTests PRIVATE src ${CMAKE_SOURCE_DIR}/ForgeTrainingTests)
    target_compile_features(RLBotHivemindTests PRIVATE cxx_std_20)
    target_compile_definitions(RLBotHivemindTests PRIVATE
        PROJECT_ROOT="${CMAKE_SOURCE_DIR}" FORGE_SOURCE_DIR="${CMAKE_SOURCE_DIR}" RLBOT_USE_GPU=0)
    target_link_libraries(RLBotHivemindTests PRIVATE GigaLearnInfer Agents RLBotCPP-static GTest::gtest)
    gtest_discover_tests(RLBotHivemindTests DISCOVERY_MODE PRE_TEST DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 120 LABELS RLBot)
endif()
