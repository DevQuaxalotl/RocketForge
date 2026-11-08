# Fills the RLBot cpp-interface submodule on configure when a clone didn't fetch submodules.
# The RLBot build and the RLBot submittal export both read its sources; training doesn't.

set(_rf_rlbot_cpp_interface_path "RLBot/inc/cpp-interface")

function(RocketForgeFetchRLBotCppInterface)
    if(EXISTS "${CMAKE_SOURCE_DIR}/${_rf_rlbot_cpp_interface_path}/CMakeLists.txt")
        return()
    endif()

    # An exported package has no Git metadata and ships its own copy of the sources.
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/.git")
        return()
    endif()

    find_package(Git QUIET)
    if(GIT_FOUND)
        message(STATUS "Fetching the RLBot cpp-interface submodule...")
        execute_process(COMMAND "${GIT_EXECUTABLE}" submodule update --init "${_rf_rlbot_cpp_interface_path}"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" OUTPUT_QUIET ERROR_QUIET)
    endif()

    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/${_rf_rlbot_cpp_interface_path}/CMakeLists.txt")
        message(WARNING "The RLBot cpp-interface submodule is empty, so RLBot builds and RLBot submittal exports won't work. "
            "Run \"git submodule update --init ${_rf_rlbot_cpp_interface_path}\", then re-run CMake.")
    endif()
endfunction()
