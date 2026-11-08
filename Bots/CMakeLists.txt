cmake_minimum_required(VERSION 3.10)

project(Bots)

# Must be defined before bot subdirectories so they can append to these targets.
add_library(Agents INTERFACE)
add_library(TrainingPlans INTERFACE)

# Auto-discovers all bot subdirectories that contain a CMakeLists.txt.
# To add a bot, drop its folder here and re-run cmake configure.
file(GLOB BOT_DIRS LIST_DIRECTORIES true "${CMAKE_CURRENT_SOURCE_DIR}/*")
foreach(BOT_DIR ${BOT_DIRS})
    if(IS_DIRECTORY "${BOT_DIR}" AND EXISTS "${BOT_DIR}/CMakeLists.txt")
        add_subdirectory("${BOT_DIR}")
    endif()
endforeach()
