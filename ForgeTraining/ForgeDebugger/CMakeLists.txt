cmake_minimum_required (VERSION 3.12)

project("ForgeDebugger")

# Add all headers and code files
file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h")
add_library(ForgeDebugger STATIC ${FILES_SRC})
target_include_directories(ForgeDebugger PUBLIC "src/")

# Set C++ version to 20
set_target_properties(ForgeDebugger PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(ForgeDebugger PROPERTIES CXX_STANDARD 20)

# ForgeDebugger only needs RLGym for Reward.h types
target_link_libraries(ForgeDebugger PUBLIC RLGym)

# Track individual resources so a no-op build performs no copy or write.
set(_vis_source "${CMAKE_CURRENT_SOURCE_DIR}/RocketSimVis")
set(_vis_destination "${CMAKE_BINARY_DIR}/ForgeDebugger/RocketSimVis")
file(GLOB_RECURSE _vis_files CONFIGURE_DEPENDS LIST_DIRECTORIES false RELATIVE "${_vis_source}" "${_vis_source}/*")
list(FILTER _vis_files EXCLUDE REGEX "(^|/)__pycache__/|\\.pyc$")

set(_vis_inputs)
set(_vis_outputs)
foreach(_vis_file IN LISTS _vis_files)
    list(APPEND _vis_inputs "${_vis_source}/${_vis_file}")
    list(APPEND _vis_outputs "${_vis_destination}/${_vis_file}")
endforeach()

string(REPLACE ";" "\n" _vis_manifest_content "${_vis_files}")
set(_vis_manifest "${CMAKE_CURRENT_BINARY_DIR}/visualizer-resources.txt")
file(GENERATE OUTPUT "${_vis_manifest}" CONTENT "${_vis_manifest_content}\n")

set(_vis_stamp "${CMAKE_CURRENT_BINARY_DIR}/visualizer-resources.stamp")
add_custom_command(OUTPUT "${_vis_stamp}" ${_vis_outputs}
    COMMAND ${CMAKE_COMMAND}
        "-DSOURCE=${_vis_source}" "-DDESTINATION=${_vis_destination}"
        "-DMANIFEST=${_vis_manifest}"
        "-DPREVIOUS_MANIFEST=${CMAKE_CURRENT_BINARY_DIR}/visualizer-resources-previous.txt"
        "-DSTAMP=${_vis_stamp}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/SyncVisualizer.cmake"
    DEPENDS ${_vis_inputs} "${_vis_manifest}" "${CMAKE_CURRENT_SOURCE_DIR}/SyncVisualizer.cmake"
    COMMENT "Updating changed RocketSimVis resources"
    VERBATIM)

add_custom_target(CopyRocketSimVis ALL DEPENDS "${_vis_stamp}" ${_vis_outputs})
add_dependencies(ForgeDebugger CopyRocketSimVis)
