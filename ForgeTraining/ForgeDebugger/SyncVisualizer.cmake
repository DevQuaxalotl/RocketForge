cmake_minimum_required(VERSION 3.12)

# Mirrors the RocketSimVis files listed in MANIFEST into DESTINATION.
# The manifest is regenerated on configure (CONFIGURE_DEPENDS), so it tracks added and removed resources.
file(STRINGS "${MANIFEST}" current_files)
if(EXISTS "${PREVIOUS_MANIFEST}")
    file(STRINGS "${PREVIOUS_MANIFEST}" previous_files)
endif()

# Validate both manifests before changing anything.
# Only previously owned files may be removed; unrelated files in the output directory are left alone.
foreach(relative IN LISTS current_files previous_files)
    if(relative MATCHES "(^|[/\\\\])\\.\\.([/\\\\]|$)" OR IS_ABSOLUTE "${relative}")
        message(FATAL_ERROR "Invalid visualizer resource path: ${relative}")
    endif()
endforeach()

foreach(relative IN LISTS previous_files)
    if(NOT relative IN_LIST current_files)
        file(REMOVE "${DESTINATION}/${relative}")
    endif()
endforeach()

foreach(relative IN LISTS current_files)
    get_filename_component(parent "${DESTINATION}/${relative}" DIRECTORY)
    file(MAKE_DIRECTORY "${parent}")
    execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${SOURCE}/${relative}" "${DESTINATION}/${relative}" RESULT_VARIABLE copy_result)
    if(NOT copy_result EQUAL 0)
        message(FATAL_ERROR "Failed to copy visualizer resource: ${relative}")
    endif()
endforeach()

configure_file("${MANIFEST}" "${PREVIOUS_MANIFEST}" COPYONLY)
file(TOUCH "${STAMP}")
