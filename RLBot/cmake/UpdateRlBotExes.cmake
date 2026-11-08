if(NOT DEFINED SOURCE_EXE)
    message(FATAL_ERROR "SOURCE_EXE was not provided.")
endif()

if(NOT DEFINED DEST_EXE)
    message(FATAL_ERROR "DEST_EXE was not provided.")
endif()

if(NOT EXISTS "${SOURCE_EXE}")
    message(FATAL_ERROR "RLBot executable does not exist: ${SOURCE_EXE}")
endif()

get_filename_component(DEST_DIR "${DEST_EXE}" DIRECTORY)
file(MAKE_DIRECTORY "${DEST_DIR}")

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${SOURCE_EXE}" "${DEST_EXE}"
    RESULT_VARIABLE COPY_RESULT)

if(NOT COPY_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to copy ${SOURCE_EXE} to ${DEST_EXE}")
endif()

message(STATUS "Created RLBot package executable: ${DEST_EXE}")
