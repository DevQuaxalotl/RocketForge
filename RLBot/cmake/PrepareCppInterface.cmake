# Keep upstream sources untouched, including in exported packages without Git metadata.
# The patch preserves packet diagnostics and hivemind fixes from 3479769 and 25f234f
# on top of upstream 8a7faf92d41e964fc3e1a4191a4bd21b433f212b.
function(prepare_cpp_interface SOURCE_DIR PATCH_FILE OUTPUT_VAR)
    find_package(Git REQUIRED)
    if(NOT EXISTS "${SOURCE_DIR}/library/Client.cpp")
        message(FATAL_ERROR "cpp-interface sources are missing. Initialize the Git submodules first.")
    endif()

    # Include only build inputs; never copy submodule metadata or local build outputs.
    file(GLOB_RECURSE INPUTS CONFIGURE_DEPENDS RELATIVE "${SOURCE_DIR}"
        "${SOURCE_DIR}/library/*"
        "${SOURCE_DIR}/examples/*"
        "${SOURCE_DIR}/benchmark/*")
    list(APPEND INPUTS CMakeLists.txt LICENSE)
    list(SORT INPUTS)
    file(SHA256 "${PATCH_FILE}" FINGERPRINT)
    file(SHA256 "${CMAKE_CURRENT_FUNCTION_LIST_FILE}" HELPER_HASH)
    string(APPEND FINGERPRINT "${HELPER_HASH}")
    foreach(INPUT IN LISTS INPUTS)
        file(SHA256 "${SOURCE_DIR}/${INPUT}" INPUT_HASH)
        string(APPEND FINGERPRINT "${INPUT}:${INPUT_HASH};")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${SOURCE_DIR}/${INPUT}")
    endforeach()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${PATCH_FILE}")
    string(SHA256 FINGERPRINT "${FINGERPRINT}")
    set(PREPARED "${CMAKE_CURRENT_BINARY_DIR}/cpp-interface-patched/${FINGERPRINT}")

    # Reuse identical inputs without touching timestamps or forcing recompilation.
    if(NOT EXISTS "${PREPARED}/.patch-complete")
        foreach(INPUT IN LISTS INPUTS)
            configure_file("${SOURCE_DIR}/${INPUT}" "${PREPARED}/${INPUT}" COPYONLY)
        endforeach()
        # Stop repository discovery: git apply otherwise silently skips paths when
        # the build directory lives inside the parent repository.
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E env "GIT_CEILING_DIRECTORIES=${CMAKE_CURRENT_BINARY_DIR}"
                "${GIT_EXECUTABLE}" apply --whitespace=nowarn "${PATCH_FILE}"
            WORKING_DIRECTORY "${PREPARED}"
            RESULT_VARIABLE PATCH_RESULT ERROR_VARIABLE PATCH_ERROR)
        if(NOT PATCH_RESULT EQUAL 0)
            message(FATAL_ERROR "RocketForge's cpp-interface patch no longer applies. Update the patch for the upstream revision.\n${PATCH_ERROR}")
        endif()
        file(WRITE "${PREPARED}/.patch-complete" "${FINGERPRINT}\n")
    endif()
    set(${OUTPUT_VAR} "${PREPARED}" PARENT_SCOPE)
endfunction()
