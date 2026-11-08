cmake_minimum_required (VERSION 3.10)

project("GigaLearn")

# Add all headers and code files
file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h" "src/*.hpp" "libsrc/*.cpp" "libsrc/.h" "libsrc/.hpp")

add_library(GigaLearn SHARED ${FILES_SRC})

target_include_directories(GigaLearn PUBLIC "src/public")
target_include_directories(GigaLearn PRIVATE "src/private")
target_compile_definitions(GigaLearn PRIVATE WITHIN_GIGALEARN)

# Set C++ version to 20
set_target_properties(GigaLearn PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(GigaLearn PROPERTIES CXX_STANDARD 20)

# Exported classes hold STL members; that's safe because the DLL and all its clients share one compiler and CRT.
if (MSVC)
    target_compile_options(GigaLearn PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/wd4251>)
endif()

# Include GigaLearnInfer
add_subdirectory(GigaLearnInfer)
target_link_libraries(GigaLearn PUBLIC GigaLearnInfer)

# Include ForgeDebugger (reward debug visualization)
target_link_libraries(GigaLearn PUBLIC ForgeDebugger)

# Include JSON
target_include_directories(GigaLearn PUBLIC "${PROJECT_SOURCE_DIR}/libsrc/json")

# Include python
find_package(Python COMPONENTS Interpreter Development)
find_package(PythonLibs)
if(NOT PYTHONLIBS_FOUND)
    message(FATAL_ERROR
        "Could not find Python's headers and library, which GigaLearn embeds for wandb metrics.\n"
        "Install Python 3.9-3.12 with: powershell -ExecutionPolicy Bypass -File .\\tools\\setup\\install_python.ps1\n"
        "then configure again from a new terminal.")
endif()
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(GigaLearn PUBLIC ${PYTHON_LIBRARIES})
message("Found Python:")
message("PYTHON_LIBRARIES: ${PYTHON_LIBRARIES}")
message("PYTHON_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS}")
message("Python_RUNTIME_LIBRARY_DIRS: ${Python_RUNTIME_LIBRARY_DIRS}")
message("Python_EXECUTABLE: ${Python_EXECUTABLE}")
add_definitions(-DPY_EXEC_PATH="${Python_EXECUTABLE}") # Give C++ access to the executable path
target_compile_definitions(GigaLearn PUBLIC PY_LIB_PATH="${Python_RUNTIME_LIBRARY_DIRS}") # Give C++ public access to the library path
# The configure_file calls below copy python_scripts/ into CMAKE_BINARY_DIR.
# Expose that path so C++ can add it to sys.path at runtime — ensures
# "import python_scripts.*" works regardless of the working directory or where
# the exe lives relative to the build root (e.g. bin/ subdir on Linux).
target_compile_definitions(GigaLearn PUBLIC GIGALEARN_BUILD_DIR="${CMAKE_BINARY_DIR}")

# Include pybind11
add_subdirectory(pybind11)
target_link_libraries(GigaLearn PUBLIC pybind11::embed)

# MSVC fails to find python DLLs even through they are in my path. Good job MSVC. Well done.
# This copies the the python DLLs to the output directory
if (MSVC)
    file(GLOB PYTHON_DLLS "${Python_RUNTIME_LIBRARY_DIRS}/*.dll")
	message("Adding Python DLLS: ${PYTHON_DLLS}")
    add_custom_command(TARGET GigaLearn
                 POST_BUILD
                 COMMAND ${CMAKE_COMMAND} -E copy_if_different
                 ${PYTHON_DLLS}
                 $<TARGET_FILE_DIR:GigaLearn>)

    if (GIGALEARN_INFER_TORCH_RUNTIME_DLLS)
        message("Adding Torch DLLS from: ${GIGALEARN_INFER_TORCH_RUNTIME_DIR}")
        add_custom_command(TARGET GigaLearn
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${GIGALEARN_INFER_TORCH_RUNTIME_DLLS}
                     $<TARGET_FILE_DIR:GigaLearn>
                     COMMAND_EXPAND_LISTS
                     VERBATIM)
    endif()
endif (MSVC)

# Make our python files copy over to our build dir
configure_file("./python_scripts/metric_receiver.py" "../../python_scripts/metric_receiver.py" COPYONLY)
configure_file("./python_scripts/render_receiver.py" "../../python_scripts/render_receiver.py" COPYONLY)