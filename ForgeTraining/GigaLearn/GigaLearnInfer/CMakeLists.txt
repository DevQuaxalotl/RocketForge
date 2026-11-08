cmake_minimum_required (VERSION 3.10)

project("GigaLearnInfer")

# Add all headers and code files
file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h")

add_library(GigaLearnInfer STATIC ${FILES_SRC})
target_compile_definitions(GigaLearnInfer PUBLIC -DGIGALEARN_INFER_STATIC)
target_compile_definitions(GigaLearnInfer PRIVATE -DWITHIN_GGL)
target_compile_definitions(GigaLearnInfer PRIVATE PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
target_include_directories(GigaLearnInfer PUBLIC
    "src/public/"
    "../libsrc/json")

# Allow override: -DLIBTORCH_ROOT="C:/path/to/libtorch"
if(NOT DEFINED LIBTORCH_ROOT)
    if(WIN32)
        set(
            LIBTORCH_ROOT
            "$ENV{LOCALAPPDATA}/libtorch"
            CACHE PATH
            "Root directory of the libtorch distribution")
    else()
        set(
            LIBTORCH_ROOT
            "/opt/libtorch"
            CACHE PATH
            "Root directory of the libtorch distribution")
    endif()
endif() 

# TorchConfig.cmake usually lives here in the official libtorch distribution:
set(Torch_DIR "${LIBTORCH_ROOT}/share/cmake/Torch")

if(NOT EXISTS "${Torch_DIR}/TorchConfig.cmake")
    if(WIN32)
        set(_TORCH_INSTALL_HINT
            "Install it with: powershell -ExecutionPolicy Bypass -File .\\tools\\setup\\install_libtorch.ps1\n"
            "or extract https://download.pytorch.org/libtorch/cu126/libtorch-win-shared-with-deps-2.13.0%2Bcu126.zip "
            "(or, without an NVIDIA GPU, https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.13.0%2Bcpu.zip) "
            "so that its libtorch folder is at %LOCALAPPDATA%\\libtorch.")
    else()
        set(_TORCH_INSTALL_HINT
            "Extract https://download.pytorch.org/libtorch/cu129/libtorch-shared-with-deps-2.13.0%2Bcu129.zip "
            "into /opt (as the Dockerfile does).")
    endif()
    string(CONCAT _TORCH_INSTALL_HINT ${_TORCH_INSTALL_HINT})
    message(FATAL_ERROR
        "Could not find TorchConfig.cmake at: ${Torch_DIR}/TorchConfig.cmake\n"
        "Check LIBTORCH_ROOT='${LIBTORCH_ROOT}' (expected libtorch layout with share/cmake/Torch).\n"
        "${_TORCH_INSTALL_HINT}\n"
        "To use libtorch from another folder, configure with -DLIBTORCH_ROOT=<path>."
    )
endif()

# A CUDA build of libtorch enables the CUDA language from its own config, which fails obscurely
# (e.g. "CMAKE_CUDA_ARCHITECTURES must be non-empty if set") when the preset's nvcc is missing.
if(EXISTS "${LIBTORCH_ROOT}/build-version")
    file(READ "${LIBTORCH_ROOT}/build-version" _TORCH_BUILD_VERSION)
    string(STRIP "${_TORCH_BUILD_VERSION}" _TORCH_BUILD_VERSION)
    if(_TORCH_BUILD_VERSION MATCHES "\\+cu" AND DEFINED CMAKE_CUDA_COMPILER AND NOT EXISTS "${CMAKE_CUDA_COMPILER}")
        message(FATAL_ERROR
            "libtorch ${_TORCH_BUILD_VERSION} at '${LIBTORCH_ROOT}' is a CUDA build, "
            "but the CUDA compiler '${CMAKE_CUDA_COMPILER}' does not exist.\n"
            "Install the CUDA 12.6 toolkit (https://developer.nvidia.com/cuda-12-6-0-download-archive), "
            "or, to build without CUDA, replace libtorch with the CPU build: "
            "powershell -ExecutionPolicy Bypass -File .\\tools\\setup\\install_libtorch.ps1 -Variant Cpu"
        )
    endif()
endif()

# A configure that failed to detect a default CUDA architecture caches it empty, and every later
# configure then fails with "must be non-empty if set". libtorch ignores the value, so drop it.
if(DEFINED CACHE{CMAKE_CUDA_ARCHITECTURES} AND "$CACHE{CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
    unset(CMAKE_CUDA_ARCHITECTURES CACHE)
endif()

find_package(Torch REQUIRED CONFIG)
 
if(TARGET Torch::Torch)
  set(_TORCH_LINK Torch::Torch)
elseif(TARGET torch)
  set(_TORCH_LINK torch)
else()
  set(_TORCH_LINK ${TORCH_LIBRARIES})
endif() 


if (TORCH_CUDA_LIBRARIES)
	message("Enabling CUDA support...")
	target_compile_definitions(GigaLearnInfer PUBLIC -DRG_CUDA_SUPPORT)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

target_link_libraries(GigaLearnInfer PUBLIC ${_TORCH_LINK})

if(WIN32)
  if(TORCH_LIBRARY)
    get_filename_component(_TORCH_RUNTIME_LIBRARY_DIR "${TORCH_LIBRARY}" DIRECTORY)
  else()
    set(_TORCH_RUNTIME_LIBRARY_DIR "${LIBTORCH_ROOT}/lib")
  endif()

  file(GLOB _TORCH_RUNTIME_DLLS "${_TORCH_RUNTIME_LIBRARY_DIR}/*.dll")

  if(_TORCH_RUNTIME_DLLS)
    set(GIGALEARN_INFER_TORCH_RUNTIME_DIR "${_TORCH_RUNTIME_LIBRARY_DIR}" PARENT_SCOPE)
    set(GIGALEARN_INFER_TORCH_RUNTIME_DLLS ${_TORCH_RUNTIME_DLLS} PARENT_SCOPE)
  else()
    message(WARNING "No Torch runtime DLLs found in ${_TORCH_RUNTIME_LIBRARY_DIR}")
  endif()
endif()

# Set C++ version to 20
set_target_properties(GigaLearnInfer PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(GigaLearnInfer PROPERTIES CXX_STANDARD 20)

target_link_libraries(GigaLearnInfer PUBLIC RLGym)