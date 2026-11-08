cmake_minimum_required(VERSION 3.10)

project("RLGym")

# Add all headers and code files
file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h")
add_library(RLGym STATIC ${FILES_SRC})
target_include_directories(RLGym PUBLIC "src/")

# Set C++ version to 20
set_target_properties(RLGym PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(RLGym PROPERTIES CXX_STANDARD 20)

# The shared RocketSim API with both engines behind it: the v3 Rust bindings come first so the Sim layer sees them.
add_subdirectory("RocketSimV3")
add_subdirectory("Sim")
add_subdirectory("RocketSim")
target_link_libraries(RLGym PUBLIC RocketForgeSim)

# Include thread pool library (https://github.com/DeveloperPaul123/thread-pool)
target_include_directories(RLGym PUBLIC "thread_pool")