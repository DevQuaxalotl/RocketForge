cmake_minimum_required (VERSION 3.10)

project("ForgeTraining")

# Add all headers and code files
file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h")
add_library(ForgeTraining STATIC ${FILES_SRC})
target_include_directories(ForgeTraining PUBLIC "src/")

# Set C++ version to 20
set_target_properties(ForgeTraining PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(ForgeTraining PROPERTIES CXX_STANDARD 20)

# Include RLGym (must be before ForgeDebugger which depends on it)
add_subdirectory(RLGym)

# The submittal export vendors RocketSim v3's crates with the cargo that builds them (empty without v3).
get_property(_rocketsim_v3_cargo GLOBAL PROPERTY ROCKETSIM_V3_CARGO_EXECUTABLE)
target_compile_definitions(ForgeTraining PRIVATE ROCKETFORGE_CARGO_EXECUTABLE="${_rocketsim_v3_cargo}")

# Include Debugger (depends on RLGym, must be before GigaLearn which uses it)
add_subdirectory(ForgeDebugger)

# Include GigaLearn (links ForgeDebugger internally via its own CMakeLists.txt)
add_subdirectory(GigaLearn)

target_link_libraries(ForgeTraining PUBLIC GigaLearn)
target_link_libraries(ForgeTraining PUBLIC GigaLearnInfer)
target_link_libraries(ForgeTraining PUBLIC ForgeDebugger)

target_compile_definitions(ForgeTraining PRIVATE PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
target_compile_definitions(ForgeTraining PUBLIC TRAINING_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")