cmake_minimum_required (VERSION 3.10)

project("ForgeConsoleTrainer")

# Add all headers and code files
file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h")
add_executable(ForgeConsoleTrainer ${FILES_SRC})

# Set C++ version to 20
set_target_properties(ForgeConsoleTrainer PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(ForgeConsoleTrainer PROPERTIES CXX_STANDARD 20)

target_include_directories(ForgeConsoleTrainer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(ForgeConsoleTrainer PUBLIC ForgeTraining Agents TrainingPlans)