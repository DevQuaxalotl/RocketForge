cmake_minimum_required (VERSION 3.10)

project("ExampleTrainingPlan")

# Add all headers and code files
file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h")

add_library(ExampleTrainingPlan OBJECT ${FILES_SRC})

# Set C++ version to 20
set_target_properties(ExampleTrainingPlan PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(ExampleTrainingPlan PROPERTIES CXX_STANDARD 20)

target_link_libraries(ExampleTrainingPlan PRIVATE ForgeTraining)