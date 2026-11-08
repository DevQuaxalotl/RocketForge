cmake_minimum_required (VERSION 3.10)

project("ExampleAgent")

# Add all headers and code files
file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h")

add_library(ExampleAgent OBJECT ${FILES_SRC})

# Set C++ version to 20
set_target_properties(ExampleAgent PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(ExampleAgent PROPERTIES CXX_STANDARD 20)

target_link_libraries(ExampleAgent PUBLIC GigaLearnInfer)

target_compile_definitions(ExampleAgent PRIVATE EXAMPLE_AGENT_MODELS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/Models")