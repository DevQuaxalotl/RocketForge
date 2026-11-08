cmake_minimum_required(VERSION 3.10)

project(PublicBotAgents)

if(RLBOT)
    return()
endif()

file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h")

add_library(PublicBotAgents OBJECT ${FILES_SRC})
set_target_properties(PublicBotAgents PROPERTIES LINKER_LANGUAGE CXX CXX_STANDARD 20)
target_link_libraries(PublicBotAgents PUBLIC GigaLearnInfer)
target_compile_definitions(PublicBotAgents PRIVATE
	PUBLIC_BOTS_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

target_sources(Agents INTERFACE $<TARGET_OBJECTS:PublicBotAgents>)
