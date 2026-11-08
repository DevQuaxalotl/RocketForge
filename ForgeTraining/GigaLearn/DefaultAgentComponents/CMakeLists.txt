cmake_minimum_required(VERSION 3.10)

project(DefaultAgentComponents)

if(NOT TARGET Agents)
    message(FATAL_ERROR "DefaultAgentComponents requires the Agents target")
endif()

add_library(DefaultAgentComponents OBJECT
    "src/DefaultAgentComponents.cpp"
)
set_target_properties(DefaultAgentComponents PROPERTIES LINKER_LANGUAGE CXX CXX_STANDARD 20)
target_link_libraries(DefaultAgentComponents PUBLIC GigaLearnInfer)

target_sources(Agents INTERFACE $<TARGET_OBJECTS:DefaultAgentComponents>)
