cmake_minimum_required(VERSION 3.10)

project(ExampleBot)

add_subdirectory(ExampleAgent)
target_sources(Agents INTERFACE $<TARGET_OBJECTS:ExampleAgent>)

if(NOT RLBOT)
    add_subdirectory(ExampleTrainingPlan)
    target_sources(TrainingPlans INTERFACE $<TARGET_OBJECTS:ExampleTrainingPlan>)
endif()
