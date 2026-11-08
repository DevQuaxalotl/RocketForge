cmake_minimum_required(VERSION 3.10)

project(RocketForgeParityProbe)

set(PARITY_PROBE_SOURCES
	src/AdapterRegistry.cpp
	src/Hash.cpp
	src/Protocol.cpp
	src/main.cpp)

# Local adapters are intentionally optional. The containing local directory is
# expected to be ignored by the parent repository and is never required to build.
file(GLOB PARITY_PROBE_LOCAL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/local/*.cpp")

add_executable(
	RocketForgeParityProbe
	EXCLUDE_FROM_ALL
	${PARITY_PROBE_SOURCES}
	${PARITY_PROBE_LOCAL_SOURCES})

target_include_directories(
	RocketForgeParityProbe
	PRIVATE
		"${CMAKE_CURRENT_SOURCE_DIR}/include"
		"${CMAKE_CURRENT_SOURCE_DIR}/local")

target_link_libraries(RocketForgeParityProbe PRIVATE GigaLearnInfer Agents)

set_target_properties(
	RocketForgeParityProbe
	PROPERTIES
		CXX_STANDARD 20
		CXX_STANDARD_REQUIRED ON
		LINKER_LANGUAGE CXX)

if(MSVC)
	target_compile_options(RocketForgeParityProbe PRIVATE /bigobj)
	file(GLOB PARITY_PROBE_TORCH_RUNTIME_DLLS "${LIBTORCH_ROOT}/lib/*.dll")
	add_custom_command(
		TARGET RocketForgeParityProbe
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${PARITY_PROBE_TORCH_RUNTIME_DLLS}
			$<TARGET_FILE_DIR:RocketForgeParityProbe>
		COMMAND_EXPAND_LISTS
		VERBATIM)
endif()
