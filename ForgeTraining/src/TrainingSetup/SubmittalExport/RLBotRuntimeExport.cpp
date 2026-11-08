#include "SubmittalExportInternal.h"

#include <sstream>

namespace fs = std::filesystem;

namespace RocketForgeTraining::SubmittalExportInternal
{
	namespace
	{
		std::string MakeMinimalRLBotCMake()
		{
			return
				"cmake_minimum_required(VERSION 3.8)\n\n"
				"project(RLBot)\n\n"
				"include(cmake/PrepareCppInterface.cmake)\n"
				"prepare_cpp_interface(\n"
				"    \"${CMAKE_CURRENT_SOURCE_DIR}/inc/cpp-interface\"\n"
				"    \"${CMAKE_CURRENT_SOURCE_DIR}/patches/cpp-interface.patch\"\n"
				"    RLBOT_CPP_SOURCE_DIR)\n"
				"add_subdirectory(\"${RLBOT_CPP_SOURCE_DIR}\" \"${CMAKE_CURRENT_BINARY_DIR}/inc/cpp-interface\")\n\n"
				"# Torch-linked runtime core.\n"
				"add_executable(RLBotCore\n"
				"    src/Runtime/PortableMain.cpp\n"
				"    src/Runtime/RLBotClient.cpp\n"
				"    src/Runtime/RLBotCapture.cpp\n"
				"    src/Runtime/RLBotDevice.cpp\n"
				"    src/Runtime/RLBotPolicyRuntime.cpp\n"
				"    src/Runtime/RunRLBot.cpp\n"
				")\n"
				"set_target_properties(RLBotCore PROPERTIES OUTPUT_NAME ${RLBOT_CORE_NAME})\n"
				"target_include_directories(RLBotCore PRIVATE src)\n"
				"target_compile_features(RLBotCore PRIVATE cxx_std_20)\n"
				"target_compile_definitions(RLBotCore PRIVATE\n"
				"    PROJECT_ROOT=\"${CMAKE_SOURCE_DIR}\"\n"
				"    RLBOT_USE_GPU=$<IF:$<BOOL:${RLBOT_USE_GPU}>,1,0>\n"
				"    RLBOT_HAS_CUDA=$<BOOL:${RLBOT_USE_GPU}>\n"
				")\n"
				"target_link_libraries(RLBotCore PUBLIC GigaLearnInfer Agents RLBotCPP-static)\n\n"
				"# Dependency-free launcher which resolves RLBot's shared Torch runtime.\n"
				"add_executable(RLBot src/Runtime/RLBotLauncher.cpp)\n"
				"target_compile_features(RLBot PRIVATE cxx_std_20)\n"
				"target_compile_definitions(RLBot PRIVATE RLBOT_USE_GPU=$<BOOL:${RLBOT_USE_GPU}>)\n"
				"add_dependencies(RLBot RLBotCore)\n"
				"if(WIN32)\n"
				"    target_link_libraries(RLBot PRIVATE shell32)\n"
				"endif()\n";
		}

		std::string MakeMinimalRLBotCppInterfaceCMake(const fs::path& sourcePath)
		{
			std::istringstream input(ReadTextFile(sourcePath));
			std::ostringstream output;
			std::string line;
			while (std::getline(input, line)) {
				const std::string trimmed = Trim(line);
				if (trimmed == "add_subdirectory(examples/ATBA)" || trimmed == "add_subdirectory(examples/ExampleBot)")
					continue;
				output << line << '\n';
			}

			return output.str();
		}

		// Unrolls the library's shared/static target loop so the package builds only the static target RLBotCore links.
		std::string MakeMinimalRLBotCppLibraryCMake(const fs::path& sourcePath)
		{
			std::istringstream input(ReadTextFile(sourcePath));
			std::ostringstream output;
			std::string line;
			while (std::getline(input, line)) {
				const std::string trimmed = Trim(line);
				if (trimmed == "add_library(${PROJECT_NAME} SHARED)" || trimmed == "endforeach()")
					continue;

				if (trimmed == "foreach(TARGET ${PROJECT_NAME} ${PROJECT_NAME}-static)") {
					output << "set(TARGET ${PROJECT_NAME}-static)\n";
					continue;
				}

				if (trimmed == "set_target_properties(${PROJECT_NAME} PROPERTIES") {
					output << "\tset_target_properties(${TARGET} PROPERTIES\n";
					continue;
				}

				output << line << '\n';
			}

			return output.str();
		}
	}

	void CopyMinimalRLBot(const fs::path& projectRoot, const fs::path& destinationRoot)
	{
		const fs::path sourceRoot = projectRoot / "RLBot";
		const std::set<fs::path> runtimeFiles = {
			"cmake/PrepareCppInterface.cmake",
			"patches/cpp-interface.patch",
			"src/Runtime/RLBotLauncher.cpp",
			"src/Runtime/PortableMain.cpp",
			"src/Runtime/RLBotClient.cpp",
			"src/Runtime/RLBotClient.h",
			"src/Runtime/RLBotCapture.cpp",
			"src/Runtime/RLBotCapture.h",
			"src/Runtime/RLBotDevice.cpp",
			"src/Runtime/RLBotDevice.h",
			"src/Runtime/RLBotClientExtension.h",
			"src/Runtime/RLBotPolicyObserver.h",
			"src/Runtime/RLBotPolicyRuntime.cpp",
			"src/Runtime/RLBotPolicyRuntime.h",
			"src/Runtime/RunRLBot.cpp",
			"src/Runtime/RunRLBot.h",
		};

		CopyFile(projectRoot / "tools" / "rlbot" / "capture" / "capture.py", destinationRoot.parent_path() / "scripts" / "rlbot_capture.py");
		CopyExplicitFiles(sourceRoot, destinationRoot, runtimeFiles);
		WriteTextFile(destinationRoot / "CMakeLists.txt", MakeMinimalRLBotCMake());

		const fs::path interfaceSource = sourceRoot / "inc" / "cpp-interface";
		const fs::path interfaceDestination = destinationRoot / "inc" / "cpp-interface";
		WriteTextFile(interfaceDestination / "CMakeLists.txt", MakeMinimalRLBotCppInterfaceCMake(interfaceSource / "CMakeLists.txt"));
		CopyFile(interfaceSource / "LICENSE", interfaceDestination / "LICENSE");
		WriteTextFile(
			interfaceDestination / "library" / "CMakeLists.txt",
			MakeMinimalRLBotCppLibraryCMake(interfaceSource / "library" / "CMakeLists.txt"));

		const std::set<fs::path> interfaceLibraryFiles = {
			"Bot.cpp", "BotContext.cpp", "BotContext.h", "BotManager.cpp", "Client.cpp",
			"Log.cpp", "Log.h", "Message.cpp", "Message.h", "Pool.cpp", "Pool.h",
			"SockAddr.cpp", "SockAddr.h", "Socket.cpp", "Socket.h", "TracyHelper.h",
			"WsaData.cpp", "WsaData.h",
			"include/rlbot/Bot.h", "include/rlbot/BotManager.h", "include/rlbot/Client.h",
			"include/rlbot/RLBotCPP.h",
		};
		CopyExplicitFiles(interfaceSource / "library", interfaceDestination / "library", interfaceLibraryFiles);
	}
}
