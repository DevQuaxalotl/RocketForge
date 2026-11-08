#include "SubmittalExportInternal.h"

#include <sstream>

namespace fs = std::filesystem;

namespace RocketForgeTraining::SubmittalExportInternal
{
	namespace
	{
		std::string MakeMinimalGigaLearnCMake(const fs::path& sourcePath)
		{
			std::istringstream input(ReadTextFile(sourcePath));
			std::ostringstream output;
			std::string line;
			bool skippingTorch = false;
			while (std::getline(input, line)) {
				const std::string trimmed = Trim(line);

				// The source's libtorch lookup runs from its "Allow override" comment up to the C++ version block;
				// the package swaps it for its portable Torch setup.
				if (trimmed.starts_with("# Allow override:")) {
					output << "include(\"${CMAKE_SOURCE_DIR}/cmake/PortableTorch.cmake\")\n"
						"rocketforge_link_portable_torch(GigaLearnInfer)\n\n";
					skippingTorch = true;
				}

				if (trimmed == "# Set C++ version to 20")
					skippingTorch = false;
				if (skippingTorch)
					continue;

				if (trimmed.starts_with("file(GLOB_RECURSE FILES_SRC"))
					continue;

				if (trimmed.starts_with("add_library(GigaLearnInfer")) {
					output <<
						"set(FILES_SRC\n"
						"    src/private/Agent.cpp\n"
						"    src/private/AgentSetupSnapshot.cpp\n"
						"    src/private/InferUnit.cpp\n"
						"    src/private/ModelManager.cpp\n"
						"    src/private/Models.cpp\n"
						")\n\n";
				}

				output << line << '\n';
			}

			return output.str();
		}
	}

	void CopyMinimalGigaLearn(const fs::path& projectRoot, const fs::path& destinationRoot)
	{
		const fs::path sourceRoot = projectRoot / "ForgeTraining" / "GigaLearn" / "GigaLearnInfer";
		WriteTextFile(destinationRoot / "GigaLearnInfer" / "CMakeLists.txt", MakeMinimalGigaLearnCMake(sourceRoot / "CMakeLists.txt"));

		const std::set<fs::path> runtimeFiles = {
			"private/Agent.cpp",
			"private/AgentSetupSnapshot.cpp",
			"private/InferUnit.cpp",
			"private/ModelManager.cpp",
			"private/Models.cpp",
			"public/GigaLearnInfer/Agent.h",
			"public/GigaLearnInfer/AgentComponentRegistry.h",
			"public/GigaLearnInfer/AgentRegistry.h",
			"public/GigaLearnInfer/AgentSetupSnapshot.h",
			"public/GigaLearnInfer/Framework.h",
			"public/GigaLearnInfer/FrameworkTorch.h",
			"public/GigaLearnInfer/InferUnit.h",
			"public/GigaLearnInfer/JsonArchive.h",
			"public/GigaLearnInfer/ModelConfig.h",
			"public/GigaLearnInfer/ModelManager.h",
			"public/GigaLearnInfer/Models.h",
		};
		CopyExplicitFiles(sourceRoot / "src", destinationRoot / "GigaLearnInfer" / "src", runtimeFiles);

		const fs::path jsonSource = projectRoot / "ForgeTraining" / "GigaLearn" / "libsrc" / "json";
		const fs::path jsonDestination = destinationRoot / "libsrc" / "json";
		CopyFile(jsonSource / "LICENSE.MIT", jsonDestination / "LICENSE.MIT");
		CopyFile(jsonSource / "nlohmann" / "json.hpp", jsonDestination / "nlohmann" / "json.hpp");
	}
}
