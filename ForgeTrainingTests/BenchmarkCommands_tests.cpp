#include <gtest/gtest.h>

#include "Cli/CliErrors.h"
#include "Commands/Commands.h"
#include "TestSupport.h"

#include <filesystem>
#include <fstream>

using namespace RocketForge::Cli;

namespace fs = std::filesystem;

TEST(ModelBenchmarkExportPath, CreatesTheMissingFolderAndLeavesNoProbe)
{
	const fs::path folder = RocketForgeTests::TestTempDir("benchmark_export_create") / "results" / "nested";
	const fs::path exportPath = folder / "run.xml";

	PrepareModelBenchmarkExportPath(exportPath);

	EXPECT_TRUE(fs::is_directory(folder));
	EXPECT_TRUE(fs::is_empty(folder));
}

TEST(ModelBenchmarkExportPath, RejectsAnExistingFolder)
{
	const fs::path folder = RocketForgeTests::TestTempDir("benchmark_export_folder") / "results";
	fs::create_directories(folder);

	EXPECT_THROW(PrepareModelBenchmarkExportPath(folder), CliUsageError);
}

TEST(ModelBenchmarkExportPath, FailsWhenTheFolderCannotBeCreated)
{
	const fs::path root = RocketForgeTests::TestTempDir("benchmark_export_blocked");
	fs::create_directories(root);
	std::ofstream(root / "blocker") << "a file where the folder should go";

	try {
		PrepareModelBenchmarkExportPath(root / "blocker" / "run.xml");
		ADD_FAILURE() << "expected CliRuntimeError";
	} catch (const CliRuntimeError& error) {
		EXPECT_NE(std::string(error.what()).find("blocker"), std::string::npos) << error.what();
	}
}
