#include <gtest/gtest.h>

#include "TestSupport.h"

#ifdef ROCKETFORGE_TESTS_HAVE_QT

#include "RemoteSync/RemoteSyncCommand.h"

#include <QDir>

#include <chrono>
#include <filesystem>
#include <fstream>

using namespace ForgeGUI;
namespace fs = std::filesystem;

namespace
{
	const QString SCRIPT = "C:/repo/tools/script.ps1";

	QStringList ExpectedPrefix(const QString& agent)
	{
		return { "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", SCRIPT, "-Agent", agent };
	}

	void WriteFile(const fs::path& path)
	{
		fs::create_directories(path.parent_path());
		std::ofstream(path) << "exe";
	}
}

TEST(RemoteSyncCommand, ScriptPathsPointAtTheToolsScripts)
{
	EXPECT_EQ(RemoteSyncScriptRelativePath(RemoteSyncKind::Pull), "tools/remote/pull_models.ps1");
	EXPECT_EQ(RemoteSyncScriptRelativePath(RemoteSyncKind::Push), "tools/remote/push_models.ps1");
	EXPECT_EQ(RemoteSyncScriptRelativePath(RemoteSyncKind::PushImage), "tools/remote/push_training_image.ps1");
}

TEST(RemoteSyncCommand, PullArgumentsMapEachOption)
{
	struct Case
	{
		const char* label;
		PullOptions options;
		QStringList expectedTail;
	};

	const Case cases[] = {
		{ "default", { "example-agent", "", false, false }, { "-PlainProgress" } },
		{ "with build", { "example-agent", "Main Build", false, false }, { "-Build", "Main Build", "-PlainProgress" } },
		{ "training only", { "example-agent", "", true, false }, { "-PlainProgress", "-TrainingOnly" } },
		{ "training only + temp", { "example-agent", "", true, true }, { "-PlainProgress", "-TrainingOnly", "-TempCheckpoints" } },
		{ "temp without training only is dropped", { "example-agent", "", false, true }, { "-PlainProgress" } },
		{ "training only + all temp", { "example-agent", "", true, true, true }, { "-PlainProgress", "-TrainingOnly", "-TempCheckpoints", "-AllTempCheckpoints" } },
		{ "all temp without temp is dropped", { "example-agent", "", true, false, true }, { "-PlainProgress", "-TrainingOnly" } },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);
		EXPECT_EQ(BuildPowerShellArguments(SCRIPT, testCase.options), ExpectedPrefix("example-agent") + testCase.expectedTail);
	}
}

TEST(RemoteSyncCommand, PushArgumentsCarryAgentAndBuild)
{
	const QStringList withoutBuild = BuildPowerShellArguments(SCRIPT, PushOptions{ "example-agent", "" });
	const QStringList withBuild = BuildPowerShellArguments(SCRIPT, PushOptions{ "example-agent", "B1" });

	EXPECT_EQ(withoutBuild, ExpectedPrefix("example-agent") + QStringList{ "-PlainProgress" });
	EXPECT_EQ(withBuild, (ExpectedPrefix("example-agent") + QStringList{ "-Build", "B1", "-PlainProgress" }));
}

TEST(RemoteSyncCommand, PushImageArgumentsMapEachOption)
{
	struct Case
	{
		const char* label;
		PushImageOptions options;
		QStringList expectedTail;
	};

	const Case cases[] = {
		{ "defaults, auto tag", { "example-agent", "", "", true, true }, { "-PlainProgress" } },
		{ "explicit tag", { "example-agent", "", "Main-v1.02", true, true }, { "-PlainProgress", "-Tag", "Main-v1.02" } },
		{ "skip latest", { "example-agent", "", "", false, true }, { "-PlainProgress", "-SkipLatest" } },
		{ "skip models", { "example-agent", "", "", true, false }, { "-PlainProgress", "-SkipModels" } },
		{ "everything", { "example-agent", "B1", "t", false, false }, { "-Build", "B1", "-PlainProgress", "-Tag", "t", "-SkipLatest", "-SkipModels" } },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);
		EXPECT_EQ(BuildPowerShellArguments(SCRIPT, testCase.options), ExpectedPrefix("example-agent") + testCase.expectedTail);
	}
}

TEST(RemoteSyncCommand, EmptyAgentProducesNoCommand)
{
	const QStringList pullArguments = BuildPowerShellArguments(SCRIPT, PullOptions{ "", "B1", true, true });
	const QStringList pushArguments = BuildPowerShellArguments(SCRIPT, PushOptions{ "  ", "B1" });
	const QStringList pushImageArguments = BuildPowerShellArguments(SCRIPT, PushImageOptions{ "", "", "t", true, true });

	EXPECT_TRUE(pullArguments.isEmpty());
	EXPECT_TRUE(pushArguments.isEmpty());
	EXPECT_TRUE(pushImageArguments.isEmpty());
}

TEST(RemoteSyncCommand, NamesWithSpacesAndQuotesStaySingleArguments)
{
	const PullOptions options{ "My Agent", "O'Brien Build", false, false };
	const QStringList arguments = BuildPowerShellArguments(SCRIPT, options);

	EXPECT_EQ(arguments.at(arguments.indexOf("-Agent") + 1), "My Agent");
	EXPECT_EQ(arguments.at(arguments.indexOf("-Build") + 1), "O'Brien Build");
}

TEST(RemoteSyncCommand, ParseBuildInfoTagFindsTheTagLine)
{
	struct Case
	{
		const char* label;
		QString output;
		QString expected;
	};

	const Case cases[] = {
		{ "normal", "BUILD_NAME=Main\nTAG=Main-v1.02\nSTATIC=x\n", "Main-v1.02" },
		{ "crlf", "BUILD_NAME=Main\r\nTAG=Main-v1.02\r\n", "Main-v1.02" },
		{ "missing", "BUILD_NAME=Main\n", "" },
		{ "empty", "", "" },
		{ "not a prefix match", "OLD_TAG=x\n", "" },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);
		EXPECT_EQ(ParseBuildInfoTag(testCase.output), testCase.expected);
	}
}

TEST(RemoteSyncCommand, FindNewestConsoleTrainerPicksTheMostRecentBuild)
{
	const fs::path root = RocketForgeTests::TestTempDir("remote-sync-exe");
	const fs::path olderExe = root / "out/build/forge-console/ForgeConsoleTrainer/ForgeConsoleTrainer.exe";
	const fs::path newerExe = root / "out/build/forge-gui/ForgeConsoleTrainer/ForgeConsoleTrainer.exe";
	WriteFile(olderExe);
	WriteFile(newerExe);

	const fs::file_time_type now = fs::file_time_type::clock::now();
	fs::last_write_time(olderExe, now - std::chrono::hours(1));
	fs::last_write_time(newerExe, now);

	const QString found = FindNewestConsoleTrainer(QString::fromStdWString(root.wstring()));

	EXPECT_EQ(QDir::cleanPath(found), QDir::cleanPath(QString::fromStdWString(newerExe.wstring())));
}

TEST(RemoteSyncCommand, FindNewestConsoleTrainerIsEmptyWhenNothingIsBuilt)
{
	const fs::path root = RocketForgeTests::TestTempDir("remote-sync-no-exe");

	EXPECT_TRUE(FindNewestConsoleTrainer(QString::fromStdWString(root.wstring())).isEmpty());
}

TEST(RemoteSyncLineSplitter, SplitsOnNewlinesAcrossChunks)
{
	RemoteSyncLineSplitter splitter;

	EXPECT_EQ(splitter.Append("first\nsec"), QStringList{ "first" });
	EXPECT_EQ(splitter.Append("ond\n\nthird"), (QStringList{ "second", "" }));
	EXPECT_EQ(splitter.Flush(), "third");
	EXPECT_EQ(splitter.Flush(), "");
}

TEST(RemoteSyncLineSplitter, TreatsCrLfAsOneLineEndEvenWhenSplit)
{
	RemoteSyncLineSplitter splitter;

	EXPECT_EQ(splitter.Append("one\r\ntwo\r"), QStringList{ "one" });
	EXPECT_EQ(splitter.Append("\nthree"), QStringList{ "two" });
	EXPECT_EQ(splitter.Flush(), "three");
}

TEST(RemoteSyncLineSplitter, DropsTextOverwrittenByABareCarriageReturn)
{
	RemoteSyncLineSplitter splitter;

	EXPECT_EQ(splitter.Append("10%\r20%\r30%\ndone\n"), (QStringList{ "30%", "done" }));
}

TEST(RemoteSyncLineSplitter, FlushKeepsTheLastRedrawnLine)
{
	RemoteSyncLineSplitter splitter;

	EXPECT_TRUE(splitter.Append("10%\r100%\r").isEmpty());
	EXPECT_EQ(splitter.Flush(), "100%");
}

#else

TEST(RemoteSyncCommand, RequiresQt)
{
	GTEST_SKIP() << "Qt 6 (Core) was not found when configuring; RemoteSyncCommand needs Qt";
}

#endif
