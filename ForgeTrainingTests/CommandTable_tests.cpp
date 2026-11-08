#include <gtest/gtest.h>

#include "Cli/CommandTable.h"
#include "Cli/TextTable.h"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace RocketForge::Cli;

namespace {

	struct CliRun {
		int exitCode = 0;
		std::string out;
		std::string err;
	};

	/** Points stream at target until destroyed, so a throwing test can't leave the stream redirected. */
	class StreamRedirect {
	public:
		StreamRedirect(std::ostream& stream, std::ostream& target) : _stream(stream), _previous(stream.rdbuf(target.rdbuf())) {}
		~StreamRedirect() { _stream.rdbuf(_previous); }

		StreamRedirect(const StreamRedirect&) = delete;
		StreamRedirect& operator=(const StreamRedirect&) = delete;

	private:
		std::ostream& _stream;
		std::streambuf* _previous;
	};

	CliRun RunCliWith(const std::vector<std::string>& arguments)
	{
		std::vector<std::string> storage = arguments;
		std::vector<char*> argv;
		for (std::string& argument : storage)
			argv.push_back(argument.data());
		argv.push_back(nullptr);

		std::ostringstream out;
		std::ostringstream err;
		CliRun run;

		{
			const StreamRedirect redirectOut(std::cout, out);
			const StreamRedirect redirectErr(std::cerr, err);
			run.exitCode = RunCli(static_cast<int>(storage.size()), argv.data());
		}

		run.out = out.str();
		run.err = err.str();
		return run;
	}

}

TEST(CommandTable, DispatchesHelpAndRejectsUnknownCommands)
{
	const CliRun help = RunCliWith({ "prog", "--help" });
	EXPECT_EQ(help.exitCode, 0);

	for (const char* group : { "Bots", "Training", "Inspect", "Models", "Debug & analysis", "Benchmarking", "Script plumbing" })
		EXPECT_NE(help.out.find(group), std::string::npos) << group;

	const CliRun unknown = RunCliWith({ "prog", "lst" });
	EXPECT_EQ(unknown.exitCode, 2);
	EXPECT_NE(unknown.err.find("list"), std::string::npos) << unknown.err;
}

TEST(CommandTable, PrintsPerCommandUsage)
{
	const CliRun viaHelp = RunCliWith({ "prog", "help", "list" });
	const CliRun viaFlag = RunCliWith({ "prog", "list", "--help" });

	EXPECT_EQ(viaHelp.exitCode, 0);
	EXPECT_EQ(viaFlag.exitCode, 0);
	EXPECT_FALSE(viaHelp.out.empty());
	EXPECT_EQ(viaHelp.out, viaFlag.out);

	const CliRun unknownTopic = RunCliWith({ "prog", "help", "lst" });
	EXPECT_EQ(unknownTopic.exitCode, 2);
}

TEST(CommandTable, MapsUsageErrorsToExitTwo)
{
	const CliRun run = RunCliWith({ "prog", "list", "agnets" });

	EXPECT_EQ(run.exitCode, 2);
	EXPECT_EQ(run.err.rfind("list: ", 0), 0u) << run.err;
	EXPECT_NE(run.err.find("agents"), std::string::npos) << run.err;
}

TEST(TextTable, PadsColumnsToTheWidestCell)
{
	const std::string table = FormatTable({ "id", "stage" }, { { "v1.10", "a" }, { "v2.0", "longer" } });

	EXPECT_EQ(table,
		"id     stage\n"
		"-----  ------\n"
		"v1.10  a\n"
		"v2.0   longer\n");
}
