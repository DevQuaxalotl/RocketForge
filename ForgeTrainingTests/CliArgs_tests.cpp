#include <gtest/gtest.h>

#include "Cli/CliArgs.h"
#include "Cli/ClosestMatches.h"
#include "Cli/CliErrors.h"

#include <functional>
#include <string>
#include <vector>

using namespace RocketForge::Cli;

TEST(CliArgs, ParsesOptionsSwitchesAndPositionals)
{
	const CliOptionSpec spec{ { "--agent" }, { "--json" }, 1 };

	const CliArgs args = CliArgs::Parse({ "segments", "--agent", "a", "--json" }, spec);

	EXPECT_EQ(args.Positionals(), std::vector<std::string>{ "segments" });
	EXPECT_EQ(args.Get("--agent"), std::optional<std::string>("a"));
	EXPECT_TRUE(args.Has("--json"));
	EXPECT_FALSE(args.Get("--build").has_value());
}

TEST(CliArgs, RejectsBadInput)
{
	const CliOptionSpec spec{ { "--agent", "--segment", "--checkpoint" }, { "--json" }, 1 };

	struct Case {
		const char* label;
		std::function<void()> action;
		const char* expectedText;
	};
	const Case cases[] = {
		{ "unknown option suggests the closest one", [&] { CliArgs::Parse({ "--segmnet", "v1.0" }, spec); }, "--segment" },
		{ "value option at the end", [&] { CliArgs::Parse({ "--agent" }, spec); }, "requires a value" },
		{ "value option followed by an option", [&] { CliArgs::Parse({ "--agent", "--json" }, spec); }, "requires a value" },
		{ "empty value", [&] { CliArgs::Parse({ "--segment", "" }, spec); }, "--segment requires a non-empty value" },
		{ "empty value before other options", [&] { CliArgs::Parse({ "--agent", "", "--json" }, spec); }, "--agent requires a non-empty value" },
		{ "repeated option", [&] { CliArgs::Parse({ "--agent", "a", "--agent", "b" }, spec); }, "given twice" },
		{ "repeated switch", [&] { CliArgs::Parse({ "--json", "--json" }, spec); }, "given twice" },
		{ "extra positional", [&] { CliArgs::Parse({ "one", "two" }, spec); }, "unexpected argument" },
		{ "non-integer value", [&] { CliArgs::Parse({ "--checkpoint", "12x" }, spec).GetUint64("--checkpoint"); }, "must be a whole number" },
		{ "negative integer value", [&] { CliArgs::Parse({ "--checkpoint", "-5" }, spec).GetUint64("--checkpoint"); }, "must be a whole number" },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);

		try {
			testCase.action();
			ADD_FAILURE() << "expected CliUsageError";
		} catch (const CliUsageError& error) {
			EXPECT_NE(std::string(error.what()).find(testCase.expectedText), std::string::npos) << error.what();
		}
	}
}

TEST(CliArgs, ParsesNumericValues)
{
	const CliOptionSpec spec{ { "--checkpoint", "--speed" }, {}, 0 };

	const CliArgs args = CliArgs::Parse({ "--checkpoint", "1200", "--speed", "0.5" }, spec);

	EXPECT_EQ(args.GetUint64("--checkpoint"), std::optional<uint64_t>(1200));
	ASSERT_TRUE(args.GetDouble("--speed").has_value());
	EXPECT_NEAR(args.GetDouble("--speed").value(), 0.5, 1e-9);
	EXPECT_THROW(CliArgs::Parse({ "--speed", "fast" }, spec).GetDouble("--speed"), CliUsageError);
}

TEST(ClosestMatches, PrefersPrefixesThenEditDistance)
{
	EXPECT_EQ(ClosestMatches("examp", { "example-agent", "other" }), std::vector<std::string>{ "example-agent" });

	const std::vector<std::string> versions = ClosestMatches("v1.3", { "v1.2", "v1.30", "v2.0" });
	ASSERT_FALSE(versions.empty());
	EXPECT_EQ(versions.front(), "v1.30");

	EXPECT_TRUE(ClosestMatches("zzz", { "abc" }).empty());
	EXPECT_EQ(ClosestMatches("EXAMPLE", { "example-agent" }), std::vector<std::string>{ "example-agent" });
}
