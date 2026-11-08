#include <gtest/gtest.h>

#include "Cli/Confirm.h"

#include <sstream>
#include <string>

using namespace RocketForge::Cli;

TEST(Confirm, RefusesNonInteractiveWithoutYes)
{
	struct Case {
		const char* label;
		bool yesFlag;
		bool interactive;
		std::string input;
		ConfirmResult expected;
	};
	const Case cases[] = {
		{ "non-interactive without --yes", false, false, "y\n", ConfirmResult::RefusedNonInteractive },
		{ "--yes skips the prompt", true, false, "", ConfirmResult::Confirmed },
		{ "interactive y", false, true, "y\n", ConfirmResult::Confirmed },
		{ "interactive blank line", false, true, "\n", ConfirmResult::Declined },
		{ "interactive EOF", false, true, "", ConfirmResult::Declined },
		{ "interactive n", false, true, "n\n", ConfirmResult::Declined },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);

		std::istringstream in(testCase.input);
		std::ostringstream out;

		const ConfirmResult result = ConfirmDestructive("Delete segment v1.0", testCase.yesFlag, testCase.interactive, in, out);

		EXPECT_EQ(result, testCase.expected);
		EXPECT_NE(out.str().find("Delete segment v1.0"), std::string::npos);

		if (testCase.expected == ConfirmResult::RefusedNonInteractive)
			EXPECT_NE(out.str().find("refusing without --yes (no interactive terminal)"), std::string::npos);
	}
}
