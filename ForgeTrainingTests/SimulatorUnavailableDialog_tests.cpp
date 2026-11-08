#include <gtest/gtest.h>

#include "TestSupport.h"

#ifdef ROCKETFORGE_TESTS_HAVE_QT

#include "SimulatorUnavailableDialog.h"

TEST(QuotedCommandFromReason, ReturnsTheFirstQuotedCommand) {
	struct Case {
		const char* label;
		const char* reason;
		const char* command;
	};
	const Case cases[] = {
		{ "missing submodule",
			"RocketSim v3 isn't built: its source (the ForgeTraining/RLGym/RocketSimV3/upstream submodule) is missing. "
			"Run \"git submodule update --init ForgeTraining/RLGym/RocketSimV3/upstream\", then re-run CMake.",
			"git submodule update --init ForgeTraining/RLGym/RocketSimV3/upstream" },
		{ "two quoted commands", "Run \"first\", then \"second\".", "first" },
		{ "nothing quoted", "RocketSim v3 isn't built: Rust isn't installed. Run tools/setup/install_rust.ps1 (Windows), then rebuild.", "" },
		{ "unterminated quote", "Run \"git submodule update", "" },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);
		EXPECT_EQ(QuotedCommandFromReason(QString::fromUtf8(testCase.reason)).toStdString(), std::string(testCase.command));
	}
}

#else

TEST(QuotedCommandFromReason, RequiresQt) {
	GTEST_SKIP() << "SimulatorUnavailableDialog tests need Qt";
}

#endif
