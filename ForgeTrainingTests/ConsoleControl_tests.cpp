#include <gtest/gtest.h>

#include "Cli/ConsoleControl.h"
#include "StageQueue/StageQueueSignals.h"

#include <chrono>
#include <thread>

using namespace RocketForge::Cli;

TEST(ConsoleControl, KeysMapToQueueActions)
{
	struct Case {
		char key;
		ConsoleKeyAction expected;
	};
	const Case cases[] = {
		{ 'q', ConsoleKeyAction::StopQueue },
		{ 'Q', ConsoleKeyAction::StopQueue },
		{ 's', ConsoleKeyAction::SkipStage },
		{ 'S', ConsoleKeyAction::SkipStage },
		{ 'x', ConsoleKeyAction::None },
		{ '\n', ConsoleKeyAction::None },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(std::string("key code ") + std::to_string((int)testCase.key));
		EXPECT_EQ(ConsoleKeyActionFor(testCase.key), testCase.expected);
	}
}

TEST(ConsoleControl, SecondStopRequestFallsThroughToDefault)
{
	ResetStopRequestCountForTests();

	EXPECT_TRUE(HandleStopRequest());
	EXPECT_FALSE(HandleStopRequest());

	ResetStopRequestCountForTests();
	RocketForgeTraining::StageQueueSignalState::Reset();
}

TEST(ConsoleControl, CloseWaitEndsWhenTheRunFinishes)
{
	ResetStopRequestCountForTests();

	EXPECT_FALSE(WaitForRunFinished(std::chrono::milliseconds(20)));

	std::thread finisher([] {
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
		NotifyRunFinished();
	});

	const auto waitStart = std::chrono::steady_clock::now();
	EXPECT_TRUE(WaitForRunFinished(std::chrono::seconds(10)));
	EXPECT_LT(std::chrono::steady_clock::now() - waitStart, std::chrono::seconds(5));
	finisher.join();

	EXPECT_TRUE(WaitForRunFinished(std::chrono::milliseconds(0)));

	ResetStopRequestCountForTests();
}
