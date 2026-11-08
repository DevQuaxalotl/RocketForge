#include <gtest/gtest.h>

#ifdef ROCKETFORGE_TESTS_HAVE_QT

#include "DebugBridge.h"

TEST(DebugBridge, GameControlsAndFocusApplyWithoutRewardCollector) {
	ForgeDebug::DebugSettings initial;
	initial.gameSpeed = 3.f;
	initial.playerIndex = 1;

	DebugBridge bridge;
	bridge.InitializeFromSettings(initial);

	EXPECT_EQ(bridge.TakePendingPlayerIndex(), 1);
	EXPECT_EQ(bridge.TakePendingPlayerIndex(), -1);

	bridge.RequestPlayerIndex(2);
	bridge.RequestGameSpeed(2.f);
	bridge.RequestMaxEpisodeTimeEnabled(true);
	bridge.RequestMaxEpisodeTime(4.f);
	bridge.FlushGameControlPending();

	EXPECT_EQ(bridge.TakePendingPlayerIndex(), 2);
	EXPECT_FLOAT_EQ(bridge.CurrentSettings().gameSpeed, 2.f);
	EXPECT_TRUE(bridge.CurrentSettings().maxEpisodeTimeEnabled);
	EXPECT_FLOAT_EQ(bridge.CurrentSettings().maxEpisodeTime, 4.f);
}

#else

TEST(DebugBridge, RequiresQt) {
	GTEST_SKIP() << "DebugBridge tests need Qt";
}

#endif
