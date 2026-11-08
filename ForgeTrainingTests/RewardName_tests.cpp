#include <gtest/gtest.h>

#include <RLGymCPP/Rewards/CommonRewards.h>

#include <string>

using namespace RLGC;

// The default Reward::GetName() derives from typeid(*this).name(), which renders the
// PlayerDataEventReward template differently on MSVC (Windows local) than GCC (Linux
// pod). That divergence changes the training-setup hash across platforms and spuriously
// forks a segment when a Windows-trained run is continued on a Linux pod. These tests
// pin PlayerDataEventReward to an explicit, compiler-independent name.

TEST(PlayerDataEventReward, GetName_IsExplicitAndStable)
{
	PlayerGoalReward goal;
	AssistReward assist;
	BumpedPenalty bumped;
	DemoedPenalty demoed;

	ASSERT_EQ(goal.GetName(),   std::string("PlayerEventGoalReward"));
	ASSERT_EQ(assist.GetName(), std::string("PlayerEventAssistReward"));
	ASSERT_EQ(bumped.GetName(), std::string("PlayerEventBumpedPenalty"));
	ASSERT_EQ(demoed.GetName(), std::string("PlayerEventDemoedPenalty"));
}

TEST(PlayerDataEventReward, GetName_HasNoCompilerTypeName)
{
	// No template angle brackets: proves we are not falling back to typeid().name().
	PlayerGoalReward goal;
	SaveReward save;

	ASSERT_TRUE(goal.GetName().find('<') == std::string::npos);
	ASSERT_TRUE(save.GetName().find('<') == std::string::npos);
}

TEST(PlayerDataEventReward, GetName_DistinctPerEvent)
{
	ShotReward shot;
	ShotPassReward shotPass;

	ASSERT_TRUE(shot.GetName() != shotPass.GetName());
}

TEST(BoostReward, PartNamesAreStable)
{
	auto boost = std::make_shared<RLGC::BoostReward>();
	ASSERT_EQ(boost->GetName(), std::string("BoostReward"));
	ASSERT_EQ(boost->Part(RLGC::BoostReward::CONT)->GetName(), std::string("BoostReward_Cont"));
	ASSERT_EQ(boost->Part(RLGC::BoostReward::EVENT)->GetName(), std::string("BoostReward_Event"));
}
