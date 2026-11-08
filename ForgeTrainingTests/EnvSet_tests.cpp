#include <gtest/gtest.h>

#include "TestSupport.h"

#include <RLGymCPP/ActionParsers/ActionParser.h>
#include <RLGymCPP/EnvSet/EnvSet.h>
#include <RLGymCPP/EnvSet/RewardHeadLayout.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>
#include <RLGymCPP/ObsBuilders/ObsBuilder.h>
#include <RLGymCPP/StateSetters/CombinedState.h>
#include <RLGymCPP/StateSetters/StateSetter.h>
#include <RLGymCPP/TerminalConditions/TerminalCondition.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <thread>

void _BumpCallback(RocketSim::Arena* arena, RocketSim::Car* bumper, RocketSim::Car* victim, bool isDemo, void* userInfo);

namespace {
	struct TinyObsBuilder : RLGC::ObsBuilder {
		RLGC::FList BuildObs(const RLGC::Player& player, const RLGC::GameState& state) override {
			return { player.team == RocketSim::Team::BLUE ? 0.f : 1.f };
		}
	};

	struct TinyActionParser : RLGC::ActionParser {
		RLGC::Action ParseAction(int actionIdx, const RLGC::Player& player, const RLGC::GameState& state) override {
			return {};
		}

		int GetActionAmount() override {
			return 1;
		}
	};

	struct ForwardActionParser : RLGC::ActionParser {
		RLGC::Action ParseAction(int actionIdx, const RLGC::Player&, const RLGC::GameState&) override {
			return { 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
		}

		int GetActionAmount() override {
			return 1;
		}
	};

	struct ResetProbeActionParser : TinyActionParser {
		int resetCount = 0;
		bool retainedEpisodeState = true;

		void ResetRuntimeActionOverride() override {
			resetCount++;
			retainedEpisodeState = false;
		}
	};

	struct ScheduledActionProbeParser : RLGC::ActionParser {
		uint64_t parsedAtTick = UINT64_MAX;

		RLGC::Action ParseAction(int, const RLGC::Player&, const RLGC::GameState& state) override {
			parsedAtTick = state.lastArena->tickCount;
			return { 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
		}

		int GetActionAmount() override {
			return 1;
		}
	};

	struct RuntimeOverrideProbeParser : RLGC::ActionParser {
		mutable int capabilityChecks = 0;
		int requirementChecks = 0;
		int overrides = 0;

		RLGC::Action ParseAction(int, const RLGC::Player&, const RLGC::GameState&) override {
			return {};
		}

		bool HasRuntimeActionOverride() const override {
			capabilityChecks++;
			return true;
		}

		bool RequiresPerTickActionOverride(const RLGC::Player&, const RLGC::GameState&) override {
			requirementChecks++;
			return true;
		}

		RLGC::Action OverrideRuntimeAction(
			const RLGC::Action&,
			const RLGC::Player&,
			const RLGC::GameState&,
			int) override {
			overrides++;
			return { 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
		}

		int GetActionAmount() override {
			return 1;
		}
	};

	struct MacroActionParser : RLGC::ActionParser {
		RLGC::Action firstAction = { 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f };
		RLGC::Action secondAction = { -1.f, 1.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f };

		RLGC::Action ParseAction(int actionIdx, const RLGC::Player& player, const RLGC::GameState& state) override {
			return firstAction;
		}

		void ParseActionMacros(int actionIdx, const RLGC::Player& player, const RLGC::GameState& state, int tickCount, std::vector<RLGC::ActionMacro>& out) override {
			out.clear();
			out.push_back({ firstAction, 4 });
			out.push_back({ secondAction, tickCount - 4 });
		}

		int GetActionAmount() override {
			return 1;
		}
	};

	struct NoOpStateSetter : RLGC::StateSetter {
		void ResetArena(RocketSim::Arena* arena) override {}
	};

	struct CappedStateSetter : RLGC::StateSetter {
		explicit CappedStateSetter(float capSeconds) {
			maxEpisodeSeconds = capSeconds;
		}

		void ResetArena(RocketSim::Arena* arena) override {}
	};

	struct AlwaysTerminalCondition : RLGC::TerminalCondition {
		bool IsTerminal(const RLGC::GameState& currentState) override {
			return true;
		}

		bool IsTruncation() override {
			return false;
		}
	};

	// Spins rather than sleeps so every call is guaranteed to take at least the given time.
	struct SlowReward : RLGC::Reward {
		std::chrono::microseconds callDuration;

		explicit SlowReward(std::chrono::microseconds callDuration) : callDuration(callDuration) {}

		float GetReward(const RLGC::Player&, const RLGC::GameState&, bool) override {
			const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now() + callDuration;
			while (std::chrono::steady_clock::now() < end) {}
			return 0.f;
		}
	};

	// Distinct per-player values so a misrouted player row or head column changes the sums.
	struct NamedPlayerReward : RLGC::Reward {
		std::string name;
		float baseValue;

		NamedPlayerReward(std::string name, float baseValue) : name(std::move(name)), baseValue(baseValue) {}

		float GetReward(const RLGC::Player& player, const RLGC::GameState&, bool) override {
			const float teamOffset = player.team == RocketSim::Team::BLUE ? 0.f : 1.f;
			return baseValue + teamOffset + 0.01f * (float)player.carId;
		}

		std::string GetName() override {
			return name;
		}
	};

	void* UserInfoForIndex(int index) {
		return reinterpret_cast<void*>(static_cast<intptr_t>(index + 1));
	}
}

TEST(EnvSet, ParallelArenaCreationKeepsCallbackMetadataAligned) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	RLGC::EnvSetConfig config = {};
	config.numArenas = 4;
	config.tickSkip = 8;
	config.actionDelay = 0;
	config.saveRewards = false;
	config.envCreateCfg = RLGC::EnvCreateConfig(config.numArenas, (uint8_t)RocketSim::GameMode::SOCCAR);
	config.envCreateFn = [](const int index, const RLGC::EnvCreateConfig& createConfig) {
		std::this_thread::sleep_for(std::chrono::milliseconds((createConfig.numArenas - index) * 5));

		RLGC::EnvCreateResult result = {};
		result.arena = RocketSim::Arena::Create(createConfig.mode);
		result.arena->AddCar(RocketSim::Team::BLUE);
		result.arena->AddCar(RocketSim::Team::ORANGE);
		result.obsBuilder = new TinyObsBuilder();
		result.actionParser = new TinyActionParser();
		result.stateSetter = new NoOpStateSetter();
		result.userInfo = UserInfoForIndex(index);
		return result;
	};

	RLGC::EnvSet envSet(config);

	for (int i = 0; i < config.numArenas; i++) {
		ASSERT_TRUE(envSet.arenas[i] != nullptr);
		ASSERT_TRUE(envSet.eventCallbackInfos[i] != nullptr);
		ASSERT_EQ(envSet.eventCallbackInfos[i]->arenaIdx, i);
		ASSERT_TRUE(envSet.eventCallbackInfos[i]->arena == envSet.arenas[i]);
		ASSERT_TRUE(envSet.eventCallbackInfos[i]->envSet == &envSet);
		ASSERT_TRUE(envSet.userInfos[i] == UserInfoForIndex(i));
		ASSERT_TRUE(envSet.state.gameStates[i].lastArena == envSet.arenas[i]);
		ASSERT_TRUE(envSet.state.gameStates[i].userInfo == UserInfoForIndex(i));
	}

	RocketSim::Car* bumper = nullptr;
	RocketSim::Car* victim = nullptr;
	for (RocketSim::Car* car : envSet.arenas[2]->_cars) {
		if (car->team == RocketSim::Team::BLUE)
			bumper = car;
		else
			victim = car;
	}
	ASSERT_TRUE(bumper != nullptr);
	ASSERT_TRUE(victim != nullptr);
	_BumpCallback(envSet.arenas[2], bumper, victim, true, envSet.eventCallbackInfos[2]);

	for (const RLGC::Player& player : envSet.state.gameStates[0].players) {
		ASSERT_FALSE(player.eventState.bump);
		ASSERT_FALSE(player.eventState.demo);
		ASSERT_FALSE(player.eventState.bumped);
		ASSERT_FALSE(player.eventState.demoed);
	}

	const RLGC::Player* bumperPlayer = nullptr;
	const RLGC::Player* victimPlayer = nullptr;
	for (const RLGC::Player& player : envSet.state.gameStates[2].players) {
		if (player.carId == bumper->id)
			bumperPlayer = &player;
		else if (player.carId == victim->id)
			victimPlayer = &player;
	}

	ASSERT_TRUE(bumperPlayer != nullptr);
	ASSERT_TRUE(victimPlayer != nullptr);
	ASSERT_TRUE(bumperPlayer->eventState.bump);
	ASSERT_TRUE(bumperPlayer->eventState.demo);
	ASSERT_TRUE(victimPlayer->eventState.bumped);
	ASSERT_TRUE(victimPlayer->eventState.demoed);
}

TEST(EnvSet, ResetArenaClearsLiveCarControlsBeforeActionDelay) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	RLGC::EnvSetConfig config = {};
	config.numArenas = 1;
	config.tickSkip = 8;
	config.actionDelay = 3;
	config.saveRewards = false;
	config.envCreateCfg = RLGC::EnvCreateConfig(
		config.numArenas,
		(uint8_t)RocketSim::GameMode::SOCCAR);
	config.envCreateFn = [](const int, const RLGC::EnvCreateConfig& createConfig) {
		RLGC::EnvCreateResult result = {};
		result.arena = RocketSim::Arena::Create(createConfig.mode);
		result.arena->AddCar(RocketSim::Team::BLUE);
		result.obsBuilder = new TinyObsBuilder();
		result.actionParser = new TinyActionParser();
		result.stateSetter = new NoOpStateSetter();
		return result;
	};

	RLGC::EnvSet envSet(config);
	RocketSim::Car* car = *envSet.arenas[0]->_cars.begin();
	car->controls.throttle = 1.f;
	car->controls.steer = -1.f;
	car->controls.jump = true;
	car->controls.boost = true;
	car->controls.handbrake = true;

	envSet.ResetArena(0);

	ASSERT_EQ(car->controls.throttle, 0.f);
	ASSERT_EQ(car->controls.steer, 0.f);
	ASSERT_FALSE(car->controls.jump);
	ASSERT_FALSE(car->controls.boost);
	ASSERT_FALSE(car->controls.handbrake);
}

TEST(EnvSet, ResetArenaClearsActionParserEpisodeState) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	RLGC::EnvSetConfig config = {};
	config.numArenas = 2;
	config.tickSkip = 8;
	config.actionDelay = 0;
	config.saveRewards = false;
	config.envCreateCfg = RLGC::EnvCreateConfig(config.numArenas, (uint8_t)RocketSim::GameMode::SOCCAR);
	config.envCreateFn = [](const int, const RLGC::EnvCreateConfig& createConfig) {
		RLGC::EnvCreateResult result = {};
		result.arena = RocketSim::Arena::Create(createConfig.mode);
		result.arena->AddCar(RocketSim::Team::BLUE);
		result.obsBuilder = new TinyObsBuilder();
		result.actionParser = new ResetProbeActionParser();
		result.stateSetter = new NoOpStateSetter();
		return result;
	};

	RLGC::EnvSet envSet(config);
	auto* first = dynamic_cast<ResetProbeActionParser*>(envSet.actionParsers[0]);
	auto* second = dynamic_cast<ResetProbeActionParser*>(envSet.actionParsers[1]);
	ASSERT_NE(first, nullptr);
	ASSERT_NE(second, nullptr);
	EXPECT_EQ(first->resetCount, 1);
	EXPECT_EQ(second->resetCount, 1);
	EXPECT_FALSE(first->retainedEpisodeState);
	EXPECT_FALSE(second->retainedEpisodeState);

	first->retainedEpisodeState = true;
	second->retainedEpisodeState = true;
	envSet.ResetArena(0);

	EXPECT_EQ(first->resetCount, 2);
	EXPECT_FALSE(first->retainedEpisodeState);
	EXPECT_EQ(second->resetCount, 1);
	EXPECT_TRUE(second->retainedEpisodeState);

	first->retainedEpisodeState = true;
	envSet.state.terminals[0] = (uint8_t)RLGC::TerminalType::NORMAL;
	envSet.state.terminals[1] = (uint8_t)RLGC::TerminalType::TRUNCATED;
	envSet.Reset();

	EXPECT_EQ(first->resetCount, 3);
	EXPECT_EQ(second->resetCount, 2);
	EXPECT_FALSE(first->retainedEpisodeState);
	EXPECT_FALSE(second->retainedEpisodeState);
}

TEST(EnvSet, MacroActionParserAppliesControlsWithinTickSkip) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	RLGC::EnvSetConfig config = {};
	config.numArenas = 1;
	config.tickSkip = 8;
	config.actionDelay = 0;
	config.saveRewards = false;
	config.envCreateCfg = RLGC::EnvCreateConfig(config.numArenas, (uint8_t)RocketSim::GameMode::SOCCAR);
	config.envCreateFn = [](const int index, const RLGC::EnvCreateConfig& createConfig) {
		RLGC::EnvCreateResult result = {};
		result.arena = RocketSim::Arena::Create(createConfig.mode);
		result.arena->AddCar(RocketSim::Team::BLUE);
		result.obsBuilder = new TinyObsBuilder();
		result.actionParser = new MacroActionParser();
		result.stateSetter = new NoOpStateSetter();
		return result;
	};

	RLGC::EnvSet envSet(config);
	envSet.StepSecondHalf(RLGC::IList{ 0 }, false);

	RocketSim::Car* car = *envSet.arenas[0]->_cars.begin();
	ASSERT_EQ(car->controls.throttle, -1.f);
	ASSERT_EQ(car->controls.steer, 1.f);
	ASSERT_EQ(car->controls.yaw, 1.f);
	ASSERT_TRUE(car->controls.handbrake);

	// prevAction reflects the LAST sub-action applied within the tick skip, matching RLBot
	// inference (which sets prevAction from the macro's back()). The next observation is taken
	// after the final macro segment ran, so "previous action" is secondAction, not firstAction.
	const RLGC::Action& prevAction = envSet.state.gameStates[0].players[0].prevAction;
	ASSERT_EQ(prevAction.throttle, -1.f);
	ASSERT_EQ(prevAction.steer, 1.f);
	ASSERT_EQ(prevAction.yaw, 1.f);
	ASSERT_EQ(prevAction.handbrake, 1.f);
	ASSERT_EQ(prevAction.jump, 0.f);
	ASSERT_EQ(envSet.arenas[0]->tickCount, (uint64_t)8);
}

TEST(EnvSet, PerPlayerActionParsersSupportHeterogeneousOpponents) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	RLGC::EnvSetConfig config = {};
	config.numArenas = 1;
	config.tickSkip = 8;
	config.actionDelay = 7;
	config.saveRewards = false;
	config.envCreateCfg = RLGC::EnvCreateConfig(
		config.numArenas,
		(uint8_t)RocketSim::GameMode::SOCCAR);
	config.envCreateFn = [](const int, const RLGC::EnvCreateConfig& createConfig) {
		RLGC::EnvCreateResult result = {};
		result.arena = RocketSim::Arena::Create(createConfig.mode);
		result.arena->AddCar(RocketSim::Team::BLUE);
		result.arena->AddCar(RocketSim::Team::ORANGE);
		result.obsBuilder = new TinyObsBuilder();
		result.actionParser = new TinyActionParser();
		result.stateSetter = new NoOpStateSetter();
		return result;
	};

	RLGC::EnvSet envSet(config);
	ForwardActionParser opponentParser;
	std::vector<RLGC::ActionParser*> parsers = {
		envSet.actionParsers[0],
		&opponentParser
	};

	envSet.StepFirstHalf(false);
	envSet.StepSecondHalf(RLGC::IList{ 0, 0 }, parsers, false);

	const auto& players = envSet.state.gameStates[0].players;
	ASSERT_EQ(players[0].prevAction.throttle, 0.f);
	ASSERT_EQ(players[1].prevAction.throttle, 1.f);
}

TEST(EnvSet, PerPlayerRuntimeOverrideCoversActionDelayAndActionWindow) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	RLGC::EnvSetConfig config = {};
	config.numArenas = 1;
	config.tickSkip = 8;
	config.actionDelay = 7;
	config.saveRewards = false;
	config.envCreateCfg = RLGC::EnvCreateConfig(
		config.numArenas,
		(uint8_t)RocketSim::GameMode::SOCCAR);
	config.envCreateFn = [](const int, const RLGC::EnvCreateConfig& createConfig) {
		RLGC::EnvCreateResult result = {};
		result.arena = RocketSim::Arena::Create(createConfig.mode);
		result.arena->AddCar(RocketSim::Team::BLUE);
		result.arena->AddCar(RocketSim::Team::ORANGE);
		result.obsBuilder = new TinyObsBuilder();
		result.actionParser = new TinyActionParser();
		result.stateSetter = new NoOpStateSetter();
		return result;
	};

	RLGC::EnvSet envSet(config);
	RuntimeOverrideProbeParser opponentParser;
	std::vector<RLGC::ActionParser*> parsers = {
		envSet.actionParsers[0],
		&opponentParser
	};

	envSet.StepFirstHalf(false, parsers);
	envSet.StepSecondHalf(RLGC::IList{ 0, 0 }, parsers, false);

	ASSERT_EQ(opponentParser.overrides, 8);
	ASSERT_EQ(envSet.arenas[0]->tickCount, (uint64_t)8);
	ASSERT_EQ(envSet.state.gameStates[0].players[1].prevAction.throttle, 1.f);
}

TEST(EnvSet, ScheduledActionAppliesAtExactNativeTick) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	RLGC::EnvSetConfig config = {};
	config.numArenas = 1;
	config.tickSkip = 4;
	config.actionDelay = 3;
	config.saveRewards = false;
	config.envCreateCfg = RLGC::EnvCreateConfig(
		config.numArenas,
		(uint8_t)RocketSim::GameMode::SOCCAR);
	config.envCreateFn = [](const int, const RLGC::EnvCreateConfig& createConfig) {
		RLGC::EnvCreateResult result = {};
		result.arena = RocketSim::Arena::Create(createConfig.mode);
		result.arena->AddCar(RocketSim::Team::BLUE);
		result.arena->AddCar(RocketSim::Team::ORANGE);
		result.arena->AddCar(RocketSim::Team::ORANGE);
		result.obsBuilder = new TinyObsBuilder();
		result.actionParser = new TinyActionParser();
		result.stateSetter = new NoOpStateSetter();
		return result;
	};

	RLGC::EnvSet envSet(config);
	ScheduledActionProbeParser zeroDelayParser;
	ScheduledActionProbeParser twoTickDelayParser;
	std::vector<RLGC::ActionParser*> parsers = {
		envSet.actionParsers[0],
		&zeroDelayParser,
		&twoTickDelayParser
	};

	envSet.StepFirstHalf(
		false,
		parsers,
		RLGC::IList{ 0, 0, 0 },
		RLGC::IList{ -1, 0, 2 });

	ASSERT_EQ(zeroDelayParser.parsedAtTick, (uint64_t)0);
	ASSERT_EQ(twoTickDelayParser.parsedAtTick, (uint64_t)2);
	ASSERT_EQ(envSet.arenas[0]->tickCount, (uint64_t)3);
	ASSERT_EQ((*std::next(envSet.arenas[0]->_cars.begin()))->controls.throttle, 1.f);
}

TEST(EnvSet, EpisodeTimeAccumulatesAndResetsWithArena) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	RLGC::EnvSetConfig config = {};
	config.numArenas = 1;
	config.tickSkip = 8;
	config.actionDelay = 0;
	config.saveRewards = false;
	config.envCreateCfg = RLGC::EnvCreateConfig(
		config.numArenas,
		(uint8_t)RocketSim::GameMode::SOCCAR);
	config.envCreateFn = [](const int, const RLGC::EnvCreateConfig& createConfig) {
		RLGC::EnvCreateResult result = {};
		result.arena = RocketSim::Arena::Create(createConfig.mode);
		result.arena->AddCar(RocketSim::Team::BLUE);
		result.obsBuilder = new TinyObsBuilder();
		result.actionParser = new TinyActionParser();
		result.stateSetter = new NoOpStateSetter();
		return result;
	};

	RLGC::EnvSet envSet(config);
	ASSERT_EQ(envSet.state.gameStates[0].episodeTime, 0.f);

	const float stepSeconds = config.tickSkip / 120.f;
	envSet.StepSecondHalf(RLGC::IList{ 0 }, false);
	ASSERT_TRUE(std::fabs(envSet.state.gameStates[0].episodeTime - stepSeconds) < 1e-5f);
	envSet.StepSecondHalf(RLGC::IList{ 0 }, false);
	ASSERT_TRUE(std::fabs(envSet.state.gameStates[0].episodeTime - 2.f * stepSeconds) < 1e-5f);

	envSet.ResetArena(0);
	ASSERT_EQ(envSet.state.gameStates[0].episodeTime, 0.f);
}

TEST(CombinedState, ForwardsChosenChildEpisodeCap) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	NoOpStateSetter child;
	child.maxEpisodeSeconds = 3.f;
	RLGC::CombinedState combined({ { &child, 1.f } });
	ASSERT_FALSE(combined.maxEpisodeSeconds > 0.f);

	RocketSim::Arena* arena = RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR);
	combined.ResetArena(arena);
	delete arena;

	ASSERT_EQ(combined.maxEpisodeSeconds, 3.f);

	RLGC::GameState capped;
	capped.episodeTime = 3.f;
	ASSERT_TRUE(combined.EpisodeCapElapsed(capped));
	capped.episodeTime = 2.9f;
	ASSERT_FALSE(combined.EpisodeCapElapsed(capped));
	ASSERT_FALSE(child.EpisodeCapElapsed(capped));

	NoOpStateSetter uncapped;
	capped.episodeTime = 1000.f;
	ASSERT_FALSE(uncapped.EpisodeCapElapsed(capped));
}

TEST(EnvSet, SetterEpisodeCapTruncatesOnceElapsed) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	RLGC::EnvSetConfig config = {};
	config.numArenas = 1;
	config.tickSkip = 8;
	config.actionDelay = 0;
	config.saveRewards = false;
	config.envCreateCfg = RLGC::EnvCreateConfig(
		config.numArenas,
		(uint8_t)RocketSim::GameMode::SOCCAR);
	config.envCreateFn = [](const int, const RLGC::EnvCreateConfig& createConfig) {
		RLGC::EnvCreateResult result = {};
		result.arena = RocketSim::Arena::Create(createConfig.mode);
		result.arena->AddCar(RocketSim::Team::BLUE);
		result.obsBuilder = new TinyObsBuilder();
		result.actionParser = new TinyActionParser();

		// 8 ticks per step is 0.0667 s, so the cap elapses on the third step
		result.stateSetter = new CappedStateSetter(0.15f);
		return result;
	};

	RLGC::EnvSet envSet(config);

	envSet.StepSecondHalf(RLGC::IList{ 0 }, false);
	ASSERT_EQ((int)envSet.state.terminals[0], (int)RLGC::TerminalType::NOT_TERMINAL);
	envSet.StepSecondHalf(RLGC::IList{ 0 }, false);
	ASSERT_EQ((int)envSet.state.terminals[0], (int)RLGC::TerminalType::NOT_TERMINAL);
	envSet.StepSecondHalf(RLGC::IList{ 0 }, false);
	ASSERT_EQ((int)envSet.state.terminals[0], (int)RLGC::TerminalType::TRUNCATED);
}

TEST(EnvSet, NormalTerminalTakesPriorityOverEpisodeCap) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	RLGC::EnvSetConfig config = {};
	config.numArenas = 1;
	config.tickSkip = 8;
	config.actionDelay = 0;
	config.saveRewards = false;
	config.envCreateCfg = RLGC::EnvCreateConfig(
		config.numArenas,
		(uint8_t)RocketSim::GameMode::SOCCAR);
	config.envCreateFn = [](const int, const RLGC::EnvCreateConfig& createConfig) {
		RLGC::EnvCreateResult result = {};
		result.arena = RocketSim::Arena::Create(createConfig.mode);
		result.arena->AddCar(RocketSim::Team::BLUE);
		result.obsBuilder = new TinyObsBuilder();
		result.actionParser = new TinyActionParser();
		result.stateSetter = new CappedStateSetter(0.01f);
		result.terminalConditions = { new AlwaysTerminalCondition() };
		return result;
	};

	RLGC::EnvSet envSet(config);

	envSet.StepSecondHalf(RLGC::IList{ 0 }, false);
	ASSERT_EQ((int)envSet.state.terminals[0], (int)RLGC::TerminalType::NORMAL);
}

namespace {
	struct CountingReward : RLGC::Reward {
		int calls = 0;

		float GetReward(const RLGC::Player&, const RLGC::GameState&, bool) override {
			calls++;
			return 1.f;
		}
	};
}

TEST(EnvSet, NonFiniteGameStateEndsEpisodeBeforeRewards) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	CountingReward* reward = nullptr;
	RLGC::EnvSetConfig config = {};
	config.numArenas = 1;
	config.tickSkip = 8;
	config.actionDelay = 0;
	config.saveRewards = false;
	config.envCreateCfg = RLGC::EnvCreateConfig(config.numArenas, (uint8_t)RocketSim::GameMode::SOCCAR);
	config.envCreateFn = [&reward](const int, const RLGC::EnvCreateConfig& createConfig) {
		RLGC::EnvCreateResult result = {};
		result.arena = RocketSim::Arena::Create(createConfig.mode);
		result.arena->AddCar(RocketSim::Team::BLUE);
		result.arena->AddCar(RocketSim::Team::ORANGE);
		result.obsBuilder = new TinyObsBuilder();
		result.actionParser = new TinyActionParser();
		result.stateSetter = new NoOpStateSetter();

		reward = new CountingReward();
		result.rewards = { RLGC::WeightedReward(reward, 1.f) };
		return result;
	};
	RLGC::EnvSet envSet(config);
	const bool async = false;

	RocketSim::BallState ballState = envSet.arenas[0]->ball->GetState();
	ballState.vel = RocketSim::Vec(NAN, 0, 0);
	envSet.arenas[0]->ball->SetState(ballState);

	envSet.StepSecondHalf(RLGC::IList(envSet.state.numPlayers, 0), async);
	ASSERT_EQ((int)envSet.state.terminals[0], (int)RLGC::TerminalType::NORMAL);
	ASSERT_EQ(reward->calls, 0);

	for (int playerIdx = 0; playerIdx < envSet.state.numPlayers; playerIdx++) {
		ASSERT_EQ(envSet.state.rewards[playerIdx], 0.f);
		for (int obsIdx = 0; obsIdx < (int)envSet.state.obs.size[1]; obsIdx++)
			ASSERT_EQ(envSet.state.obs.At(playerIdx, obsIdx), 0.f);
	}
}

TEST(EnvSet, TakeRewardTimeAccumulatesRewardCpuTimeAndClears) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	const std::chrono::microseconds rewardCallDuration(200);
	RLGC::EnvSetConfig config = {};
	config.numArenas = 4;
	config.tickSkip = 8;
	config.actionDelay = 0;
	config.saveRewards = false;
	config.envCreateCfg = RLGC::EnvCreateConfig(config.numArenas, (uint8_t)RocketSim::GameMode::SOCCAR);
	config.envCreateFn = [rewardCallDuration](const int, const RLGC::EnvCreateConfig& createConfig) {
		RLGC::EnvCreateResult result = {};
		result.arena = RocketSim::Arena::Create(createConfig.mode);
		result.arena->AddCar(RocketSim::Team::BLUE);
		result.obsBuilder = new TinyObsBuilder();
		result.actionParser = new TinyActionParser();
		result.stateSetter = new NoOpStateSetter();
		result.rewards = { RLGC::WeightedReward(new SlowReward(rewardCallDuration), 1.f) };
		return result;
	};

	RLGC::EnvSet envSet(config);

	const int numSteps = 3;
	const bool async = false;
	for (int step = 0; step < numSteps; step++)
		envSet.StepSecondHalf(RLGC::IList(envSet.state.numPlayers, 0), async);

	// One car per arena, so every arena-step makes exactly one slow reward call.
	const double minRewardSeconds = numSteps * config.numArenas * std::chrono::duration<double>(rewardCallDuration).count();
	ASSERT_TRUE(envSet.TakeRewardTime() >= minRewardSeconds);
	ASSERT_EQ(envSet.TakeRewardTime(), 0.0);
}

namespace {
	// Arena 0 is 1v1 with { Alpha, Beta }; arena 1 is 2v2 with { Beta, zero-sum Gamma }.
	RLGC::EnvSetConfig MakeMixedRewardEnvConfig() {
		RLGC::EnvSetConfig config = {};
		config.numArenas = 2;
		config.tickSkip = 8;
		config.actionDelay = 0;
		config.saveRewards = false;
		config.envCreateCfg = RLGC::EnvCreateConfig(config.numArenas, (uint8_t)RocketSim::GameMode::SOCCAR);
		config.envCreateFn = [](const int index, const RLGC::EnvCreateConfig& createConfig) {
			RLGC::EnvCreateResult result = {};
			result.arena = RocketSim::Arena::Create(createConfig.mode);

			const int teamSize = index == 0 ? 1 : 2;
			for (int i = 0; i < teamSize; i++) {
				result.arena->AddCar(RocketSim::Team::BLUE);
				result.arena->AddCar(RocketSim::Team::ORANGE);
			}

			result.obsBuilder = new TinyObsBuilder();
			result.actionParser = new TinyActionParser();
			result.stateSetter = new NoOpStateSetter();

			if (index == 0) {
				result.rewards = {
					RLGC::WeightedReward(new NamedPlayerReward("Alpha", 0.25f), 2.f),
					RLGC::WeightedReward(new NamedPlayerReward("Beta", 0.5f), 3.f),
				};
			} else {
				const float teamSpirit = 0.5f;
				result.rewards = {
					RLGC::WeightedReward(new NamedPlayerReward("Beta", 0.5f), 5.f),
					RLGC::WeightedReward(new RLGC::ZeroSumReward(new NamedPlayerReward("Gamma", 0.75f), teamSpirit), 7.f),
				};
			}
			return result;
		};
		return config;
	}
}

TEST(EnvSet, HeadRewardsAreEmptyWithoutLayout) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	RLGC::EnvSet envSet(MakeMixedRewardEnvConfig());
	const bool async = false;
	envSet.StepSecondHalf(RLGC::IList(envSet.state.numPlayers, 0), async);

	ASSERT_FALSE(envSet.rewardHeadLayout.has_value());
	ASSERT_TRUE(envSet.state.headRewards.data.empty());
}

TEST(EnvSet, WeightedHeadRewardsSumToPlayerReward) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	RLGC::EnvSet envSet(MakeMixedRewardEnvConfig());
	envSet.SetRewardHeadLayout(RLGC::RewardHeadLayout::Build(envSet.rewards));

	const bool async = false;
	envSet.StepSecondHalf(RLGC::IList(envSet.state.numPlayers, 0), async);

	const RLGC::RewardHeadLayout& layout = *envSet.rewardHeadLayout;
	RLGC::DimList2<float> weights = layout.BuildWeightTable(envSet.rewards);
	ASSERT_EQ(layout.GetNumHeads(), 3);
	ASSERT_EQ(envSet.state.headRewards.size[0], (size_t)envSet.state.numPlayers);
	ASSERT_EQ(envSet.state.headRewards.size[1], (size_t)3);

	for (int arenaIdx = 0; arenaIdx < (int)envSet.arenas.size(); arenaIdx++) {
		const int playerStartIdx = envSet.state.arenaPlayerStartIdx[arenaIdx];
		const int numArenaPlayers = (int)envSet.state.gameStates[arenaIdx].players.size();

		for (int playerIdx = playerStartIdx; playerIdx < playerStartIdx + numArenaPlayers; playerIdx++) {
			float weightedSum = 0.f;
			for (int headIdx = 0; headIdx < layout.GetNumHeads(); headIdx++)
				weightedSum += weights.At(arenaIdx, headIdx) * envSet.state.headRewards.At(playerIdx, headIdx);

			ASSERT_TRUE(std::fabs(weightedSum - envSet.state.rewards[playerIdx]) < 1e-5f);
		}
	}
}

TEST(EnvSet, AbsentHeadRewardsStayZero) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	RLGC::EnvSet envSet(MakeMixedRewardEnvConfig());
	envSet.SetRewardHeadLayout(RLGC::RewardHeadLayout::Build(envSet.rewards));

	const bool async = false;
	for (int step = 0; step < 3; step++)
		envSet.StepSecondHalf(RLGC::IList(envSet.state.numPlayers, 0), async);

	// "Gamma" (head 2) is only listed in arena 1, so arena 0's two players never receive it.
	const int gammaHead = 2;
	ASSERT_EQ(envSet.rewardHeadLayout->GetHeadNames()[gammaHead], std::string("Gamma"));
	ASSERT_EQ(envSet.state.headRewards.At(0, gammaHead), 0.f);
	ASSERT_EQ(envSet.state.headRewards.At(1, gammaHead), 0.f);
	ASSERT_TRUE(envSet.state.headRewards.At(2, gammaHead) != 0.f);
}

namespace {
	struct BufferProbeReward : RLGC::Reward {
		int step = 0;
		float* storage = nullptr;
		bool reused = true;
		int evaluations = 0;
		std::string name;

		explicit BufferProbeReward(std::string name) : name(std::move(name)) {}

		void Reset(const RLGC::GameState&) override { step = 0; }

		void PreStep(const RLGC::GameState&) override { step++; }

		std::string GetName() override { return name; }

		float GetReward(const RLGC::Player& player, const RLGC::GameState&, bool) override {
			return (float)(step + player.index);
		}

		void GetAllRewardsInto(const RLGC::GameState& state, bool final, std::vector<float>& out) override {
			out.resize(state.players.size());
			if (storage && storage != out.data())
				reused = false;

			storage = out.data();
			evaluations++;
			for (size_t i = 0; i < state.players.size(); i++)
				out[i] = GetReward(state.players[i], state, final);
		}
	};
}

TEST(EnvSet, RewardBuffersAreReusedPerArenaWithoutStaleTotalsOrDebugValues) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	RLGC::EnvSetConfig config = MakeMixedRewardEnvConfig();
	RLGC::EnvCreateFn create = config.envCreateFn;
	config.saveRewards = true;
	config.rewardSamplePlayerOverride = 0;
	config.envCreateFn = [create](int index, const RLGC::EnvCreateConfig& createConfig) {
		RLGC::EnvCreateResult result = create(index, createConfig);
		for (RLGC::WeightedReward& weighted : result.rewards)
			delete weighted.reward;

		result.rewards = {
			RLGC::WeightedReward(new BufferProbeReward("Plain"), 2.f),
			RLGC::WeightedReward(new RLGC::ZeroSumReward(new BufferProbeReward("Wrapped"), 0.5f), 3.f)
		};
		return result;
	};

	RLGC::EnvSet envSet(config);
	envSet.SetRewardHeadLayout(RLGC::RewardHeadLayout::Build(envSet.rewards));

	const bool async = false;
	const bool isFinal = false;
	for (int iteration = 0; iteration < 6; iteration++) {
		if (iteration == 3) {
			envSet.ResetArena(0);
			envSet.ResetArena(1);
		}

		envSet.StepSecondHalf(RLGC::IList(envSet.state.numPlayers, 0), async);
		const int step = iteration % 3 + 1;

		for (int arena = 0; arena < 2; arena++) {
			const RLGC::GameState& state = envSet.state.gameStates[arena];
			auto* plain = static_cast<BufferProbeReward*>(envSet.rewards[arena][0].reward);
			auto* wrapped = static_cast<RLGC::ZeroSumReward*>(envSet.rewards[arena][1].reward);
			auto* child = static_cast<BufferProbeReward*>(wrapped->child);
			ASSERT_TRUE(plain->reused);
			ASSERT_TRUE(child->reused);
			ASSERT_EQ(plain->storage, child->storage);
			ASSERT_EQ(plain->evaluations, iteration + 1);

			std::vector<float> raw = plain->GetAllRewards(state, isFinal);
			BufferProbeReward* referenceChild = new BufferProbeReward("Reference");
			referenceChild->step = step;
			RLGC::ZeroSumReward reference(referenceChild, 0.5f);
			std::vector<float> post = static_cast<RLGC::Reward&>(reference).GetAllRewards(state, isFinal);

			for (int player = 0; player < (int)state.players.size(); player++) {
				const int row = envSet.state.arenaPlayerStartIdx[arena] + player;
				EXPECT_FLOAT_EQ(raw[player], step + state.players[player].index);
				EXPECT_FLOAT_EQ(envSet.state.headRewards.At(row, 0), raw[player]);
				EXPECT_FLOAT_EQ(envSet.state.headRewards.At(row, 1), post[player]);
				EXPECT_FLOAT_EQ(envSet.state.rewards[row], raw[player] * 2.f + post[player] * 3.f);
				EXPECT_FLOAT_EQ(wrapped->_lastRewards[player], raw[player]);
			}

			EXPECT_FLOAT_EQ(envSet.state.lastRewards[arena][1], raw[0]);
			if (arena == 0)
				EXPECT_FLOAT_EQ(envSet.state.lastTrainingRewards[1], post[0]);
		}

		const auto* first = static_cast<BufferProbeReward*>(envSet.rewards[0][0].reward);
		const auto* second = static_cast<BufferProbeReward*>(envSet.rewards[1][0].reward);
		EXPECT_NE(first->storage, second->storage);
	}
}
