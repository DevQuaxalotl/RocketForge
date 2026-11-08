#include "TestSupport.h"

#include <ModelBenchmark/ModelBenchmark.h>
#include <ModelBenchmark/ModelBenchmarkLiveGames.h>
#include <ModelBenchmark/BenchmarkStartingStates.h>
#include <ModelBenchmark/ModelBenchmarkInternal.h>
#include <Settings/ForgeSettings.h>

#include <GigaLearnCPP/Eval/PolicyMatchCudaOomRecovery.h>
#include <GigaLearnInfer/ModelManager.h>
#include <GigaLearnInfer/AgentRegistry.h>
#include <GigaLearnInfer/AgentSetupSnapshot.h>
#include <TrainingSetup/TrainingSetupSnapshot.h>
#include <ATen/CPUGeneratorImpl.h>
#include <torch/cuda.h>
#ifdef RG_CUDA_SUPPORT
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

// GTest fatal assertions run in a void scope so value-returning test callbacks compile.
// TestSupport throws on a fatal assertion, aborting the entire test safely.
#define BENCHMARK_CHECK_EQ(a, b) ([&]() { ASSERT_EQ(a, b); }())
#define BENCHMARK_CHECK_FALSE(value) ([&]() { ASSERT_FALSE(value); }())
#define BENCHMARK_CHECK_GT(a, b) ([&]() { ASSERT_GT(a, b); }())
#define BENCHMARK_CHECK_TRUE(value) ([&]() { ASSERT_TRUE(value); }())

using namespace RocketForgeTraining;
using namespace RocketForgeTraining::ModelBenchmarkInternal;

TEST(ModelBenchmark, AgentSetupLoadsCurrentSavedAndCapturedSettings)
{
	AgentRegistry::Register("BenchmarkSetupFixture", []() { return std::make_unique<Agent>(); });
	ModelBenchmarkModel model;
	model.agentName = "BenchmarkSetupFixture";
	model.displayName = "Setup fixture";
	model.agentSetupSource = ModelBenchmarkAgentSetupSource::CurrentAgent;
	std::unique_ptr<Agent> current = CreateBenchmarkAgent(model);
	RocketForge::AgentSetupSnapshot expected = RocketForge::CaptureAgentSetupSnapshot(*current);
	BENCHMARK_CHECK_TRUE(RocketForge::SameAgentSetupSnapshot(expected,
		RocketForge::CaptureAgentSetupSnapshot(*CreateBenchmarkAgent(model))));

	expected.agentName = "captured-agent";
	expected.obsBuilderName = "captured-obs";
	expected.actionParserName = "captured-action";
	expected.gameMode = RocketSim::GameMode::HOOPS;
	expected.carConfig = RocketSim::CAR_CONFIG_PLANK;
	expected.tickSkip = 4;
	expected.actionDelay = 3;
	expected.maxTeamSize = 3;
	expected.useBallPrediction = true;
	expected.addLayerNorm = false;
	expected.optimType = GGL::ModelOptimType::ADAMW;
	expected.activationType = GGL::ModelActivationType::TANH;
	for (auto* config : {&expected.sharedHead, &expected.policy, &expected.critic}) {
		config->layerSizes = {128, 64};
		config->activationType = GGL::ModelActivationType::LEAKY_RELU;
		config->optimType = GGL::ModelOptimType::RMSPROP;
		config->addLayerNorm = false;
		config->addOutputLayer = false;
	}

	const std::filesystem::path root = RocketForgeTests::TestTempDir("benchmark-setup");
	model.trainingSetupPath = root / "training_setup.json";
	RocketForge::TrainingSetupSnapshot saved;
	saved.agentSetup = expected;
	BENCHMARK_CHECK_TRUE(GGL::JsonSaveToFile(saved, model.trainingSetupPath));
	model.agentSetupSource = ModelBenchmarkAgentSetupSource::SavedSegment;
	BENCHMARK_CHECK_TRUE(RocketForge::SameAgentSetupSnapshot(expected,
		RocketForge::CaptureAgentSetupSnapshot(*CreateBenchmarkAgent(model))));

	// A captured run must keep all of its settings even if the saved segment changes.
	model.agentSetupJson = RocketForge::AgentSetupSnapshotToJson(expected).dump();
	saved.agentSetup->carConfig = RocketSim::CAR_CONFIG_OCTANE;
	saved.agentSetup->tickSkip = 8;
	BENCHMARK_CHECK_TRUE(GGL::JsonSaveToFile(saved, model.trainingSetupPath));
	const std::unique_ptr<Agent> captured = CreateBenchmarkAgent(model);
	BENCHMARK_CHECK_TRUE(RocketForge::SameAgentSetupSnapshot(expected, RocketForge::CaptureAgentSetupSnapshot(*captured)));
}

TEST(ModelBenchmark, CurrentPredictionHonorsAgentAndCapturedRunsStayFrozen)
{
	namespace fs = std::filesystem;
	const fs::path root = RocketForgeTests::TestTempDir("benchmark-current-prediction");
	Agent definition;
	definition.agentName = "BenchmarkPredictionFixture";
	definition.trainingEngineVersion = 2;
	definition.ballPredictionEngineVersion = 3;
	definition.useBallPrediction = false;
	const RocketForge::AgentSetupSnapshot currentSetup = RocketForge::CaptureAgentSetupSnapshot(definition);
	AgentRegistry::Register(definition.agentName, [currentSetup] {
		std::unique_ptr<Agent> agent = std::make_unique<Agent>();
		RocketForge::ApplyAgentSetupSnapshot(*agent, currentSetup);
		return agent;
	});

	RocketForge::AgentSetupSnapshot trainedSetup = currentSetup;
	trainedSetup.trainingEngineVersion = 3;
	trainedSetup.ballPredictionEngineVersion = 2;
	trainedSetup.useBallPrediction = true;
	RocketForge::TrainingSetupSnapshot saved;
	saved.agentSetup = trainedSetup;
	ModelBenchmarkModel model;
	model.agentName = definition.agentName;
	model.checkpointPath = root / "checkpoints" / "100";
	fs::create_directories(model.checkpointPath);
	model.trainingSetupPath = root / RocketForge::TrainingSetupSnapshot::FILE_NAME;
	ASSERT_TRUE(GGL::JsonSaveToFile(saved, model.trainingSetupPath));
	model.agentSetupSource = ModelBenchmarkAgentSetupSource::SavedSegment;
	EXPECT_TRUE(RocketForge::SameAgentSetupSnapshot(trainedSetup, RocketForge::CaptureAgentSetupSnapshot(*CreateBenchmarkAgent(model))));

	// Explicit paths, discoverable snapshots, and packaged profiles cannot override Current.
	model.agentSetupSource = ModelBenchmarkAgentSetupSource::CurrentAgent;
	EXPECT_TRUE(RocketForge::SameAgentSetupSnapshot(currentSetup, RocketForge::CaptureAgentSetupSnapshot(*CreateBenchmarkAgent(model))));
	model.trainingSetupPath.clear();
	EXPECT_TRUE(RocketForge::SameAgentSetupSnapshot(currentSetup, RocketForge::CaptureAgentSetupSnapshot(*CreateBenchmarkAgent(model))));
	model.checkpointPath = root / "imported";
	fs::create_directories(model.checkpointPath);
	{
		std::ofstream out(model.checkpointPath / "agent_setup.json");
		out << RocketForge::AgentSetupSnapshotToJson(trainedSetup);
	}
	EXPECT_TRUE(RocketForge::SameAgentSetupSnapshot(currentSetup, RocketForge::CaptureAgentSetupSnapshot(*CreateBenchmarkAgent(model))));

	ModelBenchmarkConfig config;
	config.models = {model, model};
	config.models[1].agentSetupJson = RocketForge::AgentSetupSnapshotToJson(trainedSetup).dump();
	const ModelBenchmarkConfig prepared = PrepareModelBenchmarkPrediction(config);
	EXPECT_EQ(prepared.modelPredictionEngines, (std::vector<int>{3, 2}));
	EXPECT_TRUE(RocketForge::SameAgentSetupSnapshot(currentSetup,
		RocketForge::CaptureAgentSetupSnapshot(*CreateBenchmarkAgent(prepared.models[0]))));
	EXPECT_TRUE(RocketForge::SameAgentSetupSnapshot(trainedSetup,
		RocketForge::CaptureAgentSetupSnapshot(*CreateBenchmarkAgent(prepared.models[1]))));
	EXPECT_EQ(PrepareModelBenchmarkPrediction(prepared).modelPredictionEngines, prepared.modelPredictionEngines);
}

TEST(ModelBenchmark, ExplicitBudgetsSetEachPairsGameTarget)
{
	ModelBenchmarkConfig config;
	config.models.resize(3);
	config.usePairBudgets = true;
	config.pairBudgets = { { 0, 1, 1, 350 }, { 0, 2, 2, 1000 } };
	BENCHMARK_CHECK_EQ(ModelBenchmarkPairGameTarget(config, 1, 0, 1), 350);
	BENCHMARK_CHECK_EQ(ModelBenchmarkPairGameTarget(config, 0, 1, 2), 0);
	BENCHMARK_CHECK_EQ(ModelBenchmarkPairGameTarget(config, 0, 2, 2), 1000);
	BENCHMARK_CHECK_EQ(ModelBenchmarkPairGameTarget(config, 1, 2, 1), 0);
}

namespace {
	constexpr float SnapshotTolerance = 1e-3f;

	ModelBenchmarkRuntimeSignature BaseRuntimeSignature()
	{
		ModelBenchmarkRuntimeSignature signature = {};
		signature.agentName = "example-agent";
		signature.tickSkip = 8;
		signature.actionDelay = 7;
		signature.obsSize = 128;
		signature.numActions = 90;
		signature.maxTeamSize = 3;
		signature.obsBuilderType = "AdvancedObs";
		signature.actionParserType = "DefaultAction";
		signature.carConfigSignature = "octane";
		signature.policyConfigSignature = "shared/policy";
		return signature;
	}

	ModelBenchmarkModel BenchmarkModel(const std::string& name, bool primary)
	{
		ModelBenchmarkModel model = {};
		model.displayName = name;
		model.primary = primary;
		return model;
	}

	void AssertNear(float actual, float expected, float tolerance = SnapshotTolerance)
	{
		if (std::abs(actual - expected) <= tolerance)
			return;
		std::ostringstream message;
		message << "Expected " << actual << " to be within " << tolerance << " of " << expected << ".";
		FAIL() << message.str();
	}

	void AssertVecNear(const RocketSim::Vec& actual, const RocketSim::Vec& expected)
	{
		for (int axis = 0; axis < 3; axis++)
			AssertNear(actual[axis], expected[axis]);
	}

	void AssertCarStateEquivalent(const RocketSim::CarState& actual, const RocketSim::CarState& expected)
	{
		AssertVecNear(actual.pos, expected.pos);
		AssertVecNear(actual.vel, expected.vel);
		AssertVecNear(actual.angVel, expected.angVel);
		for (int row = 0; row < 3; row++)
			for (int column = 0; column < 3; column++)
				AssertNear(actual.rotMat[row][column], expected.rotMat[row][column]);
		AssertNear(actual.boost, expected.boost);
		BENCHMARK_CHECK_EQ(actual.isOnGround, expected.isOnGround);
		BENCHMARK_CHECK_EQ(actual.hasJumped, expected.hasJumped);
		BENCHMARK_CHECK_EQ(actual.hasDoubleJumped, expected.hasDoubleJumped);
		BENCHMARK_CHECK_EQ(actual.hasFlipped, expected.hasFlipped);
		BENCHMARK_CHECK_EQ(actual.isDemoed, expected.isDemoed);
	}

	void AssertSnapshotsEquivalent(const GGL::PolicyMatchSnapshot& actual, const GGL::PolicyMatchSnapshot& expected)
	{
		AssertVecNear(actual.ball.pos, expected.ball.pos);
		AssertVecNear(actual.ball.vel, expected.ball.vel);
		AssertVecNear(actual.ball.angVel, expected.ball.angVel);
		BENCHMARK_CHECK_EQ(actual.cars.size(), expected.cars.size());
		for (size_t i = 0; i < actual.cars.size(); i++) {
			BENCHMARK_CHECK_EQ(actual.cars[i].team, expected.cars[i].team);
			AssertCarStateEquivalent(actual.cars[i].state, expected.cars[i].state);
		}
	}

	bool SnapshotPositionsDiffer(const GGL::PolicyMatchSnapshot& a, const GGL::PolicyMatchSnapshot& b)
	{
		for (int axis = 0; axis < 3; axis++)
			if (std::abs(a.ball.pos[axis] - b.ball.pos[axis]) > SnapshotTolerance)
				return true;
		if (a.cars.size() != b.cars.size())
			return true;
		for (size_t carIdx = 0; carIdx < a.cars.size(); carIdx++)
			for (int axis = 0; axis < 3; axis++)
				if (std::abs(a.cars[carIdx].state.pos[axis] - b.cars[carIdx].state.pos[axis]) > SnapshotTolerance)
					return true;
		return false;
	}

	RocketSim::CarState DistinctCarState(float x, float y, float z, float boost)
	{
		RocketSim::CarState state = {};
		state.pos = { x, y, z };
		state.vel = { x * 0.25f, y * -0.125f, z * 0.5f };
		state.angVel = { 0.1f + x * 0.0001f, -0.2f + y * 0.0001f, 0.3f };
		state.boost = boost;
		state.isOnGround = z <= RocketSim::RLConst::CAR_SPAWN_REST_Z + 1.0f;
		return state;
	}

	struct BenchmarkTinyObsBuilder : RLGC::ObsBuilder {
		RLGC::FList BuildObs(const RLGC::Player& player, const RLGC::GameState& state) override
		{
			return { 0.0f };
		}
	};

	struct BenchmarkTinyActionParser : RLGC::ActionParser {
		RLGC::Action ParseAction(int actionIdx, const RLGC::Player& player, const RLGC::GameState& state) override
		{
			return {};
		}

		int GetActionAmount() override
		{
			return 1;
		}
	};

	struct KickoffActionCounts {
		bool touched = false;
		std::array<int, 2> sampledAlternatives = {};
		std::array<int, 2> afterTouchDecisions = {};
	};

	struct KickoffSamplingActionParser : RLGC::ActionParser {
		std::map<RocketSim::Arena*, KickoffActionCounts>& counts;

		explicit KickoffSamplingActionParser(std::map<RocketSim::Arena*, KickoffActionCounts>& counts)
			: counts(counts) {}

		RLGC::Action ParseAction(int actionIdx, const RLGC::Player& player, const RLGC::GameState& state) override {
			auto& count = counts[state.lastArena];
			const int side = player.team == RocketSim::Team::BLUE ? 0 : 1;
			BENCHMARK_CHECK_TRUE(actionIdx == 0 || actionIdx == 1);
			if (count.touched) {
				BENCHMARK_CHECK_EQ(actionIdx, 1);
				count.afterTouchDecisions[side]++;
			} else if (actionIdx == 0) {
				count.sampledAlternatives[side]++;
			}
			return {};
		}

		int GetActionAmount() override { return 3; }
		bool UsesActionMask() override { return true; }
		void GetActionMaskInto(const RLGC::Player&, const RLGC::GameState&, std::vector<uint8_t>& out) override {
			out = { true, true, false };
		}
	};

	struct BenchmarkModelSetOwner {
		GGL::ModelSet models;

		~BenchmarkModelSetOwner()
		{
			models.Free();
		}
	};
}

TEST(ModelBenchmark, EloFromWinProbability_IsSymmetric)
{
	double favored = ModelBenchmarkMath::EloFromWinProbability(0.75);
	double underdog = ModelBenchmarkMath::EloFromWinProbability(0.25);
	BENCHMARK_CHECK_TRUE(favored > 0.0);
	BENCHMARK_CHECK_TRUE(underdog < 0.0);
	BENCHMARK_CHECK_TRUE(std::abs(favored + underdog) < 0.001);
}

TEST(ModelBenchmark, ConfidenceToZ_UsesTwoSidedRange)
{
	double z95 = ModelBenchmarkMath::ConfidenceToZ(0.95);
	BENCHMARK_CHECK_TRUE(z95 > 1.95);
	BENCHMARK_CHECK_TRUE(z95 < 1.97);
}

TEST(ModelBenchmark, NoRatedOutcomesRemainUncertain)
{
	EloFit fit = FitElo(2, {});

	BENCHMARK_CHECK_EQ(fit.elo.size(), size_t(2));
	BENCHMARK_CHECK_EQ(fit.covariance.size(), size_t(2));
	BENCHMARK_CHECK_TRUE(fit.covariance[0][0] > 1e6);
	BENCHMARK_CHECK_TRUE(fit.covariance[1][1] > 1e6);
	BENCHMARK_CHECK_TRUE(DifferenceVariance(fit.covariance, 0, 1) > 1e6);
	BENCHMARK_CHECK_FALSE(AllRangesSmallEnough(fit, 0.95, 50.0));
	AssertNear((float)ConfidenceBetter(fit, 0, 1), 0.5f);
}

TEST(ModelBenchmark, EarlyStopMinimumUsesCommittedRatedOutcomes)
{
	BenchmarkRunState state = {};
	state.scoredGames = 100;
	BENCHMARK_CHECK_FALSE(HasMinimumRatedOutcomesForEarlyStop((int)state.outcomes.size(), 1));

	state.outcomes.push_back({ 0, 1, 1, ModelBenchmarkGoalTiming::Kickoff, 1.0 });
	BENCHMARK_CHECK_FALSE(HasMinimumRatedOutcomesForEarlyStop((int)state.outcomes.size(), 1));
	state.outcomes.push_back({ 1, 0, 1, ModelBenchmarkGoalTiming::Kickoff, 1.0 });
	BENCHMARK_CHECK_TRUE(HasMinimumRatedOutcomesForEarlyStop((int)state.outcomes.size(), 1));
}

TEST(ModelBenchmark, StatsNormalizeEventsToFiveMinuteGames)
{
	RunningStats stats = {};
	stats.gameTimeSum = 600.0;
	stats.goals = 3.0;
	stats.shots = 8.0;
	stats.assists = 2.0;
	stats.saves = 6.0;
	stats.goalsConceded = 2.0;
	stats.passes = 4.0;
	stats.passGameTimeSum = 300.0;
	stats.ballTouches = 20.0;
	stats.opponentBallTouches = 30.0;
	stats.flipResets = 1.0;
	stats.bumps = 6.0;
	stats.boostSteals = 4.0;
	stats.bigBoostPads = 10.0;
	stats.boostConsumed = 50.0;
	stats.demos = 2.0;
	stats.aerialTimeSum = 25.0;
	stats.lowAerialTimeSum = 10.0;
	stats.highAerialTimeSum = 15.0;
	stats.lowBoostTimeSum = 20.0;
	stats.playerTimeSum = 100.0;
	stats.offensiveThirdBallSteps = 4.0;
	stats.neutralThirdBallSteps = 3.0;
	stats.defensiveThirdBallSteps = 3.0;
	stats.stepSamples = 10;

	const ModelBenchmarkOptionalStats finalized = FinalizeStats(stats);
	AssertNear((float)finalized.equivalentGames, 2.0f);
	AssertNear((float)finalized.goals, 1.5f);
	AssertNear((float)finalized.shots, 4.0f);
	AssertNear((float)finalized.shotConversionPercent, 37.5f);
	AssertNear((float)finalized.assistedGoalPercent, 200.0f / 3.0f);
	AssertNear((float)finalized.saves, 3.0f);
	AssertNear((float)finalized.savePercent, 75.0f);
	AssertNear((float)finalized.passes, 4.0f);
	BENCHMARK_CHECK_TRUE(finalized.passesAvailable);
	AssertNear((float)finalized.ballTouches, 10.0f);
	AssertNear((float)finalized.touchSharePercent, 40.0f);
	AssertNear((float)finalized.flipResets, 0.5f);
	AssertNear((float)finalized.bumps, 3.0f);
	AssertNear((float)finalized.boostSteals, 2.0f);
	AssertNear((float)finalized.bigBoostPads, 5.0f);
	AssertNear((float)finalized.boostConsumptionPerMinute, 30.0f);
	AssertNear((float)finalized.demos, 1.0f);
	AssertNear((float)finalized.aerialTimePercent, 25.0f);
	AssertNear((float)finalized.lowAirTimePercent, 10.0f);
	AssertNear((float)finalized.highAirTimePercent, 15.0f);
	AssertNear((float)finalized.lowBoostTimePercent, 20.0f);
	AssertNear((float)finalized.offensiveThirdPercent, 40.0f);
	AssertNear((float)finalized.neutralThirdPercent, 30.0f);
	AssertNear((float)finalized.defensiveThirdPercent, 30.0f);
}

TEST(ModelBenchmark, StatsMergePreservesModeAndOverallRates)
{
	RunningStats mode1v1 = {};
	mode1v1.gameTimeSum = 300.0;
	mode1v1.goals = 3.0;
	mode1v1.ballTouches = 12.0;
	mode1v1.opponentBallTouches = 8.0;
	mode1v1.boostConsumed = 30.0;
	mode1v1.boostSum = 40.0;
	mode1v1.playerSamples = 1;
	mode1v1.aerialTimeSum = 60.0;
	mode1v1.lowAerialTimeSum = 45.0;
	mode1v1.highAerialTimeSum = 15.0;
	mode1v1.lowBoostTimeSum = 30.0;
	mode1v1.playerTimeSum = 300.0;

	RunningStats mode2v2 = {};
	mode2v2.gameTimeSum = 600.0;
	mode2v2.goals = 2.0;
	mode2v2.ballTouches = 18.0;
	mode2v2.opponentBallTouches = 12.0;
	mode2v2.boostConsumed = 120.0;
	mode2v2.boostSum = 180.0;
	mode2v2.playerSamples = 3;
	mode2v2.aerialTimeSum = 240.0;
	mode2v2.lowAerialTimeSum = 180.0;
	mode2v2.highAerialTimeSum = 60.0;
	mode2v2.lowBoostTimeSum = 120.0;
	mode2v2.playerTimeSum = 600.0;
	mode2v2.passes = 12.0;
	mode2v2.passGameTimeSum = 600.0;

	RunningStats overall = {};
	MergeRunningStats(overall, mode1v1);
	MergeRunningStats(overall, mode2v2);

	const ModelBenchmarkOptionalStats finalized1v1 = FinalizeStats(mode1v1);
	const ModelBenchmarkOptionalStats finalized2v2 = FinalizeStats(mode2v2);
	const ModelBenchmarkOptionalStats finalizedOverall = FinalizeStats(overall);
	AssertNear((float)finalized1v1.goals, 3.0f);
	AssertNear((float)finalized2v2.goals, 1.0f);
	AssertNear((float)finalizedOverall.equivalentGames, 3.0f);
	AssertNear((float)finalizedOverall.goals, 5.0f / 3.0f);
	AssertNear((float)finalizedOverall.ballTouches, 10.0f);
	AssertNear((float)finalizedOverall.touchSharePercent, 60.0f);
	AssertNear((float)finalizedOverall.boostConsumptionPerMinute, 10.0f);
	AssertNear((float)finalizedOverall.averageBoost, 55.0f);
	AssertNear((float)finalizedOverall.aerialTimePercent, 100.0f / 3.0f);
	AssertNear((float)finalizedOverall.lowAirTimePercent, 25.0f);
	AssertNear((float)finalizedOverall.highAirTimePercent, 25.0f / 3.0f);
	AssertNear((float)finalizedOverall.lowBoostTimePercent, 100.0f / 6.0f);
	BENCHMARK_CHECK_FALSE(finalized1v1.passesAvailable);
	BENCHMARK_CHECK_TRUE(finalized2v2.passesAvailable);
	BENCHMARK_CHECK_TRUE(finalizedOverall.passesAvailable);
	AssertNear((float)finalizedOverall.passes, 6.0f);
}

TEST(ModelBenchmark, StatsHideExpandedMetricsWhenLegacyTotalsAreMerged)
{
	RunningStats current = {};
	current.gameTimeSum = 300.0;
	current.passGameTimeSum = 300.0;
	current.passes = 4.0;

	RunningStats legacy = {};
	legacy.definitionVersion = 1;
	legacy.gameTimeSum = 300.0;
	MergeRunningStats(current, legacy);

	const ModelBenchmarkOptionalStats finalized = FinalizeStats(current);
	BENCHMARK_CHECK_FALSE(finalized.expandedStatsAvailable);
	BENCHMARK_CHECK_FALSE(finalized.efficiencyStatsAvailable);
	BENCHMARK_CHECK_FALSE(finalized.passesAvailable);
}

TEST(ModelBenchmark, StatsKeepVersionTwoMetricsWhenEfficiencyTotalsAreUnavailable)
{
	RunningStats versionTwo = {};
	versionTwo.definitionVersion = MODEL_BENCHMARK_EXPANDED_STAT_DEFINITION_VERSION;
	versionTwo.gameTimeSum = 300.0;
	versionTwo.passGameTimeSum = 300.0;
	versionTwo.passes = 4.0;
	versionTwo.bumps = 2.0;

	const ModelBenchmarkOptionalStats finalized = FinalizeStats(versionTwo);
	BENCHMARK_CHECK_TRUE(finalized.expandedStatsAvailable);
	BENCHMARK_CHECK_TRUE(finalized.passesAvailable);
	BENCHMARK_CHECK_FALSE(finalized.efficiencyStatsAvailable);
}

TEST(ModelBenchmark, CompletedPassesRequireDistanceTimeAndUncontestedPossession)
{
	const auto touch = [](uint32_t carId, RocketSim::Team team, uint64_t tick, float x) {
		return BenchmarkPassTouch{ carId, team, tick, { x, 0.0f, 100.0f } };
	};

	BenchmarkPassTracker completed = {};
	BenchmarkPassCounts counts = AccumulateCompletedPasses(completed, {
		touch(1, RocketSim::Team::BLUE, 10, 0.0f),
		touch(2, RocketSim::Team::BLUE, 250, 500.0f)
	});
	BENCHMARK_CHECK_EQ(counts.blue, 1.0);
	BENCHMARK_CHECK_EQ(counts.orange, 0.0);
	counts = AccumulateCompletedPasses(completed, {
		touch(1, RocketSim::Team::BLUE, 10, 0.0f),
		touch(2, RocketSim::Team::BLUE, 250, 500.0f)
	});
	BENCHMARK_CHECK_EQ(counts.blue, 0.0);

	BenchmarkPassTracker tooShort = {};
	counts = AccumulateCompletedPasses(tooShort, {
		touch(1, RocketSim::Team::BLUE, 10, 0.0f),
		touch(2, RocketSim::Team::BLUE, 20, 499.0f)
	});
	BENCHMARK_CHECK_EQ(counts.blue, 0.0);

	BenchmarkPassTracker tooLate = {};
	counts = AccumulateCompletedPasses(tooLate, {
		touch(1, RocketSim::Team::BLUE, 10, 0.0f),
		touch(2, RocketSim::Team::BLUE, 251, 500.0f)
	});
	BENCHMARK_CHECK_EQ(counts.blue, 0.0);

	BenchmarkPassTracker interrupted = {};
	counts = AccumulateCompletedPasses(interrupted, {
		touch(1, RocketSim::Team::BLUE, 10, 0.0f),
		touch(3, RocketSim::Team::ORANGE, 20, 250.0f),
		touch(2, RocketSim::Team::BLUE, 30, 750.0f)
	});
	BENCHMARK_CHECK_EQ(counts.blue, 0.0);
	BENCHMARK_CHECK_EQ(counts.orange, 0.0);

	BenchmarkPassTracker contested = {};
	counts = AccumulateCompletedPasses(contested, {
		touch(1, RocketSim::Team::BLUE, 10, 0.0f),
		touch(2, RocketSim::Team::BLUE, 20, 500.0f),
		touch(3, RocketSim::Team::ORANGE, 20, 500.0f),
		touch(4, RocketSim::Team::BLUE, 30, 1000.0f)
	});
	BENCHMARK_CHECK_EQ(counts.blue, 0.0);
}

TEST(ModelBenchmark, StatsSplitAirTimeFieldThirdsAndTrackBumps)
{
	RLGC::GameState previous = {};
	previous.players.resize(4);
	for (int playerIdx = 0; playerIdx < 4; playerIdx++) {
		previous.players[playerIdx].carId = (uint32_t)(playerIdx + 1);
		previous.players[playerIdx].team = playerIdx < 2
			? RocketSim::Team::BLUE
			: RocketSim::Team::ORANGE;
	}

	RLGC::GameState current = previous;
	current.players[0].isOnGround = false;
	current.players[0].pos.z = 300.0f;
	current.players[0].eventState.bump = true;
	current.players[1].isOnGround = false;
	current.players[1].pos.z = RLGC::CommonValues::GOAL_HEIGHT;
	current.players[0].eventState.goal = true;
	current.players[1].eventState.assist = true;

	RunningStats stats = {};
	for (float ballY : { 2000.0f, 0.0f, -2000.0f }) {
		current.ball.pos.y = ballY;
		AccumulateStepStats(stats, current, previous, RocketSim::Team::BLUE, 1.0f);
		previous = current;
		for (RLGC::Player& player : current.players)
			player.eventState = {};
	}

	const ModelBenchmarkOptionalStats finalized = FinalizeStats(stats);
	AssertNear((float)finalized.lowAirTimePercent, 50.0f);
	AssertNear((float)finalized.highAirTimePercent, 50.0f);
	AssertNear((float)finalized.offensiveThirdPercent, 100.0f / 3.0f);
	AssertNear((float)finalized.neutralThirdPercent, 100.0f / 3.0f);
	AssertNear((float)finalized.defensiveThirdPercent, 100.0f / 3.0f);
	AssertNear((float)stats.bumps, 1.0f);
	AssertNear((float)finalized.assistedGoalPercent, 100.0f);
}

TEST(ModelBenchmark, StatsCountOpponentHalfBigBoostAsSteal)
{
	std::unique_ptr<RocketSim::Arena> arena = GGL::MakePolicyMatchArena(
		1,
		RocketSim::CAR_CONFIG_OCTANE,
		RocketSim::CAR_CONFIG_OCTANE,
		RocketSim::EngineId::V2);
	RLGC::GameState previous(arena.get());
	RLGC::GameState current = previous;

	int stolenPadIndex = -1;
	const RocketSim::BoostPad* stolenPad = nullptr;
	for (int padIdx = 0; padIdx < (int)current.boostPadIndexMap.size(); padIdx++) {
		const int arenaPadIdx = current.boostPadIndexMap[padIdx];
		const RocketSim::BoostPad* pad = arena->_boostPads[arenaPadIdx];
		if (pad->config.isBig && pad->config.pos.y > 0.0f) {
			stolenPadIndex = padIdx;
			stolenPad = pad;
			break;
		}
	}
	BENCHMARK_CHECK_TRUE(stolenPadIndex >= 0 && stolenPad != nullptr);
	current.boostPads[stolenPadIndex] = false;
	for (RLGC::Player& player : current.players)
		player.pos = { 10000.0f, 10000.0f, 0.0f };
	for (RLGC::Player& player : current.players) {
		if (player.team == RocketSim::Team::BLUE) {
			player.pos = stolenPad->config.pos;
			break;
		}
	}

	RunningStats stats = {};
	AccumulateStepStats(stats, current, previous, RocketSim::Team::BLUE, 1.0f);
	BENCHMARK_CHECK_EQ(stats.bigBoostPads, 1.0);
	BENCHMARK_CHECK_EQ(stats.boostSteals, 1.0);
}

TEST(ModelBenchmark, StatsCountNewBallFlipResetOnce)
{
	RLGC::GameState previous = {};
	previous.players.resize(1);
	previous.players[0].carId = 17;
	previous.players[0].team = RocketSim::Team::BLUE;
	previous.players[0].isOnGround = false;
	previous.players[0].hasJumped = true;

	RLGC::GameState current = {};
	current.players.resize(1);
	current.players[0].carId = 17;
	current.players[0].team = RocketSim::Team::BLUE;
	current.players[0].isOnGround = false;
	current.players[0].hasJumped = false;
	current.players[0].ballTouchedStep = true;
	current.players[0].pos = { 0.0f, 0.0f, 200.0f };
	current.ball.pos = { 130.0f, 0.0f, 200.0f };

	RunningStats stats = {};
	AccumulateStepStats(stats, current, previous, RocketSim::Team::BLUE, 1.0f);
	BENCHMARK_CHECK_EQ(stats.flipResets, 1.0);

	previous = current;
	current.players[0].ballTouchedStep = true;
	AccumulateStepStats(stats, current, previous, RocketSim::Team::BLUE, 1.0f);
	BENCHMARK_CHECK_EQ(stats.flipResets, 1.0);

	previous.players[0].hasJumped = true;
	current.players[0].ballTouchedStep = false;
	RunningStats noBallTouchStats = {};
	AccumulateStepStats(
		noBallTouchStats,
		current,
		previous,
		RocketSim::Team::BLUE,
		1.0f);
	BENCHMARK_CHECK_EQ(noBallTouchStats.flipResets, 0.0);
}

TEST(ModelBenchmark, StatsRequireFlipResetHeightAndBallProximity)
{
	RLGC::GameState previous = {};
	previous.players.resize(1);
	previous.players[0].carId = 29;
	previous.players[0].team = RocketSim::Team::BLUE;
	previous.players[0].isOnGround = false;
	previous.players[0].hasJumped = true;

	RLGC::GameState current = previous;
	current.players[0].hasJumped = false;
	current.players[0].ballTouchedStep = true;
	current.players[0].pos = { 0.0f, 0.0f, 199.0f };
	current.ball.pos = { 0.0f, 0.0f, 199.0f };

	RunningStats stats = {};
	AccumulateStepStats(stats, current, previous, RocketSim::Team::BLUE, 1.0f);
	BENCHMARK_CHECK_EQ(stats.flipResets, 0.0);

	current.players[0].pos.z = 200.0f;
	current.ball.pos = { 131.0f, 0.0f, 200.0f };
	AccumulateStepStats(stats, current, previous, RocketSim::Team::BLUE, 1.0f);
	BENCHMARK_CHECK_EQ(stats.flipResets, 0.0);

	current.ball.pos.x = 130.0f;
	AccumulateStepStats(stats, current, previous, RocketSim::Team::BLUE, 1.0f);
	BENCHMARK_CHECK_EQ(stats.flipResets, 1.0);
}

TEST(ModelBenchmark, StatsRequireCurrentStepTouchForFlipReset)
{
	RLGC::GameState previous = {};
	previous.players.resize(1);
	previous.players[0].carId = 31;
	previous.players[0].team = RocketSim::Team::BLUE;
	previous.players[0].isOnGround = true;
	previous.players[0].hasJumped = false;
	previous.players[0].ballTouchedStep = true;

	RLGC::GameState current = previous;
	current.players[0].isOnGround = false;
	current.players[0].ballTouchedStep = false;
	current.players[0].pos = { 0.0f, 0.0f, 200.0f };
	current.ball.pos = current.players[0].pos;

	RunningStats stats = {};
	AccumulateStepStats(stats, current, previous, RocketSim::Team::BLUE, 1.0f);
	BENCHMARK_CHECK_EQ(stats.flipResets, 0.0);
}

TEST(ModelBenchmark, StatsCountBallTouchStarts)
{
	RLGC::GameState previous = {};
	previous.players.resize(1);
	previous.players[0].carId = 23;
	previous.players[0].team = RocketSim::Team::BLUE;

	RLGC::GameState current = previous;
	current.players[0].ballTouchedStep = true;
	current.players[0].ballTouchedTick = true;

	RunningStats stats = {};
	AccumulateStepStats(stats, current, previous, RocketSim::Team::BLUE, 1.0f);
	BENCHMARK_CHECK_EQ(stats.ballTouches, 1.0);

	previous = current;
	AccumulateStepStats(stats, current, previous, RocketSim::Team::BLUE, 1.0f);
	BENCHMARK_CHECK_EQ(stats.ballTouches, 1.0);

	previous.players[0].ballTouchedTick = false;
	AccumulateStepStats(stats, current, previous, RocketSim::Team::BLUE, 1.0f);
	BENCHMARK_CHECK_EQ(stats.ballTouches, 2.0);
}

TEST(ModelBenchmark, StatsTrackEfficiencyMetricDenominatorsAndBoostDepletion)
{
	RLGC::GameState previous = {};
	previous.players.resize(2);
	previous.players[0].carId = 41;
	previous.players[0].team = RocketSim::Team::BLUE;
	previous.players[0].boost = 90.0f;
	previous.players[0].tickCountSinceUpdate = 10;
	previous.players[1].carId = 42;
	previous.players[1].team = RocketSim::Team::ORANGE;

	RLGC::GameState current = previous;
	current.players[0].boost = 70.0f;
	current.players[0].tickCountSinceUpdate = 18;
	current.players[0].ballTouchedStep = true;
	current.players[1].ballTouchedStep = true;
	current.players[1].eventState.goal = true;

	RunningStats stats = {};
	AccumulateStepStats(stats, current, previous, RocketSim::Team::BLUE, 1.0f);
	BENCHMARK_CHECK_EQ(stats.goalsConceded, 1.0);
	BENCHMARK_CHECK_EQ(stats.ballTouches, 1.0);
	BENCHMARK_CHECK_EQ(stats.opponentBallTouches, 1.0);
	BENCHMARK_CHECK_EQ(stats.boostConsumed, 20.0);

	previous = current;
	current.players[0].boost = 33.0f;
	current.players[0].tickCountSinceUpdate = 2;
	current.players[0].ballTouchedStep = false;
	current.players[1].ballTouchedStep = false;
	current.players[1].eventState = {};
	AccumulateStepStats(stats, current, previous, RocketSim::Team::BLUE, 1.0f);
	BENCHMARK_CHECK_EQ(stats.boostConsumed, 20.0);
}

TEST(ModelBenchmark, AggregatedOutcomeWeightsMatchRepeatedResults)
{
	std::vector<BenchmarkOutcome> repeated;
	for (int i = 0; i < 7; i++)
		repeated.push_back({ 0, 1, 1, ModelBenchmarkGoalTiming::Kickoff, 1.0 });
	for (int i = 0; i < 3; i++)
		repeated.push_back({ 1, 0, 1, ModelBenchmarkGoalTiming::Kickoff, 1.0 });
	const std::vector<BenchmarkOutcome> aggregated = {
		{ 0, 1, 1, ModelBenchmarkGoalTiming::Kickoff, 7.0 },
		{ 1, 0, 1, ModelBenchmarkGoalTiming::Kickoff, 3.0 }
	};

	const EloFit repeatedFit = FitElo(2, repeated);
	const EloFit aggregatedFit = FitElo(2, aggregated);
	for (int model = 0; model < 2; model++) {
		AssertNear((float)aggregatedFit.elo[model], (float)repeatedFit.elo[model]);
		for (int other = 0; other < 2; other++) {
			AssertNear(
				(float)aggregatedFit.covariance[model][other],
				(float)repeatedFit.covariance[model][other]);
		}
	}
}

TEST(ModelBenchmark, LiveEstimateSharesFitWithoutChangingRankOrSummary)
{
	const std::vector<std::string> names = { "Alpha", "Bravo", "Charlie" };
	const std::vector<BenchmarkOutcome> outcomes = {
		{ 0, 1, 1, ModelBenchmarkGoalTiming::Kickoff, 1.0 },
		{ 0, 1, 2, ModelBenchmarkGoalTiming::PostKickoff, 0.5 },
		{ 1, 2, 1, ModelBenchmarkGoalTiming::PostKickoff, 1.0 },
		{ 2, 0, 3, ModelBenchmarkGoalTiming::PostKickoff, 1.0 }
	};
	const LiveEloEstimate estimate = BuildLiveEloEstimate(3, outcomes, names);
	BENCHMARK_CHECK_TRUE(estimate.hasEvidence);
	BENCHMARK_CHECK_TRUE(estimate.connected);
	BENCHMARK_CHECK_EQ(estimate.order, BuildRankOrder(3, outcomes, names));
	BENCHMARK_CHECK_EQ(FormatLiveEstimateSummary(estimate, names, 0.95), FormatLiveEstimateSummary(3, outcomes, names, 0.95));

	const LiveEloEstimate disconnected = BuildLiveEloEstimate(3, { { 0, 1, 1, ModelBenchmarkGoalTiming::PostKickoff, 1.0 } }, names);
	BENCHMARK_CHECK_TRUE(disconnected.hasEvidence);
	BENCHMARK_CHECK_FALSE(disconnected.connected);
	BENCHMARK_CHECK_EQ(disconnected.order, BuildRankOrder(3, { { 0, 1, 1, ModelBenchmarkGoalTiming::PostKickoff, 1.0 } }, names));
	BENCHMARK_CHECK_TRUE(FormatLiveEstimateSummary(disconnected, names, 0.95).find("connect every model") != std::string::npos);
}

TEST(ModelBenchmark, BetterThanSummaryShowsActualPairwiseConfidence)
{
	EloFit fit = {};
	fit.elo = { 10.0, 0.0 };
	fit.covariance = {
		{ 100.0, 0.0 },
		{ 0.0, 100.0 }
	};

	BENCHMARK_CHECK_EQ(FormatConfidenceList(fit, { "Model A", "Model B" }, 0), std::string("Model B (76.0%)"));
}

TEST(ModelBenchmark, BetterThanSummaryListsHighestEloOpponentFirst)
{
	EloFit fit = {};
	fit.elo = { 0.0, 30.0, 20.0, 10.0 };
	fit.covariance = {
		{ 100.0, 0.0, 0.0, 0.0 },
		{ 0.0, 100.0, 0.0, 0.0 },
		{ 0.0, 0.0, 100.0, 0.0 },
		{ 0.0, 0.0, 0.0, 100.0 }
	};

	BENCHMARK_CHECK_EQ(
		FormatConfidenceList(fit, { "Lowest", "Highest", "Middle", "Next" }, 1),
		std::string("Middle (76.0%), Next (92.1%), Lowest (98.3%)"));
}

TEST(ModelBenchmark, FullRankingKnown_RequiresAdjacentConfidence)
{
	std::vector<double> elo = { 100.0, 0.0, -100.0 };
	std::vector<std::vector<double>> tight = {
		{ 25.0, 0.0, 0.0 },
		{ 0.0, 25.0, 0.0 },
		{ 0.0, 0.0, 25.0 }
	};
	std::vector<std::vector<double>> loose = {
		{ 10000.0, 0.0, 0.0 },
		{ 0.0, 10000.0, 0.0 },
		{ 0.0, 0.0, 10000.0 }
	};
	BENCHMARK_CHECK_TRUE(ModelBenchmarkMath::IsFullRankingKnown(elo, tight, 0.95));
	BENCHMARK_CHECK_FALSE(ModelBenchmarkMath::IsFullRankingKnown(elo, loose, 0.95));
}

TEST(ModelBenchmark, SelectStrategyUsesSlotPathForSameRuntime)
{
	ModelBenchmarkRuntimeSignature a = BaseRuntimeSignature();
	ModelBenchmarkRuntimeSignature b = BaseRuntimeSignature();
	BENCHMARK_CHECK_EQ(SelectModelBenchmarkStrategy({ a, b }), ModelBenchmarkStrategy::SameRuntimeSlot);
	BENCHMARK_CHECK_EQ(ModelBenchmarkStrategyName(ModelBenchmarkStrategy::SameRuntimeSlot), std::string("SameRuntimeSlot"));
}

TEST(ModelBenchmark, SelectStrategyUsesHeterogeneousPathForDifferentAgents)
{
	ModelBenchmarkRuntimeSignature a = BaseRuntimeSignature();
	ModelBenchmarkRuntimeSignature b = BaseRuntimeSignature();
	b.agentName = "other-agent";
	BENCHMARK_CHECK_EQ(SelectModelBenchmarkStrategy({ a, b }), ModelBenchmarkStrategy::Heterogeneous);
	BENCHMARK_CHECK_EQ(ModelBenchmarkStrategyName(ModelBenchmarkStrategy::Heterogeneous), std::string("Heterogeneous"));
}

TEST(ModelBenchmark, SelectStrategyUsesHeterogeneousPathForDifferentObsActionShape)
{
	ModelBenchmarkRuntimeSignature a = BaseRuntimeSignature();
	ModelBenchmarkRuntimeSignature b = BaseRuntimeSignature();
	b.obsSize += 16;
	BENCHMARK_CHECK_EQ(SelectModelBenchmarkStrategy({ a, b }), ModelBenchmarkStrategy::Heterogeneous);

	b = BaseRuntimeSignature();
	b.numActions += 4;
	BENCHMARK_CHECK_EQ(SelectModelBenchmarkStrategy({ a, b }), ModelBenchmarkStrategy::Heterogeneous);
}

TEST(ModelBenchmark, SelectStrategyUsesHeterogeneousPathForDifferentHitboxes)
{
	ModelBenchmarkRuntimeSignature a = BaseRuntimeSignature();
	ModelBenchmarkRuntimeSignature b = BaseRuntimeSignature();
	b.carConfigSignature = "dominus";
	BENCHMARK_CHECK_EQ(SelectModelBenchmarkStrategy({ a, b }), ModelBenchmarkStrategy::Heterogeneous);
}

TEST(ModelBenchmark, SelectStrategyUsesHeterogeneousPathForDifferentTickSkipOrActionDelay)
{
	ModelBenchmarkRuntimeSignature a = BaseRuntimeSignature();
	ModelBenchmarkRuntimeSignature b = BaseRuntimeSignature();
	b.tickSkip++;
	BENCHMARK_CHECK_EQ(SelectModelBenchmarkStrategy({ a, b }), ModelBenchmarkStrategy::Heterogeneous);

	b = BaseRuntimeSignature();
	b.actionDelay++;
	BENCHMARK_CHECK_EQ(SelectModelBenchmarkStrategy({ a, b }), ModelBenchmarkStrategy::Heterogeneous);
}

TEST(ModelBenchmark, SelectStrategy_RejectsEmptySignatures)
{
	bool threw = false;
	try {
		(void)SelectModelBenchmarkStrategy({});
	} catch (const std::invalid_argument&) {
		threw = true;
	}
	BENCHMARK_CHECK_TRUE(threw);
}

TEST(ModelBenchmark, BuildMatchups_SkipsNonPrimaryPairs)
{
	std::vector<ModelBenchmarkModel> models = {
		BenchmarkModel("A", true),
		BenchmarkModel("B", false),
		BenchmarkModel("C", false),
		BenchmarkModel("D", true)
	};

	std::vector<std::pair<int, int>> matchups = BuildModelBenchmarkMatchups(models);
	std::vector<std::pair<int, int>> expected = {
		{ 0, 1 },
		{ 0, 2 },
		{ 0, 3 },
		{ 1, 3 },
		{ 2, 3 }
	};
	BENCHMARK_CHECK_EQ(matchups.size(), expected.size());
	for (size_t i = 0; i < expected.size(); i++)
		BENCHMARK_CHECK_TRUE(matchups[i] == expected[i]);
}

TEST(ModelBenchmark, BuildMatchups_DefaultsModelsToPrimary)
{
	std::vector<ModelBenchmarkModel> models(2);
	std::vector<std::pair<int, int>> matchups = BuildModelBenchmarkMatchups(models);
	BENCHMARK_CHECK_EQ(matchups.size(), size_t(1));
	BENCHMARK_CHECK_TRUE(matchups[0] == std::make_pair(0, 1));
}

TEST(ModelBenchmark, GoalTimingUsesSimulatedSecondsAndIncludesBoundary)
{
	AssertNear((float)FinishedGameSeconds(150, false, 8), 10.0f);
	AssertNear((float)FinishedGameSeconds(1200, true, 8), 10.0f);
	BENCHMARK_CHECK_EQ(ClassifyGoalTiming(false, 10.0, 10.0), ModelBenchmarkGoalTiming::NoGoal);
	BENCHMARK_CHECK_EQ(ClassifyGoalTiming(true, 10.0, 10.0), ModelBenchmarkGoalTiming::Kickoff);
	BENCHMARK_CHECK_EQ(ClassifyGoalTiming(true, 10.001, 10.0), ModelBenchmarkGoalTiming::PostKickoff);
}

TEST(ModelBenchmark, TimingEloRequiresConnectedComparisonGraph)
{
	std::vector<BenchmarkOutcome> outcomes = {
		{ 0, 1, 1, ModelBenchmarkGoalTiming::Kickoff, 1.0 }
	};
	BENCHMARK_CHECK_TRUE(HasConnectedOutcomeGraph(2, outcomes));
	BENCHMARK_CHECK_FALSE(HasConnectedOutcomeGraph(3, outcomes));
	BENCHMARK_CHECK_EQ(CountModelOutcomes(outcomes, 0), 1);
	BENCHMARK_CHECK_EQ(CountModelOutcomes(outcomes, 2), 0);

	outcomes.push_back({ 1, 2, 1, ModelBenchmarkGoalTiming::Kickoff, 1.0 });
	BENCHMARK_CHECK_TRUE(HasConnectedOutcomeGraph(3, outcomes));
}

TEST(ModelBenchmark, OverallEloUsesIndirectOpponentBridge)
{
	std::vector<BenchmarkOutcome> outcomes;
	for (int i = 0; i < 20; i++) {
		outcomes.push_back({ 1, 0, 1, ModelBenchmarkGoalTiming::PostKickoff, 1.0 });
		outcomes.push_back({ 0, 2, 2, ModelBenchmarkGoalTiming::PostKickoff, 1.0 });
	}

	BENCHMARK_CHECK_TRUE(HasConnectedOutcomeGraph(3, outcomes));
	EloFit fit = FitElo(3, outcomes);
	BENCHMARK_CHECK_TRUE(fit.elo[1] > fit.elo[0]);
	BENCHMARK_CHECK_TRUE(fit.elo[0] > fit.elo[2]);
}

TEST(ModelBenchmark, EligiblePairModesPreserveBridgingGraph)
{
	const std::vector<std::pair<int, int>> matchups = {
		{ 0, 1 },
		{ 0, 2 }
	};
	const std::vector<PairKey> pairModes = BuildEligiblePairModes(
		matchups,
		{ 3, 1, 2 },
		{ 1, 2, 3 });
	const std::vector<PairKey> expected = {
		{ 0, 1, 1 },
		{ 0, 2, 1 },
		{ 0, 2, 2 }
	};

	BENCHMARK_CHECK_TRUE(pairModes == expected);
	BENCHMARK_CHECK_TRUE(HasConnectedPairModeGraph(3, pairModes));
	BENCHMARK_CHECK_EQ(CountEligiblePairsForMode(pairModes, 1), 2);
	BENCHMARK_CHECK_EQ(CountEligiblePairsForMode(pairModes, 2), 1);
	BENCHMARK_CHECK_EQ(CountEligiblePairsForMode(pairModes, 3), 0);
	AssertNear((float)OverallOutcomeWeightForPair(pairModes, 0, 1), 1.0f);
	AssertNear((float)OverallOutcomeWeightForPair(pairModes, 0, 2), 0.5f);
}

TEST(ModelBenchmark, EligibilityExcludesOnlyZeroGameModels)
{
	std::vector<ModelBenchmarkModel> models(3);
	const ModelBenchmarkEligibilityAnalysis analysis = AnalyzeModelBenchmarkEligibility(
		models,
		{ 2, 2, 1 },
		{ 2 });

	BENCHMARK_CHECK_TRUE(analysis.includedModelIndices == std::vector<int>({ 0, 1 }));
	BENCHMARK_CHECK_TRUE(analysis.excludedModelIndices == std::vector<int>({ 2 }));
	BENCHMARK_CHECK_TRUE(analysis.eligibleModes == std::vector<int>({ 2 }));
	BENCHMARK_CHECK_EQ(analysis.eligibleMatchupCount, 1);
	BENCHMARK_CHECK_EQ(analysis.eligiblePairModeCount, 1);
	BENCHMARK_CHECK_TRUE(analysis.comparisonGraphConnected);
}

TEST(ModelBenchmark, EligibilityPrunesPrimaryDependencyCascade)
{
	std::vector<ModelBenchmarkModel> models(3);
	models[1].primary = false;
	models[2].primary = false;
	const ModelBenchmarkEligibilityAnalysis analysis = AnalyzeModelBenchmarkEligibility(
		models,
		{ 1, 2, 2 },
		{ 2 });

	BENCHMARK_CHECK_TRUE(analysis.includedModelIndices.empty());
	BENCHMARK_CHECK_TRUE(analysis.excludedModelIndices == std::vector<int>({ 0, 1, 2 }));
	BENCHMARK_CHECK_FALSE(analysis.comparisonGraphConnected);
}

TEST(ModelBenchmark, SavedSummaryFiltersNonPrimaryMatchupsAndStats)
{
	ModelBenchmarkConfig config;
	config.run2v2 = false;
	config.run3v3 = false;
	config.models = {
		BenchmarkModel("Primary", true),
		BenchmarkModel("Secondary A", false),
		BenchmarkModel("Secondary B", false)
	};
	auto scoredGame = [](int a, int b, int winner, uint64_t id) {
		ModelBenchmarkGameMetadata game;
		game.gameId = id;
		game.snapshotId = id;
		game.modelA = a;
		game.modelB = b;
		game.blueModel = a;
		game.orangeModel = b;
		game.mode = 1;
		game.winner = winner;
		game.goalTimeSeconds = 20.0;
		game.goalTiming = ModelBenchmarkGoalTiming::PostKickoff;
		game.goalScored = true;
		game.includedInRating = true;
		game.mirrorCompleted = true;
		return game;
	};
	const std::vector<ModelBenchmarkGameMetadata> games = {
		scoredGame(0, 1, 0, 1),
		scoredGame(0, 2, 2, 2),
		// This stored matchup must not affect any displayed result because both
		// participants are non-primary in the current saved-data selection.
		scoredGame(1, 2, 1, 3)
	};
	ModelBenchmarkStatTotals normalStats;
	normalStats.gameTimeSum = 300.0;
	normalStats.goals = 1.0;
	ModelBenchmarkStatTotals excludedStats = normalStats;
	excludedStats.goals = 100.0;
	const std::vector<ModelBenchmarkStatTotalsRow> totals = {
		{ 0, 1, 1, normalStats },
		{ 1, 0, 1, normalStats },
		{ 0, 2, 1, normalStats },
		{ 2, 0, 1, normalStats },
		{ 1, 2, 1, excludedStats },
		{ 2, 1, 1, excludedStats }
	};

	const ModelBenchmarkResult result = SummarizeModelBenchmarkData(config, games, totals);
	BENCHMARK_CHECK_EQ(result.scoredGames, 2);
	BENCHMARK_CHECK_EQ(result.pairwise.size(), size_t(4));
	BENCHMARK_CHECK_EQ(result.statTotals.size(), size_t(4));
	BENCHMARK_CHECK_EQ(result.optionalStats.size(), size_t(3));
	for (const ModelBenchmarkPairwiseRow& row : result.pairwise)
		BENCHMARK_CHECK_TRUE(row.modelIndex == 0 || row.opponentIndex == 0);
	for (const ModelBenchmarkStatRow& row : result.optionalStats) {
		AssertNear((float)row.stats.goals, 1.0f);
		AssertNear((float)row.stats.equivalentGames, row.modelIndex == 0 ? 2.0f : 1.0f);
	}
}

TEST(ModelBenchmark, RectangularPairModesKeepLegacyOutcomeWeight)
{
	const std::vector<PairKey> pairModes = BuildEligiblePairModes(
		{ { 0, 1 }, { 0, 2 }, { 1, 2 } },
		{ 3, 3, 3 },
		{ 1, 2, 3 });

	AssertNear((float)OverallOutcomeWeightForPair(pairModes, 0, 1), 1.0f);
	AssertNear((float)OverallOutcomeWeightForPair(pairModes, 0, 2), 1.0f);
	AssertNear((float)OverallOutcomeWeightForPair(pairModes, 1, 2), 1.0f);
}

TEST(ModelBenchmark, PairModeGraphRejectsDisconnectedModels)
{
	const std::vector<PairKey> pairModes = {
		{ 0, 1, 1 }
	};
	BENCHMARK_CHECK_FALSE(HasConnectedPairModeGraph(3, pairModes));
}

TEST(ModelBenchmark, TimingEloUsesOverallFloor)
{
	EloFit overallFit = {};
	overallFit.elo = { -50.0, 0.0, 50.0 };
	EloFit kickoffFit = {};
	kickoffFit.elo = { -75.0, 25.0, 50.0 };
	EloFit postKickoffFit = {};
	postKickoffFit.elo = { -25.0, -10.0, 35.0 };

	AnchorFitsToOverallFloor(overallFit, kickoffFit, postKickoffFit);

	AssertNear((float)overallFit.elo[0], 0.0f);
	AssertNear((float)overallFit.elo[1], 50.0f);
	AssertNear((float)overallFit.elo[2], 100.0f);
	AssertNear((float)kickoffFit.elo[0], -25.0f);
	AssertNear((float)kickoffFit.elo[1], 75.0f);
	AssertNear((float)postKickoffFit.elo[0], 25.0f);
	AssertNear((float)postKickoffFit.elo[1], 40.0f);
	AssertNear(
		(float)(kickoffFit.elo[1] - postKickoffFit.elo[1]),
		35.0f);
}

TEST(ModelBenchmark, FixedRunProgressUsesScheduledWork)
{
	static_assert(std::is_same_v<decltype(ModelBenchmarkProgress{}.maxGames), int64_t>);
	static_assert(std::is_same_v<decltype(ModelBenchmarkResult{}.maxGames), int64_t>);
	BENCHMARK_CHECK_FALSE(ModelBenchmarkConfig{}.emitLiveResultSnapshots);
	BENCHMARK_CHECK_TRUE(ModelBenchmarkProgress{}.liveResult == nullptr);

	AssertNear((float)BenchmarkWorkProgress(true, 20, 80, 100, 2.0), 0.4f);
	AssertNear((float)BenchmarkWorkProgress(true, 70, 80, 100, 1.0), 0.8f);
	AssertNear((float)BenchmarkWorkProgress(false, 20, 80, 100, 2.0), 0.2f);
}

TEST(ModelBenchmark, ArenaLayoutCanOnlyBeReclaimedAfterItsTargetCompletes)
{
	BENCHMARK_CHECK_FALSE(BenchmarkArenaLayoutCanBeReclaimed(99, 100));
	BENCHMARK_CHECK_TRUE(BenchmarkArenaLayoutCanBeReclaimed(100, 100));
	BENCHMARK_CHECK_TRUE(BenchmarkArenaLayoutCanBeReclaimed(101, 100));
}

TEST(ModelBenchmark, RejectsInvalidKickoffTemperatureBeforeLoadingModels)
{
	ModelBenchmarkConfig config;
	config.models.resize(2);
	config.stochasticKickoff = true;
	for (float temperature : { 0.0f, -1.0f, std::numeric_limits<float>::infinity(),
		std::numeric_limits<float>::quiet_NaN() }) {
		config.kickoffTemperature = temperature;
		bool rejected = false;
		try {
			RunModelBenchmark(config);
		} catch (const std::runtime_error& error) {
			rejected = std::string(error.what()).find("kickoff temperature") != std::string::npos;
		}
		ASSERT_TRUE(rejected);
	}
}

static void CheckStochasticKickoff(RocketSim::EngineId engine)
{
	torch::NoGradGuard noGrad;
	BenchmarkModelSetOwner modelOwner;
	GGL::PartialModelConfig policy;
	policy.layerSizes = { 4 };
	policy.addLayerNorm = false;
	GGL::ModelManager::MakeModels(false, 1, 3, {}, policy, {}, torch::kCPU, modelOwner.models);
	for (auto* model : modelOwner.models)
		for (auto parameter : model->parameters())
			parameter.zero_();
	// Action 1 is preferred; action 2 has the highest logit but is always masked.
	auto bias = modelOwner.models["policy"]->parameters().back();
	ASSERT_EQ(bias.numel(), int64_t(3));
	bias.copy_(torch::tensor({ 0.0f, 2.0f, 50.0f }));

	const std::pair<bool, int> timings[] = {{false, 0}, {false, 1}, {true, 0}, {true, 1}};
	for (const auto& [variableTiming, actionDelay] : timings) {
		SCOPED_TRACE(variableTiming ? "variable timing" : "fixed timing");
		SCOPED_TRACE(actionDelay);
		for (int scenario = 0; scenario < 3; ++scenario) {
			std::map<RocketSim::Arena*, KickoffActionCounts> counts;
			std::vector<KickoffActionCounts> completed;
			GGL::PolicyMatchRuntime runtime;
			runtime.models = &modelOwner.models;
			runtime.obsSize = 1;
			runtime.numActions = 3;
			runtime.usesActionMask = true;
			runtime.deterministic = true;
			runtime.stochasticKickoff = scenario != 0;
			runtime.kickoffTemperature = scenario == 1 ? 10.0f : 0.01f;
			runtime.policyTemperature = 1.0f;
			runtime.tickSkip = 2;
			runtime.actionDelay = actionDelay;
			runtime.actionGenerator = at::detail::createCPUGenerator(1979);
			runtime.makeRuntimeComponents = [&]() {
				return GGL::PolicyMatchRuntimeComponents{
					std::make_unique<BenchmarkTinyObsBuilder>(),
					std::make_unique<KickoffSamplingActionParser>(counts)
				};
			};
			GGL::PolicyMatchRuntime opponent = runtime;
			opponent.tickSkip = variableTiming ? 3 : 2;
			opponent.runtimeGroup = 1;

			std::vector<GGL::PolicyMatchTask> tasks(4);
			for (int i = 0; i < (int)tasks.size(); ++i) {
				auto& task = tasks[i];
				task.blueModel = 0;
				task.orangeModel = 1;
				task.gameId = i;
				task.snapshot = GGL::SamplePolicyMatchStartState(
					GGL::PolicyMatchStartType::Kickoff, 1, RocketSim::CAR_CONFIG_OCTANE, 12345, engine);
				// Ball motion alone must not disable sampling before contact.
				task.snapshot.ball.vel.x = 10.0f;
			}
			size_t scheduled = 0;
			GGL::PolicyMatchRunnerParams params;
			params.runtimes = { runtime, opponent };
			params.device = torch::kCPU;
			params.numSlots = 2;
			params.tickSkip = 2;
			params.actionDelay = actionDelay;
			params.noGoalSteps = 96;
			params.noGoalTicks = 192;
			params.makeArena = [engine](const GGL::PolicyMatchTask&) {
				return GGL::PolicyMatchArena{ GGL::MakePolicyMatchArena(
					1, RocketSim::CAR_CONFIG_OCTANE, RocketSim::CAR_CONFIG_OCTANE, engine) };
			};
			params.fillPendingTasks = [&](std::vector<GGL::PolicyMatchTask>& pending, int, int) {
				while (scheduled < tasks.size())
					pending.push_back(tasks[scheduled++]);
			};
			params.isSchedulingComplete = [&]() { return scheduled == tasks.size(); };
			params.onStep = [&](const GGL::PolicyMatchStep& step) {
				auto* arena = step.state.lastArena;
				auto& count = counts[arena];
				const int touchStep = step.task.gameId % 2 == 0 ? 32 : 64;
				if (step.steps == touchStep) {
					// Stagger a synthetic contact across arenas to exercise mixed batches.
					auto* car = *arena->_cars.begin();
					auto state = car->GetState();
					state.ballHitInfo.isValid = true;
					state.ballHitInfo.tickCountWhenHit = arena->tickCount - 1;
					car->SetState(state);
					count.touched = true;
				} else if (step.steps == touchStep + 8) {
					// Losing contact metadata or returning to center must not restart sampling.
					for (auto* car : arena->_cars) {
						auto state = car->GetState();
						state.ballHitInfo = {};
						car->SetState(state);
					}
					arena->ball->SetState(RocketSim::BallState{});
				}
			};
			params.onFinished = [&](const GGL::PolicyMatchFinish& finish) {
				auto& count = counts[finish.state.lastArena];
				completed.push_back(count);
				count = {};
				return false;
			};
			const auto result = variableTiming
				? GGL::RunPolicyMatchVariableTimingSlots(params) : GGL::RunPolicyMatchSlots(params);
			ASSERT_FALSE(result.cancelled);
			ASSERT_EQ(completed.size(), size_t(4));
			ASSERT_EQ(result.timing.slotArenaCreates, int64_t(2));
			ASSERT_EQ(result.timing.slotArenaReuses, int64_t(2));
			for (const auto& count : completed) {
				for (int side = 0; side < 2; ++side) {
					ASSERT_TRUE(count.afterTouchDecisions[side] > 0);
					if (scenario == 1)
						ASSERT_TRUE(count.sampledAlternatives[side] > 0);
					else
						ASSERT_EQ(count.sampledAlternatives[side], 0);
				}
			}
		}
	}
}

TEST(ModelBenchmark, StochasticKickoffStopsAtTouchAndResetsOnSlotReuseV2)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	CheckStochasticKickoff(RocketSim::EngineId::V2);
}

TEST(ModelBenchmark, StochasticKickoffStopsAtTouchAndResetsOnSlotReuseV3)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	ROCKETFORGE_REQUIRE_ENGINE(RocketSim::EngineId::V3);
	CheckStochasticKickoff(RocketSim::EngineId::V3);
}

TEST(ModelBenchmark, DrawsAppearInOverallAndPairwiseTotals)
{
	ModelBenchmarkConfig config;
	config.models = { BenchmarkModel("A", true), BenchmarkModel("B", true) };
	config.run2v2 = config.run3v3 = false;
	config.stochasticKickoff = true;
	config.kickoffTemperature = 0.65f;
	std::vector<ModelBenchmarkGameMetadata> games(2);
	for (int i = 0; i < 2; ++i) {
		games[i].gameId = i;
		games[i].modelA = 0;
		games[i].modelB = 1;
		games[i].blueModel = i;
		games[i].orangeModel = 1 - i;
		games[i].mirrorCompleted = true;
	}
	const ModelBenchmarkResult result = SummarizeModelBenchmarkData(config, games);
	BENCHMARK_CHECK_EQ(result.noResultGames, 2);
	BENCHMARK_CHECK_EQ(result.scoredGames, 0);
	BENCHMARK_CHECK_FALSE(result.overallEloAvailable);
	BENCHMARK_CHECK_TRUE(result.stochasticKickoff);
	BENCHMARK_CHECK_EQ(result.kickoffTemperature, 0.65f);
	BENCHMARK_CHECK_EQ(result.rankings.size(), size_t(2));
	BENCHMARK_CHECK_EQ(result.pairwise.size(), size_t(2));
	for (const auto& row : result.rankings)
		BENCHMARK_CHECK_EQ(row.totalRecord.noResults, 2);
	for (const auto& row : result.pairwise)
		BENCHMARK_CHECK_EQ(row.combined.noResults, 2);
}

TEST(ModelBenchmark, PolicyMatchRunnerReclaimsCompletedLayoutsWithoutDrainingOtherSlots)
{
	BenchmarkModelSetOwner modelOwner;
	GGL::PartialModelConfig policyConfig = {};
	policyConfig.layerSizes = { 4 };
	GGL::ModelManager::MakeModels(
		false,
		1,
		1,
		{},
		policyConfig,
		{},
		torch::Device(torch::kCPU),
		modelOwner.models);

	GGL::PolicyMatchRuntime runtime = {};
	runtime.models = &modelOwner.models;
	runtime.obsSize = 1;
	runtime.numActions = 1;
	runtime.tickSkip = 1;
	runtime.actionDelay = 0;
	runtime.runtimeGroup = 0;
	runtime.makeRuntimeComponents = []() {
		return GGL::PolicyMatchRuntimeComponents{
			std::make_unique<BenchmarkTinyObsBuilder>(),
			std::make_unique<BenchmarkTinyActionParser>()
		};
	};

	const GGL::PolicyMatchSnapshot snapshot = GGL::SamplePolicyMatchStartState(
		GGL::PolicyMatchStartType::Kickoff,
		1,
		RocketSim::CAR_CONFIG_OCTANE,
		12345,
		RocketSim::EngineId::V2);
	std::vector<GGL::PolicyMatchTask> tasks;
	auto appendLayoutTasks = [&](int layout, int resumeSteps) {
		for (int i = 0; i < 2; i++) {
			GGL::PolicyMatchTask task = {};
			task.blueModel = 0;
			task.orangeModel = 0;
			task.blueArenaKey = layout;
			task.orangeArenaKey = layout;
			task.teamSize = 1;
			task.resumeSteps = resumeSteps;
			task.snapshot = snapshot;
			task.userValues[0] = layout;
			tasks.push_back(std::move(task));
		}
	};
	// The runner pops pending tasks from the back, leaving layout 2 queued behind
	// fast layout 0 and slow layout 1 in the initial four slots.
	appendLayoutTasks(2, 2);
	appendLayoutTasks(0, 2);
	appendLayoutTasks(1, 0);

	size_t nextTask = 0;
	std::array<int, 3> completedByLayout = { 0, 0, 0 };
	int maxActiveSlots = 0;
	bool reclaimedLayoutRefilledCapacityBeforeMiddleLayoutDrained = false;
	bool pendingLayoutBeganBeforeSlowLayoutDrained = false;

	GGL::PolicyMatchRunnerParams params = {};
	params.runtimes = { runtime };
	params.device = torch::Device(torch::kCPU);
	params.numSlots = 4;
	params.tickSkip = 1;
	params.actionDelay = 0;
	params.noGoalSteps = 3;
	params.makeArena = [](const GGL::PolicyMatchTask&) {
		return GGL::PolicyMatchArena{
			GGL::MakePolicyMatchArena(
				1,
				RocketSim::CAR_CONFIG_OCTANE,
				RocketSim::CAR_CONFIG_OCTANE,
				RocketSim::EngineId::V2)
		};
	};
	params.fillPendingTasks = [&](std::vector<GGL::PolicyMatchTask>& pending, int, int) {
		while (nextTask < tasks.size())
			pending.push_back(tasks[nextTask++]);
	};
	params.isSchedulingComplete = [&]() {
		return nextTask == tasks.size();
	};
	params.isArenaLayoutComplete = [&](int teamSize, int blueArenaKey, int orangeArenaKey) {
		return teamSize == 1
			&& blueArenaKey == orangeArenaKey
			&& BenchmarkArenaLayoutCanBeReclaimed(completedByLayout[blueArenaKey], 2);
	};
	params.onStep = [&](const GGL::PolicyMatchStep& step) {
		if (step.task.userValues[0] == 2 && completedByLayout[1] < 2)
			pendingLayoutBeganBeforeSlowLayoutDrained = true;
	};
	params.onFinished = [&](const GGL::PolicyMatchFinish& finish) {
		completedByLayout[finish.task.userValues[0]]++;
		return false;
	};
	params.onStatus = [&](const GGL::PolicyMatchRunnerStatus& status) {
		maxActiveSlots = std::max(maxActiveSlots, status.activeSlots);
		if (status.slotArenaRebuilds >= 2
			&& status.activeSlots == params.numSlots
			&& completedByLayout[1] < 2) {
			reclaimedLayoutRefilledCapacityBeforeMiddleLayoutDrained = true;
		}
	};

	GGL::PolicyMatchRunnerResult result = GGL::RunPolicyMatchSlots(params);

	BENCHMARK_CHECK_FALSE(result.cancelled);
	BENCHMARK_CHECK_EQ(completedByLayout[0], 2);
	BENCHMARK_CHECK_EQ(completedByLayout[1], 2);
	BENCHMARK_CHECK_EQ(completedByLayout[2], 2);
	BENCHMARK_CHECK_EQ(maxActiveSlots, 4);
	BENCHMARK_CHECK_TRUE(reclaimedLayoutRefilledCapacityBeforeMiddleLayoutDrained);
	BENCHMARK_CHECK_TRUE(pendingLayoutBeganBeforeSlowLayoutDrained);
	BENCHMARK_CHECK_TRUE(result.timing.slotArenaRebuilds >= 2);
}

TEST(ModelBenchmark, PolicyMatchRunnerStarvationFallbackWaitsForActiveLayoutsToDrain)
{
	BenchmarkModelSetOwner modelOwner;
	GGL::PartialModelConfig policyConfig = {};
	policyConfig.layerSizes = { 4 };
	GGL::ModelManager::MakeModels(
		false,
		1,
		1,
		{},
		policyConfig,
		{},
		torch::Device(torch::kCPU),
		modelOwner.models);

	GGL::PolicyMatchRuntime runtime = {};
	runtime.models = &modelOwner.models;
	runtime.obsSize = 1;
	runtime.numActions = 1;
	runtime.tickSkip = 1;
	runtime.actionDelay = 0;
	runtime.runtimeGroup = 0;
	runtime.makeRuntimeComponents = []() {
		return GGL::PolicyMatchRuntimeComponents{
			std::make_unique<BenchmarkTinyObsBuilder>(),
			std::make_unique<BenchmarkTinyActionParser>()
		};
	};

	const GGL::PolicyMatchSnapshot snapshot = GGL::SamplePolicyMatchStartState(
		GGL::PolicyMatchStartType::Kickoff,
		1,
		RocketSim::CAR_CONFIG_OCTANE,
		67890,
		RocketSim::EngineId::V2);

	auto runRegression = [&](bool variableTiming) {
		std::vector<GGL::PolicyMatchTask> tasks;
		auto appendLayoutTasks = [&](int layout, int count, int resumeSteps) {
			for (int i = 0; i < count; i++) {
				GGL::PolicyMatchTask task = {};
				task.blueModel = 0;
				task.orangeModel = 0;
				task.blueArenaKey = layout;
				task.orangeArenaKey = layout;
				task.teamSize = 1;
				task.resumeSteps = resumeSteps;
				task.snapshot = snapshot;
				task.userValues[0] = layout;
				tasks.push_back(std::move(task));
			}
		};
		// The first four tasks include one fast, one medium, and two slow slots.
		// Layout 2 remains pending while those nonterminal layouts drain, then the
		// deadlock fallback builds bounded parallel capacity for it.
		appendLayoutTasks(0, 1, 2);
		appendLayoutTasks(1, 1, 1);
		appendLayoutTasks(1, 2, 0);
		appendLayoutTasks(2, 4, 2);

		size_t nextTask = 0;
		std::array<int, 3> completedByLayout = { 0, 0, 0 };
		bool fallbackRanBeforeActiveLayoutsDrained = false;
		bool fallbackBuiltParallelCapacityAfterDrain = false;

		GGL::PolicyMatchRunnerParams params = {};
		params.runtimes = { runtime };
		params.device = torch::Device(torch::kCPU);
		params.numSlots = 4;
		params.tickSkip = 1;
		params.actionDelay = 0;
		params.noGoalSteps = 3;
		params.noGoalTicks = 3;
		params.makeArena = [](const GGL::PolicyMatchTask&) {
			return GGL::PolicyMatchArena{
				GGL::MakePolicyMatchArena(
					1,
					RocketSim::CAR_CONFIG_OCTANE,
					RocketSim::CAR_CONFIG_OCTANE,
					RocketSim::EngineId::V2)
			};
		};
		params.fillPendingTasks = [&](std::vector<GGL::PolicyMatchTask>& pending, int active, int target) {
			while (nextTask < tasks.size() && (int)pending.size() + active < target)
				pending.push_back(tasks[nextTask++]);
		};
		params.isSchedulingComplete = [&]() {
			return nextTask == tasks.size();
		};
		params.isArenaLayoutComplete = [](int, int, int) {
			return false;
		};
		params.onFinished = [&](const GGL::PolicyMatchFinish& finish) {
			completedByLayout[finish.task.userValues[0]]++;
			return false;
		};
		params.onStatus = [&](const GGL::PolicyMatchRunnerStatus& status) {
			if (status.slotArenaRebuilds > 0 && completedByLayout[1] < 3)
				fallbackRanBeforeActiveLayoutsDrained = true;
			if (status.slotArenaRebuilds == 2
				&& status.activeSlots == 2
				&& completedByLayout[1] == 3)
				fallbackBuiltParallelCapacityAfterDrain = true;
		};

		GGL::PolicyMatchRunnerResult result = variableTiming
			? GGL::RunPolicyMatchVariableTimingSlots(params)
			: GGL::RunPolicyMatchSlots(params);

		BENCHMARK_CHECK_FALSE(result.cancelled);
		BENCHMARK_CHECK_EQ(completedByLayout[0], 1);
		BENCHMARK_CHECK_EQ(completedByLayout[1], 3);
		BENCHMARK_CHECK_EQ(completedByLayout[2], 4);
		BENCHMARK_CHECK_FALSE(fallbackRanBeforeActiveLayoutsDrained);
		BENCHMARK_CHECK_TRUE(fallbackBuiltParallelCapacityAfterDrain);
		BENCHMARK_CHECK_EQ(result.timing.slotArenaRebuilds, uint64_t(2));
	};

	runRegression(false);
	runRegression(true);
}

TEST(ModelBenchmark, PolicyMatchRunnerRequestsCompatibleWorkBeforeRebuilding)
{
	BenchmarkModelSetOwner modelOwner;
	GGL::PartialModelConfig policyConfig = {};
	policyConfig.layerSizes = { 4 };
	GGL::ModelManager::MakeModels(
		false,
		1,
		1,
		{},
		policyConfig,
		{},
		torch::Device(torch::kCPU),
		modelOwner.models);

	GGL::PolicyMatchRuntime runtime = {};
	runtime.models = &modelOwner.models;
	runtime.obsSize = 1;
	runtime.numActions = 1;
	runtime.tickSkip = 1;
	runtime.actionDelay = 0;
	runtime.runtimeGroup = 0;
	runtime.makeRuntimeComponents = []() {
		return GGL::PolicyMatchRuntimeComponents{
			std::make_unique<BenchmarkTinyObsBuilder>(),
			std::make_unique<BenchmarkTinyActionParser>()
		};
	};

	const GGL::PolicyMatchSnapshot snapshot = GGL::SamplePolicyMatchStartState(
		GGL::PolicyMatchStartType::Kickoff,
		1,
		RocketSim::CAR_CONFIG_OCTANE,
		24680,
		RocketSim::EngineId::V2);

	auto runRegression = [&](bool variableTiming) {
		auto makeTask = [&](int layout, int resumeSteps, int sequence) {
			GGL::PolicyMatchTask task = {};
			task.blueModel = 0;
			task.orangeModel = 0;
			task.blueArenaKey = layout;
			task.orangeArenaKey = layout;
			task.teamSize = 1;
			task.resumeSteps = resumeSteps;
			task.snapshot = snapshot;
			task.userValues[0] = layout;
			task.userValues[1] = sequence;
			return task;
		};

		std::array<int, 3> completedByLayout = { 0, 0, 0 };
		bool initialBatchQueued = false;
		bool incompatibleTaskQueued = false;
		bool compatibleTaskQueued = false;
		bool compatibleTaskRanBeforeSlowLayoutDrained = false;
		bool rebuiltNonterminalLayout = false;
		int demandCallbacks = 0;

		GGL::PolicyMatchRunnerParams params = {};
		params.runtimes = { runtime };
		params.device = torch::Device(torch::kCPU);
		params.numSlots = 4;
		params.tickSkip = 1;
		params.actionDelay = 0;
		params.noGoalSteps = 3;
		params.noGoalTicks = 3;
		params.makeArena = [](const GGL::PolicyMatchTask&) {
			return GGL::PolicyMatchArena{
				GGL::MakePolicyMatchArena(
					1,
					RocketSim::CAR_CONFIG_OCTANE,
					RocketSim::CAR_CONFIG_OCTANE,
					RocketSim::EngineId::V2)
			};
		};
		params.fillPendingTasks = [&](std::vector<GGL::PolicyMatchTask>& pending, int, int) {
			if (!initialBatchQueued) {
				// LIFO assignment puts the fast layout-0 task in the first slot.
				pending.push_back(makeTask(1, 0, 0));
				pending.push_back(makeTask(1, 0, 0));
				pending.push_back(makeTask(1, 0, 0));
				pending.push_back(makeTask(0, 2, 0));
				initialBatchQueued = true;
			} else if (!incompatibleTaskQueued) {
				pending.push_back(makeTask(2, 2, 0));
				incompatibleTaskQueued = true;
			}
		};
		params.fillPendingTasksForArenaLayouts = [&](
			std::vector<GGL::PolicyMatchTask>& pending,
			const std::vector<GGL::PolicyMatchArenaLayoutDemand>& demands) {
			demandCallbacks++;
			const bool layoutZeroRequested = std::any_of(
				demands.begin(),
				demands.end(),
				[](const GGL::PolicyMatchArenaLayoutDemand& demand) {
					return demand.teamSize == 1
						&& demand.blueArenaKey == 0
						&& demand.orangeArenaKey == 0
						&& demand.idleSlots == 1;
				});
			if (layoutZeroRequested && !compatibleTaskQueued) {
				pending.push_back(makeTask(0, 2, 1));
				compatibleTaskQueued = true;
			}
		};
		params.isSchedulingComplete = [&]() {
			return incompatibleTaskQueued && compatibleTaskQueued;
		};
		params.isArenaLayoutComplete = [&](int teamSize, int blueArenaKey, int orangeArenaKey) {
			if (teamSize != 1 || blueArenaKey != orangeArenaKey)
				return false;
			const std::array<int, 3> targets = { 2, 3, 1 };
			return completedByLayout[blueArenaKey] >= targets[blueArenaKey];
		};
		params.onStep = [&](const GGL::PolicyMatchStep& step) {
			if (step.task.userValues[0] == 0
				&& step.task.userValues[1] == 1
				&& completedByLayout[1] < 3) {
				compatibleTaskRanBeforeSlowLayoutDrained = true;
			}
		};
		params.onFinished = [&](const GGL::PolicyMatchFinish& finish) {
			completedByLayout[finish.task.userValues[0]]++;
			return false;
		};
		params.onStatus = [&](const GGL::PolicyMatchRunnerStatus& status) {
			if (status.slotArenaRebuilds > 0 && completedByLayout[0] < 2)
				rebuiltNonterminalLayout = true;
		};

		GGL::PolicyMatchRunnerResult result = variableTiming
			? GGL::RunPolicyMatchVariableTimingSlots(params)
			: GGL::RunPolicyMatchSlots(params);

		BENCHMARK_CHECK_FALSE(result.cancelled);
		BENCHMARK_CHECK_TRUE(demandCallbacks > 0);
		BENCHMARK_CHECK_TRUE(compatibleTaskQueued);
		BENCHMARK_CHECK_TRUE(compatibleTaskRanBeforeSlowLayoutDrained);
		BENCHMARK_CHECK_FALSE(rebuiltNonterminalLayout);
		BENCHMARK_CHECK_EQ(completedByLayout[0], 2);
		BENCHMARK_CHECK_EQ(completedByLayout[1], 3);
		BENCHMARK_CHECK_EQ(completedByLayout[2], 1);
		BENCHMARK_CHECK_EQ(result.timing.slotArenaRebuilds, uint64_t(1));
	};

	runRegression(false);
	runRegression(true);
}

TEST(ModelBenchmark, FixedSchedulerUsesDurationWeightedInitialSplitAndDemandRefill)
{
	ModelBenchmarkConfig config = {};
	config.runUntilMaxGames = true;
	ModelBenchmarkCancelCallback cancelCallback = {};
	std::vector<LoadedModel> loadedModels(2);
	loadedModels[0].name = "A";
	loadedModels[1].name = "B";
	loadedModels[0].carConfig = RocketSim::CAR_CONFIG_OCTANE;
	loadedModels[1].carConfig = RocketSim::CAR_CONFIG_OCTANE;

	const std::vector<int> modes = { 1, 2, 3 };
	const std::vector<std::pair<int, int>> matchups = { { 0, 1 } };
	const std::vector<std::string> modelNames = { "A", "B" };
	BenchmarkRunState state = {};
	state.stats.resize(2);
	state.targetGamesPerPairMode = 20;
	for (int mode : modes) {
		state.gamesByMode[mode] = 0;
		state.scheduledGamesByMode[mode] = 0;
		state.stepsByMode[mode] = 0;
		state.scheduledByPairMode[{ 0, 1, mode }] = 0;
	}

	bool runnerCalled = false;
	SlotBenchmarkParams params = {
		config,
		cancelCallback,
		loadedModels,
		modes,
		matchups,
		modelNames,
		ModelBenchmarkStrategy::SameRuntimeSlot,
		"Benchmarking",
		"test slot path",
		2,
		12,
		1,
		0,
		1,
		1,
		false,
		12345,
		[](const std::string&, const std::string&, const std::string&) {},
		[](int) { return false; },
		[]() { return false; }
	};
	params.runPolicyMatchRunner = [&](const GGL::PolicyMatchRunnerParams& runnerParams) {
		runnerCalled = true;
		BENCHMARK_CHECK_TRUE((bool)runnerParams.fillPendingTasks);
		BENCHMARK_CHECK_TRUE((bool)runnerParams.fillPendingTasksForArenaLayouts);
		std::vector<GGL::PolicyMatchTask> pendingTasks;
		runnerParams.fillPendingTasks(pendingTasks, 0, 12);

		auto taskCountForMode = [&](int mode) {
			return (int)std::count_if(
				pendingTasks.begin(),
				pendingTasks.end(),
				[&](const GGL::PolicyMatchTask& task) {
					return task.teamSize == mode;
				});
		};
		BENCHMARK_CHECK_EQ(pendingTasks.size(), size_t(12));
		BENCHMARK_CHECK_EQ(taskCountForMode(1), 2);
		BENCHMARK_CHECK_EQ(taskCountForMode(2), 4);
		BENCHMARK_CHECK_EQ(taskCountForMode(3), 6);

		runnerParams.fillPendingTasksForArenaLayouts(
			pendingTasks,
			{ { 1, 0, 0, 4 } });

		auto compatibleTaskCount = [&]() {
			return (int)std::count_if(
				pendingTasks.begin(),
				pendingTasks.end(),
				[](const GGL::PolicyMatchTask& task) {
					return task.teamSize == 1
						&& task.blueArenaKey == 0
						&& task.orangeArenaKey == 0;
				});
		};
		BENCHMARK_CHECK_EQ(compatibleTaskCount(), 4);
		BENCHMARK_CHECK_EQ(pendingTasks.size(), size_t(14));

		// Existing compatible tasks count toward the next requested capacity.
		runnerParams.fillPendingTasksForArenaLayouts(
			pendingTasks,
			{ { 1, 0, 0, 6 } });
		BENCHMARK_CHECK_EQ(compatibleTaskCount(), 6);
		BENCHMARK_CHECK_EQ(pendingTasks.size(), size_t(24));
		return GGL::PolicyMatchRunnerResult{};
	};

	RunSlotBenchmark(params, state);

	BENCHMARK_CHECK_TRUE(runnerCalled);
	BENCHMARK_CHECK_EQ(state.scheduledGamesByMode[1], 6);
	BENCHMARK_CHECK_EQ(state.scheduledGamesByMode[2], 8);
	BENCHMARK_CHECK_EQ(state.scheduledGamesByMode[3], 10);
	BENCHMARK_CHECK_EQ(state.scheduledGames, 24);
}

TEST(ModelBenchmark, SlotStatsTrackOverallAndIndividualModes)
{
	ModelBenchmarkConfig config = {};
	config.runUntilMaxGames = true;
	config.collectOptionalStats = true;
	ModelBenchmarkCancelCallback cancelCallback = {};
	std::vector<LoadedModel> loadedModels(2);
	loadedModels[0].name = "A";
	loadedModels[1].name = "B";
	loadedModels[0].carConfig = RocketSim::CAR_CONFIG_OCTANE;
	loadedModels[1].carConfig = RocketSim::CAR_CONFIG_OCTANE;

	const std::vector<int> modes = { 1, 2 };
	const std::vector<std::pair<int, int>> matchups = { { 0, 1 } };
	const std::vector<std::string> modelNames = { "A", "B" };
	BenchmarkRunState state = {};
	state.stats.resize(2);
	state.targetGamesPerPairMode = 1;
	for (int mode : modes) {
		state.gamesByMode[mode] = 0;
		state.scheduledGamesByMode[mode] = 0;
		state.stepsByMode[mode] = 0;
		state.scheduledByPairMode[{ 0, 1, mode }] = 0;
	}

	SlotBenchmarkParams params = {
		config,
		cancelCallback,
		loadedModels,
		modes,
		matchups,
		modelNames,
		ModelBenchmarkStrategy::SameRuntimeSlot,
		"Benchmarking",
		"test slot path",
		2,
		2,
		1,
		0,
		1,
		1,
		false,
		12345,
		[](const std::string&, const std::string&, const std::string&) {},
		[](int) { return false; },
		[]() { return false; }
	};
	params.runPolicyMatchRunner = [&](const GGL::PolicyMatchRunnerParams& runnerParams) {
		GGL::PolicyMatchTask task = {};
		task.blueModel = 0;
		task.orangeModel = 1;
		task.teamSize = 2;

		RLGC::GameState previous = {};
		previous.players.resize(2);
		previous.players[0].carId = 31;
		previous.players[0].team = RocketSim::Team::BLUE;
		previous.players[1].carId = 32;
		previous.players[1].team = RocketSim::Team::ORANGE;

		RLGC::GameState current = previous;
		current.players[0].eventState.goal = true;
		current.players[0].isOnGround = false;
		current.players[0].boost = 10.0f;
		runnerParams.onStep({ task, current, previous, 1, 300.0f });
		return GGL::PolicyMatchRunnerResult{};
	};

	RunSlotBenchmark(params, state);

	const ModelBenchmarkOptionalStats overall = FinalizeStats(state.stats[0]);
	const ModelBenchmarkOptionalStats mode1v1 = FinalizeStats(state.statsByMode.at(1)[0]);
	const ModelBenchmarkOptionalStats mode2v2 = FinalizeStats(state.statsByMode.at(2)[0]);
	AssertNear((float)overall.equivalentGames, 1.0f);
	AssertNear((float)overall.goals, 1.0f);
	AssertNear((float)mode1v1.equivalentGames, 0.0f);
	AssertNear((float)mode2v2.equivalentGames, 1.0f);
	AssertNear((float)mode2v2.goals, 1.0f);
}

TEST(ModelBenchmark, SchedulerOnlyQueuesEligiblePairModes)
{
	ModelBenchmarkConfig config = {};
	config.runUntilMaxGames = true;
	ModelBenchmarkCancelCallback cancelCallback = {};
	std::vector<LoadedModel> loadedModels(3);
	for (int i = 0; i < 3; i++) {
		loadedModels[i].name = std::string(1, (char)('A' + i));
		loadedModels[i].carConfig = RocketSim::CAR_CONFIG_OCTANE;
	}

	const std::vector<int> modes = { 1, 2 };
	const std::vector<std::pair<int, int>> matchups = {
		{ 0, 1 },
		{ 0, 2 },
		{ 1, 2 }
	};
	const std::vector<std::string> modelNames = { "A", "B", "C" };
	BenchmarkRunState state = {};
	state.stats.resize(3);
	state.targetGamesPerPairMode = 2;
	for (int mode : modes) {
		state.gamesByMode[mode] = 0;
		state.scheduledGamesByMode[mode] = 0;
		state.stepsByMode[mode] = 0;
	}
	state.scheduledByPairMode[{ 0, 1, 1 }] = 0;
	state.scheduledByPairMode[{ 0, 2, 2 }] = 0;

	SlotBenchmarkParams params = {
		config,
		cancelCallback,
		loadedModels,
		modes,
		matchups,
		modelNames,
		ModelBenchmarkStrategy::Heterogeneous,
		"Benchmarking",
		"test slot path",
		3,
		8,
		1,
		0,
		1,
		1,
		false,
		12345,
		[](const std::string&, const std::string&, const std::string&) {},
		[](int) { return false; },
		[]() { return false; }
	};
	params.runPolicyMatchRunner = [&](const GGL::PolicyMatchRunnerParams& runnerParams) {
		std::vector<GGL::PolicyMatchTask> pendingTasks;
		runnerParams.fillPendingTasks(pendingTasks, 0, 8);
		BENCHMARK_CHECK_EQ(pendingTasks.size(), size_t(4));

		auto taskCount = [&](int modelA, int modelB, int mode) {
			return (int)std::count_if(
				pendingTasks.begin(),
				pendingTasks.end(),
				[&](const GGL::PolicyMatchTask& task) {
					return task.userValues[0] == modelA
						&& task.userValues[1] == modelB
						&& task.teamSize == mode;
				});
		};
		BENCHMARK_CHECK_EQ(taskCount(0, 1, 1), 2);
		BENCHMARK_CHECK_EQ(taskCount(0, 2, 2), 2);
		BENCHMARK_CHECK_EQ(taskCount(0, 1, 2), 0);
		BENCHMARK_CHECK_EQ(taskCount(0, 2, 1), 0);
		BENCHMARK_CHECK_EQ(taskCount(1, 2, 1), 0);
		BENCHMARK_CHECK_EQ(taskCount(1, 2, 2), 0);
		return GGL::PolicyMatchRunnerResult{};
	};

	RunSlotBenchmark(params, state);

	BENCHMARK_CHECK_EQ(state.scheduledGamesByMode[1], 2);
	BENCHMARK_CHECK_EQ(state.scheduledGamesByMode[2], 2);
	BENCHMARK_CHECK_EQ(state.scheduledGames, 4);
}

TEST(ModelBenchmark, SchedulerUsesIndependentDeficits)
{
	ModelBenchmarkConfig config = {};
	config.runUntilMaxGames = true;
	ModelBenchmarkCancelCallback cancelCallback = {};
	std::vector<LoadedModel> loadedModels(3);
	for (int i = 0; i < 3; i++) {
		loadedModels[i].name = std::string(1, (char)('A' + i));
		loadedModels[i].carConfig = RocketSim::CAR_CONFIG_OCTANE;
	}

	const std::vector<int> modes = { 1, 2 };
	const std::vector<std::pair<int, int>> matchups = { { 0, 1 }, { 0, 2 }, { 1, 2 } };
	const std::vector<std::string> modelNames = { "A", "B", "C" };

	BenchmarkRunState state = {};
	state.stats.resize(3);
	state.targetGamesPerPairMode = 1000;
	state.targetByPairMode[{ 0, 1, 1 }] = 3;
	state.targetByPairMode[{ 0, 2, 2 }] = 1;
	for (int mode : modes) {
		state.gamesByMode[mode] = 0;
		state.scheduledGamesByMode[mode] = 0;
		state.stepsByMode[mode] = 0;
	}
	state.scheduledByPairMode[{ 0, 1, 1 }] = 0;
	state.scheduledByPairMode[{ 0, 2, 2 }] = 0;

	SlotBenchmarkParams params = {
		config,
		cancelCallback,
		loadedModels,
		modes,
		matchups,
		modelNames,
		ModelBenchmarkStrategy::Heterogeneous,
		"Benchmarking",
		"test slot path",
		3,
		8,
		1,
		0,
		1,
		1,
		false,
		12345,
		[](const std::string&, const std::string&, const std::string&) {},
		[](int) { return false; },
		[]() { return false; }
	};
	params.runPolicyMatchRunner = [&](const GGL::PolicyMatchRunnerParams& runnerParams) {
		std::vector<GGL::PolicyMatchTask> pendingTasks;
		runnerParams.fillPendingTasks(pendingTasks, 0, 8);
		BENCHMARK_CHECK_EQ(pendingTasks.size(), size_t(4));

		auto taskCount = [&](int modelA, int modelB, int mode) {
			return (int)std::count_if(pendingTasks.begin(), pendingTasks.end(), [&](const GGL::PolicyMatchTask& task) {
				return task.userValues[0] == modelA && task.userValues[1] == modelB && task.teamSize == mode;
			});
		};

		BENCHMARK_CHECK_EQ(taskCount(0, 1, 1), 3);
		BENCHMARK_CHECK_EQ(taskCount(0, 2, 2), 1);
		BENCHMARK_CHECK_EQ(taskCount(0, 1, 2), 0);
		BENCHMARK_CHECK_EQ(taskCount(0, 2, 1), 0);
		BENCHMARK_CHECK_EQ(taskCount(1, 2, 1), 0);
		BENCHMARK_CHECK_EQ(taskCount(1, 2, 2), 0);
		return GGL::PolicyMatchRunnerResult{};
	};

	RunSlotBenchmark(params, state);

	BENCHMARK_CHECK_EQ(state.scheduledGamesByMode[1], 3);
	BENCHMARK_CHECK_EQ(state.scheduledGamesByMode[2], 1);
	BENCHMARK_CHECK_EQ(state.scheduledGames, 4);
}

TEST(ModelBenchmark, MirrorGroupOnlyCommitsCompleteScoredPairs)
{
	BENCHMARK_CHECK_FALSE(BenchmarkMirrorGroupComplete(1, 2));
	BENCHMARK_CHECK_TRUE(BenchmarkMirrorGroupComplete(2, 2));
	BENCHMARK_CHECK_FALSE(IncludeBenchmarkGameInRating(false, true));
	BENCHMARK_CHECK_FALSE(IncludeBenchmarkGameInRating(true, false));
	BENCHMARK_CHECK_TRUE(IncludeBenchmarkGameInRating(true, true));

	auto makeGame = [](uint64_t gameId, int blueModel, int orangeModel, int winner, double goalTimeSeconds) {
		ModelBenchmarkGameMetadata game = {};
		game.gameId = gameId;
		game.snapshotId = 42;
		game.modelA = 0;
		game.modelB = 1;
		game.blueModel = blueModel;
		game.orangeModel = orangeModel;
		game.mode = 1;
		game.winner = winner;
		game.goalTimeSeconds = goalTimeSeconds;
		game.goalTiming = ClassifyGoalTiming(true, goalTimeSeconds, 10.0);
		game.goalScored = true;
		return game;
	};

	// This is the production cancellation state: one physical half completed,
	// but its mirror never arrived, so no rating or record may be committed.
	BenchmarkRunState cancelledState = {};
	BenchmarkMirrorBuffer cancelledBuffer;
	BENCHMARK_CHECK_FALSE(cancelledBuffer.Add(cancelledState, makeGame(100, 0, 1, 0, 4.0), 2));
	cancelledState.cancelled = true;
	BENCHMARK_CHECK_EQ(cancelledState.games.size(), size_t(1));
	BENCHMARK_CHECK_FALSE(cancelledState.games[0].mirrorCompleted);
	BENCHMARK_CHECK_FALSE(cancelledState.games[0].includedInRating);
	BENCHMARK_CHECK_TRUE(cancelledState.outcomes.empty());
	BENCHMARK_CHECK_TRUE(cancelledState.records.empty());

	BenchmarkRunState completedState = {};
	BenchmarkMirrorBuffer completedBuffer;
	BENCHMARK_CHECK_FALSE(completedBuffer.Add(completedState, makeGame(200, 0, 1, 0, 4.0), 2, 0.5));
	BENCHMARK_CHECK_TRUE(completedBuffer.Add(completedState, makeGame(201, 1, 0, 1, 15.0), 2, 0.5));
	BENCHMARK_CHECK_EQ(completedState.games.size(), size_t(2));
	BENCHMARK_CHECK_EQ(completedState.outcomes.size(), size_t(2));
	AssertNear((float)completedState.outcomes[0].weight, 0.5f);
	AssertNear((float)completedState.outcomes[1].weight, 0.5f);
	BENCHMARK_CHECK_EQ(FilterOutcomes(completedState.outcomes, ModelBenchmarkGoalTiming::Kickoff).size(), size_t(1));
	BENCHMARK_CHECK_EQ(FilterOutcomes(completedState.outcomes, ModelBenchmarkGoalTiming::PostKickoff).size(), size_t(1));
	BENCHMARK_CHECK_FALSE(completedState.records.empty());
	const ModelBenchmarkRecord& overall = GetRecord(
		completedState.records,
		0,
		1,
		1,
		BenchmarkRecordType::Overall);
	const ModelBenchmarkRecord& kickoff = GetRecord(
		completedState.records,
		0,
		1,
		1,
		BenchmarkRecordType::Kickoff);
	const ModelBenchmarkRecord& postKickoff = GetRecord(
		completedState.records,
		0,
		1,
		1,
		BenchmarkRecordType::PostKickoff);
	BENCHMARK_CHECK_EQ(overall.wins, 1);
	BENCHMARK_CHECK_EQ(overall.losses, 1);
	BENCHMARK_CHECK_EQ(kickoff.wins, 1);
	BENCHMARK_CHECK_EQ(kickoff.losses, 0);
	BENCHMARK_CHECK_EQ(postKickoff.wins, 0);
	BENCHMARK_CHECK_EQ(postKickoff.losses, 1);
	for (const ModelBenchmarkGameMetadata& game : completedState.games) {
		BENCHMARK_CHECK_TRUE(game.mirrorCompleted);
		BENCHMARK_CHECK_TRUE(game.includedInRating);
	}
}

TEST(ModelBenchmark, MirrorTasksShareSnapshotAndSwapModelSides)
{
	GGL::PolicyMatchSnapshot snapshot = GGL::SamplePolicyMatchStartState(
		GGL::PolicyMatchStartType::Kickoff,
		2,
		RocketSim::CAR_CONFIG_OCTANE,
		12345,
		RocketSim::EngineId::V2);
	std::vector<GGL::PolicyMatchTask> tasks = BuildBenchmarkMirrorTasks(
		3,
		7,
		2,
		30,
		70,
		snapshot,
		99,
		500,
		2);

	BENCHMARK_CHECK_EQ(tasks.size(), size_t(2));
	BENCHMARK_CHECK_EQ(tasks[0].blueModel, 3);
	BENCHMARK_CHECK_EQ(tasks[0].orangeModel, 7);
	BENCHMARK_CHECK_EQ(tasks[0].blueArenaKey, 30);
	BENCHMARK_CHECK_EQ(tasks[0].orangeArenaKey, 70);
	BENCHMARK_CHECK_EQ(tasks[1].blueModel, 7);
	BENCHMARK_CHECK_EQ(tasks[1].orangeModel, 3);
	BENCHMARK_CHECK_EQ(tasks[1].blueArenaKey, 70);
	BENCHMARK_CHECK_EQ(tasks[1].orangeArenaKey, 30);
	BENCHMARK_CHECK_EQ(tasks[0].snapshotId, uint64_t(99));
	BENCHMARK_CHECK_EQ(tasks[1].snapshotId, uint64_t(99));
	BENCHMARK_CHECK_EQ(tasks[0].gameId, uint64_t(500));
	BENCHMARK_CHECK_EQ(tasks[1].gameId, uint64_t(501));
	BENCHMARK_CHECK_EQ(tasks[0].mirrorGameCount, 2);
	BENCHMARK_CHECK_EQ(tasks[1].mirrorGameCount, 2);
	AssertSnapshotsEquivalent(tasks[0].snapshot, snapshot);
	AssertSnapshotsEquivalent(tasks[1].snapshot, snapshot);
}

TEST(ModelBenchmark, ApplySnapshot_PreservesTeamMappedStartState)
{
	GGL::PolicyMatchSnapshot expected = {};
	expected.ball.pos = { 125.0f, -640.0f, 720.0f };
	expected.ball.vel = { 850.0f, 1200.0f, -90.0f };
	expected.ball.angVel = { 0.5f, -1.0f, 1.5f };
	expected.cars = {
		{ RocketSim::Team::BLUE, DistinctCarState(-1200.0f, -2400.0f, 17.0f, 23.0f) },
		{ RocketSim::Team::BLUE, DistinctCarState(900.0f, -800.0f, 450.0f, 71.0f) },
		{ RocketSim::Team::ORANGE, DistinctCarState(-700.0f, 1100.0f, 320.0f, 42.0f) },
		{ RocketSim::Team::ORANGE, DistinctCarState(1300.0f, 2600.0f, 17.0f, 88.0f) }
	};

	std::unique_ptr<RocketSim::Arena> arena = GGL::MakePolicyMatchArena(
		2,
		RocketSim::CAR_CONFIG_OCTANE,
		RocketSim::CAR_CONFIG_DOMINUS,
		RocketSim::EngineId::V2);
	GGL::ApplyPolicyMatchSnapshot(arena.get(), expected);

	GGL::PolicyMatchSnapshot actual = GGL::CapturePolicyMatchSnapshot(arena.get());
	AssertSnapshotsEquivalent(actual, expected);
}

TEST(ModelBenchmark, SampleKickoffStartStateWithSeedIsReproducible)
{
	constexpr uint64_t seed = 0xC0FFEE1234ULL;
	GGL::PolicyMatchSnapshot first = GGL::SamplePolicyMatchStartState(
		GGL::PolicyMatchStartType::Kickoff,
		3,
		RocketSim::CAR_CONFIG_OCTANE,
		seed,
		RocketSim::EngineId::V2);
	GGL::PolicyMatchSnapshot second = GGL::SamplePolicyMatchStartState(
		GGL::PolicyMatchStartType::Kickoff,
		3,
		RocketSim::CAR_CONFIG_OCTANE,
		seed,
		RocketSim::EngineId::V2);
	AssertSnapshotsEquivalent(first, second);
}

TEST(ModelBenchmark, SampleRandomStartStateWithSeedIsReproducible)
{
	constexpr uint64_t seed = 0xC0FFEE1234ULL;
	GGL::PolicyMatchSnapshot first = GGL::SamplePolicyMatchStartState(
		GGL::PolicyMatchStartType::RandomState,
		3,
		RocketSim::CAR_CONFIG_OCTANE,
		seed,
		RocketSim::EngineId::V2);
	GGL::PolicyMatchSnapshot second = GGL::SamplePolicyMatchStartState(
		GGL::PolicyMatchStartType::RandomState,
		3,
		RocketSim::CAR_CONFIG_OCTANE,
		seed,
		RocketSim::EngineId::V2);
	AssertSnapshotsEquivalent(first, second);
}

TEST(ModelBenchmark, SampleStartState_WithSeedRestoresGlobalRandomEngine)
{
	std::default_random_engine& engine = RocketSim::Math::GetRandEngine();
	auto assertPreservesEngine = [&](GGL::PolicyMatchStartType startType, uint64_t seed) {
		std::default_random_engine expected = engine;
		(void)GGL::SamplePolicyMatchStartState(
			startType,
			3,
			RocketSim::CAR_CONFIG_OCTANE,
			seed,
			RocketSim::EngineId::V2);
		std::default_random_engine actual = engine;
		for (int draw = 0; draw < 8; draw++)
			BENCHMARK_CHECK_EQ(actual(), expected());
	};

	assertPreservesEngine(GGL::PolicyMatchStartType::Kickoff, 12345);
	assertPreservesEngine(GGL::PolicyMatchStartType::RandomState, 67890);
}

TEST(ModelBenchmark, SampleStartState_DifferentSeedsProduceDifferentRandomStates)
{
	GGL::PolicyMatchSnapshot first = GGL::SamplePolicyMatchStartState(
		GGL::PolicyMatchStartType::RandomState,
		3,
		RocketSim::CAR_CONFIG_OCTANE,
		12345,
		RocketSim::EngineId::V2);
	GGL::PolicyMatchSnapshot second = GGL::SamplePolicyMatchStartState(
		GGL::PolicyMatchStartType::RandomState,
		3,
		RocketSim::CAR_CONFIG_OCTANE,
		67890,
		RocketSim::EngineId::V2);

	BENCHMARK_CHECK_TRUE(SnapshotPositionsDiffer(first, second));
}

TEST(ModelBenchmark, ApplySnapshot_ClearsControlsFromPreviousEpisode)
{
	GGL::PolicyMatchSnapshot snapshot = GGL::SamplePolicyMatchStartState(
		GGL::PolicyMatchStartType::RandomState,
		1,
		RocketSim::CAR_CONFIG_OCTANE,
		12345,
		RocketSim::EngineId::V2);
	std::unique_ptr<RocketSim::Arena> arena = GGL::MakePolicyMatchArena(
		1,
		RocketSim::CAR_CONFIG_OCTANE,
		RocketSim::CAR_CONFIG_OCTANE,
		RocketSim::EngineId::V2);

	for (RocketSim::Car* car : arena->_cars) {
		car->controls.throttle = 1.0f;
		car->controls.steer = -0.75f;
		car->controls.pitch = 0.5f;
		car->controls.yaw = -0.25f;
		car->controls.roll = 1.0f;
		car->controls.jump = true;
		car->controls.boost = true;
		car->controls.handbrake = true;
	}

	GGL::ApplyPolicyMatchSnapshot(arena.get(), snapshot);

	for (RocketSim::Car* car : arena->_cars) {
		AssertNear(car->controls.throttle, 0.0f);
		AssertNear(car->controls.steer, 0.0f);
		AssertNear(car->controls.pitch, 0.0f);
		AssertNear(car->controls.yaw, 0.0f);
		AssertNear(car->controls.roll, 0.0f);
		BENCHMARK_CHECK_FALSE(car->controls.jump);
		BENCHMARK_CHECK_FALSE(car->controls.boost);
		BENCHMARK_CHECK_FALSE(car->controls.handbrake);
	}
}

TEST(ModelBenchmark, CudaOomRecovery_RetriesOnceWithoutGraph)
{
#ifndef RG_CUDA_SUPPORT
	return;
#else
	using namespace GGL::PolicyMatchRunnerInternal;
	CudaOomRecoveryState state = {};
	std::vector<bool> graphFlags;
	int cleanupCalls = 0;
	int attemptCalls = 0;

	const bool recovered = RunWithOneCudaOomRetry(
		true,
		true,
		state,
		[&](bool useCudaGraph) {
			graphFlags.push_back(useCudaGraph);
			if (attemptCalls++ == 0) {
				throw c10::AcceleratorError(
					{ __func__, __FILE__, (uint32_t)__LINE__ },
					(int32_t)cudaErrorMemoryAllocation,
					"synthetic CUDA allocation failure");
			}
		},
		IsCudaOutOfMemory,
		[&]() noexcept { cleanupCalls++; });

	BENCHMARK_CHECK_TRUE(recovered);
	BENCHMARK_CHECK_EQ(attemptCalls, 2);
	BENCHMARK_CHECK_EQ(cleanupCalls, 1);
	BENCHMARK_CHECK_EQ(graphFlags.size(), (size_t)2);
	BENCHMARK_CHECK_TRUE(graphFlags[0]);
	BENCHMARK_CHECK_FALSE(graphFlags[1]);
	BENCHMARK_CHECK_TRUE(state.used);
	BENCHMARK_CHECK_TRUE(state.graphsDisabled);
#endif
}

TEST(ModelBenchmark, CudaOomRecovery_NonMemoryErrorPropagates)
{
	using namespace GGL::PolicyMatchRunnerInternal;
	CudaOomRecoveryState state = {};
	int attemptCalls = 0;
	int cleanupCalls = 0;
	bool threw = false;

	try {
		RunWithOneCudaOomRetry(
			true,
			true,
			state,
			[&](bool) {
				attemptCalls++;
				TORCH_CHECK(false, "synthetic non-memory failure");
			},
			IsCudaOutOfMemory,
			[&]() noexcept { cleanupCalls++; });
	} catch (const c10::Error&) {
		threw = true;
	}

	BENCHMARK_CHECK_TRUE(threw);
	BENCHMARK_CHECK_EQ(attemptCalls, 1);
	BENCHMARK_CHECK_EQ(cleanupCalls, 0);
	BENCHMARK_CHECK_FALSE(state.used);
	BENCHMARK_CHECK_FALSE(state.graphsDisabled);
}

TEST(ModelBenchmark, CudaOomRecovery_SecondOomPropagates)
{
#ifndef RG_CUDA_SUPPORT
	return;
#else
	using namespace GGL::PolicyMatchRunnerInternal;
	CudaOomRecoveryState state = {};
	std::vector<bool> graphFlags;
	int cleanupCalls = 0;
	bool threw = false;

	try {
		RunWithOneCudaOomRetry(
			true,
			true,
			state,
			[&](bool useCudaGraph) {
				graphFlags.push_back(useCudaGraph);
				C10_THROW_ERROR(OutOfMemoryError, "synthetic repeated OOM");
			},
			IsCudaOutOfMemory,
			[&]() noexcept { cleanupCalls++; });
	} catch (const c10::OutOfMemoryError&) {
		threw = true;
	}

	BENCHMARK_CHECK_TRUE(threw);
	BENCHMARK_CHECK_EQ(graphFlags.size(), (size_t)2);
	BENCHMARK_CHECK_TRUE(graphFlags[0]);
	BENCHMARK_CHECK_FALSE(graphFlags[1]);
	BENCHMARK_CHECK_EQ(cleanupCalls, 1);
	BENCHMARK_CHECK_TRUE(state.used);
	BENCHMARK_CHECK_TRUE(state.graphsDisabled);
#endif
}

TEST(ModelBenchmark, CudaOomRecovery_CpuOomDoesNotRetry)
{
	using namespace GGL::PolicyMatchRunnerInternal;
	CudaOomRecoveryState state = {};
	int attemptCalls = 0;
	int cleanupCalls = 0;
	bool threw = false;

	try {
		RunWithOneCudaOomRetry(
			false,
			true,
			state,
			[&](bool) {
				attemptCalls++;
				C10_THROW_ERROR(OutOfMemoryError, "synthetic CPU OOM");
			},
			IsCudaOutOfMemory,
			[&]() noexcept { cleanupCalls++; });
	} catch (const c10::OutOfMemoryError&) {
		threw = true;
	}

	BENCHMARK_CHECK_TRUE(threw);
	BENCHMARK_CHECK_EQ(attemptCalls, 1);
	BENCHMARK_CHECK_EQ(cleanupCalls, 0);
	BENCHMARK_CHECK_FALSE(state.used);
	BENCHMARK_CHECK_FALSE(state.graphsDisabled);
}

TEST(ModelBenchmark, EnvironmentIdentityUsesModelKeys)
{
	ModelBenchmarkConfig config;
	config.models.resize(2);
	config.models[0].checkpointPath = "model-A";
	config.models[1].checkpointPath = "model-B";
	config.modelPredictionEngines = {2, 0};
	config.startingStateDirectory = "frozen starts";
	ModelBenchmarkResult result;
	result.rocketSimVersions = {3};

	const std::map<int, std::string> predictorRevisions = {{2, "native-v2-pin"}};
	RecordModelBenchmarkEnvironment(result, config, "RocketSim-v3", "physics-pin", "insertion-order", predictorRevisions);
	const std::string identity = result.evaluationIdentity;
	BENCHMARK_CHECK_TRUE(nlohmann::json::parse(identity).at("noBallRot").get<bool>());

	const bool noBallRot = false;
	RecordModelBenchmarkEnvironment(result, config, "RocketSim-v3", "physics-pin", "insertion-order", predictorRevisions, noBallRot);
	BENCHMARK_CHECK_FALSE(nlohmann::json::parse(result.evaluationIdentity).at("noBallRot").get<bool>());
	BENCHMARK_CHECK_FALSE(result.evaluationIdentity == identity);

	std::swap(config.models[0], config.models[1]);
	config.modelPredictionEngines = {0, 2};
	RecordModelBenchmarkEnvironment(result, config, "RocketSim-v3", "physics-pin", "insertion-order", predictorRevisions);
	BENCHMARK_CHECK_EQ(result.evaluationIdentity, identity);

	RecordModelBenchmarkEnvironment(result, config, "RocketSim-v3", "other-physics-pin", "insertion-order", predictorRevisions);
	BENCHMARK_CHECK_FALSE(result.evaluationIdentity == identity);
}

TEST(ModelBenchmark, CudaBatchesKeepActionsWithTheirModelsAcrossSlotReuse)
{
	if (!torch::cuda::is_available())
		return;

	torch::NoGradGuard noGrad;
	std::array<BenchmarkModelSetOwner, 3> owners;
	GGL::PartialModelConfig policy;
	policy.layerSizes = { 4 };
	policy.addLayerNorm = false;
	const bool makeCritic = false;
	for (int model = 0; model < 3; ++model) {
		GGL::ModelManager::MakeModels(makeCritic, 1, 4, {}, policy, {}, torch::kCUDA, owners[model].models);
		for (auto* network : owners[model].models)
			for (torch::Tensor parameter : network->parameters())
				parameter.zero_();

		torch::Tensor bias = owners[model].models["policy"]->parameters().back();
		bias.select(0, model).fill_(10.f);

		// The highest logit must remain masked.
		bias.select(0, 3).fill_(100.f);
	}

	struct CheckAction : RLGC::ActionParser {
		int expected;
		int& count;

		CheckAction(int expected, int& count) : expected(expected), count(count) {}

		int GetActionAmount() override { return 4; }

		bool UsesActionMask() override { return true; }

		bool SuppressPreferred(const RLGC::Player& player, const RLGC::GameState& state) const {
			return (state.lastTickCount / 4 + player.carId) % 2 != 0;
		}

		void GetActionMaskInto(const RLGC::Player& player, const RLGC::GameState& state, std::vector<uint8_t>& mask) override {
			mask = { true, true, true, false };
			mask[expected] = !SuppressPreferred(player, state);
		}

		RLGC::Action ParseAction(int action, const RLGC::Player& player, const RLGC::GameState& state) override {
			// Vary actions across rows and ticks to catch stale or reordered readbacks.
			const int wanted = SuppressPreferred(player, state) ? (expected == 0 ? 1 : 0) : expected;
			BENCHMARK_CHECK_EQ(action, wanted);
			++count;
			return {};
		}
	};

	for (bool graph : { false, true }) {
		for (bool variableTiming : { false, true }) {
			std::array<int, 3> actionCounts = {};
			GGL::PolicyMatchRunnerParams params;
			params.device = torch::kCUDA;
			params.numSlots = 4;
			params.independentSlotTiming = variableTiming;
			params.tickSkip = 4;
			params.actionDelay = 0;
			params.noGoalSteps = 24;
			params.noGoalTicks = 96;
			for (int model = 0; model < 3; ++model) {
				GGL::PolicyMatchRuntime runtime;
				runtime.models = &owners[model].models;
				runtime.obsSize = 1;
				runtime.numActions = 4;
				runtime.usesActionMask = true;
				runtime.useCudaGraph = graph;
				runtime.tickSkip = variableTiming ? model + 2 : 4;
				runtime.runtimeGroup = model;
				runtime.makeRuntimeComponents = [&, model]() {
					return GGL::PolicyMatchRuntimeComponents{
						std::make_unique<BenchmarkTinyObsBuilder>(),
						std::make_unique<CheckAction>(model, actionCounts[model]) };
				};
				params.runtimes.push_back(runtime);
			}

			std::vector<GGL::PolicyMatchTask> tasks(9);
			for (int i = 0; i < (int)tasks.size(); ++i) {
				GGL::PolicyMatchTask& task = tasks[i];
				task.blueModel = i % 3;
				task.orangeModel = (i + 1) % 3;
				task.teamSize = i % 3 + 1;
				task.gameId = i;
				task.snapshot = GGL::SamplePolicyMatchStartState(
					GGL::PolicyMatchStartType::Kickoff, task.teamSize, RocketSim::CAR_CONFIG_OCTANE, 12345 + i, RocketSim::EngineId::V2);
			}

			size_t scheduled = 0;
			int finished = 0;
			params.makeArena = [](const GGL::PolicyMatchTask& task) {
				return GGL::PolicyMatchArena{ GGL::MakePolicyMatchArena(
					task.teamSize, RocketSim::CAR_CONFIG_OCTANE, RocketSim::CAR_CONFIG_OCTANE, RocketSim::EngineId::V2) };
			};
			params.fillPendingTasks = [&](std::vector<GGL::PolicyMatchTask>& pending, int, int) {
				while (scheduled < tasks.size())
					pending.push_back(tasks[scheduled++]);
			};
			params.isSchedulingComplete = [&]() { return scheduled == tasks.size(); };
			params.onFinished = [&](const GGL::PolicyMatchFinish&) {
				++finished;
				return false;
			};

			const GGL::PolicyMatchRunnerResult result = variableTiming ? GGL::RunPolicyMatchVariableTimingSlots(params)
				: GGL::RunPolicyMatchSlots(params);
			BENCHMARK_CHECK_FALSE(result.cancelled);
			BENCHMARK_CHECK_EQ(finished, int(tasks.size()));
			for (int count : actionCounts)
				BENCHMARK_CHECK_TRUE(count > 0);

			if (graph)
				BENCHMARK_CHECK_TRUE(result.timing.cudaGraphReplays > 0);
		}
	}
}

TEST(ModelBenchmark, CudaValidRowsRetainsCommonShrinkingBatchSizes)
{
	if (!torch::cuda::is_available())
		return;

	torch::NoGradGuard noGrad;
	BenchmarkModelSetOwner owner;
	GGL::PartialModelConfig policy;
	policy.layerSizes = { 4 };
	policy.addLayerNorm = false;
	const bool makeCritic = false;
	GGL::ModelManager::MakeModels(makeCritic, 1, 4, {}, policy, {}, torch::kCUDA, owner.models);
	for (auto* network : owner.models)
		for (torch::Tensor parameter : network->parameters())
			parameter.zero_();

	owner.models["policy"]->parameters().back().select(0, 2).fill_(10.f);
	GGL::ModelManager::ClearCudaGraphCache();
	GGL::ModelManager::ResetCudaGraphStats();

	const bool deterministic = true;
	const float temperature = 1.f;
	const bool halfPrecision = false;
	const float minActionProbability = 0.f;
	const bool useCudaGraph = true;
	const bool persistentStorage = false;
	for (int repeat = 0; repeat < 2; ++repeat) {
		for (int rows = 8192; rows >= 16; rows /= 2) {
			torch::Tensor obs = torch::zeros({ rows, 1 }, torch::TensorOptions().device(torch::kCUDA));
			torch::Tensor masks = torch::ones({ rows, 4 }, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kUInt8));
			torch::Tensor actions;
			GGL::ModelManager::InferActionsFromModelsValidRows(
				owner.models,
				obs,
				masks,
				deterministic,
				temperature,
				halfPrecision,
				&actions,
				nullptr,
				minActionProbability,
				useCudaGraph,
				GGL::CudaGraphBatchPolicy::PowerOfTwo,
				persistentStorage,
				rows);
			BENCHMARK_CHECK_EQ(actions.numel(), rows);
			BENCHMARK_CHECK_TRUE(actions.eq(2).all().item<bool>());
		}
	}

	const GGL::CudaGraphStats stats = GGL::ModelManager::GetCudaGraphStats();
	BENCHMARK_CHECK_EQ(stats.captures, 10);
	BENCHMARK_CHECK_EQ(stats.replays, 20);
	BENCHMARK_CHECK_EQ(stats.variantLimitFallbacks, 0);
	GGL::ModelManager::ClearCudaGraphCache();
}

TEST(ModelBenchmark, LiveDeltasPreserveSnapshotsAndRejectMissingGames)
{
	ModelBenchmarkLiveGames history;
	auto update = [](int64_t offset, uint64_t id) {
		ModelBenchmarkProgress progress;
		std::shared_ptr<ModelBenchmarkResult> result = std::make_shared<ModelBenchmarkResult>();
		result->games.emplace_back();
		result->games.back().gameId = id;
		progress.liveGamesOffset = offset;
		progress.liveResult = result;
		return progress;
	};

	ModelBenchmarkProgress first = update(0, 101);
	history.Apply(first);
	ModelBenchmarkProgress second = update(1, 102);
	history.Apply(second);
	BENCHMARK_CHECK_EQ(first.liveResult->games.size(), size_t(1));
	BENCHMARK_CHECK_EQ(second.liveResult->games.size(), size_t(2));
	BENCHMARK_CHECK_EQ(second.liveResult->games[0].gameId, uint64_t(101));
	BENCHMARK_CHECK_EQ(second.liveResult->games[1].gameId, uint64_t(102));
	BENCHMARK_CHECK_EQ(second.liveGamesOffset, int64_t(-1));

	for (int offset : { 1, 3 }) {
		ModelBenchmarkProgress broken = update(offset, 103);
		bool rejected = false;
		try {
			history.Apply(broken);
		} catch (const std::runtime_error&) {
			rejected = true;
		}
		BENCHMARK_CHECK_TRUE(rejected);
	}

	history.Reset();
	ModelBenchmarkProgress snapshot = update(-1, 201);
	history.Apply(snapshot);
	ModelBenchmarkProgress next = update(1, 202);
	history.Apply(next);
	BENCHMARK_CHECK_EQ(next.liveResult->games.size(), size_t(2));
	BENCHMARK_CHECK_EQ(next.liveResult->games[0].gameId, uint64_t(201));
}

TEST(ModelBenchmark, IndependentTimingPreservesPerTickPhysicsAndDelays)
{
	torch::NoGradGuard noGrad;
	BenchmarkModelSetOwner owner;
	GGL::PartialModelConfig policy;
	policy.layerSizes = { 4 };
	policy.addLayerNorm = false;
	const bool makeCritic = false;
	GGL::ModelManager::MakeModels(makeCritic, 1, 1, {}, policy, {}, torch::kCPU, owner.models);

	struct DrivingAction : BenchmarkTinyActionParser {
		RLGC::Action ParseAction(int, const RLGC::Player&, const RLGC::GameState&) override {
			RLGC::Action action;
			action.throttle = 1.f;
			action.steer = 0.2f;
			return action;
		}
	};

	for (int maxStep : {1, 0}) {
		using Frames = std::array<std::vector<float>, 6>;
		std::array<Frames, 2> traces;
		std::array<int, 2> callbacks = {};
		for (int independent = 0; independent < 2; ++independent) {
			GGL::PolicyMatchRunnerParams params;
			params.device = torch::kCPU;
			params.numSlots = 3;
			params.noGoalTicks = 96;
			params.independentSlotTiming = independent != 0;
			const int skips[] = { 8, 6, 4 };
			const int delays[] = { 7, 0, 2 };
			for (int model = 0; model < 3; ++model) {
				GGL::PolicyMatchRuntime runtime;
				runtime.models = &owner.models;
				runtime.obsSize = 1;
				runtime.numActions = 1;
				runtime.tickSkip = skips[model];
				runtime.actionDelay = delays[model];
				runtime.runtimeGroup = model;

				runtime.makeRuntimeComponents = []() {
					return GGL::PolicyMatchRuntimeComponents{
						std::make_unique<BenchmarkTinyObsBuilder>(), std::make_unique<DrivingAction>() };
				};
				params.runtimes.push_back(runtime);
			}

			std::vector<GGL::PolicyMatchTask> tasks(6);
			for (int i = 0; i < 6; ++i) {
				GGL::PolicyMatchTask& task = tasks[i];
				task.gameId = i;
				task.snapshotId = i + 1;
				task.blueModel = i % 3;
				task.orangeModel = (i + 1) % 3;
				task.snapshot = GGL::SamplePolicyMatchStartState(
					GGL::PolicyMatchStartType::Kickoff, 1, RocketSim::CAR_CONFIG_OCTANE, 7919 + i, RocketSim::EngineId::V2);
			}

			size_t scheduled = 0;
			int finished = 0;
			std::array<int, 6> serialTicks = {};
			std::array<int, 6> parallelTicks = {};
			params.makeArena = [](const GGL::PolicyMatchTask&) {
				return GGL::PolicyMatchArena{
					GGL::MakePolicyMatchArena(1, RocketSim::CAR_CONFIG_OCTANE, RocketSim::CAR_CONFIG_OCTANE, RocketSim::EngineId::V2) };
			};
			params.fillPendingTasks = [&](std::vector<GGL::PolicyMatchTask>& pending, int, int) {
				while (scheduled < tasks.size())
					pending.push_back(tasks[scheduled++]);
			};
			params.isSchedulingComplete = [&]() { return scheduled == tasks.size(); };
			params.maxPhysicsStepTicks = maxStep;

			auto recordFrame = [&](const GGL::PolicyMatchStep& step) {
				std::vector<float>& frames = traces[independent][step.task.gameId];
				frames.push_back((float)step.steps);

				// RocketSim v2 stores cars in pointer order, which can differ between arenas,
				// so compare the same car across runs instead of comparing vector positions.
				std::vector<const RLGC::Player*> players;
				for (const RLGC::Player& player : step.state.players)
					players.push_back(&player);

				std::sort(players.begin(), players.end(), [](const RLGC::Player* a, const RLGC::Player* b) {
					return a->carId < b->carId;
				});
				for (const RLGC::Player* player : players) {
					for (float x : {player->pos.x, player->pos.y, player->pos.z, player->vel.x, player->vel.y, player->vel.z})
						frames.push_back(x);
				}
			};
			params.onStepParallel = [&](const GGL::PolicyMatchStep& step) {
				BENCHMARK_CHECK_TRUE(step.slotIndex >= 0 && step.slotIndex < 3);
				parallelTicks[step.task.gameId] += (int)std::lround(step.deltaSeconds * 120);
			};
			params.onStep = [&](const GGL::PolicyMatchStep& step) {
				if (maxStep == 1 || step.steps == 96)
					recordFrame(step);

				callbacks[independent]++;
				serialTicks[step.task.gameId] += (int)std::lround(step.deltaSeconds * 120);
			};
			params.onFinished = [&](const GGL::PolicyMatchFinish&) {
				++finished;
				return false;
			};

			const GGL::PolicyMatchRunnerResult result = GGL::RunPolicyMatchVariableTimingSlots(params);
			BENCHMARK_CHECK_FALSE(result.cancelled);
			BENCHMARK_CHECK_EQ(finished, 6);
			for (int i = 0; i < 6; ++i) {
				BENCHMARK_CHECK_EQ(serialTicks[i], 96);
				BENCHMARK_CHECK_EQ(parallelTicks[i], 96);
			}
		}

		for (int i = 0; i < 6; ++i) {
			BENCHMARK_CHECK_FALSE(traces[0][i].empty());
			BENCHMARK_CHECK_EQ(traces[0][i].size(), traces[1][i].size());
			for (size_t j = 0; j < traces[0][i].size(); ++j)
				AssertNear(traces[0][i][j], traces[1][i][j]);
		}

		if (maxStep == 0)
			BENCHMARK_CHECK_TRUE(callbacks[1] < callbacks[0]);
		else
			BENCHMARK_CHECK_EQ(callbacks[1], callbacks[0]);
	}
}

TEST(ModelBenchmark, LiveDeltasWaitForCompletedMirrors)
{
	ModelBenchmarkLiveGameDeltas publisher;
	ModelBenchmarkLiveGames receiver;
	std::vector<ModelBenchmarkGameMetadata> games(1);
	games[0].gameId = 1;
	auto emit = [&]() {
		std::shared_ptr<ModelBenchmarkResult> result = std::make_shared<ModelBenchmarkResult>();
		ModelBenchmarkProgress progress;
		progress.liveGamesOffset = publisher.Append(games, *result);
		progress.liveResult = result;
		receiver.Apply(progress);
		return progress;
	};

	BENCHMARK_CHECK_TRUE(emit().liveResult->games.empty());

	games.emplace_back();
	games[1].gameId = 2;
	for (ModelBenchmarkGameMetadata& game : games) {
		game.mirrorCompleted = true;
		game.includedInRating = true;
	}

	const ModelBenchmarkProgress completed = emit();
	BENCHMARK_CHECK_EQ(completed.liveResult->games.size(), size_t(2));
	for (const ModelBenchmarkGameMetadata& game : completed.liveResult->games)
		BENCHMARK_CHECK_TRUE(game.includedInRating);

	BENCHMARK_CHECK_EQ(emit().liveResult->games.size(), size_t(2));
}

TEST(ModelBenchmark, ParallelStatsFlushOnCancellationWithoutDoubleCounting)
{
	ModelBenchmarkConfig config = {};
	config.runUntilMaxGames = true;
	config.collectOptionalStats = true;
	ModelBenchmarkCancelCallback cancelCallback = {};
	std::vector<LoadedModel> loadedModels(2);
	loadedModels[0].name = "A";
	loadedModels[1].name = "B";
	loadedModels[0].carConfig = RocketSim::CAR_CONFIG_OCTANE;
	loadedModels[1].carConfig = RocketSim::CAR_CONFIG_OCTANE;

	const std::vector<int> modes = { 1, 2 };
	const std::vector<std::pair<int, int>> matchups = { { 0, 1 } };
	const std::vector<std::string> modelNames = { "A", "B" };
	BenchmarkRunState state = {};
	state.stats.resize(2);
	state.targetGamesPerPairMode = 1;
	for (int mode : modes) {
		state.gamesByMode[mode] = 0;
		state.scheduledGamesByMode[mode] = 0;
		state.stepsByMode[mode] = 0;
		state.scheduledByPairMode[{ 0, 1, mode }] = 0;
	}

	SlotBenchmarkParams params = {
		config,
		cancelCallback,
		loadedModels,
		modes,
		matchups,
		modelNames,
		ModelBenchmarkStrategy::SameRuntimeSlot,
		"Benchmarking",
		"test slot path",
		2,
		2,
		1,
		0,
		1,
		1,
		false,
		12345,
		[](const std::string&, const std::string&, const std::string&) {},
		[](int) { return false; },
		[]() { return false; }
	};
	params.runPolicyMatchRunner = [&](const GGL::PolicyMatchRunnerParams& runnerParams) {
		GGL::PolicyMatchTask task = {};
		task.blueModel = 0;
		task.orangeModel = 1;
		task.teamSize = 2;

		RLGC::GameState previous = {};
		previous.players.resize(2);
		previous.players[0].carId = 31;
		previous.players[0].team = RocketSim::Team::BLUE;
		previous.players[1].carId = 32;
		previous.players[1].team = RocketSim::Team::ORANGE;

		RLGC::GameState current = previous;
		current.players[0].eventState.goal = true;
		current.players[0].isOnGround = false;
		current.players[0].boost = 10.0f;
		GGL::PolicyMatchStep step{ task, current, previous, 1, 300.0f, 0 };
		BENCHMARK_CHECK_TRUE(bool(runnerParams.onStepParallel));
		runnerParams.onStepParallel(step);
		runnerParams.onStep(step);
		GGL::PolicyMatchRunnerResult result;
		result.cancelled = true;
		return result;
	};

	RunSlotBenchmark(params, state);
	BENCHMARK_CHECK_TRUE(state.cancelled);

	const ModelBenchmarkOptionalStats overall = FinalizeStats(state.stats[0]);
	const ModelBenchmarkOptionalStats mode1v1 = FinalizeStats(state.statsByMode.at(1)[0]);
	const ModelBenchmarkOptionalStats mode2v2 = FinalizeStats(state.statsByMode.at(2)[0]);
	AssertNear((float)overall.equivalentGames, 1.0f);
	AssertNear((float)overall.goals, 1.0f);
	AssertNear((float)mode1v1.equivalentGames, 0.0f);
	AssertNear((float)mode2v2.equivalentGames, 1.0f);
	AssertNear((float)mode2v2.goals, 1.0f);
}

TEST(ModelBenchmark, CudaActionGraphPreservesMasksTiesAndValidRows)
{
	if (!torch::cuda::is_available())
		return;

	torch::NoGradGuard noGrad;
	BenchmarkModelSetOwner owner;
	GGL::PartialModelConfig policy;
	policy.layerSizes = { 4 };
	policy.addLayerNorm = false;
	const bool makeCritic = false;
	GGL::ModelManager::MakeModels(makeCritic, 1, 4, {}, policy, {}, torch::kCUDA, owner.models);
	for (auto* network : owner.models)
		for (torch::Tensor parameter : network->parameters())
			parameter.zero_();

	GGL::ModelManager::ClearCudaGraphCache();
	torch::Tensor obs = torch::zeros({16, 1}, torch::TensorOptions().device(torch::kCUDA));
	torch::Tensor masks = torch::ones({16, 4}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kUInt8));

	const bool deterministic = true;
	const float temperature = 1.f;
	const bool halfPrecision = false;
	const float minActionProbability = 0.f;
	const bool persistentStorage = true;
	const bool withoutCudaGraph = false;
	const bool withCudaGraph = true;
	for (int repeat = 0; repeat < 3; ++repeat) {
		masks.fill_(1);
		masks.narrow(1, 0, repeat + 1).zero_();
		for (int valid : {1, 7, 16}) {
			torch::Tensor eager;
			torch::Tensor captured;
			GGL::ModelManager::InferActionsFromModelsValidRows(
				owner.models,
				obs,
				masks,
				deterministic,
				temperature,
				halfPrecision,
				&eager,
				nullptr,
				minActionProbability,
				withoutCudaGraph,
				GGL::CudaGraphBatchPolicy::Exact,
				persistentStorage,
				valid);
			GGL::ModelManager::InferActionsFromModelsValidRows(
				owner.models,
				obs,
				masks,
				deterministic,
				temperature,
				halfPrecision,
				&captured,
				nullptr,
				minActionProbability,
				withCudaGraph,
				GGL::CudaGraphBatchPolicy::Exact,
				persistentStorage,
				valid);

			BENCHMARK_CHECK_EQ(captured.numel(), valid);
			BENCHMARK_CHECK_TRUE(captured.eq(eager).all().item<bool>());
			BENCHMARK_CHECK_TRUE(captured.eq(repeat + 1).all().item<bool>());
		}
	}

	GGL::ModelManager::ClearCudaGraphCache();
}

TEST(ModelBenchmark, SampleStartStateWithSeedIsReproducible)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	constexpr uint64_t seed = 0xC0FFEE1234ULL;

	for (GGL::PolicyMatchStartType startType : { GGL::PolicyMatchStartType::Kickoff, GGL::PolicyMatchStartType::RandomState }) {
		GGL::PolicyMatchSnapshot first = GGL::SamplePolicyMatchStartState(
			startType, 3, RocketSim::CAR_CONFIG_OCTANE, seed, RocketSim::EngineId::V2);
		GGL::PolicyMatchSnapshot second = GGL::SamplePolicyMatchStartState(
			startType, 3, RocketSim::CAR_CONFIG_OCTANE, seed, RocketSim::EngineId::V2);
		AssertSnapshotsEquivalent(first, second);
	}
}

TEST(ModelBenchmark, SchedulerHonorsUnequalBudgetsAndAdvancesSnapshots)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	ModelBenchmarkConfig config = {};
	config.runUntilMaxGames = true;
	ModelBenchmarkCancelCallback cancelCallback = {};
	std::vector<LoadedModel> loadedModels(3);
	for (int i = 0; i < 3; i++) {
		loadedModels[i].name = std::string(1, (char)('A' + i));
		loadedModels[i].carConfig = RocketSim::CAR_CONFIG_OCTANE;
	}

	const std::vector<int> modes = { 1, 2 };
	const std::vector<std::pair<int, int>> matchups = { { 0, 1 }, { 0, 2 }, { 1, 2 } };
	const std::vector<std::string> modelNames = { "A", "B", "C" };

	BenchmarkRunState state = {};
	state.stats.resize(3);
	state.targetByPairMode = {{{0, 1, 1}, 3}, {{0, 2, 2}, 1}};
	state.nextSnapshotId = 101;

	for (int mode : modes) {
		state.gamesByMode[mode] = 0;
		state.scheduledGamesByMode[mode] = 0;
		state.stepsByMode[mode] = 0;
	}
	state.scheduledByPairMode[{ 0, 1, 1 }] = 0;
	state.scheduledByPairMode[{ 0, 2, 2 }] = 0;

	SlotBenchmarkParams params = {
		config,
		cancelCallback,
		loadedModels,
		modes,
		matchups,
		modelNames,
		ModelBenchmarkStrategy::Heterogeneous,
		"Benchmarking",
		"test slot path",
		3,
		8,
		1,
		0,
		1,
		1,
		false,
		12345,
		[](const std::string&, const std::string&, const std::string&) {},
		[](int) { return false; },
		[]() { return false; }
	};

	auto checkRunnerParams = [&](const GGL::PolicyMatchRunnerParams& runnerParams) {
		std::vector<GGL::PolicyMatchTask> pendingTasks;
		runnerParams.fillPendingTasks(pendingTasks, 0, 8);
		BENCHMARK_CHECK_EQ(pendingTasks.size(), size_t(4));
		runnerParams.fillPendingTasks(pendingTasks, 0, 8);
		BENCHMARK_CHECK_EQ(pendingTasks.size(), size_t(4));
		for (const GGL::PolicyMatchTask& task : pendingTasks)
			BENCHMARK_CHECK_GT(task.snapshotId, 100u);

		auto taskCount = [&](int modelA, int modelB, int mode) {
			return (int)std::count_if(pendingTasks.begin(), pendingTasks.end(), [&](const GGL::PolicyMatchTask& task) {
				return task.userValues[0] == modelA && task.userValues[1] == modelB && task.teamSize == mode;
			});
		};

		BENCHMARK_CHECK_EQ(taskCount(0, 1, 1), 3);
		BENCHMARK_CHECK_EQ(taskCount(0, 2, 2), 1);
		BENCHMARK_CHECK_EQ(taskCount(0, 1, 2), 0);
		BENCHMARK_CHECK_EQ(taskCount(0, 2, 1), 0);
		BENCHMARK_CHECK_EQ(taskCount(1, 2, 1), 0);
		BENCHMARK_CHECK_EQ(taskCount(1, 2, 2), 0);
	};
	params.runPolicyMatchRunner = [&](const GGL::PolicyMatchRunnerParams& runnerParams) {
		checkRunnerParams(runnerParams);
		return GGL::PolicyMatchRunnerResult{};
	};

	RunSlotBenchmark(params, state);

	BENCHMARK_CHECK_EQ(state.scheduledGamesByMode[1], 3);
	BENCHMARK_CHECK_EQ(state.scheduledGamesByMode[2], 1);
	BENCHMARK_CHECK_EQ(state.scheduledGames, 4);
}

TEST(ModelBenchmark, SelectStrategyPicksPathFromRuntimeSignature)
{
	struct Case {
		const char* label;
		void (*mutate)(ModelBenchmarkRuntimeSignature&);
		ModelBenchmarkStrategy expected;
	};

	const Case cases[] = {
		{ "same runtime", [](ModelBenchmarkRuntimeSignature&) {}, ModelBenchmarkStrategy::SameRuntimeSlot },
		{ "different agent", [](ModelBenchmarkRuntimeSignature& b) { b.agentName = "other-agent"; }, ModelBenchmarkStrategy::Heterogeneous },
		{ "different obs size", [](ModelBenchmarkRuntimeSignature& b) { b.obsSize += 16; }, ModelBenchmarkStrategy::Heterogeneous },
		{ "different action count", [](ModelBenchmarkRuntimeSignature& b) { b.numActions += 4; }, ModelBenchmarkStrategy::Heterogeneous },
		{ "different hitbox", [](ModelBenchmarkRuntimeSignature& b) { b.carConfigSignature = "dominus"; }, ModelBenchmarkStrategy::Heterogeneous },
		{ "different tick skip", [](ModelBenchmarkRuntimeSignature& b) { b.tickSkip++; }, ModelBenchmarkStrategy::Heterogeneous },
		{ "different action delay", [](ModelBenchmarkRuntimeSignature& b) { b.actionDelay++; }, ModelBenchmarkStrategy::Heterogeneous },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);
		ModelBenchmarkRuntimeSignature a = BaseRuntimeSignature();
		ModelBenchmarkRuntimeSignature b = BaseRuntimeSignature();
		testCase.mutate(b);

		BENCHMARK_CHECK_EQ(SelectModelBenchmarkStrategy({ a, b }), testCase.expected);
	}

	BENCHMARK_CHECK_EQ(ModelBenchmarkStrategyName(ModelBenchmarkStrategy::SameRuntimeSlot), std::string("SameRuntimeSlot"));
	BENCHMARK_CHECK_EQ(ModelBenchmarkStrategyName(ModelBenchmarkStrategy::Heterogeneous), std::string("Heterogeneous"));
}

TEST(ModelBenchmark, PortableStartingStatesRoundTripAndRejectWrongIdentity) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	ModelBenchmarkConfig config;
	config.seed = 42;
	config.snapshotOffset = 100;
	config.usePairBudgets = true;
	config.startingStateDirectory = RocketForgeTests::TestTempDir("starting-states");
	config.pairBudgets = {{0, 1, 1, 3}, {0, 1, 2, 2}, {0, 1, 3, 1}};
	PrepareBenchmarkStartingStates(config);

	std::map<std::filesystem::path, std::string> originals;
	for (const std::filesystem::directory_entry& file : std::filesystem::directory_iterator(config.startingStateDirectory)) {
		std::ifstream input(file.path());
		const std::string bytes((std::istreambuf_iterator<char>(input)), {});
		const nlohmann::json json = nlohmann::json::parse(bytes);
		const GGL::PolicyMatchSnapshot loaded =
			ReadBenchmarkStartingState(config.startingStateDirectory, json.at("mode"), json.at("snapshotSeed"));
		EXPECT_EQ(loaded.cars.size(), size_t(json.at("mode").get<int>() * 2));
		originals[file.path()] = bytes;
	}

	ASSERT_EQ(originals.size(), 4u);
	PrepareBenchmarkStartingStates(config);
	for (const auto& [path, bytes] : originals) {
		std::ifstream input(path);
		EXPECT_EQ(std::string((std::istreambuf_iterator<char>(input)), {}), bytes);
	}

	nlohmann::json malformed = nlohmann::json::parse(originals.begin()->second);
	malformed["schema"] = 99;
	{
		std::ofstream out(originals.begin()->first);
		out << malformed;
	}
	EXPECT_THROW(PrepareBenchmarkStartingStates(config), std::runtime_error);
}

TEST(ModelBenchmark, SimulatorSelectionsRoundTripAndLegacyDefaultsToV2) {
    ModelBenchmarkSettings benchmark;
    EXPECT_EQ(benchmark.engineVersion, 2);
    benchmark.engineVersion = 3;
    benchmark.includeOtherSimulatorResults = true;
    nlohmann::json saved;
    GGL::JsonSave(benchmark, saved);
    ModelBenchmarkSettings restoredBenchmark;
    EXPECT_TRUE(GGL::JsonLoad(saved, restoredBenchmark).empty());
    EXPECT_EQ(restoredBenchmark.engineVersion, 3);
    EXPECT_TRUE(restoredBenchmark.includeOtherSimulatorResults);

    EXPECT_EQ(ModelBenchmarkConfig{}.engine, RocketSim::EngineId::V2);
}

TEST(ModelBenchmark, EnvironmentDescribesTheConfiguredMatchEngine) {
    for (RocketSim::EngineId engine : { RocketSim::EngineId::V2, RocketSim::EngineId::V3 }) {
        SCOPED_TRACE(RocketSim::GetEngineInfo(engine).displayName);
        ModelBenchmarkConfig config;
        config.engine = engine;

        const ModelBenchmarkResult result = DescribeModelBenchmarkEnvironment(config);
        EXPECT_EQ(result.rocketSimVersions, std::set<int>{ static_cast<int>(engine) });
        EXPECT_EQ(result.engineFamily, engine == RocketSim::EngineId::V2 ? "RocketSim-v2" : "RocketSim-v3");
        EXPECT_EQ(result.engineRevision, RocketSim::GetEngineInfo(engine).revision);
        EXPECT_EQ(result.engineCarOrder, engine == RocketSim::EngineId::V2 ? "legacy-pointer-order" : "insertion-order");
        EXPECT_TRUE(result.integrationRevision.empty());
        EXPECT_FALSE(nlohmann::json::parse(result.evaluationIdentity).contains("integrationSourceRevision"));
    }
}

namespace {
    /** Registers an agent that trains on v2 but predicts the ball with v3, and two benchmark models of it. **/
    ModelBenchmarkConfig MakeCrossEnginePredictionConfig(const std::filesystem::path& root) {
        Agent definition;
        definition.agentName = "BenchmarkCrossEnginePredictionFixture";
        definition.trainingEngineVersion = 2;
        definition.ballPredictionEngineVersion = 3;
        const auto setup = RocketForge::CaptureAgentSetupSnapshot(definition);
        AgentRegistry::Register(definition.agentName, [setup] {
            auto agent = std::make_unique<Agent>();
            RocketForge::ApplyAgentSetupSnapshot(*agent, setup);
            return agent;
        });

        ModelBenchmarkConfig config;
        config.engine = RocketSim::EngineId::V2;
        for (int index = 0; index < 2; ++index) {
            ModelBenchmarkModel model;
            model.agentName = definition.agentName;
            model.displayName = "Model " + std::to_string(index);
            model.versionId = "v1.0" + std::to_string(index);
            model.checkpointPath = root / ("model-" + std::to_string(index));
            model.agentSetupSource = ModelBenchmarkAgentSetupSource::CurrentAgent;
            std::filesystem::create_directories(model.checkpointPath);
            config.models.push_back(model);
        }
        return config;
    }
}

TEST(ModelBenchmark, EnvironmentRecordsCrossEnginePredictorRevisions) {
    const ModelBenchmarkConfig config = MakeCrossEnginePredictionConfig(RocketForgeTests::TestTempDir());
    const ModelBenchmarkResult result = DescribeModelBenchmarkEnvironment(config);

    const auto profiles = nlohmann::json::parse(result.predictionConfigurationJson);
    ASSERT_EQ(profiles.size(), config.models.size());
    for (const auto& [model, profile] : profiles.items()) {
        SCOPED_TRACE(model);
        EXPECT_EQ(profile.at("engine"), 3);
        EXPECT_EQ(profile.at("revision"), RocketSim::GetEngineInfo(RocketSim::EngineId::V3).revision);
    }
}

TEST(ModelBenchmark, UnavailablePredictionEngineFailsBeforeLoadingModels) {
    const ModelBenchmarkConfig config = MakeCrossEnginePredictionConfig(RocketForgeTests::TestTempDir());
    EXPECT_NO_THROW(RequireModelBenchmarkEngines(config));

    const RocketSim::Testing::ScopedEngineUnavailable unavailable(RocketSim::EngineId::V3, "test reason");
    EXPECT_THROW(RequireModelBenchmarkEngines(config), RocketSim::EngineUnavailableError);

    try {
        RunModelBenchmark(config, {}, {});
        FAIL() << "A benchmark whose models predict with an unavailable engine must not run.";
    } catch (const RocketSim::EngineUnavailableError& error) {
        EXPECT_EQ(error.engine, RocketSim::EngineId::V3);
    }
}

TEST(ModelBenchmark, SamplingUsesValidRowsAndIndependentGenerators)
{
	torch::NoGradGuard noGrad;
	BenchmarkModelSetOwner owner;
	GGL::PartialModelConfig policy;
	policy.layerSizes = {4};
	policy.addLayerNorm = false;
	GGL::ModelManager::MakeModels(false, 1, 3, {}, policy, {}, torch::kCPU, owner.models);
	for (auto* model : owner.models)
		for (auto parameter : model->parameters())
			parameter.zero_();
	owner.models["policy"]->parameters().back().copy_(torch::tensor({0.0f, 2.0f, 50.0f}));

	const auto obs = torch::zeros({128, 1});
	const auto masks = torch::tensor({1, 1, 0}, torch::kUInt8).repeat({128, 1});
	const auto globalState = at::detail::getDefaultCPUGenerator().get_state();
	auto sample = [&](int rows, float temperature, at::Generator generator) {
		torch::Tensor actions;
		GGL::ModelManager::InferActionsFromModelsValidRows(owner.models, obs, masks,
			false, temperature, false, &actions, nullptr, 0.0f, false,
			GGL::CudaGraphBatchPolicy::PowerOfTwo, false, rows, generator);
		return actions;
	};

	auto generator = at::detail::createCPUGenerator(1979);
	auto repeat = at::detail::createCPUGenerator(1979);
	const auto first = sample(96, 10.0f, generator);
	EXPECT_EQ(first.numel(), 96);
	EXPECT_TRUE(torch::equal(first, sample(96, 10.0f, repeat)));

	// A padded input must consume exactly the RNG draws of its active rows.
	auto referenceGenerator = at::detail::createCPUGenerator(1979);
	torch::Tensor reference;
	GGL::ModelManager::InferActionsFromModels(owner.models, obs.narrow(0, 0, 96), masks.narrow(0, 0, 96),
		false, 10.0f, false, &reference, nullptr, 0.0f, false,
		GGL::CudaGraphBatchPolicy::PowerOfTwo, false, referenceGenerator);
	EXPECT_TRUE(torch::equal(first, reference));
	EXPECT_TRUE(torch::equal(generator.get_state(), referenceGenerator.get_state()));
	EXPECT_TRUE(torch::equal(sample(32, 10.0f, generator), sample(32, 10.0f, repeat)));
	EXPECT_TRUE((first == 0).any().item<bool>());
	EXPECT_TRUE((first == 1).any().item<bool>());
	EXPECT_FALSE((first == 2).any().item<bool>());
	EXPECT_TRUE((sample(96, 0.01f, at::detail::createCPUGenerator(1979)) == 1).all().item<bool>());
	EXPECT_TRUE(torch::equal(globalState, at::detail::getDefaultCPUGenerator().get_state()));
}

TEST(ModelBenchmark, SamplingProvenancePreservesDeterministicIdentity)
{
	ModelBenchmarkConfig config;
	const auto deterministic = DescribeModelBenchmarkEnvironment(config);
	config.kickoffTemperature = 2.0f;
	EXPECT_EQ(DescribeModelBenchmarkEnvironment(config).evaluationIdentity, deterministic.evaluationIdentity);

	config.stochasticKickoff = true;
	const auto sampled = DescribeModelBenchmarkEnvironment(config);
	EXPECT_NE(sampled.evaluationIdentity, deterministic.evaluationIdentity);
	const auto identity = nlohmann::json::parse(sampled.evaluationIdentity);
	EXPECT_TRUE(identity.at("stochasticKickoff").get<bool>());
	EXPECT_FLOAT_EQ(identity.at("kickoffTemperature").get<float>(), 2.0f);
	config.kickoffTemperature = 0.65f;
	EXPECT_NE(DescribeModelBenchmarkEnvironment(config).evaluationIdentity, sampled.evaluationIdentity);
	config.stochasticKickoff = false;
	EXPECT_EQ(DescribeModelBenchmarkEnvironment(config).evaluationIdentity, deterministic.evaluationIdentity);
}

TEST(ModelBenchmark, CudaSamplingUsesRunLocalRngOutsideGraphReplay)
{
#ifndef RG_CUDA_SUPPORT
	GTEST_SKIP() << "Built without CUDA support";
#else
	ROCKETFORGE_REQUIRE_CUDA();
	torch::NoGradGuard noGrad;
	BenchmarkModelSetOwner owner;
	GGL::PartialModelConfig policy;
	policy.layerSizes = {4};
	policy.addLayerNorm = false;
	const torch::Device device(torch::kCUDA, 0);
	GGL::ModelManager::MakeModels(false, 1, 3, {}, policy, {}, device, owner.models);
	for (auto* model : owner.models)
		for (auto parameter : model->parameters())
			parameter.zero_();
	owner.models["policy"]->parameters().back().copy_(torch::tensor({0.0f, 2.0f, 50.0f}, device));
	GGL::ModelManager::ClearCudaGraphCache();
	GGL::ModelManager::ResetCudaGraphStats();

	const auto obs = torch::zeros({128, 1}, device);
	const auto masks = torch::tensor({1, 1, 0}, torch::kUInt8).repeat({128, 1}).to(device);
	auto sample = [&](float temperature, bool deterministic, uint64_t seed) {
		auto generator = at::cuda::detail::createCUDAGenerator(0);
		generator.set_current_seed(seed);
		torch::Tensor actions;
		GGL::ModelManager::InferActionsFromModelsValidRows(owner.models, obs, masks,
			deterministic, temperature, false, &actions, nullptr, 0.0f, true,
			GGL::CudaGraphBatchPolicy::PowerOfTwo, false, 96, generator);
		return actions.cpu();
	};

	const auto first = sample(10.0f, false, 1979);
	EXPECT_EQ(first.numel(), 96);
	EXPECT_TRUE(torch::equal(first, sample(10.0f, false, 1979)));
	EXPECT_FALSE(torch::equal(first, sample(10.0f, false, 1980)));
	EXPECT_TRUE((first == 0).any().item<bool>());
	EXPECT_TRUE((first == 1).any().item<bool>());
	EXPECT_FALSE((first == 2).any().item<bool>());
	EXPECT_TRUE((sample(0.01f, false, 1979) == 1).all().item<bool>());
	EXPECT_TRUE((sample(1.0f, true, 1979) == 1).all().item<bool>());
	EXPECT_GT(GGL::ModelManager::GetCudaGraphStats().replays, 0u);
#endif
}
