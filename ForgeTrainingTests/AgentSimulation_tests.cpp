#include <gtest/gtest.h>
#include "TestSupport.h"

#include <GigaLearnInfer/AgentRegistry.h>
#include <GigaLearnInfer/AgentSetupSnapshot.h>
#include <TrainingSetup/TrainingSetupSnapshot.h>
#include <Inference/InferenceModelSource.h>
#include <StageQueue/StageQueueRunner.h>
#include <Storage/BuildStorage.h>
#include <RLGymCPP/BallPrediction/ProfiledObsBuilder.h>
#include <RLGymCPP/BallPrediction/PredictionContext.h>
#include <RocketSim/Engine.h>
#include <fstream>
#include <functional>

using namespace RocketForge;
using namespace RocketForgeTraining;
namespace fs = std::filesystem;
namespace BP = RLGC::BallPrediction;

namespace {
	void ExpectBall(const RocketSim::BallState& actual, const RocketSim::BallState& expected) {
		for (int i = 0; i < 3; ++i) {
			EXPECT_NEAR(actual.pos[i], expected.pos[i], 0.01f);
			EXPECT_NEAR(actual.vel[i], expected.vel[i], 0.01f);
			EXPECT_NEAR(actual.angVel[i], expected.angVel[i], 0.001f);
		}
	}

	class PredictionObs : public RLGC::ObsBuilder {
	public:
		RLGC::FList BuildObs(const RLGC::Player&, const RLGC::GameState& state) override {
			const RocketSim::BallState future = state.GetFutureBallState(1.f);
			return { future.pos.x, future.pos.y, future.pos.z };
		}
	};

	struct PredictionAgent : Agent {
		RLGC::ObsBuilder* MakeObsBuilder() const override { return new PredictionObs(); }
	};

	fs::path RegisterSimulationAgent(const std::string& name, int engine = 2) {
		const fs::path root = RocketForgeTests::TestTempDir(name);
		AgentRegistry::Register(name, [=] {
			std::unique_ptr<Agent> agent = std::make_unique<Agent>();
			agent->agentName = name;
			agent->modelPath = root;
			agent->trainingEngineVersion = engine;
			return agent;
		});

		return root;
	}
}

TEST(AgentSimulation, RoundTripKeepsIndependentTrainingAndPredictionEngines) {
	Agent agent;
	agent.trainingEngineVersion = 3;
	agent.ballPredictionEngineVersion = 2;

	const std::optional<AgentSetupSnapshot> setup = AgentSetupSnapshotFromJson(AgentSetupSnapshotToJson(CaptureAgentSetupSnapshot(agent)));
	ASSERT_TRUE(setup);

	Agent restored;
	ApplyAgentSetupSnapshot(restored, *setup);
	EXPECT_EQ(restored.trainingEngineVersion, 3);
	EXPECT_EQ(restored.GetBallPredictionEngineVersion(), 2);
}

TEST(AgentSimulation, LegacySnapshotsRestoreRecordedEngineAndIgnoreTheirPredictionRate) {
	for (int engine : { 0, 2, 3 }) {
		TrainingSetupSnapshot snapshot;
		snapshot.engineVersion = engine;
		snapshot.agentSetup = CaptureAgentSetupSnapshot(Agent{});

		nlohmann::json encoded;
		GGL::JsonSave(snapshot, encoded);
		encoded["agent_setup"].erase("training_engine_version");
		encoded["agent_setup"].erase("ball_prediction_engine_version");
		encoded["agent_setup"]["ball_pred_tick_rate"] = 60;

		TrainingSetupSnapshot loaded;
		ASSERT_TRUE(GGL::JsonLoad(encoded, loaded).empty());
		ASSERT_TRUE(loaded.agentSetup);
		EXPECT_EQ(loaded.agentSetup->trainingEngineVersion, engine == 0 ? 2 : engine);
		EXPECT_EQ(loaded.agentSetup->ballPredictionEngineVersion, engine == 0 ? 2 : engine);
		EXPECT_FALSE(AgentSetupSnapshotToJson(*loaded.agentSetup).contains("ball_pred_tick_rate"));
	}
}

TEST(AgentSimulation, ContradictoryRecordedEnginesThrow) {
	AgentSetupSnapshot setup = CaptureAgentSetupSnapshot(Agent{});
	setup.trainingEngineVersion = 3;
	EXPECT_THROW(ResolveAgentSimulation(setup, 2), std::runtime_error);

	setup.trainingEngineVersion = 2;
	EXPECT_NO_THROW(ResolveAgentSimulation(setup, 2));
}

TEST(AgentSimulation, RequiredEnginesIncludeThePredictorEvenWhenPredictionIsOff) {
	AgentSetupSnapshot setup = CaptureAgentSetupSnapshot(Agent{});
	setup.useBallPrediction = false;
	setup.trainingEngineVersion = 2;
	setup.ballPredictionEngineVersion = 3;
	EXPECT_EQ(RequiredEngines(setup), (std::vector<RocketSim::EngineId>{ RocketSim::EngineId::V2, RocketSim::EngineId::V3 }));

	setup.ballPredictionEngineVersion = 0;
	EXPECT_EQ(RequiredEngines(setup), std::vector<RocketSim::EngineId>{ RocketSim::EngineId::V2 });
}

TEST(AgentSimulation, ObsSizeIsMeasuredOnTheTrainingEngine) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	Agent agent;
	agent.trainingEngineVersion = 3;
	agent.ballPredictionEngineVersion = 2;

	const RocketSim::Testing::ScopedEngineUnavailable unavailable(RocketSim::EngineId::V3, "test reason");
	EXPECT_THROW(agent.GetObsSize(), RocketSim::EngineUnavailableError);
}

TEST(AgentSimulation, NewQueueUsesAgentDefinitionInsteadOfOldGlobalSelector) {
	const std::string name = "simulation-new";
	RegisterSimulationAgent(name, 3);

	ForgeSettings settings;
	settings.agentName = name;
	settings.buildName = "test";
	StageSettings stage;
	stage.stageName = "stage";
	stage.targetTotalTimesteps = 1000;
	settings.stages = { stage };

	EXPECT_EQ(ResolveTrainingSimulation(settings), std::vector<RocketSim::EngineId>{ RocketSim::EngineId::V3 });
	ASSERT_TRUE(settings.stages[0].agentSetupOverride);
	EXPECT_EQ(settings.stages[0].agentSetupOverride->ballPredictionEngineVersion, 3);
}

TEST(AgentSimulation, ContinuedQueueAndDescendantsRestoreSavedProfileWithoutRewritingHistory) {
	const std::string name = "simulation-continue";
	const fs::path root = RegisterSimulationAgent(name, 2);
	BuildStorage storage(root, "test");
	storage.EnsureBuildExists();

	TrainingSegmentMetadata segment;
	segment.segmentId = "v1.00";
	segment.stageName = "stage";
	segment.folderName = storage.MakeSegmentFolderName(segment.segmentId, segment.stageName);
	storage.SaveSegment(segment);
	fs::create_directories(storage.SegmentCheckpointsRoot(segment.folderName) / "100");

	TrainingSetupSnapshot snapshot;
	snapshot.engineVersion = 3;
	snapshot.agentSetup = CaptureAgentSetupSnapshot(*AgentRegistry::Create(name));
	snapshot.agentSetup->trainingEngineVersion = 3;
	snapshot.agentSetup->ballPredictionEngineVersion = 2;
	const fs::path path = storage.SegmentRoot(segment.folderName) / TrainingSetupSnapshot::FILE_NAME;
	ASSERT_TRUE(GGL::JsonSaveToFile(snapshot, path));
	const fs::file_time_type before = fs::last_write_time(path);

	ForgeSettings settings;
	settings.agentName = name;
	settings.buildName = "test";
	StageSettings stage;
	stage.stageName = "stage";
	stage.targetTotalTimesteps = 1000;
	stage.bootstrap.mode = BootstrapConfig::Mode::ContinueExisting;
	stage.bootstrap.selectedExistingVersionId = segment.segmentId;
	AgentSetupSnapshot current = CaptureAgentSetupSnapshot(*AgentRegistry::Create(name));
	current.policy.layerSizes = { 32 };
	stage.agentSetupOverride = current;
	settings.stages = { stage, StageSettings{} };
	settings.stages[1].stageName = "next";
	settings.stages[1].targetTotalTimesteps = 2000;

	EXPECT_EQ(ResolveTrainingSimulation(settings), (std::vector<RocketSim::EngineId>{ RocketSim::EngineId::V3, RocketSim::EngineId::V2 }));
	for (const StageSettings& entry : settings.stages) {
		ASSERT_TRUE(entry.agentSetupOverride);
		EXPECT_EQ(entry.agentSetupOverride->trainingEngineVersion, 3);
		EXPECT_EQ(entry.agentSetupOverride->ballPredictionEngineVersion, 2);
	}

	EXPECT_EQ(settings.stages[0].agentSetupOverride->policy.layerSizes, std::vector<int>{ 32 });
	EXPECT_EQ(fs::last_write_time(path), before);

	ResolvedInferenceAgent inference = ResolveInferenceAgent({ name, storage.SegmentCheckpointsRoot(segment.folderName) / "100", path });
	EXPECT_EQ(inference.agent->trainingEngineVersion, 3);
	EXPECT_EQ(inference.agent->GetBallPredictionEngineVersion(), 2);
}

TEST(AgentSimulation, ContextMatchesNativePredictionAcrossStepsAndHorizonGrowth) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	std::unique_ptr<RocketSim::Arena> arena(RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR));

	RocketSim::BallState seed;
	seed.pos = { 3900, 2000, 600 };
	seed.vel = { 1200, 1800, -300 };
	seed.angVel = { 2, 1, 0 };
	arena->ball->SetState(seed);

	std::shared_ptr<BP::Context> context = BP::Acquire(arena.get(), arena->gameMode, arena->GetEngine());
	for (int frame = 0; frame < 5; ++frame) {
		for (float time : { 0.125f, 1.f, 2.f }) {
			const RocketSim::BallState actual = context->Predict(arena->ball->GetState(), arena->tickCount, time);
			const RocketSim::BallState expected = arena->GetFutureBallState(time);
			ExpectBall(actual, expected);
			ExpectBall(context->Predict(arena->ball->GetState(), arena->tickCount, time), expected);
		}

		arena->Step(8);
	}
}

TEST(AgentSimulation, SameTickTeleportAndCounterResetDiscardOldPrediction) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	const RocketSim::GameMode mode = RocketSim::GameMode::SOCCAR;
	const RocketSim::EngineId engine = RocketSim::EngineId::V2;
	std::shared_ptr<BP::Context> context = BP::Acquire(nullptr, mode, engine);

	RocketSim::BallState seed;
	seed.pos = { 0, 0, 900 };
	seed.vel = { 100, 200, 0 };
	context->Predict(seed, 100, 1.f);

	seed.pos = { 2500, 2000, 1000 };
	std::shared_ptr<BP::Context> reference = BP::Acquire(nullptr, mode, engine);
	const RocketSim::BallState expected = reference->Predict(seed, 0, 1.f);
	const RocketSim::BallState actual = context->Predict(seed, 100, 1.f);
	for (int i = 0; i < 3; ++i)
		EXPECT_FLOAT_EQ(actual.pos[i], expected.pos[i]);

	seed.pos.x = -1200;
	const RocketSim::BallState reset = context->Predict(seed, 0, 1.f);
	const RocketSim::BallState resetExpected = reference->Predict(seed, 0, 1.f);
	for (int i = 0; i < 3; ++i)
		EXPECT_FLOAT_EQ(reset.pos[i], resetExpected.pos[i]);
}

TEST(AgentSimulation, CustomObservationUsesSavedPredictorInsteadOfPacketOrMatchEngine) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	const char* bothPredictors = std::getenv("ROCKETFORGE_TEST_BOTH_PREDICTORS");
	std::unique_ptr<RocketSim::Arena> arena(RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR));
	const int arenaEngine = static_cast<int>(arena->GetEngine());
	const int predictionEngine = bothPredictors && std::string(bothPredictors) == "1" ? (arenaEngine == 2 ? 3 : 2) : arenaEngine;
	arena->AddCar(RocketSim::Team::BLUE);

	RLGC::GameState state(arena.get());
	state.ball.pos = { 3500, 200, 800 };
	state.ball.vel = { 2000, 1000, 0 };
	state.ballPredData.resize(121);
	for (auto& ball : state.ballPredData)
		ball.pos = { 99999, 99999, 99999 };

	PredictionAgent agent;
	agent.trainingEngineVersion = predictionEngine;
	std::unique_ptr<RLGC::ObsBuilder> builder(agent.MakeConfiguredObsBuilder());
	builder->Reset(state);
	const RLGC::FList observations = builder->BuildObs(state.players[0], state);

	std::shared_ptr<BP::Context> context = BP::Acquire(arena.get(), state.gameMode, RocketSim::EngineFromVersion(predictionEngine));
	const RocketSim::BallState expected = context->Predict(state.ball, state.lastTickCount, 1.f);
	ASSERT_EQ(observations.size(), 3u);
	for (int i = 0; i < 3; ++i)
		EXPECT_FLOAT_EQ(observations[i], expected.pos[i]);

	EXPECT_NE(observations[0], 99999);
	EXPECT_EQ(state.predictionContext, nullptr);
}

TEST(AgentSimulation, ArenaPredictionStandsInOnlyForCurrentStateOfTheSameEngine) {
    ROCKETFORGE_REQUIRE_ROCKETSIM();
    std::unique_ptr<RocketSim::Arena> arena(RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR));
    arena->AddCar(RocketSim::Team::BLUE);
    arena->Step(8);
    const RocketSim::EngineId engine = arena->GetEngine();
    const RocketSim::EngineId otherEngine = engine == RocketSim::EngineId::V2 ? RocketSim::EngineId::V3 : RocketSim::EngineId::V2;

    struct Case {
        const char* label;
        RocketSim::EngineId engine;
        std::function<void(RLGC::GameState&)> edit;
        bool expected;
    };
    const Case cases[] = {
        {"same engine on the arena's current state", engine, {}, true},
        {"different prediction engine", otherEngine, {}, false},
        {"ball edited away from the arena", engine, [](RLGC::GameState& state) { state.ball.pos.x += 100; }, false},
        {"state older than the arena", engine, [](RLGC::GameState& state) { state.lastTickCount -= 1; }, false},
        {"no arena", engine, [](RLGC::GameState& state) { state.lastArena = nullptr; }, false},
    };

    for (const Case& testCase : cases) {
        SCOPED_TRACE(testCase.label);
        RLGC::GameState state(arena.get());
        if (testCase.edit)
            testCase.edit(state);
        EXPECT_EQ(state.CanUseArenaBallPrediction(testCase.engine), testCase.expected);
    }

    // Predictions are made at 120 Hz, so an arena stepping at another rate cannot stand in.
    std::unique_ptr<RocketSim::Arena> slowArena(RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR, {}, 60));
    slowArena->AddCar(RocketSim::Team::BLUE);
    slowArena->Step(8);
    const RLGC::GameState slowState(slowArena.get());
    EXPECT_FALSE(slowState.CanUseArenaBallPrediction(slowArena->GetEngine()));
}

TEST(AgentSimulation, BenchmarkRuntimeSeparatesPredictionEngines) {
	ModelBenchmarkRuntimeSignature first;
	ModelBenchmarkRuntimeSignature second;
	EXPECT_EQ(SelectModelBenchmarkStrategy({ first, second }), ModelBenchmarkStrategy::SameRuntimeSlot);

	second.ballPredictionEngineVersion = 3;
	EXPECT_EQ(SelectModelBenchmarkStrategy({ first, second }), ModelBenchmarkStrategy::Heterogeneous);
}
