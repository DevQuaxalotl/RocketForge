#include <gtest/gtest.h>

#include "TestSupport.h"

#ifdef ROCKETFORGE_TESTS_HAVE_QT

#include "TrainingSetup/TrainingWorker.h"

#include <GigaLearnInfer/AgentRegistry.h>
#include <RocketSim/Engine.h>
#include <StageRegistry.h>
#include <TrainingStage.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace RocketForgeTraining;
namespace fs = std::filesystem;

namespace {
	const std::string WORKER_AGENT = "TrainingWorkerTest";
	const std::string V3_AGENT = "TrainingWorkerV3Test";
	const std::string WORKER_STAGE = "TrainingWorkerTest";
	constexpr int TIMESTEPS_PER_ITERATION = 256;
	constexpr uint64_t TARGET_TIMESTEPS = TIMESTEPS_PER_ITERATION * 20;

	/** Models root for the fixture agents; each test points it at its own temp dir before training. */
	fs::path g_modelsRoot;

	std::unique_ptr<Agent> MakeWorkerAgent(const std::string& name, int engineVersion) {
		auto agent = std::make_unique<Agent>();
		agent->agentName = name;
		agent->trainingEngineVersion = engineVersion;
		agent->modelPath = g_modelsRoot;
		agent->sharedHead.layerSizes = { 16 };
		agent->policy.layerSizes = { 16 };
		agent->critic.layerSizes = { 16 };
		return agent;
	}

	class WorkerTestStage : public RocketForge::TrainingStage {
	public:
		explicit WorkerTestStage(std::unique_ptr<Agent> agent) : TrainingStage(std::move(agent)) {
			stageName = WORKER_STAGE;
			timestepsPerIteration = TIMESTEPS_PER_ITERATION;
			learnerConfig.numGames = 2;
			learnerConfig.deviceType = GGL::LearnerDeviceType::CPU;
			learnerConfig.ppo.tsPerItr = TIMESTEPS_PER_ITERATION;
			learnerConfig.ppo.batchSize = TIMESTEPS_PER_ITERATION;
			learnerConfig.ppo.microBatchSize = TIMESTEPS_PER_ITERATION;
			learnerConfig.ppo.epochs = 1;
			learnerConfig.ppo.maxEpisodeDuration = 1.0;
			learnerConfig.tsPerSave = TIMESTEPS_PER_ITERATION;
			learnerConfig.skillTracker.enabled = false;
			learnerConfig.sendMetrics = false;
			learnerConfig.randomSeed = 123;
		}
	};

	void RegisterWorkerFixtures(const fs::path& modelsRoot) {
		g_modelsRoot = modelsRoot;

		AgentRegistry::Register(WORKER_AGENT, [] { return MakeWorkerAgent(WORKER_AGENT, 2); });
		AgentRegistry::Register(V3_AGENT, [] { return MakeWorkerAgent(V3_AGENT, 3); });
		RocketForge::StageRegistry::Register(WORKER_STAGE, [](std::unique_ptr<Agent> agent) {
			return RocketForge::TrainingStage::Create<WorkerTestStage>(std::move(agent));
		}, {});
	}

	ForgeSettings MakeSingleStageQueue(const std::string& agentName) {
		ForgeSettings settings;
		settings.agentName = agentName;
		settings.buildName = "WorkerTest";
		settings.globals.metricsEnabled = false;
		settings.globals.skillTestingEnabled = false;

		StageSettings stage;
		stage.stageName = WORKER_STAGE;
		stage.targetTotalTimesteps = TARGET_TIMESTEPS;
		settings.stages = { stage };
		return settings;
	}

	/** Highest timestep count among durable checkpoints (<segment>/checkpoints/<timesteps>), ignoring temp autosaves. */
	uint64_t LatestCheckpointTimesteps(const fs::path& root) {
		uint64_t latest = 0;
		if (!fs::exists(root))
			return latest;

		for (const fs::directory_entry& entry : fs::recursive_directory_iterator(root)) {
			if (entry.path().filename() != "RUNNING_STATS.json" || entry.path().parent_path().parent_path().filename() != "checkpoints")
				continue;

			nlohmann::json stats;
			std::ifstream(entry.path()) >> stats;
			latest = std::max(latest, stats.at("total_timesteps").get<uint64_t>());
		}
		return latest;
	}
}

TEST(TrainingWorker, RunsQueueInProcessAndStopsCooperatively) {
	ROCKETFORGE_REQUIRE_COLLISION_MESHES();

	const fs::path root = RocketForgeTests::TestTempDir();
	RegisterWorkerFixtures(root / "Models");

	TrainingWorker worker;
	std::vector<std::string> received;

	// Stopping from StageStarted lands before the stage's learner exists, which a GUI Stop can also hit during setup.
	QObject::connect(&worker, &TrainingWorker::StageStarted, [&](int, QString, QString) {
		received.push_back("StageStarted");
		worker.RequestStop();
	});
	QObject::connect(&worker, &TrainingWorker::StopAcknowledged, [&] { received.push_back("StopAcknowledged"); });
	QObject::connect(&worker, &TrainingWorker::Finished, [&] { received.push_back("Finished"); });
	QObject::connect(&worker, &TrainingWorker::ErrorOccurred, [&](QString message, bool) {
		received.push_back("ErrorOccurred: " + message.toStdString());
	});

	worker.PrepareRun();
	worker.RunTraining(MakeSingleStageQueue(WORKER_AGENT), QString::fromStdString((root / "forge_settings.json").string()));

	EXPECT_EQ(received, (std::vector<std::string>{ "StageStarted", "StopAcknowledged", "Finished" }));

	const uint64_t savedTimesteps = LatestCheckpointTimesteps(root / "Models");
	EXPECT_GT(savedTimesteps, 0u);
	EXPECT_LT(savedTimesteps, TARGET_TIMESTEPS);
}

TEST(TrainingWorker, ReportsUnavailableSimulatorBeforeStarting) {
	const RocketSim::Testing::ScopedEngineUnavailable unavailable(RocketSim::EngineId::V3, "test reason");
	const fs::path root = RocketForgeTests::TestTempDir();
	RegisterWorkerFixtures(root / "Models");

	TrainingWorker worker;
	std::vector<std::string> received;
	QObject::connect(&worker, &TrainingWorker::StageStarted, [&](int, QString, QString) { received.push_back("StageStarted"); });
	QObject::connect(&worker, &TrainingWorker::Finished, [&] { received.push_back("Finished"); });
	QObject::connect(&worker, &TrainingWorker::ErrorOccurred, [&](QString message, bool simulatorUnavailable) {
		received.push_back("ErrorOccurred: " + message.toStdString() + (simulatorUnavailable ? " (simulator unavailable)" : ""));
	});

	worker.PrepareRun();
	worker.RunTraining(MakeSingleStageQueue(V3_AGENT), QString::fromStdString((root / "forge_settings.json").string()));

	EXPECT_EQ(received, (std::vector<std::string>{ "ErrorOccurred: test reason (simulator unavailable)" }));
	EXPECT_FALSE(fs::exists(root / "Models" / "WorkerTest" / "segments"));
}

#else

TEST(TrainingWorker, RequiresQt) {
	GTEST_SKIP() << "TrainingWorker tests need Qt";
}

#endif
