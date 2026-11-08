#include <gtest/gtest.h>

#include "TestSupport.h"

#ifdef ROCKETFORGE_TESTS_HAVE_QT

#include "ModelBenchmark/ModelBenchmarkWorker.h"
#include "ParamTesting/ParamTestingWorker.h"

#include <ForgeTraining.h>
#include <GigaLearnInfer/AgentRegistry.h>
#include <RocketSim/Engine.h>
#include <StageRegistry.h>
#include <Storage/BuildStorage.h>
#include <TrainingSetup/TrainingSetupSnapshot.h>
#include <TrainingStage.h>

#include <QCoreApplication>
#include <torch/cuda.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <string>
#include <vector>

using namespace RocketForgeTraining;
namespace fs = std::filesystem;

namespace {
	const std::string TOOL_AGENT = "ToolWorkerTest";
	const std::string TOOL_STAGE = "ToolWorkerTest";
	const std::string TOOL_BUILD = "ToolWorkerTest";
	constexpr int TIMESTEPS_PER_ITERATION = 256;

	/** Models root for the fixture agent; each test points it at its own temp dir before training. */
	fs::path g_modelsRoot;

	class ToolTestStage : public RocketForge::TrainingStage {
	public:
		explicit ToolTestStage(std::unique_ptr<Agent> agent) : TrainingStage(std::move(agent)) {
			stageName = TOOL_STAGE;
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

	void RegisterToolFixtures(const fs::path& modelsRoot) {
		g_modelsRoot = modelsRoot;

		AgentRegistry::Register(TOOL_AGENT, [] {
			auto agent = std::make_unique<Agent>();
			agent->agentName = TOOL_AGENT;
			agent->trainingEngineVersion = 2;
			agent->modelPath = g_modelsRoot;
			agent->sharedHead.layerSizes = { 16 };
			agent->policy.layerSizes = { 16 };
			agent->critic.layerSizes = { 16 };
			return agent;
		});
		RocketForge::StageRegistry::Register(TOOL_STAGE, [](std::unique_ptr<Agent> agent) {
			return RocketForge::TrainingStage::Create<ToolTestStage>(std::move(agent));
		}, {});
	}

	/** Trains one fixture iteration and returns the new segment's id, or empty when training didn't complete. */
	std::string TrainBaselineSegment(const fs::path& root) {
		ForgeSettings settings;
		settings.agentName = TOOL_AGENT;
		settings.buildName = TOOL_BUILD;
		settings.globals.metricsEnabled = false;
		settings.globals.skillTestingEnabled = false;

		StageSettings stage;
		stage.stageName = TOOL_STAGE;
		stage.targetTotalTimesteps = TIMESTEPS_PER_ITERATION;
		settings.stages = { stage };

		if (RunTraining(settings, root / "forge_settings.json", {}).status != RunStatus::Completed)
			return {};

		const std::vector<TrainingSegmentMetadata> segments = BuildStorage(g_modelsRoot, TOOL_BUILD).ListSegments();
		return segments.empty() ? std::string() : segments.front().segmentId;
	}

	/** Durable checkpoints (<segment>/checkpoints/<timesteps>) under root whose path contains the filter, ignoring temp autosaves. */
	std::vector<fs::path> DurableCheckpoints(const fs::path& root, const std::string& pathFilter = {}) {
		std::vector<fs::path> checkpoints;
		for (const fs::directory_entry& entry : fs::recursive_directory_iterator(root)) {
			const fs::path folder = entry.path().parent_path();
			if (entry.path().filename() == "RUNNING_STATS.json" && folder.parent_path().filename() == "checkpoints"
				&& folder.string().find(pathFilter) != std::string::npos)
				checkpoints.push_back(folder);
		}
		return checkpoints;
	}

	uint64_t CheckpointTimesteps(const fs::path& checkpoint) {
		nlohmann::json stats;
		std::ifstream(checkpoint / "RUNNING_STATS.json") >> stats;
		return stats.at("total_timesteps").get<uint64_t>();
	}

	ModelBenchmarkModel MakeBenchmarkModel(const std::string& displayName, const fs::path& checkpoint) {
		ModelBenchmarkModel model;
		model.displayName = displayName;
		model.agentName = TOOL_AGENT;
		model.stageName = TOOL_STAGE;
		model.versionId = displayName;
		model.checkpointPath = checkpoint;
		model.trainingSetupPath = checkpoint.parent_path().parent_path() / RocketForge::TrainingSetupSnapshot::FILE_NAME;
		model.agentSetupSource = ModelBenchmarkAgentSetupSource::SavedSegment;
		return model;
	}
}

TEST(ModelBenchmarkWorker, RunsInProcessAndStreamsGames) {
	ROCKETFORGE_REQUIRE_COLLISION_MESHES();

	// Live snapshots go out every 100 finished games, so a shorter run would only report startup and the final result.
	constexpr int benchmarkGames = 200;
	const fs::path root = RocketForgeTests::TestTempDir();
	RegisterToolFixtures(root / "Models");

	ASSERT_FALSE(TrainBaselineSegment(root).empty());
	const std::vector<fs::path> trained = DurableCheckpoints(root / "Models");
	ASSERT_FALSE(trained.empty());

	const fs::path firstCheckpoint = trained.front();
	const fs::path secondCheckpoint = firstCheckpoint.parent_path() / (firstCheckpoint.filename().string() + "_copy");
	fs::copy(firstCheckpoint, secondCheckpoint, fs::copy_options::recursive);

	ModelBenchmarkConfig config;
	config.models = { MakeBenchmarkModel("A", firstCheckpoint), MakeBenchmarkModel("B", secondCheckpoint) };
	config.numArenas = 2;
	config.usePairBudgets = true;
	config.pairBudgets = { { 0, 1, 1, benchmarkGames } };
	config.run2v2 = config.run3v3 = false;
	config.noGoalTimeoutSeconds = .1f;
	config.emitLiveResultSnapshots = true;
	config.seed = 123;

	ModelBenchmarkWorker worker;
	int progressSignals = 0;
	size_t streamedGames = 0;
	std::optional<ModelBenchmarkResult> completed;
	std::vector<std::string> failures;

	QObject::connect(&worker, &ModelBenchmarkWorker::ProgressUpdated, [&] {
		const std::optional<ModelBenchmarkProgress> progress = worker.TakeLatestProgress();
		if (!progress)
			return;

		++progressSignals;
		if (progress->liveResult)
			streamedGames = std::max(streamedGames, progress->liveResult->games.size());
	});
	QObject::connect(&worker, &ModelBenchmarkWorker::Completed, [&](ModelBenchmarkResult result) { completed = std::move(result); });
	QObject::connect(&worker, &ModelBenchmarkWorker::Cancelled, [&] { failures.push_back("Cancelled"); });
	QObject::connect(&worker, &ModelBenchmarkWorker::Failed, [&](QString message) { failures.push_back(message.toStdString()); });

	worker.RunBenchmark(config);

	EXPECT_TRUE(failures.empty()) << failures.front();
	ASSERT_TRUE(completed.has_value());
	EXPECT_EQ(completed->games.size(), (size_t)benchmarkGames);
	EXPECT_GE(progressSignals, 3);
	EXPECT_GT(streamedGames, 0u);
	EXPECT_EQ(completed->rocketSimVersions, std::set<int>{ static_cast<int>(config.engine) });

	const fs::path buildFolder = QCoreApplication::applicationDirPath().toStdString();
	EXPECT_FALSE(fs::exists(buildFolder / ("ForgeEngineHost-v" + std::to_string(static_cast<int>(config.engine)) + ".exe")));
}

TEST(ModelBenchmarkWorker, ReportsUnavailableSimulatorBeforeStarting) {
	const RocketSim::Testing::ScopedEngineUnavailable unavailable(RocketSim::EngineId::V3, "test reason");

	ModelBenchmarkConfig config;
	config.engine = RocketSim::EngineId::V3;

	ModelBenchmarkWorker worker;
	std::vector<std::string> received;
	QObject::connect(&worker, &ModelBenchmarkWorker::ProgressUpdated, [&] { received.push_back("ProgressUpdated"); });
	QObject::connect(&worker, &ModelBenchmarkWorker::Completed, [&](ModelBenchmarkResult) { received.push_back("Completed"); });
	QObject::connect(&worker, &ModelBenchmarkWorker::Failed, [&](QString message, bool simulatorUnavailable) {
		received.push_back("Failed: " + message.toStdString() + (simulatorUnavailable ? " (simulator unavailable)" : ""));
	});

	worker.RunBenchmark(config);
	EXPECT_EQ(received, (std::vector<std::string>{ "Failed: test reason (simulator unavailable)" }));
}

TEST(ParamTestingWorker, CancelDuringSegmentSetupStopsAndKeepsCheckpoint) {
	ROCKETFORGE_REQUIRE_COLLISION_MESHES();

	constexpr uint64_t segmentTimesteps = TIMESTEPS_PER_ITERATION * 20;
	const fs::path root = RocketForgeTests::TestTempDir();
	RegisterToolFixtures(root / "Models");

	const std::string baselineSegmentId = TrainBaselineSegment(root);
	ASSERT_FALSE(baselineSegmentId.empty());
	const uint64_t baselineTimesteps = CheckpointTimesteps(DurableCheckpoints(root / "Models").front());

	ParamTestingSegmentConfig trial;
	trial.label = "Trial";
	trial.stageName = TOOL_STAGE;

	ParamTestingConfig config;
	config.agentName = TOOL_AGENT;
	config.buildName = TOOL_BUILD;
	config.baselineStageName = TOOL_STAGE;
	config.baselineSegmentId = baselineSegmentId;
	config.metricsEnabled = false;
	config.segmentTimesteps = segmentTimesteps;
	config.segments = { trial };

	ParamTestingWorker worker;
	std::vector<std::string> received;
	std::string trialSegmentId;

	// Cancelling before the segment's queue starts lands in the window where the queue resets its stop signals.
	QObject::connect(&worker, &ParamTestingWorker::ProgressUpdated, [&](ParamTestingProgress progress) {
		if (progress.stageStarted)
			trialSegmentId = progress.versionId;
		else if (progress.label == "Starting test segment")
			worker.Cancel();

		received.push_back(progress.stageStarted ? "StageStarted" : "Progress");
	});
	QObject::connect(&worker, &ParamTestingWorker::Completed, [&](ParamTestingResult) { received.push_back("Completed"); });
	QObject::connect(&worker, &ParamTestingWorker::Cancelled, [&] { received.push_back("Cancelled"); });
	QObject::connect(&worker, &ParamTestingWorker::Failed, [&](QString message) { received.push_back("Failed: " + message.toStdString()); });

	worker.RunParamTesting(config);

	EXPECT_EQ(received, (std::vector<std::string>{ "Progress", "Progress", "StageStarted", "Cancelled" }));
	ASSERT_FALSE(trialSegmentId.empty());

	const std::vector<fs::path> trialCheckpoints = DurableCheckpoints(root / "Models", trialSegmentId);
	ASSERT_FALSE(trialCheckpoints.empty());

	const uint64_t savedTimesteps = CheckpointTimesteps(trialCheckpoints.back());
	EXPECT_GT(savedTimesteps, baselineTimesteps);
	EXPECT_LT(savedTimesteps, baselineTimesteps + segmentTimesteps);
}

#else

TEST(ToolWorkers, RequiresQt) {
	GTEST_SKIP() << "Tool worker tests need Qt";
}

#endif
