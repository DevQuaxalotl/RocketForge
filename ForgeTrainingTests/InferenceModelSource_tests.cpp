#include <gtest/gtest.h>

#include "TestSupport.h"

#include <DebugRender/InferenceDebugRenderLoop.h>
#include <GigaLearnInfer/AgentRegistry.h>
#include <Inference/InferenceModelSource.h>
#include <RocketSim/Engine.h>
#include <TrainingSetup/TrainingSetupSnapshot.h>
#include <StaticOpponents/StaticOpponentSelection.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>

using namespace RocketForge;

TEST(InferenceModelSource, SegmentSourceDerivesTrainingSetupPath) {
	// checkpointFolder = <segment>/checkpoints/<timesteps>
	std::filesystem::path checkpoint = "root/seg/checkpoints/1000";
	InferenceModelSource src = FromSegmentCheckpoint("ExampleAgent", checkpoint);

	ASSERT_EQ(src.agentName, "ExampleAgent");
	ASSERT_EQ(src.checkpointFolder, checkpoint);
	ASSERT_EQ(src.setupSnapshotPath, std::filesystem::path("root/seg") / TrainingSetupSnapshot::FILE_NAME);
}

TEST(InferenceModelSource, ImportedSourceHasNoSnapshotPath) {
	// ResolveModelFolder returns empty for a non-existent root; the source still records
	// the agent name and an empty snapshot path (never a training_setup.json).
	InferenceModelSource src = FromImportedModel("ExampleAgent", "does/not/exist");

	ASSERT_EQ(src.agentName, "ExampleAgent");
	ASSERT_TRUE(src.setupSnapshotPath.empty());
}

TEST(InferenceModelSource, StaticEntrySegmentDerivesTrainingSetupPath) {
	// A checkpointPath ending in .../checkpoints/<timesteps> is a saved segment version.
	RocketForgeTraining::StaticOpponentSelectionEntry entry;
	entry.sourceAgentName = "ExampleAgent";
	entry.checkpointPath = "root/seg/checkpoints/1000";

	InferenceModelSource src = FromStaticOpponentEntry(entry);

	ASSERT_EQ(src.agentName, "ExampleAgent");
	ASSERT_EQ(src.checkpointFolder, std::filesystem::path("root/seg/checkpoints/1000"));
	ASSERT_EQ(src.setupSnapshotPath, std::filesystem::path("root/seg") / TrainingSetupSnapshot::FILE_NAME);
}

TEST(InferenceModelSource, StaticEntryImportedHasNoSnapshotPath) {
	// A checkpointPath whose parent is not "checkpoints" is an imported model folder; the
	// network comes from INFERENCE_POLICY.json, so no training_setup snapshot is derived.
	RocketForgeTraining::StaticOpponentSelectionEntry entry;
	entry.sourceAgentName = "ExampleAgent";
	entry.checkpointPath = "some/imported/does_not_exist";

	InferenceModelSource src = FromStaticOpponentEntry(entry);

	ASSERT_EQ(src.agentName, "ExampleAgent");
	ASSERT_TRUE(src.setupSnapshotPath.empty());
}

TEST(InferenceModelSource, ResolveThrowsOnUnknownAgent) {
	InferenceModelSource src;
	src.agentName = "NoSuchAgent__";
	src.checkpointFolder = "unused";

	bool sawThrow = false;
	try
	{
		ResolveInferenceAgent(src);
	}
	catch (const std::runtime_error&)
	{
		sawThrow = true;
	}
	ASSERT_TRUE(sawThrow);
}

TEST(ResolveDebugEngines, UsesSavedSimulatorsAndRejectsBrokenMetadata) {
	namespace fs = std::filesystem;

	const fs::path root = RocketForgeTests::TestTempDir();
	const std::string agentName = "DebugSimulationTest";
	AgentRegistry::Register(agentName, [root] {
		auto agent = std::make_unique<Agent>();
		agent->agentName = "DebugSimulationTest";
		agent->trainingEngineVersion = 3;
		agent->modelPath = root / "Models";
		return agent;
	});

	const fs::path checkpoint = root / "segment" / "checkpoints" / "256";
	fs::create_directories(checkpoint);
	InferenceDebugRenderConfig config;
	config.agentName = agentName;
	config.checkpointFolder = checkpoint;

	// A legacy segment has no training_setup.json and always trained on v2.
	EXPECT_EQ(ResolveDebugEngines(config), std::vector<RocketSim::EngineId>{ RocketSim::EngineId::V2 });
	EXPECT_FALSE(ResolveInferenceAgent(FromSegmentCheckpoint(agentName, checkpoint)).appliedSnapshot);

	TrainingSetupSnapshot snapshot;
	snapshot.engineVersion = 3;
	snapshot.agentSetup = CaptureAgentSetupSnapshot(*AgentRegistry::Create(agentName));
	snapshot.agentSetup->trainingEngineVersion = 3;
	snapshot.agentSetup->ballPredictionEngineVersion = 2;
	const fs::path setup = checkpoint.parent_path().parent_path() / TrainingSetupSnapshot::FILE_NAME;
	ASSERT_TRUE(GGL::JsonSaveToFile(snapshot, setup));
	EXPECT_EQ(ResolveDebugEngines(config), (std::vector<RocketSim::EngineId>{ RocketSim::EngineId::V3, RocketSim::EngineId::V2 }));
	EXPECT_EQ(ResolveInferenceAgent(FromSegmentCheckpoint(agentName, checkpoint)).agent->ballPredictionEngineVersion, 2);

	std::ofstream(setup) << "broken";
	EXPECT_THROW(ResolveDebugEngines(config), std::runtime_error);

	config.isImported = true;
	EXPECT_EQ(ResolveDebugEngines(config), std::vector<RocketSim::EngineId>{ RocketSim::EngineId::V3 });
}
