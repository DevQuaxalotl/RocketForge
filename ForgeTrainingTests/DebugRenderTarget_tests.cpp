#include <gtest/gtest.h>

#include "DebugRender/DebugRenderTarget.h"
#include "Storage/BuildStorage.h"
#include "TestSupport.h"

#include <GigaLearnInfer/Agent.h>
#include <GigaLearnInfer/AgentRegistry.h>

#include <filesystem>
#include <string>

using namespace RocketForgeTraining;

namespace fs = std::filesystem;

namespace {

	const char* const TARGET_AGENT = "debug-render-target-test-agent";
	const char* const TARGET_BUILD = "V1";
	const char* const TARGET_SEGMENT = "v1.0";
	const char* const TARGET_STAGE = "example-ball-chase";

	/** Registers the test agent over a fresh temp Models folder holding segment v1.0 with checkpoints 100 and 200. */
	void MakeTargetFixture()
	{
		const fs::path modelsRoot = RocketForgeTests::TestTempDir("debug_render_target") / "Agent" / "Models";
		fs::create_directories(modelsRoot);

		AgentRegistry::Register(TARGET_AGENT, [modelsRoot] {
			auto agent = std::make_unique<Agent>();
			agent->agentName = TARGET_AGENT;
			agent->modelPath = modelsRoot;
			return agent;
		});

		BuildStorage storage(modelsRoot, TARGET_BUILD);
		storage.EnsureBuildExists();

		TrainingSegmentMetadata segment;
		segment.segmentId = TARGET_SEGMENT;
		segment.folderName = storage.MakeSegmentFolderName(TARGET_SEGMENT, TARGET_STAGE);
		segment.buildName = TARGET_BUILD;
		segment.stageName = TARGET_STAGE;

		for (int timesteps : { 100, 200 })
			fs::create_directories(storage.SegmentCheckpointsRoot(segment.folderName) / std::to_string(timesteps));

		storage.SaveSegment(segment);
	}

}

TEST(DebugRenderTarget, ResolvesNewestOrRequestedCheckpoint)
{
	MakeTargetFixture();

	const DebugRenderTarget newest = ResolveDebugRenderTarget(TARGET_AGENT, TARGET_BUILD, TARGET_SEGMENT, 0);
	ASSERT_TRUE(newest.error.empty()) << newest.error;
	EXPECT_EQ(newest.checkpointFolder.filename().string(), "200");
	EXPECT_EQ(newest.trainedStageName, TARGET_STAGE);
	EXPECT_FALSE(newest.isImported);

	const DebugRenderTarget requested = ResolveDebugRenderTarget(TARGET_AGENT, TARGET_BUILD, TARGET_SEGMENT, 100);
	ASSERT_TRUE(requested.error.empty()) << requested.error;
	EXPECT_EQ(requested.checkpointFolder.filename().string(), "100");

	const DebugRenderTarget missingCheckpoint = ResolveDebugRenderTarget(TARGET_AGENT, TARGET_BUILD, TARGET_SEGMENT, 150);
	EXPECT_FALSE(missingCheckpoint.error.empty());
	EXPECT_TRUE(missingCheckpoint.checkpointFolder.empty());

	const DebugRenderTarget unknownSegment = ResolveDebugRenderTarget(TARGET_AGENT, TARGET_BUILD, "v9.9", 0);
	EXPECT_NE(unknownSegment.error.find("v9.9"), std::string::npos) << unknownSegment.error;
	EXPECT_TRUE(unknownSegment.checkpointFolder.empty());
}
