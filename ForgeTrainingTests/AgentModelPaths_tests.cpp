#include <gtest/gtest.h>

#include "TestSupport.h"

#include "AgentModelPaths.h"

#include <GigaLearnInfer/AgentRegistry.h>

#include <filesystem>
#include <memory>
#include <string>

namespace
{
	constexpr const char* SHARED_FOLDER_AGENT_NAME = "lite-example-agent";

	std::unique_ptr<Agent> MakeSharedFolderAgent()
	{
		std::unique_ptr<Agent> agent = std::make_unique<Agent>();
		agent->agentName = SHARED_FOLDER_AGENT_NAME;
		agent->modelPath = std::filesystem::path(FORGE_SOURCE_DIR) / "Bots" / "ExampleBot" / "ExampleAgent" / "Models";
		return agent;
	}
}

// example-agent maps to Bots/ExampleBot/ExampleAgent/Models (capital Models),
// which exists in the repo. The relative path uses forward slashes regardless
// of platform so it can be handed straight to rclone.
TEST(ResolveAgentModelsRelativePath, example_agent)
{
	const std::string relativePath = RocketForgeTraining::ResolveAgentModelsRelativePath("example-agent");
	ASSERT_EQ(relativePath, std::string("Bots/ExampleBot/ExampleAgent/Models"));
}

// A variant agent can live in another agent's folder, whose name does not match the variant's.
// Directory-name matching cannot find it; only the registered agent's real modelPath does.
TEST(ResolveAgentModelsRelativePath, SharedFolderResolvesThroughModelPath)
{
	AgentRegistry::Register(SHARED_FOLDER_AGENT_NAME, MakeSharedFolderAgent);

	const std::string relativePath = RocketForgeTraining::ResolveAgentModelsRelativePath(SHARED_FOLDER_AGENT_NAME);
	ASSERT_EQ(relativePath, std::string("Bots/ExampleBot/ExampleAgent/Models"));
}

// A modelPath that does not exist yet (e.g. a stage built before its Build is created)
// has no Models ancestor on disk; resolution must fall back to it rather than climb forever.
TEST(ResolveAgentModelsFolder, MissingModelPathFallsBackToItself)
{
	Agent agent;
	agent.agentName = "missing-models-path-agent";
	agent.modelPath = RocketForgeTests::TestTempDir() / "missing" / "Agent" / "Models";

	std::filesystem::path expected = agent.modelPath;
	expected.make_preferred();
	ASSERT_EQ(RocketForgeTraining::ResolveAgentModelsFolder(&agent), expected);
}
