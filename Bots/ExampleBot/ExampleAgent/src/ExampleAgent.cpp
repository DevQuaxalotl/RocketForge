
#include "ExampleAgent.h"

#include <GigaLearnInfer/AgentRegistry.h>

// Required for Agent to be selectable by executables
REGISTER_AGENT("example-agent", ExampleTrainingAgent)

ExampleTrainingAgent::ExampleTrainingAgent()
{
	agentName = "example-agent";

	// Built-in builders are registered in DefaultAgentComponents.
	// A custom builder needs REGISTER_OBS_BUILDER in its own source file so RLBot submission export can find it.
	obsBuilderName = "AdvancedObsPadded";

	// Set to 3 for RocketSim v3 (needs Rust; see tools/setup/install_rust). A prediction engine of 0 follows the training engine.
	trainingEngineVersion = 2;
	ballPredictionEngineVersion = 0;

	// Ticks (at 120 Hz) per action: 8 = 15 actions per second.
	// actionDelay is derived from tickSkip only at construction, so set both together.
	// Changing either on an existing build changes what the trained policy sees and does.
	tickSkip = 8;
	actionDelay = tickSkip - 1;

	// To train larger team sizes, raise this together with ExampleTrainingStage's weightedTeamSizes
	// and EXAMPLE_COMPAT_INFO.minAgentTeamSize in ExampleTrainingPlan.cpp.
	// The padded obs builder adapts to it, but changing it on an existing build requires a new build.
	maxTeamSize = 1;

	sharedHead.layerSizes = { 256, 256 };
	policy.layerSizes = { 256, 256, 256 };
	critic.layerSizes = { 256, 256, 256 };

	otimType = GGL::ModelOptimType::ADAM;
	modelActivationType = GGL::ModelActivationType::RELU;

#ifdef EXAMPLE_AGENT_MODELS_PATH
	modelPath = std::filesystem::path(EXAMPLE_AGENT_MODELS_PATH).make_preferred();
#endif
}
