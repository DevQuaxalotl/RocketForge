#include "PublicBotAgents.h"

#include "PublicBotActions.h"
#include "PublicBotObs.h"

#include <GigaLearnInfer/AgentComponentRegistry.h>
#include <GigaLearnInfer/AgentRegistry.h>
#include <GigaLearnInfer/Models.h>

#include "ScriptedPolicyModel.h"

REGISTER_OBS_BUILDER("RLBotPackNectoObs", [](const Agent&) {
	return std::make_unique<NectoObs>();
})

REGISTER_OBS_BUILDER("RLBotPackNextoObs", [](const Agent&) {
	return std::make_unique<NextoObs>();
})

REGISTER_OBS_BUILDER("RLBotPackElementObs", [](const Agent&) {
	return std::make_unique<ElementObs>();
})

REGISTER_ACTION_PARSER("RLBotPackNectoAction", [](const Agent&) {
	return std::make_unique<NectoAction>();
})

REGISTER_ACTION_PARSER("RLBotPackNextoAction", [](const Agent&) {
	return std::make_unique<NextoAction>();
})

REGISTER_ACTION_PARSER("RLBotPackElementAction", [](const Agent&) {
	return std::make_unique<ElementAction>();
})

REGISTER_AGENT("Necto", NectoAgent)
REGISTER_AGENT("Nexto", NextoAgent)
REGISTER_AGENT("Element", ElementAgent)

namespace
{
	constexpr int ATTENTION_ENTITY_COUNT = 41;

	void ConfigurePublicAgent(Agent& agent)
	{
		agent.gameMode = RocketSim::GameMode::SOCCAR;
		agent.tickSkip = 8;
		agent.actionDelay = 7;
		agent.maxTeamSize = 3;
		agent.inferenceOnly = true;
		agent.trainingEngineVersion = 2;
		agent.inferenceDeterministic = true;
		agent.sharedHead.layerSizes.clear();
		agent.policy.layerSizes.clear();
		agent.critic.layerSizes.clear();
	}
}

NectoAgent::NectoAgent()
{
	ConfigurePublicAgent(*this);
	agentName = "Necto";
	obsBuilderName = "RLBotPackNectoObs";
	actionParserName = "RLBotPackNectoAction";
#ifdef PUBLIC_BOTS_ROOT
	modelPath = std::filesystem::path(PUBLIC_BOTS_ROOT)
		/ "Necto" / "Models" / "Imported Models" / "RLBotPack";
#endif
}

void NectoAgent::MakeInferenceModels(
	int obsSize,
	int numActions,
	const c10::Device& device,
	GGL::ModelSet& outModels) const
{
	outModels.Add(new GGL::ScriptedPolicyModel(
		"necto-model.pt",
		obsSize,
		numActions,
		device,
		GGL::ScriptedPolicyInput::PackedAttention,
		ATTENTION_ENTITY_COUNT));
}

NextoAgent::NextoAgent()
{
	ConfigurePublicAgent(*this);
	agentName = "Nexto";
	obsBuilderName = "RLBotPackNextoObs";
	actionParserName = "RLBotPackNextoAction";
#ifdef PUBLIC_BOTS_ROOT
	modelPath = std::filesystem::path(PUBLIC_BOTS_ROOT)
		/ "Nexto" / "Models" / "Imported Models" / "RLBotPack";
#endif
}

void NextoAgent::MakeInferenceModels(
	int obsSize,
	int numActions,
	const c10::Device& device,
	GGL::ModelSet& outModels) const
{
	outModels.Add(new GGL::ScriptedPolicyModel(
		"nexto-model.pt",
		obsSize,
		numActions,
		device,
		GGL::ScriptedPolicyInput::PackedAttention,
		ATTENTION_ENTITY_COUNT));
}

ElementAgent::ElementAgent()
{
	ConfigurePublicAgent(*this);
	agentName = "Element";
	obsBuilderName = "RLBotPackElementObs";
	actionParserName = "RLBotPackElementAction";
#ifdef PUBLIC_BOTS_ROOT
	modelPath = std::filesystem::path(PUBLIC_BOTS_ROOT)
		/ "Element" / "Models" / "Imported Models" / "RLBotPack";
#endif
}

void ElementAgent::MakeInferenceModels(
	int obsSize,
	int numActions,
	const c10::Device& device,
	GGL::ModelSet& outModels) const
{
	outModels.Add(new GGL::ScriptedPolicyModel(
		"element-model.pt",
		obsSize,
		numActions,
		device,
		GGL::ScriptedPolicyInput::Flat));
}
