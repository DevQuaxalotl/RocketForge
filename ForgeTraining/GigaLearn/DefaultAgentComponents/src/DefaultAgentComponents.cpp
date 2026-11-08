#include <GigaLearnInfer/Agent.h>
#include <GigaLearnInfer/AgentComponentRegistry.h>

#include <RLGymCPP/ActionParsers/DefaultAction.h>
#include <RLGymCPP/ObsBuilders/AdvancedObs.h>
#include <RLGymCPP/ObsBuilders/AdvancedObsPadded.h>
#include <RLGymCPP/ObsBuilders/DefaultObs.h>
#include <RLGymCPP/ObsBuilders/DefaultObsPadded.h>

REGISTER_OBS_BUILDER("DefaultObsPadded", [](const Agent& agent) {
	return std::make_unique<RLGC::DefaultObsPadded>(agent.maxTeamSize);
})

REGISTER_OBS_BUILDER("DefaultObs", [](const Agent&) {
	return std::make_unique<RLGC::DefaultObs>();
})

REGISTER_OBS_BUILDER("AdvancedObs", [](const Agent&) {
	return std::make_unique<RLGC::AdvancedObs>();
})

REGISTER_OBS_BUILDER("AdvancedObsPadded", [](const Agent& agent) {
	return std::make_unique<RLGC::AdvancedObsPadded>(agent.maxTeamSize);
})

REGISTER_ACTION_PARSER("DefaultAction", [](const Agent&) {
	return std::make_unique<RLGC::DefaultAction>();
})
