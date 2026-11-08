#pragma once

#include <GigaLearnInfer/AgentSetupSnapshot.h>

#include <filesystem>

namespace RocketForge::SimulationAnalysis
{
	/** Loads agent setup metadata from training_setup.json or a raw agent_setup JSON file. **/
	RocketForge::AgentSetupSnapshot LoadAgentSetup(const std::filesystem::path& path);
}