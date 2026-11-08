#include "AgentSetup.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>

namespace RocketForge::SimulationAnalysis
{
	AgentSetupSnapshot LoadAgentSetup(const std::filesystem::path& path)
	{
		// Accept either a full training_setup.json or a smaller agent_setup JSON.
		std::ifstream in(path);
		if (!in.good())
			throw std::runtime_error("failed to open setup path: " + path.string());

		nlohmann::json root = nlohmann::json::parse(in, nullptr, false);
		if (root.is_discarded())
			throw std::runtime_error("failed to parse setup JSON: " + path.string());

		const bool isTrainingSetup = root.contains("agent_setup");
		const nlohmann::json& setupJson = isTrainingSetup ? root["agent_setup"] : root;
		std::optional<AgentSetupSnapshot> setup = AgentSetupSnapshotFromJson(setupJson);
		if (!setup.has_value())
			throw std::runtime_error("setup JSON does not contain a valid agent_setup object");

		// A training setup without per-agent engines inherits the engine it recorded at the top level.
		int recordedEngine = 0;
		if (isTrainingSetup && root.contains("engine_version")) {
			if (!root["engine_version"].is_number_integer())
				throw std::runtime_error("engine_version must be an integer in " + path.string());
			recordedEngine = root["engine_version"].get<int>();
		}
		ResolveAgentSimulation(setup.value(), recordedEngine);
		return setup.value();
	}
}