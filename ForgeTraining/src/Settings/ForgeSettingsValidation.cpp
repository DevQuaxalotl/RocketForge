#include "ForgeSettings.h"
#include "StageRegistry.h"

#include <GigaLearnInfer/AgentRegistry.h>
#include <GigaLearnCPP/SegmentVersioning.h>

#include <algorithm>
#include <sstream>

namespace RocketForgeTraining {

	static bool AgentExists(const std::string& name) {
		auto agents = AgentRegistry::GetAvailableAgents();
		return std::find(agents.begin(), agents.end(), name) != agents.end();
	}

	static bool StageExists(const std::string& name) {
		auto stages = RocketForge::StageRegistry::GetAvailableStages();
		return std::find(stages.begin(), stages.end(), name) != stages.end();
	}

	std::vector<std::string> ValidateForgeSettings(const ForgeSettings& settings) {
		std::vector<std::string> errors;

		if (settings.schemaVersion != kForgeSettingsSchemaVersion) {
			std::ostringstream oss;
			oss << "schemaVersion " << settings.schemaVersion
				<< " is not supported (expected " << kForgeSettingsSchemaVersion << ")";
			errors.push_back(oss.str());
		}

		if (settings.agentName.empty()) {
			errors.push_back("agentName must not be empty.");
		} else if (!AgentExists(settings.agentName)) {
			errors.push_back("Unknown agent: " + settings.agentName);
		} else {
			std::unique_ptr<Agent> agent = AgentRegistry::Create(settings.agentName);
			if (agent && agent->inferenceOnly) {
				errors.push_back(
					"Agent " + settings.agentName
					+ " is inference-only and cannot be selected as the training learner.");
			}
		}

		if (settings.buildName.empty())
			errors.push_back("buildName must not be empty.");

		if (settings.stages.empty()) {
			errors.push_back("At least one stage is required.");
			return errors;
		}

		uint64_t prevTarget = 0;
		for (size_t i = 0; i < settings.stages.size(); ++i) {
			const auto& s = settings.stages[i];
			std::ostringstream prefix;
			prefix << "Stage " << (i + 1) << " (" << (s.stageName.empty() ? "<unnamed>" : s.stageName) << "): ";

			if (s.stageName.empty()) {
				errors.push_back(prefix.str() + "stageName must not be empty.");
			} else if (!StageExists(s.stageName)) {
				errors.push_back(prefix.str() + "unknown stage.");
			}

			if (s.targetTotalTimesteps == 0) {
				errors.push_back(prefix.str() + "targetTotalTimesteps must be > 0.");
			} else if (i > 0 && s.targetTotalTimesteps <= prevTarget) {
				std::ostringstream oss;
				oss << prefix.str() << "targetTotalTimesteps (" << s.targetTotalTimesteps
					<< ") must be greater than previous stage's target (" << prevTarget << ").";
				errors.push_back(oss.str());
			}
			prevTarget = s.targetTotalTimesteps;

			if (s.warmStartIterations < 0)
				errors.push_back(prefix.str() + "warmStartIterations must be >= 0.");

			if (i == 0) {
				switch (s.bootstrap.mode) {
					case BootstrapConfig::Mode::New:
						break;
					case BootstrapConfig::Mode::ContinueLatest:
						break;
					case BootstrapConfig::Mode::ContinueExisting:
						if (s.bootstrap.selectedExistingVersionId.empty())
							errors.push_back(prefix.str() + "bootstrap.mode=continueExisting requires selectedExistingVersionId.");
						break;
					case BootstrapConfig::Mode::LoadFromOtherStage:
						if (s.bootstrap.sourceStage.empty() || s.bootstrap.sourceVersionId.empty())
							errors.push_back(prefix.str() + "bootstrap.mode=loadFromOtherStage requires sourceStage and sourceVersionId.");
						break;
				}

			}

			if ((s.bootstrap.branchMode == BootstrapConfig::BranchMode::CreateNew ||
				 s.bootstrap.branchMode == BootstrapConfig::BranchMode::ContinueSelected) &&
				s.branchName.empty())
				errors.push_back(prefix.str() + "selecting or creating a branch requires branchName.");
		}

		return errors;
	}
}
