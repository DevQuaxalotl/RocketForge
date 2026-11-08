#include "ForgeSettings.h"
#include "AgentModelPaths.h"

#include <nlohmann/json.hpp>
#include <ForgeDebugger.h>

#include <filesystem>
#include <algorithm>
#include <fstream>

namespace fs = std::filesystem;
using nlohmann::json;

// TODO: Unknown field preservation. Per design spec,
// unknown JSON fields should be preserved in memory and round-tripped on save so
// older binaries don't drop fields written by newer binaries. Current implementation
// silently drops unknown fields. Acceptable while there is one schema version, but
// must be added before the schema evolves to v2.

namespace RocketForgeTraining {

	static const GGL::EnumNames<BootstrapConfig::Mode> BOOTSTRAP_MODE_NAMES = { {
		{ BootstrapConfig::Mode::New,                "new" },
		{ BootstrapConfig::Mode::ContinueLatest,     "continueLatest" },
		{ BootstrapConfig::Mode::ContinueExisting,   "continueExisting" },
		{ BootstrapConfig::Mode::LoadFromOtherStage, "loadFromOtherStage" },
	} };

	static const GGL::EnumNames<BootstrapConfig::BranchMode> BOOTSTRAP_BRANCH_MODE_NAMES = { {
		{ BootstrapConfig::BranchMode::ContinueExisting, "continueExisting" },
		{ BootstrapConfig::BranchMode::ContinueSelected, "continueSelected" },
		{ BootstrapConfig::BranchMode::CreateNew,        "createNew" },
	} };

	static const GGL::EnumNames<VersionIncrement> VERSION_INCREMENT_NAMES = { {
		{ VersionIncrement::None,  "none" },
		{ VersionIncrement::Minor, "minor" },
		{ VersionIncrement::Major, "major" },
	} };

	static const GGL::EnumNames<StageParameterOverrideKind> PARAMETER_OVERRIDE_KIND_NAMES = { {
		{ StageParameterOverrideKind::RewardWeight,   "rewardWeight" },
		{ StageParameterOverrideKind::Hyperparameter, "hyperparameter" },
	} };

	static const GGL::EnumNames<ModelBenchmarkAgentSetupSource> AGENT_SETUP_SOURCE_NAMES = { {
		{ ModelBenchmarkAgentSetupSource::SavedSegment, "savedSegment" },
		{ ModelBenchmarkAgentSetupSource::CurrentAgent, "currentAgent" },
	} };

	void BootstrapConfig::Serialize(GGL::JsonArchive& ar) {
		ar.EnumField("mode", mode, BOOTSTRAP_MODE_NAMES);
		ar.EnumField("branchMode", branchMode, BOOTSTRAP_BRANCH_MODE_NAMES);
		ar.FieldNullIfEmpty("sourceStage", sourceStage);
		ar.FieldNullIfEmpty("sourceVersionId", sourceVersionId);
		ar.FieldNullIfEmpty("selectedExistingVersionId", selectedExistingVersionId);
	}

	void StageParameterOverride::Serialize(GGL::JsonArchive& ar) {
		ar.EnumField("kind", kind, PARAMETER_OVERRIDE_KIND_NAMES);
		ar.Field("group", group);
		ar.Field("name", name);
		ar.Field("value", value);
	}

	void StageSettings::Serialize(GGL::JsonArchive& ar) {
		ar.Field("stageName", stageName);
		ar.Field("targetTotalTimesteps", targetTotalTimesteps);
		ar.Field("branchName", branchName, std::string("main"));
		ar.Field("warmStartPolicy", warmStartPolicy);
		ar.Field("warmStartIterations", warmStartIterations);
		ar.Field("keepRunId", keepRunId);
		ar.Field("allowSegmentSetupMismatch", allowSegmentSetupMismatch);
		ar.EnumField("versionIncrement", versionIncrement, VERSION_INCREMENT_NAMES);
		ar.Field("versionOverride", versionOverride);
		ar.Field("versionNotes", versionNotes);
		ar.Field("staticOpponentsEnabled", staticOpponentsEnabled);
		ar.Field("bootstrap", bootstrap);
		ar.Field("parameterOverrides", parameterOverrides);

		ar.Field("createdSegmentId", createdSegmentId);

		// The old hand-written loader fell back to the legacy key whenever the resulting VALUE
		// was empty (not only when the primary key was absent) — match that exactly here rather
		// than at the shared FieldWithLegacy level, which other tasks rely on for the absent-key rule.
		if (ar.IsLoading() && createdSegmentId.empty())
			ar.Field("createdVersionId", createdSegmentId);

		// NOTE: agentSetupOverride is transient GUI state and is intentionally not serialized.
	}

	void GlobalSettings::Serialize(GGL::JsonArchive& ar) {
		ar.Field("metricsEnabled", metricsEnabled);
		ar.Field("skillTestingEnabled", skillTestingEnabled);
		ar.Field("pruneSourceCheckpoints", pruneSourceCheckpoints);
		ar.Field("persistSegmentSnapshotsDuringTraining", persistSegmentSnapshotsDuringTraining);
		ar.Field("performanceCoresOnly", performanceCoresOnly);
		ar.Field("render", render);
		ar.Field("debug", debug);
	}

	void ModelBenchmarkSelectedModelSettings::Serialize(GGL::JsonArchive& ar) {
		ar.Field("agentName", agentName);
		ar.Field("buildName", buildName, std::string(GGL::DEFAULT_BUILD_NAME));
		ar.Field("stageName", stageName);
		ar.Field("sourceLabel", sourceLabel);
		ar.Field("versionId", versionId);
		ar.Field("checkpointPath", checkpointPath);
		ar.Field("trainingSetupPath", trainingSetupPath);
		ar.EnumField("agentSetupSource", agentSetupSource, AGENT_SETUP_SOURCE_NAMES);
		ar.Field("primary", primary);
		ar.Field("included", included, true);
		ar.Field("resultsOnly", resultsOnly, false);
		ar.Field("libraryModelId", libraryModelId, int64_t{-1});
		ar.Field("fingerprint", fingerprint);
		ar.Field("agentSetupJson", agentSetupJson);
		ar.Field("lastTrainedAt", lastTrainedAt);
		ar.Field("colorIndex", colorIndex);
		ar.Field("latestTimesteps", latestTimesteps);
		ar.Field("notes", notes);
		ar.Field("skillRating", skillRating);
		ar.Field("hasSkillRating", hasSkillRating);
	}

	void ModelBenchmarkSettings::Serialize(GGL::JsonArchive& ar) {
		ar.Field("engineVersion", engineVersion, 2);
		ar.Field("includeOtherSimulatorResults", includeOtherSimulatorResults, false);
		ar.Field("fillMissingGames", fillMissingGames, true);
		ar.Field("allMatchups", allMatchups, true);
		ar.Field("matchSavedConditions", matchSavedConditions, true);
		ar.Field("reuseSelectedEngine", reuseSelectedEngine, true);
		ar.Field("reuseEngineVersions", reuseEngineVersions);

		std::map<std::string, std::vector<std::string>> environments;
		if (ar.IsSaving())
			for (const auto& [engine, keys] : reuseEnvironmentKeys)
				environments[std::to_string(engine)] = keys;

		ar.Field("reuseEnvironmentKeys", environments);

		if (ar.IsLoading()) {
			reuseEnvironmentKeys.clear();
			for (const auto& [engine, keys] : environments) {
				try {
					reuseEnvironmentKeys.emplace(std::stoi(engine), keys);
				} catch (...) {
				}
			}

			if (!ar.Contains("reuseSelectedEngine") && includeOtherSimulatorResults) {
				reuseSelectedEngine = false;
				reuseEngineVersions = { 2, 3 };
			}
		}

		if (ar.IsLoading()) {
			const bool selectedLegacy = std::find(reuseEngineVersions.begin(), reuseEngineVersions.end(), 0) != reuseEngineVersions.end();
			const bool selectedV2 = std::find(reuseEngineVersions.begin(), reuseEngineVersions.end(), 2) != reuseEngineVersions.end();
			if (selectedLegacy) {
				const auto legacy = reuseEnvironmentKeys.find(0);
				const auto v2 = reuseEnvironmentKeys.find(2);

				// A missing entry selects all environments, so the merged v2 entry must stay missing too.
				if (legacy == reuseEnvironmentKeys.end() || (selectedV2 && v2 == reuseEnvironmentKeys.end())) {
					reuseEnvironmentKeys.erase(2);
				} else {
					std::vector<std::string> keys = legacy->second;
					for (std::string& key : keys)
						if (key == "legacy-major:0,")
							key = "legacy-major:2,";

					if (selectedV2)
						keys.insert(keys.end(), v2->second.begin(), v2->second.end());

					std::sort(keys.begin(), keys.end());
					keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
					reuseEnvironmentKeys[2] = std::move(keys);
				}

				for (int& engine : reuseEngineVersions)
					if (engine == 0)
						engine = 2;

				std::sort(reuseEngineVersions.begin(), reuseEngineVersions.end());
				reuseEngineVersions.erase(std::unique(reuseEngineVersions.begin(), reuseEngineVersions.end()), reuseEngineVersions.end());
			}

			reuseEnvironmentKeys.erase(0);
		}

		ar.Field("ratingTargetsOnly", ratingTargetsOnly, false);
		ar.Field("presetName", presetName);
		ar.Field("stageFilter", stageFilter);
		ar.Field("hideImportedModels", hideImportedModels);
		ar.Field("libraryPath", libraryPath);
		ar.Field("numArenas", numArenas);
		ar.Field("maxGamesPerMode", maxGamesPerMode);
		ar.Field("run1v1", run1v1);
		ar.Field("run2v2", run2v2);
		ar.Field("run3v3", run3v3);
		ar.Field("stochasticKickoff", stochasticKickoff, false);
		ar.Field("kickoffTemperature", kickoffTemperature, 0.70f);
		ar.Field("betterConfidence", betterConfidence);
		ar.Field("stopWhenRankingKnown", stopWhenRankingKnown);
		ar.Field("eloRangeConfidence", eloRangeConfidence);
		ar.Field("maxEloUncertaintyRange", maxEloUncertaintyRange);
		ar.Field("runUntilMaxGames", runUntilMaxGames);
		ar.Field("collectOptionalStats", collectOptionalStats);
		ar.Field("seed", seed);
		ar.Field("selectedModels", selectedModels);
	}

	std::string SerializeModelBenchmarkSettings(const ModelBenchmarkSettings& settings) {
		json value;
		GGL::JsonSave(settings, value);

		return value.dump();
	}

	ModelBenchmarkSettingsLoadResult DeserializeModelBenchmarkSettings(const std::string& text) {
		ModelBenchmarkSettingsLoadResult result;
		try {
			const json value = json::parse(text);
			result.errors = GGL::JsonLoad(value, result.value);
			if (!value.contains("reuseSelectedEngine") && result.value.includeOtherSimulatorResults) {
				result.value.reuseSelectedEngine = false;
				result.value.reuseEngineVersions = { 2, 3 };
			}

			result.ok = result.errors.empty();
		} catch (const std::exception& error) {
			result.errors.push_back(error.what());
		}

		return result;
	}

	void ForgeSettings::Serialize(GGL::JsonArchive& ar) {
		ar.Field("schemaVersion", schemaVersion);
		ar.Field("agentName", agentName);
		ar.Field("buildName", buildName, std::string(GGL::DEFAULT_BUILD_NAME));
		ar.Field("globals", globals);
		ar.Field("stages", stages);
	}

	ForgeSettingsLoadResult LoadForgeSettings(const fs::path& path) {
		return GGL::JsonLoadFromFileWithSchema<ForgeSettings>(path, "schemaVersion", kForgeSettingsSchemaVersion);
	}

	fs::path GetDefaultModelBenchmarkSettingsPath() {
		return fs::path(PROJECT_ROOT) / "forge_model_benchmark_settings.json";
	}

	ModelBenchmarkSettingsLoadResult LoadModelBenchmarkSettings(const fs::path& path) {
		// No schema key in this file; field-level errors were always ignored here, so a
		// successfully parsed file loads ok (errors are carried for diagnostics only).
		ModelBenchmarkSettingsLoadResult result;
		result.ok = GGL::JsonLoadFromFile(path, result.value, result.errors);
		return result;
	}

	fs::path GetDefaultForgeSettingsPath(const std::string& agentName) {
		fs::path modelsRoot = ResolveAgentModelsFolder(agentName);
		// modelsRoot is .../Bots/Agents/<agent>/Models — the settings file lives one level above.
		return modelsRoot.parent_path() / "forge_settings.json";
	}

	fs::path GetExampleForgeSettingsPath(const std::string& agentName) {
		fs::path modelsRoot = ResolveAgentModelsFolder(agentName);
		return modelsRoot.parent_path() / "forge_settings.example.json";
	}

	static fs::path GetAppStatePath() {
		return fs::path(PROJECT_ROOT) / "forge_app_state.json";
	}

	std::string LoadLastSelectedAgent() {
		const fs::path path = GetAppStatePath();
		if (!fs::exists(path))
			return {};

		std::ifstream fIn(path);
		if (!fIn.good())
			return {};

		json j;
		try {
			fIn >> j;
		} catch (...) {
			return {};
		}

		return j.value("lastSelectedAgent", std::string());
	}

	void SaveLastSelectedAgent(const std::string& agentName) {
		const fs::path path = GetAppStatePath();
		try {
			fs::create_directories(path.parent_path());
			std::ofstream fOut(path);
			if (!fOut.good())
				return;
			json j;
			j["lastSelectedAgent"] = agentName;
			fOut << j.dump(2);
		} catch (...) {}
	}

	ForgeSettings MakeDefaultForgeSettings(const std::string& agentName, const std::string& stageName) {
		ForgeSettings s;
		s.schemaVersion = kForgeSettingsSchemaVersion;
		s.agentName     = agentName;
		s.buildName     = GGL::DEFAULT_BUILD_NAME;
		StageSettings st;
		st.stageName            = stageName;
		st.targetTotalTimesteps = UINT64_MAX;  // "run until stopped" sentinel — no timestep cap.
		st.bootstrap.mode       = BootstrapConfig::Mode::New;
		s.stages.push_back(st);
		return s;
	}
}
