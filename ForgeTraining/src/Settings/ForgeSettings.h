#pragma once

#include "ModelBenchmark/ModelBenchmark.h"
#include "TrainingSetup/TrainingParameterOverride.h"
#include <GigaLearnInfer/AgentSetupSnapshot.h>
#include <GigaLearnInfer/JsonArchive.h>

#include <GigaLearnCPP/SegmentVersioning.h>
#include <DebugSettings.h>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <map>
#include <string>
#include <vector>

namespace RocketForgeTraining {

	/** Session-wide settings shared across all stages in a training run. */
	struct GlobalSettings {
		bool metricsEnabled         = true;
		bool skillTestingEnabled    = true;
		bool render                 = false;
		bool pruneSourceCheckpoints = false;
		bool persistSegmentSnapshotsDuringTraining = false;

		/** Windows hybrid CPUs: run only on the performance cores with power throttling off. Applied once per process when training starts; turning it off needs a restart. */
		bool performanceCoresOnly   = false;
		ForgeDebug::DebugSettings debug;

		/** Serializes all fields to/from JSON via the shared JsonArchive. */
		void Serialize(GGL::JsonArchive& ar);
	};

	/** First-stage-only bootstrap configuration. Ignored for stages 1..N. */
	struct BootstrapConfig {
		enum class Mode {
			New,
			ContinueLatest,
			ContinueExisting,
			LoadFromOtherStage,
		};

		enum class BranchMode {
			ContinueExisting,
			ContinueSelected,
			CreateNew,
		};

		Mode mode = Mode::New;
		BranchMode branchMode = BranchMode::ContinueExisting;

		// ContinueExisting:
		std::string selectedExistingVersionId;

		// LoadFromOtherStage:
		std::string sourceStage;
		std::string sourceVersionId;

		/** Serializes all fields to/from JSON via the shared JsonArchive. */
		void Serialize(GGL::JsonArchive& ar);
	};

	enum class VersionIncrement {
		None,
		Minor,
		Major,
	};

	/** The schema version this binary writes/expects. */
	constexpr int kForgeSettingsSchemaVersion = 1;

	/** All settings for a single stage in the queue. */
	struct StageSettings {
		std::string stageName;
		uint64_t    targetTotalTimesteps = 0;
		std::string branchName = "main";

		bool warmStartPolicy     = true;
		int  warmStartIterations = 20;
		bool keepRunId           = false;
		bool allowSegmentSetupMismatch = false;

		VersionIncrement                 versionIncrement = VersionIncrement::None;
		std::optional<GGL::SegmentVersionNumber> versionOverride;
		std::string                       versionNotes;
		bool                              staticOpponentsEnabled = false;

		/** Only honored for stages[0]; stages 1..N bootstrap from the previous segment. */
		BootstrapConfig bootstrap;

		/** Named reward/hyperparameter overrides applied after this stage builds defaults. **/
		std::vector<StageParameterOverride> parameterOverrides;

		/** Empty until the runner first creates a Training Segment for this queue entry. */
		std::string     createdSegmentId;

		/** Transient only: restored per-segment Agent setup selected by the GUI preflight. **/
		std::optional<RocketForge::AgentSetupSnapshot> agentSetupOverride;

		/** Serializes all fields to/from JSON via the shared JsonArchive. */
		void Serialize(GGL::JsonArchive& ar);
	};

	/** One selected model in the persisted Model Benchmarking setup. */
	struct ModelBenchmarkSelectedModelSettings {
		std::string agentName;
		std::string buildName;
		std::string stageName;
		std::string sourceLabel;
		std::string versionId;
		std::filesystem::path checkpointPath;
		std::filesystem::path trainingSetupPath;
		ModelBenchmarkAgentSetupSource agentSetupSource = ModelBenchmarkAgentSetupSource::SavedSegment;
		/** Primary models play every selected model; non-primary models only play primary models. **/
		bool primary = true;
		bool included = true;
		bool resultsOnly = false;

		/** Exact saved snapshot, scoped to the selected library; empty for untested models. **/
		int64_t libraryModelId = -1;
		std::string fingerprint;
		std::string agentSetupJson;

		/** Stable ForgeGUI palette index used to identify this model across benchmark tables. **/
		int colorIndex = -1;
		uint64_t latestTimesteps = 0;
		std::string lastTrainedAt;
		std::string notes;
		double skillRating = 0.0;
		bool hasSkillRating = false;
		void Serialize(GGL::JsonArchive& ar);
	};

	/** Persisted GUI setup for Model Benchmarking. */
	struct ModelBenchmarkSettings {
		/** Simulator selected for GUI benchmark runs. **/
		int engineVersion = 2;

		/** Legacy selection migrated into shared compatibility controls. **/
		bool includeOtherSimulatorResults = false;
		std::string stageFilter;
		bool hideImportedModels = false;

		/** Last SQLite benchmark library selected by the GUI. **/
		std::filesystem::path libraryPath;

		int numArenas = 500;
		int maxGamesPerMode = 500;
		bool fillMissingGames = true;
		bool allMatchups = true;
		bool matchSavedConditions = true;

		/** Follow the selected engine unless the user explicitly chooses reusable simulators. **/
		bool reuseSelectedEngine = true;
		std::vector<int> reuseEngineVersions;

		/** Missing engine entry accepts all environments, including future ones; an empty list accepts none. **/
		std::map<int, std::vector<std::string>> reuseEnvironmentKeys;
		bool ratingTargetsOnly = false;
		std::string presetName;
		bool run1v1 = true;
		bool run2v2 = true;
		bool run3v3 = true;

		/** Sample actions until the first car-ball touch, then use deterministic inference. **/
		bool stochasticKickoff = false;
		float kickoffTemperature = 0.70f;

		float betterConfidence = 0.95f;
		bool stopWhenRankingKnown = false;
		float eloRangeConfidence = 0.95f;
		float maxEloUncertaintyRange = 50.0f;
		bool runUntilMaxGames = true;
		bool collectOptionalStats = true;
		/** Reproducible benchmark seed. Zero selects a new seed automatically. **/
		uint64_t seed = 0;

		std::vector<ModelBenchmarkSelectedModelSettings> selectedModels;
		void Serialize(GGL::JsonArchive& ar);
	};

	/** Top-level settings file content. */
	struct ForgeSettings {
		int                        schemaVersion = kForgeSettingsSchemaVersion;
		std::string                agentName;
		std::string                buildName = GGL::DEFAULT_BUILD_NAME;
		GlobalSettings             globals;
		std::vector<StageSettings> stages;

		/** Serializes all fields to/from JSON via the shared JsonArchive. */
		void Serialize(GGL::JsonArchive& ar);
	};

	/** Result of loading a settings file: ok with 'value' populated, or 'errors'. */
	using ForgeSettingsLoadResult = GGL::JsonFileLoadResult<ForgeSettings>;

	/** Result of loading the shared Model Benchmarking setup file: ok with 'value' populated, or 'errors'. */
	using ModelBenchmarkSettingsLoadResult = GGL::JsonFileLoadResult<ModelBenchmarkSettings>;

	/** Loads ForgeSettings from a JSON file. Missing file returns ok=false with a single descriptive error. */
	ForgeSettingsLoadResult LoadForgeSettings(const std::filesystem::path& path);

	/** Returns the shared Model Benchmarking setup path at the project root. */
	std::filesystem::path GetDefaultModelBenchmarkSettingsPath();

	/** Loads the shared Model Benchmarking setup. Missing file returns ok=false with a descriptive error. */
	ModelBenchmarkSettingsLoadResult LoadModelBenchmarkSettings(const std::filesystem::path& path);
	std::string SerializeModelBenchmarkSettings(const ModelBenchmarkSettings& settings);
	ModelBenchmarkSettingsLoadResult DeserializeModelBenchmarkSettings(const std::string& text);

	/** Returns the canonical default file path for the given agent: <agentRoot>/forge_settings.json. */
	std::filesystem::path GetDefaultForgeSettingsPath(const std::string& agentName);

	/** Returns the canonical example file path for the given agent: <agentRoot>/forge_settings.example.json. */
	std::filesystem::path GetExampleForgeSettingsPath(const std::string& agentName);

	/** Returns a default ForgeSettings populated with one new-from-scratch stage. */
	ForgeSettings MakeDefaultForgeSettings(const std::string& agentName, const std::string& stageName);

	/** Validates the ForgeSettings against agent/stage registries and structural rules.
	    Returns an empty vector when valid; otherwise a list of human-readable error strings. */
	std::vector<std::string> ValidateForgeSettings(const ForgeSettings& settings);

	/** Returns the last selected agent name persisted in the app state file, or empty if none. */
	std::string LoadLastSelectedAgent();

	/** Persists the selected agent name to the app state file. */
	void SaveLastSelectedAgent(const std::string& agentName);
}
