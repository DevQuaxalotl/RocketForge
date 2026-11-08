#pragma once

#include "ModelBenchmark/ModelBenchmark.h"
#include "Settings/ForgeSettings.h"

#include <RocketSim/Engine.h>

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace RocketForgeTraining {

	/** One segment tested from the shared baseline model. */
	struct ParamTestingSegmentConfig {
		std::string label;
		std::string stageName;
		VersionIncrement versionIncrement = VersionIncrement::Minor;
		std::optional<GGL::SegmentVersionNumber> versionOverride;
		std::string versionNotes;
		std::vector<StageParameterOverride> parameterOverrides;
	};

	/** Top-level Param Testing run configuration. */
	struct ParamTestingConfig {
		std::string agentName;
		std::string buildName;
		std::string baselineStageName;
		std::string baselineSegmentId;
		std::string branchName = "param-test";
		uint64_t segmentTimesteps = 0;
		bool metricsEnabled = true;
		bool skillTestingEnabled = false;
		bool runBenchmark = false;
		int benchmarkGamesPerMode = 1000;
		std::vector<ParamTestingSegmentConfig> segments;
	};

	/** Progress update reported while Param Testing runs. */
	struct ParamTestingProgress {
		int segmentIndex = -1;
		int segmentCount = 0;
		bool benchmark = false;
		int benchmarkFinishedGames = 0;
		int benchmarkMaxGames = 0;
		bool stageStarted = false;
		bool stageCompleted = false;
		std::string stageName;
		std::string versionId;
		std::string label;
		std::string details;
	};

	/** Complete Param Testing result, including any benchmark handoff/result. */
	struct ParamTestingResult {
		std::vector<ModelBenchmarkSelectedModelSettings> selectedBenchmarkModels;
		std::filesystem::path benchmarkSettingsPath;
		bool benchmarkRan = false;
		ModelBenchmarkResult benchmarkResult = {};
	};

	using ParamTestingProgressCallback = std::function<void(const ParamTestingProgress&)>;
	using ParamTestingCancelCallback = std::function<bool()>;

	/** Builds the StageSettings rows the GUI can show while Param Testing trains. */
	std::vector<StageSettings> BuildParamTestingPreviewStages(const ParamTestingConfig& config);

	/** Resolves each test segment's training engine without starting a learner or creating a segment. */
	std::vector<RocketSim::EngineId> ResolveParamTestingSimulation(const ParamTestingConfig& config);

	/** Runs each configured test segment from the same baseline and updates Model Benchmarking setup. */
	ParamTestingResult RunParamTesting(
		const ParamTestingConfig& config,
		const ParamTestingProgressCallback& progressCallback = {},
		const ParamTestingCancelCallback& isCancelled = {});
}
