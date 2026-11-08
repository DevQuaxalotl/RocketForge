#include "ParamTesting.h"
#include <StageQueue/StageQueueRunner.h>
#include <RocketSim/Engine.h>

#include "AgentModelPaths.h"
#include "ForgeTraining.h"
#include "Storage/BuildStorage.h"
#include "TrainingSetup/TrainingSetupSnapshot.h"

#include <GigaLearnCPP/TrainingStorage.h>

#include <algorithm>
#include <filesystem>
#include <set>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace RocketForgeTraining {
namespace {

	// Reserved label of the baseline model; test segments cannot reuse it.
	constexpr const char* BASELINE_LABEL = "Baseline";

	std::string Trim(const std::string& text)
	{
		const size_t first = text.find_first_not_of(" \t\r\n");
		if (first == std::string::npos)
			return {};

		const size_t last = text.find_last_not_of(" \t\r\n");
		return text.substr(first, last - first + 1);
	}

	std::string ConfigBranchName(const ParamTestingConfig& config)
	{
		std::string branchName = Trim(config.branchName);
		return branchName.empty() ? "param-test" : branchName;
	}

	std::string MakeRuntimeSettingsStem(const ParamTestingConfig& config)
	{
		std::string stem = BuildStorage::SanitizeFolderPart(config.baselineSegmentId);
		if (stem.empty())
			stem = "param-testing";
		return stem;
	}

	fs::path RuntimeSettingsPath(const ParamTestingConfig& config)
	{
		const fs::path root = fs::path(PROJECT_ROOT) / "out" / "param_testing";
		return root / (MakeRuntimeSettingsStem(config) + "_runtime.json");
	}

	void Report(
		const ParamTestingProgressCallback& progressCallback,
		int segmentIndex,
		int segmentCount,
		const std::string& label,
		const std::string& details = {})
	{
		if (!progressCallback)
			return;

		ParamTestingProgress progress = {};
		progress.segmentIndex = segmentIndex;
		progress.segmentCount = segmentCount;
		progress.label = label;
		progress.details = details;
		progressCallback(progress);
	}

	void ThrowIfCancelled(const ParamTestingCancelCallback& isCancelled)
	{
		if (isCancelled && isCancelled())
			throw std::runtime_error("Param Testing cancelled.");
	}

	fs::path LatestCheckpointPath(const BuildStorage& storage, const TrainingSegmentMetadata& segment)
	{
		return GGL::TrainingStorage::GetMostRecentCheckpointIn(storage.SegmentCheckpointsRoot(segment.folderName));
	}

	uint64_t CheckpointTimesteps(const fs::path& checkpointPath)
	{
		if (checkpointPath.empty())
			return 0;

		const std::string filename = checkpointPath.filename().string();
		if (!GGL::TrainingStorage::IsAllDigits(filename))
			return 0;

		return static_cast<uint64_t>(std::stoull(filename));
	}

	ModelBenchmarkSelectedModelSettings MakeBenchmarkModel(
		const std::string& agentName,
		const std::string& buildName,
		const std::string& displayName,
		const BuildStorage& storage,
		const TrainingSegmentMetadata& segment)
	{
		const fs::path checkpointPath = LatestCheckpointPath(storage, segment);
		if (checkpointPath.empty())
			throw std::runtime_error("Param Testing: segment has no saved checkpoint: " + segment.segmentId);

		ModelBenchmarkSelectedModelSettings model = {};
		model.agentName = agentName;
		model.buildName = buildName;
		model.stageName = segment.stageName;
		model.sourceLabel = displayName;
		model.versionId = segment.segmentId;
		model.checkpointPath = checkpointPath;
		model.trainingSetupPath = storage.SegmentRoot(segment.folderName) / RocketForge::TrainingSetupSnapshot::FILE_NAME;
		model.agentSetupSource = ModelBenchmarkAgentSetupSource::SavedSegment;
		model.latestTimesteps = CheckpointTimesteps(checkpointPath);
		model.notes = segment.notes;
		return model;
	}

	void ValidateConfig(const ParamTestingConfig& config)
	{
		if (Trim(config.agentName).empty())
			throw std::runtime_error("Param Testing requires an agent.");
		if (Trim(config.buildName).empty())
			throw std::runtime_error("Param Testing requires a build.");
		if (Trim(config.baselineStageName).empty())
			throw std::runtime_error("Param Testing requires a baseline stage.");
		if (Trim(config.baselineSegmentId).empty())
			throw std::runtime_error("Param Testing requires a baseline model.");
		if (config.segmentTimesteps == 0)
			throw std::runtime_error("Param Testing requires segment timesteps greater than zero.");
		if (Trim(config.branchName).empty())
			throw std::runtime_error("Param Testing requires a branch name.");
		if (config.segments.empty())
			throw std::runtime_error("Param Testing requires at least one test segment.");

		std::set<std::string> names = { BASELINE_LABEL };
		for (const ParamTestingSegmentConfig& segment : config.segments) {
			if (Trim(segment.label).empty())
				throw std::runtime_error("Param Testing segment names cannot be empty.");
			if (Trim(segment.stageName).empty())
				throw std::runtime_error("Param Testing segment \"" + segment.label + "\" requires a training stage.");
			if (!names.insert(segment.label).second)
				throw std::runtime_error("Param Testing segment names must be unique: " + segment.label);
		}
	}

	std::string MakeSegmentNotes(const ParamTestingConfig& config, const ParamTestingSegmentConfig& segment)
	{
		std::ostringstream out;
		if (!Trim(segment.versionNotes).empty())
			out << Trim(segment.versionNotes) << "\n\n";

		out << "Param Testing: " << segment.label;
		out << "\nBaseline: " << config.baselineSegmentId << " (" << config.baselineStageName << ")";
		out << "\nBranch: " << ConfigBranchName(config);

		if (!segment.parameterOverrides.empty()) {
			out << "\nOverrides:";
			for (const StageParameterOverride& override : segment.parameterOverrides) {
				out << "\n- ";
				if (override.kind == StageParameterOverrideKind::RewardWeight)
					out << "Reward / ";
				else
					out << override.group << " / ";
				out << override.name << " = " << override.value;
			}
		}

		return out.str();
	}

	ForgeSettings BuildSingleSegmentSettings(
		const ParamTestingConfig& config,
		const ParamTestingSegmentConfig& segment,
		uint64_t targetTotalTimesteps)
	{
		ForgeSettings settings = {};
		settings.agentName = config.agentName;
		settings.buildName = config.buildName;
		settings.globals.metricsEnabled = config.metricsEnabled;
		settings.globals.skillTestingEnabled = config.skillTestingEnabled;
		settings.globals.render = false;

		StageSettings stage = {};
		stage.stageName = segment.stageName;
		stage.targetTotalTimesteps = targetTotalTimesteps;
		stage.branchName = ConfigBranchName(config);
		stage.versionIncrement = segment.versionIncrement;
		stage.versionOverride = segment.versionOverride;
		stage.versionNotes = MakeSegmentNotes(config, segment);
		stage.parameterOverrides = segment.parameterOverrides;

		stage.bootstrap.mode = BootstrapConfig::Mode::LoadFromOtherStage;
		stage.bootstrap.branchMode = BootstrapConfig::BranchMode::CreateNew;
		stage.bootstrap.sourceStage = config.baselineStageName;
		stage.bootstrap.sourceVersionId = config.baselineSegmentId;
		settings.stages.push_back(std::move(stage));

		return settings;
	}

	ModelBenchmarkSettings BuildBenchmarkSettings(
		const ParamTestingConfig& config,
		const std::vector<ModelBenchmarkSelectedModelSettings>& selectedModels)
	{
		const int maxBenchmarkArenas = 500;

		ModelBenchmarkSettings settings = {};
		settings.numArenas = std::max(1, std::min(maxBenchmarkArenas, config.benchmarkGamesPerMode));
		settings.maxGamesPerMode = std::max(1, config.benchmarkGamesPerMode);
		settings.runUntilMaxGames = true;
		settings.collectOptionalStats = true;
		settings.selectedModels = selectedModels;

		return settings;
	}
}

std::vector<StageSettings> BuildParamTestingPreviewStages(const ParamTestingConfig& config)
{
	ValidateConfig(config);

	BuildStorage storage(ResolveAgentModelsFolder(config.agentName), config.buildName);
	std::optional<TrainingSegmentMetadata> baselineSegment = storage.LoadSegmentById(config.baselineSegmentId);
	if (!baselineSegment.has_value())
		throw std::runtime_error("Param Testing baseline segment was not found: " + config.baselineSegmentId);
	if (baselineSegment->stageName != config.baselineStageName)
		throw std::runtime_error("Param Testing baseline stage mismatch for " + config.baselineSegmentId);

	const fs::path baselineCheckpoint = LatestCheckpointPath(storage, baselineSegment.value());
	if (baselineCheckpoint.empty())
		throw std::runtime_error("Param Testing baseline has no saved checkpoint: " + config.baselineSegmentId);

	const uint64_t baselineTimesteps = CheckpointTimesteps(baselineCheckpoint);
	const uint64_t targetTotalTimesteps = baselineTimesteps + config.segmentTimesteps;

	std::vector<StageSettings> stages;
	for (const ParamTestingSegmentConfig& segment : config.segments) {
		ForgeSettings settings = BuildSingleSegmentSettings(config, segment, targetTotalTimesteps);
		if (!settings.stages.empty())
			stages.push_back(std::move(settings.stages.front()));
	}

	return stages;
}

std::vector<RocketSim::EngineId> ResolveParamTestingSimulation(const ParamTestingConfig& config)
{
	ForgeSettings settings;
	settings.agentName = config.agentName;
	settings.buildName = config.buildName;
	settings.stages = BuildParamTestingPreviewStages(config);

	return ResolveTrainingSimulation(settings);
}

ParamTestingResult RunParamTesting(
	const ParamTestingConfig& config,
	const ParamTestingProgressCallback& progressCallback,
	const ParamTestingCancelCallback& isCancelled)
{
	ValidateConfig(config);
	ThrowIfCancelled(isCancelled);

	for (RocketSim::EngineId engine : ResolveParamTestingSimulation(config))
		RocketSim::RequireEngine(engine);

	const int segmentCount = static_cast<int>(config.segments.size());
	const int preparingSegmentIndex = -1;
	Report(progressCallback, preparingSegmentIndex, segmentCount, "Preparing Param Testing", "Loading baseline metadata.");

	BuildStorage storage(ResolveAgentModelsFolder(config.agentName), config.buildName);
	std::optional<TrainingSegmentMetadata> baselineSegment = storage.LoadSegmentById(config.baselineSegmentId);
	if (!baselineSegment.has_value())
		throw std::runtime_error("Param Testing baseline segment was not found: " + config.baselineSegmentId);
	if (baselineSegment->stageName != config.baselineStageName)
		throw std::runtime_error("Param Testing baseline stage mismatch for " + config.baselineSegmentId);

	const fs::path baselineCheckpoint = LatestCheckpointPath(storage, baselineSegment.value());
	if (baselineCheckpoint.empty())
		throw std::runtime_error("Param Testing baseline has no saved checkpoint: " + config.baselineSegmentId);

	const uint64_t baselineTimesteps = CheckpointTimesteps(baselineCheckpoint);
	const uint64_t targetTotalTimesteps = baselineTimesteps + config.segmentTimesteps;

	ParamTestingResult result = {};
	result.selectedBenchmarkModels.push_back(MakeBenchmarkModel(
		config.agentName,
		config.buildName,
		BASELINE_LABEL,
		storage,
		baselineSegment.value()));

	const fs::path settingsPath = RuntimeSettingsPath(config);
	fs::create_directories(settingsPath.parent_path());

	for (int i = 0; i < segmentCount; i++) {
		ThrowIfCancelled(isCancelled);

		const ParamTestingSegmentConfig& segment = config.segments[i];
		Report(progressCallback, i, segmentCount, "Starting test segment", segment.label + " from " + config.baselineSegmentId);

		ForgeSettings runSettings = BuildSingleSegmentSettings(config, segment, targetTotalTimesteps);
		std::string createdSegmentId;

		StageEvents events = {};
		events.onStageStarted = [&](int, const std::string& stageName, const std::string& versionId) {
			createdSegmentId = versionId;

			if (progressCallback) {
				ParamTestingProgress progress = {};
				progress.segmentIndex = i;
				progress.segmentCount = segmentCount;
				progress.stageStarted = true;
				progress.stageName = stageName;
				progress.versionId = versionId;
				progress.label = "Training test segment";
				progress.details = segment.label + " -> " + versionId + " (" + stageName + ")";
				progressCallback(progress);
			}
		};

		events.onStageCompleted = [&](int) {
			if (progressCallback) {
				ParamTestingProgress progress = {};
				progress.segmentIndex = i;
				progress.segmentCount = segmentCount;
				progress.stageCompleted = true;
				progress.versionId = createdSegmentId;
				progress.label = "Completed test segment";
				progress.details = segment.label;
				progressCallback(progress);
			}
		};

		events.onStageSkipped = [&](int, const std::string& reason) {
			Report(progressCallback, i, segmentCount, "Skipped test segment", segment.label + ": " + reason);
		};

		RunResult runResult = RunTraining(runSettings, settingsPath, events);
		if (runResult.status == RunStatus::Stopped)
			throw std::runtime_error("Param Testing cancelled.");
		if (runResult.status == RunStatus::Error && runResult.unavailableEngine)
			throw RocketSim::EngineUnavailableError(*runResult.unavailableEngine, runResult.errorMessage);
		if (runResult.status == RunStatus::Error)
			throw std::runtime_error(runResult.errorMessage);
		if (createdSegmentId.empty())
			throw std::runtime_error("Param Testing segment did not produce a model: " + segment.label);

		std::optional<TrainingSegmentMetadata> completedSegment = storage.LoadSegmentById(createdSegmentId);
		if (!completedSegment.has_value())
			throw std::runtime_error("Param Testing could not reload completed segment: " + createdSegmentId);

		completedSegment->eligibleBenchmark = true;
		storage.SaveSegment(completedSegment.value());

		result.selectedBenchmarkModels.push_back(MakeBenchmarkModel(
			config.agentName,
			config.buildName,
			segment.label,
			storage,
			completedSegment.value()));
	}

	ThrowIfCancelled(isCancelled);

	ModelBenchmarkSettings benchmarkSettings = BuildBenchmarkSettings(config, result.selectedBenchmarkModels);
	result.benchmarkSettingsPath = GetDefaultModelBenchmarkSettingsPath();

	std::string saveError;
	if (!GGL::JsonSaveToFile(benchmarkSettings, result.benchmarkSettingsPath, &saveError))
		throw std::runtime_error("Param Testing failed to save Model Benchmark setup: " + saveError);

	Report(progressCallback, segmentCount, segmentCount, "Loaded models into Model Benchmarking", result.benchmarkSettingsPath.string());

	return result;
}
}
