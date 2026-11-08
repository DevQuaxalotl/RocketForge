#include "ModelBenchmarkRoster.h"

#include "AgentModelPaths.h"
#include "CheckpointFingerprint.h"
#include "ModelBenchmarkExport.h"
#include "Storage/BuildStorage.h"
#include "Storage/ImportedModelStorage.h"
#include "TrainingSetup/TrainingSetupSnapshot.h"
#include "TrainingSetup/VersionManagement.h"

#include <GigaLearnCPP/SegmentVersioning.h>
#include <GigaLearnCPP/TrainingStorage.h>

#include <algorithm>
#include <cctype>
#include <set>

using namespace std;
using namespace RocketForgeTraining;

namespace fs = std::filesystem;

namespace {

	const char* const IMPORTED_SOURCE_LABEL = "Imported Model";
	const char* const EXTERNAL_SOURCE_LABEL = "External";

	struct ResolvedEntry {
		fs::path checkpointPath;
		fs::path trainingSetupPath;
		ModelBenchmarkExportModel exportModel;

		/** True when the entry's own checkpointPath was used, so its saved fingerprint and agent setup still describe it. */
		bool usedOwnPath = false;
	};

	/** The GUI marks an imported model by its source label, and gives it the same text as its build name. */
	bool IsImported(const ModelBenchmarkSelectedModelSettings& entry)
	{
		return entry.sourceLabel == IMPORTED_SOURCE_LABEL || entry.buildName == IMPORTED_SOURCE_LABEL;
	}

	string NormalizedLowercase(const fs::path& path)
	{
		string text = path.lexically_normal().generic_string();
		transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return static_cast<char>(tolower(c)); });

		return text;
	}

	bool HasSavedAgentSetup(const fs::path& setupPath)
	{
		error_code existsError;
		if (setupPath.empty() || !fs::exists(setupPath, existsError))
			return false;

		const optional<RocketForge::TrainingSetupSnapshot> snapshot = RocketForge::LoadTrainingSetupSnapshot(setupPath);

		return snapshot.has_value() && snapshot->agentSetup.has_value();
	}

	/** Mirrors the GUI's choice: a model without a loadable saved setup runs on the current agent. */
	ModelBenchmarkAgentSetupSource ChooseAgentSetupSource(const ModelBenchmarkSelectedModelSettings& entry, const ResolvedEntry& resolved)
	{
		if (entry.agentSetupSource == ModelBenchmarkAgentSetupSource::CurrentAgent)
			return entry.agentSetupSource;

		const bool hasCapturedSetup = resolved.usedOwnPath && !entry.agentSetupJson.empty();
		if (hasCapturedSetup || HasSavedAgentSetup(resolved.trainingSetupPath))
			return ModelBenchmarkAgentSetupSource::SavedSegment;

		return ModelBenchmarkAgentSetupSource::CurrentAgent;
	}

	string DescribeEntry(size_t index, const ModelBenchmarkSelectedModelSettings& entry)
	{
		string text = "model " + to_string(index + 1) + " (" + entry.agentName;
		if (!entry.buildName.empty())
			text += "/" + entry.buildName;

		return text + "/" + entry.versionId + ")";
	}

	ModelBenchmarkExportModel MakeExportModel(const ModelBenchmarkSelectedModelSettings& entry)
	{
		ModelBenchmarkExportModel model;
		model.agentName = entry.agentName;
		model.buildName = entry.buildName;
		model.stageName = entry.stageName;
		model.sourceLabel = entry.sourceLabel;
		model.lastTrainedAt = entry.lastTrainedAt;
		model.latestTimesteps = entry.latestTimesteps;
		model.primary = entry.primary;

		return model;
	}

	/** Fills the checkpoint from a hand-written entry's names alone. Returns an error message, or empty on success. */
	string ResolveFromStorage(const ModelBenchmarkSelectedModelSettings& entry, ResolvedEntry& resolved)
	{
		if (entry.sourceLabel != EXTERNAL_SOURCE_LABEL && (entry.agentName.empty() || entry.versionId.empty()))
			return "needs a checkpointPath, or an agentName and versionId to find one";

		if (entry.sourceLabel == EXTERNAL_SOURCE_LABEL)
			return "external model folder not found: " + entry.checkpointPath.string();

		if (IsImported(entry)) {
			const fs::path importedRoot = ResolveImportedModelRoot(entry.agentName, entry.versionId);
			const fs::path modelFolder = GGL::TrainingStorage::ResolveModelFolder(importedRoot);
			if (modelFolder.empty()) {
				const string listCommand = "`ForgeConsoleTrainer list imported-models --agent " + entry.agentName + "`";
				return "imported model '" + entry.versionId + "' not found. Run " + listCommand;
			}

			resolved.checkpointPath = modelFolder;
			resolved.exportModel.buildName = IMPORTED_SOURCE_LABEL;

			return {};
		}

		const string buildName = entry.buildName.empty() ? string(GGL::DEFAULT_BUILD_NAME) : entry.buildName;
		const bool includeInactive = true;
		const vector<GGL::SegmentVersionSummary> segments = ListBuildVersions(entry.agentName, buildName, {}, includeInactive);
		const auto found = find_if(segments.begin(), segments.end(), [&](const auto& segment) { return segment.id == entry.versionId; });

		const string listCommand = "`ForgeConsoleTrainer list segments --agent " + entry.agentName + " --build " + buildName + "`";
		if (found == segments.end())
			return "segment '" + entry.versionId + "' not found. Run " + listCommand;

		const fs::path segmentRoot = found->rootPath;
		const optional<ResolvedCheckpoint> checkpoint = BuildStorage::LatestSegmentCheckpointIn(segmentRoot);
		if (!checkpoint.has_value())
			return "segment '" + entry.versionId + "' has no checkpoints. Run " + listCommand;

		const fs::path setupPath = segmentRoot / RocketForge::TrainingSetupSnapshot::FILE_NAME;
		error_code existsError;

		resolved.checkpointPath = checkpoint->folder;
		resolved.trainingSetupPath = fs::exists(setupPath, existsError) ? setupPath : fs::path();

		ModelBenchmarkExportModel& model = resolved.exportModel;
		model.buildName = buildName;
		if (model.stageName.empty())
			model.stageName = found->stageName;
		model.branchName = found->branchName;
		model.lastTrainedAt = found->lastTrainedAt;
		model.latestTimesteps = checkpoint->timesteps;

		return {};
	}

	void WarnAboutLibraryOnlyOptions(const ModelBenchmarkSettings& settings, vector<string>& warnings)
	{
		const ModelBenchmarkSettings defaults;

		const pair<const char*, bool> ignoredOptions[] = {
			{ "fillMissingGames", settings.fillMissingGames != defaults.fillMissingGames },
			{ "matchSavedConditions", settings.matchSavedConditions != defaults.matchSavedConditions },
			// The GUI always saves its default library path, so only a different one means the user chose it.
			{ "libraryPath", !settings.libraryPath.empty() && NormalizedLowercase(settings.libraryPath) != NormalizedLowercase(DefaultModelBenchmarkLibraryPath()) },
			{ "reuseSelectedEngine", settings.reuseSelectedEngine != defaults.reuseSelectedEngine },
			{ "reuseEngineVersions", settings.reuseEngineVersions != defaults.reuseEngineVersions },
			{ "reuseEnvironmentKeys", settings.reuseEnvironmentKeys != defaults.reuseEnvironmentKeys },
			{ "ratingTargetsOnly", settings.ratingTargetsOnly != defaults.ratingTargetsOnly },
		};

		for (const auto& [field, isSet] : ignoredOptions) {
			if (isSet)
				warnings.push_back(string(field) + " needs the GUI's results library and is ignored by the console benchmark");
		}
	}

	ModelBenchmarkConfig MakeConfig(const ModelBenchmarkSettings& settings)
	{
		ModelBenchmarkConfig config;
		config.engine = RocketSim::EngineFromVersion(settings.engineVersion);
		config.numArenas = settings.numArenas;
		config.maxGamesPerMode = settings.maxGamesPerMode;
		config.seed = settings.seed;
		config.run1v1 = settings.run1v1;
		config.run2v2 = settings.run2v2;
		config.run3v3 = settings.run3v3;
		config.stochasticKickoff = settings.stochasticKickoff;
		config.kickoffTemperature = settings.kickoffTemperature;
		config.betterConfidence = settings.betterConfidence;
		config.stopWhenRankingKnown = settings.stopWhenRankingKnown;
		config.eloRangeConfidence = settings.eloRangeConfidence;
		config.maxEloUncertaintyRange = settings.maxEloUncertaintyRange;
		config.runUntilMaxGames = settings.runUntilMaxGames;
		config.collectOptionalStats = settings.collectOptionalStats;

		return config;
	}

}

ModelBenchmarkRoster RocketForgeTraining::ResolveModelBenchmarkRoster(const ModelBenchmarkSettings& settings, bool allPrimary)
{
	ModelBenchmarkRoster roster;
	roster.config = MakeConfig(settings);

	WarnAboutLibraryOnlyOptions(settings, roster.warnings);

	set<string> displayNames;

	for (size_t i = 0; i < settings.selectedModels.size(); i++) {
		const ModelBenchmarkSelectedModelSettings& entry = settings.selectedModels[i];
		if (!entry.included || entry.resultsOnly)
			continue;

		ResolvedEntry resolved;
		resolved.exportModel = MakeExportModel(entry);

		// A path the GUI wrote on another machine can be stale, so the entry's names still get a chance to resolve it.
		error_code existsError;
		if (!entry.checkpointPath.empty() && fs::exists(entry.checkpointPath, existsError)) {
			resolved.usedOwnPath = true;
			resolved.checkpointPath = entry.checkpointPath;
			resolved.trainingSetupPath = entry.trainingSetupPath;
		} else {
			const string problem = ResolveFromStorage(entry, resolved);
			if (!problem.empty()) {
				roster.errors.push_back(DescribeEntry(i, entry) + ": " + problem);
				continue;
			}

			if (!entry.checkpointPath.empty()) {
				roster.warnings.push_back(DescribeEntry(i, entry) + ": checkpointPath " + entry.checkpointPath.string() +
					" not found, using " + resolved.checkpointPath.string());
			}
		}

		const bool primary = allPrimary ? true : entry.primary;
		resolved.exportModel.primary = primary;

		const bool useSavedFingerprint = resolved.usedOwnPath && !entry.fingerprint.empty();
		resolved.exportModel.fingerprint =
			useSavedFingerprint ? entry.fingerprint : FingerprintCheckpoint(resolved.checkpointPath, resolved.trainingSetupPath);

		// Matches the GUI's "agent / build / version" so a workbook imports under the same names.
		const string idText = entry.versionId.empty() ? resolved.checkpointPath.filename().string() : entry.versionId;
		const string displayName = entry.agentName + " / " + resolved.exportModel.buildName + " / " + idText;
		if (!displayNames.insert(displayName).second) {
			roster.errors.push_back(DescribeEntry(i, entry) + ": the same model is selected twice (" + displayName + ")");
			continue;
		}

		ModelBenchmarkModel model;
		model.displayName = displayName;
		model.agentName = entry.agentName;
		model.stageName = resolved.exportModel.stageName;
		model.versionId = entry.versionId;
		model.checkpointPath = resolved.checkpointPath;
		model.trainingSetupPath = resolved.trainingSetupPath;
		model.agentSetupSource = ChooseAgentSetupSource(entry, resolved);
		model.primary = primary;
		if (resolved.usedOwnPath)
			model.agentSetupJson = entry.agentSetupJson;

		roster.config.models.push_back(std::move(model));
		roster.exportModels.push_back(std::move(resolved.exportModel));
	}

	if (roster.errors.empty() && roster.config.models.size() < 2) {
		roster.errors.push_back(
			"a benchmark needs at least 2 included models that are not results-only (found " + to_string(roster.config.models.size()) + ")");
	}

	return roster;
}
