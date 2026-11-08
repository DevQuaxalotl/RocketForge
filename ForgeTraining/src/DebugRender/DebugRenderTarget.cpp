#include "DebugRenderTarget.h"

#include "Storage/BuildStorage.h"
#include "Storage/ImportedModelStorage.h"
#include "TrainingSetup/VersionManagement.h"

#include <GigaLearnCPP/SegmentVersioning.h>

#include <algorithm>
#include <optional>
#include <vector>

using namespace std;

namespace fs = std::filesystem;

namespace
{
	using RocketForgeTraining::DebugRenderTarget;

	DebugRenderTarget Unresolved(const string& error)
	{
		DebugRenderTarget target;
		target.error = error;

		return target;
	}

	DebugRenderTarget ResolveImported(const string& agentName, const string& folderName, uint64_t checkpointTimesteps)
	{
		if (folderName.empty())
			return Unresolved("Select a build and a segment, or an imported model.");

		if (checkpointTimesteps != 0)
			return Unresolved("Imported model '" + folderName + "' has a single checkpoint; a checkpoint can't be chosen.");

		const vector<RocketForgeTraining::DiscoveredImportedModel> models = RocketForgeTraining::EnumerateImportedModels(agentName);
		const auto found = std::find_if(models.begin(), models.end(), [&](const RocketForgeTraining::DiscoveredImportedModel& model) {
			return model.folderName == folderName;
		});

		if (found == models.end())
			return Unresolved("Imported model '" + folderName + "' was not found.");
		if (!found->canUseForInference)
			return Unresolved("Imported model '" + folderName + "' has no loadable policy to render.");

		DebugRenderTarget target;
		target.checkpointFolder = found->importedRoot;
		target.isImported = true;

		return target;
	}

	/** The requested checkpoint folder, preferring a durable checkpoint over an autosave at the same timesteps. */
	optional<fs::path> FindCheckpointFolder(const fs::path& segmentRoot, uint64_t checkpointTimesteps)
	{
		using RocketForgeTraining::BuildStorage;

		for (const char* folderName : { BuildStorage::CHECKPOINTS_FOLDER_NAME, BuildStorage::TEMP_CHECKPOINTS_FOLDER_NAME }) {
			const fs::path folder = segmentRoot / folderName / to_string(checkpointTimesteps);

			std::error_code ignored;
			if (fs::is_directory(folder, ignored))
				return folder;
		}

		return nullopt;
	}
}

DebugRenderTarget RocketForgeTraining::ResolveDebugRenderTarget(
	const string& agentName,
	const string& buildName,
	const string& segmentId,
	uint64_t checkpointTimesteps)
{
	if (buildName.empty())
		return ResolveImported(agentName, segmentId, checkpointTimesteps);

	if (segmentId.empty())
		return Unresolved("Select a segment of build '" + buildName + "' to render.");

	const bool includeInactive = true;
	const vector<GGL::SegmentVersionSummary> segments = ListBuildVersions(agentName, buildName, {}, includeInactive);
	const auto found = std::find_if(segments.begin(), segments.end(), [&](const GGL::SegmentVersionSummary& summary) {
		return summary.id == segmentId;
	});

	if (found == segments.end())
		return Unresolved("Segment '" + segmentId + "' was not found in build '" + buildName + "'.");

	DebugRenderTarget target;
	target.trainedStageName = found->stageName;

	if (checkpointTimesteps == 0) {
		const optional<ResolvedCheckpoint> latest = BuildStorage::LatestSegmentCheckpointIn(found->rootPath);
		if (!latest)
			return Unresolved("Segment '" + segmentId + "' has no checkpoint to render.");

		target.checkpointFolder = latest->folder;
		return target;
	}

	const optional<fs::path> requested = FindCheckpointFolder(found->rootPath, checkpointTimesteps);
	if (!requested)
		return Unresolved("Segment '" + segmentId + "' has no checkpoint at " + to_string(checkpointTimesteps) + " timesteps.");

	target.checkpointFolder = *requested;
	return target;
}
