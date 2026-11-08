#include "SyncManifest.h"
#include "Storage/BuildStorage.h"
#include "TrainingSetup/TrainingSetupSnapshot.h"

#include <algorithm>
#include <optional>
#include <set>

namespace RocketForgeTraining {

	namespace {
		const char* const BOTS_ROOT_PREFIX = "Bots/";

		// Filter rules are relative to the Bots/ directory the consumers copy from.
		std::string BotsRelativeRule(const std::string& path)
		{
			const std::string prefix = BOTS_ROOT_PREFIX;
			if (path.rfind(prefix, 0) == 0)
				return "/" + path.substr(prefix.size());
			return "/" + path;
		}

		std::string ParentPath(const std::string& path)
		{
			const size_t slash = path.find_last_of('/');
			return slash == std::string::npos ? std::string() : path.substr(0, slash);
		}

		// Every queue id plus its parent chain, so lineage-filtered pools sync too.
		std::vector<std::string> LineageSegmentIds(const BuildStorage& storage, const std::vector<std::string>& queueIds)
		{
			std::vector<std::string> ids = {};
			std::set<std::string> seen = {};

			for (const std::string& queueId : queueIds) {
				std::string current = queueId;
				while (!current.empty() && seen.insert(current).second) {
					ids.push_back(current);
					std::optional<TrainingSegmentMetadata> segment = storage.LoadSegmentById(current);
					current = segment.has_value() ? segment->parentSegmentId : std::string();
				}
			}

			return ids;
		}
	}

	std::string RcloneGlobLiteral(const std::string& text)
	{
		std::string escaped = {};
		for (char c : text) {
			if (c == '*' || c == '?' || c == '[' || c == ']' || c == '{' || c == '}' || c == '\\')
				escaped.push_back('\\');
			escaped.push_back(c);
		}
		return escaped;
	}

	SyncManifest BuildSyncManifest(
		const ForgeSettings& settings,
		const BuildStorage& storage,
		const SegmentInspector& inspect,
		const SyncManifestContext& context)
	{
		SyncManifest manifest;

		const std::string modelsRel = RcloneGlobLiteral(context.modelsRel);
		const std::string agentRel = ParentPath(modelsRel);
		const std::string buildRel = modelsRel + "/" + RcloneGlobLiteral(storage.BuildRoot().filename().string());
		const std::string segmentsRel = buildRel + "/" + BuildStorage::SEGMENTS_FOLDER_NAME;
		const std::string recentRel = buildRel + "/" + BuildStorage::RECENT_POLICY_VERSIONS_FOLDER_NAME;
		const std::string benchmarksRel = buildRel + "/" + BuildStorage::SKILL_BENCHMARKS_FOLDER_NAME;

		manifest.metadataPaths = {
			agentRel + "/forge_settings.json",
			modelsRel + "/" + BuildStorage::BUILD_REGISTRY_FILE_NAME,
			buildRel + "/" + BuildStorage::BUILD_FILE_NAME,
			buildRel + "/" + BuildStorage::STATIC_OPPONENTS_FILE_NAME,
			buildRel + "/" + BuildStorage::SKILL_REFERENCES_FILE_NAME,
			buildRel + "/" + BuildStorage::RLBOT_EXPORTS_FILE_NAME,
			recentRel + "/" + BuildStorage::RECENT_POLICY_VERSIONS_FILE_NAME,
			benchmarksRel + "/" + BuildStorage::SKILL_BENCHMARKS_FILE_NAME,
			segmentsRel + "/*/" + BuildStorage::SEGMENT_FILE_NAME,
			segmentsRel + "/*/" + RocketForge::TrainingSetupSnapshot::FILE_NAME,
		};

		const std::vector<std::string> queueIds = ResolveQueueSegmentIds(settings, storage, inspect);
		for (const std::string& id : queueIds) {
			const std::string segmentGlob = segmentsRel + "/" + RcloneGlobLiteral(id) + "_*";
			manifest.folderGlobs.push_back(segmentGlob + "/**");
			manifest.tempCheckpointGlobs.push_back(
				segmentGlob + "/" + BuildStorage::TEMP_CHECKPOINTS_FOLDER_NAME + "/**");
		}

		for (const std::string& id : LineageSegmentIds(storage, queueIds)) {
			manifest.folderGlobs.push_back(recentRel + "/" + RcloneGlobLiteral(id) + "_*/**");
			manifest.folderGlobs.push_back(benchmarksRel + "/" + RcloneGlobLiteral(id) + "_*/**");
		}

		for (const std::string& glob : context.staticOpponentGlobs)
			manifest.folderGlobs.push_back(glob + "/**");

		return manifest;
	}

	std::string FormatRcloneFilter(const SyncManifest& manifest, SyncManifestScope scope)
	{
		std::string filter = {};

		if (scope == SyncManifestScope::TempCheckpoints) {
			for (const std::string& glob : manifest.tempCheckpointGlobs)
				filter += "+ " + BotsRelativeRule(glob) + "\n";
			filter += "- **\n";
			return filter;
		}

		// Unanchored, so temp checkpoints are dropped at any depth before any include can claim them.
		filter += "- temp_checkpoints/**\n";
		filter += "- Temp Checkpoints/**\n";

		for (const std::string& path : manifest.metadataPaths)
			filter += "+ " + BotsRelativeRule(path) + "\n";

		if (scope == SyncManifestScope::Full) {
			for (const std::string& glob : manifest.folderGlobs)
				filter += "+ " + BotsRelativeRule(glob) + "\n";
		}

		filter += "- **\n";

		return filter;
	}

	int MergeRecordedSegmentIds(ForgeSettings& local, const ForgeSettings& remote)
	{
		int updated = 0;
		const size_t sharedCount = std::min(local.stages.size(), remote.stages.size());

		for (size_t i = 0; i < sharedCount; ++i) {
			StageSettings& entry = local.stages[i];
			const StageSettings& remoteEntry = remote.stages[i];

			if (remoteEntry.stageName != entry.stageName || remoteEntry.createdSegmentId.empty())
				continue;
			if (remoteEntry.createdSegmentId == entry.createdSegmentId)
				continue;

			// The whole continue-target must move with the id: a leftover version bump would make the queue plan a new segment.
			FlipBootstrapToContinueExisting(entry, remoteEntry.createdSegmentId);
			entry.branchName = remoteEntry.branchName;
			++updated;
		}

		return updated;
	}
}
