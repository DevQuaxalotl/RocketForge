#pragma once

#include "Settings/ForgeSettings.h"
#include "StageQueue/StageQueueRunner.h"

#include <string>
#include <vector>

namespace RocketForgeTraining {
	class BuildStorage;

	/**
	 * The files and folders one queue run reads or writes, as repo-relative forward-slash paths
	 * in rclone glob syntax (literal names already escaped).
	 **/
	struct SyncManifest {
		/** Exact files, or file globs such as "<build>/segments/*\/segment.json". */
		std::vector<std::string> metadataPaths;
		/** Folder globs ending in "/**". */
		std::vector<std::string> folderGlobs;
		/**
		 * The queue segments' temp_checkpoints folders, as "/**" globs.
		 * Kept apart from folderGlobs so the default filter keeps excluding them:
		 * consumers push these autosaves as a crash backup and never pull them.
		 **/
		std::vector<std::string> tempCheckpointGlobs;
	};

	/** Which part of the manifest a rendered filter list covers. */
	enum class SyncManifestScope {
		/** Metadata files, the segment folders, the pools, and the static opponents and skill references; never temp checkpoints. */
		Full,
		/** Only the metadata files — the small set a consumer needs before it can compute the full manifest. */
		MetadataOnly,
		/** Only the queue segments' temp_checkpoints folders. */
		TempCheckpoints,
	};

	/** Inputs BuildSyncManifest cannot derive from settings and storage alone. */
	struct SyncManifestContext {
		/** The agent's Models directory relative to the repo root (see ResolveAgentModelsRelativePath). */
		std::string modelsRel;
		/** Static-opponent and skill-reference folder globs relative to the repo root (see ListOpponentModelSyncPaths). */
		std::vector<std::string> staticOpponentGlobs;
	};

	/** Escapes the characters rclone reads as pattern syntax, so text matches only itself. */
	std::string RcloneGlobLiteral(const std::string& text);

	/**
	 * Metadata files for the whole build, the full folders of every segment the queue touches,
	 * the lineage-filtered recent-policy and skill-benchmark pools, and the static opponents and skill references.
	 **/
	SyncManifest BuildSyncManifest(
		const ForgeSettings& settings,
		const BuildStorage& storage,
		const SegmentInspector& inspect,
		const SyncManifestContext& context);

	/**
	 * Renders one scope of the manifest as an rclone filter list with rules relative to the Bots/ directory,
	 * ending in "- **" so anything unlisted is excluded.
	 **/
	std::string FormatRcloneFilter(const SyncManifest& manifest, SyncManifestScope scope);

	/**
	 * Copies the segments a remote run recorded into the local queue, leaving every other local edit in place.
	 * An entry is updated only when the remote entry at the same index has the same stage and a different recorded segment;
	 * it is pointed at that segment the way the queue runner records one.
	 * Returns the number of entries updated.
	 **/
	int MergeRecordedSegmentIds(ForgeSettings& local, const ForgeSettings& remote);
}
