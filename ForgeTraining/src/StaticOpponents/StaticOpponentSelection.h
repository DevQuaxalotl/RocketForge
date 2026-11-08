#pragma once

#include <GigaLearnInfer/Framework.h>
#include <GigaLearnInfer/AgentSetupSnapshot.h>
#include <GigaLearnInfer/JsonArchive.h>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace RocketForgeTraining
{
	/** Identifies a saved training segment or imported model selected as a static opponent. */
	struct StaticOpponentSelectionEntry
	{
		std::string sourceAgentName;
		std::string sourceBuildName;
		std::string sourceStageName;
		std::string sourceVersionId;
		std::string displayName;
		std::filesystem::path checkpointPath;

		/** Exact source runtime configuration resolved from the selected model. */
		std::optional<RocketForge::AgentSetupSnapshot> agentSetup;

		/** Serializes this entry to/from the "static_opponents" array of a selection file. */
		void Serialize(GGL::JsonArchive& ar);
	};

	/**
	 * Candidate shown in the static opponent picker.
	 * This may be a saved stage version with a loadable checkpoint,
	 * or an imported model folder already present under an agent's Models/Imported Models root.
	 */
	struct StaticOpponentCandidate
	{
		StaticOpponentSelectionEntry selection;
		std::string notes;
		uint64_t latestTimesteps = 0;
		double skillRating = 0.0;
		bool hasSkillRating = false;
		bool selected = false;
	};

	/** Returns the stable identity persisted by debug and selection UIs for one entry. */
	std::string MakeStaticOpponentSelectionKey(const StaticOpponentSelectionEntry& entry);

	/** Returns all eligible versions that can be selected as static opponents for the target agent. */
	std::vector<StaticOpponentCandidate> ListStaticOpponentCandidates(const std::string& targetAgentName);

	/** Returns all eligible versions that can be selected as static opponents for the target Build. */
	std::vector<StaticOpponentCandidate> ListStaticOpponentCandidates(
		const std::string& targetAgentName,
		const std::string& targetBuildName);

	/**
	 * Returns the repo-relative folders (forward slashes, as rclone globs) holding each selected static opponent's model files,
	 * for replicating an allowlist onto a machine that has none of them.
	 * Covers imported-model and segment-sourced opponents alike,
	 * and reports folders whether or not they exist locally, including entries that can't be resolved here.
	 */
	std::vector<std::string> ListStaticOpponentSyncPaths(const std::string& targetAgentName, const std::string& targetBuildName);

	/** Loads the persisted per-agent static opponent allowlist. */
	std::vector<StaticOpponentSelectionEntry> LoadStaticOpponentSelection(const std::string& targetAgentName);

	/** Loads the persisted per-Build static opponent allowlist. */
	std::vector<StaticOpponentSelectionEntry> LoadStaticOpponentSelection(
		const std::string& targetAgentName,
		const std::string& targetBuildName);

	/** Saves the per-agent static opponent allowlist. */
	void SaveStaticOpponentSelection(const std::string& targetAgentName, const std::vector<StaticOpponentSelectionEntry>& entries);

	/** Saves the per-Build static opponent allowlist without copying source model folders. */
	void SaveStaticOpponentSelection(
		const std::string& targetAgentName,
		const std::string& targetBuildName,
		const std::vector<StaticOpponentSelectionEntry>& entries);

	/** Validates the per-agent static opponent registry. No model folders are copied. */
	void SyncStaticOpponentSelection(const std::string& targetAgentName);

	/**
	 * Rewrites the per-Build static opponent registry with each entry resolved to its source's latest checkpoint.
	 * An entry that can't be resolved on this machine is skipped by training and written back exactly as stored.
	 * No model folders are copied.
	 */
	void SyncStaticOpponentSelection(const std::string& targetAgentName, const std::string& targetBuildName);

	/** Every engine the Build's selected static opponents need, resolved from their sources as training will. */
	std::vector<RocketSim::EngineId> ResolveStaticOpponentEngines(const std::string& targetAgentName, const std::string& targetBuildName);
}
