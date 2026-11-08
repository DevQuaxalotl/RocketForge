#pragma once

#include <GigaLearnInfer/AgentSetupSnapshot.h>
#include <GigaLearnInfer/JsonArchive.h>
#include "RewardComparison.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace RLGC
{
	/** Non-intrusive serialization for RLGC types; lives here (not in RLGym) so the game-env library stays JSON-free. */
	void SerializeFields(GGL::JsonArchive& ar, RewardValueRange& range);
}

namespace RocketForge
{
	class TrainingStage;

	/** Enum<->string table for RewardTableEntry::category, matching the on-disk "event"/"continuous" strings. */
	extern const GGL::EnumNames<RLGC::RewardCategory> REWARD_CATEGORY_NAMES;

	/** Single serialized training parameter shown in the setup viewer. */
	struct TrainingSetupParameter
	{
		std::string group;
		std::string name;
		std::string value;

		void Serialize(GGL::JsonArchive& ar);
	};

	/** One saved-vs-current training setup change. */
	struct TrainingSetupDiff
	{
		std::string field;
		std::string savedValue;
		std::string currentValue;
	};

	/**
	 * Snapshot of the reward table and key hyperparameters for a saved version.
	 * It is stored next to version artifacts,
	 * so the GUI can explain what was trained without reconstructing the stage from code.
	 */
	struct TrainingSetupSnapshot
	{
		static constexpr const char* FILE_NAME = "training_setup.json";

		std::string agentName;
		std::string stageName;
		std::string versionId;
		std::string savedAt;

		/** Recorded training engine; zero preserves unknown legacy provenance. */
		int engineVersion = 0;
		std::string engineRevision;
		std::optional<AgentSetupSnapshot> agentSetup;
		std::vector<RewardTableEntry> rewards;
		std::vector<TrainingSetupParameter> hyperparameters;

		RewardComparisonTable BuildRewardTable() const;
		void Serialize(GGL::JsonArchive& ar);
	};

	/** Returns whether a serialized parameter contributes to setup identity. */
	bool TrainingSetupParameterAffectsIdentity(const TrainingSetupParameter& parameter);

	/** Returns display-ready saved -> current training setup differences. */
	std::vector<TrainingSetupDiff> DiffTrainingSetupSnapshots(const TrainingSetupSnapshot& saved, const TrainingSetupSnapshot& current);

	/** Builds a setup snapshot from the current stage configuration. */
	TrainingSetupSnapshot MakeTrainingSetupSnapshot(const TrainingStage* stage, const std::string& versionId = {});

	/** Reads a previously saved setup snapshot from disk. */
	std::optional<TrainingSetupSnapshot> LoadTrainingSetupSnapshot(const std::filesystem::path& path, std::string* error = nullptr);

	/** Returns nullopt only when `path` does not exist; throws when it exists but cannot be loaded. */
	std::optional<TrainingSetupSnapshot> LoadExistingTrainingSetupSnapshot(const std::filesystem::path& path);

	/**
	 * The agent setup a saved model runs with: the snapshot's recorded setup,
	 * or `current` with its simulators resolved from the snapshot's engine (v2 when nothing is recorded).
	 */
	AgentSetupSnapshot ResolveSavedAgentSetup(const std::optional<TrainingSetupSnapshot>& saved, AgentSetupSnapshot current);
}

namespace RocketForgeTraining {
	/** Captures the current reward table and training hyperparameters for inspection. */
	RocketForge::TrainingSetupSnapshot CaptureTrainingSetupSnapshot(const RocketForge::TrainingStage* stage);

	/** Loads the saved setup snapshot for the requested version, if it exists. */
	std::optional<RocketForge::TrainingSetupSnapshot> LoadTrainingSetupSnapshot(
		const RocketForge::TrainingStage* stage,
		const std::string& versionId);

	/** Returns whether the requested version has a saved setup snapshot. */
	bool HasTrainingSetupSnapshot(const RocketForge::TrainingStage* stage, const std::string& versionId);
}
