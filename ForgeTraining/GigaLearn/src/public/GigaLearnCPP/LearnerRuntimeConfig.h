#pragma once
#include "Util/Report.h"
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

class Agent;

namespace GGL {

	/**
	 * Learner settings that the trainer and orchestration layer fill in for each run:
	 * storage paths, build and segment identity, callbacks and stop limits.
	 * Stages and bots don't set these; held by LearnerConfig::runtime.
	 */
	struct LearnerRuntimeConfig {
		/**
		 * Checkpoints are saved here as timestep-numbered subfolders, e.g. 20,000 steps saves to "20000".
		 * Empty disables saving.
		 */
		std::filesystem::path checkpointFolder = "checkpoints";

		/** Separate load root for checkpoint resume; empty loads from checkpointFolder. */
		std::filesystem::path checkpointLoadFolder = {};

		/** Durable snapshot root that Save(true) copies the current checkpoint into. */
		std::filesystem::path checkpointSnapshotFolder = {};

		/** Also copies durable snapshots on regular autosaves, not only on Save(true). */
		bool persistCheckpointSnapshotsDuringTraining = false;

		/** Called after a checkpoint and its policy/benchmark registries have been saved. */
		std::function<void(uint64_t timesteps, const std::filesystem::path& checkpointFolder)> onCheckpointSaved = {};

		std::string versionId = {};
		std::string buildName = {};
		std::string segmentId = {};
		std::string trainingStageName = {};
		std::string versionNotes = {};
		bool resetRunIdOnLoad = false;
		std::filesystem::path recentPolicyVersionsFolder = {};
		std::vector<std::string> recentPolicyVersionSourceIds = {};
		std::filesystem::path skillBenchmarkFolder = {};
		std::filesystem::path staticOpponentsRegistryPath = {};

		/** The Build's skill reference registry; the skill test plays its references when the file lists any. */
		std::filesystem::path skillReferencesRegistryPath = {};

		/** Creates source-agent runtimes in the module that owns agent registrations. */
		std::function<std::unique_ptr<Agent>(const std::string&)> staticOpponentAgentFactory = {};

		/** Non-zero saves and exits cleanly once total timesteps reach this after an iteration. */
		uint64_t stopAtTotalTimesteps = 0;

		/** Non-zero exits after this many learner iterations, after printing the iteration report. */
		uint64_t stopAfterIterations = 0;

		/** Called once per iteration after its timing and timestep metrics are in the report. */
		std::function<void(const Report& report)> onIterationReport = {};
	};
}
