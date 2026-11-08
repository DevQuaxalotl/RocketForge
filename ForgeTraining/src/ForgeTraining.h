#pragma once

#include "Settings/ForgeSettings.h"
#include "StaticOpponents/StaticOpponentSelection.h"
#include "TrainingSetup/TrainingSetupSnapshot.h"
#include "TrainingSetup/VersionManagement.h"

#include <RocketSim/Engine.h>

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace RocketForge {
	class TrainingStage;
}

namespace RocketForgeTraining {

	/**
	 * Starts the embedded Python interpreter (leaving the GIL released) and RocketSim.
	 * Runs once per process; later calls from any thread return immediately.
	 */
	void InitializeTraining();

	/** Creates a TrainingStage, applying both global and per-stage settings to learnerConfig. */
	std::unique_ptr<RocketForge::TrainingStage> CreateTrainingStage(const std::string& agentName, const std::string& stageName, const GlobalSettings& globals, const StageSettings& stage);

	/** Status of a training run. */
	enum class RunStatus {
		Completed,    // All stages reached their targets.
		Stopped,      // RequestStop signaled.
		Error,        // Exception during training.
	};

	/** Result of a training run. */
	struct RunResult {
		RunStatus status = RunStatus::Completed;

		/** Valid only when status == Error. */
		int failedStageIndex = -1;

		std::string errorMessage;

		/** Set when the Error is a RocketSim::EngineUnavailableError; errorMessage is then its reason, unprefixed. */
		std::optional<RocketSim::EngineId> unavailableEngine;
	};

	/** Optional hooks invoked by the runner during execution. */
	struct StageEvents {
		std::function<void(int index, const std::string& stageName, const std::string& versionId)> onStageStarted;
		std::function<void(int index)> onStageCompleted;
		std::function<void(int index, const std::string& message)> onStageSkipped;

		/**
		 * Called after CreateTrainingStage but before training starts.
		 * Lets the caller modify learnerConfig before the Learner is constructed.
		 */
		std::function<void(int index, RocketForge::TrainingStage* stage)> onStageCreated;
	};

	/** Where and how a run records queue progress into the settings file. */
	struct RunPersistence {
		/** Settings file the runner records progress into; empty records nothing. */
		std::filesystem::path settingsPath;

		/** Written in place of the run's globals, so per-run overrides never reach the file. Unset writes the run's own. */
		std::optional<GlobalSettings> persistedGlobals;
	};

	/**
	 * Top-level training entry point. Walks settings.stages, dispatching as needed.
	 * settings is mutated to write back createdSegmentId and skip-rewritten targets,
	 * and is saved through persistence after each mutation.
	 */
	RunResult RunTraining(ForgeSettings& settings, const RunPersistence& persistence, const StageEvents& events = {});

	/** Runs the queue, recording progress into settingsPath. */
	RunResult RunTraining(ForgeSettings& settings, const std::filesystem::path& settingsPath, const StageEvents& events = {});

	/** Requests the running learner to save and stop. Halts the queue. */
	void RequestStop();

	/** Requests the running learner to save and stop, but advance the queue to the next stage. */
	void RequestSkipStage();
}
