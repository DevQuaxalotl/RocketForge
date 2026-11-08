#pragma once

namespace RocketForge { class TrainingStage; }

namespace RocketForgeTraining {
	/** Internal helpers shared between ForgeTraining.cpp, StageQueueRunner.cpp, and VersionManagement.cpp. Not part of the public API. */
	void LaunchNewStageTraining(
		RocketForge::TrainingStage* newStage,
		const RocketForge::TrainingStage* sourceStage,
		bool clearRunId,
		bool pruneSourceCheckpoints,
		bool confirmedDelete);

	void LaunchContinueTraining(
		RocketForge::TrainingStage* stageToTrain,
		const RocketForge::TrainingStage* sourceStage,
		bool startFromScratch,
		bool pruneSourceCheckpoints,
		bool confirmedDelete);

}
