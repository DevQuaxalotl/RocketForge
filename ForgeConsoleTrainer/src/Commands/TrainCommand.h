#pragma once

#include "Cli/CliArgs.h"
#include "ForgeTraining.h"
#include "Settings/ForgeSettings.h"
#include "StageQueue/StageQueueRunner.h"

#include <filesystem>
#include <string>
#include <vector>

namespace RocketForge::Cli {

/** Everything one `train` run needs: the settings to run and how the run records progress. */
struct TrainRequest {
	RocketForgeTraining::ForgeSettings settings;
	RocketForgeTraining::RunPersistence persistence;
	bool preview = false;
};

/** Stored segments a one-off --stage run starts from; empty ids mean the stage has none. */
struct AdHocSegments {
	/** Newest segment of --stage, which the run continues. */
	std::string stageSegmentId;

	/** Newest segment of --from-stage, which the new segment bootstraps from. */
	std::string sourceSegmentId;
};

/**
 * Builds the run from loaded settings and one-off flags. Pure: no file access.
 * Throws CliUsageError for bootstrap flags without --stage, --new together with --from-stage,
 * --keep-run-id without --from-stage, or --from-stage with no source segment.
 */
TrainRequest BuildTrainRequest(
	const CliArgs& args,
	const RocketForgeTraining::ForgeSettings& loaded,
	const std::filesystem::path& settingsPath,
	const AdHocSegments& segments = {});

/** One line per queue entry, e.g. "1. ball-chase  continue v1.3 (12.0M / 50.0M)". */
std::string FormatQueuePlan(const std::vector<RocketForgeTraining::QueuePlanEntry>& plan);

int RunTrainCommand(const CliArgs& args);

}
