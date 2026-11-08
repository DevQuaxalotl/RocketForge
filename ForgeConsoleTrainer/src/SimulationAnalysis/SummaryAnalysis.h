#pragma once

#include "SimulationAnalysisTypes.h"

#include <GigaLearnInfer/Agent.h>
#include <RLGymCPP/BasicTypes/Action.h>

#include <filesystem>
#include <vector>

namespace RocketForge::SimulationAnalysis
{
	/** Builds the wall-dash pair pre-aggregation for the summary JSON. **/
	std::vector<WallDashPairStats> BuildWallDashPairSummary(const std::vector<TraceSample>& samples);
	/** Builds fixed-length speed-flip sequence pre-aggregations for the summary JSON. **/
	std::vector<SequenceStats> BuildSpeedFlipSummary(
		const std::vector<TraceSample>& samples,
		int ticksPerAction,
		int parts);
	/** Writes the additive summary JSON next to the raw CSV trace. **/
	void WriteSummary(
		const std::filesystem::path& summaryPath,
		const SimulationAnalysisOptions& options,
		const Agent& sourceAgent,
		const std::vector<TraceSample>& samples,
		const std::vector<WallDashPairStats>& wallDashPairSummary,
		const std::vector<SequenceStats>& speedFlip4TickSummary,
		const std::vector<SequenceStats>& speedFlip2TickSummary,
		const std::vector<RLGC::Action>& actions);
}
