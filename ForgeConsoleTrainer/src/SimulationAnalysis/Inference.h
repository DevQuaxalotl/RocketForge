#pragma once

#include "SimulationAnalysisTypes.h"

#include <GigaLearnInfer/ModelManager.h>
#include <GigaLearnInfer/Models.h>
#include <RLGymCPP/ActionParsers/ActionParser.h>
#include <RLGymCPP/ObsBuilders/ObsBuilder.h>

namespace RocketForge::SimulationAnalysis
{
	/** Runs the one-player policy inference path and keeps the obs/mask used for tracing. **/
	InferenceTrace InferActionTrace(
		GGL::ModelSet& models,
		RLGC::ObsBuilder& obsBuilder,
		RLGC::ActionParser& actionParser,
		const RLGC::Player& player,
		const RLGC::GameState& state,
		int obsSize,
		int numActions,
		bool deterministic,
		float temperature);
}