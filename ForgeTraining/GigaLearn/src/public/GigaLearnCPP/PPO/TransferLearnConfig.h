#pragma once

#include <GigaLearnInfer/Framework.h>
#include <GigaLearnInfer/ModelConfig.h>
#include <RLGymCPP/ActionParsers/ActionParser.h>
#include <RLGymCPP/Gamestates/GameState.h>
#include <RLGymCPP/ObsBuilders/ObsBuilder.h>

namespace GGL {

	typedef std::function<RLGC::ObsBuilder*()> MakeObsFn;
	typedef std::function<RLGC::ActionParser*()> MakeActFn;

	typedef std::function<std::vector<int>(const RLGC::Player&, const RLGC::GameState&)> MapActionsFn;

	struct TransferLearnConfig {
		// This function should return the old obs builder (can be the same as the new)
		MakeObsFn makeOldObsFn;

		// This function should return the old action parser (can be the same as the new)
		MakeActFn makeOldActFn;

		/**
		 * One teacher index per student action, or -1 for an unmapped student action.
		 * Indices may vary by state; nonnegative indices must be unique within a row.
		 * Transfer compares distributions conditional on actions legal for both parsers.
		 * Unmapped actions receive no direct loss gradient.
		 * Omit for identical action order.
		 **/
		MapActionsFn mapActsFn = nullptr;

		PartialModelConfig oldPolicyConfig;
		PartialModelConfig oldSharedHeadConfig; // If the old model didn't have shared layers, don't set this
		std::filesystem::path oldModelsPath; // Path to the directory with the old policy model(s) in it.

		// NOTE: All of the learning parameters below mostly depend on the complexity of your old policy's gameplay
		//	For refined/skilled bots, you may want a larger batch size and a much smaller learning rate
		//	The settings below are for a very basic ballchasing bot

		float lr = 3e-4;

		/** Number of paired observation rows collected per update (whole environment steps). **/
		int batchSize = 50'000;

		/** Maximum rows evaluated at once; zero uses the full batch. Does not change optimizer steps. **/
		int microBatchSize = 0;
		int epochs = 5;

		/** Learn the teacher's top shared legal action with cross-entropy; takes priority over useKLDiv. **/
		bool useHardActionLabels = false;

		/** Forward categorical KL (teacher || student); otherwise use mean absolute probability error. **/
		bool useKLDiv = false;

		// Scale of the loss (prevents optimizers from dying since the natural loss is very low)
		float lossScale = 500.f;

		/** Positive exponent applied per row (KL/cross-entropy), or per action (absolute error). **/
		float lossExponent = 1.f;
	};
}