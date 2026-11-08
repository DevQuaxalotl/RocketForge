#pragma once

#include "ModelBenchmark.h"
#include "ModelBenchmarkExport.h"
#include "Settings/ForgeSettings.h"

#include <string>
#include <vector>

namespace RocketForgeTraining
{
	/** A persisted benchmark setup turned into something RunModelBenchmark can run, plus what to tell the user about it. */
	struct ModelBenchmarkRoster {
		/** Models and run options from the settings. */
		ModelBenchmarkConfig config;

		/** Index-aligned with config.models. */
		std::vector<ModelBenchmarkExportModel> exportModels;
		std::vector<std::string> warnings;

		/** Non-empty means the benchmark cannot run. */
		std::vector<std::string> errors;
	};

	/**
	 * Turns the persisted settings into a runnable benchmark.
	 * Entries without a checkpoint path resolve from storage: a segment's newest checkpoint and training_setup.json,
	 * or an imported model's folder.
	 */
	ModelBenchmarkRoster ResolveModelBenchmarkRoster(const ModelBenchmarkSettings& settings, bool allPrimary);
}
