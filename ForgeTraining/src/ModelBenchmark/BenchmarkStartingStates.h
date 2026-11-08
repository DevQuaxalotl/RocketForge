#pragma once

#include <GigaLearnCPP/Eval/PolicyMatchRunner.h>
#include "ModelBenchmark.h"

namespace RocketForgeTraining::ModelBenchmarkInternal {
	/** Starting-state ids are mode * stride + index, so each team size has its own id range. **/
	constexpr uint64_t BENCHMARK_START_MODE_ID_STRIDE = 100000000ULL;

	/** Load portable physical state, never backend-specific opaque caches. **/
	GGL::PolicyMatchSnapshot ReadBenchmarkStartingState(const std::filesystem::path& directory, int mode, uint64_t seed);

	void PrepareBenchmarkStartingStates(const ModelBenchmarkConfig& config);
}
