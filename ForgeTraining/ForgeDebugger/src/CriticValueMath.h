#pragma once
#include "CriticDebugData.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace GGL { class JsonArchive; }

namespace ForgeDebug {

	/** One value head's saved PopArt statistics, as RUNNING_STATS.json "value_heads" stores them. */
	struct ValueHeadStats {
		std::string name;
		bool initialized = false;
		double mean = 0.0;

		/** Running mean of squared targets; the head's spread is sqrt(meanSq - mean²). */
		double meanSq = 0.0;

		/** Same keys as GGL::PopArtHeadState. Defined in ForgeTraining, which can see JsonArchive. */
		void Serialize(GGL::JsonArchive& ar);
	};

	/** A decomposed-critic head with everything needed to turn its output into a debug value. */
	struct DecomposedCriticHead {
		ValueHeadStats stats;
		std::optional<float> weight;
	};

	/**
	 * Raw value σ·n + μ of one PopArt-normalized head output.
	 * Mirrors GGL::PopArtStats, which this module can't link:
	 * an uninitialized head uses μ = 0 and σ = 1, otherwise σ never drops below sigmaFloor.
	 */
	float DenormalizeHeadValue(float normalized, const ValueHeadStats& stats, float sigmaFloor);

	/** Builds the estimate from one row of decomposed-critic outputs, one per head in critic row order. */
	CriticDebugData CombineDecomposedCritic(
		const std::vector<float>& normalizedOutputs, const std::vector<DecomposedCriticHead>& heads, float sigmaFloor);

	/** Builds the estimate from a single-output critic, which is trained on returns divided by returnStd. */
	CriticDebugData MakeSingleCriticEstimate(float output, float returnStd);

	/** Return std the way GGL::WelfordStat::GetSTD derives it from its saved fields; 1 when unknown. */
	float WelfordStd(double runningVariance, int64_t count);
}
