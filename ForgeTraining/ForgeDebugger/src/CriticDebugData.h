#pragma once
#include <optional>
#include <string>
#include <vector>

namespace ForgeDebug {

	/** One decomposed-critic value head, in real (denormalized) return units. */
	struct CriticHeadDebugData {
		std::string name;

		/** Predicted discounted future reward from this head, unweighted. */
		float raw = 0.f;

		/** Current weight of the matching reward; empty when the stage has no reward of this name. */
		std::optional<float> weight;

		/** weight · raw, or 0 when there is no matching reward. */
		float contribution = 0.f;

		/** Sum of |contribution| over the episode so far, filled in by RewardDebugCollector like a reward's abs sum. */
		float episodeAbsContributionSum = 0.f;
	};

	/** The critic's value prediction V(s) for the debugged player, in real return units. */
	struct CriticDebugData {
		float total = 0.f;
		bool isDecomposed = false;

		/** Critic row order. Empty for a single-output critic. */
		std::vector<CriticHeadDebugData> heads;
	};
}
