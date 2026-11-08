#pragma once

#include "GigaLearnInfer/JsonArchive.h"

namespace GGL {

	enum class CriticHeadLossWeighting {
		/** Every present value head counts equally. */
		Equal,

		/** A head's error counts in proportion to |weight|·σ, its typical contribution to the combined value. */
		Contribution
	};

	inline const EnumNames<CriticHeadLossWeighting> CRITIC_HEAD_LOSS_WEIGHTING_NAMES = { {
		{ CriticHeadLossWeighting::Equal, "Equal" },
		{ CriticHeadLossWeighting::Contribution, "Contribution" },
	} };

	/**
	 * Settings for the decomposed critic, which trains one value head per reward and combines them with the current reward weights.
	 * Experimental: it is not a proven technique, so compare against a normal critic before relying on it.
	 */
	struct DecomposedCriticConfig {
		/**
		 * Set from Agent::decomposedCritic,
		 * since it changes the critic's architecture and checkpoint format.
		 */
		bool enabled = false;

		/** Fraction of the way each iteration moves a head's PopArt mean and mean-square towards that batch. */
		float popArtBeta = 0.02f;

		/** Lower bound on each head's target std, in raw (unweighted) reward units. */
		float popArtSigmaFloor = 0.01f;

		CriticHeadLossWeighting lossWeighting = CriticHeadLossWeighting::Equal;

		/**
		 * Also report each head's PopArt mean and normalized loss, and per mode its target std and contribution share.
		 * Metrics only, so it is left out of the training setup snapshot and toggling it never starts a new segment.
		 */
		bool logDetailedHeadMetrics = false;
	};
}
