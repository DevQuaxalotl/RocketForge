#pragma once
#include <string>
#include <vector>
#include <RLGymCPP/Rewards/Reward.h>

namespace ForgeDebug {

	struct RewardDebugData {
		std::string name;
		RLGC::RewardCategory category;

		/** Unweighted reward value for the sampled player */
		float rawValue;

		/** Weight from WeightedReward */
		float weight;

		/** rawValue * weight */
		float weightedValue;

		/** Running episode total of abs(weightedValue) for the sampled player */
		float episodeAbsWeightedSum;

		/** Declared output range from the reward class */
		RLGC::RewardValueRange declaredRange;

		/** Value after normalization, computed by the collector based on the active mode */
		float normalizedValue;

		/** 0 for continuous rewards; for events, decays from 1.0 to 0.0 over decayDuration */
		float eventDecayValue;

		/** True if the event fired on this exact frame */
		bool eventFiredThisFrame;

		/** 3D debug lines from the reward */
		std::vector<RLGC::DebugLine> debugLines;
	};

}
