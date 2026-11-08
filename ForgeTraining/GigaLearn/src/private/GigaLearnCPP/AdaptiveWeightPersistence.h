#pragma once

#include <GigaLearnInfer/JsonArchive.h>
#include <RLGymCPP/EnvSet/AdaptiveRewardWeightController.h>

namespace RLGC {
	/** Names for AdaptivePhase as written to RUNNING_STATS.json. */
	extern const GGL::EnumNames<AdaptivePhase> ADAPTIVE_PHASE_NAMES;

	/**
	 * Non-intrusive serialization for one "adaptive_reward_weights" entry of RUNNING_STATS.json.
	 * RLGymCPP stays JSON-free, so the entry's fields are declared here.
	 */
	void SerializeFields(GGL::JsonArchive& ar, AdaptivePersistEntry& entry);
}
