#pragma once

namespace ForgeDebug {

	enum class RewardDisplayMode {
		EARNED,   // Pre-wrapper rewards (what the agent earned from each reward function)
		TRAINING, // Post-wrapper rewards (what the agent actually trains on, e.g. after ZeroSum)
		COUNT
	};

}
