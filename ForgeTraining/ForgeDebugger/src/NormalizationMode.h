#pragma once
#include <cstdint>

namespace ForgeDebug {

	enum class NormalizationMode : uint8_t {
		RAW_WEIGHTED,                  // Show raw weighted values (rawValue * weight)
		PER_REWARD_NORMALIZED,         // Each reward normalized to [-1, 1] based on its declared range * weight
		CATEGORY_NORMALIZED,           // Continuous and event rewards each normalized against their own category's combined range
		PROPORTIONAL_EVENT_CONTINUOUS  // Shows proportional difference between event and continuous reward budgets
	};

	constexpr int NORMALIZATION_MODE_COUNT = 4;

}
