#include "AdaptiveWeightSettings.h"

#include "RLGymCPP/Framework.h"

using namespace RLGC;

static constexpr float ADAPTIVE_TICK_RATE = 120.f;

float RLGC::AdaptiveTargetContinuous(const float fullRewardSeconds, const float overSeconds) {
	return fullRewardSeconds / overSeconds;
}

float RLGC::AdaptiveTargetEvent(const float fires, const float overSeconds, const int tickSkip) {
	const float stepsPerSecond = ADAPTIVE_TICK_RATE / (float)tickSkip;
	return fires / (overSeconds * stepsPerSecond);
}

float RLGC::AdaptiveTargetFromWeightedImpact(const float weightedPerStep, const float weight) {
	RG_ASSERT(weight != 0.f);
	return weightedPerStep / weight;
}