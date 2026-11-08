#include "AdaptiveWeightSchedule.h"

#include "../Framework.h"

#include <algorithm>
#include <cmath>
#include <limits>

using namespace RLGC;

namespace {
	// A lag step ending this close to its target (as a fraction of maxWeight) lands on it, so weights settle exactly instead of decaying forever.
	constexpr float WEIGHT_SNAP_FRACTION = 1e-4f;

	// Fraction of the remaining distance a half-life process covers in `steps` timesteps.
	double HalfLifeBlend(double steps, double halfLifeSteps) {
		if (halfLifeSteps <= 0.0)
			return 1.0;
		return 1.0 - std::pow(0.5, steps / halfLifeSteps);
	}

	float DeadbandRatio(float ratio, float deadband) {
		const float offset = ratio - 1.f;
		if (std::fabs(offset) <= deadband)
			return 1.f;
		return ratio - std::copysign(deadband, offset);
	}

	// The phase this one exits to if its exit condition holds at this goodness, or `phase` itself if none does.
	AdaptivePhase LatchExitTarget(AdaptivePhase phase, float goodness, const AdaptiveWeightSettings& settings) {
		switch (phase) {
			case AdaptivePhase::Discovering:
			case AdaptivePhase::Relearning:
				return (goodness >= 1.f) ? AdaptivePhase::Settled : phase;
			case AdaptivePhase::Settled:
				if (goodness >= settings.masteredRatio)
					return AdaptivePhase::Mastered;
				if (goodness < settings.relearnRatio)
					return AdaptivePhase::Relearning;
				return phase;
			case AdaptivePhase::Mastered:
				return (goodness < 1.f) ? AdaptivePhase::Settled : phase;
		}
		return phase;
	}

	void AdvanceLatchPhase(AdaptiveKeyState& state, const AdaptiveWeightSettings& settings, float goodness, double updateTimesteps) {
		const AdaptivePhase exitTarget = LatchExitTarget(state.phase, goodness, settings);
		if (exitTarget == state.phase) {
			state.pendingExitSteps = 0.0;
			return;
		}

		state.pendingExitSteps += updateTimesteps;
		if (state.pendingExitSteps >= settings.sustainSteps) {
			state.phase = exitTarget;
			state.pendingExitSteps = 0.0;
		}
	}
}

float RLGC::ScheduleWeightMag(float ratio, const AdaptiveWeightSettings& settings, float anchorMag) {
	const float startRatio = settings.easeStartRatio;
	const float endRatio   = settings.easeEndRatio;
	const float exponent   = settings.easeExponent;

	const float minWeight = settings.minWeight;
	const float maxWeight = settings.maxWeight;

	// The curve passes through the hand-set weight at the target rate (ratio == 1).
	const float anchor = std::clamp(anchorMag, minWeight, maxWeight);

	// For a reward the weight falls as production rises, so it holds high below the band and low
	// above it. A punishment mirrors this: the magnitude rises as the bad rate exceeds the ceiling.
	const float lowHold  = settings.punishment ? minWeight : maxWeight;
	const float highHold = settings.punishment ? maxWeight : minWeight;

	if (ratio <= startRatio)
		return lowHold;

	if (ratio >= endRatio)
		return highHold;

	if (ratio <= 1.f) {
		// Ease from the low-side hold to the anchor across [startRatio, 1].
		const float u = (ratio - startRatio) / (1.f - startRatio);
		return lowHold + (anchor - lowHold) * std::pow(u, exponent);
	}

	// Ease from the anchor to the high-side hold across [1, endRatio].
	const float v = (ratio - 1.f) / (endRatio - 1.f);
	return anchor + (highHold - anchor) * (1.f - std::pow(1.f - v, exponent));
}

float RLGC::GoodnessRatio(float rawRateEma, const AdaptiveWeightSettings& settings) {
	const float target = settings.normalizedTargetPerStep;
	if (!settings.punishment)
		return rawRateEma / target;
	return (rawRateEma > 0.f) ? target / rawRateEma : std::numeric_limits<float>::infinity();
}

AdaptivePhase RLGC::InferLatchPhase(float goodness, const AdaptiveWeightSettings& settings) {
	if (goodness >= settings.masteredRatio)
		return AdaptivePhase::Mastered;
	if (goodness >= settings.relearnRatio)
		return AdaptivePhase::Settled;
	return AdaptivePhase::Discovering;
}

float RLGC::LatchTargetMag(AdaptivePhase phase, const AdaptiveWeightSettings& settings, float anchorMag) {
	switch (phase) {
		case AdaptivePhase::Settled:
			return std::clamp(anchorMag, settings.minWeight, settings.maxWeight);
		case AdaptivePhase::Mastered:
			return settings.minWeight;
		case AdaptivePhase::Discovering:
		case AdaptivePhase::Relearning:
			return settings.maxWeight;
	}
	return settings.maxWeight;
}

AdaptiveKeyState RLGC::MakeFreshKeyState(const AdaptiveWeightSettings& settings, float anchorMag) {
	AdaptiveKeyState state;
	state.weightMag = (settings.mode == AdaptiveMode::Regulate)
		? ScheduleWeightMag(0.f, settings, anchorMag)
		: LatchTargetMag(AdaptivePhase::Discovering, settings, anchorMag);
	return state;
}

void RLGC::StepKey(AdaptiveKeyState& state, const AdaptiveWeightSettings& settings, float anchorMag, float measuredRate, double updateTimesteps) {
	if (!state.rateSeeded) {
		state.rawRateEma = measuredRate;
		state.rateSeeded = true;
	} else {
		const float alpha = (float)HalfLifeBlend(updateTimesteps, settings.emaHalfLifeSteps);
		state.rawRateEma += alpha * (measuredRate - state.rawRateEma);
	}

	float targetMag;
	if (settings.mode == AdaptiveMode::Regulate) {
		const float ratio = state.rawRateEma / settings.normalizedTargetPerStep;
		targetMag = ScheduleWeightMag(DeadbandRatio(ratio, settings.regulateDeadband), settings, anchorMag);
	} else {
		AdvanceLatchPhase(state, settings, GoodnessRatio(state.rawRateEma, settings), updateTimesteps);
		targetMag = LatchTargetMag(state.phase, settings, anchorMag);
	}

	const float lag = (float)HalfLifeBlend(updateTimesteps, settings.weightHalfLifeSteps);
	state.weightMag += lag * (targetMag - state.weightMag);
	if (std::fabs(targetMag - state.weightMag) <= WEIGHT_SNAP_FRACTION * settings.maxWeight)
		state.weightMag = targetMag;
}

void RLGC::ValidateAdaptiveSettings(const AdaptiveWeightSettings& settings) {
	RG_ASSERT(settings.normalizedTargetPerStep > 0.f);
	RG_ASSERT(settings.minWeight >= 0.f && settings.minWeight <= settings.maxWeight);
	if (settings.mode == AdaptiveMode::Regulate) {
		RG_ASSERT(settings.easeStartRatio > 0.f && settings.easeStartRatio < 1.f);
		RG_ASSERT(settings.easeEndRatio > 1.f);
		RG_ASSERT(settings.regulateDeadband >= 0.f && settings.regulateDeadband < 1.f);
	} else {
		RG_ASSERT(settings.masteredRatio > 1.f);
		RG_ASSERT(settings.relearnRatio > 0.f && settings.relearnRatio < 1.f);
	}
}
