#pragma once

#include "../Rewards/AdaptiveWeightSettings.h"

#include <cstdint>

namespace RLGC {

	/** Latch-mode phase of one controlled key. The numeric values are logged as the Phase metric. */
	enum class AdaptivePhase : uint8_t {
		Discovering = 0,
		Settled = 1,
		Mastered = 2,
		Relearning = 3
	};

	constexpr int ADAPTIVE_PHASE_COUNT = 4;

	/** Dynamic state of one controlled key, advanced once per controller update by StepKey. */
	struct AdaptiveKeyState {
		/** Latch only; ignored in Regulate mode. */
		AdaptivePhase phase = AdaptivePhase::Discovering;

		/** Unsigned applied weight. */
		float weightMag = 0.f;

		float rawRateEma = 0.f;

		/** False until the first measured rate seeds rawRateEma. */
		bool rateSeeded = false;

		/** Collected timesteps the current phase's exit condition has held continuously. */
		double pendingExitSteps = 0.0;
	};

	/**
	 * Evaluates the anchored weight schedule at ratio r = smoothedRate / target, returning the unsigned weight magnitude.
	 * Monotonic decreasing in r for a reward, increasing for a punishment.
	 * Anchored at maxWeight (r <= easeStartRatio), the anchor magnitude (r == 1), and minWeight (r >= easeEndRatio);
	 * the reward/punishment holds are mirrored.
	 * anchorMag is clamped into [minWeight, maxWeight] internally, so callers may pass an unclamped configured weight.
	 */
	float ScheduleWeightMag(float ratio, const AdaptiveWeightSettings& settings, float anchorMag);

	/**
	 * Measured rate relative to target, oriented so >= 1 always means "at or better than target":
	 * rawRateEma / target for a reward, target / rawRateEma for a punishment (+inf at a zero rate).
	 */
	float GoodnessRatio(float rawRateEma, const AdaptiveWeightSettings& settings);

	/** Latch phase implied by a goodness ratio alone, for keys restored without a recorded phase. */
	AdaptivePhase InferLatchPhase(float goodness, const AdaptiveWeightSettings& settings);

	/** Unsigned weight a Latch phase holds: max while Discovering/Relearning, the anchor while Settled, min while Mastered. */
	float LatchTargetMag(AdaptivePhase phase, const AdaptiveWeightSettings& settings, float anchorMag);

	/**
	 * State of a key that has no measurements yet.
	 * Latch starts Discovering at maxWeight; Regulate starts where its curve sits at a zero rate.
	 */
	AdaptiveKeyState MakeFreshKeyState(const AdaptiveWeightSettings& settings, float anchorMag);

	/**
	 * Advances one key by one controller update:
	 * folds measuredRate into the EMA, advances the Latch phase or evaluates the Regulate curve,
	 * and moves weightMag toward the resulting target.
	 * updateTimesteps is the update's total collected timesteps, the clock for every half-life and sustain time.
	 */
	void StepKey(AdaptiveKeyState& state, const AdaptiveWeightSettings& settings, float anchorMag, float measuredRate, double updateTimesteps);

	void ValidateAdaptiveSettings(const AdaptiveWeightSettings& settings);
}
