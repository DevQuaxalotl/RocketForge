#pragma once

namespace RLGC {
	/**
	 * How the adaptive controller aggregates a reward's measured production rate across players.
	 * Average: mean absolute raw reward per player-step over the whole player population of the team-size key. 
	 * Max: per arena, the top player's rate, averaged across arenas of the key.
	 * Use Max for sparse, role-concentrated event rewards so the weight targets the most active
	 * player per team rather than the population mean (which the inactive players dilute).
	 */
	enum class AdaptiveAggregation {
		Average,
		Max
	};

	/**
	 * How the controller turns a reward's measured rate into its weight.
	 * Latch: hold maxWeight until the target rate is reached, then the configured weight,
	 * then minWeight once production is well past target; climb back to maxWeight only if production collapses.
	 * Regulate: the weight is a continuous ease of measured / target (see ScheduleWeightMag).
	 */
	enum class AdaptiveMode {
		Latch,
		Regulate
	};

	/**
	 * Opt-in configuration for the closed-loop adaptive weight controller.
	 * When enabled, the reward's weight is tuned each PPO iteration from its
	 * normalized (unweighted) production rate relative to normalizedTargetPerStep; see AdaptiveMode.
	 * State is tracked per (reward name, arena player-count), so each team size adapts its own weight.
	 * enabled == false leaves the weight fixed.
	 */
	struct AdaptiveWeightSettings {
		bool enabled = false;

		/**
		 * Desired normalized (unweighted) reward magnitude per timestep per player.
		 * Equals reward-analysis' meanWeightedPerStep divided by the reward's weight.
		 * Use the AdaptiveTarget* helpers to compute this from seconds/fires.
		 */
		float normalizedTargetPerStep = 0.f;

		/**
		 * Magnitude clamp on the tuned weight. minWeight floors it; maxWeight caps the
		 * "keep pushing until discovered" ramp. Shared across team sizes.
		 */
		float minWeight = 0.f;
		float maxWeight = 0.f;

		/** Half-life of the raw-rate EMA, in collected timesteps (the same clock for every key). */
		float emaHalfLifeSteps = 2000000.f;

		/** How the measured rate is aggregated across players. See AdaptiveAggregation. */
		AdaptiveAggregation aggregation = AdaptiveAggregation::Average;

		/** See AdaptiveMode. */
		AdaptiveMode mode = AdaptiveMode::Latch;

		/** Latch: goodness ratio (see GoodnessRatio) at or above which Settled moves to Mastered. Must be > 1. */
		float masteredRatio = 2.f;

		/** Latch: goodness ratio below which Settled moves to Relearning. Must be in (0, 1). */
		float relearnRatio = 0.5f;

		/** Latch: collected timesteps a phase's exit condition must hold continuously before the phase changes. */
		float sustainSteps = 1000000.f;

		/** Half-life, in collected timesteps, of the weight's approach to its target. <= 0 jumps straight to it. */
		float weightHalfLifeSteps = 500000.f;

		/** Regulate: measured / target ratios within this distance of 1 count as exactly on target. */
		float regulateDeadband = 0.1f;

		/**
		 * Regulate: ratio (measured / target) at or below which the weight holds at its low-side anchor:
		 * maxWeight for a reward, minWeight for a punishment.
		 */
		float easeStartRatio = 0.5f;

		/**
		 * Regulate: ratio (measured / target) at or above which the weight holds at its high-side anchor:
		 * minWeight for a reward, maxWeight for a punishment.
		 */
		float easeEndRatio = 1.5f;

		/**
		 * Regulate: curve shape for the ease between a hold and the anchor, applied symmetrically to both halves.
		 * 1 = linear; 2 (default) = smooth (flat departure from the holds); larger holds nearer the
		 * anchors and transitions harder in the middle. Set by hand — not a builder parameter.
		 */
		float easeExponent = 2.f;

		/**
		 * Inverts the adaptation direction for punishments. When true, the weight magnitude RISES as
		 * the measured rate exceeds the target and FALLS as it drops below — the opposite of the reward
		 * default. Use TargetPunishment to set this.
		 */
		bool punishment = false;

		/** Enable the controller to grow a REWARD toward a target production rate; other knobs keep defaults. */
		static AdaptiveWeightSettings TargetReward(
			float normalizedTargetPerStep,
			float minWeight,
			float maxWeight,
			AdaptiveAggregation aggregation = AdaptiveAggregation::Average) {
			return Build(normalizedTargetPerStep, minWeight, maxWeight, aggregation, /*punishment*/ false);
		}

		/** Enable the controller to grow a PUNISHMENT as its measured rate exceeds the target (inverted direction). */
		static AdaptiveWeightSettings TargetPunishment(
			float normalizedTargetPerStep,
			float minWeight,
			float maxWeight,
			AdaptiveAggregation aggregation = AdaptiveAggregation::Average) {
			return Build(normalizedTargetPerStep, minWeight, maxWeight, aggregation, /*punishment*/ true);
		}

	private:
		static AdaptiveWeightSettings Build(
			float normalizedTargetPerStep,
			float minWeight,
			float maxWeight,
			AdaptiveAggregation aggregation,
			bool punishment) {
			AdaptiveWeightSettings s;
			s.enabled = true;
			s.normalizedTargetPerStep = normalizedTargetPerStep;
			s.minWeight = minWeight;
			s.maxWeight = maxWeight;
			s.aggregation = aggregation;
			s.punishment = punishment;
			return s;
		}
	};

	/**
	 * Continuous target: "fullRewardSeconds seconds' worth of full reward per overSeconds".
	 * Assumes a full reward magnitude of ~1.0 per step, so tickSkip cancels out.
	 */
	float AdaptiveTargetContinuous(const float fullRewardSeconds, const float overSeconds);

	/** Event target: "fires full triggers per overSeconds" (tickSkip converts to a per-step rate). */
	float AdaptiveTargetEvent(const float fires, const float overSeconds, const int tickSkip);

	/** Convert a reward-impact-analysis weighted per-step value to a normalized target. */
	float AdaptiveTargetFromWeightedImpact(const float weightedPerStep, const float weight);
}