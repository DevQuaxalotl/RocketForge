#pragma once

#include "../Rewards/Reward.h"
#include "AdaptiveWeightSchedule.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace RLGC {

	/** One controlled key's persisted state, keyed by (reward name, arena player count). */
	struct AdaptivePersistEntry {
		std::string name;
		int playerCount = 0;

		/** Signed applied weight. */
		float weight = 0.f;

		float rawRateEma = 0.f;
		
		std::optional<AdaptivePhase> phase;

		double pendingExitSteps = 0.0;
	};

	/**
	 * Closed-loop adaptive reward weight controller (owned by EnvSet).
	 *
	 * During collection, Accumulate() adds one player-step's |raw reward| into arena-local storage
	 * (thread-safe: each arena writes only its own row). Once per PPO iteration, in the serial boundary
	 * between collection and learning, the Learner calls UpdateAndApply(), which merges the arena
	 * accumulators per (reward name, player count) key, folds them into a raw-rate EMA, then advances each
	 * key with StepKey on the shared timestep clock, and returns per-key stats for metrics.
	 */
	class AdaptiveRewardWeightController {
	public:
		/** Per-iteration stat for one controlled key (for wandb reporting). */
		struct ReportEntry {
			std::string name;
			int playerCount = 0;
			float weight = 0.f;   // signed effective weight after this update
			float ratio = 0.f;    // measured normalized rate / target (1 == on target)
		};

		/** Per-arena team split, used to distinguish fair (blue == orange) from uneven matchups. */
		struct ArenaTeamSizes {
			int blue = 0;
			int orange = 0;
		};

		/**
		 * Builds tracking state from the resolved per-arena reward lists and team splits.
		 * teamSizes[a] is arena a's (blue, orange) car counts. Adaptive keys are created ONLY for
		 * fair matchups (blue == orange), keyed by (GetName(), blue + orange); those arenas "own"
		 * their key and are collection-eligible. Uneven arenas create no key — each of their adaptive
		 * rewards borrows the even key for 2 * max(blue, orange) (nearest tracked even key <= that,
		 * else smallest tracked) for weight application only. A reward with no even key anywhere is
		 * left at its configured static weight.
		 */
		void Init(std::vector<std::vector<WeightedReward>>& rewards, const std::vector<ArenaTeamSizes>& teamSizes);

		/** True if this arena's local reward index is under adaptive control. */
		bool IsEnabled(int arenaIdx, int localRewardIdx) const;

		/** True if this (arena, reward) owns its key (a fair matchup) — the only sites that collect. */
		bool IsCollecting(int arenaIdx, int localRewardIdx) const;

		/** Adds one player-step's absolute reward to arena-local storage. Call once per player per step. */
		void Accumulate(int arenaIdx, int localRewardIdx, int playerIdx, float absValue);

		/**
		 * Serial-window only: merge accumulators, step every key that collected samples, rewrite weights, return stats.
		 * updateTimesteps is the iteration's total collected timesteps, the shared clock for every key.
		 */
		std::vector<ReportEntry> UpdateAndApply(std::vector<std::vector<WeightedReward>>& rewards, double updateTimesteps);

		/** Snapshots controller state (one entry per tracked key) for persistence. */
		std::vector<AdaptivePersistEntry> ExportState() const;

		/**
		 * Restores state by (name, player count) and re-applies weights.
		 * Unknown keys are ignored; Latch entries saved without a phase get an inferred one.
		 */
		void ImportState(const std::vector<AdaptivePersistEntry>& entries, std::vector<std::vector<WeightedReward>>& rewards);

		/**
		 * Persisted-state counterpart of Init's key selection: the entry an arena of (blue, orange) takes
		 * its weight from for `name` — its own key when fair, the borrowed even key otherwise.
		 * Lets offline consumers of ExportState() output (no live controller) resolve a checkpoint's
		 * effective weight for any matchup. nullptr if no entry applies.
		 */
		static const AdaptivePersistEntry* FindEntryForArena(
			const std::vector<AdaptivePersistEntry>& entries, const std::string& name, int blue, int orange);

	private:
		struct Controlled {
			std::string name;
			int playerCount = 0;
			AdaptiveWeightSettings settings;

			// Sign of the configured weight; the schedule yields a magnitude, this restores the sign.
			float sign = 1.f;

			// |configured weight|, clamped to [minWeight, maxWeight]: the reference weight a key's schedule is built around.
			float anchorMag = 0.f;

			AdaptiveKeyState state;
		};

		int FindOrCreate(const std::string& name, int playerCount, const AdaptiveWeightSettings& settings, float startWeight);

		/** Best even key for an uneven arena: largest tracked total <= targetTotal, else smallest. -1 if none. */
		int FindBorrowTarget(const std::string& name, int targetTotal) const;

		void ApplyOne(int controlledIdx, std::vector<std::vector<WeightedReward>>& rewards);
		float ComputeWeight(const Controlled& c) const;

		std::vector<Controlled> _controlled;

		/** [arenaIdx][localRewardIdx] -> index into _controlled, or -1 if untracked. */
		std::vector<std::vector<int>> _arenaRewardToControlled;

		/** [arenaIdx][localRewardIdx] -> true if this site owns its key (fair matchup). */
		std::vector<std::vector<uint8_t>> _arenaRewardIsOwner;

		/** One (arena, reward)'s per-player accumulator row, indexed by player index. */
		using PlayerSumRow   = std::vector<double>;
		using PlayerCountRow = std::vector<int64_t>;

		/** Arena-local per-player accumulators, indexed [arenaIdx][localRewardIdx] -> per-player row. */
		std::vector<std::vector<PlayerSumRow>>   _accumSum;
		std::vector<std::vector<PlayerCountRow>> _accumCount;
	};
}
