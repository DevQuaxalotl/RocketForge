#pragma once
#include "RewardDebugSnapshot.h"
#include "EventDecayTracker.h"
#include <RLGymCPP/Rewards/Reward.h>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ForgeDebug {

	struct CollectorConfig {
		NormalizationMode normMode = NormalizationMode::PER_REWARD_NORMALIZED;
		RewardDisplayMode rewardDisplayMode = RewardDisplayMode::TRAINING;
		float eventDecayDuration = 2.0f;
		bool forceDebugAll = false;
	};

	class RewardDebugCollector {
	public:
		CollectorConfig config;

		/** Initialize with the reward list from an arena. Call once after env creation. */
		void Initialize(const std::vector<RLGC::WeightedReward>& rewards);

		/**
		 * Swaps in the reward list of a replacement arena, resetting per-episode state and the player index as Initialize does.
		 * Unlike Initialize, it keeps the episode index and every debug-enabled choice made by name, including for rewards the previous list lacked.
		 */
		void Rebind(const std::vector<RLGC::WeightedReward>& rewards);

		/**
		 * Refresh cached weights from the live reward list (e.g. after the adaptive controller
		 * updates them). Leaves roster, event trackers, and episode totals intact.
		 */
		void SyncWeights(const std::vector<RLGC::WeightedReward>& src);

		/**
		 * Called each step after rewards are computed.
		 * earnedRewards: pre-wrapper values (what each reward function returned).
		 * trainingRewards: post-wrapper values (e.g. after ZeroSum transformation).
		 * Only rewards with debugEnabled are recorded in the snapshot.
		 * Event decay is always driven by earnedRewards so event firing is detected correctly.
		 */
		void RecordStep(const std::vector<float>& earnedRewards,
			const std::vector<float>& trainingRewards,
			float deltaTime,
			const RLGC::Player& player,
			const RLGC::GameState& state);

		/** Critic estimate attached to every following snapshot until replaced; empty when no critic estimate is available. */
		void SetCriticEstimate(std::optional<CriticDebugData> estimate);

		/** Marks the start of a new episode, so per-episode GUI state keyed on snapshot.episodeIndex can reset. */
		void BeginEpisode();

		/** Get the latest snapshot with normalization applied. Only contains debug-enabled rewards. */
		const RewardDebugSnapshot& GetSnapshot() const;

		/** Change normalization mode at runtime */
		void SetNormalizationMode(NormalizationMode mode);

		/** Cycle to the next normalization mode */
		void CycleNormalizationMode();

		/** Cycle the reward display mode between Earned and Training */
		void CycleRewardDisplayMode();

		/** Set the reward display mode directly */
		void SetRewardDisplayMode(RewardDisplayMode mode);

		/** Enable/disable debug on a single reward by index (the user's desired selection). */
		void SetRewardDebugEnabled(int index, bool enabled);

		/** Enable/disable debug on a reward by name; remembered for a later Rebind even if the current list lacks it. */
		void SetRewardDebugEnabled(const std::string& name, bool enabled);

		/** Replace the desired selection: each reward, now or after a later Rebind, is enabled iff its name is in enabledNames. */
		void SetDebugSelection(const std::vector<std::string>& enabledNames);

		/** Build the full roster (all rewards + desired state). Used by RecordStep and tests. */
		std::vector<RewardRosterEntry> BuildRoster() const;

		/** Force-enable or disable debug on all rewards */
		void SetForceDebugAll(bool enabled);

		/** Toggle force debug all */
		void ToggleForceDebugAll();

		/** Cycle the player index for debugging */
		void CyclePlayerIndex(const RLGC::GameState& state);

		/** Set the player index directly (clamped to [0, playerCount-1] at next RecordStep) **/
		void SetPlayerIndex(int idx);

		/** Get current player index */
		int GetPlayerIndex() const { return playerIndex; }

		/** Reset per-episode totals without changing debug settings or selected player. **/
		void ResetEpisodeTotals();

		/** 
		* Soft reset: clears frame counters, episode totals, and snapshot.
		* Preserves the reward list, desiredDebugEnabled, and event trackers — only
		* valid between episodes on the same stage (same Initialize call). 
		*/
		void Reset();

	private:
		std::vector<RLGC::WeightedReward> rewards;

		/** One per EVENT reward */
		std::vector<EventDecayTracker> eventTrackers;

		/** Maps reward index to eventTracker index (-1 if continuous) */
		std::vector<int> eventTrackerMap;

		/** Running sum of abs(weighted reward) for the selected player in the current episode. **/
		std::vector<float> episodeAbsWeightedSums;

		/** Running sum of abs(head contribution) per critic head name, over the same span as episodeAbsWeightedSums. */
		std::unordered_map<std::string, float> episodeAbsHeadContributionSums;

		/** User's per-reward selection (1 = desired on). Effective state is
		    forceDebugAll || desiredDebugEnabled[i], written to each reward. */
		std::vector<uint8_t> desiredDebugEnabled;

		/** Desired debug state chosen per reward name; outlives Rebind so a choice survives lists that lack that reward. */
		std::unordered_map<std::string, bool> desiredByName;

		/** Stable names/categories with the current desired selection. */
		std::vector<RewardRosterEntry> roster;

		RewardDebugSnapshot latestSnapshot;
		uint64_t frameNumber = 0;
		int playerIndex = 0;
		std::optional<CriticDebugData> criticEstimate;
		double gameTimeSeconds = 0.0;
		uint64_t episodeIndex = 0;

		/** Rebuilds all reward-list state from rewards, taking each desired state from desiredByName or the reward's default. */
		void BindRewards(const std::vector<RLGC::WeightedReward>& rewards);

		void ApplyNormalization(RewardDebugSnapshot& snapshot) const;

		/** Re-apply effective debug state (forceDebugAll || desired) to every reward. */
		void RecomputeEffective();
		void NormalizeRawWeighted(RewardDebugSnapshot& snapshot) const;
		void NormalizePerReward(RewardDebugSnapshot& snapshot) const;
		void NormalizeByCategory(RewardDebugSnapshot& snapshot) const;
		void NormalizeProportional(RewardDebugSnapshot& snapshot) const;
	};

}
