#include "RewardDebugCollector.h"
#include <cmath>
#include <algorithm>

namespace ForgeDebug {

	void RewardDebugCollector::Initialize(const std::vector<RLGC::WeightedReward>& inRewards) {
		desiredByName.clear();
		episodeIndex = 0;
		BindRewards(inRewards);
	}

	void RewardDebugCollector::Rebind(const std::vector<RLGC::WeightedReward>& inRewards) {
		BindRewards(inRewards);
	}

	void RewardDebugCollector::BindRewards(const std::vector<RLGC::WeightedReward>& inRewards) {
		rewards = inRewards;
		eventTrackers.clear();
		eventTrackerMap.clear();
		episodeAbsWeightedSums.assign(rewards.size(), 0.f);
		episodeAbsHeadContributionSums.clear();
		frameNumber = 0;
		playerIndex = 0;
		gameTimeSeconds = 0.0;
		criticEstimate.reset();
		latestSnapshot = {};

		for (int i = 0; i < (int)rewards.size(); i++) {
			const RLGC::WeightedReward& wr = rewards[i];
			const RLGC::RewardCategory cat = wr.reward->GetCategory();

			if (cat == RLGC::RewardCategory::EVENT) {
				EventDecayTracker tracker;
				tracker.decayDuration = config.eventDecayDuration;
				eventTrackerMap.push_back((int)eventTrackers.size());
				eventTrackers.push_back(tracker);
			} else {
				eventTrackerMap.push_back(-1);
			}
		}

		roster.clear();
		roster.reserve(rewards.size());
		for (int i = 0; i < (int)rewards.size(); ++i) {
			RewardRosterEntry entry;
			entry.name = rewards[i].reward->GetName();
			entry.category = rewards[i].reward->GetCategory();
			entry.index = i;
			roster.push_back(std::move(entry));
		}

		desiredDebugEnabled.assign(rewards.size(), 0);
		for (int i = 0; i < (int)rewards.size(); i++) {
			const auto chosen = desiredByName.find(roster[i].name);
			const bool desired = chosen != desiredByName.end() ? chosen->second : rewards[i].reward->IsDebugEnabled();
			desiredDebugEnabled[i] = desired ? 1 : 0;
		}

		RecomputeEffective();
	}

	void RewardDebugCollector::SyncWeights(const std::vector<RLGC::WeightedReward>& src) {
		const int count = std::min((int)rewards.size(), (int)src.size());
		for (int i = 0; i < count; i++)
			rewards[i].weight = src[i].weight;
	}

	void RewardDebugCollector::RecordStep(const std::vector<float>& earnedRewards,
		const std::vector<float>& trainingRewards,
		float deltaTime,
		const RLGC::Player& player,
		const RLGC::GameState& state) {

		// Step 1: Update ALL event trackers.
		// In Training mode, use training rewards so that events triggered by teammates/opponents
		// (whose value ZeroSum spreads to the sampled player) also fire the decay tracker.
		// For event rewards, ZeroSum of all-zero inputs is also zero, so no false positives.
		// In Earned mode, use earned rewards so only events the sampled player triggered fire.
		const std::vector<float>& eventSourceValues = (config.rewardDisplayMode == RewardDisplayMode::TRAINING)
			? trainingRewards : earnedRewards;

		for (int i = 0; i < (int)rewards.size(); i++) {
			const int trackerIdx = eventTrackerMap[i];
			if (trackerIdx < 0)
				continue;

			const float eventVal = (i < (int)eventSourceValues.size()) ? eventSourceValues[i] : 0.f;
			eventTrackers[trackerIdx].Update(deltaTime);
			if (std::abs(eventVal) > 1e-6f)
				eventTrackers[trackerIdx].RecordEvent(eventVal);
		}

		gameTimeSeconds += deltaTime;

		// Step 2: Select which reward values to display and detect events from, based on current mode
		const std::vector<float>& displayValues = eventSourceValues;

		// Step 3: Build snapshot from only debug-enabled rewards
		RewardDebugSnapshot snapshot;
		snapshot.frameNumber = frameNumber++;
		snapshot.roster = roster;
		snapshot.rewards.reserve(rewards.size());
		snapshot.normMode = config.normMode;
		snapshot.rewardDisplayMode = config.rewardDisplayMode;
		snapshot.forceDebugAll = config.forceDebugAll;
		snapshot.playerIndex = playerIndex;
		snapshot.playerCarId = player.carId;
		snapshot.playerTeam = (int)player.team;
		snapshot.totalContinuousWeighted = 0.f;
		snapshot.totalEventWeighted = 0.f;
		snapshot.totalWeighted = 0.f;
		snapshot.gameTimeSeconds = gameTimeSeconds;
		snapshot.episodeIndex = episodeIndex;
		snapshot.critic = criticEstimate;

		if (snapshot.critic) {
			for (CriticHeadDebugData& head : snapshot.critic->heads) {
				float& sum = episodeAbsHeadContributionSums[head.name];
				sum += std::abs(head.contribution);
				head.episodeAbsContributionSum = sum;
			}
		}

		for (int i = 0; i < (int)rewards.size(); i++) {
			const RLGC::WeightedReward& wr = rewards[i];
			const float rawVal = (i < (int)displayValues.size()) ? displayValues[i] : 0.f;
			const float weightedVal = rawVal * wr.weight;

			if (i >= (int)episodeAbsWeightedSums.size())
				episodeAbsWeightedSums.resize(rewards.size(), 0.f);
			episodeAbsWeightedSums[i] += std::abs(weightedVal);

			if (!wr.reward->IsDebugEnabled())
				continue;

			RewardDebugData data;
			data.name = roster[i].name;
			data.category = roster[i].category;
			data.rawValue = rawVal;
			data.weight = wr.weight;
			data.weightedValue = weightedVal;
			data.episodeAbsWeightedSum = episodeAbsWeightedSums[i];
			data.declaredRange = wr.reward->GetValueRange();
			data.eventFiredThisFrame = false;
			data.eventDecayValue = 0.f;
			data.normalizedValue = 0.f;
			data.debugLines = wr.reward->GetDebugLines(player, state);

			if (data.category == RLGC::RewardCategory::EVENT) {
				const int trackerIdx = eventTrackerMap[i];
				data.eventDecayValue = eventTrackers[trackerIdx].GetDecayValue();
				data.eventFiredThisFrame = std::abs(rawVal) > 1e-6f;
				snapshot.totalEventWeighted += data.weightedValue;
			} else {
				snapshot.totalContinuousWeighted += data.weightedValue;
			}

			snapshot.totalWeighted += data.weightedValue;
			snapshot.rewards.push_back(std::move(data));
		}

		ApplyNormalization(snapshot);
		latestSnapshot = std::move(snapshot);
	}

	const RewardDebugSnapshot& RewardDebugCollector::GetSnapshot() const {
		return latestSnapshot;
	}

	void RewardDebugCollector::SetCriticEstimate(std::optional<CriticDebugData> estimate) {
		criticEstimate = std::move(estimate);
	}

	void RewardDebugCollector::BeginEpisode() {
		episodeIndex++;
	}

	void RewardDebugCollector::SetNormalizationMode(NormalizationMode mode) {
		config.normMode = mode;
	}

	void RewardDebugCollector::CycleNormalizationMode() {
		const int next = ((int)config.normMode + 1) % NORMALIZATION_MODE_COUNT;
		config.normMode = (NormalizationMode)next;
	}

	void RewardDebugCollector::CycleRewardDisplayMode() {
		const int next = ((int)config.rewardDisplayMode + 1) % (int)RewardDisplayMode::COUNT;
		SetRewardDisplayMode((RewardDisplayMode)next);
	}

	void RewardDebugCollector::SetRewardDisplayMode(RewardDisplayMode mode) {
		if (config.rewardDisplayMode != mode) {
			config.rewardDisplayMode = mode;
			ResetEpisodeTotals();
		}
	}

	void RewardDebugCollector::SetForceDebugAll(bool enabled) {
		config.forceDebugAll = enabled;
		RecomputeEffective();
	}

	void RewardDebugCollector::ToggleForceDebugAll() {
		SetForceDebugAll(!config.forceDebugAll);
	}

	void RewardDebugCollector::RecomputeEffective() {
		for (int i = 0; i < (int)rewards.size(); i++) {
			const bool effective = config.forceDebugAll
				|| (i < (int)desiredDebugEnabled.size() && desiredDebugEnabled[i]);
			rewards[i].reward->SetDebugEnabled(effective);
			roster[i].desiredEnabled = desiredDebugEnabled[i] != 0;
		}
	}

	void RewardDebugCollector::SetRewardDebugEnabled(int index, bool enabled) {
		if (index < 0 || index >= (int)desiredDebugEnabled.size())
			return;
		desiredDebugEnabled[index] = enabled ? 1 : 0;
		desiredByName[roster[index].name] = enabled;
		RecomputeEffective();
	}

	void RewardDebugCollector::SetRewardDebugEnabled(const std::string& name, bool enabled) {
		desiredByName[name] = enabled;
		for (int i = 0; i < (int)roster.size(); i++)
			if (roster[i].name == name)
				desiredDebugEnabled[i] = enabled ? 1 : 0;

		RecomputeEffective();
	}

	void RewardDebugCollector::SetDebugSelection(const std::vector<std::string>& enabledNames) {
		for (auto& [name, desired] : desiredByName)
			desired = false;
		for (const RewardRosterEntry& entry : roster)
			desiredByName[entry.name] = false;
		for (const std::string& name : enabledNames)
			desiredByName[name] = true;

		for (int i = 0; i < (int)roster.size(); i++)
			desiredDebugEnabled[i] = desiredByName[roster[i].name] ? 1 : 0;

		RecomputeEffective();
	}

	std::vector<RewardRosterEntry> RewardDebugCollector::BuildRoster() const {
		return roster;
	}

	void RewardDebugCollector::CyclePlayerIndex(const RLGC::GameState& state) {
		if (state.players.empty())
			return;
		SetPlayerIndex((playerIndex + 1) % (int)state.players.size());
	}

	void RewardDebugCollector::SetPlayerIndex(int idx) {
		idx = std::max(0, idx);
		if (playerIndex != idx) {
			playerIndex = idx;
			ResetEpisodeTotals();
		}
	}

	void RewardDebugCollector::ResetEpisodeTotals() {
		std::fill(episodeAbsWeightedSums.begin(), episodeAbsWeightedSums.end(), 0.f);
		episodeAbsHeadContributionSums.clear();
	}

	void RewardDebugCollector::Reset() {
		frameNumber = 0;
		playerIndex = 0;
		gameTimeSeconds = 0.0;
		episodeIndex++;
		criticEstimate.reset();
		latestSnapshot = {};
		ResetEpisodeTotals();
		for (auto& tracker : eventTrackers)
			tracker.Reset();
	}

	void RewardDebugCollector::ApplyNormalization(RewardDebugSnapshot& snapshot) const {
		switch (config.normMode) {
		case NormalizationMode::RAW_WEIGHTED:
			NormalizeRawWeighted(snapshot);
			break;
		case NormalizationMode::PER_REWARD_NORMALIZED:
			NormalizePerReward(snapshot);
			break;
		case NormalizationMode::CATEGORY_NORMALIZED:
			NormalizeByCategory(snapshot);
			break;
		case NormalizationMode::PROPORTIONAL_EVENT_CONTINUOUS:
			NormalizeProportional(snapshot);
			break;
		}
	}

	void RewardDebugCollector::NormalizeRawWeighted(RewardDebugSnapshot& snapshot) const {
		for (auto& rd : snapshot.rewards)
			rd.normalizedValue = rd.weightedValue;
	}

	void RewardDebugCollector::NormalizePerReward(RewardDebugSnapshot& snapshot) const {
		for (auto& rd : snapshot.rewards) {
			const float weightedMin = rd.declaredRange.minValue * rd.weight;
			const float weightedMax = rd.declaredRange.maxValue * rd.weight;

			if (rd.category == RLGC::RewardCategory::EVENT) {
				// Unipolar: idle (0) maps to 0, max maps to 1
				if (std::abs(weightedMax) < 1e-9f) {
					rd.normalizedValue = 0.f;
				} else {
					rd.normalizedValue = rd.weightedValue / weightedMax;
					rd.normalizedValue = std::clamp(rd.normalizedValue, -1.f, 1.f);
				}
			} else {
				const float range = weightedMax - weightedMin;
				if (std::abs(range) < 1e-9f) {
					rd.normalizedValue = 0.f;
				} else if (weightedMin >= 0.f) {
					// Unipolar: reward only produces non-negative values; 0 stays 0, max maps to 1
					rd.normalizedValue = rd.weightedValue / weightedMax;
					rd.normalizedValue = std::clamp(rd.normalizedValue, -1.f, 1.f);
				} else {
					// Bipolar: reward can be negative; min maps to -1, max maps to 1
					rd.normalizedValue = 2.f * (rd.weightedValue - weightedMin) / range - 1.f;
					rd.normalizedValue = std::clamp(rd.normalizedValue, -1.f, 1.f);
				}
			}
		}
	}

	void RewardDebugCollector::NormalizeByCategory(RewardDebugSnapshot& snapshot) const {
		// Compute the signed min/max across each category's weighted declared ranges
		float contMin = 0.f, contMax = 0.f;
		float evtMax = 0.f;
		for (const auto& rd : snapshot.rewards) {
			const float minW = rd.declaredRange.minValue * rd.weight;
			const float maxW = rd.declaredRange.maxValue * rd.weight;
			if (rd.category == RLGC::RewardCategory::CONTINUOUS) {
				contMin = std::min(contMin, minW);
				contMax = std::max(contMax, maxW);
			} else {
				evtMax = std::max(evtMax, maxW);
			}
		}

		// Normalize all continuous rewards against the largest absolute magnitude in the category.
		// This preserves 0->0 for every reward regardless of whether its own range is unipolar or bipolar,
		// and regardless of the asymmetry of other rewards in the category.
		const float contAbsMax = std::max(std::abs(contMin), contMax);

		for (auto& rd : snapshot.rewards) {
			if (rd.category == RLGC::RewardCategory::EVENT) {
				// Unipolar: idle (0) maps to 0, catMax maps to 1
				if (std::abs(evtMax) < 1e-9f) {
					rd.normalizedValue = 0.f;
				} else {
					rd.normalizedValue = rd.weightedValue / evtMax;
					rd.normalizedValue = std::clamp(rd.normalizedValue, -1.f, 1.f);
				}
			} else {
				if (std::abs(contAbsMax) < 1e-9f) {
					rd.normalizedValue = 0.f;
				} else {
					rd.normalizedValue = rd.weightedValue / contAbsMax;
					rd.normalizedValue = std::clamp(rd.normalizedValue, -1.f, 1.f);
				}
			}
		}
	}

	void RewardDebugCollector::NormalizeProportional(RewardDebugSnapshot& snapshot) const {
		// Use the single reward with the highest weighted max as the reference, not the sum
		float globalMax = 0.f;
		for (const auto& rd : snapshot.rewards)
			globalMax = std::max(globalMax, rd.declaredRange.maxValue * rd.weight);

		if (std::abs(globalMax) < 1e-9f) {
			for (auto& rd : snapshot.rewards)
				rd.normalizedValue = 0.f;
			return;
		}

		for (auto& rd : snapshot.rewards) {
			rd.normalizedValue = rd.weightedValue / globalMax;
			rd.normalizedValue = std::clamp(rd.normalizedValue, -1.f, 1.f);
		}
	}

}
