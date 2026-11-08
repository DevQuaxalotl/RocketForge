#include "AdaptiveRewardWeightController.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <iostream>
#include <set>

using namespace RLGC;

namespace {
	// The borrow rule for an uneven arena, over any key list: among keys named `name`,
	// the largest player count <= targetTotal, else the smallest. -1 if none.
	template<class NameAt, class CountAt>
	int SelectBorrowKey(int numKeys, NameAt nameAt, CountAt countAt, const std::string& name, int targetTotal) {
		int bestBelow = -1, bestBelowTotal = -1;
		int smallest = -1, smallestTotal = INT_MAX;
		for (int i = 0; i < numKeys; i++) {
			if (nameAt(i) != name)
				continue;
			const int total = countAt(i);
			if (total <= targetTotal && total > bestBelowTotal) {
				bestBelowTotal = total;
				bestBelow = i;
			}
			if (total < smallestTotal) {
				smallestTotal = total;
				smallest = i;
			}
		}
		return (bestBelow >= 0) ? bestBelow : smallest;
	}
}

void AdaptiveRewardWeightController::Init(std::vector<std::vector<WeightedReward>>& rewards, const std::vector<ArenaTeamSizes>& teamSizes) {
	const int numArenas = (int)rewards.size();

	_controlled.clear();
	_arenaRewardToControlled.assign(numArenas, {});
	_arenaRewardIsOwner.assign(numArenas, {});
	_accumSum.assign(numArenas, {});
	_accumCount.assign(numArenas, {});

	for (int arenaIdx = 0; arenaIdx < numArenas; arenaIdx++) {
		const int numRewards = (int)rewards[arenaIdx].size();
		const int playerCount = teamSizes[arenaIdx].blue + teamSizes[arenaIdx].orange;

		_arenaRewardToControlled[arenaIdx].assign(numRewards, -1);
		_arenaRewardIsOwner[arenaIdx].assign(numRewards, 0);
		_accumSum[arenaIdx].assign(numRewards, PlayerSumRow(playerCount, 0.0));
		_accumCount[arenaIdx].assign(numRewards, PlayerCountRow(playerCount, 0));
	}

	// Pass 1: fair matchups create/own their even key.
	for (int arenaIdx = 0; arenaIdx < numArenas; arenaIdx++) {
		if (teamSizes[arenaIdx].blue != teamSizes[arenaIdx].orange)
			continue;

		const int playerCount = teamSizes[arenaIdx].blue + teamSizes[arenaIdx].orange;
		for (int localRewardIdx = 0; localRewardIdx < (int)rewards[arenaIdx].size(); localRewardIdx++) {
			WeightedReward& weighted = rewards[arenaIdx][localRewardIdx];
			if (!weighted.adaptiveSettings.enabled)
				continue;

			const int controlledIdx = FindOrCreate(
				weighted.reward->GetName(), playerCount, weighted.adaptiveSettings, weighted.weight);
			_arenaRewardToControlled[arenaIdx][localRewardIdx] = controlledIdx;
			_arenaRewardIsOwner[arenaIdx][localRewardIdx] = 1;
		}
	}

	// Pass 2: uneven matchups borrow the even key of their larger team (apply only, never own).
	for (int arenaIdx = 0; arenaIdx < numArenas; arenaIdx++) {
		if (teamSizes[arenaIdx].blue == teamSizes[arenaIdx].orange)
			continue;

		const int targetTotal = 2 * std::max(teamSizes[arenaIdx].blue, teamSizes[arenaIdx].orange);
		for (int localRewardIdx = 0; localRewardIdx < (int)rewards[arenaIdx].size(); localRewardIdx++) {
			WeightedReward& weighted = rewards[arenaIdx][localRewardIdx];
			if (!weighted.adaptiveSettings.enabled)
				continue;

			const int controlledIdx = FindBorrowTarget(weighted.reward->GetName(), targetTotal);
			if (controlledIdx < 0) {
				// No even matchup exists for this reward: leave it at its configured static weight.
				static std::set<std::string> warned;
				if (warned.insert(weighted.reward->GetName()).second)
					std::cerr << "[RLGC] Adaptive weight '" << weighted.reward->GetName()
						<< "' has no even matchup to borrow from; left at static weight." << std::endl;
				continue;
			}
			_arenaRewardToControlled[arenaIdx][localRewardIdx] = controlledIdx;
			// isOwner stays 0 -> borrower: receives an applied weight but never collects.
		}
	}

	// Sync the (clamped, signed) starting weights back onto every mapped reward (owners + borrowers).
	for (int i = 0; i < (int)_controlled.size(); i++)
		ApplyOne(i, rewards);
}

int AdaptiveRewardWeightController::FindBorrowTarget(const std::string& name, int targetTotal) const {
	return SelectBorrowKey((int)_controlled.size(),
		[&](int i) -> const std::string& { return _controlled[i].name; },
		[&](int i) { return _controlled[i].playerCount; },
		name, targetTotal);
}

const AdaptivePersistEntry* AdaptiveRewardWeightController::FindEntryForArena(
	const std::vector<AdaptivePersistEntry>& entries, const std::string& name, int blue, int orange)
{
	if (blue == orange) {
		const int playerCount = blue + orange;
		for (const AdaptivePersistEntry& entry : entries)
			if (entry.name == name && entry.playerCount == playerCount)
				return &entry;
		return nullptr;
	}

	const int idx = SelectBorrowKey((int)entries.size(),
		[&](int i) -> const std::string& { return entries[i].name; },
		[&](int i) { return entries[i].playerCount; },
		name, 2 * std::max(blue, orange));
	return (idx >= 0) ? &entries[idx] : nullptr;
}

int AdaptiveRewardWeightController::FindOrCreate(const std::string& name, int playerCount, const AdaptiveWeightSettings& settings, float startWeight) {
	ValidateAdaptiveSettings(settings);

	for (int i = 0; i < (int)_controlled.size(); i++)
		if (_controlled[i].name == name && _controlled[i].playerCount == playerCount)
			return i;

	Controlled c;
	c.name = name;
	c.playerCount = playerCount;
	c.settings = settings;
	c.sign = (startWeight < 0.f) ? -1.f : 1.f;
	c.anchorMag = std::clamp(std::fabs(startWeight), settings.minWeight, settings.maxWeight);
	c.state = MakeFreshKeyState(settings, c.anchorMag);

	_controlled.push_back(c);
	return (int)_controlled.size() - 1;
}

bool AdaptiveRewardWeightController::IsEnabled(int arenaIdx, int localRewardIdx) const {
	return !_controlled.empty() && _arenaRewardToControlled[arenaIdx][localRewardIdx] >= 0;
}

bool AdaptiveRewardWeightController::IsCollecting(int arenaIdx, int localRewardIdx) const {
	return !_controlled.empty()
		&& _arenaRewardToControlled[arenaIdx][localRewardIdx] >= 0
		&& _arenaRewardIsOwner[arenaIdx][localRewardIdx];
}

void AdaptiveRewardWeightController::Accumulate(int arenaIdx, int localRewardIdx, int playerIdx, float absValue) {
	RG_ASSERT(playerIdx >= 0 && playerIdx < (int)_accumSum[arenaIdx][localRewardIdx].size());

	_accumSum[arenaIdx][localRewardIdx][playerIdx] += absValue;
	_accumCount[arenaIdx][localRewardIdx][playerIdx] += 1;
}

std::vector<AdaptiveRewardWeightController::ReportEntry>
AdaptiveRewardWeightController::UpdateAndApply(std::vector<std::vector<WeightedReward>>& rewards, double updateTimesteps) {
	std::vector<ReportEntry> report;
	if (_controlled.empty())
		return report;

	// Merge this iteration's arena-local accumulators into per-key totals, then reset them.
	std::vector<double>  iterSum(_controlled.size(), 0.0);
	std::vector<int64_t> iterCount(_controlled.size(), 0);

	// Max aggregation: sum of each contributing arena's top-player rate, and the number of arenas
	// that contributed one (an arena contributes only if at least one of its players logged a step).
	std::vector<double> iterMaxRateSum(_controlled.size(), 0.0);
	std::vector<int>    iterMaxArenaCount(_controlled.size(), 0);

	for (int arenaIdx = 0; arenaIdx < (int)_arenaRewardToControlled.size(); arenaIdx++) {
		for (int localRewardIdx = 0; localRewardIdx < (int)_arenaRewardToControlled[arenaIdx].size(); localRewardIdx++) {
			const int idx = _arenaRewardToControlled[arenaIdx][localRewardIdx];
			if (idx < 0)
				continue;

			PlayerSumRow&   playerSums         = _accumSum[arenaIdx][localRewardIdx];
			PlayerCountRow& playerSampleCounts = _accumCount[arenaIdx][localRewardIdx];

			double arenaMaxRate = 0.0;
			bool arenaHadSteps = false;

			for (int playerIdx = 0; playerIdx < (int)playerSums.size(); playerIdx++) {
				const double playerSum    = playerSums[playerIdx];
				const int64_t playerCount = playerSampleCounts[playerIdx];

				iterSum[idx]   += playerSum;
				iterCount[idx] += playerCount;

				if (playerCount > 0) {
					const double playerRate = playerSum / (double)playerCount;
					arenaMaxRate = std::max(arenaMaxRate, playerRate);
					arenaHadSteps = true;
				}

				playerSums[playerIdx]         = 0.0;
				playerSampleCounts[playerIdx] = 0;
			}

			if (arenaHadSteps) {
				iterMaxRateSum[idx]    += arenaMaxRate;
				iterMaxArenaCount[idx] += 1;
			}
		}
	}

	report.reserve(_controlled.size());
	for (int i = 0; i < (int)_controlled.size(); i++) {
		Controlled& c = _controlled[i];
		if (iterCount[i] > 0) {
			float rawRate;
			if (c.settings.aggregation == AdaptiveAggregation::Max) {
				// Mean of the per-arena top-player rates.
				rawRate = (iterMaxArenaCount[i] > 0)
					? (float)(iterMaxRateSum[i] / (double)iterMaxArenaCount[i])
					: 0.f;
			} else {
				rawRate = (float)(iterSum[i] / (double)iterCount[i]);
			}

			StepKey(c.state, c.settings, c.anchorMag, rawRate, updateTimesteps);
		}

		ApplyOne(i, rewards);

		ReportEntry entry;
		entry.name = c.name;
		entry.playerCount = c.playerCount;
		entry.ratio = c.state.rawRateEma / c.settings.normalizedTargetPerStep;
		entry.weight = ComputeWeight(c);
		report.push_back(entry);
	}

	return report;
}

float AdaptiveRewardWeightController::ComputeWeight(const Controlled& c) const {
	return c.sign * c.state.weightMag;
}

void AdaptiveRewardWeightController::ApplyOne(int controlledIdx, std::vector<std::vector<WeightedReward>>& rewards) {
	const float weight = ComputeWeight(_controlled[controlledIdx]);

	for (int arenaIdx = 0; arenaIdx < (int)_arenaRewardToControlled.size(); arenaIdx++)
		for (int localRewardIdx = 0; localRewardIdx < (int)_arenaRewardToControlled[arenaIdx].size(); localRewardIdx++)
			if (_arenaRewardToControlled[arenaIdx][localRewardIdx] == controlledIdx)
				rewards[arenaIdx][localRewardIdx].weight = weight;
}

std::vector<AdaptivePersistEntry> AdaptiveRewardWeightController::ExportState() const {
	std::vector<AdaptivePersistEntry> entries;
	entries.reserve(_controlled.size());
	for (const Controlled& c : _controlled) {
		AdaptivePersistEntry entry;
		entry.name = c.name;
		entry.playerCount = c.playerCount;
		entry.weight = ComputeWeight(c);
		entry.rawRateEma = c.state.rawRateEma;
		if (c.settings.mode == AdaptiveMode::Latch) {
			entry.phase = c.state.phase;
			entry.pendingExitSteps = c.state.pendingExitSteps;
		}
		entries.push_back(entry);
	}
	return entries;
}

void AdaptiveRewardWeightController::ImportState(const std::vector<AdaptivePersistEntry>& entries, std::vector<std::vector<WeightedReward>>& rewards) {
	for (const AdaptivePersistEntry& entry : entries) {
		for (int i = 0; i < (int)_controlled.size(); i++) {
			Controlled& c = _controlled[i];
			if (c.name != entry.name || c.playerCount != entry.playerCount)
				continue;

			// The weight is restored rather than re-derived so a resume never jumps.
			// Clamping applies min/max edits made since the save; the lag then eases toward the phase target.
			c.state.weightMag = std::clamp(std::fabs(entry.weight), c.settings.minWeight, c.settings.maxWeight);
			c.state.rawRateEma = entry.rawRateEma;
			c.state.rateSeeded = true;

			if (c.settings.mode == AdaptiveMode::Latch) {
				if (entry.phase.has_value()) {
					c.state.phase = *entry.phase;
					c.state.pendingExitSteps = entry.pendingExitSteps;
				} else {
					c.state.phase = InferLatchPhase(GoodnessRatio(entry.rawRateEma, c.settings), c.settings);
					c.state.pendingExitSteps = 0.0;
				}
			}

			ApplyOne(i, rewards);
			break;
		}
	}
}
