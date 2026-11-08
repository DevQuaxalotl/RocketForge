#include "RewardHeadLayout.h"

#include <unordered_map>
#include <unordered_set>

using namespace std;
using namespace RLGC;

vector<string> RLGC::RewardHeadLayout::FindDuplicateNames(const vector<vector<WeightedReward>>& arenaRewards) {
	vector<string> duplicates = {};
	unordered_set<string> reported = {};

	for (const vector<WeightedReward>& rewards : arenaRewards) {
		unordered_set<string> seenInArena = {};
		for (const WeightedReward& weightedReward : rewards) {
			const string name = weightedReward.reward->GetName();
			if (!seenInArena.insert(name).second && reported.insert(name).second)
				duplicates.push_back(name);
		}
	}

	return duplicates;
}

RewardHeadLayout RLGC::RewardHeadLayout::Build(const vector<vector<WeightedReward>>& arenaRewards) {
	const vector<string> duplicates = FindDuplicateNames(arenaRewards);
	if (!duplicates.empty()) {
		RG_ERR_CLOSE("RewardHeadLayout: reward \"" << duplicates.front() << "\" is listed more than once in one arena; "
			"value heads are keyed by reward name");
	}

	RewardHeadLayout layout = {};
	unordered_map<string, int> headIndexByName = {};
	layout._arenaRewardHeads.resize(arenaRewards.size());

	for (size_t arenaIdx = 0; arenaIdx < arenaRewards.size(); arenaIdx++) {
		for (const WeightedReward& weightedReward : arenaRewards[arenaIdx]) {
			const string name = weightedReward.reward->GetName();
			auto [entry, inserted] = headIndexByName.try_emplace(name, layout.GetNumHeads());
			if (inserted)
				layout._headNames.push_back(name);

			layout._arenaRewardHeads[arenaIdx].push_back(entry->second);
		}
	}

	if (layout._headNames.empty())
		RG_ERR_CLOSE("RewardHeadLayout: no arena has any rewards, so there are no value heads to build");

	layout._presence = DimList2<uint8_t>(arenaRewards.size(), layout._headNames.size());
	for (size_t arenaIdx = 0; arenaIdx < arenaRewards.size(); arenaIdx++)
		for (int headIdx : layout._arenaRewardHeads[arenaIdx])
			layout._presence.At(arenaIdx, headIdx) = 1;

	return layout;
}

bool RLGC::RewardHeadLayout::Matches(const vector<vector<WeightedReward>>& arenaRewards) const {
	if (arenaRewards.size() != _arenaRewardHeads.size())
		return false;

	for (size_t arenaIdx = 0; arenaIdx < arenaRewards.size(); arenaIdx++) {
		if (arenaRewards[arenaIdx].size() != _arenaRewardHeads[arenaIdx].size())
			return false;
	}

	return true;
}

int RLGC::RewardHeadLayout::GetHeadIndex(int arenaIdx, int rewardIdx) const {
	return _arenaRewardHeads[arenaIdx][rewardIdx];
}

DimList2<float> RLGC::RewardHeadLayout::BuildWeightTable(const vector<vector<WeightedReward>>& arenaRewards) const {
	RG_ASSERT(arenaRewards.size() == _arenaRewardHeads.size());

	DimList2<float> weights(arenaRewards.size(), _headNames.size());
	for (size_t arenaIdx = 0; arenaIdx < arenaRewards.size(); arenaIdx++) {
		RG_ASSERT(arenaRewards[arenaIdx].size() == _arenaRewardHeads[arenaIdx].size());

		for (size_t rewardIdx = 0; rewardIdx < arenaRewards[arenaIdx].size(); rewardIdx++)
			weights.At(arenaIdx, _arenaRewardHeads[arenaIdx][rewardIdx]) = arenaRewards[arenaIdx][rewardIdx].weight;
	}

	return weights;
}
