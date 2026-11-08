#pragma once

#include "../BasicTypes/Lists.h"
#include "../Rewards/Reward.h"

#include <cstdint>
#include <string>
#include <vector>

namespace RLGC {

	/**
	 * Maps every arena's reward list onto one shared list of value heads, keyed by reward name.
	 * Heads are ordered by first appearance, scanning arenas in order and each arena's rewards in order.
	 * Arena reward lists can differ (e.g. per team size), so a head may be absent from some arenas.
	 */
	class RewardHeadLayout {
	public:
		/** Builds the layout. Fatal if an arena lists two rewards with the same name, or if there are no rewards at all. */
		static RewardHeadLayout Build(const std::vector<std::vector<WeightedReward>>& arenaRewards);

		/** Names listed more than once within a single arena's rewards, each reported once. */
		static std::vector<std::string> FindDuplicateNames(const std::vector<std::vector<WeightedReward>>& arenaRewards);

		const std::vector<std::string>& GetHeadNames() const { return _headNames; }
		int GetNumHeads() const { return (int)_headNames.size(); }

		/** True if the lists have the same arena count and per-arena reward slot counts the layout was built from. */
		bool Matches(const std::vector<std::vector<WeightedReward>>& arenaRewards) const;

		/** Head that reward slot rewardIdx of arena arenaIdx feeds. */
		int GetHeadIndex(int arenaIdx, int rewardIdx) const;

		/** [arenas × heads]: 1 where the arena's reward list contains the head's reward, else 0. */
		const DimList2<uint8_t>& GetPresence() const { return _presence; }

		/**
		 * [arenas × heads] of the rewards' current weights, 0 where a head is absent.
		 * The lists must have the shape the layout was built from; weights may have changed since.
		 */
		DimList2<float> BuildWeightTable(const std::vector<std::vector<WeightedReward>>& arenaRewards) const;

	private:
		std::vector<std::string> _headNames = {};

		/** [arena][reward slot] → head index. */
		std::vector<std::vector<int>> _arenaRewardHeads = {};

		DimList2<uint8_t> _presence = {};
	};
}
