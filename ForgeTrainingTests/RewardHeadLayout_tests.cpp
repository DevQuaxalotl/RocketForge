#include <gtest/gtest.h>

#include <RLGymCPP/EnvSet/RewardHeadLayout.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using namespace RLGC;

namespace {
	struct NamedReward : Reward {
		string name;

		explicit NamedReward(string name) : name(std::move(name)) {}

		float GetReward(const Player&, const GameState&, bool) override {
			return 0.f;
		}

		string GetName() override {
			return name;
		}
	};

	// Owns the rewards that the WeightedReward lists point at.
	struct RewardLists {
		vector<unique_ptr<NamedReward>> owned;
		vector<vector<WeightedReward>> arenas;

		void AddArena(const vector<pair<string, float>>& namedWeights) {
			vector<WeightedReward> rewards = {};
			for (const pair<string, float>& namedWeight : namedWeights) {
				owned.push_back(make_unique<NamedReward>(namedWeight.first));
				rewards.emplace_back(owned.back().get(), namedWeight.second);
			}
			arenas.push_back(std::move(rewards));
		}
	};

	bool BuildThrows(const vector<vector<WeightedReward>>& arenas) {
		try {
			RewardHeadLayout::Build(arenas);
		} catch (const runtime_error&) {
			return true;
		}
		return false;
	}
}

TEST(RewardHeadLayout, HeadsAreUniqueNamesInFirstAppearanceOrder) {
	RewardLists lists;
	lists.AddArena({ { "Touch", 1.f }, { "Goal", 10.f } });
	lists.AddArena({ { "Goal", 10.f }, { "Pass", 2.f }, { "Touch", 1.f } });

	RewardHeadLayout layout = RewardHeadLayout::Build(lists.arenas);

	ASSERT_EQ(layout.GetNumHeads(), 3);
	ASSERT_EQ(layout.GetHeadNames()[0], string("Touch"));
	ASSERT_EQ(layout.GetHeadNames()[1], string("Goal"));
	ASSERT_EQ(layout.GetHeadNames()[2], string("Pass"));
}

TEST(RewardHeadLayout, SlotsMapToTheirNamedHead) {
	RewardLists lists;
	lists.AddArena({ { "Touch", 1.f }, { "Goal", 10.f } });
	lists.AddArena({ { "Goal", 10.f }, { "Pass", 2.f }, { "Touch", 1.f } });

	RewardHeadLayout layout = RewardHeadLayout::Build(lists.arenas);

	ASSERT_EQ(layout.GetHeadIndex(0, 0), 0);
	ASSERT_EQ(layout.GetHeadIndex(0, 1), 1);
	ASSERT_EQ(layout.GetHeadIndex(1, 0), 1);
	ASSERT_EQ(layout.GetHeadIndex(1, 1), 2);
	ASSERT_EQ(layout.GetHeadIndex(1, 2), 0);
}

TEST(RewardHeadLayout, PresenceMarksOnlyListedRewards) {
	RewardLists lists;
	lists.AddArena({ { "Touch", 1.f } });
	lists.AddArena({ { "Touch", 1.f }, { "Pass", 2.f } });

	RewardHeadLayout layout = RewardHeadLayout::Build(lists.arenas);
	const DimList2<uint8_t>& presence = layout.GetPresence();

	ASSERT_EQ(presence.size[0], (size_t)2);
	ASSERT_EQ(presence.size[1], (size_t)2);
	ASSERT_EQ((int)presence.At(0, 0), 1);
	ASSERT_EQ((int)presence.At(0, 1), 0);
	ASSERT_EQ((int)presence.At(1, 0), 1);
	ASSERT_EQ((int)presence.At(1, 1), 1);
}

TEST(RewardHeadLayout, WeightTableUsesEachArenasOwnWeight) {
	RewardLists lists;
	lists.AddArena({ { "Touch", 1.5f } });
	lists.AddArena({ { "Pass", 2.f }, { "Touch", 4.f } });

	RewardHeadLayout layout = RewardHeadLayout::Build(lists.arenas);
	DimList2<float> weights = layout.BuildWeightTable(lists.arenas);

	ASSERT_EQ(weights.At(0, 0), 1.5f);
	ASSERT_EQ(weights.At(0, 1), 0.f);
	ASSERT_EQ(weights.At(1, 0), 4.f);
	ASSERT_EQ(weights.At(1, 1), 2.f);
}

TEST(RewardHeadLayout, WeightTableReadsCurrentWeights) {
	RewardLists lists;
	lists.AddArena({ { "Touch", 1.f }, { "Goal", 10.f } });

	RewardHeadLayout layout = RewardHeadLayout::Build(lists.arenas);
	lists.arenas[0][1].weight = 25.f;
	DimList2<float> weights = layout.BuildWeightTable(lists.arenas);

	ASSERT_EQ(weights.At(0, 1), 25.f);
}

TEST(RewardHeadLayout, MatchesOnlyTheShapeItWasBuiltFrom) {
	RewardLists lists;
	lists.AddArena({ { "Touch", 1.f }, { "Goal", 10.f } });
	lists.AddArena({ { "Touch", 1.f } });

	RewardHeadLayout layout = RewardHeadLayout::Build(lists.arenas);
	ASSERT_TRUE(layout.Matches(lists.arenas));

	RewardLists extraSlot = {};
	extraSlot.AddArena({ { "Touch", 1.f }, { "Goal", 10.f } });
	extraSlot.AddArena({ { "Touch", 1.f }, { "Goal", 10.f } });
	ASSERT_FALSE(layout.Matches(extraSlot.arenas));

	RewardLists extraArena = {};
	extraArena.AddArena({ { "Touch", 1.f }, { "Goal", 10.f } });
	extraArena.AddArena({ { "Touch", 1.f } });
	extraArena.AddArena({ { "Touch", 1.f } });
	ASSERT_FALSE(layout.Matches(extraArena.arenas));
}

TEST(RewardHeadLayout, SameNameAcrossArenasIsNotADuplicate) {
	RewardLists lists;
	lists.AddArena({ { "Touch", 1.f } });
	lists.AddArena({ { "Touch", 3.f } });

	ASSERT_TRUE(RewardHeadLayout::FindDuplicateNames(lists.arenas).empty());
	ASSERT_FALSE(BuildThrows(lists.arenas));
}

TEST(RewardHeadLayout, DuplicateWithinAnyArenaIsReportedOnce) {
	RewardLists lists;
	lists.AddArena({ { "Touch", 1.f } });
	lists.AddArena({ { "Velocity", 1.f }, { "Velocity", -1.f } });
	lists.AddArena({ { "Velocity", 1.f }, { "Velocity", -1.f } });

	vector<string> duplicates = RewardHeadLayout::FindDuplicateNames(lists.arenas);

	ASSERT_EQ(duplicates.size(), (size_t)1);
	ASSERT_EQ(duplicates[0], string("Velocity"));
	ASSERT_TRUE(BuildThrows(lists.arenas));
}

TEST(RewardHeadLayout, BuildWithNoRewardsIsFatal) {
	RewardLists lists;
	lists.AddArena({});
	lists.AddArena({});

	ASSERT_TRUE(BuildThrows(lists.arenas));
}
