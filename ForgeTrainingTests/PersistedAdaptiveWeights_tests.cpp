#include <gtest/gtest.h>

#include "TestSupport.h"

#include <Inference/PersistedAdaptiveWeights.h>
#include <RLGymCPP/Rewards/Reward.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace RocketForge;

namespace fs = std::filesystem;

namespace {
	struct NamedReward : RLGC::Reward {
		std::string name;
		explicit NamedReward(std::string name) : name(std::move(name)) {}
		std::string GetName() override { return name; }
	};

	struct TempCheckpoint {
		fs::path folder;

		explicit TempCheckpoint(const std::string& name)
			: folder(RocketForgeTests::TestTempDir(name)) {
			fs::remove_all(folder);
			fs::create_directories(folder);
		}
		~TempCheckpoint() { fs::remove_all(folder); }

		void WriteStats(const std::string& json) const {
			std::ofstream out(folder / "RUNNING_STATS.json");
			out << json;
		}
	};
}

TEST(PersistedAdaptiveWeights, LoadReadsRunningStatsEntries) {
	TempCheckpoint temp("load");
	temp.WriteStats(R"({
		"total_timesteps": 5,
		"adaptive_reward_weights": [
			{ "name": "Flip", "player_count": 2, "weight": 12.5, "raw_rate_ema": 0.25 },
			{ "name": "Boost", "player_count": 4, "weight": -3.0, "raw_rate_ema": 0.5 }
		]
	})");

	std::vector<RLGC::AdaptivePersistEntry> entries = LoadPersistedAdaptiveWeights(temp.folder);

	ASSERT_EQ(entries.size(), (size_t)2);
	ASSERT_EQ(entries[0].name, std::string("Flip"));
	ASSERT_EQ(entries[0].playerCount, 2);
	ASSERT_EQ(entries[0].weight, 12.5f);
	ASSERT_EQ(entries[0].rawRateEma, 0.25f);
	ASSERT_EQ(entries[1].name, std::string("Boost"));
	ASSERT_EQ(entries[1].weight, -3.f);
}

TEST(PersistedAdaptiveWeights, LoadIsEmptyWhenAbsent) {
	TempCheckpoint missingFile("missing");
	ASSERT_TRUE(LoadPersistedAdaptiveWeights(missingFile.folder).empty());

	TempCheckpoint noSection("nosection");
	noSection.WriteStats(R"({ "total_timesteps": 5 })");
	ASSERT_TRUE(LoadPersistedAdaptiveWeights(noSection.folder).empty());

	TempCheckpoint malformed("malformed");
	malformed.WriteStats("{ not json");
	ASSERT_TRUE(LoadPersistedAdaptiveWeights(malformed.folder).empty());
}

TEST(PersistedAdaptiveWeights, ApplyOverridesOnlyAdaptiveRewardsWithEntries) {
	NamedReward flip("Flip");
	NamedReward boost("Boost");
	NamedReward touch("Touch");
	RLGC::AdaptiveWeightSettings adaptive = RLGC::AdaptiveWeightSettings::TargetReward(1.f, 0.1f, 100.f);

	std::vector<RLGC::WeightedReward> rewards = {
		{ &flip, 20.f, adaptive },   // adaptive, has a 2v2 entry
		{ &boost, 5.f, adaptive },   // adaptive, no entry for any matchup
		{ &touch, 7.f }              // static: an entry under its name must not touch it
	};
	std::vector<RLGC::AdaptivePersistEntry> entries = {
		{ "Flip", 4, 42.f, 0.f },
		{ "Touch", 4, 99.f, 0.f }
	};

	ApplyPersistedAdaptiveWeights(entries, 2, 2, rewards);

	ASSERT_EQ(rewards[0].weight, 42.f);
	ASSERT_EQ(rewards[1].weight, 5.f);
	ASSERT_EQ(rewards[2].weight, 7.f);
}

TEST(PersistedAdaptiveWeights, ApplyResolvesUnevenMatchupByBorrowing) {
	NamedReward flip("Flip");
	RLGC::AdaptiveWeightSettings adaptive = RLGC::AdaptiveWeightSettings::TargetReward(1.f, 0.1f, 100.f);

	std::vector<RLGC::WeightedReward> rewards = { { &flip, 20.f, adaptive } };
	std::vector<RLGC::AdaptivePersistEntry> entries = {
		{ "Flip", 2, 11.f, 0.f },
		{ "Flip", 4, 22.f, 0.f }
	};

	// 1v2 borrows the 2v2 (total 4) key, as the live controller does.
	ApplyPersistedAdaptiveWeights(entries, 1, 2, rewards);

	ASSERT_EQ(rewards[0].weight, 22.f);
}
