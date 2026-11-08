#include <gtest/gtest.h>

#include <RewardDebugCollector.h>
#include <RLGymCPP/Rewards/Reward.h>

#include <memory>
#include <vector>

namespace {
	// Minimal reward: fixed name + category, never computes a value.
	class FakeReward : public RLGC::Reward {
	public:
		FakeReward(std::string name, RLGC::RewardCategory cat, bool debugDefault)
			: _name(std::move(name)), _cat(cat) {
			SetDebugEnabled(debugDefault);
		}

		int nameCalls = 0;
		mutable int categoryCalls = 0;
		std::string GetName() override { ++nameCalls; return _name; }
		RLGC::RewardCategory GetCategory() const override { ++categoryCalls; return _cat; }
		float GetReward(const RLGC::Player&, const RLGC::GameState&, bool) override { return 0.f; }
	private:
		std::string _name;
		RLGC::RewardCategory _cat;
	};

	struct Fixture {
		std::vector<std::unique_ptr<FakeReward>> owned;
		std::vector<RLGC::WeightedReward> weighted;

		// a=CONTINUOUS off, b=CONTINUOUS on (code default), c=EVENT off
		Fixture() {
			AddReward("a", RLGC::RewardCategory::CONTINUOUS, false);
			AddReward("b", RLGC::RewardCategory::CONTINUOUS, true);
			AddReward("c", RLGC::RewardCategory::EVENT,      false);
		}
		void AddReward(const std::string& n, RLGC::RewardCategory cat, bool dbg) {
			owned.push_back(std::make_unique<FakeReward>(n, cat, dbg));
			weighted.emplace_back(owned.back().get(), 1.0f);
		}
		bool Effective(int i) const { return owned[i]->IsDebugEnabled(); }
	};
}

TEST(Collector, InitializeSeedsDesiredFromCodeDefault) {
	Fixture f;
	ForgeDebug::RewardDebugCollector c;
	c.Initialize(f.weighted);
	ASSERT_FALSE(f.Effective(0));   // a default off
	ASSERT_TRUE(f.Effective(1));    // b default on
	ASSERT_FALSE(f.Effective(2));   // c default off

	auto roster = c.BuildRoster();
	ASSERT_EQ(roster.size(), size_t(3));
	ASSERT_EQ(roster[1].name, std::string("b"));
	ASSERT_EQ(roster[1].index, 1);
	ASSERT_TRUE(roster[1].desiredEnabled);
	ASSERT_FALSE(roster[0].desiredEnabled);
}

TEST(Collector, SetRewardDebugEnabledFlipsOnlyTarget) {
	Fixture f;
	ForgeDebug::RewardDebugCollector c;
	c.Initialize(f.weighted);

	c.SetRewardDebugEnabled(0, true);
	ASSERT_TRUE(f.Effective(0));
	ASSERT_TRUE(f.Effective(1));    // unchanged
	ASSERT_FALSE(f.Effective(2));   // unchanged

	c.SetRewardDebugEnabled(1, false);
	ASSERT_TRUE(f.Effective(0));
	ASSERT_FALSE(f.Effective(1));
}

TEST(Collector, ForceDebugAllIsNonDestructive) {
	Fixture f;
	ForgeDebug::RewardDebugCollector c;
	c.Initialize(f.weighted);
	c.SetRewardDebugEnabled(0, true);   // desired: a=on, b=on, c=off

	c.SetForceDebugAll(true);
	ASSERT_TRUE(f.Effective(0));
	ASSERT_TRUE(f.Effective(1));
	ASSERT_TRUE(f.Effective(2));        // forced on despite desired off

	c.SetForceDebugAll(false);
	ASSERT_TRUE(f.Effective(0));        // restored from desired
	ASSERT_TRUE(f.Effective(1));
	ASSERT_FALSE(f.Effective(2));       // restored to desired off, NOT all-off
}

TEST(Collector, SyncWeightsUpdatesWeightedValues) {
	Fixture f;
	ForgeDebug::RewardDebugCollector c;
	c.Initialize(f.weighted);
	c.SetRewardDebugEnabled(0, true);   // debug "a" so it shows up in the snapshot

	RLGC::Player player;
	RLGC::GameState state;
	std::vector<float> earned = { 2.f, 0.f, 0.f };
	std::vector<float> training = earned;

	c.RecordStep(earned, training, 0.f, player, state);

	const ForgeDebug::RewardDebugData* before = nullptr;
	for (const auto& rd : c.GetSnapshot().rewards)
		if (rd.name == "a") before = &rd;
	ASSERT_TRUE(before != nullptr);
	ASSERT_EQ(before->weight, 1.0f);
	ASSERT_EQ(before->weightedValue, 2.0f);   // rawValue(2) * weight(1)

	// Simulate the adaptive controller changing "a"'s live weight.
	std::vector<RLGC::WeightedReward> updated = f.weighted;
	updated[0].weight = 5.0f;
	c.SyncWeights(updated);

	c.RecordStep(earned, training, 0.f, player, state);

	const ForgeDebug::RewardDebugData* after = nullptr;
	for (const auto& rd : c.GetSnapshot().rewards)
		if (rd.name == "a") after = &rd;
	ASSERT_TRUE(after != nullptr);
	ASSERT_EQ(after->weight, 5.0f);
	ASSERT_EQ(after->weightedValue, 10.0f);   // rawValue(2) * weight(5)

	// Roster/selection state is untouched by SyncWeights.
	auto roster = c.BuildRoster();
	ASSERT_EQ(roster.size(), size_t(3));
	ASSERT_TRUE(roster[0].desiredEnabled);   // explicitly enabled above
	ASSERT_TRUE(roster[1].desiredEnabled);   // "b" defaults on (code default), unchanged
	ASSERT_FALSE(roster[2].desiredEnabled);  // "c" defaults off, unchanged
}

TEST(Collector, SyncWeightsHandlesSizeMismatch) {
	Fixture f;
	ForgeDebug::RewardDebugCollector c;
	c.Initialize(f.weighted);
	c.SetRewardDebugEnabled(0, true);   // debug "a" so it shows up in the snapshot

	RLGC::Player player;
	RLGC::GameState state;
	std::vector<float> earned = { 1.f, 0.f, 0.f };
	std::vector<float> training = earned;

	// (a) Shorter src (1 element): only index 0 is updated, no out-of-range access.
	std::vector<RLGC::WeightedReward> shorter = { f.weighted[0] };
	shorter[0].weight = 3.0f;
	c.SyncWeights(shorter);

	c.RecordStep(earned, training, 0.f, player, state);
	const ForgeDebug::RewardDebugData* afterShort = nullptr;
	for (const auto& rd : c.GetSnapshot().rewards)
		if (rd.name == "a") afterShort = &rd;
	ASSERT_TRUE(afterShort != nullptr);
	ASSERT_EQ(afterShort->weight, 3.0f);   // index 0 updated from the shorter list

	// (b) Longer src (4 elements): overlapping indices update, the extra is ignored.
	std::vector<RLGC::WeightedReward> longer = f.weighted;
	longer[0].weight = 7.0f;
	longer.emplace_back(f.weighted[0].reward, 99.0f);   // extra element beyond rewards.size()
	c.SyncWeights(longer);

	c.RecordStep(earned, training, 0.f, player, state);
	const ForgeDebug::RewardDebugData* afterLong = nullptr;
	for (const auto& rd : c.GetSnapshot().rewards)
		if (rd.name == "a") afterLong = &rd;
	ASSERT_TRUE(afterLong != nullptr);
	ASSERT_EQ(afterLong->weight, 7.0f);   // overlapping index updated; no crash from the extra
}

TEST(Collector, SetDebugSelectionMatchesByName) {
	Fixture f;
	ForgeDebug::RewardDebugCollector c;
	c.Initialize(f.weighted);

	c.SetDebugSelection({ "a", "c" });
	ASSERT_TRUE(f.Effective(0));
	ASSERT_FALSE(f.Effective(1));   // b not in selection -> off
	ASSERT_TRUE(f.Effective(2));

	auto roster = c.BuildRoster();
	ASSERT_TRUE(roster[0].desiredEnabled);
	ASSERT_FALSE(roster[1].desiredEnabled);
	ASSERT_TRUE(roster[2].desiredEnabled);
}

TEST(Collector, SnapshotCarriesCriticEstimate) {
	Fixture f;
	ForgeDebug::RewardDebugCollector c;
	c.Initialize(f.weighted);

	RLGC::Player player;
	RLGC::GameState state;
	std::vector<float> values = { 0.f, 0.f, 0.f };

	c.RecordStep(values, values, 0.f, player, state);
	ASSERT_FALSE(c.GetSnapshot().critic.has_value());

	ForgeDebug::CriticDebugData estimate;
	estimate.total = 1.5f;
	c.SetCriticEstimate(estimate);
	c.RecordStep(values, values, 0.f, player, state);
	ASSERT_TRUE(c.GetSnapshot().critic.has_value());
	ASSERT_EQ(c.GetSnapshot().critic->total, 1.5f);

	c.SetCriticEstimate(std::nullopt);
	c.RecordStep(values, values, 0.f, player, state);
	ASSERT_FALSE(c.GetSnapshot().critic.has_value());
}

TEST(Collector, GameTimeAccumulatesDeltaTime) {
	Fixture f;
	ForgeDebug::RewardDebugCollector c;
	c.Initialize(f.weighted);

	RLGC::Player player;
	RLGC::GameState state;
	std::vector<float> values = { 0.f, 0.f, 0.f };

	c.RecordStep(values, values, 0.25f, player, state);
	c.RecordStep(values, values, 0.25f, player, state);
	ASSERT_EQ(c.GetSnapshot().gameTimeSeconds, 0.5);
}

TEST(Collector, BeginEpisodeIncrementsEpisodeIndex) {
	Fixture f;
	ForgeDebug::RewardDebugCollector c;
	c.Initialize(f.weighted);

	RLGC::Player player;
	RLGC::GameState state;
	std::vector<float> values = { 0.f, 0.f, 0.f };

	c.RecordStep(values, values, 0.f, player, state);
	const uint64_t first = c.GetSnapshot().episodeIndex;

	c.BeginEpisode();
	c.RecordStep(values, values, 0.f, player, state);
	ASSERT_EQ(c.GetSnapshot().episodeIndex, first + 1);
}

TEST(Collector, HeadAbsSumsAccumulateUntilEpisodeTotalsReset) {
	Fixture f;
	ForgeDebug::RewardDebugCollector c;
	c.Initialize(f.weighted);

	RLGC::Player player;
	RLGC::GameState state;
	std::vector<float> values = { 0.f, 0.f, 0.f };

	ForgeDebug::CriticHeadDebugData head;
	head.name = "touch";
	head.contribution = -2.f;
	ForgeDebug::CriticDebugData estimate;
	estimate.isDecomposed = true;
	estimate.heads = { head };
	c.SetCriticEstimate(estimate);

	c.RecordStep(values, values, 0.f, player, state);
	c.RecordStep(values, values, 0.f, player, state);
	ASSERT_EQ(c.GetSnapshot().critic->heads[0].episodeAbsContributionSum, 4.f);

	c.ResetEpisodeTotals();
	c.RecordStep(values, values, 0.f, player, state);
	ASSERT_EQ(c.GetSnapshot().critic->heads[0].episodeAbsContributionSum, 2.f);
}

TEST(Collector, ReusesMetadataWhileSelectionAndWeightsStayLive) {
	Fixture f;
	ForgeDebug::RewardDebugCollector c;
	c.Initialize(f.weighted);
	const int nameCalls = f.owned[0]->nameCalls;
	const int categoryCalls = f.owned[0]->categoryCalls;

	c.SetDebugSelection({ "a" });
	c.SetForceDebugAll(true);
	c.SetForceDebugAll(false);
	f.weighted[0].weight = 4.f;
	c.SyncWeights(f.weighted);

	RLGC::Player player;
	RLGC::GameState state;
	for (int i = 0; i < 3; ++i) {
		c.RecordStep({ 2.f, 0.f, 0.f }, { 2.f, 0.f, 0.f }, .1f, player, state);
		std::vector<ForgeDebug::RewardRosterEntry> roster = c.BuildRoster();
		ASSERT_EQ(roster.size(), 3u);
		EXPECT_TRUE(roster[0].desiredEnabled);
		EXPECT_FALSE(roster[1].desiredEnabled);
		ASSERT_EQ(c.GetSnapshot().rewards.size(), 1u);
		EXPECT_EQ(c.GetSnapshot().rewards[0].name, "a");
		EXPECT_FLOAT_EQ(c.GetSnapshot().rewards[0].weightedValue, 8.f);

		// Callers receive their own roster; edits cannot corrupt the cached one.
		roster[0].name = "changed";
	}

	EXPECT_EQ(c.BuildRoster()[0].name, "a");
	EXPECT_EQ(f.owned[0]->nameCalls, nameCalls);
	EXPECT_EQ(f.owned[0]->categoryCalls, categoryCalls);
}

TEST(Collector, ReinitializeRefreshesMetadataAndDropsOldRewards) {
	Fixture f;
	ForgeDebug::RewardDebugCollector c;
	c.Initialize(f.weighted);
	FakeReward replacement("replacement", RLGC::RewardCategory::EVENT, true);
	c.Initialize({ RLGC::WeightedReward(&replacement, 3.f) });

	RLGC::Player player;
	RLGC::GameState state;
	c.RecordStep({ 1.f }, { 1.f }, .1f, player, state);

	const ForgeDebug::RewardDebugSnapshot& snapshot = c.GetSnapshot();
	ASSERT_EQ(snapshot.roster.size(), 1u);
	EXPECT_EQ(snapshot.roster[0].name, "replacement");
	EXPECT_EQ(snapshot.roster[0].category, RLGC::RewardCategory::EVENT);
	EXPECT_EQ(snapshot.roster[0].index, 0);
	EXPECT_TRUE(snapshot.roster[0].desiredEnabled);
	ASSERT_EQ(snapshot.rewards.size(), 1u);
	EXPECT_EQ(snapshot.rewards[0].name, "replacement");
	EXPECT_FLOAT_EQ(snapshot.totalEventWeighted, 3.f);
}

TEST(Collector, RebindKeepsEpisodeIndexAndChoicesForRewardsTheNextListLacks) {
	Fixture f;
	ForgeDebug::RewardDebugCollector c;
	c.Initialize(f.weighted);
	c.SetRewardDebugEnabled(std::string("a"), true);
	c.SetRewardDebugEnabled(std::string("b"), false);
	c.BeginEpisode();

	// A list without "a" (as when a team size filters it out), then the full list again.
	c.Rebind({ f.weighted[1], f.weighted[2] });
	ASSERT_EQ(c.BuildRoster().size(), 2u);
	EXPECT_FALSE(c.BuildRoster()[0].desiredEnabled);   // "b" keeps its explicit off over its code default
	c.Rebind(f.weighted);

	RLGC::Player player;
	RLGC::GameState state;
	c.RecordStep({ 0.f, 0.f, 0.f }, { 0.f, 0.f, 0.f }, 0.f, player, state);
	EXPECT_EQ(c.GetSnapshot().episodeIndex, 1u);

	const std::vector<ForgeDebug::RewardRosterEntry> roster = c.BuildRoster();
	EXPECT_TRUE(roster[0].desiredEnabled);
	EXPECT_FALSE(roster[1].desiredEnabled);
	EXPECT_FALSE(roster[2].desiredEnabled);
	EXPECT_TRUE(f.Effective(0));
}

TEST(Collector, ChoicesForAbsentRewardsApplyOnceTheyAppear) {
	Fixture f;
	ForgeDebug::RewardDebugCollector c;
	c.Initialize({ f.weighted[1] });

	c.SetRewardDebugEnabled(std::string("c"), true);
	c.SetDebugSelection({ "a" });   // replaces the "c" choice made above
	c.Rebind(f.weighted);

	const std::vector<ForgeDebug::RewardRosterEntry> roster = c.BuildRoster();
	EXPECT_TRUE(roster[0].desiredEnabled);
	EXPECT_FALSE(roster[1].desiredEnabled);
	EXPECT_FALSE(roster[2].desiredEnabled);
}
