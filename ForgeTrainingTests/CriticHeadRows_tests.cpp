#include <gtest/gtest.h>
#include "RewardVisualizer/CriticHeadRows.h"

#include <cmath>
#include <string>
#include <vector>

using namespace ForgeGUI;

namespace {
	constexpr bool SELECTED_HEADS_ONLY = false;

	bool Near(float a, float b) { return std::abs(a - b) < 1e-5f; }

	ForgeDebug::CriticHeadDebugData MakeHead(const std::string& name, std::optional<float> weight, float contribution) {
		ForgeDebug::CriticHeadDebugData head;
		head.name = name;
		head.weight = weight;
		head.contribution = contribution;
		return head;
	}

	ForgeDebug::CriticDebugData MakeCritic() {
		ForgeDebug::CriticDebugData critic;
		critic.isDecomposed = true;
		critic.heads = { MakeHead("a", 1.f, 2.f), MakeHead("b", 1.f, -4.f), MakeHead("gone", std::nullopt, 0.f) };
		return critic;
	}
}

TEST(CriticHeadRows, NoSavedSelectionEnablesEveryHead) {
	ValueTrendTracker trends;
	float peak = 0.f;
	ForgeDebug::CriticDebugData critic = MakeCritic();
	critic.heads[1].episodeAbsContributionSum = 7.f;
	std::vector<RewardDisplayRow> rows = BuildCriticHeadRows(critic, {}, false, SELECTED_HEADS_ONLY, trends, peak);

	ASSERT_EQ(rows.size(), size_t(3));
	ASSERT_TRUE(rows[0].enabled && rows[1].enabled && rows[2].enabled);
	ASSERT_TRUE(Near(rows[0].displayNorm, 0.5f));
	ASSERT_TRUE(Near(rows[1].displayNorm, -1.f));
	ASSERT_TRUE(Near(rows[1].absSum, 7.f));
	ASSERT_EQ(rows[1].index, 1);
}

TEST(CriticHeadRows, UnselectedHeadsAreExcludedFromScaling) {
	ValueTrendTracker trends;
	float peak = 0.f;
	std::vector<RewardDisplayRow> rows = BuildCriticHeadRows(MakeCritic(), { "a", "gone" }, true, SELECTED_HEADS_ONLY, trends, peak);

	ASSERT_TRUE(rows[0].enabled);
	ASSERT_FALSE(rows[1].enabled);
	ASSERT_FALSE(rows[1].desiredEnabled);
	ASSERT_TRUE(Near(rows[0].displayNorm, 1.f));
}

TEST(CriticHeadRows, ShowAllHeadsEnablesUnselectedHeadsButKeepsCheckbox) {
	ValueTrendTracker trends;
	float peak = 0.f;
	const bool showAllHeads = true;
	std::vector<RewardDisplayRow> rows = BuildCriticHeadRows(MakeCritic(), { "a" }, true, showAllHeads, trends, peak);

	ASSERT_TRUE(rows[1].enabled);
	ASSERT_FALSE(rows[1].desiredEnabled);
	ASSERT_TRUE(Near(rows[1].displayNorm, -1.f));
	ASSERT_TRUE(Near(rows[0].displayNorm, 0.5f));
}

TEST(CriticHeadRows, LoneHeadNormIsRelativeToItsPeakAcrossFrames) {
	ValueTrendTracker trends;
	float peak = 0.f;
	ForgeDebug::CriticDebugData critic;
	critic.isDecomposed = true;
	critic.heads = { MakeHead("punish", 1.f, -4.f) };
	BuildCriticHeadRows(critic, {}, false, SELECTED_HEADS_ONLY, trends, peak);

	critic.heads[0].contribution = -1.f;
	std::vector<RewardDisplayRow> rows = BuildCriticHeadRows(critic, {}, false, SELECTED_HEADS_ONLY, trends, peak);

	ASSERT_TRUE(Near(peak, 4.f));
	ASSERT_TRUE(Near(rows[0].displayNorm, -0.25f));
}

TEST(CriticHeadRows, HeadWithoutRewardHasNoValue) {
	ValueTrendTracker trends;
	float peak = 0.f;
	std::vector<RewardDisplayRow> rows = BuildCriticHeadRows(MakeCritic(), {}, false, SELECTED_HEADS_ONLY, trends, peak);

	ASSERT_FALSE(rows[2].hasValue);
	ASSERT_TRUE(Near(rows[2].displayNorm, 0.f));
}

TEST(CriticHeadRows, TrendIntensityIsRelativeToLargestChange) {
	ValueTrendTracker trends;
	float peak = 0.f;
	for (int i = 0; i <= 16; i++) {
		const double time = i / 16.0;
		trends.Record("a", time, (float)time);
		trends.Record("b", time, (float)time * 0.5f);
	}

	std::vector<RewardDisplayRow> rows = BuildCriticHeadRows(MakeCritic(), {}, false, SELECTED_HEADS_ONLY, trends, peak);

	ASSERT_EQ(rows[0].trendDirection, 1);
	ASSERT_TRUE(Near(rows[0].trendIntensity, 1.f));
	ASSERT_TRUE(Near(rows[1].trendIntensity, 0.5f));
}

TEST(CriticHeadRows, ToggleWithoutSavedSelectionSeedsAllHeads) {
	const std::vector<std::string> all = { "a", "b", "c" };
	const std::vector<std::string> selection = ToggleCriticHeadSelection({}, false, all, "b", false);
	ASSERT_EQ(selection, (std::vector<std::string>{ "a", "c" }));
}

TEST(CriticHeadRows, ToggleAddsAndRemovesFromSavedSelection) {
	const std::vector<std::string> all = { "a", "b", "c" };
	std::vector<std::string> selection = ToggleCriticHeadSelection({ "a" }, true, all, "c", true);
	ASSERT_EQ(selection, (std::vector<std::string>{ "a", "c" }));

	selection = ToggleCriticHeadSelection(selection, true, all, "a", false);
	ASSERT_EQ(selection, (std::vector<std::string>{ "c" }));

	selection = ToggleCriticHeadSelection(selection, true, all, "c", true);
	ASSERT_EQ(selection, (std::vector<std::string>{ "c" }));
}
