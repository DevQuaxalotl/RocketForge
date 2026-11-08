#include <gtest/gtest.h>
#include "RewardVisualizer/RewardDisplayRow.h"

#include <vector>

using namespace ForgeGUI;

namespace {
	RewardDisplayRow MakeRow(int index, const std::string& name, bool enabled, float weighted, float absSum) {
		RewardDisplayRow row;
		row.index = index;
		row.name = name;
		row.enabled = enabled;
		row.displayWeighted = weighted;
		row.displayNorm = weighted;
		row.absSum = absSum;
		return row;
	}
}

TEST(RewardDisplayRow, SortOrder) {
	struct Case {
		const char* label;
		std::vector<RewardDisplayRow> rows;
		RewardSortKey key;
		bool descending;
		int expectedFirstIndex;
		const char* expectedFirstName;
	};

	const Case cases[] = {
		{ "enabled rows first", { MakeRow(0, "a", false, 9.f, 9.f), MakeRow(1, "b", true, 1.f, 1.f) }, RewardSortKey::Weighted, true, 1, "b" },
		{ "abs sum descending", { MakeRow(0, "a", true, -1.f, 1.f), MakeRow(1, "b", true, 0.5f, 3.f) }, RewardSortKey::AbsSum, true, 1, "b" },
		{ "name ascending", { MakeRow(0, "zeta", true, 0.f, 0.f), MakeRow(1, "alpha", true, 0.f, 0.f) }, RewardSortKey::Name, false, 1, "alpha" },
		{ "ties fall back to index", { MakeRow(5, "a", true, 1.f, 1.f), MakeRow(2, "b", true, 1.f, 1.f) }, RewardSortKey::Weighted, true, 2, "b" },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);
		std::vector<RewardDisplayRow> rows = testCase.rows;
		SortRewardDisplayRows(rows, testCase.key, testCase.descending);

		ASSERT_EQ(rows[0].index, testCase.expectedFirstIndex);
		ASSERT_EQ(rows[0].name, std::string(testCase.expectedFirstName));
	}
}
