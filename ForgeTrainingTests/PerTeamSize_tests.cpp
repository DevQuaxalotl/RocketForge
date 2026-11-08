#include <gtest/gtest.h>

#include "Rewards/PerTeamSize.h"

#include <stdexcept>

namespace {
	// Concrete helper so the protected setters are reachable from tests.
	struct IntPerTeamSize : RocketForge::PerTeamSize<int> {
		void MakeShared(int v) { SetShared(v); }
		void MakePerTeamSize(std::initializer_list<int> v) { SetPerTeamSize(v); }
	};
}

TEST(PerTeamSize, SharedResolvesToSharedForEverySize) {
	IntPerTeamSize p;
	p.MakeShared(7);

	ASSERT_FALSE(p.usePerTeamSize);
	ASSERT_EQ(p.Resolve(1), 7);
	ASSERT_EQ(p.Resolve(2), 7);
	ASSERT_EQ(p.Resolve(3), 7);
}

TEST(PerTeamSize, PerSizeResolvesToCorrectSlot) {
	IntPerTeamSize p;
	p.MakePerTeamSize({ 10, 20, 30 });

	ASSERT_TRUE(p.usePerTeamSize);
	ASSERT_EQ(p.Resolve(1), 10);
	ASSERT_EQ(p.Resolve(2), 20);
	ASSERT_EQ(p.Resolve(3), 30);
}

TEST(PerTeamSize, ResolveThrowsForOutOfRangeSize) {
	IntPerTeamSize p;
	p.MakePerTeamSize({ 1, 2, 3 });

	bool threwLow = false, threwHigh = false;
	try { p.Resolve(0); } catch (const std::exception&) { threwLow = true; }
	try { p.Resolve(4); } catch (const std::exception&) { threwHigh = true; }
	ASSERT_TRUE(threwLow);
	ASSERT_TRUE(threwHigh);
}

TEST(PerTeamSize, SetPerTeamSizeRequiresExactlyThree) {
	IntPerTeamSize tooFew;
	bool threwTooFew = false;
	try { tooFew.MakePerTeamSize({ 1, 2 }); } catch (const std::exception&) { threwTooFew = true; }
	ASSERT_TRUE(threwTooFew);

	IntPerTeamSize tooMany;
	bool threwTooMany = false;
	try { tooMany.MakePerTeamSize({ 1, 2, 3, 4 }); } catch (const std::exception&) { threwTooMany = true; }
	ASSERT_TRUE(threwTooMany);
}

TEST(PerTeamSize, SwitchingModesClearsStaleRepresentation) {
	IntPerTeamSize p;
	p.MakePerTeamSize({ 10, 20, 30 });
	p.MakeShared(5);
	// Switching to shared must zero the per-size slots.
	ASSERT_FALSE(p.usePerTeamSize);
	ASSERT_EQ(p.perTeamSize[0], 0);
	ASSERT_EQ(p.perTeamSize[1], 0);
	ASSERT_EQ(p.perTeamSize[2], 0);

	p.MakePerTeamSize({ 1, 2, 3 });
	// Switching back to per-size must clear the shared value.
	ASSERT_TRUE(p.usePerTeamSize);
	ASSERT_EQ(p.shared, 0);
}

TEST(PerTeamSize, AnyChecksSharedOrEverySlot) {
	IntPerTeamSize shared;
	shared.MakeShared(0);
	ASSERT_FALSE(shared.Any([](int v) { return v != 0; }));
	shared.MakeShared(5);
	ASSERT_TRUE(shared.Any([](int v) { return v != 0; }));

	IntPerTeamSize perSize;
	perSize.MakePerTeamSize({ 0, 0, 0 });
	ASSERT_FALSE(perSize.Any([](int v) { return v != 0; }));
	perSize.MakePerTeamSize({ 0, 0, 9 });
	ASSERT_TRUE(perSize.Any([](int v) { return v != 0; }));
}
