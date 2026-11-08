#include <gtest/gtest.h>

#include <RocketSim/Engine.h>

#include <stdexcept>
#include <string>

using namespace RocketSim;

TEST(EngineRegistry, V2IsAlwaysAvailable)
{
	const EngineInfo& v2 = GetEngineInfo(EngineId::V2);
	EXPECT_TRUE(v2.available);
	EXPECT_EQ(v2.displayName, "RocketSim v2");
	EXPECT_EQ(v2.revision, "vendored");
	EXPECT_NO_THROW(RequireEngine(EngineId::V2));
}

TEST(EngineRegistry, VersionsRoundTripAndUnknownThrows)
{
	EXPECT_EQ(EngineFromVersion(2), EngineId::V2);
	EXPECT_EQ(EngineFromVersion(3), EngineId::V3);
	EXPECT_THROW(EngineFromVersion(7), std::invalid_argument);
}

TEST(EngineRegistry, UnavailableEngineThrowsItsReason)
{
	Testing::ScopedEngineUnavailable off(EngineId::V3, "test reason");

	try {
		RequireEngine(EngineId::V3);
		FAIL();
	} catch (const EngineUnavailableError& error) {
		EXPECT_EQ(std::string(error.what()), "test reason");
		EXPECT_EQ(error.engine, EngineId::V3);
	}
}

TEST(EngineRegistry, ReasonIsEmptyExactlyWhenAvailable)
{
	for (const EngineInfo& info : Engines())
		EXPECT_EQ(info.available, info.unavailableReason.empty()) << info.displayName;
}
