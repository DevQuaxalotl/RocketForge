#include <gtest/gtest.h>
#include "TestSupport.h"

#include <SimulationAnalysis/SnapshotReplay.h>

#include <RocketSim/Arena.h>

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

using namespace RocketForge::SimulationAnalysis;

namespace
{
	/** A snapshot of a one-car v2 arena with its engine key removed, as snapshots saved before the key existed read. **/
	std::string SnapshotWithoutEngine()
	{
		RocketSim::ArenaConfig config = {};
		config.engine = RocketSim::EngineId::V2;
		std::unique_ptr<RocketSim::Arena> arena(RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR, config));
		arena->AddCar(RocketSim::Team::BLUE);
		std::string snapshot = SerializeGameStateSnapshot(RLGC::GameState(arena.get()));

		const std::string key = ";engine=2";
		snapshot.erase(snapshot.find(key), key.size());
		return snapshot;
	}
}

TEST(SnapshotReplay, SnapshotsRecordTheirArenasEngine)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	RocketSim::ArenaConfig config = {};
	config.engine = RocketSim::EngineId::V2;
	std::unique_ptr<RocketSim::Arena> arena(RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR, config));
	arena->AddCar(RocketSim::Team::BLUE);

	const std::string snapshot = SerializeGameStateSnapshot(RLGC::GameState(arena.get()));
	EXPECT_NE(snapshot.find(";engine=2;"), std::string::npos) << snapshot;

	RocketSim::EngineId engine = RocketSim::EngineId::V3;
	ParseGameStateSnapshot(snapshot, std::nullopt, engine);
	EXPECT_EQ(engine, RocketSim::EngineId::V2);
}

TEST(SnapshotReplay, SnapshotsWithoutAnEngineUseTheAgentsEngineOrThrow)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	const std::string snapshot = SnapshotWithoutEngine();
	ASSERT_EQ(snapshot.find("engine="), std::string::npos);

	RocketSim::EngineId engine = RocketSim::EngineId::V2;
	ParseGameStateSnapshot(snapshot, RocketSim::EngineId::V3, engine);
	EXPECT_EQ(engine, RocketSim::EngineId::V3);

	EXPECT_THROW(ParseGameStateSnapshot(snapshot, std::nullopt, engine), std::runtime_error);
}
