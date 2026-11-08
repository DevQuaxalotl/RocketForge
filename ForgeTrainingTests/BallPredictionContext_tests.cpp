#include <gtest/gtest.h>

#include "TestSupport.h"

#include <RLGymCPP/BallPrediction/PredictionContext.h>
#include <RLGymCPP/BallPrediction/ProfiledObsBuilder.h>
#include <RLGymCPP/Gamestates/GameState.h>
#include <RocketSim/Arena.h>

#include <memory>
#include <stdexcept>

using namespace RocketSim;
namespace BP = RLGC::BallPrediction;

namespace
{
	constexpr float PROBE_SECONDS = 0.5f;

	/** Observes the predicted ball position and the context it was built under. **/
	class ProbeObs : public RLGC::ObsBuilder {
	public:
		RLGC::FList BuildObs(const RLGC::Player&, const RLGC::GameState& state) override {
			seenContext = state.predictionContext;
			if (throwOnBuild)
				throw std::runtime_error("probe");

			const BallState future = state.GetFutureBallState(PROBE_SECONDS);
			return { future.pos.x, future.pos.y, future.pos.z };
		}

		const BP::Context* seenContext = nullptr;
		bool throwOnBuild = false;
	};

	/** Falls steeply onto the field so the probe horizon spans a bounce, where engines diverge. **/
	BallState MakeBouncingBall() {
		BallState ball;
		ball.pos = Vec(300, -200, 250);
		ball.vel = Vec(900, 600, -2200);
		ball.angVel = Vec(1, -2, 0.5f);
		return ball;
	}

	Vec ProbePosition(const RLGC::FList& obs) { return Vec(obs[0], obs[1], obs[2]); }

	void ExpectSamePosition(const Vec& actual, const Vec& expected) {
		EXPECT_FLOAT_EQ(actual.x, expected.x);
		EXPECT_FLOAT_EQ(actual.y, expected.y);
		EXPECT_FLOAT_EQ(actual.z, expected.z);
	}
}

TEST(BallPredictionContext, MixedEnginePredictionStaysPerModel)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	ROCKETFORGE_REQUIRE_ENGINE(EngineId::V3);

	ArenaConfig matchConfig = {};
	matchConfig.engine = EngineId::V3;
	std::unique_ptr<Arena> arena(Arena::Create(GameMode::SOCCAR, matchConfig));
	arena->AddCar(Team::BLUE);
	arena->ball->SetState(MakeBouncingBall());
	const RLGC::GameState state(arena.get());

	RLGC::ProfiledObsBuilder v2Builder(new ProbeObs(), EngineId::V2);
	RLGC::ProfiledObsBuilder v3Builder(new ProbeObs(), EngineId::V3);

	const Vec v3Expected = arena->GetFutureBallState(PROBE_SECONDS).pos;

	// A v2 model predicts on its own v2 ball-only arena, fed the physical ball state.
	std::unique_ptr<Arena> v2Reference(Arena::Create(GameMode::SOCCAR));
	BallState v2Seed;
	v2Seed.pos = state.ball.pos;
	v2Seed.vel = state.ball.vel;
	v2Seed.angVel = state.ball.angVel;
	v2Seed.rotMat = state.ball.rotMat;
	v2Reference->ball->SetState(v2Seed);
	const Vec v2Expected = v2Reference->GetFutureBallState(PROBE_SECONDS).pos;

	// Equal expectations would let a mixed-up predictor pass unnoticed.
	const Vec separation = v3Expected - v2Expected;
	ASSERT_GT(separation.Length(), 0.01f);

	const RLGC::Player& player = state.players[0];
	ExpectSamePosition(ProbePosition(v3Builder.BuildObs(player, state)), v3Expected);
	ExpectSamePosition(ProbePosition(v2Builder.BuildObs(player, state)), v2Expected);

	// The other order must not leak either model's predictor into the other's observation.
	ExpectSamePosition(ProbePosition(v2Builder.BuildObs(player, state)), v2Expected);
	ExpectSamePosition(ProbePosition(v3Builder.BuildObs(player, state)), v3Expected);
	EXPECT_EQ(state.predictionContext, nullptr);
}

TEST(BallPredictionContext, ContextRestoredAfterBuild)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	RLGC::GameState state;
	RLGC::Player player;

	auto* probe = new ProbeObs();
	RLGC::ProfiledObsBuilder builder(probe, EngineId::V2);

	probe->throwOnBuild = true;
	EXPECT_THROW(builder.BuildObs(player, state), std::runtime_error);
	EXPECT_NE(probe->seenContext, nullptr);
	EXPECT_EQ(state.predictionContext, nullptr);

	// An outer scope's context comes back too, not just null.
	std::shared_ptr<BP::Context> outer = BP::Acquire(nullptr, GameMode::SOCCAR, EngineId::V2);
	state.predictionContext = outer.get();
	EXPECT_THROW(builder.BuildObs(player, state), std::runtime_error);
	EXPECT_NE(probe->seenContext, outer.get());
	EXPECT_EQ(state.predictionContext, outer.get());

	probe->throwOnBuild = false;
	state.predictionContext = nullptr;
	state.ball = MakeBouncingBall();
	builder.BuildObs(player, state);
	EXPECT_NE(probe->seenContext, nullptr);
	EXPECT_EQ(state.predictionContext, nullptr);
}

TEST(BallPredictionContext, SameArenaAndEngineShareOneContext)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	std::unique_ptr<Arena> arenaA(Arena::Create(GameMode::SOCCAR));
	std::unique_ptr<Arena> arenaB(Arena::Create(GameMode::SOCCAR));

	std::shared_ptr<BP::Context> first = BP::Acquire(arenaA.get(), GameMode::SOCCAR, EngineId::V2);
	std::shared_ptr<BP::Context> same = BP::Acquire(arenaA.get(), GameMode::SOCCAR, EngineId::V2);
	EXPECT_EQ(first, same);
	EXPECT_EQ(first->Engine(), EngineId::V2);

	EXPECT_NE(first, BP::Acquire(arenaB.get(), GameMode::SOCCAR, EngineId::V2));
	EXPECT_NE(first, BP::Acquire(arenaA.get(), GameMode::HOOPS, EngineId::V2));
	if (GetEngineInfo(EngineId::V3).available) {
		std::shared_ptr<BP::Context> v3 = BP::Acquire(arenaA.get(), GameMode::SOCCAR, EngineId::V3);
		EXPECT_NE(first, v3);
		EXPECT_EQ(v3->Engine(), EngineId::V3);
	}

	// Packet-driven play has no arena to share by, so every builder keeps its own predictor.
	EXPECT_NE(BP::Acquire(nullptr, GameMode::SOCCAR, EngineId::V2), BP::Acquire(nullptr, GameMode::SOCCAR, EngineId::V2));

	// The cache owns nothing, but a builder elsewhere in the process may still hold a context keyed by a reused arena address,
	// so count owners rather than expecting expiry.
	const long ownersBefore = first.use_count();
	std::weak_ptr<BP::Context> weak = first;
	first.reset();
	same.reset();
	EXPECT_EQ(weak.use_count(), ownersBefore - 2);
}

TEST(BallPredictionContext, UnavailableEngineFailsToAcquire)
{
	Testing::ScopedEngineUnavailable unavailable(EngineId::V3, "v3 is switched off for this test");
	EXPECT_THROW(BP::Acquire(nullptr, GameMode::SOCCAR, EngineId::V3), EngineUnavailableError);
}

TEST(BallPredictionContext, DropshotPredictionFollowsTheMatchTiles)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	ROCKETFORGE_REQUIRE_DROPSHOT_MESHES();
	std::unique_ptr<Arena> arena(Arena::Create(GameMode::DROPSHOT));
	BallState ball;
	ball.pos = Vec(0, 2500, 400);
	ball.vel = Vec(0, 0, -1000);
	arena->ball->SetState(ball);

	std::shared_ptr<BP::Context> context = BP::Acquire(arena.get(), GameMode::DROPSHOT, EngineId::V2);
	EXPECT_GT(context->Predict(arena->ball->GetState(), arena->tickCount, 1.5f).pos.z, 0);

	DropshotTilesState broken = arena->GetDropshotTilesState();
	for (int team = 0; team < RLConst::Dropshot::TEAM_AMOUNT; team++)
		for (int tile = 0; tile < RLConst::Dropshot::NUM_TILES_PER_TEAM; tile++)
			broken.states[team][tile].damageState = DropshotTileState::STATE_BROKEN;
	arena->SetDropshotTilesState(broken);

	EXPECT_LT(context->Predict(arena->ball->GetState(), arena->tickCount, 1.5f).pos.z, 0);
}
