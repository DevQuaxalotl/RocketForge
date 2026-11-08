#include <gtest/gtest.h>

#include "TestSupport.h"

#include "../ForgeTraining/RLGym/RocketSim/src/RocketSim.h"
#include <RocketSim/Arena.h>

#include "../ForgeTraining/RLGym/RocketSim/libsrc/bullet3-3.24/BulletCollision/BroadphaseCollision/btRSBroadphase.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {
	constexpr float FLOAT_TOLERANCE = 1e-4f;

	bool HasArenaMeshes(RocketSimV2::GameMode gameMode) {
		std::filesystem::path meshDir = RocketForgeTests::RepoCollisionMeshRoot() / RocketSimV2::GAMEMODE_STRS[static_cast<int>(gameMode)];
		if (!std::filesystem::is_directory(meshDir))
			return false;

		for (const auto& entry : std::filesystem::directory_iterator(meshDir))
			if (entry.is_regular_file() && entry.path().extension() == ".cmf")
				return true;
		return false;
	}

	class SimulationOptimizationScope {
	public:
		SimulationOptimizationScope(bool reuseWheelTransforms, bool rayAabbFilter) :
			prevReuseWheelTransforms(RocketSimV2::btVehicleRL::GetReuseWheelTransformsInUpdateVehicleFirst()),
			prevRayAabbFilter(btRSBroadphase::GetRayAabbFilterEnabled()) {
			RocketSimV2::btVehicleRL::SetReuseWheelTransformsInUpdateVehicleFirst(reuseWheelTransforms);
			btRSBroadphase::SetRayAabbFilterEnabled(rayAabbFilter);
		}

		~SimulationOptimizationScope() {
			RocketSimV2::btVehicleRL::SetReuseWheelTransformsInUpdateVehicleFirst(prevReuseWheelTransforms);
			btRSBroadphase::SetRayAabbFilterEnabled(prevRayAabbFilter);
		}

	private:
		bool prevReuseWheelTransforms;
		bool prevRayAabbFilter;
	};

	std::string WithField(const std::string& context, const std::string& field) {
		return context + " " + field;
	}

	template <typename T, typename U>
	void ExpectEqual(const T& reference, const U& optimized, const std::string& field) {
		if (!(reference == optimized)) {
			std::ostringstream oss;
			oss << field << " mismatch: reference=" << ::testing::PrintToString(reference)
				<< ", optimized=" << ::testing::PrintToString(optimized);
			FAIL() << oss.str();
		}
	}

	void ExpectFloatNear(float reference, float optimized, const std::string& field, float tolerance = FLOAT_TOLERANCE) {
		if (!(std::fabs(reference - optimized) <= tolerance)) {
			std::ostringstream oss;
			oss << field << " mismatch: reference=" << reference
				<< ", optimized=" << optimized
				<< ", diff=" << std::fabs(reference - optimized)
				<< ", tolerance=" << tolerance;
			FAIL() << oss.str();
		}
	}

	void ExpectVecNear(const RocketSimV2::Vec& reference, const RocketSimV2::Vec& optimized, const std::string& field) {
		ExpectFloatNear(reference.x, optimized.x, WithField(field, ".x"));
		ExpectFloatNear(reference.y, optimized.y, WithField(field, ".y"));
		ExpectFloatNear(reference.z, optimized.z, WithField(field, ".z"));
	}

	void ExpectRotMatNear(const RocketSimV2::RotMat& reference, const RocketSimV2::RotMat& optimized, const std::string& field) {
		ExpectVecNear(reference.forward, optimized.forward, WithField(field, ".forward"));
		ExpectVecNear(reference.right, optimized.right, WithField(field, ".right"));
		ExpectVecNear(reference.up, optimized.up, WithField(field, ".up"));
	}

	void ExpectControlsNear(const RocketSimV2::CarControls& reference, const RocketSimV2::CarControls& optimized, const std::string& field) {
		ExpectFloatNear(reference.throttle, optimized.throttle, WithField(field, ".throttle"));
		ExpectFloatNear(reference.steer, optimized.steer, WithField(field, ".steer"));
		ExpectFloatNear(reference.pitch, optimized.pitch, WithField(field, ".pitch"));
		ExpectFloatNear(reference.yaw, optimized.yaw, WithField(field, ".yaw"));
		ExpectFloatNear(reference.roll, optimized.roll, WithField(field, ".roll"));
		ExpectEqual(reference.jump, optimized.jump, WithField(field, ".jump"));
		ExpectEqual(reference.boost, optimized.boost, WithField(field, ".boost"));
		ExpectEqual(reference.handbrake, optimized.handbrake, WithField(field, ".handbrake"));
	}

	void ExpectBallHitInfoNear(const RocketSimV2::BallHitInfo& reference, const RocketSimV2::BallHitInfo& optimized, const std::string& field) {
		ExpectEqual(reference.isValid, optimized.isValid, WithField(field, ".isValid"));
		ExpectVecNear(reference.relativePosOnBall, optimized.relativePosOnBall, WithField(field, ".relativePosOnBall"));
		ExpectVecNear(reference.ballPos, optimized.ballPos, WithField(field, ".ballPos"));
		ExpectVecNear(reference.extraHitVel, optimized.extraHitVel, WithField(field, ".extraHitVel"));
		ExpectEqual(reference.tickCountWhenHit, optimized.tickCountWhenHit, WithField(field, ".tickCountWhenHit"));
		ExpectEqual(reference.tickCountWhenExtraImpulseApplied, optimized.tickCountWhenExtraImpulseApplied, WithField(field, ".tickCountWhenExtraImpulseApplied"));
	}

	void ExpectBallStateNear(const RocketSimV2::BallState& reference, const RocketSimV2::BallState& optimized, const std::string& field) {
		ExpectEqual(reference.tickCountSinceUpdate, optimized.tickCountSinceUpdate, WithField(field, ".tickCountSinceUpdate"));
		ExpectVecNear(reference.pos, optimized.pos, WithField(field, ".pos"));
		ExpectRotMatNear(reference.rotMat, optimized.rotMat, WithField(field, ".rotMat"));
		ExpectVecNear(reference.vel, optimized.vel, WithField(field, ".vel"));
		ExpectVecNear(reference.angVel, optimized.angVel, WithField(field, ".angVel"));
		ExpectFloatNear(reference.hsInfo.yTargetDir, optimized.hsInfo.yTargetDir, WithField(field, ".hsInfo.yTargetDir"));
		ExpectFloatNear(reference.hsInfo.curTargetSpeed, optimized.hsInfo.curTargetSpeed, WithField(field, ".hsInfo.curTargetSpeed"));
		ExpectFloatNear(reference.hsInfo.timeSinceHit, optimized.hsInfo.timeSinceHit, WithField(field, ".hsInfo.timeSinceHit"));
		ExpectEqual(reference.dsInfo.chargeLevel, optimized.dsInfo.chargeLevel, WithField(field, ".dsInfo.chargeLevel"));
		ExpectFloatNear(reference.dsInfo.accumulatedHitForce, optimized.dsInfo.accumulatedHitForce, WithField(field, ".dsInfo.accumulatedHitForce"));
		ExpectFloatNear(reference.dsInfo.yTargetDir, optimized.dsInfo.yTargetDir, WithField(field, ".dsInfo.yTargetDir"));
		ExpectEqual(reference.dsInfo.hasDamaged, optimized.dsInfo.hasDamaged, WithField(field, ".dsInfo.hasDamaged"));
		if (reference.dsInfo.hasDamaged || optimized.dsInfo.hasDamaged)
			ExpectEqual(reference.dsInfo.lastDamageTick, optimized.dsInfo.lastDamageTick, WithField(field, ".dsInfo.lastDamageTick"));
	}

	void ExpectCarStateNear(const RocketSimV2::CarState& reference, const RocketSimV2::CarState& optimized, const std::string& field) {
		ExpectEqual(reference.tickCountSinceUpdate, optimized.tickCountSinceUpdate, WithField(field, ".tickCountSinceUpdate"));
		ExpectVecNear(reference.pos, optimized.pos, WithField(field, ".pos"));
		ExpectRotMatNear(reference.rotMat, optimized.rotMat, WithField(field, ".rotMat"));
		ExpectVecNear(reference.vel, optimized.vel, WithField(field, ".vel"));
		ExpectVecNear(reference.angVel, optimized.angVel, WithField(field, ".angVel"));
		ExpectEqual(reference.isOnGround, optimized.isOnGround, WithField(field, ".isOnGround"));
		ExpectEqual(reference.obsOnGround, optimized.obsOnGround, WithField(field, ".obsOnGround"));
		for (int i = 0; i < 4; i++)
			ExpectEqual(reference.wheelsWithContact[i], optimized.wheelsWithContact[i], WithField(field, ".wheelsWithContact[" + std::to_string(i) + "]"));
		ExpectEqual(reference.hasJumped, optimized.hasJumped, WithField(field, ".hasJumped"));
		ExpectEqual(reference.hasDoubleJumped, optimized.hasDoubleJumped, WithField(field, ".hasDoubleJumped"));
		ExpectEqual(reference.hasFlipped, optimized.hasFlipped, WithField(field, ".hasFlipped"));
		ExpectVecNear(reference.flipRelTorque, optimized.flipRelTorque, WithField(field, ".flipRelTorque"));
		ExpectFloatNear(reference.jumpTime, optimized.jumpTime, WithField(field, ".jumpTime"));
		ExpectFloatNear(reference.flipTime, optimized.flipTime, WithField(field, ".flipTime"));
		ExpectEqual(reference.isFlipping, optimized.isFlipping, WithField(field, ".isFlipping"));
		ExpectEqual(reference.isJumping, optimized.isJumping, WithField(field, ".isJumping"));
		ExpectFloatNear(reference.airTime, optimized.airTime, WithField(field, ".airTime"));
		ExpectFloatNear(reference.airTimeSinceJump, optimized.airTimeSinceJump, WithField(field, ".airTimeSinceJump"));
		ExpectFloatNear(reference.boost, optimized.boost, WithField(field, ".boost"));
		ExpectFloatNear(reference.timeSinceBoosted, optimized.timeSinceBoosted, WithField(field, ".timeSinceBoosted"));
		ExpectEqual(reference.isBoosting, optimized.isBoosting, WithField(field, ".isBoosting"));
		ExpectFloatNear(reference.boostingTime, optimized.boostingTime, WithField(field, ".boostingTime"));
		ExpectEqual(reference.isSupersonic, optimized.isSupersonic, WithField(field, ".isSupersonic"));
		ExpectFloatNear(reference.supersonicTime, optimized.supersonicTime, WithField(field, ".supersonicTime"));
		ExpectFloatNear(reference.handbrakeVal, optimized.handbrakeVal, WithField(field, ".handbrakeVal"));
		ExpectEqual(reference.isAutoFlipping, optimized.isAutoFlipping, WithField(field, ".isAutoFlipping"));
		ExpectFloatNear(reference.autoFlipTimer, optimized.autoFlipTimer, WithField(field, ".autoFlipTimer"));
		ExpectFloatNear(reference.autoFlipTorqueScale, optimized.autoFlipTorqueScale, WithField(field, ".autoFlipTorqueScale"));
		ExpectEqual(reference.worldContact.hasContact, optimized.worldContact.hasContact, WithField(field, ".worldContact.hasContact"));
		ExpectVecNear(reference.worldContact.contactNormal, optimized.worldContact.contactNormal, WithField(field, ".worldContact.contactNormal"));
		ExpectEqual(reference.carContact.otherCarID, optimized.carContact.otherCarID, WithField(field, ".carContact.otherCarID"));
		ExpectFloatNear(reference.carContact.cooldownTimer, optimized.carContact.cooldownTimer, WithField(field, ".carContact.cooldownTimer"));
		ExpectEqual(reference.isDemoed, optimized.isDemoed, WithField(field, ".isDemoed"));
		ExpectFloatNear(reference.demoRespawnTimer, optimized.demoRespawnTimer, WithField(field, ".demoRespawnTimer"));
		ExpectBallHitInfoNear(reference.ballHitInfo, optimized.ballHitInfo, WithField(field, ".ballHitInfo"));
		ExpectControlsNear(reference.lastControls, optimized.lastControls, WithField(field, ".lastControls"));
	}

	std::vector<RocketSimV2::Car*> GetCarsSorted(RocketSimV2::Arena& arena) {
		std::vector<RocketSimV2::Car*> cars(arena.GetCars().begin(), arena.GetCars().end());
		std::sort(cars.begin(), cars.end(), [](const RocketSimV2::Car* lhs, const RocketSimV2::Car* rhs) {
			return lhs->id < rhs->id;
		});
		return cars;
	}

	void ExpectBoostPadsNear(RocketSimV2::Arena& reference, RocketSimV2::Arena& optimized, const std::string& field) {
		const auto& referencePads = reference.GetBoostPads();
		const auto& optimizedPads = optimized.GetBoostPads();
		ExpectEqual(referencePads.size(), optimizedPads.size(), WithField(field, ".size"));
		for (size_t i = 0; i < referencePads.size(); i++) {
			RocketSimV2::BoostPadState referenceState = referencePads[i]->GetState();
			RocketSimV2::BoostPadState optimizedState = optimizedPads[i]->GetState();
			const std::string padField = WithField(field, "[" + std::to_string(i) + "]");
			ExpectEqual(referenceState.isActive, optimizedState.isActive, WithField(padField, ".isActive"));
			ExpectFloatNear(referenceState.cooldown, optimizedState.cooldown, WithField(padField, ".cooldown"));
			ExpectEqual(referenceState.prevLockedCarID, optimizedState.prevLockedCarID, WithField(padField, ".prevLockedCarID"));
		}
	}

	void ExpectDropshotTilesNear(const RocketSimV2::DropshotTilesState& reference, const RocketSimV2::DropshotTilesState& optimized, const std::string& field) {
		for (int team = 0; team < RocketSimV2::RLConst::Dropshot::TEAM_AMOUNT; team++) {
			for (int i = 0; i < RocketSimV2::RLConst::Dropshot::NUM_TILES_PER_TEAM; i++) {
				ExpectEqual(
					static_cast<int>(reference.states[team][i].damageState),
					static_cast<int>(optimized.states[team][i].damageState),
					WithField(field, "[" + std::to_string(team) + "][" + std::to_string(i) + "].damageState")
				);
			}
		}
	}

	void ExpectArenasEquivalent(RocketSimV2::Arena& reference, RocketSimV2::Arena& optimized, const std::string& context) {
		ExpectEqual(reference.gameMode, optimized.gameMode, WithField(context, "gameMode"));
		ExpectFloatNear(reference.GetTickRate(), optimized.GetTickRate(), WithField(context, "tickRate"));
		ExpectEqual(reference.tickCount, optimized.tickCount, WithField(context, "tickCount"));
		ExpectEqual(reference.IsBallScored(), optimized.IsBallScored(), WithField(context, "IsBallScored"));

		ExpectBallStateNear(reference.ball->GetState(), optimized.ball->GetState(), WithField(context, "ball"));

		std::vector<RocketSimV2::Car*> referenceCars = GetCarsSorted(reference);
		std::vector<RocketSimV2::Car*> optimizedCars = GetCarsSorted(optimized);
		ExpectEqual(referenceCars.size(), optimizedCars.size(), WithField(context, "cars.size"));
		for (size_t i = 0; i < referenceCars.size(); i++) {
			const std::string carField = WithField(context, "car[" + std::to_string(i) + "]");
			ExpectEqual(referenceCars[i]->id, optimizedCars[i]->id, WithField(carField, ".id"));
			ExpectEqual(static_cast<int>(referenceCars[i]->team), static_cast<int>(optimizedCars[i]->team), WithField(carField, ".team"));
			ExpectControlsNear(referenceCars[i]->controls, optimizedCars[i]->controls, WithField(carField, ".controls"));
			ExpectCarStateNear(referenceCars[i]->GetState(), optimizedCars[i]->GetState(), WithField(carField, ".state"));
		}

		ExpectBoostPadsNear(reference, optimized, WithField(context, "boostPads"));
		ExpectDropshotTilesNear(reference.GetDropshotTilesState(), optimized.GetDropshotTilesState(), WithField(context, "dropshotTiles"));
	}

	float Wave(int tick, int salt) {
		return std::sin((static_cast<float>(tick) * 0.073f) + (static_cast<float>(salt) * 1.117f));
	}

	RocketSimV2::CarControls MakeControls(int tick, int carIndex) {
		RocketSimV2::CarControls controls;
		controls.throttle = Wave(tick, carIndex * 11 + 1);
		controls.steer = Wave(tick, carIndex * 11 + 2) * 0.85f;
		controls.pitch = Wave(tick, carIndex * 11 + 3);
		controls.yaw = Wave(tick, carIndex * 11 + 4);
		controls.roll = Wave(tick, carIndex * 11 + 5);
		controls.jump = ((tick + carIndex * 7) % 53) < 4;
		controls.boost = ((tick + carIndex * 11) % 37) < 12;
		controls.handbrake = ((tick + carIndex * 13) % 29) < 9;
		return controls;
	}

	void ApplyControls(RocketSimV2::Arena& arena, int tick) {
		std::vector<RocketSimV2::Car*> cars = GetCarsSorted(arena);
		for (size_t i = 0; i < cars.size(); i++)
			cars[i]->controls = MakeControls(tick, static_cast<int>(i));
	}

	RocketSimV2::CarState MakeCarState(
		const RocketSimV2::Vec& pos,
		const RocketSimV2::Vec& forward,
		const RocketSimV2::Vec& up,
		const RocketSimV2::Vec& vel,
		const RocketSimV2::Vec& angVel,
		float boost,
		bool onGround
	) {
		RocketSimV2::CarState state;
		state.pos = pos;
		state.rotMat = RocketSimV2::RotMat::LookAt(forward, up);
		state.vel = vel;
		state.angVel = angVel;
		state.boost = boost;
		state.isOnGround = onGround;
		state.obsOnGround = onGround;
		return state;
	}

	RocketSimV2::BallState MakeBallState(
		const RocketSimV2::Vec& pos,
		const RocketSimV2::Vec& vel,
		const RocketSimV2::Vec& angVel
	) {
		RocketSimV2::BallState state;
		state.pos = pos;
		state.rotMat = RocketSimV2::RotMat::GetIdentity();
		state.vel = vel;
		state.angVel = angVel;
		return state;
	}

	template <typename ArenaT>
	void SetBoostPadsDeterministic(ArenaT& arena) {
		const auto& pads = arena.GetBoostPads();
		for (size_t i = 0; i < pads.size(); i++) {
			auto state = pads[i]->GetState();
			state.isActive = (i % 5) != 0;
			state.cooldown = state.isActive ? 0.f : 1.25f + static_cast<float>(i % 7) * 0.1f;
			state.prevLockedCarID = 0;
			pads[i]->SetState(state);
		}
	}

	/** Shared by the direct v2 arena and the shared-API arena so both start from bit-identical states. **/
	template <typename ArenaT>
	void PopulateSoccarArena(ArenaT& arena) {
		auto* blueA = arena.AddCar(RocketSimV2::Team::BLUE, RocketSimV2::CAR_CONFIG_OCTANE);
		auto* orangeA = arena.AddCar(RocketSimV2::Team::ORANGE, RocketSimV2::CAR_CONFIG_DOMINUS);
		auto* blueB = arena.AddCar(RocketSimV2::Team::BLUE, RocketSimV2::CAR_CONFIG_MERC);
		auto* orangeB = arena.AddCar(RocketSimV2::Team::ORANGE, RocketSimV2::CAR_CONFIG_BREAKOUT);
		blueA->SetState(MakeCarState(RocketSimV2::Vec(-1200, -2200, 17), RocketSimV2::Vec(1, 0.2f, 0), RocketSimV2::Vec(0, 0, 1), RocketSimV2::Vec(1050, 850, 0), RocketSimV2::Vec(0.15f, -0.1f, 0.5f), 73.f, true));
		orangeA->SetState(MakeCarState(RocketSimV2::Vec(900, 1600, 280), RocketSimV2::Vec(-0.4f, -1, -0.1f), RocketSimV2::Vec(0, 0, 1), RocketSimV2::Vec(-650, -900, 180), RocketSimV2::Vec(-0.4f, 0.6f, 1.2f), 42.f, false));
		blueB->SetState(MakeCarState(RocketSimV2::Vec(3820, -700, 125), RocketSimV2::Vec(-0.15f, 1, 0.1f), RocketSimV2::Vec(-1, 0, 0.1f), RocketSimV2::Vec(-350, 1100, 40), RocketSimV2::Vec(0.2f, 0.7f, -1.1f), 100.f, false));
		orangeB->SetState(MakeCarState(RocketSimV2::Vec(-2600, 3000, 17), RocketSimV2::Vec(0.5f, -1, 0), RocketSimV2::Vec(0, 0, 1), RocketSimV2::Vec(800, -650, 0), RocketSimV2::Vec(-0.3f, 0.15f, 0.8f), 15.f, true));
		arena.ball->SetState(MakeBallState(RocketSimV2::Vec(160, -400, 320), RocketSimV2::Vec(1300, 900, 520), RocketSimV2::Vec(1.4f, -2.2f, 0.6f)));
		SetBoostPadsDeterministic(arena);
	}

	std::unique_ptr<RocketSimV2::Arena> MakeSoccarArena() {
		std::unique_ptr<RocketSimV2::Arena> arena(RocketSimV2::Arena::Create(RocketSimV2::GameMode::SOCCAR));
		PopulateSoccarArena(*arena);
		return arena;
	}

	std::unique_ptr<RocketSimV2::Arena> MakeHoopsArena() {
		std::unique_ptr<RocketSimV2::Arena> arena(RocketSimV2::Arena::Create(RocketSimV2::GameMode::HOOPS));
		RocketSimV2::Car* blue = arena->AddCar(RocketSimV2::Team::BLUE, RocketSimV2::CAR_CONFIG_OCTANE);
		RocketSimV2::Car* orange = arena->AddCar(RocketSimV2::Team::ORANGE, RocketSimV2::CAR_CONFIG_DOMINUS);
		RocketSimV2::Car* blueAir = arena->AddCar(RocketSimV2::Team::BLUE, RocketSimV2::CAR_CONFIG_PLANK);

		blue->SetState(MakeCarState(RocketSimV2::Vec(-800, -2500, 17), RocketSimV2::Vec(0.4f, 1, 0), RocketSimV2::Vec(0, 0, 1), RocketSimV2::Vec(950, 750, 0), RocketSimV2::Vec(0.1f, -0.2f, 0.4f), 88.f, true));
		orange->SetState(MakeCarState(RocketSimV2::Vec(700, 2500, 17), RocketSimV2::Vec(-0.4f, -1, 0), RocketSimV2::Vec(0, 0, 1), RocketSimV2::Vec(-880, -620, 0), RocketSimV2::Vec(-0.2f, 0.2f, -0.3f), 66.f, true));
		blueAir->SetState(MakeCarState(RocketSimV2::Vec(2600, -200, 700), RocketSimV2::Vec(-1, 0.2f, -0.25f), RocketSimV2::Vec(0, 0, 1), RocketSimV2::Vec(-850, 300, -250), RocketSimV2::Vec(1.3f, -0.9f, 0.6f), 35.f, false));
		arena->ball->SetState(MakeBallState(RocketSimV2::Vec(80, 200, 610), RocketSimV2::Vec(600, 1450, 450), RocketSimV2::Vec(-1.1f, 0.8f, 2.0f)));
		SetBoostPadsDeterministic(*arena);
		return arena;
	}

	std::unique_ptr<RocketSimV2::Arena> MakeDropshotArena() {
		std::unique_ptr<RocketSimV2::Arena> arena(RocketSimV2::Arena::Create(RocketSimV2::GameMode::DROPSHOT));
		RocketSimV2::Car* blue = arena->AddCar(RocketSimV2::Team::BLUE, RocketSimV2::CAR_CONFIG_OCTANE);
		RocketSimV2::Car* orange = arena->AddCar(RocketSimV2::Team::ORANGE, RocketSimV2::CAR_CONFIG_MERC);
		RocketSimV2::Car* blueWall = arena->AddCar(RocketSimV2::Team::BLUE, RocketSimV2::CAR_CONFIG_DOMINUS);

		blue->SetState(MakeCarState(RocketSimV2::Vec(-1100, -900, 19), RocketSimV2::Vec(0.8f, 0.6f, 0), RocketSimV2::Vec(0, 0, 1), RocketSimV2::Vec(900, 450, 0), RocketSimV2::Vec(0.2f, -0.1f, 0.7f), 95.f, true));
		orange->SetState(MakeCarState(RocketSimV2::Vec(1250, 850, 19), RocketSimV2::Vec(-0.7f, -0.6f, 0), RocketSimV2::Vec(0, 0, 1), RocketSimV2::Vec(-760, -420, 0), RocketSimV2::Vec(-0.1f, 0.3f, -0.8f), 54.f, true));
		blueWall->SetState(MakeCarState(RocketSimV2::Vec(3820, 120, 240), RocketSimV2::Vec(0, 1, 0.15f), RocketSimV2::Vec(-1, 0, 0.2f), RocketSimV2::Vec(-500, 920, 110), RocketSimV2::Vec(0.8f, -1.0f, 0.4f), 22.f, false));

		RocketSimV2::BallState ballState = MakeBallState(RocketSimV2::Vec(-120, 120, 720), RocketSimV2::Vec(1050, -850, -620), RocketSimV2::Vec(2.1f, -0.5f, 1.4f));
		ballState.dsInfo.chargeLevel = 2;
		ballState.dsInfo.accumulatedHitForce = 3500.f;
		ballState.dsInfo.yTargetDir = 1.f;
		arena->ball->SetState(ballState);

		RocketSimV2::DropshotTilesState tiles = arena->GetDropshotTilesState();
		tiles.states[0][3].damageState = RocketSimV2::DropshotTileState::STATE_DAMAGED;
		tiles.states[1][7].damageState = RocketSimV2::DropshotTileState::STATE_BROKEN;
		arena->SetDropshotTilesState(tiles);
		return arena;
	}

	void RunScenario(const std::string& name, int ticks, std::unique_ptr<RocketSimV2::Arena>(*makeArena)()) {
		std::unique_ptr<RocketSimV2::Arena> reference = makeArena();
		std::unique_ptr<RocketSimV2::Arena> optimized = makeArena();

		ExpectArenasEquivalent(*reference, *optimized, name + " initial");
		for (int tick = 0; tick < ticks; tick++) {
			ApplyControls(*reference, tick);
			ApplyControls(*optimized, tick);

			{
				SimulationOptimizationScope referenceMode(false, false);
				reference->Step(1);
			}
			{
				SimulationOptimizationScope optimizedMode(true, true);
				optimized->Step(1);
			}

			ExpectArenasEquivalent(*reference, *optimized, name + " tick " + std::to_string(tick + 1));
		}
	}
	struct BumpRecord {
		uint32_t bumperId;
		uint32_t victimId;
		bool isDemo;

		bool operator==(const BumpRecord& other) const {
			return bumperId == other.bumperId && victimId == other.victimId && isDemo == other.isDemo;
		}
	};

	void RecordDirectBump(RocketSimV2::Arena*, RocketSimV2::Car* bumper, RocketSimV2::Car* victim, bool isDemo, void* userInfo) {
		static_cast<std::vector<BumpRecord>*>(userInfo)->push_back({ bumper->id, victim->id, isDemo });
	}

	void RecordSharedBump(RocketSim::Arena*, RocketSim::Car* bumper, RocketSim::Car* victim, bool isDemo, void* userInfo) {
		static_cast<std::vector<BumpRecord>*>(userInfo)->push_back({ bumper->id, victim->id, isDemo });
	}

	void ExpectSharedMatchesDirect(RocketSim::Arena& shared, RocketSimV2::Arena& direct, const std::string& context) {
		ExpectEqual(shared.tickCount, direct.tickCount, WithField(context, "tickCount"));
		ExpectEqual(shared.IsBallScored(), direct.IsBallScored(), WithField(context, "IsBallScored"));
		ExpectBallStateNear(direct.ball->GetState(), shared.ball->GetState(), WithField(context, "ball"));

		std::vector<RocketSimV2::Car*> directCars = GetCarsSorted(direct);
		ExpectEqual(shared._cars.size(), directCars.size(), WithField(context, "cars.size"));
		for (size_t i = 0; i < directCars.size(); i++) {
			const std::string carField = WithField(context, "car[" + std::to_string(i) + "]");
			ExpectEqual(shared._cars[i]->id, directCars[i]->id, WithField(carField, ".id"));
			ExpectControlsNear(directCars[i]->controls, shared._cars[i]->controls, WithField(carField, ".controls"));
			ExpectCarStateNear(directCars[i]->GetState(), shared._cars[i]->GetState(), WithField(carField, ".state"));
		}

		for (size_t i = 0; i < direct._boostPads.size(); i++) {
			const RocketSimV2::BoostPadState directPad = direct._boostPads[i]->GetState();
			const RocketSim::BoostPadState sharedPad = shared._boostPads[i]->GetState();
			const std::string padField = WithField(context, "pad[" + std::to_string(i) + "]");
			ExpectEqual(sharedPad.isActive, directPad.isActive, WithField(padField, ".isActive"));
			ExpectFloatNear(directPad.cooldown, sharedPad.cooldown, WithField(padField, ".cooldown"));
			ExpectEqual(sharedPad.prevLockedCarID, directPad.prevLockedCarID, WithField(padField, ".prevLockedCarID"));
			ExpectEqual(sharedPad.curLockedCar ? sharedPad.curLockedCar->id : 0u, directPad.curLockedCar ? directPad.curLockedCar->id : 0u,
				WithField(padField, ".curLockedCar"));
		}
	}
}

TEST(SimulationEquivalence, SharedArenaMatchesDirectV2) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	constexpr int TICK_SKIP = 8;
	constexpr int NUM_STEPS = 90;
	constexpr int TELEPORT_STEP = 45;

	std::unique_ptr<RocketSimV2::Arena> direct = MakeSoccarArena();
	std::unique_ptr<RocketSim::Arena> shared(RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR));
	PopulateSoccarArena(*shared);

	std::vector<BumpRecord> directBumps;
	std::vector<BumpRecord> sharedBumps;
	direct->SetCarBumpCallback(RecordDirectBump, &directBumps);
	shared->SetCarBumpCallback(RecordSharedBump, &sharedBumps);

	ExpectSharedMatchesDirect(*shared, *direct, "initial");
	for (int step = 0; step < NUM_STEPS; step++) {
		const std::string context = "step " + std::to_string(step + 1);

		// A mid-episode state set, as state setters and replays do, including a car contact that refers to another car by id.
		if (step == TELEPORT_STEP) {
			RocketSimV2::CarState teleported = GetCarsSorted(*direct)[1]->GetState();
			teleported.pos = RocketSimV2::Vec(-1500, 400, 17);
			teleported.carContact.otherCarID = 3;
			teleported.carContact.cooldownTimer = 0.2f;
			GetCarsSorted(*direct)[1]->SetState(teleported);
			shared->_cars[1]->SetState(teleported);

			const RocketSimV2::BallState ball = MakeBallState(RocketSimV2::Vec(-1500, 600, 93), RocketSimV2::Vec(0, -300, 0), RocketSimV2::Vec());
			direct->ball->SetState(ball);
			shared->ball->SetState(ball);
		}

		std::vector<RocketSimV2::Car*> directCars = GetCarsSorted(*direct);
		for (size_t i = 0; i < directCars.size(); i++) {
			const RocketSimV2::CarControls controls = MakeControls(step * TICK_SKIP, static_cast<int>(i));
			directCars[i]->controls = controls;
			directCars[i]->ResetStepContactState();
			shared->_cars[i]->controls = controls;
			shared->_cars[i]->ResetStepContactState();
		}

		direct->Step(TICK_SKIP);
		shared->Step(TICK_SKIP);

		ExpectSharedMatchesDirect(*shared, *direct, context);
		ASSERT_EQ(sharedBumps.size(), directBumps.size()) << context;
		for (size_t i = 0; i < directBumps.size(); i++)
			EXPECT_TRUE(sharedBumps[i] == directBumps[i]) << context << " bump " << i;
	}
}

TEST(SimulationEquivalence, OptimizedRaycastsMatchReferenceAcrossModes) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	RunScenario("soccar", 300, MakeSoccarArena);
	if (HasArenaMeshes(RocketSimV2::GameMode::HOOPS))
		RunScenario("hoops", 260, MakeHoopsArena);
	if (HasArenaMeshes(RocketSimV2::GameMode::DROPSHOT))
		RunScenario("dropshot", 260, MakeDropshotArena);
}
