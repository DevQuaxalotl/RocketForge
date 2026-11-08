#pragma once

#include "Framework.h"
#include "Engine.h"
#include "GameMode.h"
#include "CarConfig.h"
#include "CarControls.h"
#include "CarState.h"
#include "BallState.h"
#include "MutatorConfig.h"
#include "DropshotTiles.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace RocketSim {

namespace EngineBackend {
	class ArenaBackend;
	struct StepResult;
}

enum class RocketSimStage : byte {
	UNINITIALIZED,
	INITIALIZING,
	INITIALIZED
};

/** Loads the collision meshes for every engine in this build. **/
void Init(std::filesystem::path collisionMeshesFolder, bool silent = false);

/** Absolute, configure-time path of this source tree's collision_meshes folder; invalid for a binary moved off this machine. **/
std::filesystem::path CollisionMeshesFolder();

/** CollisionMeshesFolder() relative to the top-level source directory, for copies of the tree that keep its layout. **/
std::filesystem::path CollisionMeshesRelativePath();

RocketSimStage GetStage();

/** True when Init loaded `engine`'s arena meshes for `gameMode`; false for an engine this build lacks. **/
bool HasCollisionMeshes(EngineId engine, GameMode gameMode);

class Arena;
struct BallPredTracker;

/** Handle to one car in an Arena. State lives in the arena's engine. **/
class Car {
public:
	CarConfig config;
	Team team = Team::BLUE;

	/** Unique within the arena and always > 0. **/
	uint32_t id = 0;

	/** The controls the next Arena::Step simulates this car with. **/
	CarControls controls;

	CarState GetState();
	void SetState(const CarState& state);

	/** Clears the per-step contact flags (wasOnGroundDuringStep). **/
	void ResetStepContactState();

	Vec GetForwardDir() const;
	Vec GetRightDir() const;
	Vec GetUpDir() const;

	Arena* _arena = nullptr;
	uint32_t _idx = 0;

	Car() = default;
	Car(const Car&) = delete;
	Car& operator=(const Car&) = delete;
};

class Ball {
public:
	BallState GetState();
	void SetState(const BallState& state);

	Arena* _arena = nullptr;
	uint32_t _idx = 0;

	Ball() = default;
	Ball(const Ball&) = delete;
	Ball& operator=(const Ball&) = delete;
};

struct BoostPadConfig {
	Vec pos;
	bool isBig = false;
};

/** The part of a pad's state that changes every step. **/
struct BoostPadTimers {
	bool isActive = true;
	float cooldown = 0;
};

struct BoostPadState {
	bool isActive = true;
	float cooldown = 0;

	Car* curLockedCar = nullptr;
	uint32_t prevLockedCarID = 0;
};

class BoostPad {
public:
	BoostPadConfig config;

	BoostPadState GetState() const;
	void SetState(const BoostPadState& state);

	Arena* _arena = nullptr;
	uint32_t _idx = 0;

	BoostPad() = default;
	BoostPad(const BoostPad&) = delete;
	BoostPad& operator=(const BoostPad&) = delete;
};

/** Speed/memory trade-off of the engine's collision caches. **/
enum class ArenaMemWeightMode : byte {
	HEAVY,
	LIGHT,
	/** v2 has no balanced mode and runs it as HEAVY. **/
	BALANCED
};

struct ArenaConfig {
	ArenaMemWeightMode memWeightMode = ArenaMemWeightMode::HEAVY;

	/** Broadphase bounds and sizing; v3 accepts only the defaults. **/
	Vec minPos = Vec(-5600, -6000, 0);
	Vec maxPos = Vec(5600, 6000, 2200);
	float maxAABBLen = 370;

	/** Skips ball rotation updates for speed. Snowday always rotates the ball. **/
	bool noBallRot = true;

	/** Uses v2's RocketSim-specific broadphase; turn off only for giant custom maps. **/
	bool useCustomBroadphase = true;

	int maxObjects = 512;

	/** Replaces the standard boost pads with customBoostPads; v3 accepts only the defaults. **/
	bool useCustomBoostPads = false;
	std::vector<BoostPadConfig> customBoostPads = {};

	EngineId engine = EngineId::V2;
};

typedef std::function<void(Arena* arena, Car* bumper, Car* victim, bool isDemo, void* userInfo)> CarBumpEventFn;

/** One simulated game: cars, ball and boost pads stepped by the engine chosen in ArenaConfig::engine. **/
class Arena {
public:
	GameMode gameMode;

	/** Ticks simulated so far. Writable: Step(n) continues from whatever value it holds. **/
	uint64_t tickCount = 0;

	float tickTime;

	Ball* ball = nullptr;

	/** In add order. Arena owns the handles. **/
	std::vector<Car*> _cars;
	std::vector<BoostPad*> _boostPads;

	uint32_t _lastCarID = 0;
	std::unordered_map<uint32_t, Car*> _carIDMap;

	struct {
		CarBumpEventFn func = nullptr;
		void* userInfo = nullptr;
	} _bumpCallback;

	std::unique_ptr<EngineBackend::ArenaBackend> _backend;

	/** Throws EngineUnavailableError when config.engine isn't built. The caller owns the result. **/
	static Arena* Create(GameMode gameMode, const ArenaConfig& config = {}, float tickRate = 120);
	~Arena();

	Arena(const Arena&) = delete;
	Arena& operator=(const Arena&) = delete;

	EngineId GetEngine() const { return _config.engine; }

	const std::vector<Car*>& GetCars() { return _cars; }
	const std::vector<BoostPad*>& GetBoostPads() { return _boostPads; }

	/** Writes one entry per pad, in _boostPads order; cheaper than each pad's GetState since it skips the locked-car lookups. **/
	void GetBoostPadTimers(BoostPadTimers* out) const;
	const MutatorConfig& GetMutatorConfig() { return _mutatorConfig; }
	const ArenaConfig& GetArenaConfig() const { return _config; }

	float GetTickRate() const { return 1 / tickTime; }

	Car* AddCar(Team team, const CarConfig& config = CAR_CONFIG_OCTANE);

	/** Returns null when no car has that id. **/
	Car* GetCar(uint32_t id);

	/** Simulates every car with its current controls, then fires the bump callback for each bump in tick order. **/
	void Step(int ticksToSimulate = 1);

	void ResetToRandomKickoff(int seed = -1);

	/** True while the ball is in a goal; works in every game mode (never true in THE_VOID). **/
	bool IsBallScored() const;

	/**
	 * Purposely overestimating straight-line shot check, like the game's own shot prediction; ignores wall and ceiling bounces.
	 * extraMargin widens the goal (negative to narrow it). Soccar, Snowday and Hoops only.
	 */
	bool IsBallProbablyGoingIn(float maxTime = 2.f, float extraMargin = 0, Team* goalTeamOut = nullptr) const;

	/** Ball state `time` seconds ahead, from a prediction tracker shared by every caller on this arena. **/
	BallState GetFutureBallState(float time);

	/** True when the last prediction scores; outputs the time to score and where. **/
	bool GetPredictedScore(float& outScoreTime, Vec& outScorePos) const;

	void SetCarBumpCallback(CarBumpEventFn callbackFn, void* userInfo = nullptr) {
		_bumpCallback.func = callbackFn;
		_bumpCallback.userInfo = userInfo;
	}

	/** Dropshot only; throws std::logic_error in other modes. **/
	DropshotTilesState GetDropshotTilesState() const;
	void SetDropshotTilesState(const DropshotTilesState& tilesState);

private:
	Arena(GameMode gameMode, const ArenaConfig& config, float tickRate);

	ArenaConfig _config;
	MutatorConfig _mutatorConfig;
	BallPredTracker* _ballPredTracker = nullptr;

	/** Reused by every Step so stepping doesn't allocate. **/
	std::vector<CarControls> _stepControls;
	std::unique_ptr<EngineBackend::StepResult> _stepResult;
};

}
