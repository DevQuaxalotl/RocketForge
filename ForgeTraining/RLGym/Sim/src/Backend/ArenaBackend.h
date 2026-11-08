#pragma once

#include <RocketSim/Arena.h>

#include <memory>
#include <vector>

namespace RocketSim::EngineBackend {

/** Filled by Step; Arena fires its bump callback from it. **/
struct StepResult {
	struct Bump {
		uint32_t bumperIdx, victimIdx;
		bool isDemo;
	};

	/** In tick order. **/
	std::vector<Bump> bumps;
};

/**
 * One engine's arena, addressed by car and pad index.
 * Arena keeps every call in the module that created it through this virtual interface,
 * since RLGym is linked into both GigaLearn.dll and the executables.
 * Car states use backend indices for car references: carContact.otherCarID is (car index + 1), or 0 for none.
 *
 * Each backend merges what happens during a step into the car states it returns:
 * ballHitInfo is the car's latest touch and persists across steps, with tickCountWhenHit counted in Step's startTick tick space;
 * wasOnGroundDuringStep accumulates over every stepped tick until ResetCarStepContact.
 */
class ArenaBackend {
public:
	virtual ~ArenaBackend() = default;

	virtual EngineId Engine() const = 0;

	/** Returns the backend car index; indices run 0, 1, 2... in add order. **/
	virtual uint32_t AddCar(Team team, const CarConfig& config) = 0;

	/** `controls` holds one entry per car index; the step's first tick is numbered `startTick`. **/
	virtual void Step(int ticks, uint64_t startTick, const CarControls* controls, StepResult& out) = 0;

	virtual CarState GetCarState(uint32_t idx) = 0;
	virtual void SetCarState(uint32_t idx, const CarState& state) = 0;
	virtual void ResetCarStepContact(uint32_t idx) = 0;

	virtual BallState GetBallState() = 0;
	virtual void SetBallState(const BallState& state) = 0;

	virtual uint32_t NumPads() = 0;
	virtual BoostPadConfig GetPadConfig(uint32_t idx) = 0;

	/** UINT32_MAX = none; Arena maps idx <-> Car*. **/
	struct BoostPadStateRaw {
		bool isActive;
		float cooldown;
		uint32_t lockedCarIdx;
		uint32_t prevLockedCarIdx;
	};

	virtual BoostPadStateRaw GetPadState(uint32_t idx) = 0;

	/** Writes NumPads() entries. **/
	virtual void GetPadTimers(BoostPadTimers* out) = 0;
	virtual void SetPadState(uint32_t idx, const BoostPadStateRaw& state) = 0;

	virtual void ResetToRandomKickoff(int seed) = 0;
	virtual bool IsBallScored() = 0;

	virtual DropshotTilesState GetTiles() = 0;
	virtual void SetTiles(const DropshotTilesState& tiles) = 0;

	/** Steps a ball-only arena from `start`; writes numTicks states; returns the first scoring index or -1. **/
	virtual int PredictBall(const BallState& start, int numTicks, BallState* out) = 0;
};

/** `arenaTickCount` is the owning Arena's tickCount, for backends whose engine keeps a clock of its own. **/
std::unique_ptr<ArenaBackend> CreateBackend(
	EngineId engine,
	GameMode mode,
	const ArenaConfig& config,
	float tickRate,
	const uint64_t& arenaTickCount);

}
