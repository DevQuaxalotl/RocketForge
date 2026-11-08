#pragma once

#include "ArenaBackend.h"

#include <rocketsim_ffi.h>

namespace RocketSim::EngineBackend {

void InitV3(const std::filesystem::path& collisionMeshesFolder, bool silent);
bool V3IsInitialized();
bool V3HasCollisionMeshes(GameMode gameMode);

/**
 * Drives an upstream v3 arena through the rsf_ C ABI.
 * Every step and setter is one Rust call; getters read a cache that the step snapshot and each setter refresh.
 *
 * The engine keeps its own tick clock, which only advances while stepping.
 * Arena::tickCount is the clock callers see, and they may rewrite it, so ball damage ticks are translated between the two
 * as ages (ticks since the last damage), which both clocks agree on.
 * Two edges follow from that.
 * A damage tick read after tickCount was rewritten below its age clamps to 0, so setting that state back shortens the age.
 * A damage tick later than tickCount, or older than the engine's whole run, is set as never having damaged.
 */
class V3Backend : public ArenaBackend {
public:
	/** `arenaTickCount` is the owning Arena's tickCount; it must outlive the backend. **/
	V3Backend(GameMode mode, const ArenaConfig& config, float tickRate, const uint64_t& arenaTickCount);
	~V3Backend() override;

	V3Backend(const V3Backend&) = delete;
	V3Backend& operator=(const V3Backend&) = delete;

	EngineId Engine() const override { return EngineId::V3; }

	uint32_t AddCar(Team team, const CarConfig& config) override;
	void Step(int ticks, uint64_t startTick, const CarControls* controls, StepResult& out) override;

	CarState GetCarState(uint32_t idx) override;
	void SetCarState(uint32_t idx, const CarState& state) override;
	void ResetCarStepContact(uint32_t idx) override;

	BallState GetBallState() override;
	void SetBallState(const BallState& state) override;

	uint32_t NumPads() override;
	BoostPadConfig GetPadConfig(uint32_t idx) override;
	BoostPadStateRaw GetPadState(uint32_t idx) override;
	void GetPadTimers(BoostPadTimers* out) override;
	void SetPadState(uint32_t idx, const BoostPadStateRaw& state) override;

	void ResetToRandomKickoff(int seed) override;
	bool IsBallScored() override;

	DropshotTilesState GetTiles() override;
	void SetTiles(const DropshotTilesState& tiles) override;

	int PredictBall(const BallState& start, int numTicks, BallState* out) override;

private:
	/** The CarState fields v3 has no native state for, kept per car between steps and setters. **/
	struct CarExtras {
		BallHitInfo ballHitInfo;
		bool wasOnGroundDuringStep = true;
		bool groundContactGraceUsed = false;
		int jumpResetBlockTicks = 0;
		uint64_t ticksSinceSet = 0;
	};

	/** Re-reads every car, the ball, the pads and the tiles without simulating; controls stay as they are. **/
	void RefreshSnapshot();

	/** Fills `out` to point at the cache, so a snapshot call writes straight into it. **/
	void PrepareStepOutput(RsfStepOutput& out, uint32_t bumpsCap);

	RsfArena* _arena = nullptr;
	RsfBallPredictor* _predictor = nullptr;

	GameMode _gameMode;
	bool _noBallRot;
	const uint64_t& _arenaTickCount;

	/** rsf_arena_tick_count, tracked here so reading it never calls into Rust. **/
	uint64_t _engineTick = 0;

	/** Clock of PredictBall's input and output damage ticks; advances by every predicted tick. **/
	uint64_t _predictTick = 0;

	std::vector<RsfCarControls> _controls;
	std::vector<RsfCarState> _cars;
	std::vector<RsfStepCarResult> _carResults;
	std::vector<CarExtras> _carExtras;
	RsfBallState _ball = {};
	uint64_t _ballTicksSinceSet = 0;
	std::vector<RsfPadConfig> _padConfigs;
	float _padCooldownBig = 0;
	float _padCooldownSmall = 0;
	std::vector<RsfPadState> _pads;
	std::vector<uint8_t> _tiles;
	std::vector<RsfBump> _bumps;
	bool _isBallScored = false;

	/** A ball set can move the ball into or out of a goal, which only the next step's snapshot would report. **/
	bool _isBallScoredStale = false;

	/** The tiles the predictor last received, so PredictBall only resends them after a change. **/
	std::vector<uint8_t> _predictorTiles;
	std::vector<RsfBallState> _predictScratch;
};

}
