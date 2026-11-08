#pragma once

#include "ArenaBackend.h"

namespace RocketSimV2 {
	class Arena;
	class Car;
}

namespace RocketSim::EngineBackend {

void InitV2(const std::filesystem::path& collisionMeshesFolder, bool silent);
RocketSimStage GetV2Stage();
bool V2HasCollisionMeshes(GameMode gameMode);

/** Forwards to a vendored v2 arena; v2 reads and writes the shared value types directly, so nothing is converted. **/
class V2Backend : public ArenaBackend {
public:
	V2Backend(GameMode mode, const ArenaConfig& config, float tickRate);
	~V2Backend() override;

	V2Backend(const V2Backend&) = delete;
	V2Backend& operator=(const V2Backend&) = delete;

	EngineId Engine() const override { return EngineId::V2; }

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
	static void OnBump(RocketSimV2::Arena* arena, RocketSimV2::Car* bumper, RocketSimV2::Car* victim, bool isDemo, void* userInfo);

	uint32_t IndexOf(const RocketSimV2::Car* car) const;
	uint32_t IndexOfId(uint32_t carId) const;

	RocketSimV2::Arena* _arena = nullptr;

	/** In add order, so the vector index is the backend car index; v2 numbers its cars from 1, so a v2 car id is that index + 1. **/
	std::vector<RocketSimV2::Car*> _cars;

	/** Set only while Step runs, since v2 reports bumps mid-tick. **/
	StepResult* _stepResult = nullptr;
};

}
