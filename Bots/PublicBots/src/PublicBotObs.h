#pragma once

#include <RLGymCPP/ObsBuilders/ObsBuilder.h>

#include <array>
#include <unordered_map>

/** Necto's attention observation, including published boost/demo timers. **/
class NectoObs final : public RLGC::ObsBuilder
{
public:
	void Reset(const RLGC::GameState& initialState) override;
	RLGC::FList BuildObs(const RLGC::Player& player, const RLGC::GameState& state) override;

private:
	std::unordered_map<uint32_t, float> demoTimers;
	std::array<float, 34> boostTimers = {};
	std::array<float, 34> observedBoostTimers = {};
	uint64_t timerTick = UINT64_MAX;

	void UpdateTimers(const RLGC::GameState& state);
};

/** Nexto's player-relative attention observation. **/
class NextoObs final : public RLGC::ObsBuilder
{
public:
	RLGC::FList BuildObs(const RLGC::Player& player, const RLGC::GameState& state) override;
};

/** Element's paired-opponent 1v1 observation. **/
class ElementObs final : public RLGC::ObsBuilder
{
public:
	RLGC::FList BuildObs(const RLGC::Player& player, const RLGC::GameState& state) override;

private:
	static void AddPlayer(
		RLGC::FList& out,
		const RLGC::Player& player,
		const PhysState& ball,
		bool inverted,
		const PhysState* relativeTo);
};
