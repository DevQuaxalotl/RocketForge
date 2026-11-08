#pragma once

#include <RLGymCPP/ActionParsers/ActionParser.h>

#include <array>
#include <unordered_map>

/** Replays Necto and Nexto's published per-tick kickoff control sequence. **/
class RecordedKickoffController
{
public:
	bool RequiresPerTickActionOverride(const RLGC::Player& player, const RLGC::GameState& state);
	RLGC::Action OverrideRuntimeAction(
		const RLGC::Action& policyAction,
		const RLGC::Player& player,
		const RLGC::GameState& state,
		int ticksElapsed);
	void Reset();

private:
	struct PlayerState
	{
		bool active = false;
		bool taker = false;
		bool hasOutput = false;
		int tick = 0;
	};

	std::unordered_map<uint32_t, PlayerState> playerStates;

	PlayerState& UpdateState(const RLGC::Player& player, const RLGC::GameState& state);
};

/** Necto's factorized discrete action parser with its recorded kickoff. **/
class NectoAction final : public RLGC::ActionParser
{
public:
	RLGC::Action ParseAction(int actionIdx, const RLGC::Player& player, const RLGC::GameState& state) override;
	bool HasRuntimeActionOverride() const override { return true; }
	bool RequiresPerTickActionOverride(const RLGC::Player& player, const RLGC::GameState& state) override;
	RLGC::Action OverrideRuntimeAction(
		const RLGC::Action& policyAction,
		const RLGC::Player& player,
		const RLGC::GameState& state,
		int ticksElapsed) override;
	void ResetRuntimeActionOverride() override { kickoff.Reset(); }
	int GetActionAmount() override { return 72; }
	std::string GetPolicyCompatibilityKey() override { return "RLBotPack.NectoAction.v1"; }

private:
	RecordedKickoffController kickoff;
};

/** Nexto's 90-entry lookup-table parser with its recorded kickoff. **/
class NextoAction final : public RLGC::ActionParser
{
public:
	NextoAction();

	RLGC::Action ParseAction(int actionIdx, const RLGC::Player& player, const RLGC::GameState& state) override;
	bool HasRuntimeActionOverride() const override { return true; }
	bool RequiresPerTickActionOverride(const RLGC::Player& player, const RLGC::GameState& state) override;
	RLGC::Action OverrideRuntimeAction(
		const RLGC::Action& policyAction,
		const RLGC::Player& player,
		const RLGC::GameState& state,
		int ticksElapsed) override;
	void ResetRuntimeActionOverride() override { kickoff.Reset(); }
	int GetActionAmount() override { return (int)actions.size(); }
	std::string GetPolicyCompatibilityKey() override { return "RLBotPack.NextoAction.v1"; }

private:
	std::vector<RLGC::Action> actions;
	RecordedKickoffController kickoff;
};

/** Element's factorized action parser and stateful speed-flip kickoff. **/
class ElementAction final : public RLGC::ActionParser
{
public:
	RLGC::Action ParseAction(int actionIdx, const RLGC::Player& player, const RLGC::GameState& state) override;
	bool HasRuntimeActionOverride() const override { return true; }
	bool RequiresPerTickActionOverride(const RLGC::Player& player, const RLGC::GameState& state) override;
	RLGC::Action OverrideRuntimeAction(
		const RLGC::Action& policyAction,
		const RLGC::Player& player,
		const RLGC::GameState& state,
		int ticksElapsed) override;
	void ResetRuntimeActionOverride() override { playerStates.clear(); }
	int GetActionAmount() override { return 1944; }
	std::string GetPolicyCompatibilityKey() override { return "RLBotPack.ElementAction.v1"; }

private:
	enum class SpeedflipState
	{
		Align,
		Drive,
		FirstJump,
		StartFlip,
		CancelFlip,
		Done
	};

	struct PlayerState
	{
		bool active = false;
		int direction = 1;
		float initialAngle = 0;
		float driveDistance = 0;
		float yawStrength = 0;
		int offGroundTicks = 0;
		RocketSim::PhysState initialCar = {};
		SpeedflipState speedflipState = SpeedflipState::Align;
	};

	std::unordered_map<uint32_t, PlayerState> playerStates;

	PlayerState& UpdateState(const RLGC::Player& player, const RLGC::GameState& state);
	static bool ShouldOverride(
		const RLGC::Player& player,
		const RLGC::GameState& state,
		const PlayerState& kickoffState);
};
