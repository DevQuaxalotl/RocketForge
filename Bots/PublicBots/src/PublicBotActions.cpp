#include "PublicBotActions.h"

#include <RLGymCPP/Gamestates/StateUtil.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace RLGC;
using namespace RocketSim;

namespace
{
	constexpr float PI = 3.14159265358979323846f;
	constexpr int RECORDED_KICKOFF_TICKS = 168;

	template<size_t N>
	std::array<int, N> DecodeMixedRadix(int value, const std::array<int, N>& radices)
	{
		std::array<int, N> decoded = {};
		for (int i = (int)N - 1; i >= 0; i--) {
			decoded[i] = value % radices[i];
			value /= radices[i];
		}
		return decoded;
	}

	bool IsKickoffBallCentered(const GameState& state)
	{
		// GameState has no RLBot kickoff-pause flag, so the centered x/y position
		// distinguishes a kickoff from an ordinary midfield crossing.
		return state.gameMode == GameMode::SOCCAR
			&& state.ball.pos.x == 0.f
			&& state.ball.pos.y == 0.f;
	}

	float PlanarDistanceToBall(const Player& player, const GameState& state)
	{
		const float dx = player.pos.x - state.ball.pos.x;
		const float dy = player.pos.y - state.ball.pos.y;
		return std::sqrt(dx * dx + dy * dy);
	}

	bool IsRecordedKickoffTaker(const Player& player, const GameState& state)
	{
		float minDistance = std::numeric_limits<float>::max();
		for (const Player& candidate : state.players)
			minDistance = std::min(minDistance, PlanarDistanceToBall(candidate, state));

		const float playerDistance = PlanarDistanceToBall(player, state);
		if (std::abs(playerDistance - minDistance) > 10.f)
			return false;

		for (const Player& candidate : state.players) {
			if (candidate.carId == player.carId || candidate.team != player.team)
				continue;
			if (std::abs(PlanarDistanceToBall(candidate, state) - playerDistance) > 10.f)
				continue;

			const bool candidateIsLeft = player.team == Team::BLUE
				? candidate.pos.x < player.pos.x
				: candidate.pos.x > player.pos.x;
			// NOTE: This preserves the published wrappers' behavior exactly,
			// including their counterintuitive "Left goes" comment.
			if (!candidateIsLeft)
				return false;
		}
		return true;
	}

	Action RecordedKickoffActionAt(int tick)
	{
		if (tick < 44)
			return { 1, 0, 0, 0, 0, 0, 1, 0 };
		if (tick < 60)
			return { 1, -1, 0, 0, 0, 0, 1, 0 };
		if (tick < 68)
			return { 1, 0, 0, 0, 0, 1, 1, 0 };
		if (tick < 72)
			return { 1, 0, 0, 0, 0, 0, 1, 0 };
		if (tick < 76)
			return { 1, 0, -0.7f, 0.8f, 0, 1, 1, 0 };
		if (tick < 128)
			return { 1, 0, 1, 0, 0, 0, 1, 0 };
		return { 1, 0, 0.5f, 0, 1, 0, 0, 0 };
	}

	const Player* ClosestTeammateToBall(const Player& player, const GameState& state)
	{
		const Player* closest = nullptr;
		float closestDistance = std::numeric_limits<float>::max();
		for (const Player& candidate : state.players) {
			if (candidate.team != player.team)
				continue;
			const float distance = candidate.pos.Dist(state.ball.pos);
			if (distance < closestDistance) {
				closest = &candidate;
				closestDistance = distance;
			}
		}
		return closest;
	}
}

RecordedKickoffController::PlayerState& RecordedKickoffController::UpdateState(
	const Player& player,
	const GameState& state)
{
	PlayerState& kickoffState = playerStates[player.carId];
	if (!IsKickoffBallCentered(state)) {
		if (kickoffState.active)
			kickoffState = {};
		return kickoffState;
	}
	if (!kickoffState.active) {
		kickoffState.active = true;
		kickoffState.taker = IsRecordedKickoffTaker(player, state);
	}
	return kickoffState;
}

bool RecordedKickoffController::RequiresPerTickActionOverride(
	const Player& player,
	const GameState& state)
{
	PlayerState& kickoffState = UpdateState(player, state);
	return kickoffState.active && kickoffState.taker && kickoffState.tick < RECORDED_KICKOFF_TICKS;
}

Action RecordedKickoffController::OverrideRuntimeAction(
	const Action& policyAction,
	const Player& player,
	const GameState& state,
	int ticksElapsed)
{
	PlayerState& kickoffState = UpdateState(player, state);
	if (!kickoffState.active || !kickoffState.taker)
		return policyAction;

	if (kickoffState.hasOutput)
		kickoffState.tick += std::max(0, ticksElapsed);
	else
		kickoffState.hasOutput = true;

	if (kickoffState.tick < 0 || kickoffState.tick >= RECORDED_KICKOFF_TICKS)
		return policyAction;
	return RecordedKickoffActionAt(kickoffState.tick);
}

void RecordedKickoffController::Reset()
{
	playerStates.clear();
}

// https://github.com/RLBot/RLBotPack/blob/0033a98d4e060670f334f60f7f8dc5ee21ba93bb/RLBotPack/Necto/Necto/agent.py
Action NectoAction::ParseAction(int actionIdx, const Player&, const GameState&)
{
	if (actionIdx < 0 || actionIdx >= GetActionAmount())
		throw std::out_of_range("NectoAction: action index is out of range");

	const std::array<int, 5> values = DecodeMixedRadix<5>(actionIdx, { 3, 3, 2, 2, 2 });
	const float throttle = (float)values[0] - 1;
	const float steer = (float)values[1] - 1;
	const float jump = (float)values[2];
	const float boost = (float)values[3];
	const float handbrake = (float)values[4];
	return {
		throttle,
		steer,
		throttle,
		steer * (1 - handbrake),
		steer * handbrake,
		jump,
		boost,
		handbrake
	};
}

bool NectoAction::RequiresPerTickActionOverride(const Player& player, const GameState& state)
{
	return kickoff.RequiresPerTickActionOverride(player, state);
}

Action NectoAction::OverrideRuntimeAction(
	const Action& policyAction,
	const Player& player,
	const GameState& state,
	int ticksElapsed)
{
	return kickoff.OverrideRuntimeAction(policyAction, player, state, ticksElapsed);
}

// https://github.com/RLBot/RLBotPack/blob/0033a98d4e060670f334f60f7f8dc5ee21ba93bb/RLBotPack/Necto/Nexto/agent.py
NextoAction::NextoAction()
{
	constexpr float binaryValues[] = { 0, 1 };
	constexpr float fullValues[] = { -1, 0, 1 };

	for (float throttle : fullValues) {
		for (float steer : fullValues) {
			for (float boost : binaryValues) {
				for (float handbrake : binaryValues) {
					if (boost == 1 && throttle != 1)
						continue;
					actions.push_back({
						throttle != 0 ? throttle : boost,
						steer,
						0,
						steer,
						0,
						0,
						boost,
						handbrake
					});
				}
			}
		}
	}

	for (float pitch : fullValues) {
		for (float yaw : fullValues) {
			for (float roll : fullValues) {
				for (float jump : binaryValues) {
					for (float boost : binaryValues) {
						if (jump == 1 && yaw != 0)
							continue;
						if (pitch == 0 && roll == 0 && jump == 0)
							continue;
						const float handbrake = jump == 1 && (pitch != 0 || yaw != 0 || roll != 0);
						actions.push_back({
							boost,
							yaw,
							pitch,
							yaw,
							roll,
							jump,
							boost,
							handbrake
						});
					}
				}
			}
		}
	}

	if (actions.size() != 90)
		throw std::runtime_error("NextoAction: lookup table did not produce 90 actions");
}

Action NextoAction::ParseAction(int actionIdx, const Player&, const GameState&)
{
	if (actionIdx < 0 || actionIdx >= GetActionAmount())
		throw std::out_of_range("NextoAction: action index is out of range");
	return actions[actionIdx];
}

bool NextoAction::RequiresPerTickActionOverride(const Player& player, const GameState& state)
{
	return kickoff.RequiresPerTickActionOverride(player, state);
}

Action NextoAction::OverrideRuntimeAction(
	const Action& policyAction,
	const Player& player,
	const GameState& state,
	int ticksElapsed)
{
	return kickoff.OverrideRuntimeAction(policyAction, player, state, ticksElapsed);
}

// https://github.com/RLBot/RLBotPack/blob/0033a98d4e060670f334f60f7f8dc5ee21ba93bb/RLBotPack/Element/bot.py
Action ElementAction::ParseAction(int actionIdx, const Player&, const GameState&)
{
	if (actionIdx < 0 || actionIdx >= GetActionAmount())
		throw std::out_of_range("ElementAction: action index is out of range");

	std::array<int, 8> values = DecodeMixedRadix<8>(
		actionIdx,
		{ 3, 3, 3, 3, 3, 2, 2, 2 });
	for (int i = 0; i < 5; i++)
		values[i] -= 1;

	const float jump = (float)values[5];
	return {
		(float)values[0],
		(float)values[1],
		(float)values[2],
		jump > 0 ? 0.f : (float)values[3],
		(float)values[4],
		jump,
		(float)values[6],
		(float)values[7]
	};
}

// https://github.com/RLBot/RLBotPack/blob/0033a98d4e060670f334f60f7f8dc5ee21ba93bb/RLBotPack/Element/sequences/speedflip.py
ElementAction::PlayerState& ElementAction::UpdateState(const Player& player, const GameState& state)
{
	PlayerState& kickoffState = playerStates[player.carId];
	if (!IsKickoffBallCentered(state)) {
		if (kickoffState.active)
			kickoffState = {};
		return kickoffState;
	}
	if (kickoffState.active)
		return kickoffState;

	kickoffState.active = true;
	kickoffState.speedflipState = SpeedflipState::Align;
	kickoffState.initialCar = InvertPhys(player, player.team == Team::ORANGE);
	kickoffState.direction = kickoffState.initialCar.pos.x > 0 ? 1 : -1;
	kickoffState.initialAngle = PI / 16.f;
	kickoffState.driveDistance = 0;

	const float absX = std::abs(kickoffState.initialCar.pos.x);
	if (absX < 25.f) {
		kickoffState.yawStrength = 1.f;
		kickoffState.driveDistance = 290.f;
		kickoffState.initialAngle = PI / 22.f;
	} else if (absX < 500.f) {
		kickoffState.yawStrength = 0.6f;
		kickoffState.driveDistance = 330.f;
		kickoffState.direction = -kickoffState.direction;
	} else {
		kickoffState.yawStrength = 0.4f;
		kickoffState.driveDistance = 240.f;
	}
	return kickoffState;
}

bool ElementAction::ShouldOverride(
	const Player& player,
	const GameState& state,
	const PlayerState& kickoffState)
{
	if (!kickoffState.active || kickoffState.speedflipState == SpeedflipState::Done)
		return false;

	const Player* closest = ClosestTeammateToBall(player, state);
	return closest
		&& closest->carId == player.carId
		&& player.pos.Dist(state.ball.pos) > 600.f;
}

bool ElementAction::RequiresPerTickActionOverride(const Player& player, const GameState& state)
{
	return ShouldOverride(player, state, UpdateState(player, state));
}

Action ElementAction::OverrideRuntimeAction(
	const Action& policyAction,
	const Player& player,
	const GameState& state,
	int ticksElapsed)
{
	PlayerState& kickoffState = UpdateState(player, state);
	if (kickoffState.active) {
		if (player.isOnGround)
			kickoffState.offGroundTicks = 0;
		else
			kickoffState.offGroundTicks += std::max(0, ticksElapsed);
	}
	if (!ShouldOverride(player, state, kickoffState))
		return policyAction;

	const PhysState car = InvertPhys(player, player.team == Team::ORANGE);
	// Match the six-tick on-ground grace in Element's published state decoder.
	const bool onGround = player.isOnGround || kickoffState.offGroundTicks <= 6;

	if (kickoffState.speedflipState == SpeedflipState::Align) {
		const float dot = std::clamp(
			car.rotMat.forward.Dot(kickoffState.initialCar.rotMat.forward),
			-1.f,
			1.f);
		if (std::acos(dot) < kickoffState.initialAngle)
			return { 1, (float)-kickoffState.direction, 0, 0, 0, 0, 1, 0 };
		kickoffState.speedflipState = SpeedflipState::Drive;
	}

	if (kickoffState.speedflipState == SpeedflipState::Drive) {
		if (car.pos.Dist(kickoffState.initialCar.pos) < kickoffState.driveDistance)
			return { 1, 0, 0, 0, 0, 0, 1, 0 };
		kickoffState.speedflipState = SpeedflipState::FirstJump;
	}

	if (kickoffState.speedflipState == SpeedflipState::FirstJump) {
		kickoffState.speedflipState = SpeedflipState::StartFlip;
		return { 1, 0, 0, 0, 0, 1, 1, 0 };
	}

	if (kickoffState.speedflipState == SpeedflipState::StartFlip) {
		if (onGround)
			return { 1, 0, 0, 0, (float)kickoffState.direction, 0, 1, 0 };
		kickoffState.speedflipState = SpeedflipState::CancelFlip;
		return { 1, 0, -1, 0, (float)kickoffState.direction, 1, 1, 0 };
	}

	if (kickoffState.speedflipState == SpeedflipState::CancelFlip) {
		if (!onGround) {
			const float boost = car.vel.Length() < 2295.f ? 1.f : 0.f;
			return {
				1,
				0,
				1,
				kickoffState.yawStrength * kickoffState.direction,
				(float)kickoffState.direction,
				0,
				boost,
				0
			};
		}
		kickoffState.speedflipState = SpeedflipState::Done;
		return { 1, 0, 0, 0, 0, 0, 1, 0 };
	}

	return policyAction;
}
