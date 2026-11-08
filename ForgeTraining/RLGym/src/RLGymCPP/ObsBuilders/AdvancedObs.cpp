#include "AdvancedObs.h"
#include <RLGymCPP/Gamestates/StateUtil.h>

void RLGC::AdvancedObs::AddPlayerToObs(FList& obs, const Player& player, bool inv, const PhysState& ball) {
	auto phys = InvertPhys(player, inv);

	obs += phys.pos * POS_COEF;
	obs += phys.rotMat.forward;
	obs += phys.rotMat.up;
	obs += phys.vel * VEL_COEF;
	obs += phys.angVel * ANG_VEL_COEF;
	obs += phys.rotMat.Dot(phys.angVel) * ANG_VEL_COEF; // Local ang vel

	// Local ball pos and vel
	obs += phys.rotMat.Dot(ball.pos - phys.pos) * POS_COEF;
	obs += phys.rotMat.Dot(ball.vel - phys.vel) * VEL_COEF;

	obs += player.boost / 100;
	obs += player.isOnGround;
	obs += player.HasFlipOrJump();
	obs += player.isDemoed;
	obs += player.hasJumped; // Allows detecting flip resets
}

RLGC::FList RLGC::AdvancedObs::BuildObs(const Player& player, const GameState& state) {
	const bool directGroups = false;
	FList obs;
	FillObs(player, state, obs, directGroups);

	return obs;
}

void RLGC::AdvancedObs::BuildObsInto(const Player& player, const GameState& state, FList& out) {
	// A derived builder may override BuildObs or have stateful AddPlayerToObs hooks.
	if (typeid(*this) != typeid(AdvancedObs)) {
		ObsBuilder::BuildObsInto(player, state, out);
		return;
	}

	const bool directGroups = true;
	FillObs(player, state, out, directGroups);
}

void RLGC::AdvancedObs::AddSharedObs(FList& obs, const Player& player, const GameState& state, bool inv, const PhysState& ball) {
	auto& pads = state.GetBoostPads(inv);
	auto& padTimers = state.GetBoostPadTimers(inv);

	obs += ball.pos * POS_COEF;
	obs += ball.vel * VEL_COEF;
	obs += ball.angVel * ANG_VEL_COEF;

	for (int i = 0; i < player.prevAction.ELEM_AMOUNT; i++)
		obs += player.prevAction[i];

	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
		// A clever trick that blends the boost pads using their timers
		if (pads[i]) {
			obs += 1.f; // Pad is already available
		} else {
			obs += 1.f / (1.f + padTimers[i]); // Approaches 1 as the pad becomes available
		}
	}
}

void RLGC::AdvancedObs::FillObs(const Player& player, const GameState& state, FList& obs, bool directGroups) {
	obs.clear();

	bool inv = player.team == Team::ORANGE;
	auto ball = InvertPhys(state.ball, inv);

	AddSharedObs(obs, player, state, inv, ball);
	AddPlayerToObs(obs, player, inv, ball);

	if (directGroups) {
		// Stock player encoding is stateless: append each team directly in the original row order.
		for (bool sameTeam : { true, false }) {
			for (const auto& otherPlayer : state.players) {
				if (otherPlayer.carId != player.carId && (otherPlayer.team == player.team) == sameTeam)
					AddPlayerToObs(obs, otherPlayer, inv, ball);
			}
		}

		return;
	}

	// Preserve hook order and temporary-vector contents for custom subclasses using BuildObs.
	FList teammates = {}, opponents = {};

	for (auto& otherPlayer : state.players) {
		if (otherPlayer.carId == player.carId)
			continue;

		AddPlayerToObs(
			(otherPlayer.team == player.team) ? teammates : opponents,
			otherPlayer, inv, ball
		);
	}

	obs += teammates;
	obs += opponents;
}