#include "DefaultObs.h"
#include "../Gamestates/StateUtil.h"

void RLGC::DefaultObs::AddPlayerToObs(FList& obs, const Player& player, bool inv) {
	auto phys = InvertPhys(player, inv);

	obs += phys.pos * posCoef;
	obs += phys.rotMat.forward;
	obs += phys.rotMat.up;
	obs += phys.vel * velCoef;
	obs += phys.angVel * angVelCoef;

	obs += player.boost / 100;
	obs += player.isOnGround;
	obs += player.HasFlipOrJump();
	obs += player.isDemoed;
}

RLGC::FList RLGC::DefaultObs::BuildObs(const Player& player, const GameState& state) {
	const bool directGroups = false;
	FList result;
	FillObs(player, state, result, directGroups);

	return result;
}

void RLGC::DefaultObs::BuildObsInto(const Player& player, const GameState& state, FList& out) {
	// A derived builder may override BuildObs or have stateful AddPlayerToObs hooks.
	if (typeid(*this) != typeid(DefaultObs)) {
		ObsBuilder::BuildObsInto(player, state, out);
		return;
	}

	const bool directGroups = true;
	FillObs(player, state, out, directGroups);
}

void RLGC::DefaultObs::FillObs(const Player& player, const GameState& state, FList& result, bool directGroups) {
	result.clear();

	bool inv = player.team == Team::ORANGE;

	auto ball = InvertPhys(state.ball, inv);
	auto& pads = state.GetBoostPads(inv);

	result += ball.pos * posCoef;
	result += ball.vel * velCoef;
	result += ball.angVel * angVelCoef;

	for (int i = 0; i < player.prevAction.ELEM_AMOUNT; i++)
		result += player.prevAction[i];

	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++)
		result += (float)pads[i];

	AddPlayerToObs(result, player, inv);

	if (directGroups) {
		// Stock player encoding is stateless: append each team directly in the original row order.
		for (bool sameTeam : { true, false }) {
			for (const auto& otherPlayer : state.players) {
				if (otherPlayer.carId != player.carId && (otherPlayer.team == player.team) == sameTeam)
					AddPlayerToObs(result, otherPlayer, inv);
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
			otherPlayer,
			inv
		);
	}

	result += teammates;
	result += opponents;
}