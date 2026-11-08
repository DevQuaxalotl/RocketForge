#include "AdvancedObsPadded.h"
#include <RLGymCPP/Gamestates/StateUtil.h>

RLGC::AdvancedObsPadded::AdvancedObsPadded(int maxPlayers) : maxPlayers(maxPlayers) {
	if (maxPlayers < 1)
		RG_ERR_CLOSE("AdvancedObsPadded: maxPlayers must be at least 1, got " << maxPlayers);
}

RLGC::FList RLGC::AdvancedObsPadded::BuildObs(const Player& player, const GameState& state) {
	FList obs;
	FillPaddedObs(player, state, obs);
	return obs;
}

void RLGC::AdvancedObsPadded::BuildObsInto(const Player& player, const GameState& state, FList& out) {
	// A derived builder may override BuildObs, which this direct path would skip.
	if (typeid(*this) != typeid(AdvancedObsPadded)) {
		ObsBuilder::BuildObsInto(player, state, out);
		return;
	}

	FillPaddedObs(player, state, out);
}

void RLGC::AdvancedObsPadded::FillPaddedObs(const Player& player, const GameState& state, FList& obs) {
	obs.clear();

	bool inv = player.team == Team::ORANGE;
	auto ball = InvertPhys(state.ball, inv);

	AddSharedObs(obs, player, state, inv, ball);

	const size_t selfStart = obs.size();
	AddPlayerToObs(obs, player, inv, ball);
	const size_t playerObsSize = obs.size() - selfStart;

	for (bool sameTeam : { true, false }) {
		const int slotCount = sameTeam ? maxPlayers - 1 : maxPlayers;
		int filledSlots = 0;

		for (const Player& otherPlayer : state.players) {
			if (otherPlayer.carId == player.carId || (otherPlayer.team == player.team) != sameTeam)
				continue;

			if (filledSlots == slotCount)
				RG_ERR_CLOSE("AdvancedObsPadded: Too many " << (sameTeam ? "teammates" : "opponents")
					<< " for Obs, maximum is " << slotCount);

			AddPlayerToObs(obs, otherPlayer, inv, ball);
			filledSlots++;
		}

		obs.resize(obs.size() + (slotCount - filledSlots) * playerObsSize, 0.f);
	}
}
