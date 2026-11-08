#pragma once
#include "../Framework.h"
#include "../BasicTypes/Action.h"

namespace RLGC {
	struct PlayerEventState {
		bool goal, save, assist, shot, shotPass, bump, bumped, demo, demoed;

		PlayerEventState() {
			memset(this, 0, sizeof(*this));
		}
	};

	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/gamestates/player_data.py
	struct Player : CarState {

		Player* prev = NULL;

		int index = -1; // Index in the gamestate players array
		uint32_t carId;
		Team team;

		PlayerEventState eventState = {};

		bool ballTouchedStep; // True if the player touched the ball during any of tick of the step
		bool ballTouchedTick; // True if the player is touching the ball on the final tick of the step

		Action prevAction = {};

		uint8_t airState = 0;            // 0 OnGround, 1 Jumping, 2 DoubleJumping, 3 Dodging, 4 InAir
		bool    hasDodged = false;       // RLBot: has_dodged
		float   dodgeTimeout = 0.f;      // RLBot: dodge_timeout (seconds remaining)
		float   demolishedTimeout = 0.f; // RLBot: demolished_timeout (seconds remaining)

		// Called before updating to reset the per-step state
		void ResetBeforeStep();

		void UpdateFromCar(Car* car, uint64_t tickCount, int tickSkip, const Action& prevAction, Player* prev);
	};
}