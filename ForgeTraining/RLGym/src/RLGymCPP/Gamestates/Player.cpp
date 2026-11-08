#include "Player.h"

namespace RLGC {
	void Player::ResetBeforeStep() {
		this->eventState = {};
	}

	void Player::UpdateFromCar(Car* car, uint64_t tickCount, int tickSkip, const Action& prevAction, Player* prev) {

		this->prev = prev;
		if (prev)
			prev->prev = NULL;

		carId = car->id;
		team = car->team;
		*(CarState*)this = car->GetState();

		if (ballHitInfo.isValid) {
			ballTouchedStep = ballHitInfo.tickCountWhenHit >= (tickCount - tickSkip);
			ballTouchedTick = ballHitInfo.tickCountWhenHit == (tickCount - 1);
		} else {
			ballTouchedStep = ballTouchedTick = false;
		}

		this->prevAction = prevAction;

		// ---- RLBot parity fields ----

		// airState mapping
		// RLBot states: OnGround, Jumping, DoubleJumping, Dodging, InAir
		if (obsOnGround)           airState = 0;
		else if (isJumping)        airState = 1;
		else if (hasDoubleJumped)  airState = 2;
		else if (isFlipping)       airState = 3;  // flip == dodge
		else                       airState = 4;

		// RLBot has_dodged means "used dodge" (not "currently dodging")
		hasDodged = hasFlipped || isFlipping;

		const bool flipAvailable =
			hasJumped &&              // <- key: require first jump used
			!hasDoubleJumped &&
			!hasDodged &&
			!obsOnGround;

		dodgeTimeout = flipAvailable
			? std::max(0.f, RLConst::DOUBLEJUMP_MAX_DELAY - airTimeSinceJump)
			: 0.f;

		// demolished_timeout
		demolishedTimeout = isDemoed ? demoRespawnTimer : 0.f;
	}
}