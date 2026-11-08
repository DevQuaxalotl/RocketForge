#pragma once
#include "StateSetter.h"

namespace RLGC {
	class RandomState : public StateSetter {
	public:
		bool
			randBallSpeed, randCarSpeed, carsOnGround;

		RandomState(bool randBallSpeed, bool randCarSpeed, bool carsOnGround) :
			randBallSpeed(randBallSpeed), randCarSpeed(randCarSpeed), carsOnGround(carsOnGround) {
			setterType = StateSetterType::Random;
		}

		virtual void ResetArena(Arena* arena);
	};
}