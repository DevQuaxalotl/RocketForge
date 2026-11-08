#include <RocketSim/PhysState.h>

namespace RocketSim {

PhysState PhysState::GetInvertedY() const {
	constexpr Vec INVERT_SCALE = Vec(-1, -1, 1);

	PhysState inverted = *this;
	inverted.pos *= INVERT_SCALE;
	for (int i = 0; i < 3; i++)
		inverted.rotMat[i] *= INVERT_SCALE;
	inverted.vel *= INVERT_SCALE;
	inverted.angVel *= INVERT_SCALE;

	return inverted;
}

}