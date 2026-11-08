#include <RocketSim/CarState.h>

namespace RocketSim {

bool CarState::HasFlipOrJump() const {
	return 
		isOnGround || 
		(!hasFlipped && !hasDoubleJumped && airTimeSinceJump < RLConst::DOUBLEJUMP_MAX_DELAY);
}

bool CarState::HasFlipReset() const {
	return !isOnGround && HasFlipOrJump() && !hasJumped;
}

bool CarState::GotFlipReset() const {
	return !isOnGround && !hasJumped;
}

void CarState::Serialize(DataStreamOut& out) const {
	ballHitInfo.Serialize(out);

	out.WriteMultiple(
		CARSTATE_SERIALIZATION_FIELDS
	);
}

void CarState::Deserialize(DataStreamIn& in) {

	ballHitInfo.Deserialize(in);

	in.ReadMultiple(
		CARSTATE_SERIALIZATION_FIELDS
	);
	wasOnGroundDuringStep = isOnGround;
}

}
