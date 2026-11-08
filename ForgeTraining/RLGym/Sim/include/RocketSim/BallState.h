#pragma once
#include "PhysState.h"
#include "RLConst.h"
#include "DataStream.h"

namespace RocketSim {

struct BallState : public PhysState {
	// Incremented every update, reset when SetState() is called
	// Used for telling if a stateset occured
	// Not serialized
	uint64_t tickCountSinceUpdate = 0;

	struct HeatseekerInfo {
		// Which net the ball should seek towards
		// When 0, no net
		float yTargetDir = 0;

		float curTargetSpeed = RLConst::Heatseeker::INITIAL_TARGET_SPEED;
		float timeSinceHit = 0;
	};
	HeatseekerInfo hsInfo;

	struct DropshotInfo {
		// Charge level number, which controls the radius of damage when hitting tiles
		// 1 = damages r=1 -> 1 tile
		// 2 = damages r=2 -> 7 tiles
		// 3 = damages r=3 -> 19 tiles
		int chargeLevel = 1;

		float accumulatedHitForce = 0; // Resets when a tile is damaged
		float yTargetDir = 0; // Which side of the field the ball can damage (0=none, -1=blue, 1=orange)
		
		bool hasDamaged = false;
		uint64_t lastDamageTick; // Only valid if hasDamaged
	};
	DropshotInfo dsInfo;

	BallState() : PhysState() {
		pos.z = RLConst::BALL_REST_Z;
	}

	bool Matches(const BallState& other, float marginPos = 0.8, float marginVel = 0.4, float marginAngVel = 0.02) const;

	void Serialize(DataStreamOut& out);
	void Deserialize(DataStreamIn& in);
};

#define BALLSTATE_SERIALIZATION_FIELDS \
pos, rotMat, vel, angVel, \
hsInfo.yTargetDir, hsInfo.curTargetSpeed, hsInfo.timeSinceHit, \
dsInfo.chargeLevel, dsInfo.accumulatedHitForce, dsInfo.yTargetDir, dsInfo.hasDamaged, dsInfo.lastDamageTick \

}
