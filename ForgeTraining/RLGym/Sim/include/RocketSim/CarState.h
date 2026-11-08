#pragma once
#include "PhysState.h"
#include "CarControls.h"
#include "BallHitInfo.h"
#include "RLConst.h"
#include "DataStream.h"

namespace RocketSim {

struct CarState : public PhysState {

	// Incremented every update, reset when SetState() is called
	// Used for telling if a stateset occured
	// Not serialized
	uint64_t tickCountSinceUpdate = 0;

	// True if 3 or more wheels have contact, with one-tick RL air-state grace after contact loss
	bool isOnGround = true;

	// True if the grounded air-state was active at any point in the current step
	bool wasOnGroundDuringStep = true;

	/** True while the current grounded state is the one-tick post-contact grace frame. **/
	bool groundContactGraceUsed = false;

	// False as soon as a jump starts to match RLBot behavior
	bool obsOnGround = true;

	// Whether each of the 4 wheels have contact
	// First two are front
	// If your car has 3 wheels, the 4th bool will always be false
	bool wheelsWithContact[4] = {}; 

	// Whether we jumped to get into the air
	// Can be false while airborne, if we left the ground with a flip reset
	bool hasJumped = false;

	// True if we have double jumped and are still in the air
	// NOTE: Flips DO NOT COUNT as double jumps! This is not RLBot.
	bool hasDoubleJumped = false;

	// True if we are in the air, and (have flipped or are currently flipping)
	bool hasFlipped = false;

	// Relative torque direction of the flip
	// Forward flip will have positive Y
	Vec flipRelTorque = { 0, 0, 0 };

	float jumpTime = 0; // When currently jumping, the time since we started jumping, else 0
	float flipTime = 0; // When currently flipping, the time since we started flipping, else 0

	// True during a flip (not an auto-flip, and not after a flip)
	bool isFlipping = false;

	// True during a jump
	bool isJumping = false;

	// Total time spent in the air
	float airTime = 0;

	// Time spent in the air once !isJumping
	// If we never jumped, it is 0
	float airTimeSinceJump = 0;

	// Goes from 0 to 100
	float boost = RLConst::BOOST_SPAWN_AMOUNT;

	// Used for recharge boost, counts up from 0 on spawn (in seconds)
	float timeSinceBoosted = 0.f;

	// True if we boosted that tick
	// There exists a minimum boosting time, thus why we must track boosting time
	bool isBoosting = false;
	float boostingTime = 0;

	bool isSupersonic = false;

	// Time spent supersonic, for checking with the supersonic maintain time (see RLConst.h)
	float supersonicTime = 0;

	// This is a state variable due to the rise/fall rate of handbrake inputs (see RLConst.h)
	float handbrakeVal = 0;

	bool isAutoFlipping = false;
	float autoFlipTimer = 0; // Counts down when auto-flipping
	float autoFlipTorqueScale = 0;

	struct {
		bool hasContact = false;
		Vec contactNormal;
	} worldContact;

	struct {
		uint32_t otherCarID = 0;
		float cooldownTimer = 0;
	} carContact;

	bool isDemoed = false;
	float demoRespawnTimer = 0;

	BallHitInfo ballHitInfo = BallHitInfo();

	// Controls from last tick, set to this->controls after simulation
	CarControls lastControls = CarControls();

	/** Blocks jump and dodge edges through the packet after a grounded hasJumped reset. **/
	int jumpResetBlockTicks = 0;

	CarState() : PhysState() {
		pos.z = RLConst::CAR_SPAWN_REST_Z;
	}

	// Returns true if the car is currently able to jump, double-jump, or flip
	bool HasFlipOrJump() const;

	// Returns true if we currently have a flip, and got it from a flip reset (instead of a jump)
	bool HasFlipReset() const;

	// Returns true if we are in the air, and became airborne from a flip reset
	// This returns true regardless of whether or not we still have the flip
	bool GotFlipReset() const;

	void Serialize(DataStreamOut& out) const;
	void Deserialize(DataStreamIn& in);
};

#define CARSTATE_SERIALIZATION_FIELDS \
pos, rotMat, vel, angVel, isOnGround, hasJumped, hasDoubleJumped, hasFlipped, \
flipRelTorque, jumpTime, isFlipping, flipTime, isJumping, airTimeSinceJump, \
boost, isBoosting, boostingTime, timeSinceBoosted, supersonicTime, handbrakeVal, isAutoFlipping, \
autoFlipTimer, autoFlipTorqueScale, isDemoed, demoRespawnTimer, groundContactGraceUsed, lastControls, jumpResetBlockTicks, \
worldContact.hasContact, worldContact.contactNormal, \
carContact.otherCarID, carContact.cooldownTimer

enum class Team : byte {
	BLUE = 0,
	ORANGE = 1
};

#define RS_OPPOSITE_TEAM(team) ((team) == Team::BLUE ? Team::ORANGE : Team::BLUE)
#define RS_TEAM_FROM_Y(y) ((y) < 0 ? Team::BLUE : Team::ORANGE)

}
