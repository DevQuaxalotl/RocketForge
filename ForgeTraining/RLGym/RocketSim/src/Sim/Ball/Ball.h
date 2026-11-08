#pragma once
#include "../../BaseInc.h"
#include <RocketSim/BallState.h>
#include <RocketSim/MutatorConfig.h>

#include "../../../libsrc/bullet3-3.24/BulletDynamics/Dynamics/btRigidBody.h"
#include "../../../libsrc/bullet3-3.24/BulletCollision/CollisionShapes/btSphereShape.h"
#include "../Arena/DropshotTiles/DropshotTiles.h"

class btDynamicsWorld;

RS_NS_START

class Ball {
public:

	BallState _internalState;
	BallState GetState();
	void SetState(const BallState& state);

	btRigidBody _rigidBody;
	btCollisionShape* _collisionShape;

	// For construction by Arena
	static Ball* _AllocBall() { return new Ball(); }

	// For removal by Arena
	static void _DestroyBall(Ball* ball) { delete ball; }

	void _BulletSetup(GameMode gameMode, btDynamicsWorld* bulletWorld, const MutatorConfig& mutatorConfig, bool noRot);

	bool _groundStickApplied = false;
	Vec _velocityImpulseCache = { 0,0,0 };
	void _FinishPhysicsTick(const MutatorConfig& mutatorConfig);

	bool IsSphere() const;

	// Returns radius in BulletPhysics units
	float GetRadiusBullet() const;

	// Returns radius in Unreal Engine units (uu)
	float GetRadius() const {
		return GetRadiusBullet() * BT_TO_UU;
	}

	// Returns mass
	float GetMass() const;

	void _PreTickUpdate(GameMode gameMode, float tickTime);
	void _OnHit(
		class Car* car, Vec relPos,
		float& outFriction, float& outRestitution,
		GameMode gameMode, const MutatorConfig& mutatorConfig, uint64_t tickCount
	);
	void _OnWorldCollision(GameMode gameMode, Vec normal, float tickTime);
	// Returns true if the tiles state was modified
	bool _OnDropshotTileCollision(
		DropshotTilesState& tilesState, int tileTotalIndex, const btCollisionObject* tileObj, 
		uint64_t tickCount, float tickTime
	);
		
	Ball(const Ball& other) = delete;
	Ball& operator=(const Ball& other) = delete;

	~Ball() {
		delete _collisionShape;
	}

private:
	Ball() {}
};

RS_NS_END
