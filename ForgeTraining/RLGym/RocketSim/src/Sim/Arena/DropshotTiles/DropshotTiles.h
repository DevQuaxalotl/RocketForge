#pragma once
#include "../../../BaseInc.h"
#include <RocketSim/DropshotTiles.h>
#include <RocketSim/RLConst.h>
#include "../../../../libsrc/bullet3-3.24/BulletDynamics/Dynamics/btRigidBody.h"

RS_NS_START

namespace DropshotTiles {
	void Init();

	Vec GetTilePos(int team, int index);
	std::vector<btCollisionShape*> MakeTileShapes();
	std::vector<int> GetNeighborIndices(int startIdx, int radius);
};

RS_NS_END