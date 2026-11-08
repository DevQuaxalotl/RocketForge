#pragma once

// BulletLink.h: Includes most basic bullet headers, and defines some convenient typedefs/wrappers/etc.

// Bullet v3 (https://github.com/bulletphysics/bullet3)
#include "../libsrc/bullet3-3.24/LinearMath/btVector3.h"
#include "../libsrc/bullet3-3.24/LinearMath/btMatrix3x3.h"

#include <ostream>

// BT_TO_UU and UU_TO_BT
#include <RocketSim/MathTypes.h>

// Enum values for Bullet btCollisionObject userinfo usage
enum : int {
	BT_USERINFO_NONE,

	BT_USERINFO_TYPE_CAR,
	BT_USERINFO_TYPE_BALL,
	BT_USERINFO_TYPE_DROPSHOT_TILE,
};
