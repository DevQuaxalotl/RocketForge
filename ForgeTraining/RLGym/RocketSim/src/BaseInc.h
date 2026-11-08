#pragma once

// Header file for basic includes

#include "Framework.h"
#include "BulletLink.h"
#include <RocketSim/MathTypes.h>

// Declared up front so elaborated references such as `class Car*` stay in v2 when the shared API is already visible.
RS_NS_START
	class Arena;
	class Car;
	class Ball;
	class BoostPad;
RS_NS_END
