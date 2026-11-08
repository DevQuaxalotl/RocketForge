#pragma once
#include "MathTypes.h"

namespace RocketSim {

struct LinearPieceCurve {
	std::map<float, float> valueMappings;

	float GetOutput(float input, float defaultOutput = 1) const;
};

namespace Math {
	btVector3 RoundVec(btVector3 vec, float precision);

	// NOTE: min is inclusive, max is exclusive
	// Seed will be used if not -1
	int RandInt(int min, int max, int seed = -1);

	float RandFloat(float min = 0, float max = 1);

	btVector3 RandVec(const btVector3& min, const btVector3& max);

	std::default_random_engine& GetRandEngine();

	float WrapNormalizeFloat(float val, float minmax);

	// Simulates aggressive UE3 rotator rounding when converting from a UE3 rotator to vector
	Angle RoundAngleUE3(Angle ang);
}

}