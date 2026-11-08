#include <RocketSim/BallHitInfo.h>

namespace RocketSim {

void BallHitInfo::Serialize(DataStreamOut& out) const {
	out.Write<bool>(isValid);

	if (isValid)
		out.WriteMultiple(BALLHITINFO_SERIALIZATION_FIELDS);
}

void BallHitInfo::Deserialize(DataStreamIn& in) {
	in.Read<bool>(isValid);

	if (isValid)
		in.ReadMultiple(BALLHITINFO_SERIALIZATION_FIELDS);
}

}