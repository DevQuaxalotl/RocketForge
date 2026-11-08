#include <RocketSim/Engine.h>

#include <EngineConfig.h>

#include <utility>

namespace RocketSim {

namespace {
	std::vector<EngineInfo>& MutableEngines() {
		const bool v3Available = RF_ENGINE_V3_AVAILABLE != 0;

		static std::vector<EngineInfo> engines = {
			{ EngineId::V2, "RocketSim v2", true, "vendored", "" },
			{ EngineId::V3, "RocketSim v3", v3Available, RF_ENGINE_V3_REVISION, v3Available ? "" : RF_ENGINE_V3_REASON },
		};
		return engines;
	}

	EngineInfo& FindEngine(EngineId id) {
		for (EngineInfo& info : MutableEngines())
			if (info.id == id)
				return info;

		throw std::invalid_argument("Unknown RocketSim engine id " + std::to_string(static_cast<int>(id)));
	}
}

EngineUnavailableError::EngineUnavailableError(EngineId engine, const std::string& reason)
	: std::runtime_error(reason), engine(engine) {}

const std::vector<EngineInfo>& Engines() {
	return MutableEngines();
}

const EngineInfo& GetEngineInfo(EngineId id) {
	return FindEngine(id);
}

EngineId EngineFromVersion(int version) {
	for (const EngineInfo& info : Engines())
		if (static_cast<int>(info.id) == version)
			return info.id;

	throw std::invalid_argument("Unknown RocketSim engine version " + std::to_string(version));
}

void RequireEngine(EngineId id) {
	const EngineInfo& info = GetEngineInfo(id);
	if (!info.available)
		throw EngineUnavailableError(id, info.unavailableReason);
}

Testing::ScopedEngineUnavailable::ScopedEngineUnavailable(EngineId id, std::string reason) : _id(id) {
	EngineInfo& info = FindEngine(id);
	_previousAvailable = info.available;
	_previousReason = std::move(info.unavailableReason);

	info.available = false;
	info.unavailableReason = std::move(reason);
}

Testing::ScopedEngineUnavailable::~ScopedEngineUnavailable() {
	EngineInfo& info = FindEngine(_id);
	info.available = _previousAvailable;
	info.unavailableReason = std::move(_previousReason);
}

}
