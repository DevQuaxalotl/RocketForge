#include "ArenaBackend.h"
#include "V2Backend.h"

#include <EngineConfig.h>

#if RF_ENGINE_V3_AVAILABLE
#include "V3Backend.h"
#endif

namespace RocketSim::EngineBackend {

std::unique_ptr<ArenaBackend> CreateBackend(
	EngineId engine,
	GameMode mode,
	const ArenaConfig& config,
	float tickRate,
	const uint64_t& arenaTickCount) {
	RequireEngine(engine);

	switch (engine) {
	case EngineId::V2:
		return std::make_unique<V2Backend>(mode, config, tickRate);
#if RF_ENGINE_V3_AVAILABLE
	case EngineId::V3:
		return std::make_unique<V3Backend>(mode, config, tickRate, arenaTickCount);
#endif
	default:
		throw EngineUnavailableError(engine, "No arena backend is compiled for " + GetEngineInfo(engine).displayName + ".");
	}
}

}
