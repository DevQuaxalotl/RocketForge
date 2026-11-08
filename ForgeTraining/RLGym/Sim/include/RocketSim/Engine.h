#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace RocketSim {

/** Physics engine behind an arena. The values are the integers saved as engine versions. */
enum class EngineId : int {
	V2 = 2,
	V3 = 3
};

struct EngineInfo {
	EngineId id;

	/** "RocketSim v2" / "RocketSim v3". */
	std::string displayName;

	bool available;

	/** "vendored" for v2; the submodule commit or crate version for v3. */
	std::string revision;

	/** Empty when available; otherwise a user-facing message that names the fix. */
	std::string unavailableReason;
};

/** Thrown when work needs an engine this build doesn't have. what() is the engine's unavailableReason. */
class EngineUnavailableError : public std::runtime_error {
public:
	EngineUnavailableError(EngineId engine, const std::string& reason);

	EngineId engine;
};

/** Every engine RocketForge knows, whether or not this build has it. */
const std::vector<EngineInfo>& Engines();

const EngineInfo& GetEngineInfo(EngineId id);

/** Maps a saved engine version to its id; throws std::invalid_argument for unknown versions. */
EngineId EngineFromVersion(int version);

/** Throws EngineUnavailableError when `id` isn't available in this build. */
void RequireEngine(EngineId id);

namespace Testing {
	/** Test-only: makes `id` report unavailable with `reason` for the guard's lifetime. Lets error paths run on a box that has every engine. **/
	class ScopedEngineUnavailable {
	public:
		ScopedEngineUnavailable(EngineId id, std::string reason);
		~ScopedEngineUnavailable();

		ScopedEngineUnavailable(const ScopedEngineUnavailable&) = delete;
		ScopedEngineUnavailable& operator=(const ScopedEngineUnavailable&) = delete;

	private:
		EngineId _id;
		bool _previousAvailable;
		std::string _previousReason;
	};
}

}
