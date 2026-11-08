#include "ModelBenchmarkInternal.h"
#include "BenchmarkStartingStates.h"

namespace RocketForgeTraining::ModelBenchmarkInternal
{
	ArenaSnapshot CaptureSnapshot(Arena* arena) {
		return GGL::CapturePolicyMatchSnapshot(arena);
	}

	void ApplySnapshot(Arena* arena, const ArenaSnapshot& snapshot) {
		GGL::ApplyPolicyMatchSnapshot(arena, snapshot);
	}

	std::unique_ptr<Arena> MakeArena(int teamSize, const CarConfig& blueCarConfig, const CarConfig& orangeCarConfig, EngineId engine) {
		return GGL::MakePolicyMatchArena(teamSize, blueCarConfig, orangeCarConfig, engine);
	}

	ArenaSnapshot SampleKickoffState(int teamSize, const CarConfig& carConfig, EngineId engine) {
		return GGL::SamplePolicyMatchStartState(
			GGL::PolicyMatchStartType::Kickoff,
			teamSize,
			carConfig,
			engine);
	}

	ArenaSnapshot SampleKickoffState(
		int teamSize,
		const CarConfig& carConfig,
		uint64_t seed,
		EngineId engine,
		const std::filesystem::path& startingStateDirectory) {
		if (!startingStateDirectory.empty())
			return ReadBenchmarkStartingState(startingStateDirectory, teamSize, seed);

		return GGL::SamplePolicyMatchStartState(
			GGL::PolicyMatchStartType::Kickoff,
			teamSize,
			carConfig,
			seed,
			engine);
	}
}
