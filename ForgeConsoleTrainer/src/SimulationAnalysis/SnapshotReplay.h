#pragma once

#include "SimulationAnalysisTypes.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace RocketForge::SimulationAnalysis
{
	/** Serializes the observation GameState, including its arena's engine, into the trace snapshot column. Throws for a state with no arena. **/
	std::string SerializeGameStateSnapshot(const RLGC::GameState& state);
	/**
	 * Parses one trace snapshot and writes the engine it was recorded on to `engineOut`.
	 * A snapshot that records no engine uses `fallbackEngine`; with none known, parsing throws.
	 */
	RLGC::GameState ParseGameStateSnapshot(
		const std::string& snapshot,
		std::optional<RocketSim::EngineId> fallbackEngine,
		RocketSim::EngineId& engineOut);
	/**
	 * Loads a RocketSim replay seed from one trace CSV snapshot row.
	 * A snapshot that records no engine replays on `fallbackEngine`; with none known, loading throws.
	 */
	SnapshotReplaySeed LoadSnapshotReplaySeed(
		const std::filesystem::path& csvPath,
		int snapshotTick,
		int snapshotBotIndex,
		int64_t snapshotCarId,
		std::optional<RocketSim::EngineId> fallbackEngine);
	/** Loads per-tick controls from an RLBot trace CSV for deterministic physics replay. **/
	ControlReplayTrace LoadControlReplayTrace(
		const std::filesystem::path& csvPath,
		uint32_t tracedCarId,
		int startTick,
		int maxTicks);
	/** Builds an arena from a replay snapshot and returns the traced car by output pointer. **/
	std::unique_ptr<RocketSim::Arena> BuildArenaFromSnapshot(
		const SnapshotReplaySeed& seed,
		const RocketSim::CarConfig& carConfig,
		RocketSim::Car*& tracedCarOut);
}