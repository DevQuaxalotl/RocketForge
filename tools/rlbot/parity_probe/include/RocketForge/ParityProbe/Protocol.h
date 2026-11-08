#pragma once

#include <RLGymCPP/BasicTypes/Action.h>
#include <RLGymCPP/Gamestates/GameState.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace RocketForge::ParityProbe
{
	inline constexpr const char* FIXTURE_SCHEMA = "rocketforge.parity.fixture.v1";
	inline constexpr const char* RESULT_SCHEMA = "rocketforge.parity.result.v1";

	/** Optional runtime evidence captured with a fixture frame. **/
	struct RuntimeFixture
	{
		std::optional<RLGC::Action> appliedAction;
	};

	/** One canonical game state and the deterministic inputs used to probe it. **/
	struct FrameFixture
	{
		nlohmann::json frameId;
		uint32_t targetCarId = 0;
		uint64_t obsRngSeed = 0;
		bool resetObsBuilder = false;
		nlohmann::json source;
		RLGC::GameState state;
		RuntimeFixture runtime;
	};

	/** Ordered frames share observation-builder state and previous-state links. **/
	struct SequenceFixture
	{
		nlohmann::json sequenceId;
		std::vector<FrameFixture> frames;
	};

	/** Versioned, implementation-neutral input for parity probes. **/
	struct FixtureDocument
	{
		nlohmann::json fixtureId;
		std::vector<SequenceFixture> sequences;
	};

	/** One parser-emitted control segment and its physics-tick duration. **/
	struct ActionPlanEntry
	{
		RLGC::Action action = {};
		int ticks = 0;
	};

	/** Canonical policy result produced by either a RocketForge or reference adapter. **/
	struct FrameResult
	{
		nlohmann::json fixtureId;
		nlohmann::json sequenceId;
		nlohmann::json frameId;
		uint32_t targetCarId = 0;
		uint64_t obsRngSeed = 0;
		std::vector<float> observation;
		std::vector<uint8_t> actionMask;
		std::vector<float> sharedFeatures;
		std::vector<float> rawLogits;
		std::vector<float> maskedLogits;
		int argmaxActionId = -1;
		float top1Margin = 0.f;
		RLGC::Action parsedAction = {};
		std::vector<ActionPlanEntry> actionPlan;
		int tickSkip = 0;
		int actionDelay = 0;
		int actionTicks = 0;
		RLGC::Action queuedAction = {};
		RLGC::Action appliedAction = {};
	};

	/** Reads and validates a structured canonical fixture document. **/
	FixtureDocument LoadFixtureDocument(const std::filesystem::path& path);

	/** Serializes a result using the stable JSONL interchange contract. **/
	nlohmann::json FrameResultToJson(const FrameResult& result, const std::string& producer);
}
