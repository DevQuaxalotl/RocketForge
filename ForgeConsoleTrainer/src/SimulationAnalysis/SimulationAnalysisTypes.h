#pragma once

#include <RLGymCPP/ActionParsers/ActionParser.h>
#include <RLGymCPP/BasicTypes/Lists.h>
#include <RLGymCPP/Gamestates/GameState.h>

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace RocketForge::SimulationAnalysis
{
	using RLGC::Action;
	using RLGC::FList;
	using RLGC::GameState;
	using RocketSim::Vec;

	/**
	 * CLI options for tracing a saved policy without starting a training run.
	 * Paths are absolute once parsed; build and segment are empty when explicit paths replaced the segment.
	 **/
	struct SimulationAnalysisOptions
	{
		std::string agentName;
		std::string buildName;
		std::string segmentId;
		std::filesystem::path setupPath;
		std::filesystem::path checkpointPath;
		std::filesystem::path outPath = "action_trace.csv";
		int episodes = 8;
		int maxTicks = 720;
		int teamSize = 1;
		uint64_t seed = 1234;
		bool deterministic = true;
		bool kickoffMode = false;
		bool wallProbeMode = false;
		float temperature = 1.f;
		std::string stageResetName;
		std::filesystem::path snapshotCsvPath;
		int snapshotTick = -1;
		int snapshotBotIndex = -1;
		int64_t snapshotCarId = -1;
		std::filesystem::path actionSequenceCsvPath;
		std::filesystem::path controlReplayCsvPath;
		int controlReplayInputDelayTicks = 0;
		bool controlReplayUseLastInput = false;
		bool controlReplayFocusPolicy = false;
		int forcedActionId = -1;
	};

	/**
	 * One row in the output CSV.
	 * Records both the selected policy action and the action applied on this physics tick.
	 **/
	struct TraceSample
	{
		std::string source = "rocketsim";
		int botIndex = 0;
		int botTeam = 0;
		uint32_t carId = 0;
		int episode = 0;
		int physicsTick = 0;
		int policyStep = 0;
		int policyTick = 0;
		int tickSkip = 0;
		int actionDelay = 0;
		int ticksElapsed = 1;
		float secondsElapsed = 0.f;
		float gameTimeRemaining = -1.f;
		float deltaTime = 1.f / 120.f;
		bool inferAction = false;
		bool applyQueuedAction = false;
		int selectedActionId = -1;
		bool selectedIsMacro = false;
		int selectedMacroIndex = -1;
		int appliedActionId = -1;
		int appliedBaseActionId = -1;
		int macroSegmentIndex = -1;
		int macroSegmentTick = -1;
		int macroSegmentTicks = -1;
		int macroTicksLeft = 0;
		std::string macroPlan;
		int maskAllowedCount = 0;
		uint64_t maskHash = 0;
		std::string maskBits;
		float maskBoost = 0.f;
		bool maskOnGround = false;
		bool maskWasOnGroundDuringStep = false;
		bool maskObsOnGround = false;
		bool maskIsJumping = false;
		bool maskIsFlipping = false;
		bool maskHasJumped = false;
		bool maskHasFlipped = false;
		bool maskHasDodged = false;
		bool maskHasDoubleJumped = false;
		int maskAirState = 0;
		float maskJumpTime = 0.f;
		float maskAirTime = 0.f;
		float maskAirTimeSinceJump = 0.f;
		float maskDodgeTimeout = 0.f;
		int obsSize = 0;
		uint64_t obsHash = 0;
		std::string obsValues;
		Action selectedAction = {};
		Action appliedAction = {};
		Action prevAction = {};
		Vec pos = {};
		Vec vel = {};
		Vec angVel = {};
		float speed = 0.f;
		float boost = 0.f;
		bool onGround = false;
		bool obsOnGround = false;
		int wheelsContactCount = 0;
		std::string wheelsContactBits;
		bool worldContact = false;
		bool wallContact = false;
		bool nearWall = false;
		Vec contactNormal = {};
		bool isJumping = false;
		bool isFlipping = false;
		bool hasJumped = false;
		bool hasFlipped = false;
		bool hasDodged = false;
		bool hasDoubleJumped = false;
		int airState = 0;
		float jumpTime = 0.f;
		float airTime = 0.f;
		float airTimeSinceJump = 0.f;
		float dodgeTimeout = 0.f;
		int rawAirState = -1;
		float rawDodgeTimeout = -1.f;
		float rawDemoTimeout = -1.f;
		uint32_t frameNum = 0;
		Action rlbotLastInput = {};
		float rawDodgeElapsed = -1.f;
		float rawDodgeDirX = 0.f;
		float rawDodgeDirY = 0.f;
		bool jumpEdge = false;
		bool ballTouchedTick = false;
		std::string stateSnapshot;
		int modelSelectedActionId = -1;
		Action modelSelectedAction = {};
		int modelAppliedActionId = -1;
		Action modelAppliedAction = {};
		std::string postStateSnapshot;
	};

	struct InferenceTrace
	{
		int actionId = -1;
		FList obs;
		std::vector<uint8_t> actionMask;
	};

	struct ControlReplayFrame
	{
		uint32_t carId = 0;
		int botIndex = -1;
		int botTeam = 0;
		int physicsTick = 0;
		int policyStep = 0;
		int policyTick = 0;
		int tickSkip = 0;
		int actionDelay = 0;
		int ticksElapsed = 1;
		float secondsElapsed = 0.f;
		float gameTimeRemaining = -1.f;
		float deltaTime = 1.f / 120.f;
		bool inferAction = false;
		bool applyQueuedAction = false;
		int selectedActionId = -1;
		bool selectedIsMacro = false;
		int selectedMacroIndex = -1;
		int appliedActionId = -1;
		int appliedBaseActionId = -1;
		int macroSegmentIndex = -1;
		int macroSegmentTick = -1;
		int macroSegmentTicks = -1;
		int macroTicksLeft = 0;
		std::string macroPlan;
		Action selectedAction = {};
		Action appliedAction = {};
		Action prevAction = {};
		uint32_t frameNum = 0;
		Action rlbotLastInput = {};
	};

	struct SnapshotReplaySeed
	{
		GameState state;

		/** Engine the snapshot was recorded on, or the loaded agent's training engine for snapshots that don't record one. **/
		RocketSim::EngineId engine = RocketSim::EngineId::V2;

		uint32_t tracedCarId = 0;
		int tracedBotIndex = -1;
		int sourcePhysicsTick = 0;
		int sourcePolicyStep = 0;
		ControlReplayFrame sourceFrame;
		Action previousAppliedAction = {};
		bool hasPreviousAppliedAction = false;
	};

	struct ControlReplayTrace
	{
		std::map<int, std::map<uint32_t, ControlReplayFrame>> framesByTick;
		int exactRows = 0;
		int tracedRows = 0;
		int startTick = 0;
		int endTick = 0;
	};

	struct WallDashPairStats
	{
		int firstActionId = -1;
		int secondActionId = -1;
		int count = 0;
		float speedDeltaSum = 0.f;
		float startSpeedSum = 0.f;
		float endSpeedSum = 0.f;
		int jumpEdgeSum = 0;
		int flipTickSum = 0;
		int wallTickSum = 0;
		int exampleEpisode = 0;
		int exampleTick = 0;
	};

	struct SequenceStats
	{
		std::vector<int> actionIds;
		int ticksPerAction = 0;
		int count = 0;
		float speedDeltaSum = 0.f;
		float startSpeedSum = 0.f;
		float endSpeedSum = 0.f;
		int jumpEdgeSum = 0;
		int flipTickSum = 0;
		int ballTouchTickSum = 0;
		int exampleEpisode = 0;
		int exampleTick = 0;
	};
}
