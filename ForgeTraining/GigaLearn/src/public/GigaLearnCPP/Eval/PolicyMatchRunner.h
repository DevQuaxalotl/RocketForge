#pragma once

#include <GigaLearnInfer/FrameworkTorch.h>
#include <GigaLearnInfer/Models.h>
#include <RocketSim/Includes.h>
#include <RLGymCPP/ActionParsers/ActionParser.h>
#include <RLGymCPP/ObsBuilders/ObsBuilder.h>
#include <RLGymCPP/Rewards/Reward.h>
#include <RLGymCPP/StateSetters/StateSetter.h>
#include <RLGymCPP/TerminalConditions/TerminalCondition.h>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace GGL {

	/** Start-state family to sample for policy match episodes. **/
	enum class PolicyMatchStartType {
		Kickoff,
		RandomState
	};

	/** Saved car state within a reusable policy-match start snapshot. **/
	struct PolicyMatchCarSnapshot {
		RocketSim::Team team = RocketSim::Team::BLUE;
		RocketSim::CarState state = {};
	};

	/** Saved arena state used to reset reusable slots without rebuilding arenas. **/
	struct PolicyMatchSnapshot {
		RocketSim::BallState ball = {};
		std::vector<PolicyMatchCarSnapshot> cars;
	};

	/** Per-slot observation and action runtime for one compatible runtime group. **/
	struct PolicyMatchRuntimeComponents {
		std::unique_ptr<RLGC::ObsBuilder> obsBuilder;
		std::unique_ptr<RLGC::ActionParser> actionParser;
	};

	/** One policy model plus the runtime objects required to produce actions for it. **/
	struct PolicyMatchRuntime {
		ModelSet* models = nullptr;
		RocketSim::CarConfig carConfig = RocketSim::CAR_CONFIG_OCTANE;
		int obsSize = 0;
		int numActions = 0;
		bool usesActionMask = false;
		bool deterministic = true;

		/** Sample policy actions until the first car-ball contact in each episode. **/
		bool stochasticKickoff = false;
		float kickoffTemperature = 0.70f;

		/** Optional run-local RNG, so evaluation does not consume the training RNG. **/
		std::optional<at::Generator> actionGenerator;

		bool useHalfPrecision = false;
		bool useCudaGraph = false;
		float policyTemperature = 1.0f;
		float minActionProbability = 0.0f;
		int tickSkip = 1;
		int actionDelay = 0;
		int runtimeGroup = 0;
		std::function<PolicyMatchRuntimeComponents()> makeRuntimeComponents;
	};

	/** Episode assigned to a reusable slot. userValues are owned by the caller. **/
	struct PolicyMatchTask {
		int blueModel = -1;
		int orangeModel = -1;
		int blueArenaKey = -1;
		int orangeArenaKey = -1;
		int teamSize = 1;
		int resumeSteps = 0;
		PolicyMatchSnapshot snapshot = {};
		std::array<int, 4> userValues = { 0, 0, 0, 0 };
		uint64_t gameId = 0;
		uint64_t snapshotId = 0;
		int mirrorGameCount = 1;
	};

	/** Optional arena construction result for callers that attach GameState userInfo. **/
	struct PolicyMatchArena {
		std::unique_ptr<RocketSim::Arena> arena;
		void* userInfo = nullptr;
		PolicyMatchRuntimeComponents runtimeComponents = {};
		int runtimeGroup = 0;

		/** Optional environment objects kept alive with the arena. **/
		std::vector<std::unique_ptr<RLGC::Reward>> ownedRewards;
		std::vector<std::unique_ptr<RLGC::TerminalCondition>> ownedTerminalConditions;
		std::unique_ptr<RLGC::StateSetter> ownedStateSetter;
	};

	/** Per-step callback payload after physics and state updates complete. **/
	struct PolicyMatchStep {
		const PolicyMatchTask& task;
		const RLGC::GameState& state;
		const RLGC::GameState& prevState;
		int steps = 0;
		float deltaSeconds = 0.0f;

		/** Slot-local storage index for parallel statistics. **/
		int slotIndex = -1;
	};

	/** Episode completion callback payload. **/
	struct PolicyMatchFinish {
		const PolicyMatchTask& task;
		const RLGC::GameState& state;
		int steps = 0;
		bool goalScored = false;
		RocketSim::Team scoringTeam = RocketSim::Team::BLUE;
		int winningModel = -1;
		int losingModel = -1;
	};

	/** Timing buckets for the reusable slot runner. **/
	struct PolicyMatchTiming {
		double physicsSeconds = 0.0;
		double obsMaskSeconds = 0.0;
		double gpuTransferSeconds = 0.0;
		double inferenceSeconds = 0.0;
		double actionParseSeconds = 0.0;
		double actionStepSeconds = 0.0;
		double finishSeconds = 0.0;
		double schedulerSeconds = 0.0;
		double slotAssignSeconds = 0.0;
		double slotArenaSetupSeconds = 0.0;
		double slotSideRuntimeSeconds = 0.0;
		double slotSnapshotSeconds = 0.0;
		double slotStateInitSeconds = 0.0;
		double slotBuilderResetSeconds = 0.0;
		double batchPrepSeconds = 0.0;
		int64_t cudaOomRecoveries = 0;
		bool cudaGraphsDisabledAfterOom = false;
		uint64_t cudaGraphEntries = 0;
		uint64_t cudaGraphCaptures = 0;
		uint64_t cudaGraphReplays = 0;
		uint64_t cudaGraphFallbacks = 0;
		uint64_t cudaGraphVariantLimitFallbacks = 0;
		uint64_t cudaGraphCaptureFailures = 0;
		int64_t slotArenaCreates = 0;
		int64_t slotArenaRebuilds = 0;
		int64_t slotArenaReuses = 0;
	};

	/** Lightweight live state for runner telemetry. **/
	struct PolicyMatchRunnerStatus {
		int totalSlots = 0;
		int activeSlots = 0;
		int pendingTasks = 0;
		int64_t slotArenaCreates = 0;
		int64_t slotArenaRebuilds = 0;
		int64_t slotArenaReuses = 0;
		int64_t cudaOomRecoveries = 0;
		bool cudaGraphsDisabledAfterOom = false;
		uint64_t cudaGraphEntries = 0;
		uint64_t cudaGraphCaptures = 0;
		uint64_t cudaGraphReplays = 0;
		uint64_t cudaGraphFallbacks = 0;
		uint64_t cudaGraphVariantLimitFallbacks = 0;
		uint64_t cudaGraphCaptureFailures = 0;
		std::vector<std::pair<int, int>> activeSlotsByTeamSize;
	};

	/** Idle arena demand for an exact physical layout. **/
	struct PolicyMatchArenaLayoutDemand {
		int teamSize = 1;
		int blueArenaKey = -1;
		int orangeArenaKey = -1;
		int idleSlots = 0;
	};

	/** Configuration and callbacks for running policy match episodes through reusable slots. **/
	struct PolicyMatchRunnerParams {
		std::vector<PolicyMatchRuntime> runtimes;
		torch::Device device = torch::Device(torch::kCUDA);
		int numSlots = 1;
		int tickSkip = 1;
		int actionDelay = 0;
		int noGoalSteps = 1;

		std::function<PolicyMatchArena(const PolicyMatchTask& task)> makeArena;
		std::function<void(std::vector<PolicyMatchTask>& pendingTasks, int activeSlots, int targetQueueSize)> fillPendingTasks;
		std::function<bool()> isSchedulingComplete;
		std::function<bool()> isCancelled;
		std::function<void(const PolicyMatchStep& step)> onStep;
		std::function<bool(const PolicyMatchFinish& finish)> onFinished;
		int noGoalTicks = 0;

		/**
		 * Returns true only after a physical arena layout has no remaining work.
		 * When provided, idle slots from terminal layouts may be rebuilt in batches.
		 **/
		std::function<bool(int teamSize, int blueArenaKey, int orangeArenaKey)> isArenaLayoutComplete;
		std::function<void(const PolicyMatchRunnerStatus& status)> onStatus;

		/**
		 * Optionally schedules compatible work for idle, nonterminal layouts even
		 * when the normal global pending queue has reached its capacity target.
		 **/
		std::function<void(
			std::vector<PolicyMatchTask>& pendingTasks,
			const std::vector<PolicyMatchArenaLayoutDemand>& demands)> fillPendingTasksForArenaLayouts;

		/** Each arena advances to its own decision boundary when enabled. **/
		bool independentSlotTiming = false;
		int maxPhysicsStepTicks = 0;
		std::function<void(const PolicyMatchStep& step)> onStepParallel;
	};

	struct PolicyMatchRunnerResult {
		bool cancelled = false;
		PolicyMatchTiming timing = {};
	};

	GGL_TRAIN_IMEXPORT std::unique_ptr<RocketSim::Arena> MakePolicyMatchArena(
		int teamSize,
		const RocketSim::CarConfig& blueCarConfig,
		const RocketSim::CarConfig& orangeCarConfig,
		RocketSim::EngineId engine);
	GGL_TRAIN_IMEXPORT PolicyMatchSnapshot CapturePolicyMatchSnapshot(RocketSim::Arena* arena);
	GGL_TRAIN_IMEXPORT void ApplyPolicyMatchSnapshot(RocketSim::Arena* arena, const PolicyMatchSnapshot& snapshot);
	GGL_TRAIN_IMEXPORT PolicyMatchSnapshot SamplePolicyMatchStartState(
		PolicyMatchStartType startType,
		int teamSize,
		const RocketSim::CarConfig& carConfig,
		RocketSim::EngineId engine);
	GGL_TRAIN_IMEXPORT PolicyMatchSnapshot SamplePolicyMatchStartState(
		PolicyMatchStartType startType,
		int teamSize,
		const RocketSim::CarConfig& carConfig,
		uint64_t seed,
		RocketSim::EngineId engine);
	GGL_TRAIN_IMEXPORT PolicyMatchRunnerResult RunPolicyMatchSlots(const PolicyMatchRunnerParams& params);
	GGL_TRAIN_IMEXPORT PolicyMatchRunnerResult RunPolicyMatchVariableTimingSlots(const PolicyMatchRunnerParams& params);
}
