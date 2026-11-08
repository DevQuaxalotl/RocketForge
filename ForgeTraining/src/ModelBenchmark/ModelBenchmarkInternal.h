#pragma once

#include "ModelBenchmark.h"

#include <GigaLearnCPP/Eval/PolicyMatchRunner.h>
#include <GigaLearnInfer/Agent.h>
#include <GigaLearnInfer/FrameworkTorch.h>
#include <GigaLearnInfer/Models.h>
#include <RLGymCPP/Gamestates/StateUtil.h>
#include <RocketSim/Includes.h>

#include <array>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace RocketForgeTraining::ModelBenchmarkInternal
{
	using namespace RLGC;
	using namespace RocketSim;

	inline constexpr double LOGIT_TO_ELO = 173.71779276130073;

	using CarSnapshot = GGL::PolicyMatchCarSnapshot;
	using ArenaSnapshot = GGL::PolicyMatchSnapshot;

	/** Creates the effective agent used by both benchmark preflight and inference. **/
	std::unique_ptr<Agent> CreateBenchmarkAgent(const ModelBenchmarkModel& model);

	/** Frees every model owned by a benchmark ModelSet before deleting the container. **/
	struct BenchmarkModelSetDeleter {
		void operator()(GGL::ModelSet* modelSet) const noexcept;
	};

	using BenchmarkModelSetPtr = std::unique_ptr<GGL::ModelSet, BenchmarkModelSetDeleter>;

	/** Releases benchmark CUDA graphs and cached allocator memory when a run exits. **/
	class BenchmarkCudaCleanupGuard {
	public:
		BenchmarkCudaCleanupGuard() = default;
		~BenchmarkCudaCleanupGuard() noexcept;
		void Cleanup() noexcept;

		BenchmarkCudaCleanupGuard(const BenchmarkCudaCleanupGuard&) = delete;
		BenchmarkCudaCleanupGuard& operator=(const BenchmarkCudaCleanupGuard&) = delete;

	private:
		bool cleaned = false;
	};

	struct BenchmarkOutcome {
		int winner = -1;
		int loser = -1;
		int mode = 1;
		ModelBenchmarkGoalTiming goalTiming = ModelBenchmarkGoalTiming::PostKickoff;
		double weight = 1.0;
	};

	struct PairKey {
		int a = 0;
		int b = 0;
		int mode = 1;

		bool operator<(const PairKey& other) const {
			if (a != other.a) return a < other.a;
			if (b != other.b) return b < other.b;
			return mode < other.mode;
		}

		bool operator==(const PairKey& other) const {
			return a == other.a && b == other.b && mode == other.mode;
		}
	};

	struct BenchmarkWorkTarget {
		int mode = 1;
		bool valid = false;
	};

	struct LoadedModel {
		std::string name;
		std::string agentName;
		std::filesystem::path checkpointPath;
		std::unique_ptr<Agent> agent;
		BenchmarkModelSetPtr modelSet;
		CarConfig carConfig = CAR_CONFIG_OCTANE;
		float policyTemperature = 1.0f;
		int tickSkip = 1;
		int actionDelay = 0;
		int maxTeamSize = 1;
		int obsSize = 0;
		int numActions = 0;
		bool usesActionMask = false;
		std::string obsBuilderType;
		std::string actionParserType;
	};

	using RunningStats = ModelBenchmarkStatTotals;

	/** One exact car-ball touch used to evaluate completed benchmark passes. **/
	struct BenchmarkPassTouch {
		uint32_t carId = 0;
		Team team = Team::BLUE;
		uint64_t tick = 0;
		Vec ballPosition = {};
	};

	/** Per-episode possession chain retained between benchmark simulation steps. **/
	struct BenchmarkPassTracker {
		bool hasProcessedTouchTick = false;
		uint64_t processedThroughTick = 0;
		bool hasLastUncontestedTouch = false;
		BenchmarkPassTouch lastUncontestedTouch = {};
	};

	/** Completed pass counts discovered for both teams during one update. **/
	struct BenchmarkPassCounts {
		double blue = 0.0;
		double orange = 0.0;

		double ForTeam(Team team) const {
			return team == Team::BLUE ? blue : orange;
		}
	};

	struct BenchmarkTiming {
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
		double postProcessingSeconds = 0.0;
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

	enum class BenchmarkRecordType {
		Overall,
		Kickoff,
		PostKickoff
	};

	using BenchmarkRecords = std::map<PairKey, std::array<std::array<ModelBenchmarkRecord, 2>, 3>>;

	struct EloFit {
		std::vector<double> elo;
		std::vector<std::vector<double>> covariance;
	};

	/** One shared live Elo fit used by both progress ranking and summary text. **/
	struct LiveEloEstimate {
		bool hasEvidence = false;
		bool connected = false;
		EloFit fit;
		std::vector<int> order;
	};

	struct BenchmarkRunState {
		std::vector<BenchmarkOutcome> outcomes;
		std::vector<ModelBenchmarkGameMetadata> games;
		BenchmarkRecords records;
		std::vector<RunningStats> stats;
		std::map<int, std::vector<RunningStats>> statsByMode;
		std::map<PairKey, std::array<RunningStats, 2>> statsByPairMode;
		int scoredGames = 0;
		int noResultGames = 0;
		bool cancelled = false;
		std::string stopReason;
		std::map<int, int> gamesByMode;
		std::map<int, int> scheduledGamesByMode;
		std::map<int, int64_t> stepsByMode;
		std::map<int, int64_t> processedPhysicsTicksByMode;
		std::map<int, int64_t> completedPhysicsTicksByMode;
		std::map<PairKey, int> scheduledByPairMode;
		int targetGamesPerPairMode = 0;
		std::map<PairKey, int> targetByPairMode;
		uint64_t nextGameId = 1;
		uint64_t nextSnapshotId = 1;
		int scheduledGames = 0;
		int activeArenas = 0;
		int totalArenas = 0;
		int pendingGames = 0;
		int64_t arenaCreates = 0;
		int64_t arenaRebuilds = 0;
		int64_t arenaReuses = 0;
		int64_t cudaOomRecoveries = 0;
		bool cudaGraphsDisabledAfterOom = false;
		uint64_t cudaGraphEntries = 0;
		uint64_t cudaGraphCaptures = 0;
		uint64_t cudaGraphReplays = 0;
		uint64_t cudaGraphFallbacks = 0;
		uint64_t cudaGraphVariantLimitFallbacks = 0;
		uint64_t cudaGraphCaptureFailures = 0;
		int active1v1Arenas = 0;
		int active2v2Arenas = 0;
		int active3v3Arenas = 0;
		BenchmarkTiming timing;
	};

	/** The pair's own target when one was set, otherwise the uniform per-pair target. **/
	inline int PairGameTarget(const BenchmarkRunState& state, const PairKey& pair) {
		const auto found = state.targetByPairMode.find(pair);
		return found == state.targetByPairMode.end() ? state.targetGamesPerPairMode : found->second;
	}

	inline int64_t ModeGameTarget(const BenchmarkRunState& state, int mode) {
		int64_t target = 0;
		for (const auto& [pair, scheduled] : state.scheduledByPairMode)
			if (pair.mode == mode)
				target += PairGameTarget(state, pair);

		return target;
	}

	/**
	 * Holds unfinished mirrored starts outside the rated result set.
	 * A group is committed atomically only after every scheduled side completes.
	 **/
	class BenchmarkMirrorBuffer {
	public:
		bool Add(
			BenchmarkRunState& state,
			ModelBenchmarkGameMetadata metadata,
			int expectedGames,
			double outcomeWeight = 1.0);

	private:
		struct Key {
			uint64_t snapshotId = 0;
			int modelA = -1;
			int modelB = -1;
			int mode = 1;

			bool operator<(const Key& other) const {
				if (snapshotId != other.snapshotId) return snapshotId < other.snapshotId;
				if (modelA != other.modelA) return modelA < other.modelA;
				if (modelB != other.modelB) return modelB < other.modelB;
				return mode < other.mode;
			}
		};

		struct PendingGroup {
			int expectedGames = 0;
			std::vector<size_t> metadataIndices;
		};

		std::map<Key, PendingGroup> pendingGroups;
	};

	struct SlotBenchmarkParams {
		const ModelBenchmarkConfig& config;
		const ModelBenchmarkCancelCallback& isCancelled;
		std::vector<LoadedModel>& loadedModels;
		const std::vector<int>& modes;
		const std::vector<std::pair<int, int>>& matchups;
		const std::vector<std::string>& modelNames;
		ModelBenchmarkStrategy benchmarkStrategy = ModelBenchmarkStrategy::SameRuntimeSlot;
		std::string benchmarkingLabel;
		std::string slotPathLabel;
		int numModels = 0;
		int numArenas = 1;
		int tickSkip = 1;
		int actionDelay = 0;
		int noGoalSteps = 1;
		int noGoalTicks = 1;
		bool useVariableTiming = false;
		uint64_t runSeed = 0;
		std::function<void(const std::string&, const std::string&, const std::string&)> reportProgress;
		std::function<bool(int)> shouldReportLiveProgress;
		std::function<bool()> shouldStopEarly;
		/** Optional runner override used by focused scheduler integration tests. **/
		std::function<GGL::PolicyMatchRunnerResult(const GGL::PolicyMatchRunnerParams&)> runPolicyMatchRunner;
		/** Must match the device loadedModels were loaded onto. **/
		torch::Device device = torch::kCPU;
	};

	double SecondsSince(std::chrono::steady_clock::time_point start);
	std::string FormatBenchmarkTiming(const BenchmarkTiming& timing, double simulationSeconds, int finishedGames);
	std::string ModeName(int mode);
	double DefaultModeDurationWeight(int mode);
	double FinishedGameSeconds(int finishSteps, bool useVariableTiming, int tickSkip);
	ModelBenchmarkGoalTiming ClassifyGoalTiming(
		bool goalScored,
		double goalTimeSeconds,
		double kickoffGoalWindowSeconds);
	double BenchmarkWorkProgress(
		bool runUntilMaxGames,
		int completedGames,
		int scheduledGames,
		int targetGames,
		double modeDurationWeight);
	bool BenchmarkArenaLayoutCanBeReclaimed(int completedGames, int targetGames);
	bool BenchmarkMirrorGroupComplete(int completedGames, int expectedGames);
	bool IncludeBenchmarkGameInRating(bool goalScored, bool mirrorGroupComplete);
	std::vector<GGL::PolicyMatchTask> BuildBenchmarkMirrorTasks(
		int modelA,
		int modelB,
		int teamSize,
		int modelAArenaKey,
		int modelBArenaKey,
		const ArenaSnapshot& snapshot,
		uint64_t snapshotId,
		uint64_t firstGameId,
		int gameCount,
		bool modelABlueFirst = true);

	double ClampProbability(double p);
	double NormalCDF(double x);
	double DifferenceVariance(const std::vector<std::vector<double>>& covariance, int a, int b);
	EloFit FitElo(int numModels, const std::vector<BenchmarkOutcome>& outcomes);
	double ConfidenceBetter(const EloFit& fit, int a, int b);
	std::string FormatConfidenceList(const EloFit& fit, const std::vector<std::string>& names, int modelIdx);
	EloFit MakeRelative(EloFit fit);
	/** Anchors overall and timing Elo fits to the worst overall model's rating. **/
	void AnchorFitsToOverallFloor(EloFit& overallFit, EloFit& kickoffFit, EloFit& postKickoffFit);
	std::vector<int> BuildRankOrderFromFit(int numModels, const EloFit& fit, const std::vector<std::string>& names);
	std::vector<int> BuildRankOrder(int numModels, const std::vector<BenchmarkOutcome>& outcomes, const std::vector<std::string>& names);
	LiveEloEstimate BuildLiveEloEstimate(
		int numModels,
		const std::vector<BenchmarkOutcome>& outcomes,
		const std::vector<std::string>& names);
	std::string FormatLiveEstimateSummary(
		const LiveEloEstimate& estimate,
		const std::vector<std::string>& names,
		double eloRangeConfidence);
	std::string FormatLiveEstimateSummary(
		int numModels,
		const std::vector<BenchmarkOutcome>& outcomes,
		const std::vector<std::string>& names,
		double eloRangeConfidence);
	bool AllRangesSmallEnough(const EloFit& fit, double confidence, double maxRange);
	bool HasMinimumRatedOutcomesForEarlyStop(int ratedOutcomeCount, int matchupCount);
	bool HasConnectedOutcomeGraph(int numModels, const std::vector<BenchmarkOutcome>& outcomes);
	int CountModelOutcomes(const std::vector<BenchmarkOutcome>& outcomes, int modelIdx);

	std::unique_ptr<Arena> MakeArena(int teamSize, const CarConfig& blueCarConfig, const CarConfig& orangeCarConfig, EngineId engine);
	ArenaSnapshot CaptureSnapshot(Arena* arena);
	void ApplySnapshot(Arena* arena, const ArenaSnapshot& snapshot);
	ArenaSnapshot SampleKickoffState(int teamSize, const CarConfig& carConfig, EngineId engine);
	ArenaSnapshot SampleKickoffState(
		int teamSize,
		const CarConfig& carConfig,
		uint64_t seed,
		EngineId engine,
		const std::filesystem::path& startingStateDirectory = {});

	ModelBenchmarkRuntimeSignature RuntimeSignature(const LoadedModel& model);
	std::string SameRuntimeSlotMismatchReason(const std::vector<LoadedModel>& models);
	BenchmarkModelSetPtr LoadModelSet(const LoadedModel& model, torch::Device device);
	std::vector<int> SelectedModes(const ModelBenchmarkConfig& config, int maxTeamSize);
	std::vector<PairKey> BuildEligiblePairModes(
		const std::vector<std::pair<int, int>>& matchups,
		const std::vector<int>& modelMaxTeamSizes,
		const std::vector<int>& selectedModes);
	ModelBenchmarkEligibilityAnalysis AnalyzeModelBenchmarkEligibility(
		const std::vector<ModelBenchmarkModel>& models,
		const std::vector<int>& modelMaxTeamSizes,
		const std::vector<int>& selectedModes);
	bool HasConnectedPairModeGraph(int numModels, const std::vector<PairKey>& pairModes);
	int CountEligiblePairsForMode(const std::vector<PairKey>& pairModes, int mode);
	double OverallOutcomeWeightForPair(
		const std::vector<PairKey>& pairModes,
		int modelA,
		int modelB);

	ModelBenchmarkRecord& GetRecord(BenchmarkRecords& records, int model, int opponent, int mode, BenchmarkRecordType recordType);
	void AddResultToRecords(BenchmarkRecords& records, int winner, int loser, int mode, ModelBenchmarkGoalTiming goalTiming);
	void AddNoResultToRecords(BenchmarkRecords& records, int modelA, int modelB, int mode);
	double WinPct(const ModelBenchmarkRecord& record);
	std::string FormatRecordSummary(
		const BenchmarkRecords& records,
		const std::vector<std::string>& names,
		const std::vector<int>& rankOrder = {});
	std::vector<BenchmarkOutcome> FilterOutcomes(
		const std::vector<BenchmarkOutcome>& outcomes,
		ModelBenchmarkGoalTiming goalTiming);
	std::vector<BenchmarkOutcome> FilterOutcomesByMode(const std::vector<BenchmarkOutcome>& outcomes, int mode);

	void AccumulateStepStats(RunningStats& stats, const RLGC::GameState& state, const RLGC::GameState& prev, Team team, float deltaSeconds);
	BenchmarkPassCounts AccumulateCompletedPasses(
		BenchmarkPassTracker& tracker,
		std::vector<BenchmarkPassTouch> touches);
	BenchmarkPassCounts AccumulateCompletedPasses(
		BenchmarkPassTracker& tracker,
		const RLGC::GameState& state);
	void MergeRunningStats(RunningStats& destination, const RunningStats& source);
	ModelBenchmarkOptionalStats FinalizeStats(const RunningStats& stats);

	void RunSlotBenchmark(const SlotBenchmarkParams& params, BenchmarkRunState& state);
}
