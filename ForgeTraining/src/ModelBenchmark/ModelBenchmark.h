#pragma once

#include <RocketSim/Engine.h>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace RocketForgeTraining {
	struct ModelBenchmarkResult;

	/** Goal timing bucket used for kickoff-versus-gameplay benchmark reporting. **/
	enum class ModelBenchmarkGoalTiming {
		NoGoal,
		Kickoff,
		PostKickoff
	};

	/** Execution strategy chosen from the selected models' runtime compatibility. **/
	enum class ModelBenchmarkStrategy {
		SameRuntimeSlot,
		Heterogeneous
	};

	/** Agent setup source to use when constructing a selected benchmark model. **/
	enum class ModelBenchmarkAgentSetupSource {
		SavedSegment,
		CurrentAgent
	};

	/** Runtime interface fields used to decide whether models can share the same slot path. **/
	struct ModelBenchmarkRuntimeSignature {
		std::string agentName;
		int tickSkip = 1;
		int actionDelay = 0;
		int obsSize = 0;
		int numActions = 0;
		int maxTeamSize = 1;
		std::string obsBuilderType;
		std::string actionParserType;
		std::string carConfigSignature;
		std::string policyConfigSignature;
		int ballPredictionEngineVersion = 2;
	};

	/** One model checkpoint selected for a benchmark run. **/
	struct ModelBenchmarkModel {
		std::string displayName;
		std::string agentName;
		std::string stageName;
		std::string versionId;
		std::filesystem::path checkpointPath;
		std::filesystem::path trainingSetupPath;
		ModelBenchmarkAgentSetupSource agentSetupSource = ModelBenchmarkAgentSetupSource::SavedSegment;

		/** Primary models play every selected model; non-primary models only play primary models. **/
		bool primary = true;

		/** Captured effective Agent setup for repeatable runs, independent of mutable source files. **/
		std::string agentSetupJson;
	};

	/** Explicit new-game budget for one unordered model pair and team size. **/
	struct ModelBenchmarkPairBudget {
		int modelA = 0;
		int modelB = 1;
		int mode = 1;
		int games = 0;
	};

	/** User-facing benchmark run options. **/
	struct ModelBenchmarkConfig {
		std::vector<ModelBenchmarkModel> models;

		/** Match physics for every game; models still predict the ball with their own engines. **/
		RocketSim::EngineId engine = RocketSim::EngineId::V2;

		int numArenas = 500;
		int maxGamesPerMode = 500;

		/** When enabled, absent pair-modes have no work; saved games stay outside this run. **/
		bool usePairBudgets = false;
		std::vector<ModelBenchmarkPairBudget> pairBudgets;

		/** Advances fixed-seed top-ups beyond recorded starting snapshots. **/
		uint64_t snapshotOffset = 0;

		/** Frozen portable starts. Empty samples starting states during the run instead. **/
		std::filesystem::path startingStateDirectory;

		/** Per-model prediction engine identity; zero selects match physics. **/
		std::vector<int> modelPredictionEngines;

		bool run1v1 = true;
		bool run2v2 = true;
		bool run3v3 = true;

		/** Goals at or before this simulated game time are classified as kickoff goals. **/
		float kickoffGoalWindowSeconds = 10.0f;
		/** Sample actions until the first car-ball touch, then use deterministic inference. **/
		bool stochasticKickoff = false;
		float kickoffTemperature = 0.70f;

		float betterConfidence = 0.95f;
		bool stopWhenRankingKnown = false;
		float eloRangeConfidence = 0.95f;
		float maxEloUncertaintyRange = 50.0f;
		bool runUntilMaxGames = false;

		bool collectOptionalStats = true;
		float noGoalTimeoutSeconds = 300.0f;

		/** Emits throttled result snapshots through the progress callback. **/
		bool emitLiveResultSnapshots = false;

		/** Reproducible benchmark seed. Zero selects a new seed automatically. **/
		uint64_t seed = 0;

		/** Opt-in live transport: snapshots contain only games after liveGamesOffset. **/
		bool emitLiveGameDeltas = false;
	};

	/** Preflight summary of the models and pair-modes that can participate. **/
	struct ModelBenchmarkEligibilityAnalysis {
		std::vector<int> includedModelIndices;
		std::vector<int> excludedModelIndices;
		std::vector<int> eligibleModes;
		int eligibleMatchupCount = 0;
		int eligiblePairModeCount = 0;
		std::vector<ModelBenchmarkPairBudget> eligiblePairModes;
		bool comparisonGraphConnected = false;
	};

	/** Throttled CUDA allocator and graph-cache snapshot for benchmark diagnostics. **/
	struct ModelBenchmarkGpuTelemetry {
		bool available = false;
		int64_t allocatedBytes = 0;
		int64_t reservedBytes = 0;
		int64_t peakAllocatedBytes = 0;
		int64_t peakReservedBytes = 0;
		int64_t freeBytes = 0;
		int64_t totalBytes = 0;
		int64_t allocationRetries = 0;
		int64_t allocatorOoms = 0;
		uint64_t graphEntries = 0;
		uint64_t graphCaptures = 0;
		uint64_t graphReplays = 0;
		uint64_t graphFallbacks = 0;
		uint64_t graphVariantLimitFallbacks = 0;
		uint64_t graphCaptureFailures = 0;
	};

	/** Lightweight status payload emitted while a benchmark is running. **/
	struct ModelBenchmarkProgress {
		int scoredGames = 0;
		int noResultGames = 0;
		int64_t maxGames = 0;
		double elapsedSeconds = 0.0;
		double snapshotGamesPerSecond = 0.0;
		std::string label;
		std::string details;
		std::string estimateSummary;
		double estimatedRemainingSeconds = -1.0;
		int activeArenas = 0;
		int totalArenas = 0;
		int pendingGames = 0;
		int scheduledGames = 0;
		int64_t arenaCreates = 0;
		int64_t arenaRebuilds = 0;
		int64_t arenaReuses = 0;
		int active1v1Arenas = 0;
		int active2v2Arenas = 0;
		int active3v3Arenas = 0;
		int64_t cudaOomRecoveries = 0;
		bool cudaGraphsDisabledAfterOom = false;
		ModelBenchmarkGpuTelemetry gpu;

		/** Throttled summary and game records for live GUI table updates. **/
		std::shared_ptr<const ModelBenchmarkResult> liveResult;

		/** -1 denotes a complete snapshot; nonnegative offsets append to prior live games. **/
		int64_t liveGamesOffset = -1;
	};

	/** Win/loss/no-result totals for one model side. **/
	struct ModelBenchmarkRecord {
		int wins = 0;
		int losses = 0;
		int noResults = 0;
	};

	/** First lossless-total version containing the expanded gameplay statistics. **/
	inline constexpr int MODEL_BENCHMARK_EXPANDED_STAT_DEFINITION_VERSION = 2;
	/** Version of the optional-stat definitions stored in lossless benchmark totals. **/
	inline constexpr int MODEL_BENCHMARK_STAT_DEFINITION_VERSION = 3;

	/** Optional gameplay statistics collected per model. Event counts are per five-minute game. **/
	struct ModelBenchmarkOptionalStats {
		/** Simulated exposure, with 300 seconds equal to one game. **/
		double equivalentGames = 0.0;
		double goals = 0.0;
		double shots = 0.0;
		double shotConversionPercent = 0.0;
		double assists = 0.0;
		double assistedGoalPercent = 0.0;
		double saves = 0.0;
		double savePercent = 0.0;
		/** Completed teammate-to-teammate passes per eligible 2v2/3v3 game. **/
		double passes = 0.0;
		double ballTouches = 0.0;
		double touchSharePercent = 0.0;
		/** Flip resets acquired from ball contact per game. **/
		double flipResets = 0.0;
		double bumps = 0.0;
		double demos = 0.0;
		double boostSteals = 0.0;
		double bigBoostPads = 0.0;
		double smallBoostPads = 0.0;
		double boostConsumptionPerMinute = 0.0;
		double averageBoost = 0.0;
		double averageSpeed = 0.0;
		double aerialTimePercent = 0.0;
		double lowAirTimePercent = 0.0;
		double highAirTimePercent = 0.0;
		double lowBoostTimePercent = 0.0;
		double averageSupersonicTime = 0.0;
		double fieldDomination = 0.0;
		double offensiveThirdPercent = 0.0;
		double neutralThirdPercent = 0.0;
		double defensiveThirdPercent = 0.0;
		double averageTouchHeight = 0.0;
		double averageSpeedTowardsBall = 0.0;
		double averageDistanceToBall = 0.0;
		int samples = 0;
		/** False when aggregated legacy totals predate the expanded stat definitions. **/
		bool expandedStatsAvailable = true;
		/** False when totals lack the denominators introduced for efficiency statistics. **/
		bool efficiencyStatsAvailable = true;
		/** False for 1v1-only exposure, where a teammate pass is impossible. **/
		bool passesAvailable = false;
	};

	/** Lossless statistic accumulators retained for saved-result aggregation. **/
	struct ModelBenchmarkStatTotals {
		int definitionVersion = MODEL_BENCHMARK_STAT_DEFINITION_VERSION;
		double goals = 0.0;
		double shots = 0.0;
		double assists = 0.0;
		double saves = 0.0;
		double goalsConceded = 0.0;
		double passes = 0.0;
		double passGameTimeSum = 0.0;
		double ballTouches = 0.0;
		double opponentBallTouches = 0.0;
		double flipResets = 0.0;
		double bumps = 0.0;
		double demos = 0.0;
		double boostSteals = 0.0;
		double bigBoostPads = 0.0;
		double smallBoostPads = 0.0;
		double boostConsumed = 0.0;
		double gameTimeSum = 0.0;
		double boostSum = 0.0;
		double speedSum = 0.0;
		double aerialTimeSum = 0.0;
		double lowAerialTimeSum = 0.0;
		double highAerialTimeSum = 0.0;
		double lowBoostTimeSum = 0.0;
		double supersonicTimeSum = 0.0;
		double playerTimeSum = 0.0;
		double opponentHalfBallSteps = 0.0;
		double offensiveThirdBallSteps = 0.0;
		double neutralThirdBallSteps = 0.0;
		double defensiveThirdBallSteps = 0.0;
		double touchHeightSum = 0.0;
		double touchCount = 0.0;
		double speedTowardsBallSum = 0.0;
		double distanceToBallSum = 0.0;
		int64_t stepSamples = 0;
		int64_t playerSamples = 0;
	};

	/** Raw statistics for one model/opponent/mode contribution. **/
	struct ModelBenchmarkStatTotalsRow {
		int modelIndex = -1;
		int opponentIndex = -1;
		int mode = 1;
		ModelBenchmarkStatTotals totals;
	};

	/** Overall ranking row shown in GUI and export results. **/
	struct ModelBenchmarkRankingRow {
		int rank = 0;
		int modelIndex = -1;
		std::string modelName;
		double relativeElo = 0.0;
		double elo1v1 = 0.0;
		double elo2v2 = 0.0;
		double elo3v3 = 0.0;
		double rangeLow = 0.0;
		double rangeHigh = 0.0;
		std::string betterThanSummary;
		ModelBenchmarkRecord totalRecord;
		/** Per-mode Elo is only available when rated outcomes connect that mode's participants. **/
		bool elo1v1Available = false;
		bool elo2v2Available = false;
		bool elo3v3Available = false;
	};

	/** Kickoff/post-kickoff comparison row for one model. **/
	struct ModelBenchmarkBreakdownRow {
		int modelIndex = -1;
		std::string modelName;
		double combinedElo = 0.0;
		double kickoffElo = 0.0;
		double kickoffRangeLow = 0.0;
		double kickoffRangeHigh = 0.0;
		double postKickoffElo = 0.0;
		double postKickoffRangeLow = 0.0;
		double postKickoffRangeHigh = 0.0;
		double kickoffDelta = 0.0;
		int kickoffGames = 0;
		int postKickoffGames = 0;
		bool kickoffEloAvailable = false;
		bool postKickoffEloAvailable = false;
	};

	/** Pairwise result row for one ordered model/opponent/mode tuple. **/
	struct ModelBenchmarkPairwiseRow {
		int modelIndex = -1;
		int opponentIndex = -1;
		std::string modelName;
		std::string opponentName;
		std::string modeName;
		double modeElo = 0.0;
		ModelBenchmarkRecord combined;
		ModelBenchmarkRecord kickoff;
		ModelBenchmarkRecord postKickoff;
		double kickoffVsPostKickoffDeltaPercent = 0.0;
		bool timingDeltaAvailable = false;
	};

	/** Optional stats row for one model. **/
	struct ModelBenchmarkStatRow {
		int modelIndex = -1;
		std::string modelName;
		ModelBenchmarkOptionalStats stats;
	};

	/** Per-game identity and pairing metadata retained for reproducible result exports. **/
	struct ModelBenchmarkGameMetadata {
		uint64_t gameId = 0;
		uint64_t snapshotId = 0;
		int modelA = -1;
		int modelB = -1;
		int blueModel = -1;
		int orangeModel = -1;
		int mode = 1;
		int winner = -1;
		double goalTimeSeconds = 0.0;
		ModelBenchmarkGoalTiming goalTiming = ModelBenchmarkGoalTiming::NoGoal;
		bool goalScored = false;
		bool includedInRating = false;
		bool mirrorCompleted = false;
	};

	/** Complete benchmark result, ready for GUI presentation or export. **/
	struct ModelBenchmarkResult {
		/** Legacy scalar retained for existing callers and serialized histories. **/
		int engineVersion = 0;

		/** Simulator versions used by contributing runs. Missing/zero historical versions resolve to v2. **/
		std::set<int> rocketSimVersions;

		/** Empty values preserve unidentified historical provenance. **/
		std::string engineFamily;
		std::string engineRevision;
		std::string engineCarOrder;
		std::string integrationRevision;
		std::string predictionConfigurationJson;
		std::string evaluationIdentity;

		std::string agentName;
		std::string stageName;
		int scoredGames = 0;
		int noResultGames = 0;
		double elapsedSeconds = 0.0;
		double simulationSeconds = 0.0;
		double gamesPerSecond = 0.0;
		std::string stopReason;
		std::string benchmarkStrategy;
		std::string timingSummary;
		float kickoffGoalWindowSeconds = 10.0f;

		/** Sample actions until the first car-ball touch, then use deterministic inference. **/
		bool stochasticKickoff = false;
		float kickoffTemperature = 1.0f;

		/** Saved aggregates can combine runs with different kickoff inference settings. **/
		bool kickoffInferenceSettingsMixed = false;
		float eloRangeConfidence = 0.95f;
		int kickoffGoalCount = 0;
		int postKickoffGoalCount = 0;
		bool kickoffEloAvailable = false;
		bool postKickoffEloAvailable = false;
		/** Overall Elo is only available when rated outcomes connect every selected model. **/
		bool overallEloAvailable = false;
		bool has1v1 = false;
		bool has2v2 = false;
		bool has3v3 = false;
		std::vector<std::string> modelNames;
		std::vector<std::string> modelVersionIds;
		std::vector<std::filesystem::path> modelCheckpointPaths;
		std::vector<ModelBenchmarkRankingRow> rankings;
		std::vector<ModelBenchmarkBreakdownRow> breakdown;
		std::vector<ModelBenchmarkPairwiseRow> pairwise;
		/** All selected modes combined. **/
		std::vector<ModelBenchmarkStatRow> optionalStats;
		/** Per-mode statistics keyed by team size (1, 2, or 3). **/
		std::map<int, std::vector<ModelBenchmarkStatRow>> optionalStatsByMode;
		/** Per-matchup totals used to filter and merge saved benchmark data exactly. **/
		std::vector<ModelBenchmarkStatTotalsRow> statTotals;
		/** False for legacy imports whose exported statistics were already rounded. **/
		bool statTotalsComplete = true;
		uint64_t seed = 0;
		int64_t maxGames = 0;
		int64_t arenaCreates = 0;
		int64_t arenaRebuilds = 0;
		int64_t arenaReuses = 0;
		int64_t cudaOomRecoveries = 0;
		bool cudaGraphsDisabledAfterOom = false;
		ModelBenchmarkGpuTelemetry gpu;
		std::vector<ModelBenchmarkGameMetadata> games;
	};

	/** Records each model's ball-prediction engine and setup JSON from its agent. **/
	ModelBenchmarkConfig PrepareModelBenchmarkPrediction(const ModelBenchmarkConfig& config);

	/** Describes the config's match engine environment without loading checkpoints or simulating games. **/
	ModelBenchmarkResult DescribeModelBenchmarkEnvironment(const ModelBenchmarkConfig& config);

	/** Throws EngineUnavailableError unless this build has the match engine and every model's prediction engine. **/
	void RequireModelBenchmarkEngines(const ModelBenchmarkConfig& config);

	/** Records the match engine, its revision and car order, plus each model's predictor engine. **/
	void RecordModelBenchmarkEnvironment(
		ModelBenchmarkResult& result,
		const ModelBenchmarkConfig& config,
		const std::string& family,
		const std::string& revision,
		const std::string& carOrder,
		const std::map<int, std::string>& predictorRevisions = {},
		bool noBallRot = true);

	/** Resolves a new-game budget; unspecified pairs are skipped in an explicit schedule. **/
	int ModelBenchmarkPairGameTarget(const ModelBenchmarkConfig& config, int modelA, int modelB, int mode);

	using ModelBenchmarkProgressCallback = std::function<void(const ModelBenchmarkProgress&)>;
	using ModelBenchmarkCancelCallback = std::function<bool()>;

	/** Resolves historical simulator metadata; pre-versioned RocketForge only supported v2. **/
	std::set<int> ResolveModelBenchmarkSimulatorVersions(const std::set<int>& versions, int legacyEngineVersion = 0);

	/** Formats simulator provenance, including legacy v2 and mixed saved results. **/
	std::string ModelBenchmarkRocketSimVersionsText(const std::set<int>& versions);

	/** Returns the stable display name for a benchmark strategy. **/
	std::string ModelBenchmarkStrategyName(ModelBenchmarkStrategy strategy);

	/** Selects the compatible execution strategy for the given model signatures. **/
	ModelBenchmarkStrategy SelectModelBenchmarkStrategy(const std::vector<ModelBenchmarkRuntimeSignature>& signatures);

	/** Builds benchmark model pairs, skipping pairs where neither model is primary. **/
	std::vector<std::pair<int, int>> BuildModelBenchmarkMatchups(const std::vector<ModelBenchmarkModel>& models);

	/** Resolves model capabilities and reports which selections can play an eligible game. **/
	ModelBenchmarkEligibilityAnalysis AnalyzeModelBenchmarkEligibility(const ModelBenchmarkConfig& config);

	/** Runs the selected model checkpoint benchmark and returns aggregate results. **/
	ModelBenchmarkResult RunModelBenchmark(
		const ModelBenchmarkConfig& config,
		const ModelBenchmarkProgressCallback& progressCallback = {},
		const ModelBenchmarkCancelCallback& isCancelled = {});

	/** Rebuilds a complete result from stored games and raw statistic contributions. **/
	ModelBenchmarkResult SummarizeModelBenchmarkData(
		const ModelBenchmarkConfig& config,
		const std::vector<ModelBenchmarkGameMetadata>& games,
		const std::vector<ModelBenchmarkStatTotalsRow>& statTotals = {},
		bool statTotalsComplete = true);

	namespace ModelBenchmarkMath {
		/** Converts a two-sided confidence value to a normal-distribution z score. **/
		double ConfidenceToZ(double confidence);

		/** Converts a win probability to an Elo delta. **/
		double EloFromWinProbability(double winProbability);

		/** Returns true when each adjacent ranking gap clears the requested confidence. **/
		bool IsFullRankingKnown(const std::vector<double>& elo, const std::vector<std::vector<double>>& covariance, double confidence);
	}
}
