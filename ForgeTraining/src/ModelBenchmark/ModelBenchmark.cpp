#include "ModelBenchmark.h"

#include "ModelBenchmarkInternal.h"
#include "ModelBenchmarkLiveGames.h"
#include "ModelBenchmarkEngine.h"

#include <GigaLearnInfer/AgentRegistry.h>
#include <GigaLearnInfer/AgentSetupSnapshot.h>
#include <TrainingSetup/TrainingSetupSnapshot.h>
#include <Inference/InferenceModelSource.h>
#include <fstream>
#include <limits>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <filesystem>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <typeinfo>

#ifdef RG_CUDA_SUPPORT
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime_api.h>
#endif

using namespace RocketForge;
using namespace RLGC;
using namespace RocketSim;
using namespace RocketForgeTraining::ModelBenchmarkInternal;

namespace RocketForgeTraining {
static std::map<int, std::string> PredictionRevisions(const ModelBenchmarkConfig& config) {
	std::map<int, std::string> revisions;
	revisions[static_cast<int>(config.engine)] = GetEngineInfo(config.engine).revision;
	for (int engine : config.modelPredictionEngines)
		if (engine != 0)
			revisions[engine] = GetEngineInfo(EngineFromVersion(engine)).revision;

	return revisions;
}

static void RecordMatchEnvironment(ModelBenchmarkResult& result, const ModelBenchmarkConfig& config) {
	result.rocketSimVersions = { static_cast<int>(config.engine) };
	RecordModelBenchmarkEnvironment(
		result,
		config,
		ModelBenchmarkEngineFamily(config.engine),
		GetEngineInfo(config.engine).revision,
		ModelBenchmarkEngineCarOrder(config.engine),
		PredictionRevisions(config),
		ArenaConfig{}.noBallRot);
}

ModelBenchmarkConfig PrepareModelBenchmarkPrediction(const ModelBenchmarkConfig& input) {
	ModelBenchmarkConfig config = input;
	config.modelPredictionEngines.clear();
	for (ModelBenchmarkModel& model : config.models) {
		std::unique_ptr<Agent> agent = ModelBenchmarkInternal::CreateBenchmarkAgent(model);
		config.modelPredictionEngines.push_back(agent->GetBallPredictionEngineVersion());
		model.agentSetupJson = AgentSetupSnapshotToJson(CaptureAgentSetupSnapshot(*agent)).dump();
	}

	return config;
}

ModelBenchmarkResult DescribeModelBenchmarkEnvironment(const ModelBenchmarkConfig& input) {
	const ModelBenchmarkConfig config = PrepareModelBenchmarkPrediction(input);
	ModelBenchmarkResult result;
	RecordMatchEnvironment(result, config);

	return result;
}

static void RequirePreparedModelBenchmarkEngines(const ModelBenchmarkConfig& prepared) {
	RequireEngine(prepared.engine);
	for (int engine : prepared.modelPredictionEngines)
		RequireEngine(EngineFromVersion(engine));
}

void RequireModelBenchmarkEngines(const ModelBenchmarkConfig& config) {
	RequirePreparedModelBenchmarkEngines(PrepareModelBenchmarkPrediction(config));
}

int ModelBenchmarkPairGameTarget(const ModelBenchmarkConfig& config, int modelA, int modelB, int mode)
{
	if (!config.usePairBudgets)
		return std::max(1, config.maxGamesPerMode);

	for (const ModelBenchmarkPairBudget& pair : config.pairBudgets) {
		const bool samePair = (pair.modelA == modelA && pair.modelB == modelB) || (pair.modelA == modelB && pair.modelB == modelA);
		if (pair.mode == mode && samePair)
			return std::max(0, pair.games);
	}

	return 0;
}

namespace {
	ModelBenchmarkGpuTelemetry SampleBenchmarkGpuTelemetry() noexcept
	{
		ModelBenchmarkGpuTelemetry telemetry = {};
#ifdef RG_CUDA_SUPPORT
		try {
			int deviceIndex = 0;
			if (cudaGetDevice(&deviceIndex) != cudaSuccess) {
				cudaGetLastError();
				return telemetry;
			}

			const auto allocatorStats =
				c10::cuda::CUDACachingAllocator::getDeviceStats(deviceIndex);
			constexpr size_t aggregate =
				(size_t)c10::CachingAllocator::StatType::AGGREGATE;
			telemetry.allocatedBytes = allocatorStats.allocated_bytes[aggregate].current;
			telemetry.reservedBytes = allocatorStats.reserved_bytes[aggregate].current;
			telemetry.peakAllocatedBytes = allocatorStats.allocated_bytes[aggregate].peak;
			telemetry.peakReservedBytes = allocatorStats.reserved_bytes[aggregate].peak;
			telemetry.allocationRetries = allocatorStats.num_alloc_retries;
			telemetry.allocatorOoms = allocatorStats.num_ooms;

			size_t freeBytes = 0;
			size_t totalBytes = 0;
			if (cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess) {
				telemetry.freeBytes = (int64_t)freeBytes;
				telemetry.totalBytes = (int64_t)totalBytes;
			} else {
				cudaGetLastError();
			}

			telemetry.available = true;
		} catch (...) {
		}
#endif
		return telemetry;
	}

	void ApplyCudaGraphTelemetry(
		const BenchmarkRunState& state,
		ModelBenchmarkGpuTelemetry& telemetry) noexcept
	{
		telemetry.graphEntries = state.cudaGraphEntries;
		telemetry.graphCaptures = state.cudaGraphCaptures;
		telemetry.graphReplays = state.cudaGraphReplays;
		telemetry.graphFallbacks = state.cudaGraphFallbacks;
		telemetry.graphVariantLimitFallbacks = state.cudaGraphVariantLimitFallbacks;
		telemetry.graphCaptureFailures = state.cudaGraphCaptureFailures;
	}

	void ResetBenchmarkGpuTelemetry() noexcept
	{
		try {
#ifdef RG_CUDA_SUPPORT
			int deviceIndex = 0;
			if (cudaGetDevice(&deviceIndex) == cudaSuccess) {
				c10::cuda::CUDACachingAllocator::resetAccumulatedStats(deviceIndex);
				c10::cuda::CUDACachingAllocator::resetPeakStats(deviceIndex);
			} else {
				cudaGetLastError();
			}
#endif
		} catch (...) {
		}
	}

	uint64_t ResolveBenchmarkSeed(uint64_t configuredSeed)
	{
		if (configuredSeed != 0)
			return configuredSeed;

		std::random_device randomDevice;
		const uint64_t randomBits =
			((uint64_t)randomDevice() << 32)
			^ (uint64_t)randomDevice();
		const uint64_t clockBits = (uint64_t)std::chrono::high_resolution_clock::now()
			.time_since_epoch()
			.count();
		const uint64_t resolvedSeed = randomBits ^ clockBits;
		return resolvedSeed != 0 ? resolvedSeed : 1;
	}

	int ResolveBenchmarkMaxTeamSize(const Agent& agent)
	{
		return std::clamp(
			std::min(agent.maxTeamSize, agent.GetMaxBenchmarkTeamSize()),
			1,
			3);
	}

	std::string MakeBenchmarkingLabel(const std::vector<int>& modes) {
		std::ostringstream out;
		out << "Benchmarking ";
		for (int i = 0; i < (int)modes.size(); i++) {
			if (i)
				out << ", ";
			out << ModeName(modes[i]);
		}
		return out.str();
	}

	BenchmarkRunState MakeInitialRunState(
		const ModelBenchmarkConfig& config,
		int numModels,
		const std::vector<int>& modes,
		const std::vector<PairKey>& eligiblePairModes,
		int maxGamesPerMode) {
		BenchmarkRunState state = {};
		state.stats.resize(numModels);
		for (int mode : modes)
			state.statsByMode[mode].resize(numModels);
		state.stopReason = config.runUntilMaxGames
			? "Max games per mode reached"
			: "Terminated early: max games per agent per mode reached before benchmark confidence or Elo uncertainty target was reached.";
		state.targetGamesPerPairMode = maxGamesPerMode;
		state.nextSnapshotId = config.snapshotOffset + 1;

		for (int mode : modes)
			state.gamesByMode[mode] = 0;
		for (int mode : modes)
			state.scheduledGamesByMode[mode] = 0;
		for (int mode : modes)
			state.stepsByMode[mode] = 0;
		for (int mode : modes)
			state.processedPhysicsTicksByMode[mode] = 0;
		for (int mode : modes)
			state.completedPhysicsTicksByMode[mode] = 0;
		for (const PairKey& pairMode : eligiblePairModes) {
			state.scheduledByPairMode[pairMode] = 0;
			state.targetByPairMode[pairMode] = ModelBenchmarkPairGameTarget(config, pairMode.a, pairMode.b, pairMode.mode);
		}
		return state;
	}

	void PopulateOptionalStats(
		ModelBenchmarkResult& result,
		const BenchmarkRunState& state,
		const std::vector<int>& modes,
		const std::vector<std::string>& modelNames) {
		for (int modelIdx = 0; modelIdx < (int)modelNames.size(); modelIdx++) {
			result.optionalStats.push_back({
				modelIdx,
				modelNames[modelIdx],
				FinalizeStats(state.stats[modelIdx])
			});
		}

		for (int mode : modes) {
			const auto modeItr = state.statsByMode.find(mode);
			if (modeItr == state.statsByMode.end())
				continue;
			auto& rows = result.optionalStatsByMode[mode];
			for (int modelIdx = 0; modelIdx < (int)modelNames.size(); modelIdx++) {
				if (modeItr->second[modelIdx].gameTimeSum <= 0.0)
					continue;
				rows.push_back({
					modelIdx,
					modelNames[modelIdx],
					FinalizeStats(modeItr->second[modelIdx])
				});
			}
		}
	}

	void PopulateStatTotals(ModelBenchmarkResult& result, const BenchmarkRunState& state) {
		result.statTotals.clear();
		for (const auto& [key, totalsBySide] : state.statsByPairMode) {
			result.statTotals.push_back({ key.a, key.b, key.mode, totalsBySide[0] });
			result.statTotals.push_back({ key.b, key.a, key.mode, totalsBySide[1] });
		}
	}

	std::string FormatModeDistributionSummary(const BenchmarkRunState& state, const std::vector<int>& modes) {
		if (modes.empty())
			return {};

		std::ostringstream out;
		out << "Final mode distribution:";
		for (int mode : modes)
			out << "\n  " << ModeName(mode) << " " << state.gamesByMode.at(mode);
		return out.str();
	}

	struct ModeEloFit {
		EloFit fit;
		std::vector<bool> available;
	};

	ModeEloFit FitModeElo(
		int numModels,
		const std::vector<BenchmarkOutcome>& outcomes,
		bool outcomesAreAggregated = false) {
		ModeEloFit result = {};
		result.fit.elo.assign(numModels, 0.0);
		result.fit.covariance.assign(numModels, std::vector<double>(numModels, 0.0));
		result.available.assign(numModels, false);

		std::vector<int> activeModels;
		std::vector<int> remap(numModels, -1);
		for (const BenchmarkOutcome& outcome : outcomes) {
			for (int model : { outcome.winner, outcome.loser }) {
				if (model < 0 || model >= numModels || remap[model] >= 0)
					continue;
				remap[model] = (int)activeModels.size();
				activeModels.push_back(model);
			}
		}
		if (activeModels.size() < 2)
			return result;

		std::vector<BenchmarkOutcome> remappedOutcomes;
		remappedOutcomes.reserve(outcomes.size());
		for (BenchmarkOutcome outcome : outcomes) {
			if (outcome.winner < 0 || outcome.loser < 0
				|| outcome.winner >= numModels || outcome.loser >= numModels) {
				continue;
			}
			outcome.winner = remap[outcome.winner];
			outcome.loser = remap[outcome.loser];
			// Overall Elo is pair-balanced. A per-mode fit uses every result at its
			// natural unit weight because there is no cross-mode imbalance to remove.
			// Aggregated live outcomes store their natural result count in weight.
			if (!outcomesAreAggregated)
				outcome.weight = 1.0;
			remappedOutcomes.push_back(outcome);
		}
		if (!HasConnectedOutcomeGraph((int)activeModels.size(), remappedOutcomes))
			return result;

		EloFit activeFit = MakeRelative(FitElo((int)activeModels.size(), remappedOutcomes));
		for (int activeIdx = 0; activeIdx < (int)activeModels.size(); activeIdx++) {
			const int modelIdx = activeModels[activeIdx];
			result.fit.elo[modelIdx] = activeFit.elo[activeIdx];
			result.available[modelIdx] = true;
		}
		return result;
	}

	ModelBenchmarkResult BuildLiveResultSnapshot(
		const ModelBenchmarkConfig& config,
		const BenchmarkRunState& state,
		const std::vector<int>& modes,
		const std::vector<std::pair<int, int>>& matchups,
		const std::vector<PairKey>& eligiblePairModes,
		const std::vector<std::string>& modelNames) {
		const int numModels = (int)modelNames.size();
		auto findRecord = [&](int model, int opponent, int mode, BenchmarkRecordType recordType)
			-> const ModelBenchmarkRecord& {
			static const ModelBenchmarkRecord empty = {};
			const PairKey key{ std::min(model, opponent), std::max(model, opponent), mode };
			const auto itr = state.records.find(key);
			if (itr == state.records.end())
				return empty;
			const int modelSlot = model == key.a ? 0 : 1;
			const int recordTypeIndex = recordType == BenchmarkRecordType::Kickoff
				? 1
				: (recordType == BenchmarkRecordType::PostKickoff ? 2 : 0);
			return itr->second[recordTypeIndex][modelSlot];
		};
		auto appendRecordOutcomes = [](
			std::vector<BenchmarkOutcome>& outcomes,
			const ModelBenchmarkRecord& record,
			int model,
			int opponent,
			int mode,
			ModelBenchmarkGoalTiming goalTiming,
			double unitWeight) {
			if (record.wins > 0)
				outcomes.push_back({ model, opponent, mode, goalTiming, unitWeight * record.wins });
			if (record.losses > 0)
				outcomes.push_back({ opponent, model, mode, goalTiming, unitWeight * record.losses });
		};

		// Collapse identical results into weighted outcomes. Elo fitting then scales
		// with pair-mode count instead of the total number of completed games.
		std::vector<BenchmarkOutcome> overallOutcomes;
		std::vector<BenchmarkOutcome> kickoffOutcomes;
		std::vector<BenchmarkOutcome> postKickoffOutcomes;
		std::map<int, std::vector<BenchmarkOutcome>> modeOutcomes;
		std::vector<int> kickoffGamesByModel(numModels, 0);
		std::vector<int> postKickoffGamesByModel(numModels, 0);
		int kickoffGoalCount = 0;
		int postKickoffGoalCount = 0;
		for (const PairKey& pairMode : eligiblePairModes) {
			const ModelBenchmarkRecord& overall = findRecord(
				pairMode.a,
				pairMode.b,
				pairMode.mode,
				BenchmarkRecordType::Overall);
			const ModelBenchmarkRecord& kickoff = findRecord(
				pairMode.a,
				pairMode.b,
				pairMode.mode,
				BenchmarkRecordType::Kickoff);
			const ModelBenchmarkRecord& postKickoff = findRecord(
				pairMode.a,
				pairMode.b,
				pairMode.mode,
				BenchmarkRecordType::PostKickoff);
			const double overallUnitWeight = OverallOutcomeWeightForPair(
				eligiblePairModes,
				pairMode.a,
				pairMode.b);
			appendRecordOutcomes(
				overallOutcomes,
				overall,
				pairMode.a,
				pairMode.b,
				pairMode.mode,
				ModelBenchmarkGoalTiming::NoGoal,
				overallUnitWeight);
			appendRecordOutcomes(
				modeOutcomes[pairMode.mode],
				overall,
				pairMode.a,
				pairMode.b,
				pairMode.mode,
				ModelBenchmarkGoalTiming::NoGoal,
				1.0);
			appendRecordOutcomes(
				kickoffOutcomes,
				kickoff,
				pairMode.a,
				pairMode.b,
				pairMode.mode,
				ModelBenchmarkGoalTiming::Kickoff,
				overallUnitWeight);
			appendRecordOutcomes(
				postKickoffOutcomes,
				postKickoff,
				pairMode.a,
				pairMode.b,
				pairMode.mode,
				ModelBenchmarkGoalTiming::PostKickoff,
				overallUnitWeight);

			const int kickoffGames = kickoff.wins + kickoff.losses;
			const int postKickoffGames = postKickoff.wins + postKickoff.losses;
			kickoffGoalCount += kickoffGames;
			postKickoffGoalCount += postKickoffGames;
			kickoffGamesByModel[pairMode.a] += kickoffGames;
			kickoffGamesByModel[pairMode.b] += kickoffGames;
			postKickoffGamesByModel[pairMode.a] += postKickoffGames;
			postKickoffGamesByModel[pairMode.b] += postKickoffGames;
		}

		const bool overallEloAvailable = HasConnectedOutcomeGraph(numModels, overallOutcomes);
		EloFit combinedFit = FitElo(numModels, overallOutcomes);
		EloFit kickoffFit = FitElo(numModels, kickoffOutcomes);
		EloFit postKickoffFit = FitElo(numModels, postKickoffOutcomes);
		AnchorFitsToOverallFloor(combinedFit, kickoffFit, postKickoffFit);
		const bool kickoffEloAvailable = HasConnectedOutcomeGraph(numModels, kickoffOutcomes);
		const bool postKickoffEloAvailable = HasConnectedOutcomeGraph(numModels, postKickoffOutcomes);
		std::map<int, ModeEloFit> modeFits;
		for (int mode : modes)
			modeFits[mode] = FitModeElo(numModels, modeOutcomes[mode], true);

		const std::vector<int> order = overallEloAvailable
			? BuildRankOrderFromFit(numModels, combinedFit, modelNames)
			: BuildRankOrder(numModels, overallOutcomes, modelNames);
		const double z = ModelBenchmarkMath::ConfidenceToZ(config.eloRangeConfidence);
		auto pairModeEligible = [&](const std::pair<int, int>& matchup, int mode) {
			const PairKey key = {
				std::min(matchup.first, matchup.second),
				std::max(matchup.first, matchup.second),
				mode
			};
			return std::find(eligiblePairModes.begin(), eligiblePairModes.end(), key) != eligiblePairModes.end();
		};

		// The progress publisher attaches raw games and statistics to this summary
		// so the GUI can recompute combined saved and live results.
		ModelBenchmarkResult result = {};
		result.scoredGames = state.scoredGames;
		result.noResultGames = state.noResultGames;
		result.kickoffGoalWindowSeconds = config.kickoffGoalWindowSeconds;
		result.stochasticKickoff = config.stochasticKickoff;
		result.kickoffTemperature = config.kickoffTemperature;
		result.eloRangeConfidence = config.eloRangeConfidence;
		result.kickoffGoalCount = kickoffGoalCount;
		result.postKickoffGoalCount = postKickoffGoalCount;
		result.kickoffEloAvailable = kickoffEloAvailable;
		result.postKickoffEloAvailable = postKickoffEloAvailable;
		result.overallEloAvailable = overallEloAvailable;
		result.has1v1 = std::find(modes.begin(), modes.end(), 1) != modes.end();
		result.has2v2 = std::find(modes.begin(), modes.end(), 2) != modes.end();
		result.has3v3 = std::find(modes.begin(), modes.end(), 3) != modes.end();
		result.modelNames = modelNames;
		for (const ModelBenchmarkModel& model : config.models) {
			result.modelVersionIds.push_back(model.versionId);
			result.modelCheckpointPaths.push_back(model.checkpointPath);
		}

		for (int rank = 0; rank < (int)order.size(); rank++) {
			const int modelIdx = order[rank];
			ModelBenchmarkRecord total = {};
			for (const auto& matchup : matchups) {
				if (matchup.first != modelIdx && matchup.second != modelIdx)
					continue;
				const int opponent = matchup.first == modelIdx ? matchup.second : matchup.first;
				for (int mode : modes) {
					if (!pairModeEligible(matchup, mode))
						continue;
					const ModelBenchmarkRecord& overall = findRecord(
						modelIdx,
						opponent,
						mode,
						BenchmarkRecordType::Overall);
					total.wins += overall.wins;
					total.losses += overall.losses;
					total.noResults += overall.noResults;
				}
			}

			const double se =
				std::sqrt(std::max(1e-6, combinedFit.covariance[modelIdx][modelIdx]));
			ModelBenchmarkRankingRow row = {};
			row.rank = rank + 1;
			row.modelIndex = modelIdx;
			row.modelName = modelNames[modelIdx];
			row.relativeElo = combinedFit.elo[modelIdx];
			row.elo1v1 = modeFits.count(1) ? modeFits[1].fit.elo[modelIdx] : 0.0;
			row.elo2v2 = modeFits.count(2) ? modeFits[2].fit.elo[modelIdx] : 0.0;
			row.elo3v3 = modeFits.count(3) ? modeFits[3].fit.elo[modelIdx] : 0.0;
			row.rangeLow = combinedFit.elo[modelIdx] - z * se;
			row.rangeHigh = combinedFit.elo[modelIdx] + z * se;
			row.betterThanSummary = overallEloAvailable
				? FormatConfidenceList(combinedFit, modelNames, modelIdx)
				: "N/A";
			row.totalRecord = total;
			row.elo1v1Available = modeFits.count(1) && modeFits[1].available[modelIdx];
			row.elo2v2Available = modeFits.count(2) && modeFits[2].available[modelIdx];
			row.elo3v3Available = modeFits.count(3) && modeFits[3].available[modelIdx];
			result.rankings.push_back(std::move(row));
		}

		for (int modelIdx = 0; modelIdx < numModels; modelIdx++) {
			const double kickoffSe =
				std::sqrt(std::max(1e-6, kickoffFit.covariance[modelIdx][modelIdx]));
			const double postKickoffSe =
				std::sqrt(std::max(1e-6, postKickoffFit.covariance[modelIdx][modelIdx]));
			ModelBenchmarkBreakdownRow row = {};
			row.modelIndex = modelIdx;
			row.modelName = modelNames[modelIdx];
			row.combinedElo = combinedFit.elo[modelIdx];
			row.kickoffElo = kickoffFit.elo[modelIdx];
			row.kickoffRangeLow = row.kickoffElo - z * kickoffSe;
			row.kickoffRangeHigh = row.kickoffElo + z * kickoffSe;
			row.postKickoffElo = postKickoffFit.elo[modelIdx];
			row.postKickoffRangeLow = row.postKickoffElo - z * postKickoffSe;
			row.postKickoffRangeHigh = row.postKickoffElo + z * postKickoffSe;
			row.kickoffDelta = row.kickoffElo - row.postKickoffElo;
			row.kickoffGames = kickoffGamesByModel[modelIdx];
			row.postKickoffGames = postKickoffGamesByModel[modelIdx];
			row.kickoffEloAvailable = kickoffEloAvailable;
			row.postKickoffEloAvailable = postKickoffEloAvailable;
			result.breakdown.push_back(std::move(row));
		}

		for (const auto& matchup : matchups) {
			for (int side = 0; side < 2; side++) {
				const int modelIdx = side == 0 ? matchup.first : matchup.second;
				const int opponent = side == 0 ? matchup.second : matchup.first;
				for (int mode : modes) {
					if (!pairModeEligible(matchup, mode))
						continue;
					ModelBenchmarkPairwiseRow row = {};
					row.modelIndex = modelIdx;
					row.opponentIndex = opponent;
					row.modelName = modelNames[modelIdx];
					row.opponentName = modelNames[opponent];
					row.modeName = ModeName(mode);
					row.modeElo = modeFits.count(mode)
						&& modeFits[mode].available[modelIdx]
						&& modeFits[mode].available[opponent]
						? modeFits[mode].fit.elo[modelIdx] - modeFits[mode].fit.elo[opponent]
						: 0.0;
					row.combined = findRecord(
						modelIdx,
						opponent,
						mode,
						BenchmarkRecordType::Overall);
					row.kickoff = findRecord(
						modelIdx,
						opponent,
						mode,
						BenchmarkRecordType::Kickoff);
					row.postKickoff = findRecord(
						modelIdx,
						opponent,
						mode,
						BenchmarkRecordType::PostKickoff);
					const int kickoffScored = row.kickoff.wins + row.kickoff.losses;
					const int postKickoffScored = row.postKickoff.wins + row.postKickoff.losses;
					row.timingDeltaAvailable = kickoffScored > 0 && postKickoffScored > 0;
					if (row.timingDeltaAvailable) {
						row.kickoffVsPostKickoffDeltaPercent =
							(WinPct(row.kickoff) - WinPct(row.postKickoff)) * 100.0;
					}
					result.pairwise.push_back(std::move(row));
				}
			}
		}

		if (config.collectOptionalStats)
			PopulateOptionalStats(result, state, modes, modelNames);
		return result;
	}
}

std::unique_ptr<Agent> ModelBenchmarkInternal::CreateBenchmarkAgent(const ModelBenchmarkModel& model)
{
	if (!model.agentSetupJson.empty()) {
		const std::optional<AgentSetupSnapshot> setup = AgentSetupSnapshotFromJson(nlohmann::json::parse(model.agentSetupJson));
		if (!setup)
			throw std::runtime_error("Invalid captured benchmark agent setup");

		std::unique_ptr<Agent> agent = AgentRegistry::Create(model.agentName);
		if (!agent)
			throw std::runtime_error("Unknown benchmark agent: " + model.agentName);

		ApplyAgentSetupSnapshot(*agent, *setup);
		return agent;
	}

	InferenceModelSource source;
	source.agentName = model.agentName;
	source.checkpointFolder = model.checkpointPath;

	const bool isSavedSegment = model.agentSetupSource == ModelBenchmarkAgentSetupSource::SavedSegment;

	// Only SavedSegment models apply a saved setup snapshot;
	// Current uses the complete compiled agent setup, including its predictor profile.
	if (isSavedSegment) {
		if (model.trainingSetupPath.empty())
			throw std::runtime_error("ModelBenchmark: missing saved Agent setup path for " + model.displayName + ".");
		source.setupSnapshotPath = model.trainingSetupPath;
	}

	ResolvedInferenceAgent resolved = ResolveInferenceAgent(source);
	if (isSavedSegment && !resolved.appliedSnapshot.has_value())
		throw std::runtime_error("ModelBenchmark: missing saved Agent setup metadata for " + model.displayName + ".");

	return std::move(resolved.agent);
}

std::set<int> ResolveModelBenchmarkSimulatorVersions(const std::set<int>& versions, int legacyEngineVersion) {
	// Explicit simulator metadata takes precedence over the pre-versioned v2 assumption.
	if (versions.empty() || versions == std::set<int>{0}) {
		if (legacyEngineVersion == -1)
			return { 2, 3 };
		if (legacyEngineVersion > 0)
			return { legacyEngineVersion };
	}

	std::set<int> resolved;
	for (int version : versions)
		resolved.insert(version > 0 ? version : 2);
	if (resolved.empty())
		resolved.insert(2);

	return resolved;
}

std::string ModelBenchmarkRocketSimVersionsText(const std::set<int>& versions) {
	std::string text;
	for (int version : ResolveModelBenchmarkSimulatorVersions(versions)) {
		if (!text.empty())
			text += ", ";
		text += "v" + std::to_string(version);
	}
	return text;
}

std::vector<std::pair<int, int>> BuildModelBenchmarkMatchups(const std::vector<ModelBenchmarkModel>& models) {
	std::vector<std::pair<int, int>> matchups;
	matchups.reserve(models.size() * (models.size() - 1) / 2);
	for (int a = 0; a < (int)models.size(); a++) {
		for (int b = a + 1; b < (int)models.size(); b++) {
			if (models[a].primary || models[b].primary)
				matchups.push_back({ a, b });
		}
	}
	return matchups;
}

ModelBenchmarkResult SummarizeModelBenchmarkData(
	const ModelBenchmarkConfig& config,
	const std::vector<ModelBenchmarkGameMetadata>& games,
	const std::vector<ModelBenchmarkStatTotalsRow>& statTotals,
	bool statTotalsComplete) {
	if (config.models.size() < 2)
		throw std::runtime_error("ModelBenchmark: select at least two saved models.");

	const int numModels = (int)config.models.size();
	std::vector<std::string> modelNames;
	modelNames.reserve(config.models.size());
	for (const ModelBenchmarkModel& model : config.models)
		modelNames.push_back(model.displayName);

	const std::vector<std::pair<int, int>> matchups = BuildModelBenchmarkMatchups(config.models);
	std::set<std::pair<int, int>> allowedPairs;
	for (const auto& matchup : matchups)
		allowedPairs.insert(matchup);

	auto modeEnabled = [&](int mode) {
		return (mode == 1 && config.run1v1)
			|| (mode == 2 && config.run2v2)
			|| (mode == 3 && config.run3v3);
	};
	auto validPair = [&](int a, int b) {
		if (a < 0 || b < 0 || a >= numModels || b >= numModels || a == b)
			return false;
		return allowedPairs.count({ std::min(a, b), std::max(a, b) }) > 0;
	};

	BenchmarkRunState state = {};
	state.stats.resize(numModels);
	std::set<PairKey> pairModes;
	std::set<int> modeSet;
	for (const ModelBenchmarkGameMetadata& game : games) {
		if (!modeEnabled(game.mode) || !validPair(game.modelA, game.modelB))
			continue;
		state.games.push_back(game);
		modeSet.insert(game.mode);
		if (game.goalScored)
			state.scoredGames++;
		else
			state.noResultGames++;
		if (!game.mirrorCompleted)
			continue;
		const PairKey key{ std::min(game.modelA, game.modelB), std::max(game.modelA, game.modelB), game.mode };
		pairModes.insert(key);
		if (game.goalScored && game.includedInRating
			&& (game.winner == game.modelA || game.winner == game.modelB)) {
			const int loser = game.winner == game.modelA ? game.modelB : game.modelA;
			AddResultToRecords(state.records, game.winner, loser, game.mode, game.goalTiming);
		} else if (!game.goalScored) {
			AddNoResultToRecords(state.records, game.modelA, game.modelB, game.mode);
		}
	}

	for (const ModelBenchmarkStatTotalsRow& row : statTotals) {
		if (!modeEnabled(row.mode) || !validPair(row.modelIndex, row.opponentIndex))
			continue;
		modeSet.insert(row.mode);
		state.statsByMode[row.mode].resize(numModels);
		MergeRunningStats(state.stats[row.modelIndex], row.totals);
		MergeRunningStats(state.statsByMode[row.mode][row.modelIndex], row.totals);
		const PairKey key{
			std::min(row.modelIndex, row.opponentIndex),
			std::max(row.modelIndex, row.opponentIndex),
			row.mode
		};
		const int side = row.modelIndex == key.a ? 0 : 1;
		MergeRunningStats(state.statsByPairMode[key][side], row.totals);
	}

	std::vector<int> modes(modeSet.begin(), modeSet.end());
	std::vector<PairKey> eligiblePairModes(pairModes.begin(), pairModes.end());
	ModelBenchmarkResult result = BuildLiveResultSnapshot(
		config,
		state,
		modes,
		matchups,
		eligiblePairModes,
		modelNames);
	result.agentName = "Multiple";
	result.stageName = "Multiple";
	result.stopReason = "Aggregated saved benchmark data";
	result.benchmarkStrategy = "Saved benchmark aggregation";
	result.timingSummary = "Combined from saved benchmark runs.";
	result.games = state.games;
	result.statTotalsComplete = statTotalsComplete;
	PopulateStatTotals(result, state);
	return result;
}

ModelBenchmarkEligibilityAnalysis AnalyzeModelBenchmarkEligibility(const ModelBenchmarkConfig& config) {
	std::vector<int> selectedModes;
	if (config.run1v1) selectedModes.push_back(1);
	if (config.run2v2) selectedModes.push_back(2);
	if (config.run3v3) selectedModes.push_back(3);
	if (selectedModes.empty())
		throw std::runtime_error("ModelBenchmark: select at least one supported benchmark mode.");

	std::vector<int> modelMaxTeamSizes;
	modelMaxTeamSizes.reserve(config.models.size());
	for (const ModelBenchmarkModel& model : config.models) {
		std::unique_ptr<Agent> agent = CreateBenchmarkAgent(model);
		modelMaxTeamSizes.push_back(ResolveBenchmarkMaxTeamSize(*agent));
	}
	return ModelBenchmarkInternal::AnalyzeModelBenchmarkEligibility(
		config.models,
		modelMaxTeamSizes,
		selectedModes);
}

ModelBenchmarkResult RunModelBenchmark(
	const ModelBenchmarkConfig& inputConfig,
	const ModelBenchmarkProgressCallback& progressCallback,
	const ModelBenchmarkCancelCallback& isCancelled) {
	RequireEngine(inputConfig.engine);
	ModelBenchmarkConfig config = inputConfig;
	const auto benchmarkStart = std::chrono::steady_clock::now();
	// NOTE: This guard must outlive loadedModels so model weights are destroyed
	// before graph storage and the CUDA allocator cache are released.
	BenchmarkCudaCleanupGuard cudaCleanupGuard;
	auto elapsedSeconds = [&]() {
		return std::chrono::duration<double>(std::chrono::steady_clock::now() - benchmarkStart).count();
	};
	const uint64_t runSeed = ResolveBenchmarkSeed(config.seed);

	auto throwIfCancelled = [&]() {
		if (isCancelled && isCancelled())
			throw std::runtime_error("ModelBenchmark: cancelled.");
	};

	auto reportStartupProgress = [&](const std::string& label, const std::string& details = {}) {
		if (progressCallback) {
			ModelBenchmarkProgress progress = {};
			progress.elapsedSeconds = elapsedSeconds();
			progress.label = label;
			progress.details = details;
			progress.totalArenas = std::max(1, config.numArenas);
			progressCallback(progress);
		}
	};

	reportStartupProgress(
		"Starting benchmark",
		"Validating selected benchmark models.\nRun seed: " + std::to_string(runSeed));
	throwIfCancelled();

	if (config.models.size() < 2)
		throw std::runtime_error("ModelBenchmark: select at least two model versions.");
	if (!std::isfinite(config.kickoffGoalWindowSeconds) || config.kickoffGoalWindowSeconds <= 0.0f)
		throw std::runtime_error("ModelBenchmark: kickoff goal window must be a positive number of seconds.");

	if (config.stochasticKickoff
		&& (!std::isfinite(config.kickoffTemperature) || config.kickoffTemperature <= 0.0f))
		throw std::runtime_error("ModelBenchmark: kickoff temperature must be positive and finite.");

	std::set<std::string> names;
	for (const ModelBenchmarkModel& model : config.models) {
		if (model.displayName.empty())
			throw std::runtime_error("ModelBenchmark: selected models must have display names.");
		if (model.agentName.empty() || model.versionId.empty())
			throw std::runtime_error("ModelBenchmark: each selected model must include agent and version.");
		if (!names.insert(model.displayName).second)
			throw std::runtime_error("ModelBenchmark: duplicate model name: " + model.displayName);
		if (model.checkpointPath.empty() || !std::filesystem::exists(model.checkpointPath))
			throw std::runtime_error("ModelBenchmark: missing checkpoint for " + model.displayName + ": " + model.checkpointPath.string());
	}
	throwIfCancelled();

	config = PrepareModelBenchmarkPrediction(config);
	RequirePreparedModelBenchmarkEngines(config);

	const int numModels = (int)config.models.size();
	const int numArenas = std::max(1, config.numArenas);
	const int maxGamesPerMode = std::max(1, config.maxGamesPerMode);

	std::vector<LoadedModel> loadedModels;
	loadedModels.reserve(config.models.size());
	int sharedTickSkip = -1;
	int sharedActionDelay = -1;
	bool sharedTiming = true;
	for (int modelLoadIdx = 0; modelLoadIdx < (int)config.models.size(); modelLoadIdx++) {
		const ModelBenchmarkModel& model = config.models[modelLoadIdx];
		reportStartupProgress(
			"Loading benchmark model",
			"[" + std::to_string(modelLoadIdx + 1) + "/" + std::to_string(config.models.size()) + "] " +
			model.displayName + " from " + model.checkpointPath.string());
		throwIfCancelled();

		std::unique_ptr<Agent> agent = CreateBenchmarkAgent(model);
		throwIfCancelled();

		const int modelTickSkip = std::max(1, agent->tickSkip);
		const int modelActionDelay = std::clamp(agent->actionDelay, 0, modelTickSkip);
		if (sharedTickSkip < 0) {
			sharedTickSkip = modelTickSkip;
			sharedActionDelay = modelActionDelay;
		} else if (sharedTickSkip != modelTickSkip || sharedActionDelay != modelActionDelay) {
			sharedTiming = false;
		}

		agent->modelPath = model.checkpointPath;
		reportStartupProgress("Creating model components", model.displayName + ": observation builder and action parser.");
		auto obsBuilder = std::unique_ptr<ObsBuilder>(agent->MakeObsBuilder());
		auto actionParser = std::unique_ptr<ActionParser>(agent->MakeActionParser());
		throwIfCancelled();

		LoadedModel loaded = {};
		loaded.name = model.displayName;
		loaded.agentName = agent->agentName.empty() ? model.agentName : agent->agentName;
		loaded.checkpointPath = model.checkpointPath;
		loaded.carConfig = agent->carConfig;
		loaded.policyTemperature = 1.0f;
		loaded.tickSkip = modelTickSkip;
		loaded.actionDelay = modelActionDelay;
		loaded.maxTeamSize = std::clamp(agent->maxTeamSize, 1, 3);
		loaded.obsSize = agent->GetObsSize();
		loaded.numActions = actionParser->GetActionAmount();
		loaded.usesActionMask = actionParser->UsesActionMask();
		loaded.obsBuilderType = agent->obsBuilderName + ":" + typeid(*obsBuilder).name();
		loaded.actionParserType = typeid(*actionParser).name();
		loaded.agent = std::move(agent);
		loadedModels.push_back(std::move(loaded));
		reportStartupProgress("Loaded benchmark model metadata", model.displayName);
	}

	const std::vector<int> selectedModes = SelectedModes(config, 3);
	const std::vector<std::pair<int, int>> matchups = BuildModelBenchmarkMatchups(config.models);
	if (matchups.empty())
		throw std::runtime_error("ModelBenchmark: no eligible matchups; mark at least one selected model as Primary.");
	std::vector<int> modelMaxTeamSizes;
	modelMaxTeamSizes.reserve(loadedModels.size());
	for (const LoadedModel& model : loadedModels)
		modelMaxTeamSizes.push_back(ResolveBenchmarkMaxTeamSize(*model.agent));
	std::vector<PairKey> eligiblePairModes = BuildEligiblePairModes(
		matchups,
		modelMaxTeamSizes,
		selectedModes);
	std::erase_if(eligiblePairModes, [&](const PairKey& pair) {
		return ModelBenchmarkPairGameTarget(config, pair.a, pair.b, pair.mode) <= 0;
	});
	if (eligiblePairModes.empty()) {
		throw std::runtime_error(
			"ModelBenchmark: the selected models do not share any selected benchmark mode.");
	}

	if (!config.usePairBudgets && !HasConnectedPairModeGraph(numModels, eligiblePairModes)) {
		throw std::runtime_error(
			"ModelBenchmark: the selected modes do not connect every model through a direct or indirect matchup. "
			"Enable a bridging mode or remove models that cannot participate.");
	}
	std::vector<int> modes;
	for (int mode : selectedModes)
		if (CountEligiblePairsForMode(eligiblePairModes, mode) > 0)
			modes.push_back(mode);
	const std::string benchmarkingLabel = MakeBenchmarkingLabel(modes);
	const int tickSkip = std::max(1, sharedTickSkip);
	const int actionDelay = std::clamp(sharedActionDelay, 0, tickSkip);
	const int noGoalSteps = std::max(1, (int)std::ceil(config.noGoalTimeoutSeconds * 120.0f / tickSkip));
	const int noGoalTicks = std::max(1, (int)std::ceil(config.noGoalTimeoutSeconds * 120.0f));

	int64_t maxGames = 0;
	for (const PairKey& pair : eligiblePairModes)
		maxGames += ModelBenchmarkPairGameTarget(config, pair.a, pair.b, pair.mode);

	std::vector<std::string> modelNames;
	for (const LoadedModel& model : loadedModels)
		modelNames.push_back(model.name);

	BenchmarkRunState state = MakeInitialRunState(
		config,
		numModels,
		modes,
		eligiblePairModes,
		maxGamesPerMode);

	const int progressUpdateEveryGames = 100;
	std::deque<std::pair<double, int>> progressSamples;
	std::deque<std::pair<double, double>> workProgressSamples;
	double lastProgressReportElapsed = 0.0;
	double lastLiveResultElapsed = -1.0;
	int lastLiveResultFinishedGames = -1;
	ModelBenchmarkGpuTelemetry latestGpuTelemetry = {};
	double lastGpuTelemetryElapsed = -1.0;
	int64_t lastGpuTelemetryOomRecoveries = 0;
	std::chrono::steady_clock::time_point simulationStart = {};
	double completedSimulationSeconds = -1.0;

	auto simulationElapsedSeconds = [&]() {
		if (completedSimulationSeconds >= 0.0)
			return completedSimulationSeconds;
		if (simulationStart == std::chrono::steady_clock::time_point{})
			return 0.0;
		return std::chrono::duration<double>(std::chrono::steady_clock::now() - simulationStart).count();
	};

	auto estimatedRemainingSeconds = [&](double simulationElapsed) {
		if (!config.runUntilMaxGames)
			return -1.0;

		const int64_t finishedGames =
			(int64_t)state.scoredGames + (int64_t)state.noResultGames;
		const int64_t remainingGameCount = std::max<int64_t>(0, maxGames - finishedGames);
		if (remainingGameCount <= 0)
			return 0.0;
		const double cumulativeGameEta =
			finishedGames > 0 && simulationElapsed > 0.0
				? simulationElapsed * (double)remainingGameCount / (double)finishedGames
				: -1.0;

		double processedWork = 0.0;
		double observedScaleNumerator = 0.0;
		double observedScaleDenominator = 0.0;
		for (int mode : modes) {
			const double modeWork = (double)state.completedPhysicsTicksByMode[mode];
			const int completedGames = state.gamesByMode[mode];
			processedWork += (double)state.processedPhysicsTicksByMode[mode];
			if (modeWork > 0.0 && completedGames > 0) {
				observedScaleNumerator += modeWork;
				observedScaleDenominator +=
					completedGames * DefaultModeDurationWeight(mode);
			}
		}

		const double defaultDurationScale = observedScaleDenominator > 0.0
			? observedScaleNumerator / observedScaleDenominator
			: 0.0;
		double remainingWork = 0.0;
		for (int mode : modes) {
			const int64_t targetGamesPerMode = ModeGameTarget(state, mode);
			const int completedGames = state.gamesByMode[mode];
			const int64_t remainingGames =
				std::max<int64_t>(0, targetGamesPerMode - completedGames);
			if (remainingGames <= 0)
				continue;

			double estimatedGameWork = 0.0;
			if (completedGames > 0 && state.completedPhysicsTicksByMode[mode] > 0)
				estimatedGameWork =
					(double)state.completedPhysicsTicksByMode[mode] / completedGames;
			else if (defaultDurationScale > 0.0)
				estimatedGameWork =
					DefaultModeDurationWeight(mode) * defaultDurationScale;
			if (estimatedGameWork <= 0.0)
				return -1.0;

			// Remaining games includes episodes that are already active. Subtract
			// their live processed work so a long in-flight episode advances ETA
			// before it reaches the completion callback.
			const double activePartialWork =
				(double)(state.processedPhysicsTicksByMode[mode]
					- state.completedPhysicsTicksByMode[mode]);
			remainingWork += std::max(
				0.0,
				remainingGames * estimatedGameWork - activePartialWork);
		}
		if (remainingWork <= 0.0)
			return cumulativeGameEta;
		if (processedWork <= 0.0 || simulationElapsed <= 0.0)
			return -1.0;

		workProgressSamples.push_back({ simulationElapsed, processedWork });
		// Keep one sample immediately before the window so sparse, long-running
		// games still produce a usable rate when reports are more than 30s apart.
		while (workProgressSamples.size() > 2
			&& simulationElapsed - workProgressSamples[1].first > 30.0) {
			workProgressSamples.pop_front();
		}
		if (workProgressSamples.size() < 2)
			return -1.0;

		const auto& baseline = workProgressSamples.front();
		const double windowSeconds = simulationElapsed - baseline.first;
		const double windowWork = processedWork - baseline.second;
		if (windowSeconds < 5.0)
			return -1.0;
		const double recentWorkPerSecond = windowSeconds > 0.0 ? windowWork / windowSeconds : 0.0;
		if (recentWorkPerSecond <= 0.0)
			return cumulativeGameEta;
		const double workEta = remainingWork / recentWorkPerSecond;
		return cumulativeGameEta >= 0.0
			? std::max(workEta, cumulativeGameEta)
			: workEta;
	};

	auto snapshotGamesPerSecond = [&](double elapsed, int finishedGames) {
		progressSamples.push_back({ elapsed, finishedGames });
		while (progressSamples.size() > 2 && elapsed - progressSamples[1].first > 10.0)
			progressSamples.pop_front();

		const auto* baseline = &progressSamples.front();
		for (const auto& sample : progressSamples) {
			if (elapsed - sample.first >= 5.0)
				baseline = &sample;
			else
				break;
		}

		const double windowSeconds = elapsed - baseline->first;
		const int windowGames = finishedGames - baseline->second;
		if (windowSeconds <= 0.0 || windowGames <= 0)
			return 0.0;
		return windowGames / windowSeconds;
	};

	ModelBenchmarkLiveGameDeltas liveGames;
	auto reportProgress = [&](const std::string& label, const std::string& details = {}, const std::string& estimateSummary = {}) {
		if (progressCallback) {
			const int finishedGames = state.scoredGames + state.noResultGames;
			const double elapsed = elapsedSeconds();
			const double simulationElapsed = simulationElapsedSeconds();
			lastProgressReportElapsed = elapsed;
			int scheduledGames = 0;
			for (const auto& mode : state.scheduledGamesByMode)
				scheduledGames += mode.second;

			const bool forceGpuSample = !latestGpuTelemetry.available
				|| simulationElapsed - lastGpuTelemetryElapsed >= 5.0
				|| state.cudaOomRecoveries != lastGpuTelemetryOomRecoveries
				|| label != benchmarkingLabel;
			if (forceGpuSample) {
				latestGpuTelemetry = SampleBenchmarkGpuTelemetry();
				lastGpuTelemetryElapsed = simulationElapsed;
				lastGpuTelemetryOomRecoveries = state.cudaOomRecoveries;
			}
			// Graphs execute inside GigaLearn.dll, which has its own statics because
			// GigaLearnInfer is a static library. The runner forwards those counters.
			ApplyCudaGraphTelemetry(state, latestGpuTelemetry);

			ModelBenchmarkProgress progress = {};
			progress.scoredGames = state.scoredGames;
			progress.noResultGames = state.noResultGames;
			progress.maxGames = maxGames;
			progress.elapsedSeconds = elapsed;
			progress.snapshotGamesPerSecond =
				snapshotGamesPerSecond(simulationElapsed, finishedGames);
			progress.label = label;
			progress.details = details;
			progress.estimateSummary = estimateSummary;
			progress.estimatedRemainingSeconds = estimatedRemainingSeconds(simulationElapsed);
			progress.activeArenas = state.activeArenas;
			progress.totalArenas = state.totalArenas > 0 ? state.totalArenas : numArenas;
			progress.pendingGames = state.pendingGames;
			progress.scheduledGames = scheduledGames;
			progress.arenaCreates = state.arenaCreates;
			progress.arenaRebuilds = state.arenaRebuilds;
			progress.arenaReuses = state.arenaReuses;
			progress.active1v1Arenas = state.active1v1Arenas;
			progress.active2v2Arenas = state.active2v2Arenas;
			progress.active3v3Arenas = state.active3v3Arenas;
			progress.cudaOomRecoveries = state.cudaOomRecoveries;
			progress.cudaGraphsDisabledAfterOom = state.cudaGraphsDisabledAfterOom;
			progress.gpu = latestGpuTelemetry;
			if (config.emitLiveResultSnapshots
				&& completedSimulationSeconds < 0.0
				&& finishedGames > 0
				&& finishedGames != lastLiveResultFinishedGames
				&& (lastLiveResultElapsed < 0.0 || simulationElapsed - lastLiveResultElapsed >= 0.5)) {
				const auto snapshotStart = std::chrono::steady_clock::now();
				auto liveResult = std::make_shared<ModelBenchmarkResult>(BuildLiveResultSnapshot(
					config,
					state,
					modes,
					matchups,
					eligiblePairModes,
					modelNames));
				liveResult->seed = runSeed;
				liveResult->maxGames = maxGames;
				liveResult->elapsedSeconds = elapsed;
				liveResult->simulationSeconds = simulationElapsed;
				liveResult->gamesPerSecond = progress.snapshotGamesPerSecond;
				RecordMatchEnvironment(*liveResult, config);

				if (config.emitLiveGameDeltas) {
					progress.liveGamesOffset = liveGames.Append(state.games, *liveResult);
				} else {
					liveResult->games = state.games;
				}

				liveResult->statTotalsComplete = config.collectOptionalStats;
				PopulateStatTotals(*liveResult, state);
				progress.liveResult = std::move(liveResult);
				state.timing.postProcessingSeconds += SecondsSince(snapshotStart);
				lastLiveResultFinishedGames = finishedGames;
				lastLiveResultElapsed = simulationElapsed;
			}
			progressCallback(progress);
		}
	};

	auto shouldReportLiveProgress = [&](int finishedGames) {
		const double sinceLastReport = elapsedSeconds() - lastProgressReportElapsed;
		if (finishedGames % progressUpdateEveryGames == 0 && sinceLastReport >= 0.05)
			return true;

		return sinceLastReport >= 5.0;
	};

	std::vector<ModelBenchmarkRuntimeSignature> runtimeSignatures;
	runtimeSignatures.reserve(loadedModels.size());
	for (const LoadedModel& model : loadedModels)
		runtimeSignatures.push_back(RuntimeSignature(model));
	const ModelBenchmarkStrategy benchmarkStrategy = SelectModelBenchmarkStrategy(runtimeSignatures);
	const bool useVariableTiming = !sharedTiming;
	const bool useSameRuntimeSlotPath = benchmarkStrategy == ModelBenchmarkStrategy::SameRuntimeSlot;
	const std::string slotPathLabel = useSameRuntimeSlotPath
		? "same-runtime slot path"
		: (useVariableTiming ? "heterogeneous timing slot path" : "heterogeneous slot path");
	const std::string strategyDetails = useSameRuntimeSlotPath
		? "same-runtime slot path: selected models share the runtime interface."
		: slotPathLabel + ": " + SameRuntimeSlotMismatchReason(loadedModels) + ".";
	reportStartupProgress("Selecting benchmark strategy", strategyDetails);

	const torch::Device device(ResolveInferenceUseGPU() ? torch::kCUDA : torch::kCPU);
	for (LoadedModel& model : loadedModels) {
		reportStartupProgress("Loading benchmark model", model.name + ": constructing/loading policy network.");
		model.modelSet = LoadModelSet(model, device);
		throwIfCancelled();
	}

	std::set<std::pair<int, int>> eligibleMatchups;
	for (const PairKey& pairMode : eligiblePairModes)
		eligibleMatchups.insert({ pairMode.a, pairMode.b });
	auto modesHaveBaselineCoverage = [&]() {
		if (modes.size() <= 1)
			return true;
		for (int mode : modes) {
			const int minCompletedGames =
				CountEligiblePairsForMode(eligiblePairModes, mode)
				* std::min(maxGamesPerMode, 2);
			if (state.gamesByMode[mode] < minCompletedGames)
				return false;
		}
		return true;
	};

	auto shouldStopEarly = [&]() {
		if (config.runUntilMaxGames)
			return false;
		if (!modesHaveBaselineCoverage())
			return false;

		if (!HasConnectedOutcomeGraph(numModels, state.outcomes))
			return false;

		EloFit fit = FitElo(numModels, state.outcomes);
		if (config.stopWhenRankingKnown && ModelBenchmarkMath::IsFullRankingKnown(fit.elo, fit.covariance, config.betterConfidence)) {
			state.stopReason = "Model ranking reached target benchmark confidence";
			return true;
		}
		if (!config.stopWhenRankingKnown
			&& HasMinimumRatedOutcomesForEarlyStop(
				(int)state.outcomes.size(),
				(int)eligibleMatchups.size())
			&& AllRangesSmallEnough(fit, config.eloRangeConfidence, config.maxEloUncertaintyRange)) {
			state.stopReason = "Benchmark Elo uncertainty target reached";
			return true;
		}
		return false;
	};

	simulationStart = std::chrono::steady_clock::now();
	ResetBenchmarkGpuTelemetry();
	latestGpuTelemetry = SampleBenchmarkGpuTelemetry();
	lastGpuTelemetryElapsed = 0.0;
	RunSlotBenchmark(SlotBenchmarkParams{
		config,
		isCancelled,
		loadedModels,
		modes,
		matchups,
		modelNames,
		benchmarkStrategy,
		benchmarkingLabel,
		slotPathLabel,
		numModels,
		numArenas,
		tickSkip,
		actionDelay,
		noGoalSteps,
		noGoalTicks,
		useVariableTiming,
		runSeed,
		reportProgress,
		shouldReportLiveProgress,
		shouldStopEarly,
		{},
		device
	}, state);
	completedSimulationSeconds = simulationElapsedSeconds();

	auto postProcessingStart = std::chrono::steady_clock::now();
	std::vector<BenchmarkOutcome> kickoffOutcomes =
		FilterOutcomes(state.outcomes, ModelBenchmarkGoalTiming::Kickoff);
	std::vector<BenchmarkOutcome> postKickoffOutcomes =
		FilterOutcomes(state.outcomes, ModelBenchmarkGoalTiming::PostKickoff);
	const bool overallEloAvailable = HasConnectedOutcomeGraph(numModels, state.outcomes);
	EloFit combinedFit = FitElo(numModels, state.outcomes);
	EloFit kickoffFit = FitElo(numModels, kickoffOutcomes);
	EloFit postKickoffFit = FitElo(numModels, postKickoffOutcomes);
	AnchorFitsToOverallFloor(combinedFit, kickoffFit, postKickoffFit);
	const bool kickoffEloAvailable = HasConnectedOutcomeGraph(numModels, kickoffOutcomes);
	const bool postKickoffEloAvailable = HasConnectedOutcomeGraph(numModels, postKickoffOutcomes);
	std::map<int, ModeEloFit> modeFits;
	for (int mode : modes)
		modeFits[mode] = FitModeElo(numModels, FilterOutcomesByMode(state.outcomes, mode));

	std::vector<int> order = overallEloAvailable
		? BuildRankOrderFromFit(numModels, combinedFit, modelNames)
		: BuildRankOrder(numModels, state.outcomes, modelNames);
	double z = ModelBenchmarkMath::ConfidenceToZ(config.eloRangeConfidence);
	state.timing.postProcessingSeconds += SecondsSince(postProcessingStart);
	const int finishedGames = state.scoredGames + state.noResultGames;
	const std::string timingSummary =
		FormatBenchmarkTiming(state.timing, completedSimulationSeconds, finishedGames);

	ModelBenchmarkResult result = {};
	RecordMatchEnvironment(result, config);
	result.agentName = "Multiple";
	result.stageName = "Multiple";
	result.scoredGames = state.scoredGames;
	result.noResultGames = state.noResultGames;
	result.simulationSeconds = completedSimulationSeconds;
	result.gamesPerSecond = completedSimulationSeconds > 0.0
		? finishedGames / completedSimulationSeconds
		: 0.0;
	result.stopReason = state.stopReason;
	result.benchmarkStrategy = ModelBenchmarkStrategyName(benchmarkStrategy);
	result.timingSummary = timingSummary;
	result.seed = runSeed;
	result.maxGames = maxGames;
	result.arenaCreates = state.timing.slotArenaCreates;
	result.arenaRebuilds = state.timing.slotArenaRebuilds;
	result.arenaReuses = state.timing.slotArenaReuses;
	result.cudaOomRecoveries = state.timing.cudaOomRecoveries;
	result.cudaGraphsDisabledAfterOom = state.timing.cudaGraphsDisabledAfterOom;
	result.kickoffGoalWindowSeconds = config.kickoffGoalWindowSeconds;
	result.stochasticKickoff = config.stochasticKickoff;
	result.kickoffTemperature = config.kickoffTemperature;
	result.eloRangeConfidence = config.eloRangeConfidence;
	result.kickoffGoalCount = (int)kickoffOutcomes.size();
	result.postKickoffGoalCount = (int)postKickoffOutcomes.size();
	result.kickoffEloAvailable = kickoffEloAvailable;
	result.postKickoffEloAvailable = postKickoffEloAvailable;
	result.overallEloAvailable = overallEloAvailable;
	result.has1v1 = std::find(modes.begin(), modes.end(), 1) != modes.end();
	result.has2v2 = std::find(modes.begin(), modes.end(), 2) != modes.end();
	result.has3v3 = std::find(modes.begin(), modes.end(), 3) != modes.end();
	result.modelNames = modelNames;
	for (const ModelBenchmarkModel& model : config.models) {
		result.modelVersionIds.push_back(model.versionId);
		result.modelCheckpointPaths.push_back(model.checkpointPath);
	}
	result.games = std::move(state.games);
	auto pairModeEligible = [&](const std::pair<int, int>& matchup, int mode) {
		const PairKey key = {
			std::min(matchup.first, matchup.second),
			std::max(matchup.first, matchup.second),
			mode
		};
		return std::find(eligiblePairModes.begin(), eligiblePairModes.end(), key)
			!= eligiblePairModes.end();
	};

	for (int rank = 0; rank < (int)order.size(); rank++) {
		int modelIdx = order[rank];
		ModelBenchmarkRecord total = {};
		for (const auto& matchup : matchups) {
			if (matchup.first != modelIdx && matchup.second != modelIdx)
				continue;
			const int opponent = matchup.first == modelIdx ? matchup.second : matchup.first;
			for (int mode : modes) {
				if (!pairModeEligible(matchup, mode))
					continue;
				const ModelBenchmarkRecord& overall = GetRecord(
					state.records,
					modelIdx,
					opponent,
					mode,
					BenchmarkRecordType::Overall);
				total.wins += overall.wins;
				total.losses += overall.losses;
				total.noResults += overall.noResults;
			}
		}

		double se = std::sqrt(std::max(1e-6, combinedFit.covariance[modelIdx][modelIdx]));
		ModelBenchmarkRankingRow row = {};
		row.rank = rank + 1;
		row.modelIndex = modelIdx;
		row.modelName = modelNames[modelIdx];
		row.relativeElo = combinedFit.elo[modelIdx];
		row.elo1v1 = modeFits.count(1) ? modeFits[1].fit.elo[modelIdx] : 0.0;
		row.elo2v2 = modeFits.count(2) ? modeFits[2].fit.elo[modelIdx] : 0.0;
		row.elo3v3 = modeFits.count(3) ? modeFits[3].fit.elo[modelIdx] : 0.0;
		row.rangeLow = combinedFit.elo[modelIdx] - z * se;
		row.rangeHigh = combinedFit.elo[modelIdx] + z * se;
		row.betterThanSummary = overallEloAvailable
			? FormatConfidenceList(combinedFit, modelNames, modelIdx)
			: "N/A";
		row.totalRecord = total;
		row.elo1v1Available = modeFits.count(1) && modeFits[1].available[modelIdx];
		row.elo2v2Available = modeFits.count(2) && modeFits[2].available[modelIdx];
		row.elo3v3Available = modeFits.count(3) && modeFits[3].available[modelIdx];
		result.rankings.push_back(std::move(row));
	}

	for (int modelIdx = 0; modelIdx < numModels; modelIdx++) {
		const double kickoffSe =
			std::sqrt(std::max(1e-6, kickoffFit.covariance[modelIdx][modelIdx]));
		const double postKickoffSe =
			std::sqrt(std::max(1e-6, postKickoffFit.covariance[modelIdx][modelIdx]));
		ModelBenchmarkBreakdownRow row = {};
		row.modelIndex = modelIdx;
		row.modelName = modelNames[modelIdx];
		row.combinedElo = combinedFit.elo[modelIdx];
		row.kickoffElo = kickoffFit.elo[modelIdx];
		row.kickoffRangeLow = row.kickoffElo - z * kickoffSe;
		row.kickoffRangeHigh = row.kickoffElo + z * kickoffSe;
		row.postKickoffElo = postKickoffFit.elo[modelIdx];
		row.postKickoffRangeLow = row.postKickoffElo - z * postKickoffSe;
		row.postKickoffRangeHigh = row.postKickoffElo + z * postKickoffSe;
		row.kickoffDelta = row.kickoffElo - row.postKickoffElo;
		row.kickoffGames = CountModelOutcomes(kickoffOutcomes, modelIdx);
		row.postKickoffGames = CountModelOutcomes(postKickoffOutcomes, modelIdx);
		row.kickoffEloAvailable = kickoffEloAvailable;
		row.postKickoffEloAvailable = postKickoffEloAvailable;
		result.breakdown.push_back(std::move(row));
	}

	for (const auto& matchup : matchups) {
		for (int side = 0; side < 2; side++) {
			const int modelIdx = side == 0 ? matchup.first : matchup.second;
			const int opponent = side == 0 ? matchup.second : matchup.first;
			for (int mode : modes) {
				if (!pairModeEligible(matchup, mode))
					continue;
				ModelBenchmarkPairwiseRow row = {};
				row.modelIndex = modelIdx;
				row.opponentIndex = opponent;
				row.modelName = modelNames[modelIdx];
				row.opponentName = modelNames[opponent];
				row.modeName = ModeName(mode);
				row.modeElo = modeFits.count(mode)
					&& modeFits[mode].available[modelIdx]
					&& modeFits[mode].available[opponent]
					? modeFits[mode].fit.elo[modelIdx] - modeFits[mode].fit.elo[opponent]
					: 0.0;
				row.combined = GetRecord(
					state.records,
					modelIdx,
					opponent,
					mode,
					BenchmarkRecordType::Overall);
				row.kickoff = GetRecord(
					state.records,
					modelIdx,
					opponent,
					mode,
					BenchmarkRecordType::Kickoff);
				row.postKickoff = GetRecord(
					state.records,
					modelIdx,
					opponent,
					mode,
					BenchmarkRecordType::PostKickoff);
				const int kickoffScored = row.kickoff.wins + row.kickoff.losses;
				const int postKickoffScored = row.postKickoff.wins + row.postKickoff.losses;
				row.timingDeltaAvailable = kickoffScored > 0 && postKickoffScored > 0;
				if (row.timingDeltaAvailable) {
					row.kickoffVsPostKickoffDeltaPercent =
						(WinPct(row.kickoff) - WinPct(row.postKickoff)) * 100.0;
				}
				result.pairwise.push_back(std::move(row));
			}
		}
	}

	if (config.collectOptionalStats)
		PopulateOptionalStats(result, state, modes, modelNames);
	result.statTotalsComplete = config.collectOptionalStats;
	PopulateStatTotals(result, state);

	// Release model weights and cached allocator blocks before the completion
	// sample so current VRAM reports teardown while peak VRAM still describes the run.
	loadedModels.clear();
	cudaCleanupGuard.Cleanup();

	const std::string finalProgressLabel = state.stopReason == "Cancelled"
		? "Benchmark cancelled"
		: (state.stopReason.rfind("Terminated early:", 0) == 0
			? "Benchmark terminated early"
			: "Benchmark complete");
	reportProgress(
		finalProgressLabel,
		ModelBenchmarkStrategyName(benchmarkStrategy)
		+ "\n"
		+ FormatRecordSummary(state.records, modelNames, order)
		+ "\n"
		+ FormatModeDistributionSummary(state, modes)
		+ "\n"
		+ timingSummary,
		FormatLiveEstimateSummary(numModels, state.outcomes, modelNames, config.eloRangeConfidence));
	latestGpuTelemetry = SampleBenchmarkGpuTelemetry();
	ApplyCudaGraphTelemetry(state, latestGpuTelemetry);
	result.gpu = latestGpuTelemetry;
	result.elapsedSeconds = elapsedSeconds();
	return result;
}
}
