#include "ModelBenchmarkInternal.h"
#include "BenchmarkStartingStates.h"
#include "ModelBenchmarkEngine.h"
#include <GigaLearnInfer/RuntimeOptions.h>

#include <ATen/CPUGeneratorImpl.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

#ifdef RG_CUDA_SUPPORT
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

namespace RocketForgeTraining::ModelBenchmarkInternal
{
	double BenchmarkWorkProgress(
		bool runUntilMaxGames,
		int completedGames,
		int scheduledGames,
		int targetGames,
		double modeDurationWeight) {
		const double denominator = std::max(1, targetGames);
		if (!runUntilMaxGames)
			return completedGames / denominator;
		return scheduledGames / (denominator * std::max(1e-9, modeDurationWeight));
	}

	bool BenchmarkArenaLayoutCanBeReclaimed(int completedGames, int targetGames) {
		return targetGames > 0 && completedGames >= targetGames;
	}

	bool BenchmarkMirrorGroupComplete(int completedGames, int expectedGames) {
		return expectedGames > 0 && completedGames >= expectedGames;
	}

	bool IncludeBenchmarkGameInRating(bool goalScored, bool mirrorGroupComplete) {
		return goalScored && mirrorGroupComplete;
	}

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
		bool modelABlueFirst) {
		if (gameCount < 1 || gameCount > 2)
			throw std::invalid_argument("ModelBenchmark: mirror task count must be one or two.");

		std::vector<GGL::PolicyMatchTask> tasks;
		tasks.reserve(gameCount);
		for (int side = 0; side < gameCount; side++) {
			const bool modelABlue = side == 0 ? modelABlueFirst : !modelABlueFirst;
			GGL::PolicyMatchTask task = {};
			task.blueModel = modelABlue ? modelA : modelB;
			task.orangeModel = modelABlue ? modelB : modelA;
			task.blueArenaKey = modelABlue ? modelAArenaKey : modelBArenaKey;
			task.orangeArenaKey = modelABlue ? modelBArenaKey : modelAArenaKey;
			task.teamSize = teamSize;
			task.snapshot = snapshot;
			task.userValues = {
				modelA,
				modelB,
				teamSize
			};
			task.gameId = firstGameId + side;
			task.snapshotId = snapshotId;
			task.mirrorGameCount = gameCount;
			tasks.push_back(std::move(task));
		}
		return tasks;
	}

	namespace
	{
		enum BenchmarkTaskValue {
			TaskValueModelA = 0,
			TaskValueModelB = 1,
			TaskValueMode = 2
		};

		struct ArenaLayoutKey {
			int teamSize = 1;
			int blueArenaKey = -1;
			int orangeArenaKey = -1;

			bool operator<(const ArenaLayoutKey& other) const {
				return std::tie(teamSize, blueArenaKey, orangeArenaKey)
					< std::tie(other.teamSize, other.blueArenaKey, other.orangeArenaKey);
			}

			bool operator==(const ArenaLayoutKey& other) const {
				return teamSize == other.teamSize
					&& blueArenaKey == other.blueArenaKey
					&& orangeArenaKey == other.orangeArenaKey;
			}
		};

		PairKey MakePairModeKey(const std::pair<int, int>& matchup, int mode) {
			return PairKey{ std::min(matchup.first, matchup.second), std::max(matchup.first, matchup.second), mode };
		}

		bool SameWheelPairConfig(const WheelPairConfig& a, const WheelPairConfig& b) {
			return a.wheelRadius == b.wheelRadius
				&& a.suspensionRestLength == b.suspensionRestLength
				&& a.connectionPointOffset == b.connectionPointOffset;
		}

		bool SameCarConfig(const CarConfig& a, const CarConfig& b) {
			return a.hitboxSize == b.hitboxSize
				&& a.hitboxPosOffset == b.hitboxPosOffset
				&& SameWheelPairConfig(a.frontWheels, b.frontWheels)
				&& SameWheelPairConfig(a.backWheels, b.backWheels)
				&& a.threeWheels == b.threeWheels
				&& a.dodgeDeadzone == b.dodgeDeadzone;
		}

		void AddTiming(BenchmarkTiming& out, const GGL::PolicyMatchTiming& timing) {
			out.physicsSeconds += timing.physicsSeconds;
			out.obsMaskSeconds += timing.obsMaskSeconds;
			out.gpuTransferSeconds += timing.gpuTransferSeconds;
			out.inferenceSeconds += timing.inferenceSeconds;
			out.actionParseSeconds += timing.actionParseSeconds;
			out.actionStepSeconds += timing.actionStepSeconds;
			out.finishSeconds += timing.finishSeconds;
			out.schedulerSeconds += timing.schedulerSeconds;
			out.slotAssignSeconds += timing.slotAssignSeconds;
			out.slotArenaSetupSeconds += timing.slotArenaSetupSeconds;
			out.slotSideRuntimeSeconds += timing.slotSideRuntimeSeconds;
			out.slotSnapshotSeconds += timing.slotSnapshotSeconds;
			out.slotStateInitSeconds += timing.slotStateInitSeconds;
			out.slotBuilderResetSeconds += timing.slotBuilderResetSeconds;
			out.batchPrepSeconds += timing.batchPrepSeconds;
			out.cudaOomRecoveries += timing.cudaOomRecoveries;
			out.cudaGraphsDisabledAfterOom =
				out.cudaGraphsDisabledAfterOom || timing.cudaGraphsDisabledAfterOom;
			out.cudaGraphEntries = timing.cudaGraphEntries;
			out.cudaGraphCaptures += timing.cudaGraphCaptures;
			out.cudaGraphReplays += timing.cudaGraphReplays;
			out.cudaGraphFallbacks += timing.cudaGraphFallbacks;
			out.cudaGraphVariantLimitFallbacks += timing.cudaGraphVariantLimitFallbacks;
			out.cudaGraphCaptureFailures += timing.cudaGraphCaptureFailures;
			out.slotArenaCreates += timing.slotArenaCreates;
			out.slotArenaRebuilds += timing.slotArenaRebuilds;
			out.slotArenaReuses += timing.slotArenaReuses;
		}

		uint64_t MixBenchmarkSeed(uint64_t value) {
			value += 0x9e3779b97f4a7c15ULL;
			value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
			value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
			return value ^ (value >> 31);
		}

		uint64_t BenchmarkSnapshotSeed(uint64_t runSeed, uint64_t snapshotId) {
			// SplitMix64 keeps nearby snapshot IDs from producing nearby engine seeds.
			return MixBenchmarkSeed(runSeed + snapshotId);
		}

		bool SingletonModelABlue(
			uint64_t runSeed,
			const std::pair<int, int>& matchup,
			int mode) {
			uint64_t value = runSeed;
			value ^= (uint64_t)(uint32_t)matchup.first << 32;
			value ^= (uint32_t)matchup.second;
			value ^= (uint64_t)(uint32_t)mode << 16;
			return (MixBenchmarkSeed(value) & 1ULL) == 0;
		}

	}

	bool BenchmarkMirrorBuffer::Add(
		BenchmarkRunState& state,
		ModelBenchmarkGameMetadata metadata,
		int expectedGames,
		double outcomeWeight) {
		expectedGames = std::max(1, expectedGames);
		const Key key = {
			metadata.snapshotId,
			metadata.modelA,
			metadata.modelB,
			metadata.mode
		};
		const size_t metadataIndex = state.games.size();
		state.games.push_back(std::move(metadata));

		PendingGroup& group = pendingGroups[key];
		if (group.expectedGames == 0)
			group.expectedGames = expectedGames;
		else if (group.expectedGames != expectedGames)
			throw std::runtime_error("ModelBenchmark: mirror group has inconsistent expected game counts.");
		group.metadataIndices.push_back(metadataIndex);
		if ((int)group.metadataIndices.size() > expectedGames)
			throw std::runtime_error("ModelBenchmark: mirror group completed more games than scheduled.");
		if (!BenchmarkMirrorGroupComplete((int)group.metadataIndices.size(), expectedGames))
			return false;

		for (size_t index : group.metadataIndices) {
			ModelBenchmarkGameMetadata& game = state.games[index];
			game.mirrorCompleted = true;
			if (game.goalScored) {
				if (game.winner != game.modelA && game.winner != game.modelB)
					throw std::runtime_error("ModelBenchmark: scored game winner is not part of its matchup.");
				const int loser = game.winner == game.modelA ? game.modelB : game.modelA;
				state.outcomes.push_back({
					game.winner,
					loser,
					game.mode,
					game.goalTiming,
					outcomeWeight
				});
				AddResultToRecords(
					state.records,
					game.winner,
					loser,
					game.mode,
					game.goalTiming);
			} else {
				AddNoResultToRecords(
					state.records,
					game.modelA,
					game.modelB,
					game.mode);
			}
			game.includedInRating = IncludeBenchmarkGameInRating(
				game.goalScored,
				game.mirrorCompleted);
		}
		pendingGroups.erase(key);
		return true;
	}

	void RunSlotBenchmark(const SlotBenchmarkParams& params, BenchmarkRunState& state) {
		for (int mode : params.modes)
			state.statsByMode[mode].resize(params.numModels);

		std::vector<int> modelArenaKeys(params.loadedModels.size(), -1);
		std::vector<CarConfig> arenaKeyConfigs;
		for (int modelIdx = 0; modelIdx < (int)params.loadedModels.size(); modelIdx++) {
			const CarConfig& carConfig = params.loadedModels[modelIdx].carConfig;
			int arenaKey = -1;
			for (int key = 0; key < (int)arenaKeyConfigs.size(); key++) {
				if (SameCarConfig(carConfig, arenaKeyConfigs[key])) {
					arenaKey = key;
					break;
				}
			}
			if (arenaKey < 0) {
				arenaKey = (int)arenaKeyConfigs.size();
				arenaKeyConfigs.push_back(carConfig);
			}
			modelArenaKeys[modelIdx] = arenaKey;
		}

		std::vector<PairKey> eligiblePairModes;
		eligiblePairModes.reserve(state.scheduledByPairMode.size());
		for (const auto& [pairMode, scheduledGames] : state.scheduledByPairMode) {
			(void)scheduledGames;
			eligiblePairModes.push_back(pairMode);
		}
		auto eligiblePairMode = [&](const std::pair<int, int>& matchup, int mode) {
			return state.scheduledByPairMode.find(MakePairModeKey(matchup, mode))
				!= state.scheduledByPairMode.end();
		};
		std::map<std::pair<int, int>, double> overallOutcomeWeights;
		for (const auto& matchup : params.matchups) {
			const std::pair<int, int> pair = {
				std::min(matchup.first, matchup.second),
				std::max(matchup.first, matchup.second)
			};
			if (std::any_of(
				eligiblePairModes.begin(),
				eligiblePairModes.end(),
				[&](const PairKey& pairMode) {
					return pairMode.a == pair.first && pairMode.b == pair.second;
				})) {
				overallOutcomeWeights[pair] = OverallOutcomeWeightForPair(
					eligiblePairModes,
					pair.first,
					pair.second);
			}
		}

		std::map<ArenaLayoutKey, int> targetGamesByArenaLayout;
		for (int mode : params.modes) {
			for (const auto& matchup : params.matchups) {
				if (!eligiblePairMode(matchup, mode))
					continue;

				const int pairTarget = PairGameTarget(state, MakePairModeKey(matchup, mode));
				const int mirroredSideGames = pairTarget / 2;
				const int modelAKey = modelArenaKeys[matchup.first];
				const int modelBKey = modelArenaKeys[matchup.second];
				targetGamesByArenaLayout[{ mode, modelAKey, modelBKey }] += mirroredSideGames;
				targetGamesByArenaLayout[{ mode, modelBKey, modelAKey }] += mirroredSideGames;

				if ((pairTarget % 2) != 0) {
					const bool modelABlue = SingletonModelABlue(
						params.runSeed,
						matchup,
						mode);
					targetGamesByArenaLayout[{
						mode,
						modelABlue ? modelAKey : modelBKey,
						modelABlue ? modelBKey : modelAKey
					}]++;
				}
			}
		}
		std::map<ArenaLayoutKey, int> completedGamesByArenaLayout;
		BenchmarkMirrorBuffer mirrorBuffer;
		int stopCheckedThroughGames = state.scoredGames + state.noResultGames;

		auto allPairModesScheduled = [&]() {
			for (const auto& pair : state.scheduledByPairMode)
				if (pair.second < PairGameTarget(state, pair.first))
					return false;

			return true;
		};

		auto modeHasRemainingPairs = [&](int mode) {
			for (const auto& matchup : params.matchups) {
				const auto scheduled = state.scheduledByPairMode.find(MakePairModeKey(matchup, mode));
				if (scheduled != state.scheduledByPairMode.end()
					&& scheduled->second < PairGameTarget(state, scheduled->first)) {
					return true;
				}
			}

			return false;
		};

		auto estimatedModeGameLength = [&](int mode) {
			if (state.gamesByMode[mode] > 0 && state.stepsByMode[mode] > 0)
				return std::max(1.0, (double)state.stepsByMode[mode] / state.gamesByMode[mode]);
			return DefaultModeDurationWeight(mode);
		};

		auto chooseWorkTarget = [&]() {
			BenchmarkWorkTarget best = {};
			double bestWorkProgress = std::numeric_limits<double>::infinity();
			double bestTieProgress = std::numeric_limits<double>::infinity();
			double bestModeLength = -1.0;

			// Duration-normalized scheduled progress gives longer modes more arena
			// capacity up front. Fixed-count runs use it throughout for deterministic
			// scheduling; interruptible runs use completed progress first so partial
			// result counts remain balanced.
			for (int mode : params.modes) {
				if (!modeHasRemainingPairs(mode))
					continue;
				const double modeLength = params.config.runUntilMaxGames
					? DefaultModeDurationWeight(mode)
					: estimatedModeGameLength(mode);
				const int64_t targetTotal = std::max<int64_t>(1, ModeGameTarget(state, mode));
				const double scheduledProgress = BenchmarkWorkProgress(
					true,
					0,
					state.scheduledGamesByMode[mode],
					targetTotal,
					modeLength);
				const double workProgress = BenchmarkWorkProgress(
					params.config.runUntilMaxGames,
					state.gamesByMode[mode],
					state.scheduledGamesByMode[mode],
					targetTotal,
					modeLength);
				const double tieProgress = params.config.runUntilMaxGames
					? 0.0
					: scheduledProgress;
				if (!best.valid
					|| workProgress < bestWorkProgress - 1e-9
					|| (std::abs(workProgress - bestWorkProgress) <= 1e-9 && tieProgress < bestTieProgress - 1e-9)
					|| (std::abs(workProgress - bestWorkProgress) <= 1e-9
						&& std::abs(tieProgress - bestTieProgress) <= 1e-9
						&& modeLength > bestModeLength + 1e-9)) {
					best = { mode, true };
					bestWorkProgress = workProgress;
					bestTieProgress = tieProgress;
					bestModeLength = modeLength;
				}
			}
			return best;
		};

		uint64_t nextModeSnapshotId[4] = {};

		auto scheduleWorkTarget = [&](std::vector<GGL::PolicyMatchTask>& pendingTasks, const BenchmarkWorkTarget& target) {
			if (!target.valid)
				return false;
			if (params.isCancelled && params.isCancelled()) {
				state.cancelled = true;
				state.stopReason = "Cancelled";
				return false;
			}

			const uint64_t snapshotId = params.config.startingStateDirectory.empty()
				? state.nextSnapshotId++
				: params.config.snapshotOffset + uint64_t(target.mode) * BENCHMARK_START_MODE_ID_STRIDE + nextModeSnapshotId[target.mode]++;
			ArenaSnapshot snapshot = params.runSeed != 0
				? SampleKickoffState(
					target.mode,
					params.loadedModels.front().carConfig,
					BenchmarkSnapshotSeed(params.runSeed, snapshotId),
					params.config.engine,
					params.config.startingStateDirectory)
				: SampleKickoffState(target.mode, params.loadedModels.front().carConfig, params.config.engine);
			const size_t pendingBefore = pendingTasks.size();
			for (const auto& matchup : params.matchups) {
				PairKey scheduleKey = MakePairModeKey(matchup, target.mode);
				if (state.scheduledByPairMode.find(scheduleKey) == state.scheduledByPairMode.end())
					continue;

				const int remainingGames = PairGameTarget(state, scheduleKey) - state.scheduledByPairMode[scheduleKey];
				const int mirrorGameCount = std::min(2, remainingGames);
				if (mirrorGameCount <= 0)
					continue;
				const bool modelABlueFirst = mirrorGameCount == 2
					|| SingletonModelABlue(
						params.runSeed,
						matchup,
						target.mode);

				std::vector<GGL::PolicyMatchTask> mirrorTasks = BuildBenchmarkMirrorTasks(
					matchup.first,
					matchup.second,
					target.mode,
					modelArenaKeys[matchup.first],
					modelArenaKeys[matchup.second],
					snapshot,
					snapshotId,
					state.nextGameId,
					mirrorGameCount,
					modelABlueFirst);
				state.nextGameId += mirrorTasks.size();
				for (GGL::PolicyMatchTask& task : mirrorTasks) {
					pendingTasks.push_back(std::move(task));
					state.scheduledByPairMode[scheduleKey]++;
					state.scheduledGamesByMode[target.mode]++;
					state.scheduledGames++;
				}
			}
			return pendingTasks.size() > pendingBefore;
		};

		auto fillPendingTasks = [&](std::vector<GGL::PolicyMatchTask>& pendingTasks, int activeSlots, int targetQueueSize) {
			targetQueueSize = std::clamp(targetQueueSize, 1, params.numArenas);
			while (!state.cancelled
				&& !allPairModesScheduled()
				&& (int)(pendingTasks.size() + activeSlots) < targetQueueSize) {
				if (!scheduleWorkTarget(pendingTasks, chooseWorkTarget()))
					return;
			}
		};

		auto taskUsesDemand = [](
			const GGL::PolicyMatchTask& task,
			const GGL::PolicyMatchArenaLayoutDemand& demand) {
			return task.teamSize == demand.teamSize
				&& task.blueArenaKey == demand.blueArenaKey
				&& task.orangeArenaKey == demand.orangeArenaKey;
		};

		auto demandHasUnscheduledWork = [&](
			const GGL::PolicyMatchArenaLayoutDemand& demand) {
			for (const auto& matchup : params.matchups) {
				const PairKey scheduleKey = MakePairModeKey(matchup, demand.teamSize);
				if (state.scheduledByPairMode.find(scheduleKey) == state.scheduledByPairMode.end())
					continue;

				const int remainingGames = PairGameTarget(state, scheduleKey) - state.scheduledByPairMode[scheduleKey];
				if (remainingGames <= 0)
					continue;

				const int modelAKey = modelArenaKeys[matchup.first];
				const int modelBKey = modelArenaKeys[matchup.second];
				const bool includesForward =
					demand.blueArenaKey == modelAKey
					&& demand.orangeArenaKey == modelBKey;
				const bool includesReverse =
					demand.blueArenaKey == modelBKey
					&& demand.orangeArenaKey == modelAKey;
				if (remainingGames >= 2 && (includesForward || includesReverse))
					return true;
				if (remainingGames == 1) {
					const bool modelABlue = SingletonModelABlue(
						params.runSeed,
						matchup,
						demand.teamSize);
					if ((modelABlue && includesForward)
						|| (!modelABlue && includesReverse)) {
						return true;
					}
				}
			}
			return false;
		};

		auto fillPendingTasksForArenaLayouts = [&](
			std::vector<GGL::PolicyMatchTask>& pendingTasks,
			const std::vector<GGL::PolicyMatchArenaLayoutDemand>& demands) {
			auto pendingTaskCountForDemand = [&](const GGL::PolicyMatchArenaLayoutDemand& demand) {
				return (int)std::count_if(
					pendingTasks.begin(),
					pendingTasks.end(),
					[&](const GGL::PolicyMatchTask& task) {
						return taskUsesDemand(task, demand);
					});
			};

			std::vector<GGL::PolicyMatchArenaLayoutDemand> unmetDemands;
			for (const auto& demand : demands) {
				if (pendingTaskCountForDemand(demand) < demand.idleSlots
					&& demandHasUnscheduledWork(demand)) {
					unmetDemands.push_back(demand);
				}
			}

			// Generate the same deterministic global work sequence as normal refill.
			// Quotas make this loop finite even when exposing compatible work also
			// schedules other modes or physical layouts along the way.
			while (!unmetDemands.empty() && !allPairModesScheduled()) {
				if (!scheduleWorkTarget(pendingTasks, chooseWorkTarget()))
					break;
				unmetDemands.erase(
					std::remove_if(
						unmetDemands.begin(),
						unmetDemands.end(),
						[&](const GGL::PolicyMatchArenaLayoutDemand& demand) {
							return pendingTaskCountForDemand(demand) >= demand.idleSlots
								|| !demandHasUnscheduledWork(demand);
						}),
					unmetDemands.end());
			}
		};

		std::vector<GGL::PolicyMatchRuntime> runtimes;
		runtimes.reserve(params.loadedModels.size());
		for (int modelIdx = 0; modelIdx < (int)params.loadedModels.size(); modelIdx++) {
			LoadedModel& model = params.loadedModels[modelIdx];
			GGL::PolicyMatchRuntime runtime = {};
			runtime.models = model.modelSet.get();
			runtime.carConfig = model.carConfig;
			runtime.obsSize = model.obsSize;
			runtime.numActions = model.numActions;
			runtime.usesActionMask = model.usesActionMask;
			runtime.deterministic = true;
			runtime.stochasticKickoff = params.config.stochasticKickoff;
			runtime.kickoffTemperature = params.config.kickoffTemperature;
			if (runtime.stochasticKickoff) {
				const uint64_t actionSeed = MixBenchmarkSeed(params.runSeed + modelIdx);
#ifdef RG_CUDA_SUPPORT
				if (params.device.is_cuda()) {
					runtime.actionGenerator = at::cuda::detail::createCUDAGenerator(params.device.index());
					runtime.actionGenerator->set_current_seed(actionSeed);
				} else
#endif
					runtime.actionGenerator = at::detail::createCPUGenerator(actionSeed);
			}

			runtime.useHalfPrecision = false;
			runtime.useCudaGraph = params.device.is_cuda();
			runtime.policyTemperature = model.policyTemperature;
			runtime.tickSkip = model.tickSkip;
			runtime.actionDelay = model.actionDelay;
			runtime.runtimeGroup = modelIdx;
			runtime.makeRuntimeComponents = [&params, modelIdx]() {
				return GGL::PolicyMatchRuntimeComponents{
					std::unique_ptr<ObsBuilder>(params.loadedModels[modelIdx].agent->MakeConfiguredObsBuilder()),
					std::unique_ptr<ActionParser>(params.loadedModels[modelIdx].agent->MakeActionParser())
				};
			};
			runtimes.push_back(std::move(runtime));
		}

		params.reportProgress(
			"Preparing benchmark episodes",
			params.slotPathLabel + ": creating reusable arena slots before simulation starts.",
			{});

		const LiveEloEstimate initialEstimate = BuildLiveEloEstimate(params.numModels, state.outcomes, params.modelNames);
		params.reportProgress(
			"Benchmark simulation started",
			params.slotPathLabel + "\n" + FormatRecordSummary(state.records, params.modelNames, initialEstimate.order),
			FormatLiveEstimateSummary(initialEstimate, params.modelNames, params.config.eloRangeConfidence));

		std::unordered_map<uint64_t, BenchmarkPassTracker> passTrackers;
		passTrackers.reserve(params.numArenas);

		struct SlotStats {
			bool assigned = false;
			uint64_t gameId = 0;
			int blueModel = 0;
			int orangeModel = 0;
			int mode = 1;
			BenchmarkPassTracker passes;
			std::array<RunningStats, 2> totals = {};
		};

		std::vector<SlotStats> slotStats(params.numArenas);
		const bool parallelStatsByDefault = true;
		const bool parallelStats = GGL::RuntimeOptionEnabled("RF_BENCHMARK_PARALLEL_STATS", parallelStatsByDefault);

		auto mergeStats = [&](int blue, int orange, int mode, const std::array<RunningStats, 2>& totals) {
			const PairKey key{ std::min(blue, orange), std::max(blue, orange), mode };
			const int models[] = { blue, orange };
			for (int side = 0; side < 2; ++side) {
				const int model = models[side];
				MergeRunningStats(state.stats[model], totals[side]);
				MergeRunningStats(state.statsByMode.at(mode)[model], totals[side]);
				MergeRunningStats(state.statsByPairMode[key][model == key.a ? 0 : 1], totals[side]);
			}
		};

		auto flushStats = [&]() {
			// Workers have joined before any finish/progress callback.
			// Keep pass history across flushes; only a new episode resets that history.
			for (SlotStats& slot : slotStats) {
				if (!slot.assigned)
					continue;

				mergeStats(slot.blueModel, slot.orangeModel, slot.mode, slot.totals);
				slot.totals = {};
			}
		};

		auto finishBenchmarkTask = [&](const GGL::PolicyMatchFinish& finish) {
			flushStats();
			const GGL::PolicyMatchTask& task = finish.task;
			passTrackers.erase(task.gameId);
			const int modelA = task.userValues[TaskValueModelA];
			const int modelB = task.userValues[TaskValueModelB];
			const int mode = task.userValues[TaskValueMode];
			const double goalTimeSeconds = FinishedGameSeconds(
				finish.steps,
				params.useVariableTiming,
				params.tickSkip);
			const ModelBenchmarkGoalTiming goalTiming = ClassifyGoalTiming(
				finish.goalScored,
				goalTimeSeconds,
				params.config.kickoffGoalWindowSeconds);

			completedGamesByArenaLayout[{ task.teamSize, task.blueArenaKey, task.orangeArenaKey }]++;
			state.stepsByMode[mode] += finish.steps;
			const int64_t completedPhysicsTicks = (int64_t)finish.steps
				* (params.useVariableTiming ? int64_t{ 1 } : (int64_t)params.tickSkip);
			state.completedPhysicsTicksByMode[mode] += completedPhysicsTicks;
			state.gamesByMode[mode]++;
			if (finish.goalScored)
				state.scoredGames++;
			else
				state.noResultGames++;

			const std::pair<int, int> pair = {
				std::min(modelA, modelB),
				std::max(modelA, modelB)
			};
			const bool mirrorCompleted = mirrorBuffer.Add(state, {
				task.gameId,
				task.snapshotId,
				modelA,
				modelB,
				task.blueModel,
				task.orangeModel,
				mode,
				finish.goalScored ? finish.winningModel : -1,
				goalTimeSeconds,
				goalTiming,
				finish.goalScored,
				false,
				false
			}, task.mirrorGameCount, overallOutcomeWeights.at(pair));
			bool stopRequested = false;
			if (mirrorCompleted) {
				const int stopCheckEveryGames = 25;
				const int finishedGames = state.scoredGames + state.noResultGames;
				if (finishedGames / stopCheckEveryGames > stopCheckedThroughGames / stopCheckEveryGames) {
					stopCheckedThroughGames = finishedGames;
					stopRequested = params.shouldStopEarly();
				}
			}

			const int finishedGames = state.scoredGames + state.noResultGames;
			if (params.shouldReportLiveProgress(finishedGames)) {
				const LiveEloEstimate estimate = BuildLiveEloEstimate(params.numModels, state.outcomes, params.modelNames);
				params.reportProgress(
					params.benchmarkingLabel,
					ModelBenchmarkStrategyName(params.benchmarkStrategy) + "\n"
						+ FormatRecordSummary(state.records, params.modelNames, estimate.order),
					FormatLiveEstimateSummary(estimate, params.modelNames, params.config.eloRangeConfidence));
			}

			return stopRequested;
		};

		auto isArenaLayoutComplete = [&](int teamSize, int blueArenaKey, int orangeArenaKey) {
			if (!params.config.runUntilMaxGames)
				return false;
			const ArenaLayoutKey key = { teamSize, blueArenaKey, orangeArenaKey };
			const auto targetItr = targetGamesByArenaLayout.find(key);
			if (targetItr == targetGamesByArenaLayout.end())
				return false;
			return BenchmarkArenaLayoutCanBeReclaimed(
				completedGamesByArenaLayout[key],
				targetItr->second);
		};

		auto updateRunnerStatus = [&](const GGL::PolicyMatchRunnerStatus& status) {
			state.activeArenas = status.activeSlots;
			state.totalArenas = status.totalSlots;
			state.pendingGames = status.pendingTasks;
			state.arenaCreates = status.slotArenaCreates;
			state.arenaRebuilds = status.slotArenaRebuilds;
			state.arenaReuses = status.slotArenaReuses;
			state.cudaOomRecoveries = status.cudaOomRecoveries;
			state.cudaGraphsDisabledAfterOom = status.cudaGraphsDisabledAfterOom;
			state.cudaGraphEntries = status.cudaGraphEntries;
			state.cudaGraphCaptures = status.cudaGraphCaptures;
			state.cudaGraphReplays = status.cudaGraphReplays;
			state.cudaGraphFallbacks = status.cudaGraphFallbacks;
			state.cudaGraphVariantLimitFallbacks = status.cudaGraphVariantLimitFallbacks;
			state.cudaGraphCaptureFailures = status.cudaGraphCaptureFailures;
			state.active1v1Arenas = 0;
			state.active2v2Arenas = 0;
			state.active3v3Arenas = 0;
			for (const auto& [teamSize, activeSlots] : status.activeSlotsByTeamSize) {
				if (teamSize == 1)
					state.active1v1Arenas = activeSlots;
				else if (teamSize == 2)
					state.active2v2Arenas = activeSlots;
				else if (teamSize == 3)
					state.active3v3Arenas = activeSlots;
			}
		};

		GGL::PolicyMatchRunnerParams runnerParams = {
			runtimes,
			params.device,
			params.numArenas,
			params.tickSkip,
			params.actionDelay,
			params.noGoalSteps,
			[&params](const GGL::PolicyMatchTask& task) {
				return GGL::PolicyMatchArena{
					MakeArena(
						task.teamSize,
						params.loadedModels[task.blueModel].carConfig,
						params.loadedModels[task.orangeModel].carConfig,
						params.config.engine),
					nullptr
				};
			},
			fillPendingTasks,
			allPairModesScheduled,
			params.isCancelled,
			[&](const GGL::PolicyMatchStep& step) {
				const int64_t processedPhysicsTicks = step.deltaSeconds > 0.0f
					? std::max<int64_t>(1, (int64_t)std::llround((double)step.deltaSeconds * 120.0))
					: (int64_t)params.tickSkip;
				state.processedPhysicsTicksByMode[step.task.teamSize] += processedPhysicsTicks;
				if (!params.config.collectOptionalStats || (parallelStats && step.slotIndex >= 0))
					return;

				std::array<RunningStats, 2> totals = {};
				const BenchmarkPassCounts passes = step.task.teamSize > 1
					? AccumulateCompletedPasses(passTrackers[step.task.gameId], step.state)
					: BenchmarkPassCounts{};
				const float delta = step.deltaSeconds > 0 ? step.deltaSeconds : params.tickSkip / 120.0f;
				AccumulateStepStats(totals[0], step.state, step.prevState, Team::BLUE, delta);
				AccumulateStepStats(totals[1], step.state, step.prevState, Team::ORANGE, delta);
				totals[0].passes += passes.ForTeam(Team::BLUE);
				totals[1].passes += passes.ForTeam(Team::ORANGE);
				mergeStats(step.task.blueModel, step.task.orangeModel, step.task.teamSize, totals);
			},
			[&](const GGL::PolicyMatchFinish& finish) {
				return finishBenchmarkTask(finish);
			}
		};

		const bool independentTimingByDefault = true;
		runnerParams.independentSlotTiming = GGL::RuntimeOptionEnabled("RF_BENCHMARK_INDEPENDENT_TIMING", independentTimingByDefault);

		if (parallelStats && params.config.collectOptionalStats) {
			runnerParams.onStepParallel = [&](const GGL::PolicyMatchStep& step) {
				SlotStats& slot = slotStats.at(step.slotIndex);
				if (!slot.assigned || slot.gameId != step.task.gameId) {
					slot = {};
					slot.assigned = true;
					slot.gameId = step.task.gameId;
					slot.blueModel = step.task.blueModel;
					slot.orangeModel = step.task.orangeModel;
					slot.mode = step.task.teamSize;
				}

				const BenchmarkPassCounts passes = slot.mode > 1
					? AccumulateCompletedPasses(slot.passes, step.state)
					: BenchmarkPassCounts{};
				AccumulateStepStats(slot.totals[0], step.state, step.prevState, Team::BLUE, step.deltaSeconds);
				AccumulateStepStats(slot.totals[1], step.state, step.prevState, Team::ORANGE, step.deltaSeconds);
				slot.totals[0].passes += passes.ForTeam(Team::BLUE);
				slot.totals[1].passes += passes.ForTeam(Team::ORANGE);
			};
		}

		runnerParams.noGoalTicks = params.noGoalTicks;
		if (!params.config.startingStateDirectory.empty())
			runnerParams.maxPhysicsStepTicks = 1;

		if (params.config.runUntilMaxGames) {
			runnerParams.isArenaLayoutComplete = isArenaLayoutComplete;
			runnerParams.fillPendingTasksForArenaLayouts = fillPendingTasksForArenaLayouts;
		}
		runnerParams.onStatus = updateRunnerStatus;

		GGL::PolicyMatchRunnerResult result = params.runPolicyMatchRunner
			? params.runPolicyMatchRunner(runnerParams)
			: (params.useVariableTiming
				? GGL::RunPolicyMatchVariableTimingSlots(runnerParams)
				: GGL::RunPolicyMatchSlots(runnerParams));

		flushStats();
		if (result.cancelled) {
			state.cancelled = true;
			state.stopReason = "Cancelled";
		}
		AddTiming(state.timing, result.timing);
		state.cudaOomRecoveries = result.timing.cudaOomRecoveries;
		state.cudaGraphsDisabledAfterOom = result.timing.cudaGraphsDisabledAfterOom;
		state.cudaGraphEntries = result.timing.cudaGraphEntries;
		state.cudaGraphCaptures = result.timing.cudaGraphCaptures;
		state.cudaGraphReplays = result.timing.cudaGraphReplays;
		state.cudaGraphFallbacks = result.timing.cudaGraphFallbacks;
		state.cudaGraphVariantLimitFallbacks = result.timing.cudaGraphVariantLimitFallbacks;
		state.cudaGraphCaptureFailures = result.timing.cudaGraphCaptureFailures;
		state.activeArenas = 0;
		state.pendingGames = 0;
		state.active1v1Arenas = 0;
		state.active2v2Arenas = 0;
		state.active3v3Arenas = 0;
	}
}
