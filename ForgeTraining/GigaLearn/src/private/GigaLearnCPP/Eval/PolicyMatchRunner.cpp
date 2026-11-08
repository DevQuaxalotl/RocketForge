#include <GigaLearnCPP/Eval/PolicyMatchRunner.h>

#include "PolicyMatchCudaOomRecovery.h"

#include <GigaLearnInfer/ModelManager.h>
#include <RLGymCPP/Math.h>
#include <RLGymCPP/StateSetters/FuzzedKickoffState.h>
#include <RLGymCPP/StateSetters/RandomState.h>
#include <RLGymCPP/ThreadPool.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <exception>
#include <filesystem>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <tuple>

#ifdef RG_CUDA_SUPPORT
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>
#endif

namespace GGL {
namespace {
	using PolicyMatchRunnerInternal::CudaOomRecoveryState;
	using PolicyMatchRunnerInternal::IsCudaOutOfMemory;
	using PolicyMatchRunnerInternal::RunWithOneCudaOomRetry;

	struct SlotActionTarget {
		int slotIdx = 0;
		int playerIdx = 0;
		int batchRow = 0;
	};

	struct SlotModelBatch {
		int runtimeIndex = 0;
		bool stochasticKickoff = false;
		std::vector<SlotActionTarget> targets;
		std::vector<float> obs;
		std::vector<uint8_t> actionMasks;
		std::vector<int> actionIndices;
	};

	struct ScopedCudaGraphCacheCleanup {
		bool enabled = false;

		void Cleanup() noexcept {
			if (!enabled)
				return;
			enabled = false;
			try {
				ModelManager::ClearCudaGraphCache();
			} catch (...) {
			}
		}

		~ScopedCudaGraphCacheCleanup() noexcept {
			Cleanup();
		}
	};

	bool AnyRuntimeUsesCudaGraph(const std::vector<PolicyMatchRuntime>& runtimes) {
		return std::any_of(runtimes.begin(), runtimes.end(), [](const PolicyMatchRuntime& runtime) {
			return runtime.useCudaGraph;
		});
	}

	void CopyCudaGraphStats(PolicyMatchTiming& timing) {
		const CudaGraphStats stats = ModelManager::GetCudaGraphStats();
		timing.cudaGraphEntries = stats.cacheEntries;
		timing.cudaGraphCaptures = stats.captures;
		timing.cudaGraphReplays = stats.replays;
		timing.cudaGraphFallbacks = stats.fallbacks;
		timing.cudaGraphVariantLimitFallbacks = stats.variantLimitFallbacks;
		timing.cudaGraphCaptureFailures = stats.captureFailures;
	}

	void CopyCudaGraphStats(PolicyMatchRunnerStatus& status) {
		const CudaGraphStats stats = ModelManager::GetCudaGraphStats();
		status.cudaGraphEntries = stats.cacheEntries;
		status.cudaGraphCaptures = stats.captures;
		status.cudaGraphReplays = stats.replays;
		status.cudaGraphFallbacks = stats.fallbacks;
		status.cudaGraphVariantLimitFallbacks = stats.variantLimitFallbacks;
		status.cudaGraphCaptureFailures = stats.captureFailures;
	}

	struct PolicyMatchSlot : PolicyMatchTask {
		bool active = false;
		bool ballTouched = false;
		int steps = 0;
		int arenaTeamSize = 0;
		int arenaBlueKey = -1;
		int arenaOrangeKey = -1;
		void* userInfo = nullptr;
		std::array<int, 2> sideRuntimeGroups = { -1, -1 };
		std::array<RLGC::ObsBuilder*, 2> sideObsBuilders = { nullptr, nullptr };
		std::array<RLGC::ActionParser*, 2> sideActionParsers = { nullptr, nullptr };
		std::array<bool, 2> sideUsesRuntimeActionOverride = { false, false };
		bool usesRuntimeActionOverride = false;
		std::map<int, PolicyMatchRuntimeComponents> runtimeComponentsByGroup;
		std::unique_ptr<RocketSim::Arena> arena;
		std::vector<std::unique_ptr<RLGC::Reward>> ownedRewards;
		std::vector<std::unique_ptr<RLGC::TerminalCondition>> ownedTerminalConditions;
		std::unique_ptr<RLGC::StateSetter> ownedStateSetter;
		std::unique_ptr<RocketSim::GameEventTracker> eventTracker;
		RLGC::GameState state = {};
		RLGC::GameState prevState = {};
		// Variable-timing matches need side-specific observation windows so slow
		// models do not lose touch/event history to faster opponents' updates.
		std::array<RLGC::GameState, 2> sideObsStates = {};
		std::array<RLGC::GameState, 2> sidePrevObsStates = {};
		std::array<std::vector<RLGC::PlayerEventState>, 2> sideEventStates = {};
		std::vector<RLGC::Action> actions;
		std::vector<RLGC::Action> obsActions;
		std::vector<RLGC::Action> pendingActions;
		std::vector<std::vector<RLGC::ActionMacro>> actionMacros;
		std::vector<std::vector<RLGC::ActionMacro>> pendingActionMacros;
		std::vector<std::vector<RLGC::ActionMacro>> activeActionMacros;
		std::vector<int> activeMacroIndices;
		std::vector<int> activeMacroNextTicks;
		std::vector<int> playerBatchIndices;
		std::vector<int> playerBatchRows;
		std::array<int, 2> sideNextDecisionTicks = { 0, 0 };
		std::array<int, 2> sidePendingApplyTicks = { -1, -1 };
		int physicsTicks = 0;
		RLGC::FList obsScratch;
		std::vector<uint8_t> actionMaskScratch;
	};

	void UpdateBallTouched(PolicyMatchSlot& slot) {
		if (slot.ballTouched)
			return;

		// Read the arena to include contacts during action delay; latch until slot reuse.
		for (RocketSim::Car* car : slot.arena->_cars) {
			if (car->GetState().ballHitInfo.isValid) {
				slot.ballTouched = true;
				return;
			}
		}
	}

	double SecondsSince(std::chrono::steady_clock::time_point start) {
		return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
	}

	std::vector<RocketSim::Car*> SortedCars(RocketSim::Arena* arena) {
		std::vector<RocketSim::Car*> cars(arena->_cars.begin(), arena->_cars.end());
		std::sort(cars.begin(), cars.end(), [](const RocketSim::Car* a, const RocketSim::Car* b) {
			if (a->team != b->team)
				return (int)a->team < (int)b->team;
			return a->id < b->id;
		});
		return cars;
	}

	void CanonicalizeKickoffAssignments(RocketSim::Arena* arena) {
		std::map<RocketSim::Team, std::vector<RocketSim::CarState>> statesByTeam;
		for (RocketSim::Car* car : arena->_cars)
			statesByTeam[car->team].push_back(car->GetState());

		auto stateLess = [](const RocketSim::CarState& a, const RocketSim::CarState& b) {
			for (int axis = 0; axis < 3; axis++) {
				if (a.pos[axis] != b.pos[axis])
					return a.pos[axis] < b.pos[axis];
			}
			return a.boost < b.boost;
		};
		for (auto& [team, states] : statesByTeam)
			std::sort(states.begin(), states.end(), stateLess);

		std::map<RocketSim::Team, int> nextByTeam;
		for (RocketSim::Car* car : SortedCars(arena))
			car->SetState(statesByTeam[car->team][nextByTeam[car->team]++]);
	}

	RocketSim::Vec RandomUnitVector() {
		return RLGC::Math::RandVec(
			RocketSim::Vec(-1, -1, -1),
			RocketSim::Vec(1, 1, 1)).Normalized();
	}

	void ResetToSeededFuzzedKickoff(RocketSim::Arena* arena, int kickoffSeed) {
		arena->ResetToRandomKickoff(kickoffSeed);
		CanonicalizeKickoffAssignments(arena);

		// Fuzz is assigned after canonicalization so each sorted car receives the
		// same random draws even when RocketSim's pointer-order storage changes.
		for (RocketSim::Car* car : SortedCars(arena)) {
			RocketSim::CarState state = car->GetState();
			for (int axis = 0; axis < 3; axis++) {
				state.pos[axis] += RocketSim::Math::RandFloat(
					-RLGC::FuzzedKickoffState::FUZZ_POS_RANGE,
					RLGC::FuzzedKickoffState::FUZZ_POS_RANGE);
			}
			car->SetState(state);
		}
	}

	void ResetToSeededRandomState(RocketSim::Arena* arena, int kickoffSeed) {
		// This explicit seed keeps the setup/reset portion independent from the
		// caller RNG. All sampled states below are assigned in stable car order.
		arena->ResetToRandomKickoff(kickoffSeed);

		constexpr float
			X_MAX = 3500,
			Y_MAX = 4000,
			Z_MAX = 1820,
			CAR_Z_MIN = 150,
			PITCH_MAX = M_PI / 2,
			YAW_MAX = M_PI,
			ROLL_MAX = M_PI,
			ANGVEL_MAX = 5.5f;

		RocketSim::BallState ballState = {};
		ballState.pos = RLGC::Math::RandVec(
			RocketSim::Vec(-X_MAX, -Y_MAX, RLGC::CommonValues::BALL_RADIUS),
			RocketSim::Vec(X_MAX, Y_MAX, Z_MAX));
		ballState.vel = RandomUnitVector() * RocketSim::Math::RandFloat(0, 4000);
		ballState.angVel = RLGC::Math::RandVec(
			RocketSim::Vec(-4, -4, -4),
			RocketSim::Vec(4, 4, 4));
		arena->ball->SetState(ballState);

		for (RocketSim::Car* car : SortedCars(arena)) {
			RocketSim::CarState carState = {};
			carState.pos = RLGC::Math::RandVec(
				RocketSim::Vec(-X_MAX, -Y_MAX, CAR_Z_MIN),
				RocketSim::Vec(X_MAX, Y_MAX, Z_MAX));
			carState.vel = RandomUnitVector()
				* RocketSim::Math::RandFloat(0, RocketSim::RLConst::CAR_MAX_SPEED);
			carState.angVel = RandomUnitVector() * ANGVEL_MAX;

			RocketSim::Angle angle(
				RocketSim::Math::RandFloat(-YAW_MAX, YAW_MAX),
				RocketSim::Math::RandFloat(-PITCH_MAX, PITCH_MAX),
				RocketSim::Math::RandFloat(-ROLL_MAX, ROLL_MAX));
			const bool onGround = RocketSim::Math::RandFloat() > 0.5f;
			if (onGround) {
				carState.pos.z = 17;
				angle.pitch = angle.roll = 0;
				carState.vel.z = 0;
				carState.angVel = {};
			}

			carState.rotMat = angle.ToRotMat();
			carState.boost = RocketSim::Math::RandFloat(0, 100);
			car->SetState(carState);
		}
	}

	void ResetStepContactState(RocketSim::Arena* arena) {
		for (RocketSim::Car* car : arena->_cars)
			car->ResetStepContactState();
	}

	template<bool RLGC::PlayerEventState::* DATA_VAR>
	void SetSlotPlayerEvent(RocketSim::Car* car, void* userInfoPtr) {
		if (!car || !userInfoPtr)
			return;
		auto* slot = (PolicyMatchSlot*)userInfoPtr;
		for (RLGC::Player& player : slot->state.players) {
			if (player.carId == car->id) {
				(player.eventState.*DATA_VAR) = true;
				return;
			}
		}
	}

	void SlotShotEventCallback(RocketSim::Arena*, RocketSim::Car* shooter, RocketSim::Car* passer, void* userInfo) {
		SetSlotPlayerEvent<&RLGC::PlayerEventState::shot>(shooter, userInfo);
		SetSlotPlayerEvent<&RLGC::PlayerEventState::shotPass>(passer, userInfo);
	}

	void SlotGoalEventCallback(RocketSim::Arena*, RocketSim::Car* scorer, RocketSim::Car* passer, void* userInfo) {
		SetSlotPlayerEvent<&RLGC::PlayerEventState::goal>(scorer, userInfo);
		SetSlotPlayerEvent<&RLGC::PlayerEventState::assist>(passer, userInfo);
	}

	void SlotSaveEventCallback(RocketSim::Arena*, RocketSim::Car* saver, void* userInfo) {
		SetSlotPlayerEvent<&RLGC::PlayerEventState::save>(saver, userInfo);
	}

	void SlotBumpCallback(RocketSim::Arena*, RocketSim::Car* bumper, RocketSim::Car* victim, bool isDemo, void* userInfo) {
		if (!bumper || !victim || bumper->team == victim->team)
			return;
		SetSlotPlayerEvent<&RLGC::PlayerEventState::bump>(bumper, userInfo);
		SetSlotPlayerEvent<&RLGC::PlayerEventState::bumped>(victim, userInfo);
		if (isDemo) {
			SetSlotPlayerEvent<&RLGC::PlayerEventState::demo>(bumper, userInfo);
			SetSlotPlayerEvent<&RLGC::PlayerEventState::demoed>(victim, userInfo);
		}
	}

	template <typename T>
	void TensorToVecInto(torch::Tensor tensor, std::vector<T>& out) {
		assert(tensor.dim() == 1);
		tensor = tensor.contiguous().cpu().detach().to(torch::CppTypeToScalarType<T>());
		const int64_t size = tensor.size(0);
		out.resize((size_t)size);
		if (size <= 0)
			return;
		T* data = tensor.data_ptr<T>();
		std::copy(data, data + size, out.begin());
	}

	void RecoverCudaMemoryAfterOom() noexcept {
#ifdef RG_CUDA_SUPPORT
		// Drop captured graph pools before flushing allocator blocks. Keeping the
		// order is important because graph-owned blocks are not reusable by eager inference.
		C10_CUDA_CLEAR_ERROR();
		try {
			ModelManager::ClearCudaGraphCache();
		} catch (...) {
		}
		try {
			c10::cuda::CUDACachingAllocator::emptyCache();
		} catch (...) {
		}
		C10_CUDA_CLEAR_ERROR();
#endif
	}

	void InferModelBatch(
		const PolicyMatchRunnerParams& params,
		const PolicyMatchRuntime& runtime,
		SlotModelBatch& batch,
		PolicyMatchTiming& timing,
		CudaOomRecoveryState& recoveryState) {
		auto attempt = [&](bool useCudaGraph) {
			auto transferStart = std::chrono::steady_clock::now();
			torch::Tensor tObsCpu = torch::from_blob(
				batch.obs.data(),
				{ (int64_t)batch.targets.size(), runtime.obsSize },
				torch::TensorOptions().dtype(torch::kFloat32));
			torch::Tensor tMasksCpu = torch::from_blob(
				batch.actionMasks.data(),
				{ (int64_t)batch.targets.size(), runtime.numActions },
				torch::TensorOptions().dtype(torch::kUInt8));
			torch::Tensor tObs = tObsCpu.to(params.device, true);
			torch::Tensor tMasks = tMasksCpu.to(params.device, true);
			timing.gpuTransferSeconds += SecondsSince(transferStart);

			auto inferenceStart = std::chrono::steady_clock::now();
			const bool inputsHavePersistentStorage = false;
			torch::Tensor tActions;
			ModelManager::InferActionsFromModelsValidRows(
				*runtime.models,
				tObs,
				tMasks,
				batch.stochasticKickoff ? false : runtime.deterministic,
				batch.stochasticKickoff ? runtime.kickoffTemperature : runtime.policyTemperature,
				runtime.useHalfPrecision,
				&tActions,
				nullptr,
				runtime.minActionProbability,
				useCudaGraph,
				CudaGraphBatchPolicy::PowerOfTwo,
				inputsHavePersistentStorage,
				(int64_t)batch.targets.size(), runtime.actionGenerator);

			TensorToVecInto(tActions, batch.actionIndices);
			if (batch.actionIndices.size() != batch.targets.size())
				throw std::runtime_error("PolicyMatchRunner: inference returned an unexpected action count.");
			timing.inferenceSeconds += SecondsSince(inferenceStart);
		};

		const bool recovered = RunWithOneCudaOomRetry(
			params.device.is_cuda(),
			runtime.useCudaGraph,
			recoveryState,
			attempt,
			IsCudaOutOfMemory,
			RecoverCudaMemoryAfterOom);
		if (recovered) {
			timing.cudaOomRecoveries++;
			timing.cudaGraphsDisabledAfterOom = true;
		}
	}

	void ValidateParams(const PolicyMatchRunnerParams& params) {
		if (params.runtimes.empty())
			throw std::invalid_argument("PolicyMatchRunner: at least one runtime is required.");
		if (params.numSlots <= 0)
			throw std::invalid_argument("PolicyMatchRunner: numSlots must be positive.");
		if (params.tickSkip <= 0)
			throw std::invalid_argument("PolicyMatchRunner: tickSkip must be positive.");
		if (params.actionDelay < 0 || params.actionDelay > params.tickSkip)
			throw std::invalid_argument("PolicyMatchRunner: actionDelay must be between 0 and tickSkip.");
		if (params.noGoalSteps <= 0)
			throw std::invalid_argument("PolicyMatchRunner: noGoalSteps must be positive.");
		if (!params.makeArena)
			throw std::invalid_argument("PolicyMatchRunner: makeArena callback is required.");
		if (!params.fillPendingTasks)
			throw std::invalid_argument("PolicyMatchRunner: fillPendingTasks callback is required.");
		if (!params.isSchedulingComplete)
			throw std::invalid_argument("PolicyMatchRunner: isSchedulingComplete callback is required.");
		if (!params.onFinished)
			throw std::invalid_argument("PolicyMatchRunner: onFinished callback is required.");

		for (int i = 0; i < (int)params.runtimes.size(); i++) {
			const PolicyMatchRuntime& runtime = params.runtimes[i];
			if (runtime.stochasticKickoff
				&& (!std::isfinite(runtime.kickoffTemperature) || runtime.kickoffTemperature <= 0.0f))
				throw std::invalid_argument("PolicyMatchRunner: kickoff temperature must be positive and finite.");

			if (!runtime.models)
				throw std::invalid_argument("PolicyMatchRunner: runtime " + std::to_string(i) + " has no model set.");
			if (runtime.obsSize <= 0 || runtime.numActions <= 0)
				throw std::invalid_argument("PolicyMatchRunner: runtime " + std::to_string(i) + " has invalid tensor shape.");
			if (runtime.tickSkip <= 0)
				throw std::invalid_argument("PolicyMatchRunner: runtime " + std::to_string(i) + " has invalid tick skip.");
			if (runtime.actionDelay < 0 || runtime.actionDelay > runtime.tickSkip)
				throw std::invalid_argument("PolicyMatchRunner: runtime " + std::to_string(i) + " has invalid action delay.");
			if (!runtime.makeRuntimeComponents)
				throw std::invalid_argument("PolicyMatchRunner: runtime " + std::to_string(i) + " is missing runtime factories.");
		}
	}

	void EnsureTaskValid(const PolicyMatchRunnerParams& params, const PolicyMatchTask& task) {
		if (task.blueModel < 0 || task.blueModel >= (int)params.runtimes.size()
			|| task.orangeModel < 0 || task.orangeModel >= (int)params.runtimes.size()) {
			throw std::runtime_error("PolicyMatchRunner: task references an invalid model index.");
		}
		if (task.teamSize <= 0)
			throw std::runtime_error("PolicyMatchRunner: task team size must be positive.");
	}

	int TaskBlueArenaKey(const PolicyMatchTask& task) {
		return task.blueArenaKey >= 0 ? task.blueArenaKey : task.blueModel;
	}

	int TaskOrangeArenaKey(const PolicyMatchTask& task) {
		return task.orangeArenaKey >= 0 ? task.orangeArenaKey : task.orangeModel;
	}

	bool TaskUsesArenaLayout(
		const PolicyMatchTask& task,
		int teamSize,
		int blueArenaKey,
		int orangeArenaKey) {
		return task.teamSize == teamSize
			&& TaskBlueArenaKey(task) == blueArenaKey
			&& TaskOrangeArenaKey(task) == orangeArenaKey;
	}

	int SideIndex(RocketSim::Team team) {
		return team == RocketSim::Team::BLUE ? 0 : 1;
	}

	RocketSim::Team SideTeam(int sideIdx) {
		return sideIdx == 0 ? RocketSim::Team::BLUE : RocketSim::Team::ORANGE;
	}

	int SideModelIndex(const PolicyMatchSlot& slot, int sideIdx) {
		return sideIdx == 0 ? slot.blueModel : slot.orangeModel;
	}

	bool IsPlayerOnSide(const RLGC::Player& player, int sideIdx) {
		return player.team == SideTeam(sideIdx);
	}

	void ApplySlotControls(PolicyMatchSlot& slot);
	void StepRuntimeActionDelay(PolicyMatchSlot& slot, int tickCount);

	std::vector<RLGC::ActionMacro> BuildActionMacros(
		RLGC::ActionParser& actionParser,
		int actionIdx,
		const RLGC::Player& player,
		const RLGC::GameState& state,
		int tickCount) {
		if (tickCount <= 0)
			return { { actionParser.ParseAction(actionIdx, player, state), 0 } };

		std::vector<RLGC::ActionMacro> macros;
		actionParser.ParseActionMacros(actionIdx, player, state, tickCount, macros);
		if (macros.empty())
			macros.push_back({ actionParser.ParseAction(actionIdx, player, state), tickCount });

		int macroTicks = 0;
		for (const RLGC::ActionMacro& macro : macros) {
			if (macro.ticks < 0)
				throw std::runtime_error("PolicyMatchRunner: action parser returned a macro with negative ticks.");
			macroTicks += macro.ticks;
		}
		if (macroTicks != tickCount)
			throw std::runtime_error("PolicyMatchRunner: action parser returned macros that do not match the action window.");

		return macros;
	}

	bool HasSegmentedMacro(const std::vector<RLGC::ActionMacro>& macros, int tickCount) {
		return macros.size() != 1 || (!macros.empty() && macros.front().ticks != tickCount);
	}

	bool RequiresPerTickActionOverride(PolicyMatchSlot& slot, const RLGC::GameState& state) {
		if (!slot.usesRuntimeActionOverride)
			return false;

		bool required = false;
		for (int playerIdx = 0; playerIdx < (int)state.players.size(); playerIdx++) {
			const int sideIdx = SideIndex(state.players[playerIdx].team);
			if (!slot.sideUsesRuntimeActionOverride[sideIdx])
				continue;
			RLGC::ActionParser* parser = slot.sideActionParsers[sideIdx];
			required = parser->RequiresPerTickActionOverride(state.players[playerIdx], state) || required;
		}
		return required;
	}

	void ApplyActionMacrosWithinWindow(
		PolicyMatchSlot& slot,
		const std::vector<std::vector<RLGC::ActionMacro>>& macrosByPlayer,
		int tickCount) {
		const bool hasSegmentedMacro = std::any_of(
			macrosByPlayer.begin(),
			macrosByPlayer.end(),
			[tickCount](const std::vector<RLGC::ActionMacro>& macros) {
				return HasSegmentedMacro(macros, tickCount);
			});
		const bool hasRuntimeOverride = RequiresPerTickActionOverride(slot, slot.state);

		if (!hasSegmentedMacro && !hasRuntimeOverride) {
			ApplySlotControls(slot);
			slot.arena->Step(tickCount);
			return;
		}

		std::vector<int> macroIndices(macrosByPlayer.size(), 0);
		std::vector<int> macroTicksLeft(macrosByPlayer.size(), 0);
		for (int i = 0; i < (int)macrosByPlayer.size(); i++)
			if (!macrosByPlayer[i].empty())
				macroTicksLeft[i] = macrosByPlayer[i].front().ticks;

		if (!hasRuntimeOverride) {
			for (int tick = 0; tick < tickCount; tick++) {
				auto carItr = slot.arena->_cars.begin();
				for (int i = 0; i < (int)macrosByPlayer.size(); i++, carItr++) {
					const std::vector<RLGC::ActionMacro>& macros = macrosByPlayer[i];
					while (macroTicksLeft[i] <= 0 && macroIndices[i] + 1 < (int)macros.size()) {
						macroIndices[i]++;
						macroTicksLeft[i] = macros[macroIndices[i]].ticks;
					}
					(*carItr)->controls = (RocketSim::CarControls)macros[macroIndices[i]].action;
				}

				slot.arena->Step(1);
				for (int i = 0; i < (int)macroTicksLeft.size(); i++)
					if (macroTicksLeft[i] > 0)
						macroTicksLeft[i]--;
			}
			return;
		}

		RLGC::GameState controlState = slot.state;
		RLGC::GameState previousControlState;
		std::vector<RLGC::Action> appliedActions(controlState.players.size());
		for (int tick = 0; tick < tickCount; tick++) {
			auto carItr = slot.arena->_cars.begin();
			for (int i = 0; i < (int)macrosByPlayer.size(); i++, carItr++) {
				const std::vector<RLGC::ActionMacro>& macros = macrosByPlayer[i];
				while (macroTicksLeft[i] <= 0 && macroIndices[i] + 1 < (int)macros.size()) {
					macroIndices[i]++;
					macroTicksLeft[i] = macros[macroIndices[i]].ticks;
				}
				const int sideIdx = SideIndex(controlState.players[i].team);
				appliedActions[i] = macros[macroIndices[i]].action;
				if (slot.sideUsesRuntimeActionOverride[sideIdx]) {
					appliedActions[i] = slot.sideActionParsers[sideIdx]->OverrideRuntimeAction(
						appliedActions[i],
						controlState.players[i],
						controlState,
						1);
				}
				(*carItr)->controls = (RocketSim::CarControls)appliedActions[i];
			}

			slot.arena->Step(1);

			for (int i = 0; i < (int)macroTicksLeft.size(); i++)
				if (macroTicksLeft[i] > 0)
					macroTicksLeft[i]--;

			if (tick + 1 < tickCount) {
				previousControlState = controlState;
				controlState.ResetBeforeStep();
				controlState.UpdateFromArena(slot.arena.get(), appliedActions, &previousControlState);
				controlState.userInfo = slot.userInfo;
			}
		}
		for (int i = 0; i < (int)slot.state.players.size(); i++) {
			const int sideIdx = SideIndex(slot.state.players[i].team);
			if (slot.sideUsesRuntimeActionOverride[sideIdx])
				slot.obsActions[i] = appliedActions[i];
		}
	}

	void AdvancePlayerMacroToTick(PolicyMatchSlot& slot, int playerIdx) {
		while (playerIdx >= 0
			&& playerIdx < (int)slot.activeActionMacros.size()
			&& slot.activeMacroNextTicks[playerIdx] >= 0
			&& slot.activeMacroNextTicks[playerIdx] <= slot.physicsTicks) {
			const int nextIdx = slot.activeMacroIndices[playerIdx] + 1;
			if (nextIdx >= (int)slot.activeActionMacros[playerIdx].size()) {
				slot.activeActionMacros[playerIdx].clear();
				slot.activeMacroIndices[playerIdx] = 0;
				slot.activeMacroNextTicks[playerIdx] = -1;
				return;
			}

			const int nextStartTick = slot.activeMacroNextTicks[playerIdx];
			slot.activeMacroIndices[playerIdx] = nextIdx;
			const RLGC::ActionMacro& macro = slot.activeActionMacros[playerIdx][nextIdx];
			slot.actions[playerIdx] = macro.action;
			slot.activeMacroNextTicks[playerIdx] = nextStartTick + macro.ticks;
		}
	}

	void SetSideControlsFromPending(PolicyMatchSlot& slot, int sideIdx) {
		for (int playerIdx = 0; playerIdx < (int)slot.state.players.size(); playerIdx++) {
			if (!IsPlayerOnSide(slot.state.players[playerIdx], sideIdx))
				continue;

			std::vector<RLGC::ActionMacro>& pendingMacros = slot.pendingActionMacros[playerIdx];
			if (!pendingMacros.empty()) {
				slot.activeActionMacros[playerIdx] = std::move(pendingMacros);
				pendingMacros.clear();
				slot.activeMacroIndices[playerIdx] = 0;
				slot.actions[playerIdx] = slot.activeActionMacros[playerIdx].front().action;
				slot.obsActions[playerIdx] = slot.activeActionMacros[playerIdx].back().action;
				slot.state.players[playerIdx].prevAction = slot.obsActions[playerIdx];
				slot.activeMacroNextTicks[playerIdx] =
					slot.physicsTicks + slot.activeActionMacros[playerIdx].front().ticks;
				AdvancePlayerMacroToTick(slot, playerIdx);
			} else {
				slot.actions[playerIdx] = slot.pendingActions[playerIdx];
				slot.obsActions[playerIdx] = slot.pendingActions[playerIdx];
				slot.state.players[playerIdx].prevAction = slot.obsActions[playerIdx];
				slot.activeActionMacros[playerIdx].clear();
				slot.activeMacroIndices[playerIdx] = 0;
				slot.activeMacroNextTicks[playerIdx] = -1;
			}
		}
	}

	void ApplySlotControls(PolicyMatchSlot& slot) {
		auto carItr = slot.arena->_cars.begin();
		if (!slot.usesRuntimeActionOverride) {
			for (int i = 0; i < (int)slot.state.players.size(); i++, carItr++)
				(*carItr)->controls = (RocketSim::CarControls)slot.actions[i];
			return;
		}

		for (int i = 0; i < (int)slot.state.players.size(); i++, carItr++) {
			const int sideIdx = SideIndex(slot.state.players[i].team);
			RLGC::Action output = slot.actions[i];
			if (slot.sideUsesRuntimeActionOverride[sideIdx]) {
				output = slot.sideActionParsers[sideIdx]->OverrideRuntimeAction(
					output,
					slot.state.players[i],
					slot.state,
					1);
				slot.obsActions[i] = output;
			}
			(*carItr)->controls = (RocketSim::CarControls)output;
		}
	}

	void StepRuntimeActionDelay(PolicyMatchSlot& slot, int tickCount) {
		if (tickCount <= 0)
			return;
		if (!RequiresPerTickActionOverride(slot, slot.state)) {
			slot.arena->Step(tickCount);
			return;
		}

		RLGC::GameState controlState = slot.state;
		RLGC::GameState previousControlState;
		std::vector<RLGC::Action> appliedActions(controlState.players.size());
		for (int tick = 0; tick < tickCount; tick++) {
			auto carItr = slot.arena->_cars.begin();
			for (int i = 0; i < (int)controlState.players.size(); i++, carItr++) {
				const int sideIdx = SideIndex(controlState.players[i].team);
				appliedActions[i] = slot.actions[i];
				if (slot.sideUsesRuntimeActionOverride[sideIdx]) {
					appliedActions[i] = slot.sideActionParsers[sideIdx]->OverrideRuntimeAction(
						appliedActions[i],
						controlState.players[i],
						controlState,
						1);
				}
				(*carItr)->controls = (RocketSim::CarControls)appliedActions[i];
			}
			slot.arena->Step(1);

			if (tick + 1 < tickCount) {
				previousControlState = controlState;
				controlState.ResetBeforeStep();
				controlState.UpdateFromArena(slot.arena.get(), appliedActions, &previousControlState);
				controlState.userInfo = slot.userInfo;
			}
		}
		slot.obsActions = std::move(appliedActions);
	}

	void MergePlayerEventState(RLGC::PlayerEventState& dst, const RLGC::PlayerEventState& src) {
		dst.goal = dst.goal || src.goal;
		dst.save = dst.save || src.save;
		dst.assist = dst.assist || src.assist;
		dst.shot = dst.shot || src.shot;
		dst.shotPass = dst.shotPass || src.shotPass;
		dst.bump = dst.bump || src.bump;
		dst.bumped = dst.bumped || src.bumped;
		dst.demo = dst.demo || src.demo;
		dst.demoed = dst.demoed || src.demoed;
	}

	void AccumulateSlotEvents(PolicyMatchSlot& slot) {
		for (int sideIdx = 0; sideIdx < 2; sideIdx++) {
			if (slot.sideEventStates[sideIdx].size() != slot.state.players.size())
				slot.sideEventStates[sideIdx].assign(slot.state.players.size(), RLGC::PlayerEventState{});
			for (int playerIdx = 0; playerIdx < (int)slot.state.players.size(); playerIdx++)
				MergePlayerEventState(slot.sideEventStates[sideIdx][playerIdx], slot.state.players[playerIdx].eventState);
		}
	}

	void RefreshSideObsState(PolicyMatchSlot& slot, int sideIdx) {
		RLGC::GameState& obsState = slot.sideObsStates[sideIdx];
		if (obsState.lastTickCount != slot.arena->tickCount) {
			slot.sidePrevObsStates[sideIdx] = obsState;
			obsState.ResetBeforeStep();
			obsState.UpdateFromArena(slot.arena.get(), slot.obsActions, &slot.sidePrevObsStates[sideIdx]);
			obsState.userInfo = slot.userInfo;
		}

		if (slot.sideEventStates[sideIdx].size() == obsState.players.size()) {
			for (int playerIdx = 0; playerIdx < (int)obsState.players.size(); playerIdx++)
				obsState.players[playerIdx].eventState = slot.sideEventStates[sideIdx][playerIdx];
			std::fill(
				slot.sideEventStates[sideIdx].begin(),
				slot.sideEventStates[sideIdx].end(),
				RLGC::PlayerEventState{});
		}
	}

	void EnsureRocketSimInitialized(RocketSim::EngineId engine) {
		RocketSim::RequireEngine(engine);
		if (RocketSim::GetStage() != RocketSim::RocketSimStage::INITIALIZED) {
			RocketSim::Init(RocketSim::CollisionMeshesFolder(), true);
		}

		if (!RocketSim::HasCollisionMeshes(engine, RocketSim::GameMode::SOCCAR)) {
			throw std::runtime_error(
				"PolicyMatchRunner: " + RocketSim::GetEngineInfo(engine).displayName + " is initialized without soccar collision meshes.");
		}
	}
}

std::unique_ptr<RocketSim::Arena> MakePolicyMatchArena(
	int teamSize,
	const RocketSim::CarConfig& blueCarConfig,
	const RocketSim::CarConfig& orangeCarConfig,
	RocketSim::EngineId engine) {
	EnsureRocketSimInitialized(engine);
	RocketSim::ArenaConfig arenaConfig = {};
	arenaConfig.engine = engine;
	std::unique_ptr<RocketSim::Arena> arena(RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR, arenaConfig));
	for (int i = 0; i < teamSize; i++)
		arena->AddCar(RocketSim::Team::BLUE, blueCarConfig);
	for (int i = 0; i < teamSize; i++)
		arena->AddCar(RocketSim::Team::ORANGE, orangeCarConfig);
	return arena;
}

PolicyMatchSnapshot CapturePolicyMatchSnapshot(RocketSim::Arena* arena) {
	PolicyMatchSnapshot snapshot = {};
	snapshot.ball = arena->ball->GetState();
	for (RocketSim::Car* car : SortedCars(arena))
		snapshot.cars.push_back({ car->team, car->GetState() });
	return snapshot;
}

void ApplyPolicyMatchSnapshot(RocketSim::Arena* arena, const PolicyMatchSnapshot& snapshot) {
	arena->ball->SetState(snapshot.ball);
	std::vector<RocketSim::Car*> cars = SortedCars(arena);
	if (cars.size() != snapshot.cars.size())
		throw std::runtime_error("PolicyMatchRunner: snapshot car count does not match arena car count.");

	std::map<RocketSim::Team, std::vector<const PolicyMatchCarSnapshot*>> sourceByTeam;
	for (const PolicyMatchCarSnapshot& car : snapshot.cars)
		sourceByTeam[car.team].push_back(&car);

	std::map<RocketSim::Team, int> nextByTeam;
	for (RocketSim::Car* car : cars) {
		int& next = nextByTeam[car->team];
		auto& source = sourceByTeam[car->team];
		if (next >= (int)source.size())
			throw std::runtime_error("PolicyMatchRunner: snapshot team layout does not match arena team layout.");
		car->SetState(source[next++]->state);
		// NOTE: CarState excludes live controls. Reused arenas must not carry the
		// previous episode's input through the new episode's action-delay window.
		car->controls = {};
	}

	for (RocketSim::BoostPad* pad : arena->_boostPads)
		pad->SetState(RocketSim::BoostPadState());
}

PolicyMatchSnapshot SamplePolicyMatchStartState(
	PolicyMatchStartType startType,
	int teamSize,
	const RocketSim::CarConfig& carConfig,
	RocketSim::EngineId engine) {
	std::unique_ptr<RocketSim::Arena> arena = MakePolicyMatchArena(teamSize, carConfig, carConfig, engine);
	if (startType == PolicyMatchStartType::Kickoff) {
		RLGC::FuzzedKickoffState setter;
		setter.ResetArena(arena.get());
	} else {
		const bool randBallSpeed = true;
		const bool randCarSpeed = true;
		const bool carsOnGround = false;
		RLGC::RandomState setter(randBallSpeed, randCarSpeed, carsOnGround);
		setter.ResetArena(arena.get());
	}
	return CapturePolicyMatchSnapshot(arena.get());
}

PolicyMatchSnapshot SamplePolicyMatchStartState(
	PolicyMatchStartType startType,
	int teamSize,
	const RocketSim::CarConfig& carConfig,
	uint64_t seed,
	RocketSim::EngineId engine) {
	std::default_random_engine& randEngine = RocketSim::Math::GetRandEngine();
	const std::default_random_engine savedRandEngine = randEngine;
	std::seed_seq seedSequence{
		(uint32_t)(seed & 0xffffffffULL),
		(uint32_t)(seed >> 32)
	};
	randEngine.seed(seedSequence);
	try {
		std::unique_ptr<RocketSim::Arena> arena = MakePolicyMatchArena(teamSize, carConfig, carConfig, engine);
		const int kickoffSeed = (int)(randEngine() % (std::numeric_limits<int>::max)());
		if (startType == PolicyMatchStartType::Kickoff)
			ResetToSeededFuzzedKickoff(arena.get(), kickoffSeed);
		else
			ResetToSeededRandomState(arena.get(), kickoffSeed);
		PolicyMatchSnapshot snapshot = CapturePolicyMatchSnapshot(arena.get());
		randEngine = savedRandEngine;
		return snapshot;
	} catch (...) {
		randEngine = savedRandEngine;
		throw;
	}
}

PolicyMatchRunnerResult RunPolicyMatchSlots(const PolicyMatchRunnerParams& params) {
	ValidateParams(params);
	RG_NO_GRAD;

	ScopedCudaGraphCacheCleanup graphCleanup{ AnyRuntimeUsesCudaGraph(params.runtimes) };
	ModelManager::ResetCudaGraphStats();
	PolicyMatchRunnerResult result = {};
	PolicyMatchTiming& timing = result.timing;
	CudaOomRecoveryState cudaOomRecovery;
	const int numSlots = std::max(1, params.numSlots);
	const int slotThreadTarget = std::max(1, RLGC::g_ThreadPool.GetNumThreads() * 2);
	std::vector<PolicyMatchTask> pendingTasks;
	std::vector<PolicyMatchSlot> slots(numSlots);
	const int numModels = (int)params.runtimes.size();
	const bool stochasticKickoff = std::any_of(params.runtimes.begin(), params.runtimes.end(),
		[](const PolicyMatchRuntime& runtime) { return runtime.stochasticKickoff; });

	// A model can serve opening and post-touch decisions in the same iteration.
	std::vector<SlotModelBatch> modelBatches(numModels * (stochasticKickoff ? 2 : 1));
	std::vector<int> activeIterationSlots;
	activeIterationSlots.reserve(numSlots);

	const int maxTargetsPerModel = numSlots * 3;
	for (int batchIdx = 0; batchIdx < (int)modelBatches.size(); batchIdx++) {
		SlotModelBatch& batch = modelBatches[batchIdx];
		batch.runtimeIndex = batchIdx % numModels;
		batch.stochasticKickoff = batchIdx >= numModels;
		const PolicyMatchRuntime& runtime = params.runtimes[batch.runtimeIndex];
		batch.targets.reserve(maxTargetsPerModel);
		batch.obs.reserve((size_t)maxTargetsPerModel * runtime.obsSize);
		batch.actionMasks.reserve((size_t)maxTargetsPerModel * runtime.numActions);
		batch.actionIndices.reserve(maxTargetsPerModel);
	}

	int activeSlots = 0;
	std::map<int, int> activeSlotsByTeamSize;
	const int taskCreateBatchSize = std::min(numSlots, slotThreadTarget);
	int taskQueueTarget = taskCreateBatchSize;
	bool stopRequested = false;
	std::mutex threadExceptionMutex;
	std::exception_ptr threadException = nullptr;

	auto captureThreadException = [&](std::exception_ptr exception) {
		std::scoped_lock lock(threadExceptionMutex);
		if (!threadException)
			threadException = exception;
	};

	auto throwThreadException = [&]() {
		if (threadException)
			std::rethrow_exception(threadException);
	};

	auto runBatchedJobs = [&](const std::function<void(int)>& job, int count) {
		auto guardedJob = [&](int idx) {
			try {
				job(idx);
			} catch (...) {
				captureThreadException(std::current_exception());
			}
		};
		RLGC::g_ThreadPool.StartBatchedJobs(guardedJob, count, false);
		throwThreadException();
	};

	auto emitStatus = [&]() {
		if (!params.onStatus)
			return;

		PolicyMatchRunnerStatus status = {};
		status.totalSlots = numSlots;
		status.activeSlots = activeSlots;
		status.pendingTasks = (int)pendingTasks.size();
		status.slotArenaCreates = timing.slotArenaCreates;
		status.slotArenaRebuilds = timing.slotArenaRebuilds;
		status.slotArenaReuses = timing.slotArenaReuses;
		status.cudaOomRecoveries = timing.cudaOomRecoveries;
		status.cudaGraphsDisabledAfterOom = timing.cudaGraphsDisabledAfterOom;
		CopyCudaGraphStats(status);
		status.activeSlotsByTeamSize.assign(activeSlotsByTeamSize.begin(), activeSlotsByTeamSize.end());
		params.onStatus(status);
	};

	auto ensureSlotSideRuntime = [&](PolicyMatchSlot& slot, int sideIdx, int modelIdx) {
		const PolicyMatchRuntime& runtime = params.runtimes[modelIdx];
		const int runtimeGroup = runtime.runtimeGroup;
		if (slot.sideRuntimeGroups[sideIdx] == runtimeGroup
			&& slot.sideObsBuilders[sideIdx]
			&& slot.sideActionParsers[sideIdx]) {
			return;
		}

		PolicyMatchRuntimeComponents& components = slot.runtimeComponentsByGroup[runtimeGroup];
		if (!components.obsBuilder || !components.actionParser)
			components = runtime.makeRuntimeComponents();
		if (!components.obsBuilder || !components.actionParser)
			throw std::runtime_error("PolicyMatchRunner: runtime factory returned null.");

		slot.sideRuntimeGroups[sideIdx] = runtimeGroup;
		slot.sideObsBuilders[sideIdx] = components.obsBuilder.get();
		slot.sideActionParsers[sideIdx] = components.actionParser.get();
		slot.sideUsesRuntimeActionOverride[sideIdx] =
			components.actionParser->HasRuntimeActionOverride();
		slot.usesRuntimeActionOverride =
			slot.sideUsesRuntimeActionOverride[0]
			|| slot.sideUsesRuntimeActionOverride[1];
	};

	auto ensureSlotArena = [&](PolicyMatchSlot& slot) {
		if (slot.arena
			&& slot.arenaTeamSize == slot.teamSize
			&& slot.arenaBlueKey == TaskBlueArenaKey(slot)
			&& slot.arenaOrangeKey == TaskOrangeArenaKey(slot)) {
			return;
		}

		PolicyMatchArena created = params.makeArena(slot);
		if (!created.arena)
			throw std::runtime_error("PolicyMatchRunner: makeArena returned null.");
		if ((created.runtimeComponents.obsBuilder && !created.runtimeComponents.actionParser)
			|| (!created.runtimeComponents.obsBuilder && created.runtimeComponents.actionParser)) {
			throw std::runtime_error("PolicyMatchRunner: makeArena returned partial runtime components.");
		}

		slot.sideRuntimeGroups = { -1, -1 };
		slot.sideObsBuilders = { nullptr, nullptr };
		slot.sideActionParsers = { nullptr, nullptr };
		slot.sideUsesRuntimeActionOverride = { false, false };
		slot.usesRuntimeActionOverride = false;
		slot.arena = std::move(created.arena);
		slot.userInfo = created.userInfo;
		slot.ownedRewards = std::move(created.ownedRewards);
		slot.ownedTerminalConditions = std::move(created.ownedTerminalConditions);
		slot.ownedStateSetter = std::move(created.ownedStateSetter);
		if (created.runtimeComponents.obsBuilder)
			slot.runtimeComponentsByGroup[created.runtimeGroup] = std::move(created.runtimeComponents);
		slot.eventTracker = std::make_unique<RocketSim::GameEventTracker>();
		slot.eventTracker->SetShotCallback(SlotShotEventCallback, &slot);
		slot.eventTracker->SetGoalCallback(SlotGoalEventCallback, &slot);
		slot.eventTracker->SetSaveCallback(SlotSaveEventCallback, &slot);
		slot.arena->SetCarBumpCallback(SlotBumpCallback, &slot);
		slot.arenaTeamSize = slot.teamSize;
		slot.arenaBlueKey = TaskBlueArenaKey(slot);
		slot.arenaOrangeKey = TaskOrangeArenaKey(slot);
	};

	auto fillPendingTasksTimed = [&](int targetQueueSize) {
		targetQueueSize = std::clamp(targetQueueSize, 1, numSlots);
		auto timingStart = std::chrono::steady_clock::now();
		const size_t pendingBefore = pendingTasks.size();
		while (!stopRequested
			&& !result.cancelled
			&& !params.isSchedulingComplete()
			&& (int)(pendingTasks.size() + activeSlots) < targetQueueSize) {
			if (params.isCancelled && params.isCancelled()) {
				result.cancelled = true;
				emitStatus();
				return;
			}

			const size_t before = pendingTasks.size();
			params.fillPendingTasks(pendingTasks, activeSlots, targetQueueSize);
			for (size_t taskIdx = before; taskIdx < pendingTasks.size(); taskIdx++)
				EnsureTaskValid(params, pendingTasks[taskIdx]);
			if (pendingTasks.size() == before)
				break;
		}
		timing.schedulerSeconds += SecondsSince(timingStart);
		if (pendingTasks.size() != pendingBefore)
			emitStatus();
	};

	auto pendingTaskIndexForSlot = [&](const PolicyMatchSlot& slot, bool requireCompatible) {
		if (pendingTasks.empty())
			return -1;

		if (slot.arena) {
			for (int taskIdx = (int)pendingTasks.size() - 1; taskIdx >= 0; taskIdx--) {
				const PolicyMatchTask& task = pendingTasks[taskIdx];
				if (slot.arenaTeamSize == task.teamSize
					&& slot.arenaBlueKey == TaskBlueArenaKey(task)
					&& slot.arenaOrangeKey == TaskOrangeArenaKey(task)) {
					return taskIdx;
				}
			}
		}

		return requireCompatible ? -1 : (int)pendingTasks.size() - 1;
	};

	auto fillPendingTasksForIdleArenaLayoutsTimed = [&]() {
		if (!params.fillPendingTasksForArenaLayouts
			|| stopRequested
			|| result.cancelled
			|| params.isSchedulingComplete()) {
			return;
		}

		std::set<std::tuple<int, int, int>> pendingLayouts;
		for (const PolicyMatchTask& task : pendingTasks) {
			pendingLayouts.emplace(
				task.teamSize,
				TaskBlueArenaKey(task),
				TaskOrangeArenaKey(task));
		}

		std::vector<PolicyMatchArenaLayoutDemand> demands;
		for (const PolicyMatchSlot& slot : slots) {
			if (slot.active || !slot.arena)
				continue;
			if (params.isArenaLayoutComplete
				&& params.isArenaLayoutComplete(
					slot.arenaTeamSize,
					slot.arenaBlueKey,
					slot.arenaOrangeKey)) {
				continue;
			}

			const auto layout = std::make_tuple(
				slot.arenaTeamSize,
				slot.arenaBlueKey,
				slot.arenaOrangeKey);
			if (pendingLayouts.count(layout) > 0)
				continue;

			auto demandItr = std::find_if(
				demands.begin(),
				demands.end(),
				[&](const PolicyMatchArenaLayoutDemand& demand) {
					return demand.teamSize == slot.arenaTeamSize
						&& demand.blueArenaKey == slot.arenaBlueKey
						&& demand.orangeArenaKey == slot.arenaOrangeKey;
				});
			if (demandItr == demands.end()) {
				demands.push_back({
					slot.arenaTeamSize,
					slot.arenaBlueKey,
					slot.arenaOrangeKey,
					1
				});
			} else {
				demandItr->idleSlots++;
			}
		}
		if (demands.empty())
			return;

		auto timingStart = std::chrono::steady_clock::now();
		const size_t pendingBefore = pendingTasks.size();
		params.fillPendingTasksForArenaLayouts(pendingTasks, demands);
		for (size_t taskIdx = pendingBefore; taskIdx < pendingTasks.size(); taskIdx++)
			EnsureTaskValid(params, pendingTasks[taskIdx]);
		timing.schedulerSeconds += SecondsSince(timingStart);
		if (pendingTasks.size() != pendingBefore)
			emitStatus();
	};

	enum class SlotAssignMode {
		CompatibleArena,
		EmptyArena,
		TerminalLayout
	};

	auto pendingTaskIndexForAssignMode = [&](const PolicyMatchSlot& slot, SlotAssignMode mode) {
		if (mode == SlotAssignMode::EmptyArena)
			return slot.arena ? -1 : pendingTaskIndexForSlot(slot, false);
		if (mode == SlotAssignMode::CompatibleArena)
			return pendingTaskIndexForSlot(slot, true);
		if (mode == SlotAssignMode::TerminalLayout) {
			if (!slot.arena
				|| !params.isArenaLayoutComplete
				|| !params.isArenaLayoutComplete(slot.arenaTeamSize, slot.arenaBlueKey, slot.arenaOrangeKey)) {
				return -1;
			}
		}
		return pendingTaskIndexForSlot(slot, false);
	};

	std::vector<int> assignedSlotIndices;
	assignedSlotIndices.reserve(numSlots);

	auto assignPendingTask = [&](int slotIdx, int taskIdx) {
		auto assignStart = std::chrono::steady_clock::now();
		PolicyMatchSlot& slot = slots[slotIdx];
		static_cast<PolicyMatchTask&>(slot) = std::move(pendingTasks[taskIdx]);
		if (taskIdx != (int)pendingTasks.size() - 1)
			pendingTasks[taskIdx] = std::move(pendingTasks.back());
		pendingTasks.pop_back();

		// Reserve the slot immediately so later assignment passes do not reuse it before setup finishes.
		slot.active = true;
		assignedSlotIndices.push_back(slotIdx);
		timing.slotAssignSeconds += SecondsSince(assignStart);
		return true;
	};

	auto assignSlot = [&](int slotIdx, SlotAssignMode mode) {
		int taskIdx = pendingTaskIndexForAssignMode(slots[slotIdx], mode);
		return taskIdx >= 0 && assignPendingTask(slotIdx, taskIdx);
	};

	auto assignStarvationFallbackBatch = [&]() {
		if (pendingTasks.empty())
			return;

		const PolicyMatchTask& targetTask = pendingTasks.back();
		const int targetTeamSize = targetTask.teamSize;
		const int targetBlueArenaKey = TaskBlueArenaKey(targetTask);
		const int targetOrangeArenaKey = TaskOrangeArenaKey(targetTask);
		int activeTargetSlots = 0;
		for (const PolicyMatchSlot& slot : slots) {
			if (slot.active
				&& slot.arena
				&& slot.arenaTeamSize == targetTeamSize
				&& slot.arenaBlueKey == targetBlueArenaKey
				&& slot.arenaOrangeKey == targetOrangeArenaKey) {
				activeTargetSlots++;
			}
		}

		// NOTE: Rebuild at most one setup batch and retain at least half of the
		// existing arena layouts. The rebuilt slots all target one layout so the
		// unavoidable fallback cost creates useful parallel capacity.
		const int rebuildLimit = std::min(taskCreateBatchSize, std::max(1, numSlots / 2));
		const int availableRebuilds = rebuildLimit - activeTargetSlots;
		if (availableRebuilds <= 0)
			return;
		int assigned = 0;
		for (int slotIdx = 0; slotIdx < numSlots && assigned < availableRebuilds; slotIdx++) {
			if (slots[slotIdx].active)
				continue;

			int taskIdx = -1;
			for (int pendingIdx = (int)pendingTasks.size() - 1; pendingIdx >= 0; pendingIdx--) {
				if (TaskUsesArenaLayout(
					pendingTasks[pendingIdx],
					targetTeamSize,
					targetBlueArenaKey,
					targetOrangeArenaKey)) {
					taskIdx = pendingIdx;
					break;
				}
			}
			if (taskIdx < 0)
				break;
			assignPendingTask(slotIdx, taskIdx);
			assigned++;
		}
	};

	auto prepareAssignedSlots = [&]() {
		if (assignedSlotIndices.empty())
			return;

		auto assignStart = std::chrono::steady_clock::now();
		auto componentStart = std::chrono::steady_clock::now();
		for (int slotIdx : assignedSlotIndices) {
			const PolicyMatchSlot& slot = slots[slotIdx];
			if (slot.arena
				&& slot.arenaTeamSize == slot.teamSize
				&& slot.arenaBlueKey == TaskBlueArenaKey(slot)
				&& slot.arenaOrangeKey == TaskOrangeArenaKey(slot)) {
				timing.slotArenaReuses++;
			} else if (slot.arena) {
				timing.slotArenaRebuilds++;
			} else {
				timing.slotArenaCreates++;
			}
		}
		auto setupArena = [&](int assignedIdx) {
			ensureSlotArena(slots[assignedSlotIndices[assignedIdx]]);
		};
		runBatchedJobs(setupArena, (int)assignedSlotIndices.size());
		timing.slotArenaSetupSeconds += SecondsSince(componentStart);

		componentStart = std::chrono::steady_clock::now();
		for (int slotIdx : assignedSlotIndices) {
			PolicyMatchSlot& slot = slots[slotIdx];
			ensureSlotSideRuntime(slot, 0, slot.blueModel);
			ensureSlotSideRuntime(slot, 1, slot.orangeModel);
		}
		timing.slotSideRuntimeSeconds += SecondsSince(componentStart);

		componentStart = std::chrono::steady_clock::now();
		auto applySnapshot = [&](int assignedIdx) {
			PolicyMatchSlot& slot = slots[assignedSlotIndices[assignedIdx]];
			ApplyPolicyMatchSnapshot(slot.arena.get(), slot.snapshot);
		};
		runBatchedJobs(applySnapshot, (int)assignedSlotIndices.size());
		timing.slotSnapshotSeconds += SecondsSince(componentStart);

		componentStart = std::chrono::steady_clock::now();
		auto initState = [&](int assignedIdx) {
			PolicyMatchSlot& slot = slots[assignedSlotIndices[assignedIdx]];
			slot.state = RLGC::GameState(slot.arena.get());
			slot.state.userInfo = slot.userInfo;
			slot.prevState.MakeEmpty();
			slot.ballTouched = false;
			slot.actions.assign(slot.state.players.size(), RLGC::Action{});
			slot.obsActions.assign(slot.state.players.size(), RLGC::Action{});
			slot.actionMacros.assign(slot.state.players.size(), {});
			slot.playerBatchIndices.assign(slot.state.players.size(), -1);
			slot.playerBatchRows.assign(slot.state.players.size(), -1);
			slot.obsScratch.reserve(std::max(params.runtimes[slot.blueModel].obsSize, params.runtimes[slot.orangeModel].obsSize));
			slot.actionMaskScratch.reserve(std::max(
				params.runtimes[slot.blueModel].numActions,
				params.runtimes[slot.orangeModel].numActions));
			slot.steps = slot.resumeSteps;
		};
		runBatchedJobs(initState, (int)assignedSlotIndices.size());
		timing.slotStateInitSeconds += SecondsSince(componentStart);

		componentStart = std::chrono::steady_clock::now();
		auto resetRuntime = [&](int assignedIdx) {
			PolicyMatchSlot& slot = slots[assignedSlotIndices[assignedIdx]];
			if (slot.eventTracker)
				slot.eventTracker->ResetPersistentInfo();
			slot.sideObsBuilders[0]->Reset(slot.state);
			slot.sideObsBuilders[1]->Reset(slot.state);
			for (int sideIdx = 0; sideIdx < 2; sideIdx++)
				if (slot.sideUsesRuntimeActionOverride[sideIdx])
					slot.sideActionParsers[sideIdx]->ResetRuntimeActionOverride();
		};
		runBatchedJobs(resetRuntime, (int)assignedSlotIndices.size());
		timing.slotBuilderResetSeconds += SecondsSince(componentStart);

		for (int slotIdx : assignedSlotIndices)
			activeSlotsByTeamSize[slots[slotIdx].teamSize]++;
		activeSlots += (int)assignedSlotIndices.size();
		timing.slotAssignSeconds += SecondsSince(assignStart);
		assignedSlotIndices.clear();
		emitStatus();
	};

	fillPendingTasksTimed(taskQueueTarget);
	for (int slotIdx = 0; slotIdx < numSlots && !pendingTasks.empty(); slotIdx++)
		assignSlot(slotIdx, SlotAssignMode::EmptyArena);
	prepareAssignedSlots();

	while ((activeSlots > 0 || !pendingTasks.empty() || !params.isSchedulingComplete()) && !result.cancelled && !stopRequested) {
		if (params.isCancelled && params.isCancelled()) {
			result.cancelled = true;
			break;
		}

		for (int slotIdx = 0; slotIdx < numSlots && !pendingTasks.empty(); slotIdx++)
			if (!slots[slotIdx].active)
				assignSlot(slotIdx, SlotAssignMode::CompatibleArena);
		fillPendingTasksForIdleArenaLayoutsTimed();
		for (int slotIdx = 0; slotIdx < numSlots && !pendingTasks.empty(); slotIdx++)
			if (!slots[slotIdx].active)
				assignSlot(slotIdx, SlotAssignMode::CompatibleArena);
		for (int slotIdx = 0; slotIdx < numSlots && !pendingTasks.empty(); slotIdx++)
			if (!slots[slotIdx].active)
				assignSlot(slotIdx, SlotAssignMode::EmptyArena);
		for (int slotIdx = 0; slotIdx < numSlots && !pendingTasks.empty(); slotIdx++)
			if (!slots[slotIdx].active)
				assignSlot(slotIdx, SlotAssignMode::TerminalLayout);
		prepareAssignedSlots();

		if (activeSlots <= 0 && !pendingTasks.empty()) {
			assignStarvationFallbackBatch();
			prepareAssignedSlots();
		}

		if (activeSlots <= 0) {
			taskQueueTarget = std::min(numSlots, taskQueueTarget + taskCreateBatchSize);
			fillPendingTasksTimed(taskQueueTarget);
			continue;
		}

		activeIterationSlots.clear();
		for (int slotIdx = 0; slotIdx < (int)slots.size(); slotIdx++)
			if (slots[slotIdx].active)
				activeIterationSlots.push_back(slotIdx);

		auto bucketStart = std::chrono::steady_clock::now();
		auto stepFirstHalf = [&](int activeIdx) {
			PolicyMatchSlot& slot = slots[activeIterationSlots[activeIdx]];
			slot.prevState = slot.state;
			slot.state.ResetBeforeStep();
			ResetStepContactState(slot.arena.get());
			if (slot.usesRuntimeActionOverride)
				StepRuntimeActionDelay(slot, params.actionDelay);
			else
				slot.arena->Step(params.actionDelay);
		};
		runBatchedJobs(stepFirstHalf, (int)activeIterationSlots.size());
		timing.physicsSeconds += SecondsSince(bucketStart);

		bucketStart = std::chrono::steady_clock::now();
		for (SlotModelBatch& batch : modelBatches)
			batch.targets.clear();

		for (int slotIdx : activeIterationSlots) {
			PolicyMatchSlot& slot = slots[slotIdx];
			if (stochasticKickoff)
				UpdateBallTouched(slot);

			for (int playerIdx = 0; playerIdx < (int)slot.state.players.size(); playerIdx++) {
				const RLGC::Player& player = slot.state.players[playerIdx];
				int modelIdx = player.team == RocketSim::Team::BLUE ? slot.blueModel : slot.orangeModel;
				const int batchIdx = modelIdx + ((!slot.ballTouched && params.runtimes[modelIdx].stochasticKickoff) ? numModels : 0);
				int batchRow = (int)modelBatches[batchIdx].targets.size();
				modelBatches[batchIdx].targets.push_back({ slotIdx, playerIdx, batchRow });
				slot.playerBatchIndices[playerIdx] = batchIdx;
				slot.playerBatchRows[playerIdx] = batchRow;
			}
		}

		for (SlotModelBatch& batch : modelBatches) {
			const PolicyMatchRuntime& runtime = params.runtimes[batch.runtimeIndex];
			batch.obs.resize(batch.targets.size() * runtime.obsSize);
			const size_t maskSize = batch.targets.size() * (size_t)runtime.numActions;
			if (runtime.usesActionMask)
				batch.actionMasks.resize(maskSize);
			else if (batch.actionMasks.size() != maskSize)
				batch.actionMasks.assign(maskSize, true);
		}
		timing.batchPrepSeconds += SecondsSince(bucketStart);

		std::atomic<int> batchBuildError = 0;
		bucketStart = std::chrono::steady_clock::now();
		auto buildSlotBatchRows = [&](int activeIdx) {
			if (batchBuildError.load(std::memory_order_relaxed) != 0)
				return;
			PolicyMatchSlot& slot = slots[activeIterationSlots[activeIdx]];

			for (int playerIdx = 0; playerIdx < (int)slot.state.players.size(); playerIdx++) {
				const RLGC::Player& player = slot.state.players[playerIdx];
				const int batchIdx = slot.playerBatchIndices[playerIdx];
				const int batchRow = slot.playerBatchRows[playerIdx];
				const int sideIdx = player.team == RocketSim::Team::BLUE ? 0 : 1;
				if (batchIdx < 0 || batchRow < 0)
					continue;

				SlotModelBatch& batch = modelBatches[batchIdx];
				const PolicyMatchRuntime& runtime = params.runtimes[batch.runtimeIndex];
				slot.sideObsBuilders[sideIdx]->BuildObsInto(player, slot.state, slot.obsScratch);
				if ((int)slot.obsScratch.size() != runtime.obsSize) {
					batchBuildError.store(1, std::memory_order_relaxed);
					return;
				}
				std::copy(
					slot.obsScratch.begin(),
					slot.obsScratch.end(),
					batch.obs.begin() + (size_t)batchRow * runtime.obsSize);

				if (runtime.usesActionMask) {
					slot.sideActionParsers[sideIdx]->GetActionMaskInto(player, slot.state, slot.actionMaskScratch);
					if ((int)slot.actionMaskScratch.size() != runtime.numActions) {
						batchBuildError.store(2, std::memory_order_relaxed);
						return;
					}
					std::copy(
						slot.actionMaskScratch.begin(),
						slot.actionMaskScratch.end(),
						batch.actionMasks.begin() + (size_t)batchRow * runtime.numActions);
				}
			}
		};
		runBatchedJobs(buildSlotBatchRows, (int)activeIterationSlots.size());
		const int buildError = batchBuildError.load(std::memory_order_relaxed);
		if (buildError == 1)
			throw std::runtime_error("PolicyMatchRunner: obs builder produced an unexpected observation size.");
		if (buildError == 2)
			throw std::runtime_error("PolicyMatchRunner: action parser produced an unexpected mask size.");
		timing.obsMaskSeconds += SecondsSince(bucketStart);

		for (SlotModelBatch& batch : modelBatches) {
			if (batch.targets.empty())
				continue;

			const PolicyMatchRuntime& runtime = params.runtimes[batch.runtimeIndex];
			InferModelBatch(params, runtime, batch, timing, cudaOomRecovery);

			bucketStart = std::chrono::steady_clock::now();
			for (int i = 0; i < (int)batch.targets.size(); i++) {
				const SlotActionTarget& target = batch.targets[i];
				PolicyMatchSlot& slot = slots[target.slotIdx];
				const RLGC::Player& player = slot.state.players[target.playerIdx];
				const int sideIdx = player.team == RocketSim::Team::BLUE ? 0 : 1;
				RLGC::ActionParser* actionParser = slot.sideActionParsers[sideIdx];
				const int actionIndex = batch.actionIndices[i];
				if (params.actionDelay != 0 && actionParser->IsMacroAction(actionIndex, player, slot.state)) {
					throw std::runtime_error(
						"PolicyMatchRunner: action macros require actionDelay == 0.");
				}
				slot.actionMacros[target.playerIdx] = BuildActionMacros(
					*actionParser,
					actionIndex,
					player,
					slot.state,
					params.tickSkip - params.actionDelay);
				if (params.actionDelay != 0 && HasSegmentedMacro(slot.actionMacros[target.playerIdx], params.tickSkip - params.actionDelay)) {
					throw std::runtime_error(
						"PolicyMatchRunner: action macros require actionDelay == 0.");
				}
				slot.actions[target.playerIdx] = slot.actionMacros[target.playerIdx].front().action;
				slot.obsActions[target.playerIdx] = slot.actionMacros[target.playerIdx].back().action;
			}
			timing.actionParseSeconds += SecondsSince(bucketStart);
		}

		bucketStart = std::chrono::steady_clock::now();
		auto stepSecondHalf = [&](int activeIdx) {
			PolicyMatchSlot& slot = slots[activeIterationSlots[activeIdx]];
			ApplyActionMacrosWithinWindow(slot, slot.actionMacros, params.tickSkip - params.actionDelay);
			if (slot.eventTracker)
				slot.eventTracker->Update(slot.arena.get());
			slot.state.UpdateFromArena(slot.arena.get(), slot.obsActions, &slot.prevState);
			slot.state.userInfo = slot.userInfo;
			slot.steps++;

			if (params.onStepParallel)
				params.onStepParallel(
					{ slot, slot.state, slot.prevState, slot.steps, params.tickSkip / 120.f, activeIterationSlots[activeIdx] });
		};
		runBatchedJobs(stepSecondHalf, (int)activeIterationSlots.size());
		timing.actionStepSeconds += SecondsSince(bucketStart);

		bucketStart = std::chrono::steady_clock::now();
		std::vector<int> finishedSlots;
		for (int slotIdx : activeIterationSlots) {
			PolicyMatchSlot& slot = slots[slotIdx];
			if (params.onStep)
				params.onStep({ slot, slot.state, slot.prevState, slot.steps, params.tickSkip / 120.f, slotIdx });

			if (slot.state.goalScored || slot.steps >= params.noGoalSteps)
				finishedSlots.push_back(slotIdx);
		}

		for (int slotIdx : finishedSlots) {
			PolicyMatchSlot& slot = slots[slotIdx];
			slot.active = false;
			activeSlots--;
			auto countItr = activeSlotsByTeamSize.find(slot.teamSize);
			assert(countItr != activeSlotsByTeamSize.end() && countItr->second > 0);
			if (--countItr->second == 0)
				activeSlotsByTeamSize.erase(countItr);
		}
		if (!finishedSlots.empty())
			emitStatus();

		for (int slotIdx : finishedSlots) {
			PolicyMatchSlot& slot = slots[slotIdx];
			PolicyMatchFinish finish = {
				slot,
				slot.state,
				slot.steps,
				slot.state.goalScored,
				RocketSim::Team::BLUE,
				-1,
				-1
			};
			if (slot.state.goalScored) {
				RocketSim::Team concededTeam = RS_TEAM_FROM_Y(slot.state.ball.pos.y);
				finish.scoringTeam = RS_OPPOSITE_TEAM(concededTeam);
				finish.winningModel = finish.scoringTeam == RocketSim::Team::BLUE ? slot.blueModel : slot.orangeModel;
				finish.losingModel = finish.winningModel == slot.blueModel ? slot.orangeModel : slot.blueModel;
			}

			stopRequested = params.onFinished(finish) || stopRequested;
		}
		timing.finishSeconds += SecondsSince(bucketStart);

		taskQueueTarget = std::min(numSlots, taskQueueTarget + taskCreateBatchSize);
		fillPendingTasksTimed(taskQueueTarget);
	}

	graphCleanup.Cleanup();
	CopyCudaGraphStats(timing);
	return result;
}

PolicyMatchRunnerResult RunPolicyMatchVariableTimingSlots(const PolicyMatchRunnerParams& params) {
	ValidateParams(params);
	RG_NO_GRAD;

	ScopedCudaGraphCacheCleanup graphCleanup{ AnyRuntimeUsesCudaGraph(params.runtimes) };
	ModelManager::ResetCudaGraphStats();
	PolicyMatchRunnerResult result = {};
	PolicyMatchTiming& timing = result.timing;
	CudaOomRecoveryState cudaOomRecovery;
	const int numSlots = std::max(1, params.numSlots);
	const int slotThreadTarget = std::max(1, RLGC::g_ThreadPool.GetNumThreads() * 2);
	const int noGoalTicks = params.noGoalTicks > 0
		? params.noGoalTicks
		: std::max(1, params.noGoalSteps * std::max(1, params.tickSkip));
	std::vector<PolicyMatchTask> pendingTasks;
	std::vector<PolicyMatchSlot> slots(numSlots);
	const int numModels = (int)params.runtimes.size();
	const bool stochasticKickoff = std::any_of(params.runtimes.begin(), params.runtimes.end(),
		[](const PolicyMatchRuntime& runtime) { return runtime.stochasticKickoff; });

	// A model can serve opening and post-touch decisions in the same iteration.
	std::vector<SlotModelBatch> modelBatches(numModels * (stochasticKickoff ? 2 : 1));
	std::vector<int> activeIterationSlots;
	activeIterationSlots.reserve(numSlots);

	const int maxTargetsPerModel = numSlots * 3;
	for (int batchIdx = 0; batchIdx < (int)modelBatches.size(); batchIdx++) {
		SlotModelBatch& batch = modelBatches[batchIdx];
		batch.runtimeIndex = batchIdx % numModels;
		batch.stochasticKickoff = batchIdx >= numModels;
		const PolicyMatchRuntime& runtime = params.runtimes[batch.runtimeIndex];
		batch.targets.reserve(maxTargetsPerModel);
		batch.obs.reserve((size_t)maxTargetsPerModel * runtime.obsSize);
		batch.actionMasks.reserve((size_t)maxTargetsPerModel * runtime.numActions);
		batch.actionIndices.reserve(maxTargetsPerModel);
	}

	int activeSlots = 0;
	std::map<int, int> activeSlotsByTeamSize;
	const int taskCreateBatchSize = std::min(numSlots, slotThreadTarget);
	int taskQueueTarget = taskCreateBatchSize;
	bool stopRequested = false;
	std::mutex threadExceptionMutex;
	std::exception_ptr threadException = nullptr;

	auto captureThreadException = [&](std::exception_ptr exception) {
		std::scoped_lock lock(threadExceptionMutex);
		if (!threadException)
			threadException = exception;
	};

	auto throwThreadException = [&]() {
		if (threadException)
			std::rethrow_exception(threadException);
	};

	auto runBatchedJobs = [&](const std::function<void(int)>& job, int count) {
		auto guardedJob = [&](int idx) {
			try {
				job(idx);
			} catch (...) {
				captureThreadException(std::current_exception());
			}
		};
		RLGC::g_ThreadPool.StartBatchedJobs(guardedJob, count, false);
		throwThreadException();
	};

	auto emitStatus = [&]() {
		if (!params.onStatus)
			return;

		PolicyMatchRunnerStatus status = {};
		status.totalSlots = numSlots;
		status.activeSlots = activeSlots;
		status.pendingTasks = (int)pendingTasks.size();
		status.slotArenaCreates = timing.slotArenaCreates;
		status.slotArenaRebuilds = timing.slotArenaRebuilds;
		status.slotArenaReuses = timing.slotArenaReuses;
		status.cudaOomRecoveries = timing.cudaOomRecoveries;
		status.cudaGraphsDisabledAfterOom = timing.cudaGraphsDisabledAfterOom;
		CopyCudaGraphStats(status);
		status.activeSlotsByTeamSize.assign(activeSlotsByTeamSize.begin(), activeSlotsByTeamSize.end());
		params.onStatus(status);
	};

	auto ensureSlotSideRuntime = [&](PolicyMatchSlot& slot, int sideIdx, int modelIdx) {
		const PolicyMatchRuntime& runtime = params.runtimes[modelIdx];
		const int runtimeGroup = runtime.runtimeGroup;
		if (slot.sideRuntimeGroups[sideIdx] == runtimeGroup
			&& slot.sideObsBuilders[sideIdx]
			&& slot.sideActionParsers[sideIdx]) {
			return;
		}

		PolicyMatchRuntimeComponents& components = slot.runtimeComponentsByGroup[runtimeGroup];
		if (!components.obsBuilder || !components.actionParser)
			components = runtime.makeRuntimeComponents();
		if (!components.obsBuilder || !components.actionParser)
			throw std::runtime_error("PolicyMatchRunner: runtime factory returned null.");

		slot.sideRuntimeGroups[sideIdx] = runtimeGroup;
		slot.sideObsBuilders[sideIdx] = components.obsBuilder.get();
		slot.sideActionParsers[sideIdx] = components.actionParser.get();
		slot.sideUsesRuntimeActionOverride[sideIdx] =
			components.actionParser->HasRuntimeActionOverride();
		slot.usesRuntimeActionOverride =
			slot.sideUsesRuntimeActionOverride[0]
			|| slot.sideUsesRuntimeActionOverride[1];
	};

	auto ensureSlotArena = [&](PolicyMatchSlot& slot) {
		if (slot.arena
			&& slot.arenaTeamSize == slot.teamSize
			&& slot.arenaBlueKey == TaskBlueArenaKey(slot)
			&& slot.arenaOrangeKey == TaskOrangeArenaKey(slot)) {
			return;
		}

		PolicyMatchArena created = params.makeArena(slot);
		if (!created.arena)
			throw std::runtime_error("PolicyMatchRunner: makeArena returned null.");
		if ((created.runtimeComponents.obsBuilder && !created.runtimeComponents.actionParser)
			|| (!created.runtimeComponents.obsBuilder && created.runtimeComponents.actionParser)) {
			throw std::runtime_error("PolicyMatchRunner: makeArena returned partial runtime components.");
		}

		slot.sideRuntimeGroups = { -1, -1 };
		slot.sideObsBuilders = { nullptr, nullptr };
		slot.sideActionParsers = { nullptr, nullptr };
		slot.sideUsesRuntimeActionOverride = { false, false };
		slot.usesRuntimeActionOverride = false;
		slot.arena = std::move(created.arena);
		slot.userInfo = created.userInfo;
		slot.ownedRewards = std::move(created.ownedRewards);
		slot.ownedTerminalConditions = std::move(created.ownedTerminalConditions);
		slot.ownedStateSetter = std::move(created.ownedStateSetter);
		if (created.runtimeComponents.obsBuilder)
			slot.runtimeComponentsByGroup[created.runtimeGroup] = std::move(created.runtimeComponents);
		slot.eventTracker = std::make_unique<RocketSim::GameEventTracker>();
		slot.eventTracker->SetShotCallback(SlotShotEventCallback, &slot);
		slot.eventTracker->SetGoalCallback(SlotGoalEventCallback, &slot);
		slot.eventTracker->SetSaveCallback(SlotSaveEventCallback, &slot);
		slot.arena->SetCarBumpCallback(SlotBumpCallback, &slot);
		slot.arenaTeamSize = slot.teamSize;
		slot.arenaBlueKey = TaskBlueArenaKey(slot);
		slot.arenaOrangeKey = TaskOrangeArenaKey(slot);
	};

	auto fillPendingTasksTimed = [&](int targetQueueSize) {
		targetQueueSize = std::clamp(targetQueueSize, 1, numSlots);
		auto timingStart = std::chrono::steady_clock::now();
		const size_t pendingBefore = pendingTasks.size();
		while (!stopRequested
			&& !result.cancelled
			&& !params.isSchedulingComplete()
			&& (int)(pendingTasks.size() + activeSlots) < targetQueueSize) {
			if (params.isCancelled && params.isCancelled()) {
				result.cancelled = true;
				emitStatus();
				return;
			}

			const size_t before = pendingTasks.size();
			params.fillPendingTasks(pendingTasks, activeSlots, targetQueueSize);
			for (size_t taskIdx = before; taskIdx < pendingTasks.size(); taskIdx++)
				EnsureTaskValid(params, pendingTasks[taskIdx]);
			if (pendingTasks.size() == before)
				break;
		}
		timing.schedulerSeconds += SecondsSince(timingStart);
		if (pendingTasks.size() != pendingBefore)
			emitStatus();
	};

	auto pendingTaskIndexForSlot = [&](const PolicyMatchSlot& slot, bool requireCompatible) {
		if (pendingTasks.empty())
			return -1;

		if (slot.arena) {
			for (int taskIdx = (int)pendingTasks.size() - 1; taskIdx >= 0; taskIdx--) {
				const PolicyMatchTask& task = pendingTasks[taskIdx];
				if (slot.arenaTeamSize == task.teamSize
					&& slot.arenaBlueKey == TaskBlueArenaKey(task)
					&& slot.arenaOrangeKey == TaskOrangeArenaKey(task)) {
					return taskIdx;
				}
			}
		}

		return requireCompatible ? -1 : (int)pendingTasks.size() - 1;
	};

	auto fillPendingTasksForIdleArenaLayoutsTimed = [&]() {
		if (!params.fillPendingTasksForArenaLayouts
			|| stopRequested
			|| result.cancelled
			|| params.isSchedulingComplete()) {
			return;
		}

		std::set<std::tuple<int, int, int>> pendingLayouts;
		for (const PolicyMatchTask& task : pendingTasks) {
			pendingLayouts.emplace(
				task.teamSize,
				TaskBlueArenaKey(task),
				TaskOrangeArenaKey(task));
		}

		std::vector<PolicyMatchArenaLayoutDemand> demands;
		for (const PolicyMatchSlot& slot : slots) {
			if (slot.active || !slot.arena)
				continue;
			if (params.isArenaLayoutComplete
				&& params.isArenaLayoutComplete(
					slot.arenaTeamSize,
					slot.arenaBlueKey,
					slot.arenaOrangeKey)) {
				continue;
			}

			const auto layout = std::make_tuple(
				slot.arenaTeamSize,
				slot.arenaBlueKey,
				slot.arenaOrangeKey);
			if (pendingLayouts.count(layout) > 0)
				continue;

			auto demandItr = std::find_if(
				demands.begin(),
				demands.end(),
				[&](const PolicyMatchArenaLayoutDemand& demand) {
					return demand.teamSize == slot.arenaTeamSize
						&& demand.blueArenaKey == slot.arenaBlueKey
						&& demand.orangeArenaKey == slot.arenaOrangeKey;
				});
			if (demandItr == demands.end()) {
				demands.push_back({
					slot.arenaTeamSize,
					slot.arenaBlueKey,
					slot.arenaOrangeKey,
					1
				});
			} else {
				demandItr->idleSlots++;
			}
		}
		if (demands.empty())
			return;

		auto timingStart = std::chrono::steady_clock::now();
		const size_t pendingBefore = pendingTasks.size();
		params.fillPendingTasksForArenaLayouts(pendingTasks, demands);
		for (size_t taskIdx = pendingBefore; taskIdx < pendingTasks.size(); taskIdx++)
			EnsureTaskValid(params, pendingTasks[taskIdx]);
		timing.schedulerSeconds += SecondsSince(timingStart);
		if (pendingTasks.size() != pendingBefore)
			emitStatus();
	};

	enum class SlotAssignMode {
		CompatibleArena,
		EmptyArena,
		TerminalLayout
	};

	auto pendingTaskIndexForAssignMode = [&](const PolicyMatchSlot& slot, SlotAssignMode mode) {
		if (mode == SlotAssignMode::EmptyArena)
			return slot.arena ? -1 : pendingTaskIndexForSlot(slot, false);
		if (mode == SlotAssignMode::CompatibleArena)
			return pendingTaskIndexForSlot(slot, true);
		if (mode == SlotAssignMode::TerminalLayout) {
			if (!slot.arena
				|| !params.isArenaLayoutComplete
				|| !params.isArenaLayoutComplete(slot.arenaTeamSize, slot.arenaBlueKey, slot.arenaOrangeKey)) {
				return -1;
			}
		}
		return pendingTaskIndexForSlot(slot, false);
	};

	std::vector<int> assignedSlotIndices;
	assignedSlotIndices.reserve(numSlots);

	auto assignPendingTask = [&](int slotIdx, int taskIdx) {
		auto assignStart = std::chrono::steady_clock::now();
		PolicyMatchSlot& slot = slots[slotIdx];
		static_cast<PolicyMatchTask&>(slot) = std::move(pendingTasks[taskIdx]);
		if (taskIdx != (int)pendingTasks.size() - 1)
			pendingTasks[taskIdx] = std::move(pendingTasks.back());
		pendingTasks.pop_back();

		slot.active = true;
		assignedSlotIndices.push_back(slotIdx);
		timing.slotAssignSeconds += SecondsSince(assignStart);
		return true;
	};

	auto assignSlot = [&](int slotIdx, SlotAssignMode mode) {
		int taskIdx = pendingTaskIndexForAssignMode(slots[slotIdx], mode);
		return taskIdx >= 0 && assignPendingTask(slotIdx, taskIdx);
	};

	auto assignStarvationFallbackBatch = [&]() {
		if (pendingTasks.empty())
			return;

		const PolicyMatchTask& targetTask = pendingTasks.back();
		const int targetTeamSize = targetTask.teamSize;
		const int targetBlueArenaKey = TaskBlueArenaKey(targetTask);
		const int targetOrangeArenaKey = TaskOrangeArenaKey(targetTask);
		int activeTargetSlots = 0;
		for (const PolicyMatchSlot& slot : slots) {
			if (slot.active
				&& slot.arena
				&& slot.arenaTeamSize == targetTeamSize
				&& slot.arenaBlueKey == targetBlueArenaKey
				&& slot.arenaOrangeKey == targetOrangeArenaKey) {
				activeTargetSlots++;
			}
		}

		// NOTE: Rebuild at most one setup batch and retain at least half of the
		// existing arena layouts. The rebuilt slots all target one layout so the
		// unavoidable fallback cost creates useful parallel capacity.
		const int rebuildLimit = std::min(taskCreateBatchSize, std::max(1, numSlots / 2));
		const int availableRebuilds = rebuildLimit - activeTargetSlots;
		if (availableRebuilds <= 0)
			return;
		int assigned = 0;
		for (int slotIdx = 0; slotIdx < numSlots && assigned < availableRebuilds; slotIdx++) {
			if (slots[slotIdx].active)
				continue;

			int taskIdx = -1;
			for (int pendingIdx = (int)pendingTasks.size() - 1; pendingIdx >= 0; pendingIdx--) {
				if (TaskUsesArenaLayout(
					pendingTasks[pendingIdx],
					targetTeamSize,
					targetBlueArenaKey,
					targetOrangeArenaKey)) {
					taskIdx = pendingIdx;
					break;
				}
			}
			if (taskIdx < 0)
				break;
			assignPendingTask(slotIdx, taskIdx);
			assigned++;
		}
	};

	auto prepareAssignedSlots = [&]() {
		if (assignedSlotIndices.empty())
			return;

		auto assignStart = std::chrono::steady_clock::now();
		auto componentStart = std::chrono::steady_clock::now();
		for (int slotIdx : assignedSlotIndices) {
			const PolicyMatchSlot& slot = slots[slotIdx];
			if (slot.arena
				&& slot.arenaTeamSize == slot.teamSize
				&& slot.arenaBlueKey == TaskBlueArenaKey(slot)
				&& slot.arenaOrangeKey == TaskOrangeArenaKey(slot)) {
				timing.slotArenaReuses++;
			} else if (slot.arena) {
				timing.slotArenaRebuilds++;
			} else {
				timing.slotArenaCreates++;
			}
		}
		auto setupArena = [&](int assignedIdx) {
			ensureSlotArena(slots[assignedSlotIndices[assignedIdx]]);
		};
		runBatchedJobs(setupArena, (int)assignedSlotIndices.size());
		timing.slotArenaSetupSeconds += SecondsSince(componentStart);

		componentStart = std::chrono::steady_clock::now();
		for (int slotIdx : assignedSlotIndices) {
			PolicyMatchSlot& slot = slots[slotIdx];
			ensureSlotSideRuntime(slot, 0, slot.blueModel);
			ensureSlotSideRuntime(slot, 1, slot.orangeModel);
		}
		timing.slotSideRuntimeSeconds += SecondsSince(componentStart);

		componentStart = std::chrono::steady_clock::now();
		auto applySnapshot = [&](int assignedIdx) {
			PolicyMatchSlot& slot = slots[assignedSlotIndices[assignedIdx]];
			ApplyPolicyMatchSnapshot(slot.arena.get(), slot.snapshot);
		};
		runBatchedJobs(applySnapshot, (int)assignedSlotIndices.size());
		timing.slotSnapshotSeconds += SecondsSince(componentStart);

		componentStart = std::chrono::steady_clock::now();
		auto initState = [&](int assignedIdx) {
			PolicyMatchSlot& slot = slots[assignedSlotIndices[assignedIdx]];
			slot.state = RLGC::GameState(slot.arena.get());
			slot.state.userInfo = slot.userInfo;
			slot.prevState.MakeEmpty();
			slot.ballTouched = false;
			slot.sideObsStates[0] = slot.state;
			slot.sideObsStates[1] = slot.state;
			slot.sidePrevObsStates[0].MakeEmpty();
			slot.sidePrevObsStates[1].MakeEmpty();
			slot.sideEventStates[0].assign(slot.state.players.size(), RLGC::PlayerEventState{});
			slot.sideEventStates[1].assign(slot.state.players.size(), RLGC::PlayerEventState{});
			slot.actions.assign(slot.state.players.size(), RLGC::Action{});
			slot.obsActions.assign(slot.state.players.size(), RLGC::Action{});
			slot.pendingActions.assign(slot.state.players.size(), RLGC::Action{});
			slot.pendingActionMacros.assign(slot.state.players.size(), {});
			slot.activeActionMacros.assign(slot.state.players.size(), {});
			slot.activeMacroIndices.assign(slot.state.players.size(), 0);
			slot.activeMacroNextTicks.assign(slot.state.players.size(), -1);
			slot.playerBatchIndices.assign(slot.state.players.size(), -1);
			slot.playerBatchRows.assign(slot.state.players.size(), -1);
			slot.obsScratch.reserve(std::max(params.runtimes[slot.blueModel].obsSize, params.runtimes[slot.orangeModel].obsSize));
			slot.actionMaskScratch.reserve(std::max(
				params.runtimes[slot.blueModel].numActions,
				params.runtimes[slot.orangeModel].numActions));
			slot.steps = slot.resumeSteps;
			slot.physicsTicks = slot.resumeSteps;
			slot.sideNextDecisionTicks = { slot.physicsTicks, slot.physicsTicks };
			slot.sidePendingApplyTicks = { -1, -1 };
		};
		runBatchedJobs(initState, (int)assignedSlotIndices.size());
		timing.slotStateInitSeconds += SecondsSince(componentStart);

		componentStart = std::chrono::steady_clock::now();
		auto resetRuntime = [&](int assignedIdx) {
			PolicyMatchSlot& slot = slots[assignedSlotIndices[assignedIdx]];
			if (slot.eventTracker)
				slot.eventTracker->ResetPersistentInfo();
			slot.sideObsBuilders[0]->Reset(slot.state);
			slot.sideObsBuilders[1]->Reset(slot.state);
			for (int sideIdx = 0; sideIdx < 2; sideIdx++)
				if (slot.sideUsesRuntimeActionOverride[sideIdx])
					slot.sideActionParsers[sideIdx]->ResetRuntimeActionOverride();
		};
		runBatchedJobs(resetRuntime, (int)assignedSlotIndices.size());
		timing.slotBuilderResetSeconds += SecondsSince(componentStart);

		for (int slotIdx : assignedSlotIndices)
			activeSlotsByTeamSize[slots[slotIdx].teamSize]++;
		activeSlots += (int)assignedSlotIndices.size();
		timing.slotAssignSeconds += SecondsSince(assignStart);
		assignedSlotIndices.clear();
		emitStatus();
	};

	auto refreshActiveIterationSlots = [&]() {
		activeIterationSlots.clear();
		for (int slotIdx = 0; slotIdx < (int)slots.size(); slotIdx++)
			if (slots[slotIdx].active)
				activeIterationSlots.push_back(slotIdx);
	};

	auto applyDueControls = [&]() {
		auto bucketStart = std::chrono::steady_clock::now();
		for (int slotIdx : activeIterationSlots) {
			PolicyMatchSlot& slot = slots[slotIdx];
			for (int sideIdx = 0; sideIdx < 2; sideIdx++) {
				if (slot.sidePendingApplyTicks[sideIdx] >= 0
					&& slot.sidePendingApplyTicks[sideIdx] <= slot.physicsTicks) {
					SetSideControlsFromPending(slot, sideIdx);
					slot.sidePendingApplyTicks[sideIdx] = -1;
				}
			}
			for (int playerIdx = 0; playerIdx < (int)slot.state.players.size(); playerIdx++)
				AdvancePlayerMacroToTick(slot, playerIdx);
		}
		timing.actionStepSeconds += SecondsSince(bucketStart);
	};

	struct DueDecision {
		int slotIdx = 0;
		int sideIdx = 0;
	};

	auto runDueDecisions = [&]() {
		std::vector<DueDecision> dueDecisions;
		for (int slotIdx : activeIterationSlots) {
			PolicyMatchSlot& slot = slots[slotIdx];
			for (int sideIdx = 0; sideIdx < 2; sideIdx++) {
				if (slot.sideNextDecisionTicks[sideIdx] <= slot.physicsTicks)
					dueDecisions.push_back({ slotIdx, sideIdx });
			}
		}
		if (dueDecisions.empty())
			return;

		auto bucketStart = std::chrono::steady_clock::now();
		for (SlotModelBatch& batch : modelBatches)
			batch.targets.clear();

		for (const DueDecision& decision : dueDecisions) {
			PolicyMatchSlot& slot = slots[decision.slotIdx];
			if (stochasticKickoff)
				UpdateBallTouched(slot);

			RefreshSideObsState(slot, decision.sideIdx);
			const int modelIdx = SideModelIndex(slot, decision.sideIdx);
			const RLGC::GameState& obsState = slot.sideObsStates[decision.sideIdx];
			for (int playerIdx = 0; playerIdx < (int)obsState.players.size(); playerIdx++) {
				const RLGC::Player& player = obsState.players[playerIdx];
				if (!IsPlayerOnSide(player, decision.sideIdx))
					continue;
				const int batchIdx = modelIdx + ((!slot.ballTouched && params.runtimes[modelIdx].stochasticKickoff) ? numModels : 0);
				int batchRow = (int)modelBatches[batchIdx].targets.size();
				modelBatches[batchIdx].targets.push_back({ decision.slotIdx, playerIdx, batchRow });
				slot.playerBatchIndices[playerIdx] = batchIdx;
				slot.playerBatchRows[playerIdx] = batchRow;
			}
		}

		for (SlotModelBatch& batch : modelBatches) {
			const PolicyMatchRuntime& runtime = params.runtimes[batch.runtimeIndex];
			batch.obs.resize(batch.targets.size() * runtime.obsSize);
			const size_t maskSize = batch.targets.size() * (size_t)runtime.numActions;
			if (runtime.usesActionMask)
				batch.actionMasks.resize(maskSize);
			else if (batch.actionMasks.size() != maskSize)
				batch.actionMasks.assign(maskSize, true);
		}
		timing.batchPrepSeconds += SecondsSince(bucketStart);

		std::atomic<int> batchBuildError = 0;
		bucketStart = std::chrono::steady_clock::now();
		auto buildSlotBatchRows = [&](int activeIdx) {
			if (batchBuildError.load(std::memory_order_relaxed) != 0)
				return;
			PolicyMatchSlot& slot = slots[activeIterationSlots[activeIdx]];

			for (int playerIdx = 0; playerIdx < (int)slot.state.players.size(); playerIdx++) {
				const int batchIdx = slot.playerBatchIndices[playerIdx];
				const int batchRow = slot.playerBatchRows[playerIdx];
				if (batchIdx < 0 || batchRow < 0)
					continue;

				const int sideIdx = SideIndex(slot.state.players[playerIdx].team);
				const RLGC::GameState& obsState = slot.sideObsStates[sideIdx];
				const RLGC::Player& player = obsState.players[playerIdx];
				SlotModelBatch& batch = modelBatches[batchIdx];
				const PolicyMatchRuntime& runtime = params.runtimes[batch.runtimeIndex];
				slot.sideObsBuilders[sideIdx]->BuildObsInto(player, obsState, slot.obsScratch);
				if ((int)slot.obsScratch.size() != runtime.obsSize) {
					batchBuildError.store(1, std::memory_order_relaxed);
					return;
				}
				std::copy(
					slot.obsScratch.begin(),
					slot.obsScratch.end(),
					batch.obs.begin() + (size_t)batchRow * runtime.obsSize);

				if (runtime.usesActionMask) {
					slot.sideActionParsers[sideIdx]->GetActionMaskInto(player, obsState, slot.actionMaskScratch);
					if ((int)slot.actionMaskScratch.size() != runtime.numActions) {
						batchBuildError.store(2, std::memory_order_relaxed);
						return;
					}
					std::copy(
						slot.actionMaskScratch.begin(),
						slot.actionMaskScratch.end(),
						batch.actionMasks.begin() + (size_t)batchRow * runtime.numActions);
				}
			}
		};
		runBatchedJobs(buildSlotBatchRows, (int)activeIterationSlots.size());
		const int buildError = batchBuildError.load(std::memory_order_relaxed);
		if (buildError == 1)
			throw std::runtime_error("PolicyMatchRunner: obs builder produced an unexpected observation size.");
		if (buildError == 2)
			throw std::runtime_error("PolicyMatchRunner: action parser produced an unexpected mask size.");
		timing.obsMaskSeconds += SecondsSince(bucketStart);

		for (SlotModelBatch& batch : modelBatches) {
			if (batch.targets.empty())
				continue;

			const PolicyMatchRuntime& runtime = params.runtimes[batch.runtimeIndex];
			InferModelBatch(params, runtime, batch, timing, cudaOomRecovery);

			bucketStart = std::chrono::steady_clock::now();
			for (int i = 0; i < (int)batch.targets.size(); i++) {
				const SlotActionTarget& target = batch.targets[i];
				PolicyMatchSlot& slot = slots[target.slotIdx];
				const int sideIdx = SideIndex(slot.state.players[target.playerIdx].team);
				const RLGC::GameState& obsState = slot.sideObsStates[sideIdx];
				const RLGC::Player& player = obsState.players[target.playerIdx];
				const PolicyMatchRuntime& targetRuntime = params.runtimes[SideModelIndex(slot, sideIdx)];
				RLGC::ActionParser* actionParser = slot.sideActionParsers[sideIdx];
				const int actionIndex = batch.actionIndices[i];
				if (targetRuntime.actionDelay != 0 && actionParser->IsMacroAction(actionIndex, player, obsState)) {
					throw std::runtime_error(
						"PolicyMatchRunner: action macros require actionDelay == 0.");
				}
				slot.pendingActionMacros[target.playerIdx] = BuildActionMacros(
					*actionParser,
					actionIndex,
					player,
					obsState,
					targetRuntime.tickSkip - targetRuntime.actionDelay);
				if (targetRuntime.actionDelay != 0 && HasSegmentedMacro(slot.pendingActionMacros[target.playerIdx], targetRuntime.tickSkip - targetRuntime.actionDelay)) {
					throw std::runtime_error(
						"PolicyMatchRunner: action macros require actionDelay == 0.");
				}
				slot.pendingActions[target.playerIdx] = slot.pendingActionMacros[target.playerIdx].front().action;
			}
			timing.actionParseSeconds += SecondsSince(bucketStart);
		}

		bucketStart = std::chrono::steady_clock::now();
		for (const DueDecision& decision : dueDecisions) {
			PolicyMatchSlot& slot = slots[decision.slotIdx];
			const PolicyMatchRuntime& runtime = params.runtimes[SideModelIndex(slot, decision.sideIdx)];
			slot.sidePendingApplyTicks[decision.sideIdx] = slot.physicsTicks + runtime.actionDelay;
			do {
				slot.sideNextDecisionTicks[decision.sideIdx] += runtime.tickSkip;
			} while (slot.sideNextDecisionTicks[decision.sideIdx] <= slot.physicsTicks);

			if (slot.sidePendingApplyTicks[decision.sideIdx] <= slot.physicsTicks) {
				SetSideControlsFromPending(slot, decision.sideIdx);
				slot.sidePendingApplyTicks[decision.sideIdx] = -1;
			}
		}
		timing.actionStepSeconds += SecondsSince(bucketStart);

		for (int slotIdx : activeIterationSlots) {
			PolicyMatchSlot& slot = slots[slotIdx];
			std::fill(slot.playerBatchIndices.begin(), slot.playerBatchIndices.end(), -1);
			std::fill(slot.playerBatchRows.begin(), slot.playerBatchRows.end(), -1);
		}
	};

	auto nextSlotEventDeltaTicks = [&](PolicyMatchSlot& slot) {
		int bestDelta = std::numeric_limits<int>::max();
		if (RequiresPerTickActionOverride(slot, slot.state))
			bestDelta = 1;
		auto considerTick = [&](int tick) {
			if (tick > slot.physicsTicks)
				bestDelta = std::min(bestDelta, tick - slot.physicsTicks);
		};
		for (int sideIdx = 0; sideIdx < 2; sideIdx++) {
			considerTick(slot.sideNextDecisionTicks[sideIdx]);
			if (slot.sidePendingApplyTicks[sideIdx] >= 0)
				considerTick(slot.sidePendingApplyTicks[sideIdx]);
		}

		for (int tick : slot.activeMacroNextTicks)
			if (tick >= 0)
				considerTick(tick);

		considerTick(noGoalTicks);

		return bestDelta == std::numeric_limits<int>::max() ? 1 : bestDelta;
	};

	std::vector<int> slotStepTicks(numSlots);

	auto finishSlotsIfNeeded = [&]() {
		std::vector<int> finishedSlots;
		for (int slotIdx : activeIterationSlots) {
			PolicyMatchSlot& slot = slots[slotIdx];
			if (slot.state.goalScored || slot.physicsTicks >= noGoalTicks)
				finishedSlots.push_back(slotIdx);
		}

		for (int slotIdx : finishedSlots) {
			PolicyMatchSlot& slot = slots[slotIdx];
			slot.active = false;
			activeSlots--;
			auto countItr = activeSlotsByTeamSize.find(slot.teamSize);
			assert(countItr != activeSlotsByTeamSize.end() && countItr->second > 0);
			if (--countItr->second == 0)
				activeSlotsByTeamSize.erase(countItr);
		}
		if (!finishedSlots.empty())
			emitStatus();

		for (int slotIdx : finishedSlots) {
			PolicyMatchSlot& slot = slots[slotIdx];
			PolicyMatchFinish finish = {
				slot,
				slot.state,
				slot.physicsTicks,
				slot.state.goalScored,
				RocketSim::Team::BLUE,
				-1,
				-1
			};
			if (slot.state.goalScored) {
				RocketSim::Team concededTeam = RS_TEAM_FROM_Y(slot.state.ball.pos.y);
				finish.scoringTeam = RS_OPPOSITE_TEAM(concededTeam);
				finish.winningModel = finish.scoringTeam == RocketSim::Team::BLUE ? slot.blueModel : slot.orangeModel;
				finish.losingModel = finish.winningModel == slot.blueModel ? slot.orangeModel : slot.blueModel;
			}

			stopRequested = params.onFinished(finish) || stopRequested;
		}
	};

	fillPendingTasksTimed(taskQueueTarget);
	for (int slotIdx = 0; slotIdx < numSlots && !pendingTasks.empty(); slotIdx++)
		assignSlot(slotIdx, SlotAssignMode::EmptyArena);
	prepareAssignedSlots();

	while ((activeSlots > 0 || !pendingTasks.empty() || !params.isSchedulingComplete()) && !result.cancelled && !stopRequested) {
		if (params.isCancelled && params.isCancelled()) {
			result.cancelled = true;
			break;
		}

		for (int slotIdx = 0; slotIdx < numSlots && !pendingTasks.empty(); slotIdx++)
			if (!slots[slotIdx].active)
				assignSlot(slotIdx, SlotAssignMode::CompatibleArena);
		fillPendingTasksForIdleArenaLayoutsTimed();
		for (int slotIdx = 0; slotIdx < numSlots && !pendingTasks.empty(); slotIdx++)
			if (!slots[slotIdx].active)
				assignSlot(slotIdx, SlotAssignMode::CompatibleArena);
		for (int slotIdx = 0; slotIdx < numSlots && !pendingTasks.empty(); slotIdx++)
			if (!slots[slotIdx].active)
				assignSlot(slotIdx, SlotAssignMode::EmptyArena);
		for (int slotIdx = 0; slotIdx < numSlots && !pendingTasks.empty(); slotIdx++)
			if (!slots[slotIdx].active)
				assignSlot(slotIdx, SlotAssignMode::TerminalLayout);
		prepareAssignedSlots();

		if (activeSlots <= 0 && !pendingTasks.empty()) {
			assignStarvationFallbackBatch();
			prepareAssignedSlots();
		}

		if (activeSlots <= 0) {
			taskQueueTarget = std::min(numSlots, taskQueueTarget + taskCreateBatchSize);
			fillPendingTasksTimed(taskQueueTarget);
			continue;
		}

		refreshActiveIterationSlots();
		applyDueControls();
		runDueDecisions();
		finishSlotsIfNeeded();
		if (activeSlots <= 0 || result.cancelled || stopRequested) {
			taskQueueTarget = std::min(numSlots, taskQueueTarget + taskCreateBatchSize);
			fillPendingTasksTimed(taskQueueTarget);
			continue;
		}

		refreshActiveIterationSlots();

		int commonDelta = std::numeric_limits<int>::max();
		for (int slotIdx : activeIterationSlots) {
			int delta = std::max(1, nextSlotEventDeltaTicks(slots[slotIdx]));
			if (params.maxPhysicsStepTicks > 0)
				delta = std::min(delta, params.maxPhysicsStepTicks);

			slotStepTicks[slotIdx] = delta;
			commonDelta = std::min(commonDelta, delta);
		}

		if (!params.independentSlotTiming)
			for (int slotIdx : activeIterationSlots)
				slotStepTicks[slotIdx] = commonDelta;

		auto bucketStart = std::chrono::steady_clock::now();
		auto stepSlots = [&](int activeIdx) {
			const int slotIdx = activeIterationSlots[activeIdx];
			PolicyMatchSlot& slot = slots[slotIdx];
			const int stepDeltaTicks = slotStepTicks[slotIdx];
			ApplySlotControls(slot);
			slot.prevState = slot.state;
			slot.state.ResetBeforeStep();
			ResetStepContactState(slot.arena.get());
			slot.arena->Step(stepDeltaTicks);
			if (slot.eventTracker)
				slot.eventTracker->Update(slot.arena.get());
			slot.state.UpdateFromArena(slot.arena.get(), slot.obsActions, &slot.prevState);
			slot.state.userInfo = slot.userInfo;
			AccumulateSlotEvents(slot);
			slot.physicsTicks += stepDeltaTicks;
			slot.steps = slot.physicsTicks;

			if (params.onStepParallel)
				params.onStepParallel({ slot, slot.state, slot.prevState, slot.physicsTicks, stepDeltaTicks / 120.f, slotIdx });
		};
		runBatchedJobs(stepSlots, (int)activeIterationSlots.size());
		timing.physicsSeconds += SecondsSince(bucketStart);

		bucketStart = std::chrono::steady_clock::now();
		for (int slotIdx : activeIterationSlots) {
			PolicyMatchSlot& slot = slots[slotIdx];
			if (params.onStep) {
				params.onStep({
					slot,
					slot.state,
					slot.prevState,
					slot.physicsTicks,
					slotStepTicks[slotIdx] / 120.0f,
					slotIdx
				});
			}
		}
		finishSlotsIfNeeded();
		timing.finishSeconds += SecondsSince(bucketStart);

		taskQueueTarget = std::min(numSlots, taskQueueTarget + taskCreateBatchSize);
		fillPendingTasksTimed(taskQueueTarget);
	}

	graphCleanup.Cleanup();
	CopyCudaGraphStats(timing);
	return result;
}
}
