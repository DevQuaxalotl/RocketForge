#include "EnvSet.h"
#include  "../Rewards/ZeroSumReward.h"
#include <bit>
#include <chrono>

template<bool RLGC::PlayerEventState::* DATA_VAR>
void IncPlayerCounter(Car* car, void* userInfoPtr) {
	if (!car)
		return;

	auto userInfo = (RLGC::EnvSet::CallbackUserInfo*)userInfoPtr;

	auto& gs = userInfo->envSet->state.gameStates[userInfo->arenaIdx];
	for (auto& player : gs.players)
		if (player.carId == car->id)
			(player.eventState.*DATA_VAR) = true;
}

void _ShotEventCallback(Arena* arena, Car* shooter, Car* passer, void* userInfo) {
	IncPlayerCounter<&RLGC::PlayerEventState::shot>(shooter, userInfo);
	IncPlayerCounter<&RLGC::PlayerEventState::shotPass>(passer, userInfo);
}

void _GoalEventCallback(Arena* arena, Car* scorer, Car* passer, void* userInfo) {
	IncPlayerCounter<&RLGC::PlayerEventState::goal>(scorer, userInfo);
	IncPlayerCounter<&RLGC::PlayerEventState::assist>(passer, userInfo);
}

void _SaveEventCallback(Arena* arena, Car* saver, void* userInfo) {
	IncPlayerCounter<&RLGC::PlayerEventState::save>(saver, userInfo);
}

void _BumpCallback(Arena* arena, Car* bumper, Car* victim, bool isDemo, void* userInfo) {
	if (bumper->team == victim->team)
		return;

	IncPlayerCounter<&RLGC::PlayerEventState::bump>(bumper, userInfo);
	IncPlayerCounter<&RLGC::PlayerEventState::bumped>(victim, userInfo);

	if (isDemo) {
		IncPlayerCounter<&RLGC::PlayerEventState::demo>(bumper, userInfo);
		IncPlayerCounter<&RLGC::PlayerEventState::demoed>(victim, userInfo);
	}
}

void _ResetStepContactState(Arena* arena) {
	for (Car* car : arena->_cars)
		car->ResetStepContactState();
}

namespace {
	// Tests the exponent bits so the check survives fast-math builds, where std::isfinite may fold to true.
	bool IsFiniteFloat(float value) {
		return (std::bit_cast<uint32_t>(value) & 0x7f800000u) != 0x7f800000u;
	}

	bool IsFiniteVec(const Vec& vec) {
		return IsFiniteFloat(vec.x) && IsFiniteFloat(vec.y) && IsFiniteFloat(vec.z);
	}

	bool IsFinitePhys(const PhysState& phys) {
		return IsFiniteVec(phys.pos) && IsFiniteVec(phys.vel) && IsFiniteVec(phys.angVel)
			&& IsFiniteVec(phys.rotMat.forward) && IsFiniteVec(phys.rotMat.right) && IsFiniteVec(phys.rotMat.up);
	}

	bool IsFiniteGameState(const RLGC::GameState& state) {
		if (!IsFinitePhys(state.ball))
			return false;

		for (const RLGC::Player& player : state.players)
			if (!IsFinitePhys(player) || !IsFiniteFloat(player.boost))
				return false;

		return true;
	}
}

/////////////////////////////

RLGC::EnvSet::EnvSet(const EnvSetConfig& config) : config(config) {

	RG_ASSERT(config.tickSkip > 0);
	RG_ASSERT(config.actionDelay >= 0 && config.actionDelay <= config.tickSkip);

	arenas.resize(config.numArenas);
	eventTrackers.resize(config.numArenas);
	eventCallbackInfos.resize(config.numArenas);
	userInfos.resize(config.numArenas);
	rewards.resize(config.numArenas);
	terminalConditions.resize(config.numArenas);
	obsBuilders.resize(config.numArenas);
	actionParsers.resize(config.numArenas);
	stateSetters.resize(config.numArenas);

	auto fnCreateArenas = [&](int idx) {
		auto createResult = config.envCreateFn(idx, config.envCreateCfg);
		auto arena = createResult.arena;

		arenas[idx] = arena;

		auto userInfo = new CallbackUserInfo();
		userInfo->arena = arena;
		userInfo->arenaIdx = idx;
		userInfo->envSet = this;
		eventCallbackInfos[idx] = userInfo;
		arena->SetCarBumpCallback(_BumpCallback, userInfo);

		if (arena->gameMode != GameMode::HEATSEEKER) {
			GameEventTracker* tracker = new GameEventTracker({});
			eventTrackers[idx] = tracker;

			tracker->SetShotCallback(_ShotEventCallback, userInfo);
			tracker->SetGoalCallback(_GoalEventCallback, userInfo);
			tracker->SetSaveCallback(_SaveEventCallback, userInfo);
		}

		userInfos[idx] = createResult.userInfo;

		rewards[idx] = createResult.rewards;
		terminalConditions[idx] = createResult.terminalConditions;
		obsBuilders[idx] = createResult.obsBuilder;
		actionParsers[idx] = createResult.actionParser;
		stateSetters[idx] = createResult.stateSetter;
	};
	g_ThreadPool.StartBatchedJobs(fnCreateArenas, config.numArenas, false);

	state.Resize(arenas);
	_rewardTimes.resize(config.numArenas);
	_rewardScratch.resize(config.numArenas);
	_arenaScratch.resize(config.numArenas);

	// Determine obs size and action amount, initialize arrays accordingly
	{
		stateSetters[0]->ResetArena(arenas[0]);
		GameState testState = GameState(arenas[0]);
		testState.userInfo = userInfos[0];
		obsBuilders[0]->Reset(testState);
		FList obsScratch;
		obsBuilders[0]->BuildObsInto(testState.players[0], testState, obsScratch);
		obsSize = obsScratch.size();
		state.obs = DimList2<float>(state.numPlayers, obsSize);

		// Cache the runtime interface widths so callers can validate compatibility.
		numActions = actionParsers[0]->GetActionAmount();
		state.actionMasks = DimList2<uint8_t>(state.numPlayers, numActions);
	}

	// Build the per-(reward, team-size) tracking state and apply the initial clamped weights before collection begins.
	{
		std::vector<AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes(arenas.size());
		for (int i = 0; i < (int)arenas.size(); i++) {
			for (Car* car : arenas[i]->_cars)
				(car->team == RocketSim::Team::BLUE ? teamSizes[i].blue : teamSizes[i].orange)++;
		}
		adaptiveWeightController.Init(rewards, teamSizes);
	}

	// Reset all arenas initially
	g_ThreadPool.StartBatchedJobs(
		std::bind(&RLGC::EnvSet::ResetArena, this, std::placeholders::_1),
		config.numArenas, false
	);
	
}

void RLGC::EnvSet::EnsureRuntimeActionStorage() {
	if (!policyActions.empty())
		return;

	policyActions.resize(arenas.size());
	for (int arenaIdx = 0; arenaIdx < (int)arenas.size(); arenaIdx++) {
		policyActions[arenaIdx].reserve(arenas[arenaIdx]->_cars.size());
		for (Car* car : arenas[arenaIdx]->_cars)
			policyActions[arenaIdx].emplace_back(car->controls);
	}
}

double RLGC::EnvSet::TakeRewardTime() {
	double total = 0;
	for (double& arenaRewardTime : _rewardTimes) {
		total += arenaRewardTime;
		arenaRewardTime = 0;
	}

	return total;
}

void RLGC::EnvSet::EndNonFiniteEpisode(int arenaIdx) {
	const int playerStartIdx = state.arenaPlayerStartIdx[arenaIdx];
	const int playerCount = (int)state.gameStates[arenaIdx].players.size();

	for (int i = 0; i < playerCount; i++) {
		const int playerIdx = playerStartIdx + i;
		state.rewards[playerIdx] = 0;
		std::fill_n(&state.obs.At(playerIdx, 0), state.obs.size[1], 0.f);

		if (rewardHeadLayout)
			std::fill_n(&state.headRewards.At(playerIdx, 0), state.headRewards.size[1], 0.f);
	}

	state.terminals[arenaIdx] = TerminalType::NORMAL;
}

void RLGC::EnvSet::StepFirstHalf(bool async) {
	if (config.actionDelay == 0)
		return;

	auto fnStepArena = [&](int arenaIdx) {
		Arena* arena = arenas[arenaIdx];
		auto& gs = state.gameStates[arenaIdx];

		{
			// Set previous gamestates
			state.prevGameStates[arenaIdx] = gs;
		}

		gs.ResetBeforeStep();

		// Step arena with old actions
		_ResetStepContactState(arena);
		arena->Step(config.actionDelay);
	};

	g_ThreadPool.StartBatchedJobs(fnStepArena, arenas.size(), async);
}

void RLGC::EnvSet::StepFirstHalf(
	bool async,
	const std::vector<ActionParser*>& actionParsersByPlayer)
{
	RG_ASSERT(actionParsersByPlayer.size() == state.numPlayers);
	if (config.actionDelay == 0)
		return;

	EnsureRuntimeActionStorage();
	auto fnStepArena = [&](int arenaIdx) {
		Arena* arena = arenas[arenaIdx];
		auto& gs = state.gameStates[arenaIdx];

		{
			// Set previous gamestates
			state.prevGameStates[arenaIdx] = gs;
		}

		gs.ResetBeforeStep();

		_ResetStepContactState(arena);
		const int playerStartIdx = state.arenaPlayerStartIdx[arenaIdx];
		bool needsPerTickOverride = false;
		for (int i = 0; i < (int)gs.players.size(); i++) {
			ActionParser* parser = actionParsersByPlayer[playerStartIdx + i];
			if (parser->HasRuntimeActionOverride()) {
				needsPerTickOverride = parser->RequiresPerTickActionOverride(gs.players[i], gs)
					|| needsPerTickOverride;
			}
		}

		if (!needsPerTickOverride) {
			arena->Step(config.actionDelay);
			return;
		}

		GameState controlState = gs;
		GameState previousControlState;
		std::vector<Action>& appliedActions = _arenaScratch[arenaIdx].appliedActions;
		appliedActions.resize(gs.players.size());
		for (int tick = 0; tick < config.actionDelay; tick++) {
			auto carItr = arena->_cars.begin();
			for (int i = 0; i < (int)controlState.players.size(); i++, carItr++) {
				ActionParser* parser = actionParsersByPlayer[playerStartIdx + i];
				appliedActions[i] = policyActions[arenaIdx][i];
				if (parser->HasRuntimeActionOverride()) {
					appliedActions[i] = parser->OverrideRuntimeAction(
						appliedActions[i],
						controlState.players[i],
						controlState,
						1);
				}
				(*carItr)->controls = (CarControls)appliedActions[i];
			}
			arena->Step(1);

			if (tick + 1 < config.actionDelay) {
				previousControlState = controlState;
				controlState.ResetBeforeStep();
				controlState.UpdateFromArena(arena, appliedActions, &previousControlState);
				controlState.userInfo = userInfos[arenaIdx];
			}
		}
	};

	g_ThreadPool.StartBatchedJobs(fnStepArena, arenas.size(), async);
}

void RLGC::EnvSet::StepFirstHalf(
	bool async,
	const std::vector<ActionParser*>& actionParsersByPlayer,
	const IList& scheduledActionIndices,
	const IList& scheduledActionApplyTicks)
{
	RG_ASSERT(actionParsersByPlayer.size() == state.numPlayers);
	RG_ASSERT(scheduledActionIndices.size() == state.numPlayers);
	RG_ASSERT(scheduledActionApplyTicks.size() == state.numPlayers);
	if (config.actionDelay == 0)
		return;

	EnsureRuntimeActionStorage();
	auto fnStepArena = [&](int arenaIdx) {
		Arena* arena = arenas[arenaIdx];
		auto& gs = state.gameStates[arenaIdx];
		state.prevGameStates[arenaIdx] = gs;
		gs.ResetBeforeStep();
		_ResetStepContactState(arena);

		const int playerStartIdx = state.arenaPlayerStartIdx[arenaIdx];
		bool needsPerTickStep = false;
		for (int i = 0; i < (int)gs.players.size(); i++) {
			const int globalPlayerIdx = playerStartIdx + i;
			const int applyTick = scheduledActionApplyTicks[globalPlayerIdx];
			RG_ASSERT(applyTick < config.actionDelay);
			needsPerTickStep = applyTick >= 0 || needsPerTickStep;

			ActionParser* parser = actionParsersByPlayer[globalPlayerIdx];
			RG_ASSERT(parser != nullptr);
			if (parser->HasRuntimeActionOverride()) {
				needsPerTickStep = parser->RequiresPerTickActionOverride(gs.players[i], gs)
					|| needsPerTickStep;
			}
		}

		if (!needsPerTickStep) {
			arena->Step(config.actionDelay);
			return;
		}

		GameState controlState = gs;
		GameState previousControlState;
		std::vector<Action>& appliedActions = _arenaScratch[arenaIdx].appliedActions;
		appliedActions.resize(gs.players.size());
		for (int tick = 0; tick < config.actionDelay; tick++) {
			auto carItr = arena->_cars.begin();
			for (int i = 0; i < (int)controlState.players.size(); i++, carItr++) {
				const int globalPlayerIdx = playerStartIdx + i;
				ActionParser* parser = actionParsersByPlayer[globalPlayerIdx];
				if (scheduledActionApplyTicks[globalPlayerIdx] == tick) {
					const int actionIndex = scheduledActionIndices[globalPlayerIdx];
					if (parser->IsMacroAction(actionIndex, gs.players[i], gs)) {
						RG_ERR_CLOSE(
							"EnvSet::StepFirstHalf(): scheduled native-timing actions cannot be macros"
						);
					}
					policyActions[arenaIdx][i] = parser->ParseAction(actionIndex, gs.players[i], gs);
				}

				appliedActions[i] = policyActions[arenaIdx][i];
				if (parser->HasRuntimeActionOverride()) {
					appliedActions[i] = parser->OverrideRuntimeAction(
						appliedActions[i],
						controlState.players[i],
						controlState,
						1);
				}
				(*carItr)->controls = (CarControls)appliedActions[i];
			}
			arena->Step(1);

			if (tick + 1 < config.actionDelay) {
				previousControlState = controlState;
				controlState.ResetBeforeStep();
				controlState.UpdateFromArena(arena, appliedActions, &previousControlState);
				controlState.userInfo = userInfos[arenaIdx];
			}
		}
	};

	g_ThreadPool.StartBatchedJobs(fnStepArena, arenas.size(), async);
}

void RLGC::EnvSet::StepSecondHalf(const IList& actionIndices, bool async) {
	StepSecondHalfImpl<false>(actionIndices, nullptr, async);
}

void RLGC::EnvSet::StepSecondHalf(
	const IList& actionIndices,
	const std::vector<ActionParser*>& actionParsersByPlayer,
	bool async)
{
	RG_ASSERT(actionParsersByPlayer.size() == state.numPlayers);
	EnsureRuntimeActionStorage();
	StepSecondHalfImpl<true>(actionIndices, &actionParsersByPlayer, async);
}

template<bool USE_PLAYER_PARSERS>
void RLGC::EnvSet::StepSecondHalfImpl(
	const IList& actionIndices,
	const std::vector<ActionParser*>* actionParsersByPlayer,
	bool async)
{
	auto fnStepArenas = [&](int arenaIdx) {

		Arena* arena = arenas[arenaIdx];
		auto& gs = state.gameStates[arenaIdx];
		int playerStartIdx = state.arenaPlayerStartIdx[arenaIdx];

		if (config.actionDelay == 0) {
			state.prevGameStates[arenaIdx] = gs;
			gs.ResetBeforeStep();
			_ResetStepContactState(arena);
		}
			
		// Parse actions. Most parsers return one macro and keep the existing fast path.
		const int actionTicks = config.tickSkip - config.actionDelay;
		ArenaScratch& scratch = _arenaScratch[arenaIdx];
		std::vector<Action>& actions = scratch.actions;
		auto& actionMacros = scratch.actionMacros;
		actions.resize(gs.players.size());
		actionMacros.resize(gs.players.size());
		bool hasMacroAction = false;
		for (int i = 0; i < gs.players.size(); i++) {
			auto& player = gs.players[i];
			auto& macros = actionMacros[i];
			// The buffer is reused, and custom parsers may append without clearing it first.
			macros.clear();
			const int globalPlayerIdx = playerStartIdx + i;
			const int actionIndex = actionIndices[globalPlayerIdx];
			ActionParser* parser = actionParsers[arenaIdx];
			if constexpr (USE_PLAYER_PARSERS) {
				parser = (*actionParsersByPlayer)[globalPlayerIdx];
				RG_ASSERT(parser != nullptr);
			}
			if (config.actionDelay != 0 && parser->IsMacroAction(actionIndex, player, gs)) {
				RG_ERR_CLOSE(
					"EnvSet::StepSecondHalf(): action macros require actionDelay == 0 "
					<< "(tickSkip=" << config.tickSkip << ", actionDelay=" << config.actionDelay << ")"
				);
			}
			parser->ParseActionMacros(actionIndex, player, gs, actionTicks, macros);

			if (macros.empty())
				macros.push_back({ parser->ParseAction(actionIndex, player, gs), actionTicks });

			int macroTicks = 0;
			for (const ActionMacro& macro : macros) {
				RG_ASSERT(macro.ticks >= 0);
				macroTicks += macro.ticks;
			}
			RG_ASSERT(macroTicks == actionTicks);

			actions[i] = macros.back().action;
			if (macros.size() != 1 || macros.front().ticks != actionTicks)
				hasMacroAction = true;
			if constexpr (USE_PLAYER_PARSERS)
				policyActions[arenaIdx][i] = actions[i];
		}

		if constexpr (!USE_PLAYER_PARSERS) {
			// Preserve the original learner path when every player shares one parser.
			if (!hasMacroAction) {
				auto carItr = arena->_cars.begin();
				for (int i = 0; i < gs.players.size(); i++, carItr++)
					(*carItr)->controls = (CarControls)actions[i];
				arena->Step(actionTicks);
			} else {
				IList& macroIndices = scratch.macroIndices;
				IList& macroTicksLeft = scratch.macroTicksLeft;
				macroIndices.assign(gs.players.size(), 0);
				macroTicksLeft.resize(gs.players.size());
				for (int i = 0; i < gs.players.size(); i++)
					macroTicksLeft[i] = actionMacros[i].front().ticks;

				for (int tick = 0; tick < actionTicks; tick++) {
					auto carItr = arena->_cars.begin();
					for (int i = 0; i < gs.players.size(); i++, carItr++) {
						auto& macros = actionMacros[i];
						while (macroTicksLeft[i] <= 0 && macroIndices[i] + 1 < (int)macros.size()) {
							macroIndices[i]++;
							macroTicksLeft[i] = macros[macroIndices[i]].ticks;
						}
						(*carItr)->controls = (CarControls)macros[macroIndices[i]].action;
					}

					arena->Step(1);

					for (int i = 0; i < gs.players.size(); i++)
						if (macroTicksLeft[i] > 0)
							macroTicksLeft[i]--;
				}
			}
		} else {
			GameState controlState = gs;
			GameState previousControlState;
			if (config.actionDelay > 0) {
				auto& currentActions = scratch.appliedActions;
				currentActions.resize(gs.players.size());
				auto carItr = arena->_cars.begin();
				for (int i = 0; i < (int)currentActions.size(); i++, carItr++)
					currentActions[i] = Action((*carItr)->controls);
				controlState.UpdateFromArena(arena, currentActions, &gs);
				controlState.userInfo = userInfos[arenaIdx];
			}

			bool needsPerTickOverride = false;
			for (int i = 0; i < (int)controlState.players.size(); i++) {
				ActionParser* parser = (*actionParsersByPlayer)[playerStartIdx + i];
				if (parser->HasRuntimeActionOverride()) {
					needsPerTickOverride = parser->RequiresPerTickActionOverride(
						controlState.players[i],
						controlState) || needsPerTickOverride;
				}
			}

			if (!hasMacroAction && !needsPerTickOverride) {
				auto carItr = arena->_cars.begin();
				for (int i = 0; i < gs.players.size(); i++, carItr++)
					(*carItr)->controls = (CarControls)actions[i];
				arena->Step(actionTicks);
			} else {
				IList& macroIndices = scratch.macroIndices;
				IList& macroTicksLeft = scratch.macroTicksLeft;
				macroIndices.assign(gs.players.size(), 0);
				macroTicksLeft.resize(gs.players.size());
				for (int i = 0; i < gs.players.size(); i++)
					macroTicksLeft[i] = actionMacros[i].front().ticks;

				for (int tick = 0; tick < actionTicks; tick++) {
					auto carItr = arena->_cars.begin();
					for (int i = 0; i < gs.players.size(); i++, carItr++) {
						auto& macros = actionMacros[i];
						while (macroTicksLeft[i] <= 0 && macroIndices[i] + 1 < (int)macros.size()) {
							macroIndices[i]++;
							macroTicksLeft[i] = macros[macroIndices[i]].ticks;
						}
						ActionParser* parser = (*actionParsersByPlayer)[playerStartIdx + i];
						actions[i] = macros[macroIndices[i]].action;
						if (parser->HasRuntimeActionOverride()) {
							actions[i] = parser->OverrideRuntimeAction(
								actions[i],
								controlState.players[i],
								controlState,
								1);
						}
						(*carItr)->controls = (CarControls)actions[i];
					}

					arena->Step(1);

					for (int i = 0; i < gs.players.size(); i++)
						if (macroTicksLeft[i] > 0)
							macroTicksLeft[i]--;

					if (tick + 1 < actionTicks) {
						previousControlState = controlState;
						controlState.ResetBeforeStep();
						controlState.UpdateFromArena(arena, actions, &previousControlState);
						controlState.userInfo = userInfos[arenaIdx];
					}
				}
			}
		}

		if (eventTrackers[arenaIdx])
			eventTrackers[arenaIdx]->Update(arena);

		GameState* gsPrev = &state.prevGameStates[arenaIdx];
		if (gsPrev->IsEmpty())
			gsPrev = NULL;

		gs.UpdateFromArena(arena, actions, gsPrev);

		// Rewards and adaptive weights would carry the bad values forward, so stop before either sees them
		if (!IsFiniteGameState(gs)) {
			EndNonFiniteEpisode(arenaIdx);
			return;
		}

		// Update terminal
		uint8_t terminalType = TerminalType::NOT_TERMINAL;
		for (auto cond : terminalConditions[arenaIdx]) {
			if (cond->IsTerminal(gs)) {
				bool isTrunc = cond->IsTruncation();
				uint8_t curTerminalType = isTrunc ? TerminalType::TRUNCATED : TerminalType::NORMAL;
				if (terminalType == TerminalType::NOT_TERMINAL) {
					terminalType = curTerminalType;
				} else {
					// We already know this state is terminal
					// However, if we only know it is a truncated terminal, we should let normal terminals take priority
					// (Normal terminals are better information than truncations)
					if (curTerminalType == TerminalType::NORMAL)
						terminalType = curTerminalType;
				}

				// NOTE: We can't break since terminal conditions are guaranteed to be called once per step
			}
		}

		// The setter's episode cap is only a truncation, so a terminal the conditions already reported takes priority
		if (terminalType == TerminalType::NOT_TERMINAL && stateSetters[arenaIdx]->EpisodeCapElapsed(gs))
			terminalType = TerminalType::TRUNCATED;

		state.terminals[arenaIdx] = terminalType;

		// Pre-step rewards
		const std::chrono::steady_clock::time_point rewardStart = std::chrono::steady_clock::now();
		for (auto& weighted : rewards[arenaIdx])
			weighted.reward->PreStep(gs);

		// Update rewards
		{
			RewardScratch& scratch = _rewardScratch[arenaIdx];
			FList& allRewards = scratch.totals;
			allRewards.assign(gs.players.size(), 0.f);

			// Adaptive-weight collection samples only episodes whose setter gives a representative signal: 
			// kickoffs (a real game start) and random states (noise that doesn't consistently bias the mean)
			// Bespoke scenario drills are excluded so they don't skew the target.
			const bool collectFromEpisode = IsNormalGameplayEpisode(stateSetters[arenaIdx]->setterType);

			for (int rewardIdx = 0; rewardIdx < rewards[arenaIdx].size(); rewardIdx++) {
				auto& weightedReward = rewards[arenaIdx][rewardIdx];
				FList& output = scratch.output;
				weightedReward.reward->GetAllRewardsInto(gs, terminalType, output);
				RG_ASSERT(output.size() == gs.players.size());
				for (int i = 0; i < gs.players.size(); i++)
					allRewards[i] += output[i] * weightedReward.weight;

				if (rewardHeadLayout) {
					const int headIdx = rewardHeadLayout->GetHeadIndex(arenaIdx, rewardIdx);
					for (int i = 0; i < gs.players.size(); i++)
						state.headRewards.At(playerStartIdx + i, headIdx) = output[i];
				}

				// Zero-sum rewards redistribute value between players.
				// Both the adaptive-weight tracker and the reward-save path below want the pre-redistribution value.
				ZeroSumReward* zeroSum = dynamic_cast<ZeroSumReward*>(weightedReward.reward);

				// Track absolute per-player usage for the adaptive weight controller.
				// Uses the pre-zero-sum-wrapper value so redistribution doesn't skew the signal.
				if (collectFromEpisode && adaptiveWeightController.IsCollecting(arenaIdx, rewardIdx)) {
					for (int i = 0; i < gs.players.size(); i++) {
						const float rawReward = zeroSum ? zeroSum->_lastRewards[i] : output[i];
						adaptiveWeightController.Accumulate(arenaIdx, rewardIdx, i, std::fabs(rawReward));
					}
				}

				// Save the reward
				if (config.saveRewards) {
					int playerSampleIndex;
					if (config.rewardSamplePlayerOverride >= 0 && config.rewardSamplePlayerOverride < (int)output.size()) {
						playerSampleIndex = config.rewardSamplePlayerOverride;
					} else if (config.shuffleRewardSampling) {
						playerSampleIndex = Math::RandInt(0, output.size());
					} else {
						// Find player with the lowest id
						playerSampleIndex = 0;
						uint32_t lowestID = gs.players[0].carId;
						for (int i = 1; i < gs.players.size(); i++) {
							auto id = gs.players[i].carId;
							if (id < lowestID) {
								lowestID = id;
								playerSampleIndex = i;
							}
						}
					}
					// We will only take the reward from a random player
					float rewardToSave = output[playerSampleIndex];

					// Save training rewards (post-wrapper) for arena 0 only — used by the debug collector
					if (arenaIdx == 0) {
						if (state.lastTrainingRewards.empty())
							state.lastTrainingRewards.resize(rewards[arenaIdx].size());
						state.lastTrainingRewards[rewardIdx] = rewardToSave;
					}

					// If zero-sum, use the inner (pre-wrapper) reward for earned display
					if (zeroSum)
						rewardToSave = zeroSum->_lastRewards[playerSampleIndex];

					// If needed, initialize last rewards
					if (state.lastRewards[arenaIdx].empty())
						state.lastRewards[arenaIdx].resize(rewards[arenaIdx].size());

					state.lastRewards[arenaIdx][rewardIdx] = rewardToSave;
				}
			}

			for (int i = 0; i < gs.players.size(); i++)
				state.rewards[playerStartIdx + i] = allRewards[i];
		}
		_rewardTimes[arenaIdx] += std::chrono::duration<double>(std::chrono::steady_clock::now() - rewardStart).count();

		BuildArenaObs(arenaIdx);
	};

	g_ThreadPool.StartBatchedJobs(fnStepArenas, arenas.size(), async);
}

void RLGC::EnvSet::SetRewardHeadLayout(RewardHeadLayout layout) {
	RG_ASSERT(layout.Matches(rewards));

	// Absent heads are never written by the step loop, so they rely on this zero fill.
	state.headRewards = DimList2<float>(state.numPlayers, layout.GetNumHeads());
	rewardHeadLayout = std::move(layout);
}

void RLGC::EnvSet::ResetArena(int index) {
	stateSetters[index]->ResetArena(arenas[index]);
	// State setters replace CarState, but live controls are stored separately on Car.
	// Clear them before the next action-delay window so the prior episode cannot jump,
	// boost, or steer a newly reassigned player at its reset spawn.
	for (RocketSim::Car* car : arenas[index]->_cars)
		car->controls = {};
	GameState newState = GameState(arenas[index]);
	newState.userInfo = userInfos[index];
	state.gameStates[index] = newState;
	if (!policyActions.empty())
		policyActions[index].assign(newState.players.size(), Action{});

	// Update event tracker
	if (eventTrackers[index])
		eventTrackers[index]->ResetPersistentInfo();

	// Reset all the other stuff
	actionParsers[index]->ResetRuntimeActionOverride();
	obsBuilders[index]->Reset(newState);
	for (auto& cond : terminalConditions[index])
		cond->Reset(newState);
	for (auto& weightedReward : rewards[index])
		weightedReward.reward->Reset(newState);

	BuildArenaObs(index);

	// Remove previous state
	state.prevGameStates[index].MakeEmpty();
}

void RLGC::EnvSet::BuildArenaObs(int arenaIdx) {
	const GameState& gs = state.gameStates[arenaIdx];
	const int playerStartIdx = state.arenaPlayerStartIdx[arenaIdx];

	FList& obsScratch = _arenaScratch[arenaIdx].obs;
	obsScratch.clear();
	for (int i = 0; i < gs.players.size(); i++) {
		obsBuilders[arenaIdx]->BuildObsInto(gs.players[i], gs, obsScratch);
		// The obs table has one fixed width, measured on the first arena, so a builder whose size follows the
		// player count (such as AdvancedObs) fails here as soon as arenas with different team sizes are mixed.
		if ((int)obsScratch.size() != obsSize) {
			RG_ERR_CLOSE(
				"EnvSet::BuildArenaObs(): the obs builder produced " << obsScratch.size() << " values for a "
				<< gs.players.size() << "-player arena but " << obsSize << " for the first arena. "
				"Its size must not depend on the player count; use a padded builder such as DefaultObsPadded "
				"when a stage mixes team sizes"
			);
		}
		state.obs.Set(playerStartIdx + i, obsScratch);
	}

	std::vector<uint8_t>& actionMaskScratch = _arenaScratch[arenaIdx].actionMask;
	actionMaskScratch.clear();
	if (actionParsers[arenaIdx]->UsesActionMask()) {
		for (int i = 0; i < gs.players.size(); i++) {
			actionParsers[arenaIdx]->GetActionMaskInto(gs.players[i], gs, actionMaskScratch);
			state.actionMasks.Set(playerStartIdx + i, actionMaskScratch);
		}
	} else {
		actionMaskScratch.assign(numActions, true);
		for (int i = 0; i < gs.players.size(); i++)
			state.actionMasks.Set(playerStartIdx + i, actionMaskScratch);
	}
}

void RLGC::EnvSet::Reset() {
	for (int i = 0; i < arenas.size(); i++)
		if (state.terminals[i])
			g_ThreadPool.StartJobAsync(std::bind(&EnvSet::ResetArena, this, std::placeholders::_1), i);
	std::fill(state.terminals.begin(), state.terminals.end(), 0);
	g_ThreadPool.WaitUntilDone();
}
