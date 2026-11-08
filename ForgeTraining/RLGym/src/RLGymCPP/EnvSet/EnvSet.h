
#pragma once
#include "../Gamestates/GameState.h"
#include "../BasicTypes/Action.h"
#include "../TerminalConditions/TerminalCondition.h"
#include "../Rewards/Reward.h"
#include "../ObsBuilders/ObsBuilder.h"
#include "../ActionParsers/ActionParser.h"
#include "../StateSetters/StateSetter.h"
#include "../ThreadPool.h"
#include "AdaptiveRewardWeightController.h"
#include "RewardHeadLayout.h"
#include <optional>
#include <RLGymCPP/Rewards/Reward.h>

namespace RLGC {

	struct EnvCreateResult {
		Arena* arena;
		std::vector<WeightedReward> rewards;
		std::vector<TerminalCondition*> terminalConditions;
		ObsBuilder* obsBuilder;
		ActionParser* actionParser;
		StateSetter* stateSetter;

		void* userInfo; // Optional userinfo pointer if you want to track some data per-env
	};

	struct EnvCreateConfig {
		int numArenas = 1;
		RocketSim::GameMode mode = RocketSim::GameMode::SOCCAR;
		int teamSizeOverride = 0;
		int maxTeamSize = 0;
		RocketSim::EngineId engine = RocketSim::EngineId::V2;

		EnvCreateConfig() {}
		EnvCreateConfig(const int numArenas, uint8_t gameMode) : numArenas(numArenas), mode(RocketSim::GameMode(gameMode)) { }
	};

	typedef std::function<EnvCreateResult(const int index, const EnvCreateConfig& config)> EnvCreateFn;

	struct EnvSetConfig {
		EnvCreateFn envCreateFn;
		EnvCreateConfig envCreateCfg;
		int numArenas;
		int tickSkip;
		int actionDelay;
		bool saveRewards;
		bool shuffleRewardSampling = true;
		int rewardSamplePlayerOverride = -1; // When >= 0, use this player index for reward sampling instead of random/lowest
	};

	struct EnvState {
		int numPlayers;
		std::vector<GameState> gameStates;
		std::vector<GameState> prevGameStates;
		DimList2<float> obs;
		DimList2<uint8_t> actionMasks;
		std::vector<float> rewards;

		/** [players × value heads] unweighted, post-zero-sum reward per head. Empty until a reward head layout is set. */
		DimList2<float> headRewards;

		std::vector<std::vector<float>> lastRewards;  // Pre-wrapper (earned) rewards, one entry per arena
		std::vector<float> lastTrainingRewards;       // Post-wrapper (training) rewards, debug only (arena 0)
		std::vector<uint8_t> terminals;

		std::vector<int> arenaPlayerStartIdx = {};

		void Resize(std::vector<Arena*>& arenas) {
			numPlayers = 0;
			for (int i = 0; i < arenas.size(); i++) {
				arenaPlayerStartIdx.push_back(numPlayers);
				numPlayers += arenas[i]->_cars.size();
			}

			gameStates.resize(arenas.size());
			prevGameStates.resize(arenas.size());
			rewards.resize(numPlayers);
			lastRewards.resize(arenas.size());
			terminals.resize(arenas.size());
		}
	};

	struct EnvSet {

		struct CallbackUserInfo {
			RLGC::EnvSet* envSet;
			Arena* arena;
			int arenaIdx;
		};

		std::vector<Arena*> arenas;
		std::vector<GameEventTracker*> eventTrackers;

		std::vector<CallbackUserInfo*> eventCallbackInfos;
		std::vector<void*> userInfos;

		EnvSetConfig config;

		int obsSize = 0;
		int numActions = 0;

		std::vector<std::vector<WeightedReward>> rewards;
		std::vector<std::vector<TerminalCondition*>> terminalConditions;
		std::vector<ObsBuilder*> obsBuilders;
		std::vector<ActionParser*> actionParsers;
		std::vector<StateSetter*> stateSetters;
		std::vector<std::vector<Action>> policyActions;

		EnvState state = {};

		AdaptiveRewardWeightController adaptiveWeightController;

		/** Set only when per-head rewards are wanted; see SetRewardHeadLayout(). */
		std::optional<RewardHeadLayout> rewardHeadLayout = std::nullopt;

		EnvSet(const EnvSetConfig& config);

		RG_NO_COPY(EnvSet);

		~EnvSet() {
			for (Arena* arena : arenas)
				delete arena;

			for (auto& eventTracker : eventTrackers)
				delete eventTracker;
			for (auto& eventCallbackInfo : eventCallbackInfos)
				delete eventCallbackInfo;
		}

		////////////////////
		
		void StepFirstHalf(bool async);
		void StepFirstHalf(
			bool async,
			const std::vector<ActionParser*>& actionParsersByPlayer);
		/**
		 * Applies selected per-player actions at exact ticks inside the action-delay
		 * window. A negative apply tick leaves that player's current action unchanged.
		 **/
		void StepFirstHalf(
			bool async,
			const std::vector<ActionParser*>& actionParsersByPlayer,
			const IList& scheduledActionIndices,
			const IList& scheduledActionApplyTicks);
		void StepSecondHalf(const IList& actionIndices, bool async);
		/**
		 * Steps with a parser selected per global player index.
		 * This supports frozen opponents whose action table differs from the learner.
		 **/
		void StepSecondHalf(
			const IList& actionIndices,
			const std::vector<ActionParser*>& actionParsersByPlayer,
			bool async);
		void Sync() { g_ThreadPool.WaitUntilDone(); }
		void ResetArena(int index);
		void Reset();

		/** Starts writing state.headRewards each step. The layout must have been built from this EnvSet's rewards. */
		void SetRewardHeadLayout(RewardHeadLayout layout);

		/** Rebuilds state.obs and state.actionMasks rows for every player in the arena from its current game state. */
		void BuildArenaObs(int arenaIdx);

		/** Returns the CPU seconds spent in reward PreStep/GetAllRewards across all arenas since the last call, and clears it. */
		double TakeRewardTime();

	private:
		/**
		 * Ends the arena's episode as a normal terminal with zero reward.
		 * The step's observation rows are zeroed; the next Reset() replaces them.
		 **/
		void EndNonFiniteEpisode(int arenaIdx);

		/** One per arena; each is written only by the job stepping that arena. */
		std::vector<double> _rewardTimes;

		/** Per-arena storage reused by its stepping job; totals are cleared before every step. */
		struct RewardScratch {
			FList totals;
			FList output;
		};
		std::vector<RewardScratch> _rewardScratch;

		/**
		 * One entry per arena, accessed only by that arena's current job.
		 * Capacity survives steps and resets; contents are replaced before use.
		 * Observation and action storage stay separate because stepping rebuilds observations.
		 **/
		struct ArenaScratch {
			FList obs;
			std::vector<uint8_t> actionMask;
			std::vector<Action> actions;
			std::vector<Action> appliedActions;
			std::vector<std::vector<ActionMacro>> actionMacros;
			IList macroIndices;
			IList macroTicksLeft;
		};
		std::vector<ArenaScratch> _arenaScratch;

		void EnsureRuntimeActionStorage();
		template<bool USE_PLAYER_PARSERS>
		void StepSecondHalfImpl(
			const IList& actionIndices,
			const std::vector<ActionParser*>* actionParsersByPlayer,
			bool async);
	};
}
