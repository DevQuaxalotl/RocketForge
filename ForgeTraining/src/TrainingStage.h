#pragma once

#include "TrainingStageTypes.h"
#include "TrainingSetup/TrainingParameterOverride.h"
#include "Rewards/RewardSpec.h"
#include "Rewards/RewardCache.h"

#include <DebugSettings.h>

#include <GigaLearnCPP/LearnerConfig.h>
#include <GigaLearnCPP/TrainingStorage.h>
#include <GigaLearnCPP/Util/Report.h>

#include <GigaLearnInfer/Agent.h>

#include <RLGymCPP/EnvSet/EnvSet.h>

#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace RocketForge
{
	/**
	 * One phase of an Agent's training curriculum: what it is rewarded for, which situations episodes start from,
	 * when episodes end, and the learner's hyperparameters.
	 * A bot's stages run in order from its stage queue, and usually share one base stage that declares every reward and setter,
	 * with each subclass changing only weights and settings.
	 * See Documentation/Features/TrainingStages.md.
	 */
	class TrainingStage
	{
	public:
		TrainingStage(std::unique_ptr<Agent> agent);

		/**
		 * Constructs a stage and refreshes its learner config from the finished stage properties.
		 * Every stage should be created through this (REGISTER_STAGE does), so constructors don't need to call UpdateConfigs.
		 */
		template<typename StageT, typename... Args>
		static std::unique_ptr<StageT> Create(std::unique_ptr<Agent> agent, Args&&... args)
		{
			std::unique_ptr<StageT> stage = std::make_unique<StageT>(std::move(agent), std::forward<Args>(args)...);
			// Called through the base: a subclass that redeclares UpdateConfigs as protected or private makes it inaccessible through StageT.
			static_cast<TrainingStage&>(*stage).UpdateConfigs();
			return stage;
		}

		/**
		 * Learner settings for this stage.
		 * Some fields are overwritten from stage and agent properties, by the constructor and UpdateConfigs.
		 */
		GGL::LearnerConfig learnerConfig;

		std::unique_ptr<Agent> agent;

		/** Override to provide a per-arena reward cache, accessible via GameState::userInfo. */
		virtual std::unique_ptr<RewardCacheBase> CreateRewardCache() const {
			return std::make_unique<RewardCacheBase>();
		}

		/** Function used to create the environment for each learning process. */
		virtual RLGC::EnvCreateResult CreateEnvironment(const int index, const RLGC::EnvCreateConfig& config);

		/**
		 * Frees the reward cache CreateEnvironment handed out as EnvCreateResult::userInfo.
		 * For envs that are discarded before the stage.
		 */
		void ReleaseRewardCache(const void* userInfo);

		/**
		 * Returns all weighted rewards to be trained with during this stage.
		 * The caller owns the returned Reward pointers and is responsible for deleting them.
		 */
		virtual std::vector<RLGC::WeightedReward> GetRewards() const;

		/**
		 * Returns all weighted rewards to be trained with for a specific max team size.
		 * The caller owns the returned Reward pointers and is responsible for deleting them.
		 */
		virtual std::vector<RLGC::WeightedReward> GetRewards(int maxTeamSize) const;

		/** Stores named overrides that should be applied after stage defaults are built. */
		void SetParameterOverrides(const std::vector<RocketForgeTraining::StageParameterOverride>& overrides);

		/** Reapplies hyperparameter overrides after config refreshes such as ApplySegmentId(). */
		void ApplyParameterOverrides();

		/** Returns rewards with any configured reward-weight overrides applied. */
		std::vector<RLGC::WeightedReward> GetConfiguredRewards() const;

		/** Returns team-size-specific rewards with any configured reward-weight overrides applied. */
		std::vector<RLGC::WeightedReward> GetConfiguredRewards(int maxTeamSize) const;

		/** Returns state setter that defines initial game setup during this stage. */
		virtual std::unique_ptr<RLGC::StateSetter> GetStateSetter(const bool filter = true) const;

		/** Returns state setter for a specific arena matchup. */
		virtual std::unique_ptr<RLGC::StateSetter> GetStateSetter(int blueSize, int orangeSize, const bool filter = true) const;

		/** Returns a CombinedState of only this stage's normal-gameplay (Kickoff/Random) setters. */
		std::unique_ptr<RLGC::StateSetter> GetNormalGameplayStateSetter(int blueSize, int orangeSize) const;

		/** Returns active state setters available to the debug render force-setter dropdown. */
		std::vector<DebugStateSetterOption> GetDebugStateSetterOptions() const;

		/** Returns conditions for ending each episode during this stage. */
		virtual std::vector<std::unique_ptr<RLGC::TerminalCondition>> GetTerminalConditions() const;

		/**
		 * Sets the active debug settings (non-owning); pass nullptr to disable.
		 * The pointed-to DebugSettings must outlive this stage.
		 */
		void SetDebugSettings(const ForgeDebug::DebugSettings* settings);

		/** Returns the path to the most recent checkpoint folder. */
		virtual std::filesystem::path GetMostRecentCheckpoint() const;

		/** Returns the filesystem helper for this stage's shared storage layout. */
		GGL::TrainingStorage GetTrainingStorage() const;

		std::string GetStageName() const { return stageName; }
		std::string GetRunName() const { return runName; }
		int GetTimestepsPerIteration() const { return timestepsPerIteration; }
		float GetLearningRate() const { return learningRate; }
		const std::vector<WeightedTeamSize>& GetWeightedTeamSizes() const { return weightedTeamSizes; }

		/** Applies an already allocated segment label and refreshes derived config paths. */
		void ApplySegmentId(const std::string& segmentId, const std::string& notes = {});

		/**
		 * Returns team sizes for both teams, randomly selected based on weights.
		 * A selectionValue in 0-1 replaces the random value for team size selection.
		 */
		virtual std::pair<int, int> ChooseTeamSize(const float selectionValue = -1) const;

		/** Creates the environment for each training thread. */
		RLGC::EnvCreateFn environmentCreateFunc;

		/** Called after each step. Typically used for metrics. */
		StepCallbackFn stepCallbackFunc;

	protected:
		/** Team sizes along with the weight used to pick them. Normalized by each matchup's max team size. */
		std::vector<WeightedTeamSize> weightedTeamSizes = { { 1, 1.f } };

		/** Name of the stage. Used for wandb group name and checkpoint saving. */
		std::string stageName = "unnamed-runs";

		/** Name of the run. Allows for differentiating versions of a stage. Used for wandb run name and checkpoint saving. */
		std::string runName = "rocketforge-run";

		/** Sets learnerConfig.ppo.tsPerItr and batchSize. */
		int timestepsPerIteration = 50'000;

		/** Sets learnerConfig.ppo.policyLR and criticLR. */
		float learningRate = 1.5e-4;

		/** When true, kickoff state setters are removed from uneven team-size matchups. */
		bool disableKickoffsForUnfairMatchups = false;

		/** Returns reward specs for this stage, allowing weights to vary by team size without duplicating reward lists. */
		virtual std::vector<RewardSpec> GetRewardSpecs() const;

		/** Returns array of state setters for this stage along with their weight. Used by GetStateSetter to make a CombinedState. */
		virtual std::vector<WeightedStateSetter> GetStateSetters() const;

		/**
		 * Copies stage and agent properties (stageName, runName, timestepsPerIteration, learningRate, the engines) into learnerConfig,
		 * and prepares weightedTeamSizes for team-size selection.
		 * Create() runs it once construction finishes; override it to derive further config from stage properties.
		 */
		virtual void UpdateConfigs();

		/** Helper function for filtering out rewards with 0 weight. */
		virtual std::vector<RLGC::WeightedReward> GetFilteredRewards(int maxTeamSize = 1);

		/** Helper function for child stages to append new rewards to parent rewards. Removes any base rewards with a weight of 0. */
		virtual std::vector<RLGC::WeightedReward> CombineRewards(const std::vector<RLGC::WeightedReward>& baseRewards, const std::vector<RLGC::WeightedReward>& newRewards) const;

		/** Helper function for child stages to append new reward specs to parent reward specs. Removes specs with only zero weights. */
		virtual std::vector<RewardSpec> CombineRewardSpecs(const std::vector<RewardSpec>& baseRewards, const std::vector<RewardSpec>& newRewards) const;

		/** Helper function for child stages to append new state setters to parent setters. */
		virtual std::vector<WeightedStateSetter> CombineStateSetters(const std::vector<WeightedStateSetter>& baseSetters, const std::vector<WeightedStateSetter>& newSetters) const;

		/** Returns whether a state setter is allowed for the selected team sizes. */
		virtual bool ShouldUseStateSetter(const WeightedStateSetter& setter, int blueSize, int orangeSize) const;

		/** Normalizes WeightedTeamSizes values to be used when picking a team size. */
		void CalculateTeamSizeWeights();

		/** Resolves reward specs into runtime WeightedReward instances for a specific max team size. */
		std::vector<RLGC::WeightedReward> ResolveRewardSpecs(const std::vector<RewardSpec>& rewardSpecs, int maxTeamSize) const;

		/** Applies any named reward-weight overrides to an already-created reward list. */
		void ApplyRewardWeightOverrides(std::vector<RLGC::WeightedReward>& rewards) const;

		/** Cached value of tick skip agent uses for training. */
		int agentTickSkip = 8;

	private:
		/** Total of all normalized team size weights added together. */
		float _totalTeamWeight = 0.f;

		/** Running total of the normalized team size weights, parallel to weightedTeamSizes. */
		std::vector<float> _cumulativeTeamWeights;

		/** Protects _rewardCaches, which arena creation fills from several threads at once. */
		mutable std::mutex _rewardCacheMutex;

		/** Owns every per-arena reward cache until ReleaseRewardCache or the stage's destruction. */
		mutable std::vector<std::unique_ptr<RewardCacheBase>> _rewardCaches;

		/** Param Testing overrides persisted on the queue entry and applied before launch. */
		std::vector<RocketForgeTraining::StageParameterOverride> _parameterOverrides;

		const ForgeDebug::DebugSettings* _debugSettings = nullptr;
	};
}
