#pragma once
#include <RLGymCPP/BasicTypes/Lists.h>
#include <RocketSim/Engine.h>
#include "PPO/PPOLearnerConfig.h"
#include "PPG/PPGLearnerConfig.h"
#include "LearnerRuntimeConfig.h"
#include "SkillTrackerConfig.h"
#include <cstdint>
#include <string>

namespace GGL {
	enum class LearnerDeviceType {
		AUTO,
		CPU,
		GPU_CUDA
	};

	enum class LearnerAlgorithm {
		PPO,

		/** Currently not functioning; use PPO. */
		PPG
	};

	/**
	 * Training run settings: environment count, saving, self-play, metrics.
	 * The algorithm's own learning settings (entropy, epochs, GAE, clipping) are in ppo (PPOLearnerConfig).
	 * The TrainingStage constructor sets its own defaults for some of these, which differ from the ones below.
	 */
	struct LearnerConfig {

		/**************** Commonly tuned ****************/

		/** Arenas simulated in parallel. Training speed is usually limited by CPU clock, so this is the main speed lever. */
		int numGames = 300;

		PPOLearnerConfig ppo = {};

		/** Timesteps between checkpoint saves; 0 saves every iteration. */
		int64_t tsPerSave = 1'000'000;

		/** Checkpoints kept before the oldest are deleted; -1 keeps all. */
		int checkpointsToKeep = 8;

		/** Saves a copy of the policy every tsPerVersion timesteps, for self-play and skill tracking. */
		bool savePolicyVersions = false;
		int64_t tsPerVersion = 25'000'000;
		int maxRecentVersions = 32;

		/** Self-play against saved older versions of this policy (turns savePolicyVersions on automatically). */
		bool trainAgainstRecentVersions = false;

		/** Target fraction (0-1) of learner samples collected against recent versions. */
		float trainAgainstRecentPercent = 0.15f;

		/** Recent versions that can be chosen to train against; one more draining version may remain active. */
		int recentOpponentCohortSize = 3;

		/** Learner timesteps between random one-policy cohort rotations. */
		uint64_t recentOpponentRotationTimesteps = 10'000'000;

		/** Trains against the models in the static-opponent registry. */
		bool trainAgainstStaticOpponents = false;

		/** Target fraction (0-1) of learner samples collected against static opponents. */
		float trainAgainstStaticPercent = 0.1f;

		/** Static versions selectable at once; one more draining version may remain active. */
		int staticOpponentCohortSize = 4;

		/** Learner timesteps between random one-policy cohort rotations. */
		uint64_t staticOpponentRotationTimesteps = 10'000'000;

		/** Empty permits every model in the static-opponent registry. */
		std::string staticOpponentSelectionKey = {};

		/** Prefers a different compatible model after every episode reset. */
		bool resampleStaticOpponentEveryEpisode = false;

		/**
		 * Static opponents build observations with the learner's ball predictor by default, so collection steps only one predictor engine.
		 * True uses each opponent's own saved predictor: closer to what it trained on, but slower when that engine differs.
		 */
		bool staticOpponentsUseOwnBallPrediction = false;

		/** Lowers the policy LR for the first iterations after loading a checkpoint, to avoid destabilizing updates. */
		bool warmStartPolicy = true;
		int warmStartIterations = 20;
		float warmStartPolicyLRMult = 0.1f;

		SkillTrackerConfig skillTracker = {};

		/** Sends metrics to the Python metrics receiver, which logs them to wandb. */
		bool sendMetrics = true;

		/**************** Less commonly changed ****************/

		PPGLearnerConfig ppg = {};

		/** PPG is currently not functioning; keep PPO. */
		LearnerAlgorithm algorithm = LearnerAlgorithm::PPO;

		/** AUTO uses a CUDA GPU when one is available. */
		LearnerDeviceType deviceType = LearnerDeviceType::AUTO;

		/** Game mode to learn; see RocketSim::GameMode. */
		uint8_t gameMode = 0;

		/** -1 seeds from the current time. */
		int64_t randomSeed = -1;

		/** Standardizes observation values (hasn't helped much in testing). */
		bool standardizeObs = false;
		float minObsSTD = 1 / 10.f;
		float maxObsMeanRange = 3;
		int maxObsSamples = 100;

		/**
		 * Standardizes returns to help the critic; leave on unless you know why not.
		 * Ignored with the decomposed critic, whose PopArt stats scale each value head's targets instead.
		 */
		bool standardizeReturns = true;
		int maxReturnSamples = 150;

		/** Adds each reward to the metrics automatically. */
		bool addRewardsToMetrics = true;

		/** Maximum reward samples per step for reward metrics. */
		int maxRewardSamples = 50;

		/** Randomized interval range, in steps, between reward samples. */
		int rewardSampleRandInterval = 8;

		/**************** Set by TrainingStage ****************/

		/** Copied from the Agent in the TrainingStage constructor. */
		int tickSkip = 8;
		int actionDelay = 7;

		/** Physics engine of the training arenas. Set from Agent::trainingEngineVersion. */
		RocketSim::EngineId engine = RocketSim::EngineId::V2;

		/** Ball predictor engine of the learner's observations, which can differ from engine. Set from the Agent. */
		RocketSim::EngineId ballPredictionEngine = RocketSim::EngineId::V2;

		/** Names for the Python metrics receiver; set from the agent name, stage name and run name. */
		std::string metricsProjectName = "rocketforge";
		std::string metricsGroupName = "unnamed-runs";
		std::string metricsRunName = "rocketforge-run";

		/** Filled in by the trainer and orchestration layer; stages never set these. */
		LearnerRuntimeConfig runtime = {};
	};
}
