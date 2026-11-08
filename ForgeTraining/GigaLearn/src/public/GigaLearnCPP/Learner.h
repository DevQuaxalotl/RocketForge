#pragma once

#include <atomic>
#include <memory>
#include <RLGymCPP/EnvSet/EnvSet.h>
#include "Util/MetricSender.h"
#include "LearnerConfig.h"
#include "PPO/TransferLearnConfig.h"

namespace GGL {

	class PopArtStats;

	typedef std::function<void(class Learner*, const std::vector<RLGC::GameState>& states, Report& report)> StepCallbackFn;

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/learner.py
	class GGL_TRAIN_IMEXPORT Learner {
	public:
		LearnerConfig config;

		RLGC::EnvSet* envSet;

		class PPOLearner* ppo;
		struct PolicyVersionManager* versionMgr;
		struct PolicyVersionManager* staticOpponentMgr;
		/** Skill references for the skill test; null when the Build has none. **/
		struct PolicyVersionManager* skillReferenceMgr;

		RLGC::EnvCreateFn envCreateFn;
		MetricSender* metricSender;

		int obsSize;
		int numActions;

		int warmStartIterationsLeft = 0;
		bool warmStartActive = false;

		struct WelfordStat* returnStat;
		struct BatchedWelfordStat* obsStat;

		/** Per-head critic target statistics; null unless the decomposed critic is on. */
		std::unique_ptr<PopArtStats> popArtStats;

		std::string runID = {};

		uint64_t
			totalTimesteps = 0,
			totalIterations = 0;

		StepCallbackFn stepCallback = NULL;

		std::atomic<bool> stopRequested{ false };

		/** Invoked when the metrics run is stopped remotely (the wandb Stop button). Falls back to RequestStop() when unset. */
		std::function<void()> onRemoteStopRequested = {};

		Learner(RLGC::EnvCreateFn envCreateFunc, LearnerConfig config, StepCallbackFn stepCallback = NULL);
		void Start();

		void StartTransferLearn(const TransferLearnConfig& transferLearnConfig);

		void RequestStop();

		/** Sends the report to the metrics run, then routes a remote stop request if one arrived. */
		void SendMetrics(const Report& report);

		void Save(bool saveVersionSnapshot = false);
		void Load();
		void SaveStats(std::filesystem::path path);

		/** Returns the checkpoint's critic head names when its value-head layout differs from this run's, else empty. */
		std::vector<std::string> LoadStats(std::filesystem::path path);

		RG_NO_COPY(Learner);

		~Learner();
	};
}
