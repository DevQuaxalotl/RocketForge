#include "TrainingStage.h"

#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/StateSetters/CombinedState.h>
#include <RLGymCPP/StateSetters/FuzzedKickoffState.h>
#include <RLGymCPP/StateSetters/KickoffState.h>
#include <RLGymCPP/StateSetters/RandomState.h>

#include <GigaLearnCPP/Learner.h>

#include "AgentModelPaths.h"
#include "StaticOpponents/StaticOpponentSelection.h"

#include <cctype>
#ifdef __GNUG__
#include <cxxabi.h>
#endif

#include <cstdlib>
#include <regex>
#include <sstream>
#include <typeinfo>

using namespace std;
using namespace GGL; // GigaLearn
using namespace RLGC; // RLGymCPP

namespace RocketForge
{
	namespace
	{
		std::string StripVersionSuffix(const std::string& name)
		{
			static const std::regex versionSuffixPattern(R"((.*?)(?:-?v\d+\.\d+)$)", std::regex_constants::icase);
			std::smatch match;
			if (std::regex_match(name, match, versionSuffixPattern) && match.size() > 1 && !match[1].str().empty())
				return match[1].str();

			return name;
		}

		std::filesystem::path MakePreferredCopy(std::filesystem::path path)
		{
			path.make_preferred();
			return path;
		}

		std::string Trim(const std::string& text)
		{
			const size_t first = text.find_first_not_of(" \t\r\n");
			if (first == std::string::npos)
				return {};

			const size_t last = text.find_last_not_of(" \t\r\n");
			return text.substr(first, last - first + 1);
		}

		std::string CleanNumberText(const std::string& text)
		{
			std::string result;
			result.reserve(text.size());

			for (char ch : Trim(text)) {
				if (ch == ',' || ch == '\'' || ch == '_')
					continue;
				result.push_back(ch);
			}

			return result;
		}

		std::string Lower(const std::string& text)
		{
			std::string result = Trim(text);
			for (char& ch : result)
				ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

			return result;
		}

		float ParseFloatOverride(const RocketForgeTraining::StageParameterOverride& override)
		{
			const std::string text = CleanNumberText(override.value);
			try {
				size_t consumed = 0;
				const float value = std::stof(text, &consumed);
				if (consumed == text.size())
					return value;
			} catch (...) {}

			RG_ERR_CLOSE("Param Testing override \"" << override.name << "\" requires a float value, got \"" << override.value << "\"");
			return 0.f;
		}

		int64_t ParseInt64Override(const RocketForgeTraining::StageParameterOverride& override)
		{
			const std::string text = CleanNumberText(override.value);
			try {
				size_t consumed = 0;
				const int64_t value = std::stoll(text, &consumed);
				if (consumed == text.size())
					return value;
			} catch (...) {}

			RG_ERR_CLOSE("Param Testing override \"" << override.name << "\" requires an integer value, got \"" << override.value << "\"");
			return 0;
		}

		uint64_t ParseUint64Override(const RocketForgeTraining::StageParameterOverride& override)
		{
			const std::string text = CleanNumberText(override.value);
			try {
				size_t consumed = 0;
				const uint64_t value = std::stoull(text, &consumed);
				if (consumed == text.size())
					return value;
			} catch (...) {}

			RG_ERR_CLOSE("Param Testing override \"" << override.name << "\" requires an unsigned integer value, got \"" << override.value << "\"");
			return 0;
		}

		bool ParseBoolOverride(const RocketForgeTraining::StageParameterOverride& override)
		{
			const std::string text = Lower(override.value);
			if (text == "true" || text == "1" || text == "yes" || text == "on")
				return true;

			if (text == "false" || text == "0" || text == "no" || text == "off")
				return false;

			RG_ERR_CLOSE("Param Testing override \"" << override.name << "\" requires a bool value, got \"" << override.value << "\"");
			return false;
		}

		GGL::CriticHeadLossWeighting ParseCriticHeadLossWeightingOverride(const RocketForgeTraining::StageParameterOverride& override)
		{
			GGL::CriticHeadLossWeighting weighting = GGL::CriticHeadLossWeighting::Equal;
			if (GGL::CRITIC_HEAD_LOSS_WEIGHTING_NAMES.ValueOf(Trim(override.value), weighting))
				return weighting;

			RG_ERR_CLOSE("Param Testing override \"" << override.name << "\" must be Equal or Contribution, got \""
				<< override.value << "\"");
			return weighting;
		}

		std::string MakePlayerMetricPrefix(const GameState& state)
		{
			int playersOnTeams[2] = { 0, 0 };
			for (const Player& player : state.players)
				playersOnTeams[(int)player.team]++;

			const int higherCount = RS_MAX(playersOnTeams[0], playersOnTeams[1]);
			const int lowerCount = RS_MIN(playersOnTeams[0], playersOnTeams[1]);

			return "Player" + std::to_string(higherCount) + "v" + std::to_string(lowerCount) + "/";
		}

		struct PlayerMetricKeys
		{
			std::string inAirRatio;
			std::string ballTouchRatio;
			std::string demoedRatio;
			std::string shotRatio;
			std::string saveRatio;
			std::string passShotRatio;
			std::string speed;
			std::string distanceToBall;
			std::string speedTowardsBall;
			std::string boost;
			std::string touchHeight;
			std::string waveDashRatio;
		};

		PlayerMetricKeys MakePlayerMetricKeys(const std::string& prefix)
		{
			return {
				prefix + "In Air Ratio",
				prefix + "Ball Touch Ratio",
				prefix + "Demoed Ratio",
				prefix + "Shot Ratio",
				prefix + "Save Ratio",
				prefix + "Pass Shot Ratio",
				prefix + "Speed",
				prefix + "Distance to Ball",
				prefix + "Speed Towards Ball",
				prefix + "Boost",
				prefix + "Touch Height",
				prefix + "Wave Dash Ratio",
			};
		}

		const PlayerMetricKeys* GetCachedPlayerMetricKeys(const GameState& state)
		{
			int playersOnTeams[2] = { 0, 0 };
			for (const Player& player : state.players)
				playersOnTeams[(int)player.team]++;

			const int higherCount = RS_MAX(playersOnTeams[0], playersOnTeams[1]);
			const int lowerCount = RS_MIN(playersOnTeams[0], playersOnTeams[1]);

			static const PlayerMetricKeys oneVOne = MakePlayerMetricKeys("Player1v1/");
			static const PlayerMetricKeys twoVTwo = MakePlayerMetricKeys("Player2v2/");
			static const PlayerMetricKeys threeVThree = MakePlayerMetricKeys("Player3v3/");

			if (higherCount == 1 && lowerCount == 1)
				return &oneVOne;
			if (higherCount == 2 && lowerCount == 2)
				return &twoVTwo;
			if (higherCount == 3 && lowerCount == 3)
				return &threeVThree;

			return nullptr;
		}

		void AddPlayerMetrics(Report& report, const PlayerMetricKeys& metric, const Player& player, const GameState& state)
		{
			report.AddAvg(metric.inAirRatio, !player.isOnGround);
			report.AddAvg(metric.ballTouchRatio, player.ballTouchedStep);
			report.AddAvg(metric.demoedRatio, player.isDemoed);
			report.AddAvg(metric.shotRatio, player.eventState.shot);
			report.AddAvg(metric.saveRatio, player.eventState.save);
			report.AddAvg(metric.passShotRatio, player.eventState.shotPass);

			report.AddAvg(metric.speed, player.vel.Length());

			const Vec playerToBall = state.ball.pos - player.pos;
			report.AddAvg(metric.distanceToBall, playerToBall.Length());
			const Vec dirToBall = playerToBall.Normalized();
			report.AddAvg(metric.speedTowardsBall, RS_MAX(0, player.vel.Dot(dirToBall)));

			report.AddAvg(metric.boost, player.boost);

			if (player.ballTouchedStep)
				report.AddAvg(metric.touchHeight, state.ball.pos.z);

			report.AddAvg(metric.waveDashRatio, player.isOnGround && player.prev && (player.prev->isFlipping && !player.prev->isOnGround));
		}

		void DeleteStateSetter(WeightedStateSetter& setter)
		{
			delete setter.stateSetter;
			setter.stateSetter = nullptr;
		}

		std::string CleanStateSetterTypeName(const StateSetter* setter)
		{
			if (!setter)
				return "State Setter";

			std::string name;
#ifdef __GNUG__
			int status = 0;
			char* demangled = abi::__cxa_demangle(typeid(*setter).name(), nullptr, nullptr, &status);
			name = (status == 0 && demangled) ? demangled : typeid(*setter).name();
			std::free(demangled);
#else
			name = typeid(*setter).name();
#endif

			const std::string classPrefix = "class ";
			const std::string structPrefix = "struct ";

			if (name.rfind(classPrefix, 0) == 0)
				name.erase(0, classPrefix.size());
			else if (name.rfind(structPrefix, 0) == 0)
				name.erase(0, structPrefix.size());

			const size_t namespacePos = name.rfind("::");
			if (namespacePos != std::string::npos)
				name = name.substr(namespacePos + 2);

			return name.empty() ? "State Setter" : name;
		}

		std::string MakeStateSetterOptionLabel(const WeightedStateSetter& setter, int displayIndex)
		{
			std::ostringstream oss;
			oss << displayIndex << ". " << CleanStateSetterTypeName(setter.stateSetter) << " (weight " << setter.weight << ")";

			return oss.str();
		}

		std::unique_ptr<StateSetter> TakeForcedStateSetter(std::vector<WeightedStateSetter>& setters, int selectedIndex)
		{
			StateSetter* selectedSetter = nullptr;
			int optionIndex = 0;

			for (WeightedStateSetter& setter : setters) {
				if (setter.weight == 0.f) {
					DeleteStateSetter(setter);
					continue;
				}

				if (optionIndex == selectedIndex) {
					selectedSetter = setter.stateSetter;
					setter.stateSetter = nullptr;
				} else {
					DeleteStateSetter(setter);
				}

				optionIndex++;
			}

			return std::unique_ptr<StateSetter>(selectedSetter);
		}
	}

	void StepCallback(Learner* learner, const std::vector<GameState>& states, Report& report)
	{
		// Expensive metrics only run on a quarter of steps to limit their cost.
		const bool doExpensiveMetrics = (rand() % 4) == 0;

		for (int arenaIdx = 0; arenaIdx < (int)states.size(); arenaIdx++) {
			const GameState& state = states[arenaIdx];
			const bool isNormalGameplay = learner && learner->envSet &&
				arenaIdx < (int)learner->envSet->stateSetters.size() &&
				learner->envSet->stateSetters[arenaIdx] &&
				IsNormalGameplayEpisode(learner->envSet->stateSetters[arenaIdx]->setterType);

			if (doExpensiveMetrics && isNormalGameplay) {
				PlayerMetricKeys fallbackKeys;
				const PlayerMetricKeys* metric = GetCachedPlayerMetricKeys(state);
				if (!metric) {
					fallbackKeys = MakePlayerMetricKeys(MakePlayerMetricPrefix(state));
					metric = &fallbackKeys;
				}

				for (const Player& player : state.players)
					AddPlayerMetrics(report, *metric, player, state);
			}

			if (state.goalScored)
				report.AddAvg("Game/Goal Speed", state.ball.vel.Length());
		}
	}

	TrainingStage::TrainingStage(std::unique_ptr<Agent> inAgent)
		: agent(std::move(inAgent))
	{
		if (!agent)
			agent = std::make_unique<Agent>();

		agentTickSkip = agent->tickSkip;

		environmentCreateFunc = [this](const int index, const EnvCreateConfig& config) {
			return CreateEnvironment(index, config);
		};

		stepCallbackFunc = StepCallback;

		learnerConfig = {};

		learnerConfig.deviceType = LearnerDeviceType::AUTO;

		// The optimal count is machine-specific; more games consume more RAM.
		learnerConfig.numGames = 256;

		// -1 uses a random seed each run. The seed can have a strong effect on the outcome of a run.
		learnerConfig.randomSeed = 123;

		// Lower this if too much VRAM is being allocated.
		learnerConfig.ppo.microBatchSize = 50'000;

		// 2 epochs has been the best trade-off between training time and skill.
		learnerConfig.ppo.epochs = 2;

		// Unlike "ent_coef" in other frameworks, this scales normalized entropy,
		// so it doesn't need retuning when actions are added.
		learnerConfig.ppo.entropyScale = 0.035f;

		learnerConfig.ppo.gaeGamma = 0.99;

		learnerConfig.ppo.policyLR = 1.5e-4;
		learnerConfig.ppo.criticLR = 1.5e-4;

		learnerConfig.algorithm = LearnerAlgorithm::PPO;

		learnerConfig.sendMetrics = true;

		learnerConfig.tsPerSave = 25'000'000;

		learnerConfig.tsPerVersion = 25'000'000;

		learnerConfig.skillTracker.enabled = true;

		// Update config values from the agent
		learnerConfig.tickSkip = agent->tickSkip;
		learnerConfig.actionDelay = agent->actionDelay;

		learnerConfig.ppo.sharedHead = agent->sharedHead;
		learnerConfig.ppo.policy = agent->policy;
		learnerConfig.ppo.critic = agent->critic;
		learnerConfig.ppo.decomposedCritic.enabled = agent->decomposedCritic;

		learnerConfig.ppo.policy.optimType = agent->otimType;
		learnerConfig.ppo.critic.optimType = agent->otimType;
		learnerConfig.ppo.sharedHead.optimType = agent->otimType;

		learnerConfig.ppo.policy.activationType = agent->modelActivationType;
		learnerConfig.ppo.critic.activationType = agent->modelActivationType;
		learnerConfig.ppo.sharedHead.activationType = agent->modelActivationType;

		learnerConfig.ppo.policy.addLayerNorm = agent->addLayerNorm;
		learnerConfig.ppo.critic.addLayerNorm = agent->addLayerNorm;
		learnerConfig.ppo.sharedHead.addLayerNorm = agent->addLayerNorm;

		UpdateConfigs();
	}

	EnvCreateResult TrainingStage::CreateEnvironment(const int index, const EnvCreateConfig& config)
	{
		const float stageIndexPercent = config.numArenas > 1 ? index / float(config.numArenas - 1) : -1.f;
		auto [teamSizeA, teamSizeB] = config.teamSizeOverride > 0
			? std::make_pair(config.teamSizeOverride, config.teamSizeOverride)
			: ChooseTeamSize(stageIndexPercent);

		const bool doSwap = (teamSizeA != teamSizeB) && RocketSim::Math::RandFloat(0, 1) < 0.5f;
		int blueSize   = doSwap ? teamSizeB : teamSizeA;
		int orangeSize = doSwap ? teamSizeA : teamSizeB;
		const int maxTeamSize = std::max(teamSizeA, teamSizeB);

		if (std::max(blueSize, orangeSize) > agent->maxTeamSize) {
			RG_ERR_CLOSE("TrainingStage: Requested team size " << std::max(blueSize, orangeSize)
				<< " exceeds Agent maxTeamSize of " << agent->maxTeamSize);
		}

		RocketSim::ArenaConfig arenaCfg = {};
		arenaCfg.engine = config.engine;
		Arena* arena = Arena::Create(config.mode, arenaCfg);

		for (int i = 0; i < blueSize;   i++) arena->AddCar(Team::BLUE, agent->carConfig);
		for (int i = 0; i < orangeSize; i++) arena->AddCar(Team::ORANGE, agent->carConfig);

		EnvCreateResult result = {};
		result.actionParser = agent->MakeActionParser();
		result.obsBuilder = agent->MakeConfiguredObsBuilder();

		// EnvSet takes raw pointers, so ownership is released here.
		result.stateSetter = GetStateSetter(blueSize, orangeSize).release();

		vector<unique_ptr<TerminalCondition>> ownedConditions = GetTerminalConditions();
		result.terminalConditions.reserve(ownedConditions.size());
		for (unique_ptr<TerminalCondition>& condition : ownedConditions)
			result.terminalConditions.push_back(condition.release());

		result.rewards = GetFilteredRewards(maxTeamSize);
		result.arena = arena;

		std::unique_ptr<RewardCacheBase> cache = CreateRewardCache();
		result.userInfo = cache.get();

		if (cache) {
			std::lock_guard<std::mutex> lock(_rewardCacheMutex);
			_rewardCaches.push_back(std::move(cache));
		}

		return result;
	}

	void TrainingStage::UpdateConfigs()
	{
		GGL::TrainingStorage storage = GetTrainingStorage();

		learnerConfig.ppo.tsPerItr = timestepsPerIteration;
		learnerConfig.ppo.batchSize = timestepsPerIteration;

		learnerConfig.engine = RocketSim::EngineFromVersion(agent->trainingEngineVersion);
		learnerConfig.ballPredictionEngine = RocketSim::EngineFromVersion(agent->GetBallPredictionEngineVersion());
		learnerConfig.metricsProjectName = agent->agentName;
		learnerConfig.metricsGroupName = stageName;
		learnerConfig.metricsRunName = runName;

		learnerConfig.ppo.policyLR = learningRate;
		learnerConfig.ppo.criticLR = learningRate;

		GGL::LearnerRuntimeConfig& runtimeConfig = learnerConfig.runtime;
		runtimeConfig.trainingStageName = stageName;
		runtimeConfig.checkpointFolder = storage.GetAgentCheckpointFolder();
		runtimeConfig.checkpointLoadFolder = runtimeConfig.checkpointFolder;

		CalculateTeamSizeWeights();
	}

	GGL::TrainingStorage TrainingStage::GetTrainingStorage() const
	{
		std::filesystem::path agentModelsFolder = RocketForgeTraining::ResolveAgentModelsFolder(agent.get());
		std::filesystem::path stageRootFolder = MakePreferredCopy(
			agentModelsFolder / GGL::TrainingStorage::TRAINING_STAGES_FOLDER_NAME / stageName);

		return GGL::TrainingStorage(std::move(stageRootFolder), std::move(agentModelsFolder));
	}

	void TrainingStage::ApplySegmentId(const std::string& segmentId, const std::string& notes)
	{
		const std::string runPrefix = StripVersionSuffix(runName);
		runName = runPrefix + "-" + segmentId;
		learnerConfig.runtime.versionId = segmentId;
		learnerConfig.runtime.versionNotes = notes;
		UpdateConfigs();
	}

	void TrainingStage::SetParameterOverrides(const std::vector<RocketForgeTraining::StageParameterOverride>& overrides)
	{
		_parameterOverrides = overrides;
		ApplyParameterOverrides();
	}

	void TrainingStage::ApplyParameterOverrides()
	{
		for (const RocketForgeTraining::StageParameterOverride& override : _parameterOverrides) {
			if (override.kind != RocketForgeTraining::StageParameterOverrideKind::Hyperparameter)
				continue;

			const std::string& group = override.group;
			const std::string& name = override.name;

			if (group == "Environment") {
				if (name == "Games") {
					learnerConfig.numGames = static_cast<int>(ParseInt64Override(override));
				} else if (name == "Tick Skip") {
					learnerConfig.tickSkip = static_cast<int>(ParseInt64Override(override));
				} else if (name == "Action Delay") {
					learnerConfig.actionDelay = static_cast<int>(ParseInt64Override(override));
				} else {
					RG_ERR_CLOSE("Unsupported Param Testing environment override: " << name);
				}
			} else if (group == "Training") {
				if (name == "Timesteps / Iteration") {
					timestepsPerIteration = static_cast<int>(ParseInt64Override(override));
					learnerConfig.ppo.tsPerItr = timestepsPerIteration;
					learnerConfig.ppo.batchSize = timestepsPerIteration;
				} else if (name == "Batch Size") {
					learnerConfig.ppo.batchSize = ParseInt64Override(override);
				} else if (name == "Micro Batch Size") {
					learnerConfig.ppo.microBatchSize = ParseInt64Override(override);
				} else if (name == "Epochs") {
					learnerConfig.ppo.epochs = static_cast<int>(ParseInt64Override(override));
				} else if (name == "Random Seed") {
					learnerConfig.randomSeed = ParseInt64Override(override);
				} else if (name == "Learning Rate") {
					learningRate = ParseFloatOverride(override);
					learnerConfig.ppo.policyLR = learningRate;
					learnerConfig.ppo.criticLR = learningRate;
				} else if (name == "Policy LR") {
					learnerConfig.ppo.policyLR = ParseFloatOverride(override);
				} else if (name == "Critic LR") {
					learnerConfig.ppo.criticLR = ParseFloatOverride(override);
				} else if (name == "Entropy Scale") {
					learnerConfig.ppo.entropyScale = ParseFloatOverride(override);
				} else if (name == "GAE Gamma") {
					learnerConfig.ppo.gaeGamma = ParseFloatOverride(override);
				} else if (name == "GAE Lambda") {
					learnerConfig.ppo.gaeLambda = ParseFloatOverride(override);
				} else if (name == "Clip Range") {
					learnerConfig.ppo.clipRange = ParseFloatOverride(override);
				} else if (name == "Normalize Advantages") {
					learnerConfig.ppo.normalizeAdvantages = ParseBoolOverride(override);
				} else if (name == "Policy Grad Clip Norm") {
					learnerConfig.ppo.policyGradClipNorm = ParseFloatOverride(override);
				} else if (name == "Critic Head Loss Weighting") {
					learnerConfig.ppo.decomposedCritic.lossWeighting = ParseCriticHeadLossWeightingOverride(override);
				} else if (name == "Reward Clip Range") {
					learnerConfig.ppo.rewardClipRange = ParseFloatOverride(override);
				} else if (name == "Policy Temperature") {
					learnerConfig.ppo.policyTemperature = ParseFloatOverride(override);
				} else if (name == "Min Action Probability") {
					learnerConfig.ppo.minActionProbability = ParseFloatOverride(override);
				} else if (name == "Mask Entropy") {
					learnerConfig.ppo.maskEntropy = ParseBoolOverride(override);
				} else if (name == "Deterministic") {
					learnerConfig.ppo.deterministic = ParseBoolOverride(override);
				} else if (name == "Use Half Precision") {
					learnerConfig.ppo.useHalfPrecision = ParseBoolOverride(override);
				} else if (name == "Overbatching") {
					learnerConfig.ppo.overbatching = ParseBoolOverride(override);
				} else if (name == "Max Episode Duration") {
					learnerConfig.ppo.maxEpisodeDuration = ParseFloatOverride(override);
				} else {
					RG_ERR_CLOSE("Unsupported Param Testing training override: " << name);
				}
			} else if (group == "Schedules") {
				if (name == "PPO/Entropy Scale Target") {
					learnerConfig.ppo.entropyScale.target = ParseFloatOverride(override);
				} else if (name == "PPO/Entropy Scale Amplitude") {
					learnerConfig.ppo.entropyScale.amplitude = ParseFloatOverride(override);
				} else if (name == "PPO/Entropy Scale Period Timesteps") {
					learnerConfig.ppo.entropyScale.period = ParseUint64Override(override);
				} else if (name == "PPO/Entropy Scale Start Timestep") {
					learnerConfig.ppo.entropyScale.startTimestep = ParseUint64Override(override);
				} else {
					RG_ERR_CLOSE("Unsupported Param Testing schedule override: " << name);
				}
			} else if (group == "PPG") {
				if (name == "Policy Epochs") {
					learnerConfig.ppg.policyEpochs = static_cast<int>(ParseInt64Override(override));
				} else if (name == "Critic Epochs") {
					learnerConfig.ppg.criticEpochs = static_cast<int>(ParseInt64Override(override));
				} else if (name == "Aux Phase Interval") {
					learnerConfig.ppg.auxPhaseInterval = static_cast<int>(ParseInt64Override(override));
				} else if (name == "Aux Epochs") {
					learnerConfig.ppg.auxEpochs = static_cast<int>(ParseInt64Override(override));
				} else if (name == "Aux Batch Size") {
					learnerConfig.ppg.auxBatchSize = ParseInt64Override(override);
				} else if (name == "Aux Policy LR") {
					learnerConfig.ppg.auxPolicyLR = ParseFloatOverride(override);
				} else if (name == "Aux Critic LR") {
					learnerConfig.ppg.auxCriticLR = ParseFloatOverride(override);
				} else if (name == "Aux Value Loss Scale") {
					learnerConfig.ppg.auxValueLossScale = ParseFloatOverride(override);
				} else if (name == "Aux Policy Clone Loss Scale") {
					learnerConfig.ppg.auxPolicyCloneLossScale = ParseFloatOverride(override);
				} else if (name == "Aux True Critic Loss Scale") {
					learnerConfig.ppg.auxTrueCriticLossScale = ParseFloatOverride(override);
				} else if (name == "Aux Grad Clip") {
					learnerConfig.ppg.auxGradClip = ParseFloatOverride(override);
				} else {
					RG_ERR_CLOSE("Unsupported Param Testing PPG override: " << name);
				}
			} else if (group == "Saving") {
				if (name == "Timesteps / Save") {
					learnerConfig.tsPerSave = ParseInt64Override(override);
				} else if (name == "Timesteps / Version") {
					learnerConfig.tsPerVersion = ParseInt64Override(override);
				} else if (name == "Checkpoints To Keep") {
					learnerConfig.checkpointsToKeep = static_cast<int>(ParseInt64Override(override));
				} else if (name == "Save Policy Versions") {
					learnerConfig.savePolicyVersions = ParseBoolOverride(override);
				} else if (name == "Max Recent Versions") {
					learnerConfig.maxRecentVersions = static_cast<int>(ParseInt64Override(override));
				} else if (name == "Train Against Recent Versions") {
					learnerConfig.trainAgainstRecentVersions = ParseBoolOverride(override);
				} else if (name == "Train Against Recent Percent") {
					learnerConfig.trainAgainstRecentPercent = ParseFloatOverride(override);
				} else if (name == "Recent Opponent Cohort Size") {
					learnerConfig.recentOpponentCohortSize = static_cast<int>(ParseInt64Override(override));
				} else if (name == "Recent Opponent Rotation Timesteps") {
					learnerConfig.recentOpponentRotationTimesteps = ParseUint64Override(override);
				} else if (name == "Train Against Static Opponents") {
					learnerConfig.trainAgainstStaticOpponents = ParseBoolOverride(override);
				} else if (name == "Train Against Static Percent") {
					learnerConfig.trainAgainstStaticPercent = ParseFloatOverride(override);
				} else if (name == "Static Opponent Cohort Size") {
					learnerConfig.staticOpponentCohortSize = static_cast<int>(ParseInt64Override(override));
				} else if (name == "Static Opponent Rotation Timesteps") {
					learnerConfig.staticOpponentRotationTimesteps = ParseUint64Override(override);
				} else if (name == "Warm Start Policy") {
					learnerConfig.warmStartPolicy = ParseBoolOverride(override);
				} else if (name == "Warm Start Iterations") {
					learnerConfig.warmStartIterations = static_cast<int>(ParseInt64Override(override));
				} else if (name == "Warm Start Policy LR Multiplier") {
					learnerConfig.warmStartPolicyLRMult = ParseFloatOverride(override);
				} else {
					RG_ERR_CLOSE("Unsupported Param Testing saving override: " << name);
				}
			} else if (group == "Normalization") {
				if (name == "Standardize Obs") {
					learnerConfig.standardizeObs = ParseBoolOverride(override);
				} else if (name == "Standardize Returns") {
					learnerConfig.standardizeReturns = ParseBoolOverride(override);
				} else if (name == "Min Obs STD") {
					learnerConfig.minObsSTD = ParseFloatOverride(override);
				} else if (name == "Max Obs Mean Range") {
					learnerConfig.maxObsMeanRange = ParseFloatOverride(override);
				} else if (name == "Max Obs Samples") {
					learnerConfig.maxObsSamples = static_cast<int>(ParseInt64Override(override));
				} else if (name == "Max Return Samples") {
					learnerConfig.maxReturnSamples = static_cast<int>(ParseInt64Override(override));
				} else {
					RG_ERR_CLOSE("Unsupported Param Testing normalization override: " << name);
				}
			} else if (group == "Skill Test") {
				if (name == "Target Confidence") {
					learnerConfig.skillTracker.targetConfidence = ParseFloatOverride(override);
				} else if (name == "Reference Games") {
					learnerConfig.skillTracker.referenceGamesPerOpponent = static_cast<int>(ParseInt64Override(override));
				} else if (name == "Reference Calibration Games") {
					learnerConfig.skillTracker.referenceCalibrationGames = static_cast<int>(ParseInt64Override(override));
				} else if (name == "Rating Drift Per 100M") {
					learnerConfig.skillTracker.ratingDriftSigmaPer100MTimesteps = ParseFloatOverride(override);
				} else if (name == "Anchor Retire Win Prob") {
					learnerConfig.skillTracker.anchorRetireExpectedWinProb = ParseFloatOverride(override);
				} else {
					RG_ERR_CLOSE("Unsupported Param Testing skill test override: " << name);
				}
			} else {
				RG_ERR_CLOSE("Unsupported Param Testing hyperparameter override group: " << group << " / " << name);
			}
		}
	}

	vector<WeightedReward> TrainingStage::GetRewards() const
	{
		const int defaultMaxTeamSize = 1;
		vector<RewardSpec> rewardSpecs = GetRewardSpecs();
		if (!rewardSpecs.empty())
			return ResolveRewardSpecs(rewardSpecs, defaultMaxTeamSize);

		return {
			// Movement
			{ new AirReward(), 0.25f },

			// Player-ball
			{ new FaceBallReward(), 0.25f },
			{ new VelocityPlayerToBallReward(), 4.f },
			{ new StrongTouchReward(20, 100), 60 },

			// Ball-goal
			{ new ZeroSumReward(new VelocityBallToGoalReward(), 1), 2.0f },

			// Boost
			{ new PickupBoostReward(), 10.f },
			{ new SaveBoostReward(), 0.2f },

			// Game events
			{ new ZeroSumReward(new BumpReward(), 0.5f), 20 },
			{ new ZeroSumReward(new DemoReward(), 0.5f), 80 },
			{ new GoalReward(), 150 }
		};
	}

	vector<WeightedReward> TrainingStage::GetRewards(int maxTeamSize) const
	{
		vector<RewardSpec> rewardSpecs = GetRewardSpecs();
		if (!rewardSpecs.empty())
			return ResolveRewardSpecs(rewardSpecs, maxTeamSize);

		return GetRewards();
	}

	vector<WeightedReward> TrainingStage::GetConfiguredRewards() const
	{
		vector<WeightedReward> rewards = GetRewards();
		ApplyRewardWeightOverrides(rewards);

		return rewards;
	}

	vector<WeightedReward> TrainingStage::GetConfiguredRewards(int maxTeamSize) const
	{
		vector<WeightedReward> rewards = GetRewards(maxTeamSize);
		ApplyRewardWeightOverrides(rewards);

		return rewards;
	}

	std::vector<RLGC::WeightedReward> TrainingStage::GetFilteredRewards(int maxTeamSize)
	{
		vector<WeightedReward> filteredRewards = GetConfiguredRewards(maxTeamSize);

		filteredRewards.erase(
			remove_if(filteredRewards.begin(), filteredRewards.end(), [](const WeightedReward& reward) { return reward.weight == 0.f; }),
			filteredRewards.end());

		return filteredRewards;
	}

	void TrainingStage::ApplyRewardWeightOverrides(vector<WeightedReward>& rewards) const
	{
		for (const RocketForgeTraining::StageParameterOverride& override : _parameterOverrides) {
			if (override.kind != RocketForgeTraining::StageParameterOverrideKind::RewardWeight)
				continue;

			bool matched = false;
			const float weight = ParseFloatOverride(override);

			for (WeightedReward& reward : rewards) {
				if (!reward.reward || reward.reward->GetName() != override.name)
					continue;

				reward.weight = weight;
				matched = true;
			}

			if (!matched)
				RG_ERR_CLOSE("Param Testing reward override references unknown active reward: " << override.name);
		}
	}

	vector<WeightedReward> TrainingStage::CombineRewards(
		const vector<WeightedReward>& baseRewards,
		const vector<WeightedReward>& newRewards) const
	{
		vector<WeightedReward> combinedRewards = baseRewards;

		combinedRewards.erase(
			remove_if(combinedRewards.begin(), combinedRewards.end(), [](const WeightedReward& reward) { return reward.weight == 0.f; }),
			combinedRewards.end());

		combinedRewards.insert(combinedRewards.end(), newRewards.begin(), newRewards.end());

		return combinedRewards;
	}

	vector<RewardSpec> TrainingStage::GetRewardSpecs() const
	{
		return {};
	}

	vector<RewardSpec> TrainingStage::CombineRewardSpecs(const vector<RewardSpec>& baseRewards, const vector<RewardSpec>& newRewards) const
	{
		vector<RewardSpec> combinedRewards = {};
		combinedRewards.reserve(baseRewards.size() + newRewards.size());

		for (const RewardSpec& rewardSpec : baseRewards) {
			if (rewardSpec.HasAnyWeight())
				combinedRewards.push_back(rewardSpec);
		}

		for (const RewardSpec& rewardSpec : newRewards) {
			if (rewardSpec.HasAnyWeight())
				combinedRewards.push_back(rewardSpec);
		}

		return combinedRewards;
	}

	vector<WeightedReward> TrainingStage::ResolveRewardSpecs(const vector<RewardSpec>& rewardSpecs, int maxTeamSize) const
	{
		vector<WeightedReward> resolvedRewards = {};
		resolvedRewards.reserve(rewardSpecs.size());

		for (const RewardSpec& rewardSpec : rewardSpecs) {
			const float resolvedWeight = rewardSpec.ResolveWeight(maxTeamSize);
			if (resolvedWeight == 0.f)
				continue;

			Reward* reward = rewardSpec.ReleaseReward();
			if (!reward)
				RG_ERR_CLOSE("TrainingStage: RewardSpec reward was released before resolution");

			resolvedRewards.push_back({ reward, resolvedWeight, rewardSpec.adaptiveSettings.Resolve(maxTeamSize) });
		}

		return resolvedRewards;
	}

	vector<unique_ptr<TerminalCondition>> TrainingStage::GetTerminalConditions() const
	{
		const float noTouchTimeoutSeconds = 15;

		vector<unique_ptr<TerminalCondition>> conditions;
		conditions.emplace_back(make_unique<NoTouchCondition>(noTouchTimeoutSeconds));
		conditions.emplace_back(make_unique<GoalScoreCondition>());

		return conditions;
	}

	std::vector<DebugStateSetterOption> TrainingStage::GetDebugStateSetterOptions() const
	{
		std::vector<DebugStateSetterOption> options;
		std::vector<WeightedStateSetter> stateSetters = GetStateSetters();
		int optionIndex = 0;

		for (WeightedStateSetter& setter : stateSetters) {
			if (setter.weight == 0.f) {
				DeleteStateSetter(setter);
				continue;
			}

			options.push_back({
				.index = optionIndex,
				.name = CleanStateSetterTypeName(setter.stateSetter),
				.label = MakeStateSetterOptionLabel(setter, optionIndex + 1),
				.weight = setter.weight
			});

			DeleteStateSetter(setter);
			optionIndex++;
		}

		return options;
	}

	unique_ptr<StateSetter> TrainingStage::GetStateSetter(const bool filter) const
	{
		return GetStateSetter(1, 1, filter);
	}

	unique_ptr<StateSetter> TrainingStage::GetStateSetter(int blueSize, int orangeSize, const bool filter) const
	{
		// NOTE: A forced debug selection intentionally bypasses matchup filtering.
		// An index with no matching active setter falls back to the stage's weighted mix.
		if (_debugSettings && _debugSettings->forceSingleStateSetter) {
			vector<WeightedStateSetter> forcedStateSetters = GetStateSetters();
			const int forcedIndex = _debugSettings->forceStateSetterIndex;
			std::unique_ptr<StateSetter> forcedSetter = TakeForcedStateSetter(forcedStateSetters, forcedIndex);

			if (forcedSetter)
				return forcedSetter;
		}

		vector<WeightedStateSetter> stateSetters = GetStateSetters();

		if (filter) {
			for (auto it = stateSetters.begin(); it != stateSetters.end();) {
				if (it->weight == 0.f || !ShouldUseStateSetter(*it, blueSize, orangeSize)) {
					DeleteStateSetter(*it);
					it = stateSetters.erase(it);
				} else {
					++it;
				}
			}
		}

		if (stateSetters.empty())
			RG_ERR_CLOSE("TrainingStage: No valid state setters remain for " << blueSize << "v" << orangeSize << " after filtering");

		vector<pair<StateSetter*, float>> combinedStateSetters;
		combinedStateSetters.reserve(stateSetters.size());

		for (const WeightedStateSetter& setter : stateSetters) {
			if (!setter.stateSetter)
				RG_ERR_CLOSE("TrainingStage: State setter entry has a null stateSetter");

			combinedStateSetters.push_back({ setter.stateSetter, setter.weight });
		}

		return make_unique<CombinedState>(combinedStateSetters);
	}

	unique_ptr<StateSetter> TrainingStage::GetNormalGameplayStateSetter(int blueSize, int orangeSize) const
	{
		vector<WeightedStateSetter> stateSetters = GetStateSetters();

		for (auto it = stateSetters.begin(); it != stateSetters.end();) {
			const bool keep = it->weight > 0.f
				&& it->stateSetter
				&& ShouldUseStateSetter(*it, blueSize, orangeSize)
				&& IsNormalGameplayEpisode(it->stateSetter->GetSetterType());
			if (!keep) {
				DeleteStateSetter(*it);
				it = stateSetters.erase(it);
			} else {
				++it;
			}
		}

		if (stateSetters.empty())
			return nullptr;

		vector<pair<StateSetter*, float>> combinedStateSetters;
		combinedStateSetters.reserve(stateSetters.size());
		for (const WeightedStateSetter& setter : stateSetters)
			combinedStateSetters.push_back({ setter.stateSetter, setter.weight });

		return make_unique<CombinedState>(combinedStateSetters);
	}

	vector<WeightedStateSetter> TrainingStage::GetStateSetters() const
	{
		return { { new KickoffState(), 1.f } };
	}

	vector<WeightedStateSetter> TrainingStage::CombineStateSetters(
		const vector<WeightedStateSetter>& baseSetters,
		const vector<WeightedStateSetter>& newSetters) const
	{
		vector<WeightedStateSetter> combinedSetters = baseSetters;

		combinedSetters.insert(combinedSetters.end(), newSetters.begin(), newSetters.end());

		return combinedSetters;
	}

	bool TrainingStage::ShouldUseStateSetter(const WeightedStateSetter& setter, int blueSize, int orangeSize) const
	{
		const bool isKickoffSetter = setter.stateSetter && setter.stateSetter->setterType == RLGC::StateSetterType::Kickoff;

		return !(disableKickoffsForUnfairMatchups && isKickoffSetter && blueSize != orangeSize);
	}

	void TrainingStage::ReleaseRewardCache(const void* userInfo)
	{
		std::lock_guard<std::mutex> lock(_rewardCacheMutex);
		std::erase_if(_rewardCaches, [&](const std::unique_ptr<RewardCacheBase>& cache) { return cache.get() == userInfo; });
	}

	void TrainingStage::SetDebugSettings(const ForgeDebug::DebugSettings* settings)
	{
		_debugSettings = settings;
	}

	std::pair<int, int> TrainingStage::ChooseTeamSize(const float selectionValue) const
	{
		if (_debugSettings && _debugSettings->teamSizeOverrideIndex >= 0) {
			const int overrideIndex = _debugSettings->teamSizeOverrideIndex;
			if (overrideIndex < (int)weightedTeamSizes.size()) {
				const WeightedTeamSize& teamSize = weightedTeamSizes[overrideIndex];
				return { teamSize.teamSizeA, teamSize.teamSizeB };
			}
		}

		if (_cumulativeTeamWeights.size() != weightedTeamSizes.size())
			RG_ERR_CLOSE(
				"TrainingStage: CumulativeTeamWeights size \"" << _cumulativeTeamWeights.size()
				<< "\" doesn't match WeightedTeamSizes size of " << weightedTeamSizes.size());

		const float selectedWeightValue = selectionValue >= 0.f
			? selectionValue * _totalTeamWeight
			: RocketSim::Math::RandFloat(0, _totalTeamWeight);

		for (int i = 0; i < weightedTeamSizes.size(); i++) {
			if (selectedWeightValue <= _cumulativeTeamWeights[i])
				return { weightedTeamSizes[i].teamSizeA, weightedTeamSizes[i].teamSizeB };
		}

		RG_ERR_CLOSE(
			"TrainingStage::ChooseTeamSize ran out of team sizes before the matching cumulative weight was found "
			"(this should never happen)");

		return { 1, 1 };
	}

	void TrainingStage::CalculateTeamSizeWeights()
	{
		_totalTeamWeight = 0.f;
		_cumulativeTeamWeights.clear();

		for (const WeightedTeamSize& teamSize : weightedTeamSizes) {
			if (teamSize.teamSizeA <= 0 || teamSize.teamSizeB <= 0) {
				RG_ERR_CLOSE("TrainingStage: Weighted Team Size \"" << teamSize.teamSizeA << "v" << teamSize.teamSizeB
					<< "\" is negative or invalid");
			}

			if (teamSize.weight <= 0 || std::isnan(teamSize.weight) || std::isinf(teamSize.weight)) {
				RG_ERR_CLOSE("TrainingStage: Weighted Team Size \"" << teamSize.teamSizeA << "v" << teamSize.teamSizeB
					<< "\" has a negative or invalid weight of " << teamSize.weight);
			}

			if (std::max(teamSize.teamSizeA, teamSize.teamSizeB) > agent->maxTeamSize) {
				RG_ERR_CLOSE("TrainingStage: Weighted Team Size \"" << teamSize.teamSizeA << "v" << teamSize.teamSizeB
					<< "\" exceeds Agent maxTeamSize of " << agent->maxTeamSize);
			}

			// Normalize by max team size so equal weights run equally often regardless of team imbalance.
			_totalTeamWeight += teamSize.weight / float(std::max(teamSize.teamSizeA, teamSize.teamSizeB));
			_cumulativeTeamWeights.push_back(_totalTeamWeight);
		}
	}

	filesystem::path TrainingStage::GetMostRecentCheckpoint() const
	{
		filesystem::path checkpointRoot = learnerConfig.runtime.checkpointLoadFolder.empty()
			? learnerConfig.runtime.checkpointFolder
			: learnerConfig.runtime.checkpointLoadFolder;

		return GGL::TrainingStorage::GetMostRecentCheckpointIn(checkpointRoot);
	}
}
