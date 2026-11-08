#include "Learner.h"

#ifndef _MSC_VER
#include <csignal>
#endif

#include <GigaLearnCPP/PPO/PPOLearner.h>
#include <GigaLearnCPP/PPO/TransferLearning.h>
#include <GigaLearnCPP/PPG/PPGLearner.h>
#include <GigaLearnCPP/PPO/ExperienceBuffer.h>
#include <GigaLearnInfer/ModelManager.h>
#include <GigaLearnInfer/RecentPolicyEnsemble.h>
#include <GigaLearnInfer/JsonArchive.h>

#include <torch/cuda.h>
#include <ATen/Parallel.h>
#include <ATen/ops/index_select.h>
#include <RLGymCPP/ThreadPool.h>
#include <nlohmann/json.hpp>
#include <bit>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>

#ifdef RG_CUDA_SUPPORT
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime_api.h>
#endif
#include <GigaLearnCPP/PPO/GAE.h>
#include <GigaLearnCPP/PPO/PerModeCriticStats.h>
#include <GigaLearnCPP/PPO/DecomposedCritic.h>
#include <GigaLearnCPP/PPO/DecomposedGAE.h>
#include <GigaLearnCPP/PPO/PopArtStats.h>
#include <GigaLearnCPP/PolicyVersionManager.h>
#include <GigaLearnCPP/SkillReferenceRatings.h>
#include <GigaLearnCPP/AdaptiveWeightPersistence.h>

#include <GigaLearnCPP/RecentOpponentScheduler.h>
#include <GigaLearnCPP/Util/WelfordStat.h>
#include "Util/AvgTracker.h"

using namespace RLGC;

namespace {
	constexpr uint64_t CUDA_INFERENCE_TIMING_SAMPLE_INTERVAL = 16;
	constexpr int64_t LEARNER_CUDA_GRAPH_BATCH_QUANTUM = 64;
	constexpr int RECENT_ROWS_PER_SLOT_QUANTUM = 16;
	constexpr int PLAYERS_PER_APPEND_JOB = 64;

	struct PersistentPolicyInputBucket {
		torch::Tensor states;
		torch::Tensor actionMasks;
	};

	int64_t GetCudaGraphBatchBucket(int64_t batchSize) {
		if (batchSize <= LEARNER_CUDA_GRAPH_BATCH_QUANTUM)
			return LEARNER_CUDA_GRAPH_BATCH_QUANTUM;
		if (batchSize > INT64_MAX - (LEARNER_CUDA_GRAPH_BATCH_QUANTUM - 1))
			return batchSize;
		return ((batchSize + LEARNER_CUDA_GRAPH_BATCH_QUANTUM - 1)
			/ LEARNER_CUDA_GRAPH_BATCH_QUANTUM) * LEARNER_CUDA_GRAPH_BATCH_QUANTUM;
	}

	// An absent registry or an empty list means no references; a registry that exists but is unusable is fatal.
	bool RegistryListsSkillReferences(const std::filesystem::path& registryPath) {
		if (registryPath.empty() || !std::filesystem::exists(registryPath))
			return false;

		std::ifstream fIn(registryPath);
		if (!fIn.good())
			RG_ERR_CLOSE("Failed to open skill reference registry: " << registryPath);

		const nlohmann::json registry = nlohmann::json::parse(fIn, nullptr, false);
		if (registry.is_discarded() || !registry.is_object())
			RG_ERR_CLOSE("Failed to parse skill reference registry: " << registryPath);

		const nlohmann::json references = registry.value(GGL::SKILL_REFERENCES_LIST_KEY, nlohmann::json::array());
		if (!references.is_array())
			RG_ERR_CLOSE("Skill reference registry " << registryPath << " has a non-array " << GGL::SKILL_REFERENCES_LIST_KEY);

		return !references.empty();
	}

	std::string SanitizeMetricSegment(const std::string& text) {
		std::string result = {};
		bool lastWasSeparator = false;
		for (unsigned char c : text) {
			if (std::isalnum(c)) {
				result.push_back((char)std::tolower(c));
				lastWasSeparator = false;
			} else if (!lastWasSeparator) {
				result.push_back('_');
				lastWasSeparator = true;
			}
		}

		while (!result.empty() && result.front() == '_')
			result.erase(result.begin());
		while (!result.empty() && result.back() == '_')
			result.pop_back();

		return result.empty() ? "unknown" : result;
	}

	std::string MakeStaticMetricKeySuffix(const GGL::PolicyVersion& version) {
		const std::string sourceAgent = version.sourceAgentName.empty() ? "unknown-agent" : version.sourceAgentName;
		const std::string sourceStage = version.sourceStageName.empty() ? "unknown-stage" : version.sourceStageName;
		// A source version may contain multiple distinct imported variants. Prefer the
		// selected model label so their metrics cannot collide.
		const std::string sourceVersion = !version.label.empty()
			? version.label
			: (version.sourceVersionId.empty() ? "unknown-version" : version.sourceVersionId);
		return SanitizeMetricSegment(sourceAgent) + "/"
			+ SanitizeMetricSegment(sourceStage) + "/"
			+ SanitizeMetricSegment(sourceVersion);
	}

	int FindNeutralActionIndex(ActionParser& parser, const Player& player, const GameState& state) {
		int bestIndex = 0;
		float bestMagnitude = INFINITY;
		for (int actionIndex = 0; actionIndex < parser.GetActionAmount(); actionIndex++) {
			const Action action = parser.ParseAction(actionIndex, player, state);
			float magnitude = 0.f;
			for (int element = 0; element < Action::ELEM_AMOUNT; element++)
				magnitude += std::abs(action[element]);
			if (magnitude < bestMagnitude) {
				bestMagnitude = magnitude;
				bestIndex = actionIndex;
			}
		}
		return bestIndex;
	}

	void SetStepsPerSecond(GGL::Report& report, const std::string& key, double steps, double seconds) {
		if (seconds > 0)
			report[key] = steps / seconds;
	}

	bool ContainsNonFiniteFloat(const std::vector<float>& values) {
		for (float value : values) {
			const uint32_t bits = std::bit_cast<uint32_t>(value);
			if ((bits & 0x7f800000u) == 0x7f800000u)
				return true;
		}

		return false;
	}

	template <typename T>
	void AppendDimListRow(std::vector<T>& dest, const DimList2<T>& source, size_t rowIdx) {
		RG_ASSERT(rowIdx < source.size[0]);
		size_t rowSize = source.size[1];
		auto startItr = source.data.begin() + (rowIdx * rowSize);
		dest.insert(dest.end(), startItr, startItr + rowSize);
	}

	void ApplyScheduledFloat(
		GGL::ScheduledFloat& configured,
		GGL::ScheduledFloat& live,
		uint64_t totalTimesteps,
		const char* name,
		const char* metricKey,
		GGL::Report& report) {
		if (!configured.HasFiniteInputs())
			RG_ERR_CLOSE("ScheduledFloat \"" << name << "\" has non-finite schedule inputs");
		if (configured.amplitude != 0.f && configured.period == 0)
			RG_ERR_CLOSE("ScheduledFloat \"" << name << "\" needs a positive period when amplitude is non-zero");

		const float value = configured.Update(totalTimesteps);
		if (!std::isfinite(value))
			RG_ERR_CLOSE("ScheduledFloat \"" << name << "\" produced a non-finite value");

		live = configured;
		if (configured.IsScheduled())
			report[metricKey] = value;
	}

	void ApplyScheduledConfigValues(GGL::Learner& learner, GGL::Report& report) {
		ApplyScheduledFloat(
			learner.config.ppo.entropyScale,
			learner.ppo->config.entropyScale,
			learner.totalTimesteps,
			"PPO/Entropy Scale",
			"Scheduled/PPO/Entropy Scale",
			report);
	}
	
	std::string TeamSizeLabel(int playerCount) {
		if (playerCount > 0 && playerCount % 2 == 0) {
			int perTeam = playerCount / 2;
			return std::to_string(perTeam) + "v" + std::to_string(perTeam);
		}
		return std::to_string(playerCount) + "p";
	}
}

GGL::Learner::Learner(EnvCreateFn envCreateFn, LearnerConfig config, StepCallbackFn stepCallback) :
	envCreateFn(envCreateFn), config(config), stepCallback(stepCallback)
{
#ifndef NDEBUG
	RG_LOG("===========================");
	RG_LOG("WARNING: GigaLearn runs extremely slowly in debug, and there are often bizzare issues with debug-mode torch.");
	RG_LOG("It is recommended that you compile in release mode without optimization for debugging.");
	RG_SLEEP(1000);
#endif

	const LearnerRuntimeConfig& runtimeConfig = config.runtime;

	if (config.tsPerSave == 0)
		config.tsPerSave = config.ppo.tsPerItr;

	float enabledRecentPercent = config.trainAgainstRecentVersions ? config.trainAgainstRecentPercent : 0.0f;
	float enabledStaticPercent = config.trainAgainstStaticOpponents ? config.trainAgainstStaticPercent : 0.0f;
	if (enabledRecentPercent < 0 || enabledRecentPercent > 1)
		RG_ERR_CLOSE("trainAgainstRecentPercent must be between 0 and 1 when enabled");
	if (config.trainAgainstRecentVersions && config.recentOpponentCohortSize <= 0)
		RG_ERR_CLOSE("recentOpponentCohortSize must be positive when training against recent versions");
	if (config.trainAgainstRecentVersions && config.recentOpponentRotationTimesteps == 0)
		RG_ERR_CLOSE("recentOpponentRotationTimesteps must be positive when training against recent versions");
	if (enabledStaticPercent < 0 || enabledStaticPercent > 1)
		RG_ERR_CLOSE("trainAgainstStaticPercent must be between 0 and 1 when enabled");
	if (config.trainAgainstStaticOpponents && config.staticOpponentCohortSize <= 0)
		RG_ERR_CLOSE("staticOpponentCohortSize must be positive when training against static opponents");
	if (config.trainAgainstStaticOpponents && config.staticOpponentRotationTimesteps == 0)
		RG_ERR_CLOSE("staticOpponentRotationTimesteps must be positive when training against static opponents");
	if (enabledRecentPercent + enabledStaticPercent > 1)
		RG_ERR_CLOSE("Enabled trainAgainstRecentPercent + trainAgainstStaticPercent must be <= 1");
	if (config.trainAgainstStaticOpponents && runtimeConfig.staticOpponentsRegistryPath.empty())
		RG_ERR_CLOSE("Cannot train against static opponents with no static opponent registry");

	RG_LOG("Learner::Learner():");

	// Mirror the collection pool's CPU policy for libtorch's own worker threads: pin
	// this (main) thread to one NUMA node so libtorch's threads inherit the affinity,
	// and cap intra-op parallelism to the CPU budget instead of the host's full core
	// count (which oversubscribes the cgroup quota on shared many-core hosts).
	RLGC::PinCurrentThreadToPreferredNode();
	at::set_num_threads((int)RLGC::EffectiveThreadCount());

	if (config.randomSeed == -1)
		config.randomSeed = RS_CUR_MS();

	RG_LOG("\tCheckpoint Save/Load Dir: " << runtimeConfig.checkpointFolder);
	if (!runtimeConfig.checkpointLoadFolder.empty() && runtimeConfig.checkpointLoadFolder != runtimeConfig.checkpointFolder)
		RG_LOG("\tCheckpoint Resume Dir: " << runtimeConfig.checkpointLoadFolder);

	torch::manual_seed(config.randomSeed);

	at::Device device = at::Device(at::kCPU);
	if (
		config.deviceType == LearnerDeviceType::GPU_CUDA || 
		(config.deviceType == LearnerDeviceType::AUTO && torch::cuda::is_available())
		) {
		RG_LOG("\tUsing CUDA GPU device...");

		// Test out moving a tensor to GPU and back to make sure the device is working
		torch::Tensor t;
		bool deviceTestFailed = false;
		try {
			t = torch::tensor(0);
			t = t.to(at::Device(at::kCUDA));
			t = t.cpu();
		} catch (...) {
			deviceTestFailed = true;
		}

		if (!torch::cuda::is_available() || deviceTestFailed)
			RG_ERR_CLOSE(
				"Learner::Learner(): Can't use CUDA GPU because " <<
				(torch::cuda::is_available() ? "libtorch cannot access the GPU" : "CUDA is not available to libtorch") << ".\n" <<
				"Make sure your libtorch comes with CUDA support, and that CUDA is installed properly."
			)
		device = at::Device(at::kCUDA);
	} else {
		RG_LOG("\tUsing CPU device...");
#ifdef RG_CUDA_SUPPORT
		// A CUDA build landing here usually means a driver or install problem rather than a deliberate CPU run.
		if (config.deviceType == LearnerDeviceType::AUTO)
			RG_LOG("\tWARNING: libtorch was built with CUDA, but no CUDA GPU is available; training on the CPU is much slower.");
#endif
		device = at::Device(at::kCPU);
	}

	if (RocketSim::GetStage() != RocketSimStage::INITIALIZED) {
		RG_LOG("\tInitializing RocketSim...");
		RocketSim::Init(RocketSim::CollisionMeshesFolder(), true);
	}

	{
		RG_LOG("\tCreating envs...");
		EnvSetConfig envSetConfig = {};
		envSetConfig.envCreateFn = envCreateFn;
		envSetConfig.numArenas = config.numGames;
		envSetConfig.envCreateCfg = EnvCreateConfig(envSetConfig.numArenas, config.gameMode);
		envSetConfig.envCreateCfg.engine = config.engine;
		envSetConfig.tickSkip = config.tickSkip;
		envSetConfig.actionDelay = config.actionDelay;
		envSetConfig.saveRewards = config.addRewardsToMetrics;
		envSet = new RLGC::EnvSet(envSetConfig);
		obsSize = envSet->state.obs.size[1];
		numActions = envSet->actionParsers[0]->GetActionAmount();

		int maxCreatedTeamSize = 0;
		for (auto& state : envSet->state.gameStates) {
			int playersOnTeams[2] = { 0, 0 };
			for (auto& player : state.players)
				playersOnTeams[(int)player.team]++;
			maxCreatedTeamSize = RS_MAX(maxCreatedTeamSize, RS_MAX(playersOnTeams[0], playersOnTeams[1]));
		}
		envSet->config.envCreateCfg.maxTeamSize = maxCreatedTeamSize;

		if (config.ppo.decomposedCritic.enabled) {
			ValidateDecomposedCriticConfig(config);

			if (config.ppo.rewardClipRange != PPOLearnerConfig{}.rewardClipRange)
				RG_LOG("Warning: ppo.rewardClipRange is ignored with the decomposed critic");

			envSet->SetRewardHeadLayout(RLGC::RewardHeadLayout::Build(envSet->rewards));

			std::stringstream headList;
			for (const std::string& name : envSet->rewardHeadLayout->GetHeadNames())
				headList << "\n\t\t" << name;
			RG_LOG("\tCritic value heads (" << envSet->rewardHeadLayout->GetNumHeads() << "):" << headList.str());

			const DecomposedCriticConfig& decomposedConfig = config.ppo.decomposedCritic;
			popArtStats = std::make_unique<PopArtStats>(
				envSet->rewardHeadLayout->GetHeadNames(), decomposedConfig.popArtBeta, decomposedConfig.popArtSigmaFloor);
		} else {
			for (const std::string& name : RLGC::RewardHeadLayout::FindDuplicateNames(envSet->rewards))
				RG_LOG("Warning: reward \"" << name << "\" is listed more than once in one arena; "
					"its metrics and Param Testing weight overrides are shared");
		}
	}

	{
		if (config.standardizeReturns && !popArtStats) {
			this->returnStat = new WelfordStat();
		} else {
			this->returnStat = NULL;
		}

		if (config.standardizeObs) {
			this->obsStat = new BatchedWelfordStat(obsSize);
		} else {
			this->obsStat = NULL;
		}
	}

	try {
		if (config.algorithm == LearnerAlgorithm::PPG) {
			RG_LOG("\tMaking PPG learner...");
			ppo = new PPGLearner(obsSize, numActions, config.ppo, config.ppg, device);
		} else {
			RG_LOG("\tMaking PPO learner...");
			const bool initializeModels = true;
			const int numCriticOutputs = popArtStats ? popArtStats->GetNumHeads() : 1;
			ppo = new PPOLearner(obsSize, numActions, config.ppo, device, initializeModels, numCriticOutputs);
		}
	} catch (std::exception& e) {
		RG_ERR_CLOSE("Failed to create policy-gradient learner: " << e.what());
	}

	// Production recent-opponent training keeps creating snapshots as before. A caller
	// with only an explicit recent-policy folder can deliberately load it read-only,
	// which is used by the non-mutating benchmark path.
	if (config.trainAgainstRecentVersions && !runtimeConfig.checkpointFolder.empty())
		config.savePolicyVersions = true;

	if (config.trainAgainstRecentVersions || config.savePolicyVersions || config.skillTracker.enabled) {
		if (runtimeConfig.checkpointFolder.empty() && runtimeConfig.recentPolicyVersionsFolder.empty())
			RG_ERR_CLOSE("Cannot load Recent Policy Versions with no checkpoint or recent-policy folder");
		if (config.skillTracker.enabled && runtimeConfig.checkpointFolder.empty())
			RG_ERR_CLOSE("Cannot load Skill Benchmarks with no checkpoint save folder");
		TrainingStorage storage = {};
		if (!runtimeConfig.recentPolicyVersionsFolder.empty())
			storage.SetRecentPolicyVersionsFolder(runtimeConfig.recentPolicyVersionsFolder);
		if (!runtimeConfig.skillBenchmarkFolder.empty())
			storage.SetSkillBenchmarkFolder(runtimeConfig.skillBenchmarkFolder);

		std::filesystem::path recentPolicyVersionRoot = !runtimeConfig.recentPolicyVersionsFolder.empty()
			? runtimeConfig.recentPolicyVersionsFolder
			: (runtimeConfig.checkpointFolder / TrainingStorage::RECENT_POLICY_VERSIONS_FOLDER_NAME);
		versionMgr = new PolicyVersionManager(
			recentPolicyVersionRoot, config.maxRecentVersions, config.tsPerVersion,
			config.skillTracker, envSet->config, config.savePolicyVersions
		);
		versionMgr->storage = storage;
		versionMgr->versionId = runtimeConfig.segmentId.empty() ? runtimeConfig.versionId : runtimeConfig.segmentId;
		versionMgr->recentPolicyVersionSourceIds = runtimeConfig.recentPolicyVersionSourceIds;
		versionMgr->sourceStageName = runtimeConfig.trainingStageName;
	} else {
		versionMgr = NULL;
	}

	if (config.trainAgainstStaticOpponents) {
		staticOpponentMgr = new PolicyVersionManager(
			runtimeConfig.staticOpponentsRegistryPath.parent_path(), -1, 0,
			SkillTrackerConfig{}, envSet->config, false
		);
		staticOpponentMgr->staticOpponentsRegistryPath = runtimeConfig.staticOpponentsRegistryPath;
		if (!config.staticOpponentsUseOwnBallPrediction)
			staticOpponentMgr->staticOpponentBallPredictionEngine = config.ballPredictionEngine;
	} else {
		staticOpponentMgr = NULL;
	}

	if (versionMgr && config.skillTracker.enabled && RegistryListsSkillReferences(runtimeConfig.skillReferencesRegistryPath)) {
		skillReferenceMgr = new PolicyVersionManager(
			runtimeConfig.skillReferencesRegistryPath.parent_path(), -1, 0,
			SkillTrackerConfig{}, envSet->config, false
		);
		skillReferenceMgr->staticOpponentsRegistryPath = runtimeConfig.skillReferencesRegistryPath;
		skillReferenceMgr->registryListKey = SKILL_REFERENCES_LIST_KEY;
		if (!config.staticOpponentsUseOwnBallPrediction)
			skillReferenceMgr->staticOpponentBallPredictionEngine = config.ballPredictionEngine;
	} else {
		skillReferenceMgr = NULL;
	}

	if (!runtimeConfig.checkpointFolder.empty() || !runtimeConfig.checkpointLoadFolder.empty())
		Load();

	if (runtimeConfig.resetRunIdOnLoad)
		runID.clear();

	if (config.trainAgainstRecentVersions || config.savePolicyVersions) {
		auto models = ppo->GetPolicyModels();
		versionMgr->LoadVersions(models, totalTimesteps);
	}

	if (config.skillTracker.enabled && versionMgr) {
		auto models = ppo->GetPolicyModels();
		versionMgr->LoadBenchmarks(models, totalTimesteps);
		versionMgr->EnsureInitialSkillBenchmark(models, totalTimesteps);
	}

	if (config.trainAgainstStaticOpponents && staticOpponentMgr) {
		auto models = ppo->GetPolicyModels();
		staticOpponentMgr->LoadStaticOpponents(models, runtimeConfig.staticOpponentAgentFactory);
	}

	if (skillReferenceMgr) {
		auto models = ppo->GetPolicyModels();
		skillReferenceMgr->LoadStaticOpponents(models, runtimeConfig.staticOpponentAgentFactory);
		versionMgr->skillReferences = &skillReferenceMgr->versions;
		versionMgr->skillReferencesRegistryPath = runtimeConfig.skillReferencesRegistryPath;
	}

	if (config.sendMetrics) {
		if (!runID.empty())
			RG_LOG("\tRun ID: " << runID);
		metricSender = new MetricSender(config.metricsProjectName, config.metricsGroupName, config.metricsRunName, runID);
	} else {
		metricSender = NULL;
	}

	RG_LOG(RG_DIVIDER);
}

void GGL::Learner::SaveStats(std::filesystem::path path) {
	using namespace nlohmann;

	constexpr const char* ERROR_PREFIX = "Learner::SaveStats(): ";

	std::ofstream fOut(path);
	if (!fOut.good())
		RG_ERR_CLOSE(ERROR_PREFIX << "Can't open file at " << path);

	json j = {};

	j["total_timesteps"] = totalTimesteps;
	j["total_iterations"] = totalIterations;

	if (config.sendMetrics)
		j["run_id"] = metricSender->curRunID;

	if (returnStat)
		j["return_stat"] = returnStat->ToJSON();
	if (obsStat)
		j["obs_stat"] = obsStat->ToJSON();
	if (popArtStats)
		SaveValueHeads(j, *popArtStats);

	if (versionMgr)
		versionMgr->AddRunningStatsToJSON(j);

	if (envSet) {
		std::vector<RLGC::AdaptivePersistEntry> adaptiveEntries = envSet->adaptiveWeightController.ExportState();
		if (!adaptiveEntries.empty()) {
			GGL::JsonArchive ar = GGL::JsonArchive::MakeSaver(j);
			ar.Field("adaptive_reward_weights", adaptiveEntries);
		}
	}

	// NOTE: scheduled_values is diagnostic-only (never read back by LoadStats), so it stays a
	// raw JSON write like the wandb/metrics payloads rather than a JsonArchive round-trip.
	if (config.ppo.entropyScale.IsScheduled()) {
		j["scheduled_values"] = json::array();
		const GGL::ScheduledFloat& scheduled = config.ppo.entropyScale;
		j["scheduled_values"].push_back({
			{ "name", "PPO/Entropy Scale" },
			{ "target", scheduled.target },
			{ "amplitude", scheduled.amplitude },
			{ "period_timesteps", scheduled.period },
			{ "start_timestep", scheduled.startTimestep },
			{ "phase_offset_radians", scheduled.phaseOffsetRadians },
			{ "last_value", scheduled.Get() },
			{ "last_timestep", totalTimesteps }
		});
	}

	std::string jStr = j.dump(4);
	fOut << jStr;
}

std::vector<std::string> GGL::Learner::LoadStats(std::filesystem::path path) {
	// TODO: Repetitive code, merge repeated code into one function called from both SaveStats() and LoadStats()

	using namespace nlohmann;
	constexpr const char* ERROR_PREFIX = "Learner::LoadStats(): ";

	std::ifstream fIn(path);
	if (!fIn.good())
		RG_ERR_CLOSE(ERROR_PREFIX << "Can't open file at " << path);

	json j = json::parse(fIn);
	totalTimesteps = j["total_timesteps"];
	totalIterations = j["total_iterations"];

	if (j.contains("run_id"))
		runID = j["run_id"];

	if (returnStat)
		returnStat->ReadFromJSON(j["return_stat"]);
	if (obsStat)
		obsStat->ReadFromJSON(j["obs_stat"]);

	if (versionMgr)
		versionMgr->LoadRunningStatsFromJSON(j);

	if (envSet && j.contains("adaptive_reward_weights")) {
		GGL::JsonArchive ar = GGL::JsonArchive::MakeLoader(j);
		std::vector<RLGC::AdaptivePersistEntry> entries;
		ar.Field("adaptive_reward_weights", entries);

		// A malformed entry no longer throws (JsonArchive substitutes a default and records an
		// error instead) -- that's not fatal like a missing total_timesteps would be, but it must
		// not be silent, since it means one reward's calibrated weight was just discarded.
		for (const std::string& error : ar.GetErrors())
			RG_LOG(ERROR_PREFIX << "adaptive_reward_weights load error: " << error);

		envSet->adaptiveWeightController.ImportState(entries, envSet->rewards);
	}

	const ValueHeadsLoadResult valueHeads = LoadValueHeads(j, popArtStats.get());
	return valueHeads.layoutChanged ? valueHeads.savedHeadNames : std::vector<std::string>();
}

// Different than RLGym-PPO to show that they are not compatible
constexpr const char* STATS_FILE_NAME = "RUNNING_STATS.json";

void GGL::Learner::Save(bool saveVersionSnapshot) {
	const LearnerRuntimeConfig& runtimeConfig = config.runtime;

	if (runtimeConfig.checkpointFolder.empty())
		RG_ERR_CLOSE("Learner::Save(): Cannot save because config.runtime.checkpointFolder is not set");

	std::filesystem::path saveFolder = runtimeConfig.checkpointFolder / std::to_string(totalTimesteps);
	std::filesystem::create_directories(saveFolder);

	RG_LOG("Saving to folder " << saveFolder << "...");
	SaveStats(saveFolder / STATS_FILE_NAME);
	ppo->SaveTo(saveFolder);

	std::filesystem::path registeredSaveFolder = saveFolder;
	const bool saveDurableSnapshot = saveVersionSnapshot || runtimeConfig.persistCheckpointSnapshotsDuringTraining;
	if (saveDurableSnapshot && !runtimeConfig.checkpointSnapshotFolder.empty()) {
		std::filesystem::path snapshotSaveFolder = runtimeConfig.checkpointSnapshotFolder / std::to_string(totalTimesteps);
		if (runtimeConfig.checkpointSnapshotFolder != runtimeConfig.checkpointFolder) {
			std::filesystem::create_directories(runtimeConfig.checkpointSnapshotFolder);
			std::filesystem::copy(
				saveFolder,
				snapshotSaveFolder,
				std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing);
		}
		registeredSaveFolder = snapshotSaveFolder;
	}

	// Remove old checkpoints
	if (config.checkpointsToKeep != -1) {
		std::set<int64_t> allSavedTimesteps = Utils::FindNumberedDirs(runtimeConfig.checkpointFolder);
		while (allSavedTimesteps.size() > config.checkpointsToKeep) {
			int64_t lowestCheckpointTS = INT64_MAX;
			for (int64_t savedTimesteps : allSavedTimesteps)
				lowestCheckpointTS = RS_MIN(lowestCheckpointTS, savedTimesteps);

			std::filesystem::path removePath = runtimeConfig.checkpointFolder / std::to_string(lowestCheckpointTS);
			try {
				std::filesystem::remove_all(removePath);
			} catch (std::exception& e) {
				RG_ERR_CLOSE("Failed to remove old checkpoint from " << removePath << ", exception: " << e.what());
			}
			allSavedTimesteps.erase(lowestCheckpointTS);
		}
	}

	if (versionMgr)
		versionMgr->SaveVersions(totalTimesteps);
	if (versionMgr)
		versionMgr->SaveBenchmarks();
	if (runtimeConfig.onCheckpointSaved)
		runtimeConfig.onCheckpointSaved(totalTimesteps, registeredSaveFolder);

	RG_LOG(" > Done.");
}

void GGL::Learner::Load() {
	std::filesystem::path checkpointLoadRoot = config.runtime.checkpointLoadFolder.empty()
		? config.runtime.checkpointFolder
		: config.runtime.checkpointLoadFolder;
	if (checkpointLoadRoot.empty())
		RG_ERR_CLOSE("Learner::Load(): Cannot load because config.checkpointLoadFolder is not set");

	RG_LOG("Loading most recent checkpoint in " << checkpointLoadRoot << "...");

	int64_t highest = -1;
	std::set<int64_t> allSavedTimesteps = Utils::FindNumberedDirs(checkpointLoadRoot);
	for (int64_t timesteps : allSavedTimesteps)
		highest = RS_MAX(timesteps, highest);

	if (highest != -1) {
		std::filesystem::path loadFolder = checkpointLoadRoot / std::to_string(highest);
		RG_LOG(" > Loading checkpoint " << loadFolder << "...");
		const std::vector<std::string> remappedSavedHeadNames = LoadStats(loadFolder / STATS_FILE_NAME);
		if (!remappedSavedHeadNames.empty()) {
			const std::vector<std::string>& headNames = envSet->rewardHeadLayout->GetHeadNames();
			RG_LOG(" > Critic heads remapped: " << DescribeCriticHeadRemap(remappedSavedHeadNames, headNames));
			ppo->LoadFromRemappingCriticHeads(loadFolder, remappedSavedHeadNames, headNames);
		} else {
			ppo->LoadFrom(loadFolder);
		}

		if (config.warmStartPolicy && config.warmStartIterations > 0) {
			warmStartIterationsLeft = config.warmStartIterations;
			RG_LOG("[WarmStartPolicy] Activated on checkpoint load (" << config.warmStartIterations << " iterations)");
		}

		RG_LOG(" > Done.");
	} else {
		RG_LOG(" > No checkpoints found, starting new model.")
	}
}

void GGL::Learner::RequestStop() {
	stopRequested.store(true, std::memory_order_relaxed);
}

void GGL::Learner::SendMetrics(const Report& report) {
	if (!metricSender)
		return;

	metricSender->Send(report);

	if (!metricSender->remoteStopRequested)
		return;

	RG_LOG("Run stopped from the metrics website, will save and exit next iteration.");

	if (onRemoteStopRequested)
		onRemoteStopRequested();
	else
		RequestStop();
}

void GGL::Learner::StartTransferLearn(const TransferLearnConfig& tlConfig) {
	const LearnerRuntimeConfig& runtimeConfig = config.runtime;

	RG_LOG("Starting transfer learning...");
	TransferLearning::ValidateConfig(tlConfig);
	TORCH_CHECK(tlConfig.makeOldObsFn && tlConfig.makeOldActFn, "Transfer requires teacher observation and action factories");

	// TODO: Lots of manual obs builder stuff going on which is quite volatile
	//	Although I can't really think another way to do this

	std::vector<std::unique_ptr<ObsBuilder>> oldObsBuilders = {};
	for (int i = 0; i < envSet->arenas.size(); i++) {
		oldObsBuilders.emplace_back(tlConfig.makeOldObsFn());
		TORCH_CHECK(oldObsBuilders.back(), "Teacher observation factory returned null");
	}

	// Reset all obs builders initially
	for (int i = 0; i < envSet->arenas.size(); i++)
		oldObsBuilders[i]->Reset(envSet->state.gameStates[i]);

	std::vector<std::unique_ptr<ActionParser>> oldActionParsers = {};
	for (int i = 0; i < envSet->arenas.size(); i++) {
		oldActionParsers.emplace_back(tlConfig.makeOldActFn());
		TORCH_CHECK(oldActionParsers.back(), "Teacher action factory returned null");
		oldActionParsers.back()->ResetRuntimeActionOverride();
	}

	int oldNumActions = oldActionParsers[0]->GetActionAmount();

	if (oldNumActions != numActions) {
		if (!tlConfig.mapActsFn) {
			RG_ERR_CLOSE(
				"StartTransferLearn: Old and new action parsers have a different number of actions, but tlConfig.mapActsFn is NULL.\n" <<
				"You must implement this function to translate the action indices."
			);
		};
	}

	// Determine old obs size
	int oldObsSize;
	{
		GameState testState = envSet->state.gameStates[0];
		oldObsSize = oldObsBuilders[0]->BuildObs(testState.players[0], testState).size();
		oldObsBuilders[0]->Reset(envSet->state.gameStates[0]);
	}

	ModelSet oldModels = {};
	struct TeacherCleanup {
		ModelSet& models;
		~TeacherCleanup() { models.Free(); }
	} teacherCleanup{ oldModels };

	{
		RG_NO_GRAD;
		const bool makeCritic = false;
		ModelManager::MakeModels(
			makeCritic, oldObsSize, oldNumActions, tlConfig.oldSharedHeadConfig, tlConfig.oldPolicyConfig, {}, ppo->device, oldModels);

		oldModels.Load(tlConfig.oldModelsPath, false);
	}

	try {
		while (true) {
			Report report = {};

			// Collect obs
			std::vector<float> allNewObs = {};
			std::vector<float> allOldObs = {};
			std::vector<uint8_t> allNewActionMasks = {};
			std::vector<uint8_t> allOldActionMasks = {};
			std::vector<int> allActionMaps = {};
			int stepsCollected;
			{
				RG_NO_GRAD;
				for (stepsCollected = 0; stepsCollected < tlConfig.batchSize; stepsCollected += envSet->state.numPlayers) {
					
					auto terminals = envSet->state.terminals; // Backup
					envSet->Reset();
					for (int i = 0; i < envSet->arenas.size(); i++) // Manually reset old obs builders
						if (terminals[i]) {
							oldObsBuilders[i]->Reset(envSet->state.gameStates[i]);
							oldActionParsers[i]->ResetRuntimeActionOverride();
						}

					torch::Tensor tActions, tLogProbs;
					torch::Tensor tStates = DIMLIST2_TO_TENSOR<float>(envSet->state.obs);
					torch::Tensor tActionMasks = DIMLIST2_TO_TENSOR<uint8_t>(envSet->state.actionMasks);

					envSet->StepFirstHalf(true);

					allNewObs += envSet->state.obs.data;
					allNewActionMasks += envSet->state.actionMasks.data;

					// Run all old obs and old action parser on each player
					// TODO: Could be multithreaded
					for (int arenaIdx = 0; arenaIdx < envSet->arenas.size(); arenaIdx++) {
						auto& gs = envSet->state.gameStates[arenaIdx];
						for (auto& player : gs.players) {
							FList obs = oldObsBuilders[arenaIdx]->BuildObs(player, gs);
							std::vector<uint8_t> mask = oldActionParsers[arenaIdx]->GetActionMask(player, gs);
							TORCH_CHECK(obs.size() == oldObsSize && mask.size() == oldNumActions,
								"Teacher observation or action width changed during collection");
							allOldObs += obs;
							allOldActionMasks += mask;

							if (tlConfig.mapActsFn) {
								auto curMap = tlConfig.mapActsFn(player, gs);
								if (curMap.size() != numActions)
									RG_ERR_CLOSE("StartTransferLearn: Your action map must have the same size as the new action parser's actions");
								allActionMaps += curMap;
							}
						}
					}

					std::vector<int> curActions;
					const int64_t inferenceChunk = tlConfig.microBatchSize > 0 ? tlConfig.microBatchSize : tStates.size(0);

					for (int64_t start = 0; start < tStates.size(0); start += inferenceChunk) {
						const int64_t end = std::min(start + inferenceChunk, tStates.size(0));
						ModelManager::InferActionsFromModels(
							ppo->models,
							tStates.slice(0, start, end).to(ppo->device), tActionMasks.slice(0, start, end).to(ppo->device),
							ppo->config.deterministic, ppo->config.policyTemperature, ppo->config.useHalfPrecision,
							&tActions, &tLogProbs, ppo->config.minActionProbability);
						curActions += TENSOR_TO_VEC<int>(tActions);
					}

					envSet->Sync();
					envSet->StepSecondHalf(curActions, false);

					if (stepCallback)
						stepCallback(this, envSet->state.gameStates, report);
				}
			}

			// Keep the full collection on CPU; TransferLearn moves only one microbatch to the device.
			torch::Tensor tNewObs = torch::tensor(allNewObs).reshape({ -1, obsSize });
			torch::Tensor tOldObs = torch::tensor(allOldObs).reshape({ -1, oldObsSize });
			torch::Tensor tNewActionMasks = torch::tensor(allNewActionMasks).reshape({ -1, numActions });
			torch::Tensor tOldActionMasks = torch::tensor(allOldActionMasks).reshape({ -1, oldNumActions });

			torch::Tensor tActionMaps = {};
			if (!allActionMaps.empty())
				tActionMaps = torch::tensor(allActionMaps).reshape({ -1, numActions });

			// Transfer learn
			ppo->TransferLearn(oldModels, tNewObs, tOldObs, tNewActionMasks, tOldActionMasks, tActionMaps, report, tlConfig);

			uint64_t prevTimesteps = totalTimesteps;
			totalTimesteps += stepsCollected;
			report["Total Timesteps"] = totalTimesteps;
			report["Collected Timesteps"] = stepsCollected;
			totalIterations++;
			report["Total Iterations"] = totalIterations;

			if (versionMgr)
				versionMgr->OnIteration(ppo, report, totalTimesteps, prevTimesteps);

			// RocketForge: target-driven exit. Use the same stop path as RequestStop.
			if (runtimeConfig.stopAtTotalTimesteps > 0 && totalTimesteps >= runtimeConfig.stopAtTotalTimesteps) {
				stopRequested.store(true, std::memory_order_relaxed);
			}

			if (stopRequested.load(std::memory_order_relaxed)) {
				if (!runtimeConfig.checkpointFolder.empty())
					Save(true);
				return;
			}

			if (!runtimeConfig.checkpointFolder.empty()) {
				if (totalTimesteps / config.tsPerSave > prevTimesteps / config.tsPerSave) {
					// Auto-save
					Save();
				}
			}

			report.Finish();

			SendMetrics(report);

			report.Display(
				{
					"Transfer Learn Accuracy",
					"Transfer Learn Loss",
					"",
					"LossAndEntropy/Policy Entropy",
					"LossAndEntropy/Old Policy Entropy",
					"LossAndEntropy/Policy Update Magnitude",
					"",
					"Collected Timesteps",
					"Total Timesteps",
					"Total Iterations",
					"SkillTest/NextRunTimestep",
				}
			);
		}

	} catch (std::exception& e) {
		RG_ERR_CLOSE("Exception thrown during transfer learn loop: " << e.what());
	}
}

void GGL::Learner::Start() {
	const LearnerRuntimeConfig& runtimeConfig = config.runtime;

	RG_LOG("Learner::Start():");
	RG_LOG("\tObs size: " << obsSize);
	RG_LOG("\tAction amount: " << numActions);

	try {
		ExperienceBuffer experience = ExperienceBuffer(config.randomSeed, torch::kCPU);

		int numPlayers = envSet->state.numPlayers;

		struct Trajectory {
			FList states, nextStates, rewards, logProbs;

			// Filled only when the env set has a reward head layout: numHeads values and one arena index per step.
			FList headRewards;
			std::vector<int32_t> arenaIndices;
			std::vector<uint8_t> actionMasks;
			std::vector<int8_t> terminals;
			std::vector<int8_t> playerCounts;
			std::vector<int64_t> actions;

			void Clear() {
				// Keep capacity so long runs don't repeatedly allocate trajectory buffers every episode.
				states.clear();
				nextStates.clear();
				rewards.clear();
				logProbs.clear();
				headRewards.clear();
				arenaIndices.clear();
				actionMasks.clear();
				terminals.clear();
				playerCounts.clear();
				actions.clear();
			}

			void Append(const Trajectory& other) {
				states += other.states;
				nextStates += other.nextStates;
				rewards += other.rewards;
				logProbs += other.logProbs;
				headRewards += other.headRewards;
				arenaIndices += other.arenaIndices;
				actionMasks += other.actionMasks;
				terminals += other.terminals;
				playerCounts += other.playerCounts;
				actions += other.actions;
			}

			void Reserve(size_t timesteps, int obsSize, int numActions, int numHeads) {
				states.reserve(timesteps * obsSize);
				rewards.reserve(timesteps);
				logProbs.reserve(timesteps);
				headRewards.reserve(timesteps * numHeads);
				arenaIndices.reserve(numHeads > 0 ? timesteps : 0);
				actionMasks.reserve(timesteps * numActions);
				terminals.reserve(timesteps);
				playerCounts.reserve(timesteps);
				actions.reserve(timesteps);
			}

			size_t Length() const {
				return actions.size();
			}
		};

		auto trajectories = std::vector<Trajectory>(numPlayers, Trajectory{});

		// Collection staging lives for the entire learner run so exact CUDA graphs can
		// safely capture the device input allocations instead of copying into another
		// graph-owned buffer on every environment step.
		const bool usePinnedCollectionStaging = ppo->device.is_cuda();
		torch::TensorOptions collectionStateOptions = torch::TensorOptions()
			.dtype(torch::kFloat32).pinned_memory(usePinnedCollectionStaging);
		torch::TensorOptions collectionMaskOptions = torch::TensorOptions()
			.dtype(torch::kUInt8).pinned_memory(usePinnedCollectionStaging);
		torch::Tensor collectionHostStates = torch::empty(
			{ numPlayers, obsSize }, collectionStateOptions);
		torch::Tensor collectionHostActionMasks = torch::empty(
			{ numPlayers, numActions }, collectionMaskOptions);
		torch::Tensor collectionDeviceStates;
		torch::Tensor collectionDeviceActionMasks;
		std::map<int64_t, PersistentPolicyInputBucket> collectionLearnerInputBuckets;
		if (ppo->device.is_cuda()) {
			collectionDeviceStates = torch::empty(
				{ numPlayers, obsSize }, collectionStateOptions.device(ppo->device).pinned_memory(false));
			collectionDeviceActionMasks = torch::empty(
				{ numPlayers, numActions }, collectionMaskOptions.device(ppo->device).pinned_memory(false));
		}

		enum class ArenaOpponentKind {
			NONE,
			RECENT,
			STATIC
		};

		struct ArenaOpponentAssignment {
			ArenaOpponentKind kind = ArenaOpponentKind::NONE;
			Team opponentTeam = Team::BLUE;
			PolicyVersionKey recentVersionKey = {};
			int staticVersionIndex = -1;
			int staticCadenceRatio = 1;
			int staticCadenceStep = 0;
			int staticActionApplyStep = 0;
			int staticActionApplyTick = 0;
			bool staticRequiresEarlyAction = false;
			std::unique_ptr<ObsBuilder> obsBuilder;
			std::unique_ptr<ActionParser> actionParser;
		};

		std::vector<ArenaOpponentAssignment> arenaOpponentAssignments(envSet->arenas.size());
		std::vector<int> staticAppliedActionIndices(numPlayers, 0);
		std::vector<int> staticPendingActionIndices(numPlayers, 0);
		const uint64_t opponentSampleWindow = std::max<uint64_t>(
			(uint64_t)std::max<int64_t>(config.ppo.tsPerItr, 1) * 32,
			1);
		OpponentSampleTracker recentSampleTracker(
			config.trainAgainstRecentVersions ? config.trainAgainstRecentPercent : 0.f,
			opponentSampleWindow);
		OpponentSampleTracker staticSampleTracker(
			config.trainAgainstStaticOpponents ? config.trainAgainstStaticPercent : 0.f,
			opponentSampleWindow);
		RecentOpponentStack recentOpponentStack(
			std::max(config.recentOpponentCohortSize, 1),
			std::max<uint64_t>(config.recentOpponentRotationTimesteps, 1),
			static_cast<uint64_t>(config.randomSeed) ^ 0xA7B4'39C2'D15E'608FULL);
		const int recentEnsembleSlotCount = std::max(config.recentOpponentCohortSize + 1, 2);
		std::unique_ptr<RecentPolicyEnsemble> recentPolicyEnsemble =
			config.trainAgainstRecentVersions
			? std::make_unique<RecentPolicyEnsemble>(recentEnsembleSlotCount)
			: nullptr;
		std::vector<std::optional<PolicyVersionKey>> recentEnsembleSlotKeys(recentEnsembleSlotCount);
		std::unique_ptr<StaticOpponentCohort> staticOpponentCohort =
			config.trainAgainstStaticOpponents
			? std::make_unique<StaticOpponentCohort>(
				std::max(config.staticOpponentCohortSize, 1),
				std::max<uint64_t>(config.staticOpponentRotationTimesteps, 1),
				static_cast<uint64_t>(config.randomSeed) ^ 0x5C91'E2A8'74D3'0B6FULL)
			: nullptr;
		std::vector<StaticOpponentCandidate> availableStaticOpponents;
		if (config.trainAgainstStaticOpponents && staticOpponentMgr) {
			availableStaticOpponents.reserve(staticOpponentMgr->versions.size());
			bool selectedStaticOpponentFound = config.staticOpponentSelectionKey.empty();
			for (int versionIdx = 0; versionIdx < (int)staticOpponentMgr->versions.size(); versionIdx++) {
				const PolicyVersion& version = staticOpponentMgr->versions[versionIdx];
				if (!config.staticOpponentSelectionKey.empty()
					&& version.staticSelectionKey != config.staticOpponentSelectionKey)
					continue;
				selectedStaticOpponentFound = true;
				availableStaticOpponents.push_back({
					versionIdx,
					version.inferenceAgent ? version.inferenceAgent->maxTeamSize : 3
				});
			}
			if (!selectedStaticOpponentFound)
				RG_ERR_CLOSE("Selected debug static opponent is no longer present in the Build registry");
		}
		bool recentPoolWasAvailable = false;
		bool opponentAssignmentsInitialized = false;
		int maxEpisodeLength = (int)(config.ppo.maxEpisodeDuration * (120.f / config.tickSkip));
		double ppgCycleSeconds = 0;
		uint64_t ppgCycleTimesteps = 0;
		double ppgEffectiveStepsPerSecond = 0;

		while (true) {
			Report report = {};
			const CudaGraphStats modelGraphStatsBefore = ModelManager::GetCudaGraphStats();
			const RecentPolicyEnsembleStats recentEnsembleStatsBefore = recentPolicyEnsemble
				? recentPolicyEnsemble->GetStats()
				: RecentPolicyEnsembleStats{};

			bool isFirstIteration = (totalTimesteps == 0);
			ApplyScheduledConfigValues(*this, report);

			std::vector<PolicyVersionKey> availableRecentVersions = {};
			if (config.trainAgainstRecentVersions && versionMgr && !versionMgr->versions.empty()) {
				for (const PolicyVersion& version : versionMgr->versions)
					availableRecentVersions.push_back({ version.sourceVersionId, version.timesteps });
			}

			std::map<PolicyVersionKey, int> activeRecentArenaCounts;
			for (const ArenaOpponentAssignment& assignment : arenaOpponentAssignments) {
				if (assignment.kind == ArenaOpponentKind::RECENT)
					activeRecentArenaCounts[assignment.recentVersionKey]++;
			}
			recentOpponentStack.Update(
				availableRecentVersions,
				activeRecentArenaCounts,
				totalTimesteps);

			if (staticOpponentCohort) {
				std::map<int, int> activeStaticArenaCounts;
				for (const ArenaOpponentAssignment& assignment : arenaOpponentAssignments) {
					if (assignment.kind == ArenaOpponentKind::STATIC)
						activeStaticArenaCounts[assignment.staticVersionIndex]++;
				}
				staticOpponentCohort->Update(
					availableStaticOpponents,
					activeStaticArenaCounts,
					totalTimesteps);
			}

			const bool recentPoolAvailable = !recentOpponentStack.GetSelectable().empty();
			if (recentPoolAvailable && !recentPoolWasAvailable)
				recentSampleTracker.Reset();
			recentPoolWasAvailable = recentPoolAvailable;

			int recentEpisodesStarted = 0;
			int recentEpisodesCompleted = 0;
			int staticEpisodesStarted = 0;
			int staticEpisodesCompleted = 0;
			int normalEpisodesStarted = 0;
			int unexpectedOwnershipChanges = 0;
			int staticRuntimeReuses = 0;
			int staticRuntimeRebuilds = 0;
			const size_t staticVersionCount = staticOpponentMgr
				? staticOpponentMgr->versions.size()
				: 0;
			std::vector<uint64_t> staticEpisodesStartedByVersion(staticVersionCount, 0);
			std::vector<uint64_t> staticLearnerSamplesByVersion(staticVersionCount, 0);
			std::vector<uint64_t> staticInferenceCallsByVersion(staticVersionCount, 0);
			std::vector<uint64_t> staticInferenceRowsByVersion(staticVersionCount, 0);
			std::vector<uint64_t> staticCadenceSkippedRowsByVersion(staticVersionCount, 0);
			uint64_t staticInferenceCalls = 0;
			uint64_t staticInferenceRows = 0;
			uint64_t staticInferenceRequestedRows = 0;
			uint64_t staticCadenceSkippedRows = 0;
			double staticObservationBuildTime = 0;
			double earlyStaticInferenceTime = 0;

			auto chooseOpponentTeam = [&](int arenaIdx) {
				const GameState& state = envSet->state.gameStates[arenaIdx];
				const int focusCarId = envSet->stateSetters[arenaIdx]->trainingFocusCarId;
				if (focusCarId >= 0) {
					for (const Player& player : state.players) {
						if ((int)player.carId == focusCarId)
							return player.team == Team::BLUE ? Team::ORANGE : Team::BLUE;
					}
				}
				return Team(RocketSim::Math::RandInt(0, 2));
			};

			auto assignOpponentForEpisode = [&](int arenaIdx) {
				ArenaOpponentAssignment& assignment = arenaOpponentAssignments[arenaIdx];
				const ArenaOpponentKind previousKind = assignment.kind;
				const int previousStaticVersionIndex = assignment.staticVersionIndex;
				std::unique_ptr<ObsBuilder> previousObsBuilder = std::move(assignment.obsBuilder);
				std::unique_ptr<ActionParser> previousActionParser = std::move(assignment.actionParser);
				assignment = ArenaOpponentAssignment{};

				const GameState& state = envSet->state.gameStates[arenaIdx];
				const Team opponentTeam = chooseOpponentTeam(arenaIdx);
				int opponentPlayers = 0;
				for (const Player& player : state.players)
					opponentPlayers += player.team == opponentTeam;
				const bool hasFocusCar = envSet->stateSetters[arenaIdx]->trainingFocusCarId >= 0;
				const int normalLearnerPlayers = hasFocusCar ? 1 : (int)state.players.size();
				const int opponentLearnerPlayers = hasFocusCar
					? 1
					: (int)state.players.size() - opponentPlayers;

				const bool hasCompatibleStaticVersion = config.trainAgainstStaticOpponents
					&& staticOpponentMgr
					&& staticOpponentCohort
					&& staticOpponentCohort->HasCompatibleVersion(opponentPlayers);

				// Convert the two experience-share targets together, then split the resulting
				// opponent-episode probability. This keeps assignments mutually exclusive and
				// accounts for self-play episodes contributing experience from both teams.
				float staticSampleTarget = hasCompatibleStaticVersion
					? staticSampleTracker.GetCorrectedTargetFraction()
					: 0.f;
				float recentSampleTarget = recentPoolAvailable
					? recentSampleTracker.GetCorrectedTargetFraction()
					: 0.f;
				float combinedSampleTarget = staticSampleTarget + recentSampleTarget;
				if (combinedSampleTarget > 1.f) {
					const float scale = 1.f / combinedSampleTarget;
					staticSampleTarget *= scale;
					recentSampleTarget *= scale;
					combinedSampleTarget = 1.f;
				}

				const float opponentEpisodeChance = OpponentSampleTracker::SampleFractionToEpisodeProbability(
					combinedSampleTarget,
					normalLearnerPlayers,
					opponentLearnerPlayers);
				const float staticEpisodeChance = combinedSampleTarget > 0
					? opponentEpisodeChance * staticSampleTarget / combinedSampleTarget
					: 0.f;
				const float recentEpisodeChance = combinedSampleTarget > 0
					? opponentEpisodeChance * recentSampleTarget / combinedSampleTarget
					: 0.f;
				const float opponentRoll = RocketSim::Math::RandFloat();

				if (staticEpisodeChance > 0 && opponentRoll < staticEpisodeChance) {
					assignment.kind = ArenaOpponentKind::STATIC;
					assignment.opponentTeam = opponentTeam;
					std::optional<int> selectedVersion;
					if (!config.resampleStaticOpponentEveryEpisode
						&& previousKind == ArenaOpponentKind::STATIC
						&& staticOpponentCohort->CanContinueEpisode(
							previousStaticVersionIndex,
							opponentPlayers)) {
						selectedVersion = previousStaticVersionIndex;
					} else {
						selectedVersion = staticOpponentCohort->SelectForEpisode(
							opponentPlayers,
							config.resampleStaticOpponentEveryEpisode
								&& previousKind == ArenaOpponentKind::STATIC
								? std::optional<int>(previousStaticVersionIndex)
								: std::nullopt);
					}
					if (!selectedVersion.has_value())
						RG_ERR_CLOSE("Static opponent cohort became incompatible during episode assignment");
					assignment.staticVersionIndex = selectedVersion.value();
					PolicyVersion& version = staticOpponentMgr->versions[assignment.staticVersionIndex];
					if (version.inferenceAgent) {
						const bool canReuseRuntime = previousKind == ArenaOpponentKind::STATIC
							&& previousStaticVersionIndex == assignment.staticVersionIndex
							&& previousObsBuilder
							&& previousActionParser;
						if (canReuseRuntime) {
							assignment.obsBuilder = std::move(previousObsBuilder);
							assignment.actionParser = std::move(previousActionParser);
							staticRuntimeReuses++;
						} else {
							assignment.obsBuilder.reset(version.inferenceAgent->MakeConfiguredObsBuilder());
							assignment.actionParser.reset(version.inferenceAgent->MakeActionParser());
							staticRuntimeRebuilds++;
						}
						assignment.obsBuilder->Reset(state);
						assignment.actionParser->ResetRuntimeActionOverride();
						assignment.staticCadenceRatio = version.inferenceCadenceRatio;
						assignment.staticActionApplyStep = version.inferenceActionApplyStep;
						assignment.staticActionApplyTick = version.inferenceActionApplyTick;
						assignment.staticRequiresEarlyAction = version.inferenceRequiresEarlyAction;
					}

					ActionParser* staticParser = assignment.actionParser
						? assignment.actionParser.get()
						: envSet->actionParsers[arenaIdx];
					const int playerStartIdx = envSet->state.arenaPlayerStartIdx[arenaIdx];
					for (int playerIdx = 0; playerIdx < (int)state.players.size(); playerIdx++) {
						if (state.players[playerIdx].team != opponentTeam)
							continue;
						const int globalPlayerIdx = playerStartIdx + playerIdx;
						const int neutralAction = FindNeutralActionIndex(
							*staticParser,
							state.players[playerIdx],
							state);
						staticAppliedActionIndices[globalPlayerIdx] = neutralAction;
						staticPendingActionIndices[globalPlayerIdx] = neutralAction;
					}
					staticEpisodesStarted++;
					staticEpisodesStartedByVersion[assignment.staticVersionIndex]++;
					return;
				}

				if (recentEpisodeChance > 0
					&& opponentRoll < staticEpisodeChance + recentEpisodeChance) {
					assignment.kind = ArenaOpponentKind::RECENT;
					assignment.opponentTeam = opponentTeam;
					std::optional<PolicyVersionKey> selectedVersion = recentOpponentStack.SelectForEpisode();
					if (!selectedVersion.has_value())
						RG_ERR_CLOSE("Recent opponent stack became empty during episode assignment");
					assignment.recentVersionKey = selectedVersion.value();
					recentEpisodesStarted++;
					return;
				}

				normalEpisodesStarted++;
			};

			if (!opponentAssignmentsInitialized) {
				for (int arenaIdx = 0; arenaIdx < (int)arenaOpponentAssignments.size(); arenaIdx++)
					assignOpponentForEpisode(arenaIdx);
				opponentAssignmentsInitialized = true;
			}

			struct OpponentInferenceGroup {
				PolicyVersion* version = nullptr;
				int staticVersionIndex = -1;
				std::vector<int> playerIndices;
				torch::Tensor devicePlayerIndices;
			};

			std::vector<int> learnerPlayerIndices;
			std::vector<int> learnerLogProbPositions(numPlayers, -1);
			std::vector<int> playerArenaIndices(numPlayers, -1);
			std::vector<uint8_t> playerUsesRecentOpponent(numPlayers, 0);
			std::vector<uint8_t> playerUsesStaticOpponent(numPlayers, 0);
			torch::Tensor deviceLearnerPlayerIndices;
			std::vector<OpponentInferenceGroup> opponentGroups;
			std::vector<std::vector<int>> recentPlayerIndicesBySlot(recentEnsembleSlotCount);
			torch::Tensor deviceRecentPaddedPlayerIndices;
			torch::Tensor deviceRecentValidFlatPositions;
			torch::Tensor deviceRecentPlayerIndices;
			int recentActiveSlotCount = 0;
			int recentRowsPerSlot = 0;
			int recentOpponentPlayerCount = 0;
			std::vector<ActionParser*> actionParsersByPlayer(numPlayers, nullptr);
			bool hasIndependentOpponentRuntime = false;
			bool hasEarlyStaticActionRuntime = false;

			auto rebuildInferenceLayout = [&]() {
				learnerPlayerIndices.clear();
				std::fill(learnerLogProbPositions.begin(), learnerLogProbPositions.end(), -1);
				std::fill(playerUsesRecentOpponent.begin(), playerUsesRecentOpponent.end(), 0);
				std::fill(playerUsesStaticOpponent.begin(), playerUsesStaticOpponent.end(), 0);
				opponentGroups.clear();
				for (auto& indices : recentPlayerIndicesBySlot)
					indices.clear();
				recentActiveSlotCount = 0;
				recentRowsPerSlot = 0;
				recentOpponentPlayerCount = 0;
				hasIndependentOpponentRuntime = false;
				hasEarlyStaticActionRuntime = false;
				std::map<PolicyVersion*, int> groupByVersion;
				std::map<PolicyVersionKey, PolicyVersion*> requiredRecentVersions;
				for (const ArenaOpponentAssignment& assignment : arenaOpponentAssignments) {
					if (assignment.kind != ArenaOpponentKind::RECENT)
						continue;
					PolicyVersion* version = versionMgr ? versionMgr->FindVersion(assignment.recentVersionKey) : nullptr;
					if (!version)
						RG_ERR_CLOSE("An active arena's recent opponent version was pruned");
					requiredRecentVersions[assignment.recentVersionKey] = version;
				}

				// Compact active versions into the first slots. The backing parameter tensors
				// stay fixed at selectable+draining capacity, so graph weight addresses never
				// move even when the active graph shape switches between three and four slots.
				if ((int)requiredRecentVersions.size() > recentEnsembleSlotCount)
					RG_ERR_CLOSE("Recent opponent stack exceeded its fixed inference slot capacity");
				std::map<PolicyVersionKey, int> recentSlotByVersion;
				int nextRecentSlot = 0;
				for (const auto& pair : requiredRecentVersions) {
					if (!recentEnsembleSlotKeys[nextRecentSlot].has_value()
						|| recentEnsembleSlotKeys[nextRecentSlot].value() != pair.first) {
						recentPolicyEnsemble->SetSlot(nextRecentSlot, pair.second->models);
						recentEnsembleSlotKeys[nextRecentSlot] = pair.first;
					}
					recentSlotByVersion[pair.first] = nextRecentSlot++;
				}
				recentActiveSlotCount = nextRecentSlot;
				for (int slot = recentActiveSlotCount; slot < recentEnsembleSlotCount; slot++) {
					if (recentEnsembleSlotKeys[slot].has_value()) {
						recentPolicyEnsemble->ClearSlot(slot);
						recentEnsembleSlotKeys[slot].reset();
					}
				}

				for (int arenaIdx = 0; arenaIdx < (int)envSet->state.gameStates.size(); arenaIdx++) {
					ArenaOpponentAssignment& assignment = arenaOpponentAssignments[arenaIdx];
					PolicyVersion* version = nullptr;
					int recentSlot = -1;
					if (assignment.kind == ArenaOpponentKind::RECENT) {
						version = versionMgr ? versionMgr->FindVersion(assignment.recentVersionKey) : nullptr;
						if (!version)
							RG_ERR_CLOSE("An active arena's recent opponent version was pruned");
						recentSlot = recentSlotByVersion.at(assignment.recentVersionKey);
					} else if (assignment.kind == ArenaOpponentKind::STATIC) {
						if (!staticOpponentMgr || assignment.staticVersionIndex < 0
							|| assignment.staticVersionIndex >= (int)staticOpponentMgr->versions.size())
							RG_ERR_CLOSE("An active arena has an invalid static opponent version");
						version = &staticOpponentMgr->versions[assignment.staticVersionIndex];
					}

					int groupIdx = -1;
					if (version && assignment.kind == ArenaOpponentKind::STATIC) {
						auto existing = groupByVersion.find(version);
						if (existing == groupByVersion.end()) {
							groupIdx = (int)opponentGroups.size();
							groupByVersion[version] = groupIdx;
							opponentGroups.push_back({ version, assignment.staticVersionIndex });
						} else {
							groupIdx = existing->second;
						}
					}

					const int playerStartIdx = envSet->state.arenaPlayerStartIdx[arenaIdx];
					const GameState& state = envSet->state.gameStates[arenaIdx];
					for (int playerIdx = 0; playerIdx < (int)state.players.size(); playerIdx++) {
						const int globalPlayerIdx = playerStartIdx + playerIdx;
						playerArenaIndices[globalPlayerIdx] = arenaIdx;
						const bool usesOpponent = version && state.players[playerIdx].team == assignment.opponentTeam;
						if (usesOpponent) {
							if (assignment.kind == ArenaOpponentKind::RECENT)
								recentPlayerIndicesBySlot[recentSlot].push_back(globalPlayerIdx);
							else
								opponentGroups[groupIdx].playerIndices.push_back(globalPlayerIdx);
						} else {
							learnerLogProbPositions[globalPlayerIdx] = (int)learnerPlayerIndices.size();
							learnerPlayerIndices.push_back(globalPlayerIdx);
							playerUsesRecentOpponent[globalPlayerIdx] =
								assignment.kind == ArenaOpponentKind::RECENT;
							playerUsesStaticOpponent[globalPlayerIdx] =
								assignment.kind == ArenaOpponentKind::STATIC;
						}

						actionParsersByPlayer[globalPlayerIdx] = usesOpponent && assignment.actionParser
							? assignment.actionParser.get()
							: envSet->actionParsers[arenaIdx];
					}
					hasIndependentOpponentRuntime = hasIndependentOpponentRuntime
						|| (version && version->inferenceAgent != nullptr);
					hasEarlyStaticActionRuntime = hasEarlyStaticActionRuntime
						|| (version && version->inferenceRequiresEarlyAction);
				}

				deviceLearnerPlayerIndices = torch::tensor(learnerPlayerIndices).to(ppo->device);
				for (OpponentInferenceGroup& group : opponentGroups)
					group.devicePlayerIndices = torch::tensor(group.playerIndices).to(ppo->device);

				int maxRecentRows = 0;
				for (int slot = 0; slot < recentActiveSlotCount; slot++) {
					maxRecentRows = std::max(maxRecentRows, (int)recentPlayerIndicesBySlot[slot].size());
					recentOpponentPlayerCount += (int)recentPlayerIndicesBySlot[slot].size();
				}
				if (maxRecentRows > 0) {
					// All active slots share one rows-per-slot graph bucket. Padding player index
					// zero is harmless because padded actions are removed before the final scatter.
					recentRowsPerSlot = std::max(
						RECENT_ROWS_PER_SLOT_QUANTUM,
						((maxRecentRows + RECENT_ROWS_PER_SLOT_QUANTUM - 1)
							/ RECENT_ROWS_PER_SLOT_QUANTUM) * RECENT_ROWS_PER_SLOT_QUANTUM);

					std::vector<int> paddedPlayerIndices(recentActiveSlotCount * recentRowsPerSlot, 0);
					std::vector<int> validFlatPositions;
					std::vector<int> recentPlayerIndices;
					validFlatPositions.reserve(recentOpponentPlayerCount);
					recentPlayerIndices.reserve(recentOpponentPlayerCount);
					for (int slot = 0; slot < recentActiveSlotCount; slot++) {
						for (int row = 0; row < (int)recentPlayerIndicesBySlot[slot].size(); row++) {
							const int flatPosition = slot * recentRowsPerSlot + row;
							paddedPlayerIndices[flatPosition] = recentPlayerIndicesBySlot[slot][row];
							validFlatPositions.push_back(flatPosition);
							recentPlayerIndices.push_back(recentPlayerIndicesBySlot[slot][row]);
						}
					}
					deviceRecentPaddedPlayerIndices = torch::tensor(paddedPlayerIndices).to(ppo->device);
					deviceRecentValidFlatPositions = torch::tensor(validFlatPositions).to(ppo->device);
					deviceRecentPlayerIndices = torch::tensor(recentPlayerIndices).to(ppo->device);
				} else {
					deviceRecentPaddedPlayerIndices = torch::Tensor();
					deviceRecentValidFlatPositions = torch::Tensor();
					deviceRecentPlayerIndices = torch::Tensor();
				}
			};

			rebuildInferenceLayout();

			struct StaticOpponentWinLoss {
				const GGL::PolicyVersion* version = NULL;
				int wins = 0;
				int losses = 0;
			};
			std::map<std::string, StaticOpponentWinLoss> staticOpponentWinLoss = {};
			int totalStaticDecidedGames = 0;

			int stepsCollected = 0;
			int recentStepsCollected = 0;
			int staticStepsCollected = 0;
			uint64_t learnerInferenceActualRows = 0;
			uint64_t learnerInferenceEvaluatedRows = 0;
			uint64_t recentInferenceActualRows = 0;
			uint64_t recentInferencePaddedRows = 0;
			auto recordStaticInference = [&](const OpponentInferenceGroup& group, uint64_t rows) {
				staticInferenceCalls++;
				staticInferenceRows += rows;
				staticInferenceCallsByVersion[group.staticVersionIndex]++;
				staticInferenceRowsByVersion[group.staticVersionIndex] += rows;
				const bool graphEligible = ppo->config.useCudaGraphs
					&& ppo->device.is_cuda()
					&& !ppo->config.useHalfPrecision;
				staticInferenceRequestedRows += graphEligible
					? ModelManager::GetCudaGraphBatchCapacity(
						rows,
						CudaGraphBatchPolicy::PowerOfTwo)
					: rows;
			};
			{ // Generate experience

				const bool nonBlockingTransfer = true;

				// Only contains complete episodes
				Trajectory combinedTraj = {};
				const size_t targetTimesteps = (size_t)config.ppo.tsPerItr;
				const size_t reservedTimesteps = targetTimesteps + (size_t)numPlayers;
				const int numValueHeads = envSet->rewardHeadLayout ? envSet->rewardHeadLayout->GetNumHeads() : 0;
				combinedTraj.Reserve(reservedTimesteps, obsSize, numActions, numValueHeads);

				Timer collectionTimer = {};
				{ // Collect timesteps
					RG_NO_GRAD;

					double inferTime = 0;
					double inferGpuSampleTime = 0;
					uint64_t inferenceCalls = 0;
					uint64_t inferGpuTimingSamples = 0;
					float envStepTime = 0;
					torch::Tensor& tStates = collectionHostStates;
					torch::Tensor& tActionMasks = collectionHostActionMasks;
#ifdef RG_CUDA_SUPPORT
					at::cuda::CUDAEvent inferGpuStart(cudaEventDefault);
					at::cuda::CUDAEvent inferGpuEnd(cudaEventDefault);
#endif
					RG_ASSERT(envSet->state.obs.numel         == (size_t)numPlayers * obsSize);
					RG_ASSERT(envSet->state.actionMasks.numel == (size_t)numPlayers * numActions);

					std::vector<uint8_t> curTerminals(numPlayers, 0);
					std::vector<uint8_t> finishedPlayers(numPlayers, 0);
					std::vector<uint8_t> terminalsBeforeReset(envSet->state.terminals.size(), 0);
					const bool asyncFirstHalf = true;
					const bool asyncSecondHalf = false;
					for (int step = 0; combinedTraj.Length() < targetTimesteps; step++) {
						Timer resetTimer = {};
						terminalsBeforeReset = envSet->state.terminals;
						envSet->Reset();

						bool assignmentChanged = false;
						for (int arenaIdx = 0; arenaIdx < (int)terminalsBeforeReset.size(); arenaIdx++) {
							if (!terminalsBeforeReset[arenaIdx])
								continue;

							ArenaOpponentAssignment& oldAssignment = arenaOpponentAssignments[arenaIdx];
							if (oldAssignment.kind == ArenaOpponentKind::RECENT)
								recentEpisodesCompleted++;
							else if (oldAssignment.kind == ArenaOpponentKind::STATIC)
								staticEpisodesCompleted++;

							const int playerStartIdx = envSet->state.arenaPlayerStartIdx[arenaIdx];
							const int playerCount = (int)envSet->state.gameStates[arenaIdx].players.size();
							for (int playerIdx = 0; playerIdx < playerCount; playerIdx++) {
								Trajectory& trajectory = trajectories[playerStartIdx + playerIdx];
								if (trajectory.Length() == 0)
									continue;

								// This should only be reachable after external state manipulation. Never
								// connect experience across an interval controlled by another policy.
								unexpectedOwnershipChanges++;
								trajectory.Clear();
							}

							assignOpponentForEpisode(arenaIdx);
							assignmentChanged = true;
						}
						if (assignmentChanged)
							rebuildInferenceLayout();

						// Scenario setters may designate one focus car. Other cars still run the
						// policy and influence the simulation, but do not dilute this PPO batch.
						std::vector<int> experiencePlayerIndices;
						experiencePlayerIndices.reserve(learnerPlayerIndices.size());
						int globalPlayerIdx = 0;
						for (int arenaIdx = 0; arenaIdx < (int)envSet->state.gameStates.size(); ++arenaIdx) {
							const int focusCarId = envSet->stateSetters[arenaIdx]->trainingFocusCarId;
							for (const Player& player : envSet->state.gameStates[arenaIdx].players) {
								if (learnerLogProbPositions[globalPlayerIdx] >= 0 &&
									(focusCarId < 0 || (int)player.carId == focusCarId))
									experiencePlayerIndices.push_back(globalPlayerIdx);
								globalPlayerIdx++;
							}
						}
						stepsCollected += (int)experiencePlayerIndices.size();
						int recentStepSamples = 0;
						int staticStepSamples = 0;
						for (int playerIdx : experiencePlayerIndices) {
							if (playerUsesRecentOpponent[playerIdx]) {
								recentStepSamples++;
							} else if (playerUsesStaticOpponent[playerIdx]) {
								staticStepSamples++;
								const ArenaOpponentAssignment& assignment =
									arenaOpponentAssignments[playerArenaIndices[playerIdx]];
								staticLearnerSamplesByVersion[assignment.staticVersionIndex]++;
							}
						}
						recentStepsCollected += recentStepSamples;
						staticStepsCollected += staticStepSamples;
						recentSampleTracker.AddSamples(recentStepSamples, experiencePlayerIndices.size());
						staticSampleTracker.AddSamples(staticStepSamples, experiencePlayerIndices.size());
						envStepTime += resetTimer.Elapsed();

						if (ContainsNonFiniteFloat(envSet->state.obs.data))
							RG_ERR_CLOSE("Obs builder produced a NaN/inf value");

						if (obsStat) {
							// TODO: This samples from recent-version matchups too
							int numSamples = RS_MAX(envSet->state.numPlayers, config.maxObsSamples);
							for (int i = 0; i < numSamples; i++) {
								int idx = Math::RandInt(0, envSet->state.numPlayers);
								obsStat->IncrementRow(&envSet->state.obs.At(idx, 0));
							}

							std::vector<double> mean = obsStat->GetMean();
							std::vector<double> std = obsStat->GetSTD();
							for (double& f : mean)
								f = RS_CLAMP(f, -config.maxObsMeanRange, config.maxObsMeanRange);
							for (double& f : std)
								f = RS_MAX(f, config.minObsSTD);
							for (int i = 0; i < envSet->state.numPlayers; i++) {
								for (int j = 0; j < obsSize; j++) {
									float& obsVal = envSet->state.obs.At(i, j);
									obsVal = (obsVal - mean[j]) / std[j];
								}
							}
						}

						std::vector<std::vector<float>> independentOpponentObs(opponentGroups.size());
						std::vector<std::vector<uint8_t>> independentOpponentActionMasks(opponentGroups.size());
						std::vector<std::vector<int>> independentOpponentInferencePlayers(opponentGroups.size());
						std::optional<Timer> staticObservationTimer;
						if (!opponentGroups.empty())
							staticObservationTimer.emplace();
						for (int groupIdx = 0; groupIdx < (int)opponentGroups.size(); groupIdx++) {
							OpponentInferenceGroup& group = opponentGroups[groupIdx];
							auto& groupObs = independentOpponentObs[groupIdx];
							auto& groupMasks = independentOpponentActionMasks[groupIdx];
							auto& inferencePlayers = independentOpponentInferencePlayers[groupIdx];
							groupObs.reserve((size_t)group.playerIndices.size() * group.version->inferenceObsSize);
							groupMasks.reserve((size_t)group.playerIndices.size() * group.version->inferenceNumActions);
							inferencePlayers.reserve(group.playerIndices.size());
							for (int playerIdx : group.playerIndices) {
								const int arenaIdx = playerArenaIndices[playerIdx];
								ArenaOpponentAssignment& assignment = arenaOpponentAssignments[arenaIdx];
								if (!ShouldInferStaticOpponentAction(assignment.staticCadenceStep))
									continue;

								inferencePlayers.push_back(playerIdx);
								if (group.version->inferenceAgent) {
									const int localPlayerIdx = playerIdx - envSet->state.arenaPlayerStartIdx[arenaIdx];
									GameState& state = envSet->state.gameStates[arenaIdx];
									Player& player = state.players[localPlayerIdx];
									groupObs += assignment.obsBuilder->BuildObs(player, state);
									groupMasks += assignment.actionParser->GetActionMask(player, state);
								}
							}
							if (group.version->inferenceAgent) {
								RG_ASSERT(groupObs.size() == inferencePlayers.size() * group.version->inferenceObsSize);
								RG_ASSERT(groupMasks.size() == inferencePlayers.size() * group.version->inferenceNumActions);
							}
							const uint64_t skippedRows = group.playerIndices.size() - inferencePlayers.size();
							staticCadenceSkippedRows += skippedRows;
							staticCadenceSkippedRowsByVersion[group.staticVersionIndex] += skippedRows;
						}
						if (staticObservationTimer.has_value())
							staticObservationBuildTime += staticObservationTimer->Elapsed();

						torch::Tensor tActions, tLogProbs;
						// Must remain after the obs normalization block above so it captures already-normalized obs.
						std::memcpy(
							tStates.data_ptr<float>(),
							envSet->state.obs.data.data(),
							(size_t)numPlayers * obsSize * sizeof(float));
						std::memcpy(
							tActionMasks.data_ptr<uint8_t>(),
							envSet->state.actionMasks.data.data(),
							(size_t)numPlayers * numActions * sizeof(uint8_t));

						// The pool is idle here, and each job writes only its own players' trajectories.
						const bool asyncAppendJobs = false;
						const int numExperiencePlayers = (int)experiencePlayerIndices.size();
						const int numAppendJobs = (numExperiencePlayers + PLAYERS_PER_APPEND_JOB - 1) / PLAYERS_PER_APPEND_JOB;
						auto forEachExperiencePlayer = [&](auto&& appendPlayer) {
							g_ThreadPool.StartBatchedJobs([&](int jobIdx) {
								const int firstPlayer = jobIdx * PLAYERS_PER_APPEND_JOB;
								const int endPlayer = std::min(firstPlayer + PLAYERS_PER_APPEND_JOB, numExperiencePlayers);
								for (int i = firstPlayer; i < endPlayer; i++)
									appendPlayer(experiencePlayerIndices[i]);
							}, numAppendJobs, asyncAppendJobs);
						};

						forEachExperiencePlayer([&](int newPlayerIdx) {
							AppendDimListRow(trajectories[newPlayerIdx].states, envSet->state.obs, newPlayerIdx);
							AppendDimListRow(trajectories[newPlayerIdx].actionMasks, envSet->state.actionMasks, newPlayerIdx);
						});

						IList earlyStaticActionIndices;
						IList earlyStaticActionApplyTicks;
						if (hasEarlyStaticActionRuntime) {
							earlyStaticActionIndices.assign(numPlayers, 0);
							earlyStaticActionApplyTicks.assign(numPlayers, -1);

							// Earlier native boundaries must be ready before physics advances. Only
							// these exceptional static groups leave the normal inference-overlap path.
							Timer earlyStaticInferTimer = {};
							for (int groupIdx = 0; groupIdx < (int)opponentGroups.size(); groupIdx++) {
								OpponentInferenceGroup& group = opponentGroups[groupIdx];
								if (!group.version->inferenceRequiresEarlyAction)
									continue;

								const std::vector<int>& inferencePlayers =
									independentOpponentInferencePlayers[groupIdx];
								if (inferencePlayers.empty())
									continue;

								torch::Tensor tdOpponentStates = torch::tensor(independentOpponentObs[groupIdx])
									.reshape({ -1, group.version->inferenceObsSize })
									.to(ppo->device, nonBlockingTransfer);
								torch::Tensor tdOpponentActionMasks = torch::tensor(
									independentOpponentActionMasks[groupIdx])
									.reshape({ -1, group.version->inferenceNumActions })
									.to(ppo->device, nonBlockingTransfer);
								torch::Tensor tOpponentActions;
								ModelManager::InferActionsFromModels(
									group.version->models,
									tdOpponentStates,
									tdOpponentActionMasks,
									group.version->inferenceDeterministic || ppo->config.deterministic,
									ppo->config.policyTemperature,
									ppo->config.useHalfPrecision,
									&tOpponentActions,
									NULL,
									0,
									ppo->config.useCudaGraphs,
									CudaGraphBatchPolicy::PowerOfTwo);
								recordStaticInference(group, inferencePlayers.size());

								const IList inferredActions = TENSOR_TO_VEC<int>(tOpponentActions.cpu());
								RG_ASSERT(inferredActions.size() == inferencePlayers.size());
								for (int row = 0; row < (int)inferencePlayers.size(); row++)
									staticPendingActionIndices[inferencePlayers[row]] = inferredActions[row];
							}
							const double earlyStaticElapsed = earlyStaticInferTimer.Elapsed();
							inferTime += earlyStaticElapsed;
							earlyStaticInferenceTime += earlyStaticElapsed;

							for (int arenaIdx = 0; arenaIdx < (int)arenaOpponentAssignments.size(); arenaIdx++) {
								ArenaOpponentAssignment& assignment = arenaOpponentAssignments[arenaIdx];
								if (!assignment.staticRequiresEarlyAction
									|| assignment.staticCadenceStep != assignment.staticActionApplyStep)
									continue;

								const int playerStartIdx = envSet->state.arenaPlayerStartIdx[arenaIdx];
								const GameState& state = envSet->state.gameStates[arenaIdx];
								for (int playerIdx = 0; playerIdx < (int)state.players.size(); playerIdx++) {
									if (state.players[playerIdx].team != assignment.opponentTeam)
										continue;

									const int globalPlayerIdx = playerStartIdx + playerIdx;
									staticAppliedActionIndices[globalPlayerIdx] =
										staticPendingActionIndices[globalPlayerIdx];
									earlyStaticActionIndices[globalPlayerIdx] =
										staticAppliedActionIndices[globalPlayerIdx];
									earlyStaticActionApplyTicks[globalPlayerIdx] =
										assignment.staticActionApplyTick;
								}
							}
						}

						Timer stepFirstTimer = {};
						if (hasEarlyStaticActionRuntime) {
							envSet->StepFirstHalf(
								asyncFirstHalf,
								actionParsersByPlayer,
								earlyStaticActionIndices,
								earlyStaticActionApplyTicks);
						} else if (hasIndependentOpponentRuntime) {
							envSet->StepFirstHalf(asyncFirstHalf, actionParsersByPlayer);
						} else {
							envSet->StepFirstHalf(asyncFirstHalf);
						}
						envStepTime += stepFirstTimer.Elapsed();

						bool sampleGpuInferenceTime = false;
#ifdef RG_CUDA_SUPPORT
						sampleGpuInferenceTime = ppo->device.is_cuda()
							&& inferenceCalls % CUDA_INFERENCE_TIMING_SAMPLE_INTERVAL == 0;
						if (sampleGpuInferenceTime)
							inferGpuStart.record();
#endif
						Timer inferTimer = {};
						torch::Tensor tdStates;
						torch::Tensor tdActionMasks;
						if (ppo->device.is_cuda()) {
							collectionDeviceStates.copy_(tStates, nonBlockingTransfer);
							collectionDeviceActionMasks.copy_(tActionMasks, nonBlockingTransfer);
							tdStates = collectionDeviceStates;
							tdActionMasks = collectionDeviceActionMasks;
						} else {
							tdStates = tStates;
							tdActionMasks = tActionMasks;
						}

						if (!opponentGroups.empty() || recentOpponentPlayerCount > 0) {
							const int64_t learnerBatchSize = deviceLearnerPlayerIndices.size(0);
							const bool usePersistentLearnerInputs =
								ppo->device.is_cuda() && ppo->config.useCudaGraphs;
							torch::Tensor tdLearnerStates;
							torch::Tensor tdLearnerActionMasks;
							int64_t learnerInferenceBatchSize = learnerBatchSize;
							if (usePersistentLearnerInputs) {
								learnerInferenceBatchSize = GetCudaGraphBatchBucket(learnerBatchSize);
								auto [bucketItr, inserted] = collectionLearnerInputBuckets.try_emplace(
									learnerInferenceBatchSize);
								PersistentPolicyInputBucket& bucket = bucketItr->second;
								if (inserted) {
									bucket.states = torch::zeros(
										{ learnerInferenceBatchSize, obsSize }, tdStates.options());
									bucket.actionMasks = torch::ones(
										{ learnerInferenceBatchSize, numActions }, tdActionMasks.options());
								}

								// Gather directly into caller-owned graph storage. Padded rows stay
								// initialized and are discarded after row-independent inference.
								torch::Tensor activeLearnerStates = bucket.states.narrow(0, 0, learnerBatchSize);
								torch::Tensor activeLearnerActionMasks = bucket.actionMasks.narrow(0, 0, learnerBatchSize);
								at::index_select_out(activeLearnerStates, tdStates, 0, deviceLearnerPlayerIndices);
								at::index_select_out(activeLearnerActionMasks, tdActionMasks, 0, deviceLearnerPlayerIndices);
								tdLearnerStates = bucket.states;
								tdLearnerActionMasks = bucket.actionMasks;
							} else {
								tdLearnerStates = tdStates.index_select(0, deviceLearnerPlayerIndices);
								tdLearnerActionMasks = tdActionMasks.index_select(0, deviceLearnerPlayerIndices);
							}
							torch::Tensor tLearnerActions;

							const float opponentMinActionProbability = 0;
							ModelManager::InferActionsFromModels(
								ppo->models, tdLearnerStates, tdLearnerActionMasks,
								ppo->config.deterministic, ppo->config.policyTemperature, ppo->config.useHalfPrecision,
								&tLearnerActions, &tLogProbs,
								ppo->config.minActionProbability,
								ppo->config.useCudaGraphs,
								usePersistentLearnerInputs
									? CudaGraphBatchPolicy::Exact
									: CudaGraphBatchPolicy::PowerOfTwo,
								usePersistentLearnerInputs
							);
							if (learnerInferenceBatchSize != learnerBatchSize) {
								tLearnerActions = tLearnerActions.narrow(0, 0, learnerBatchSize);
								if (tLogProbs.defined())
									tLogProbs = tLogProbs.narrow(0, 0, learnerBatchSize);
							}

							learnerInferenceActualRows += learnerBatchSize;
							learnerInferenceEvaluatedRows += learnerInferenceBatchSize;

							torch::Tensor tdActions = torch::empty(numPlayers, tLearnerActions.options());
							tdActions.index_copy_(0, deviceLearnerPlayerIndices, tLearnerActions);
							if (recentOpponentPlayerCount > 0) {
								torch::Tensor tdRecentStates = tdStates.index_select(
									0, deviceRecentPaddedPlayerIndices).view(
										{ recentActiveSlotCount, recentRowsPerSlot, obsSize });
								torch::Tensor tdRecentActionMasks = tdActionMasks.index_select(
									0, deviceRecentPaddedPlayerIndices).view(
										{ recentActiveSlotCount, recentRowsPerSlot, numActions });
								torch::Tensor tRecentPaddedActions;
								recentPolicyEnsemble->InferActions(
									tdRecentStates,
									tdRecentActionMasks,
									ppo->config.deterministic,
									ppo->config.policyTemperature,
									&tRecentPaddedActions,
									opponentMinActionProbability,
									ppo->config.useCudaGraphs);
								recentInferenceActualRows += recentOpponentPlayerCount;
								recentInferencePaddedRows +=
									(uint64_t)recentActiveSlotCount * recentRowsPerSlot;
								torch::Tensor tRecentActions = tRecentPaddedActions.flatten().index_select(
									0, deviceRecentValidFlatPositions);
								tdActions.index_copy_(0, deviceRecentPlayerIndices, tRecentActions);
							}
							for (int groupIdx = 0; groupIdx < (int)opponentGroups.size(); groupIdx++) {
								OpponentInferenceGroup& group = opponentGroups[groupIdx];
								if (group.version->inferenceRequiresEarlyAction)
									continue;
								const std::vector<int>& inferencePlayers =
									independentOpponentInferencePlayers[groupIdx];
								if (inferencePlayers.empty())
									continue;
								torch::Tensor deviceInferencePlayers = inferencePlayers.size() == group.playerIndices.size()
									? group.devicePlayerIndices
									: torch::tensor(inferencePlayers).to(ppo->device);

								torch::Tensor tdOpponentStates;
								torch::Tensor tdOpponentActionMasks;
								if (group.version->inferenceAgent) {
									tdOpponentStates = torch::tensor(independentOpponentObs[groupIdx])
										.reshape({ -1, group.version->inferenceObsSize })
										.to(ppo->device, nonBlockingTransfer);
									tdOpponentActionMasks = torch::tensor(independentOpponentActionMasks[groupIdx])
										.reshape({ -1, group.version->inferenceNumActions })
										.to(ppo->device, nonBlockingTransfer);
								} else {
									tdOpponentStates = tdStates.index_select(0, deviceInferencePlayers);
									tdOpponentActionMasks = tdActionMasks.index_select(0, deviceInferencePlayers);
								}

								torch::Tensor tOpponentActions;
								ModelManager::InferActionsFromModels(
									group.version->models, tdOpponentStates, tdOpponentActionMasks,
									group.version->inferenceDeterministic || ppo->config.deterministic,
									ppo->config.policyTemperature, ppo->config.useHalfPrecision,
									&tOpponentActions, NULL,
									opponentMinActionProbability,
									ppo->config.useCudaGraphs,
									CudaGraphBatchPolicy::PowerOfTwo
								);
								recordStaticInference(group, inferencePlayers.size());
								tdActions.index_copy_(0, deviceInferencePlayers, tOpponentActions);
							}
							tActions = tdActions;
						} else {
							ModelManager::InferActionsFromModels(
								ppo->models, tdStates, tdActionMasks,
								ppo->config.deterministic, ppo->config.policyTemperature, ppo->config.useHalfPrecision,
								&tActions, &tLogProbs,
								ppo->config.minActionProbability,
								ppo->config.useCudaGraphs,
								CudaGraphBatchPolicy::Exact,
								ppo->device.is_cuda()
							);
						}

						tActions = tActions.cpu();
						if (tLogProbs.defined())
							tLogProbs = tLogProbs.cpu();
						inferTime += inferTimer.Elapsed();
#ifdef RG_CUDA_SUPPORT
						if (sampleGpuInferenceTime) {
							inferGpuEnd.record();
							inferGpuEnd.synchronize();
							inferGpuSampleTime += inferGpuStart.elapsed_time(inferGpuEnd) / 1000.0;
							inferGpuTimingSamples++;
						}
#endif
						inferenceCalls++;

						auto curActions = TENSOR_TO_VEC<int>(tActions);
						FList newLogProbs;
						if (tLogProbs.defined())
							newLogProbs = TENSOR_TO_VEC<float>(tLogProbs);

						// Static policies keep their native cadence. Boundary-aligned policies use
						// the normal delayed-action path; earlier actions were already published by
						// StepFirstHalf at their exact native tick.
						for (int arenaIdx = 0; arenaIdx < (int)arenaOpponentAssignments.size(); arenaIdx++) {
							ArenaOpponentAssignment& assignment = arenaOpponentAssignments[arenaIdx];
							if (assignment.kind != ArenaOpponentKind::STATIC
								|| (!assignment.staticRequiresEarlyAction && assignment.staticCadenceRatio <= 1))
								continue;

							const bool capturePending = !assignment.staticRequiresEarlyAction
								&& assignment.staticCadenceStep == 0;
							const bool applyPending =
								!assignment.staticRequiresEarlyAction
								&& assignment.staticCadenceStep == assignment.staticActionApplyStep;
							const int playerStartIdx = envSet->state.arenaPlayerStartIdx[arenaIdx];
							const GameState& state = envSet->state.gameStates[arenaIdx];
							for (int playerIdx = 0; playerIdx < (int)state.players.size(); playerIdx++) {
								if (state.players[playerIdx].team != assignment.opponentTeam)
									continue;

								const int globalPlayerIdx = playerStartIdx + playerIdx;
								if (capturePending)
									staticPendingActionIndices[globalPlayerIdx] = curActions[globalPlayerIdx];
								if (applyPending)
									staticAppliedActionIndices[globalPlayerIdx] = staticPendingActionIndices[globalPlayerIdx];
								curActions[globalPlayerIdx] = staticAppliedActionIndices[globalPlayerIdx];
							}

							assignment.staticCadenceStep = AdvanceStaticOpponentCadence(
								assignment.staticCadenceStep,
								assignment.staticCadenceRatio);
						}

						Timer stepSecondTimer = {};
						envSet->Sync(); // Make sure the first half is done
						if (hasIndependentOpponentRuntime)
							envSet->StepSecondHalf(curActions, actionParsersByPlayer, asyncSecondHalf);
						else
							envSet->StepSecondHalf(curActions, asyncSecondHalf);
						envStepTime += stepSecondTimer.Elapsed();

						for (int arenaIdx = 0; arenaIdx < (int)envSet->state.gameStates.size(); arenaIdx++) {
							ArenaOpponentAssignment& assignment = arenaOpponentAssignments[arenaIdx];
							if (assignment.kind != ArenaOpponentKind::STATIC)
								continue;

							auto& gs = envSet->state.gameStates[arenaIdx];
							if (!gs.goalScored)
								continue;

							if (!staticOpponentMgr || assignment.staticVersionIndex < 0
								|| assignment.staticVersionIndex >= (int)staticOpponentMgr->versions.size())
									continue;

							PolicyVersion* version = &staticOpponentMgr->versions[assignment.staticVersionIndex];
							const bool learnerWon = RS_TEAM_FROM_Y(gs.ball.pos.y) == assignment.opponentTeam;
							const std::string metricSuffix = MakeStaticMetricKeySuffix(*version);
							auto& result = staticOpponentWinLoss[metricSuffix];
							result.version = version;
							if (learnerWon)
								result.wins++;
							else
								result.losses++;
							totalStaticDecidedGames++;
						}

						if (stepCallback) {
							stepCallback(this, envSet->state.gameStates, report);
						}

						// Calc average rewards
						if (config.addRewardsToMetrics && (Math::RandInt(0, config.rewardSampleRandInterval) == 0)) {
							int numSamples = RS_MIN(envSet->arenas.size(), config.maxRewardSamples);
							std::unordered_map<std::string, AvgTracker> avgRewards = {};
							for (int i = 0; i < numSamples; i++) {
								int arenaIdx = Math::RandInt(0, envSet->arenas.size());
								auto& prevRewards = envSet->state.lastRewards[arenaIdx];
								const std::string metricCategory = IsNormalGameplayEpisode(
									envSet->stateSetters[arenaIdx]->setterType)
									? "Rewards/"
									: "StateSetterRewards/";

								for (int j = 0; j < envSet->rewards[arenaIdx].size(); j++) {
									std::string rewardName = envSet->rewards[arenaIdx][j].reward->GetName();
									avgRewards[metricCategory + rewardName] += prevRewards[j];
								}
							}

							for (auto& pair : avgRewards)
								report.AddAvg(pair.first, pair.second.Get());
						}

						std::fill(curTerminals.begin(), curTerminals.end(), 0);
						for (int idx = 0; idx < envSet->arenas.size(); idx++) {
							uint8_t terminalType = envSet->state.terminals[idx];
							if (!terminalType)
								continue;

							auto playerStartIdx = envSet->state.arenaPlayerStartIdx[idx];
							int playersInArena = envSet->state.gameStates[idx].players.size();
							for (int i = 0; i < playersInArena; i++)
								curTerminals[playerStartIdx + i] = terminalType;
						}

						// Now that we've inferred and stepped the env, we can add that stuff to the trajectories
						forEachExperiencePlayer([&](int newPlayerIdx) {
							auto& traj = trajectories[newPlayerIdx];

							traj.actions.push_back(curActions[newPlayerIdx]);
							traj.rewards += envSet->state.rewards[newPlayerIdx];
							traj.logProbs += newLogProbs[learnerLogProbPositions[newPlayerIdx]];

							if (envSet->rewardHeadLayout)
								AppendDimListRow(traj.headRewards, envSet->state.headRewards, newPlayerIdx);

							int8_t terminalType = curTerminals[newPlayerIdx];
							if (!terminalType && traj.Length() >= maxEpisodeLength) {
								// Episode is too long, truncate it here
								// This won't actually reset the env, but rather will just add it to experience buffer as truncated
								terminalType = RLGC::TerminalType::TRUNCATED;
							}

							traj.terminals.push_back(terminalType);
							traj.playerCounts.push_back(
								(int8_t)envSet->state.gameStates[playerArenaIndices[newPlayerIdx]].players.size());
							if (envSet->rewardHeadLayout)
								traj.arenaIndices.push_back(playerArenaIndices[newPlayerIdx]);

							if (terminalType == RLGC::TerminalType::TRUNCATED) {
								// Truncation requires an additional next state for the critic
								AppendDimListRow(traj.nextStates, envSet->state.obs, newPlayerIdx);
							}

							finishedPlayers[newPlayerIdx] = terminalType != 0;
						});

						// Serial and in ascending player order, so combinedTraj's layout does not depend on job scheduling.
						for (int newPlayerIdx : experiencePlayerIndices) {
							if (!finishedPlayers[newPlayerIdx])
								continue;

							combinedTraj.Append(trajectories[newPlayerIdx]);
							trajectories[newPlayerIdx].Clear();
						}
					}

					report["Inference Time"] = inferTime;
					report["Inference Calls"] = (double)inferenceCalls;
					if (inferGpuTimingSamples > 0) {
						report["Inference GPU Time Per Call"] =
							inferGpuSampleTime / inferGpuTimingSamples;
					}
					report["Env Step Time"] = envStepTime;

					// Rewards run in parallel per arena, so spread their CPU time over the pool
					// to estimate their share of Env Step Time.
					report["Reward Time"] = envSet->TakeRewardTime() / g_ThreadPool.GetNumThreads();
				}

				// Adaptive reward weights: update once per iteration (serial), applied to the next
				// iteration's collection, and report per-team-size weight + measured/target ratio.
				if (envSet) {
					for (const auto& entry : envSet->adaptiveWeightController.UpdateAndApply(envSet->rewards, (double)stepsCollected)) {
						// Skip metrics for uneven team sizes (odd player counts like 3p/5p); their weights still adapt.
						if (entry.playerCount % 2 != 0)
							continue;

						const std::string group = "Adaptive Weights/" + TeamSizeLabel(entry.playerCount) + "/" + entry.name;
						report[group + "/Weight"] = entry.weight;
						report[group + "/Ratio"] = entry.ratio;
					}
				}

				float collectionTime = collectionTimer.Elapsed();

				Timer consumptionTimer = {};
				{ // Process timesteps
					RG_NO_GRAD;

					// Make and transpose tensors
					const int64_t trajectoryLength = (int64_t)combinedTraj.Length();
					torch::Tensor tStates;
					torch::Tensor tActionMasks;
					torch::Tensor tActions;
					torch::Tensor tLogProbs;
					torch::Tensor tRewards;
					torch::Tensor tTerminals;
					tStates = torch::from_blob(
						combinedTraj.states.data(), {trajectoryLength, (int64_t)obsSize},
						torch::TensorOptions().dtype(torch::kFloat32));
					tActionMasks = torch::from_blob(
						combinedTraj.actionMasks.data(), {trajectoryLength, (int64_t)numActions},
						torch::TensorOptions().dtype(torch::kUInt8));
					tActions = torch::from_blob(
						combinedTraj.actions.data(), {trajectoryLength},
						torch::TensorOptions().dtype(torch::kInt64));
					tLogProbs = torch::from_blob(
						combinedTraj.logProbs.data(), {trajectoryLength},
						torch::TensorOptions().dtype(torch::kFloat32));
					tRewards = torch::from_blob(
						combinedTraj.rewards.data(), {trajectoryLength},
						torch::TensorOptions().dtype(torch::kFloat32));
					tTerminals = torch::from_blob(
						combinedTraj.terminals.data(), {trajectoryLength},
						torch::TensorOptions().dtype(torch::kInt8));

					// States we truncated at (there could be none)
					torch::Tensor tNextTruncStates;
					if (!combinedTraj.nextStates.empty())
						tNextTruncStates = torch::from_blob(
							combinedTraj.nextStates.data(), {(int64_t)(combinedTraj.nextStates.size() / obsSize), (int64_t)obsSize},
							torch::TensorOptions().dtype(torch::kFloat32));

					report["Average Step Reward"] = tRewards.mean().item<float>();
					report["Collected Timesteps"] = stepsCollected;
					
					const bool decomposedCritic = popArtStats != nullptr;
					auto inferValues = [&](torch::Tensor states) {
						return decomposedCritic
							? ModelManager::InferCriticHeads(states, ppo->models, ppo->config.useHalfPrecision)
							: ModelManager::InferCritic(states, ppo->models, ppo->config.useHalfPrecision);
					};

					torch::Tensor tValPreds;
					torch::Tensor tTruncValPreds;
					torch::Tensor tdStates;
					torch::Tensor tdActionMasks;
					torch::Tensor tdNextTruncStates;
					const bool trainingDeviceIsCpu = ppo->device.is_cpu();

					if (!trainingDeviceIsCpu) {
						tdStates = tStates.to(ppo->device, nonBlockingTransfer);
						tdActionMasks = tActionMasks.to(ppo->device, nonBlockingTransfer);
						if (tNextTruncStates.defined())
							tdNextTruncStates = tNextTruncStates.to(ppo->device, nonBlockingTransfer);
					}

					if (trainingDeviceIsCpu) {
						// Predict values all at once
						tValPreds = inferValues(tStates.to(ppo->device, nonBlockingTransfer)).cpu();
						if (tNextTruncStates.defined())
							tTruncValPreds = inferValues(tNextTruncStates.to(ppo->device, nonBlockingTransfer)).cpu();
					} else {
						// Predict values using memory microbatches.
						tValPreds = decomposedCritic
							? torch::zeros({ trajectoryLength, (int64_t)popArtStats->GetNumHeads() })
							: torch::zeros({ trajectoryLength });
						for (int64_t i = 0; i < trajectoryLength; i += ppo->config.microBatchSize) {
							int64_t start = i;
							int64_t end = RS_MIN(i + ppo->config.microBatchSize, trajectoryLength);
							torch::Tensor tStatesPart = tdStates.defined()
								? tdStates.slice(0, start, end)
								: tStates.slice(0, start, end).to(ppo->device, nonBlockingTransfer);

							torch::Tensor valPredsPart = inferValues(tStatesPart).cpu();
							RG_ASSERT(valPredsPart.size(0) == (end - start));
							tValPreds.slice(0, start, end).copy_(valPredsPart, nonBlockingTransfer);
						}

						if (tNextTruncStates.defined()) {
							// This really just should never happen
							// If this is ever actually a real problem in a legitimate use case, ping Zealan in the dead of night
							RG_ASSERT(tNextTruncStates.size(0) <= ppo->config.microBatchSize);

							tTruncValPreds = inferValues(
								tdNextTruncStates.defined()
									? tdNextTruncStates
									: tNextTruncStates.to(ppo->device, nonBlockingTransfer)).cpu();
						}
					}

					// Value heads output normalized values; GAE needs raw reward units.
					if (decomposedCritic) {
						tValPreds = popArtStats->Denormalize(tValPreds);
						if (tTruncValPreds.defined())
							tTruncValPreds = popArtStats->Denormalize(tTruncValPreds);
					}

					// Include truncations so timeout-heavy drill stages report their actual reset cadence.
					float terminalRate = (tTerminals != 0).to(torch::kFloat32).mean().item<float>();
					report["Episode Length"] = terminalRate > 0.f ? 1.f / terminalRate : (float)maxEpisodeLength;

					// Run GAE
					torch::Tensor tAdvantages, tTargetVals, tReturns;
					DecomposedGAEResult headGAE = {};
					torch::Tensor tSampleWeights, tSamplePresence;

					if (decomposedCritic) {
						const RLGC::RewardHeadLayout& layout = *envSet->rewardHeadLayout;
						const int64_t numHeads = layout.GetNumHeads();
						RG_ASSERT(combinedTraj.headRewards.size() == (size_t)trajectoryLength * numHeads);
						RG_ASSERT(combinedTraj.arenaIndices.size() == (size_t)trajectoryLength);

						torch::Tensor tHeadRewards = torch::from_blob(
							combinedTraj.headRewards.data(), { trajectoryLength, numHeads },
							torch::TensorOptions().dtype(torch::kFloat32));
						torch::Tensor tArenaIndices = torch::from_blob(
							combinedTraj.arenaIndices.data(), { trajectoryLength },
							torch::TensorOptions().dtype(torch::kInt32)).to(torch::kInt64);

						// Current weights, so a weight change made by UpdateAndApply this iteration already applies to this batch.
						RLGC::DimList2<float> weightTable = layout.BuildWeightTable(envSet->rewards);
						const RLGC::DimList2<uint8_t>& presenceTable = layout.GetPresence();
						const int64_t numArenas = (int64_t)weightTable.size[0];

						tSampleWeights = torch::from_blob(
							weightTable.data.data(), { numArenas, numHeads },
							torch::TensorOptions().dtype(torch::kFloat32)).index_select(0, tArenaIndices);
						tSamplePresence = torch::from_blob(
							const_cast<uint8_t*>(presenceTable.data.data()), { numArenas, numHeads },
							torch::TensorOptions().dtype(torch::kUInt8)).index_select(0, tArenaIndices).to(torch::kFloat32);

						headGAE = ComputeDecomposedGAE(
							tHeadRewards, tTerminals, tValPreds, tTruncValPreds, tSampleWeights, config.ppo.gaeGamma, config.ppo.gaeLambda);
						tAdvantages = headGAE.advantages;
						tTargetVals = headGAE.targetValues;
						tReturns = headGAE.returns;
					} else {
						float rewClipPortion;
						GAE::Compute(
							tRewards, tTerminals, tValPreds, tTruncValPreds,
							tAdvantages, tTargetVals, tReturns, rewClipPortion,
							config.ppo.gaeGamma, config.ppo.gaeLambda, returnStat ? returnStat->GetSTD() : 1, config.ppo.rewardClipRange
						);
						report["Clipped Reward Portion"] = rewClipPortion;
					}

					if (returnStat) {
						report["GAE/Returns STD"] = returnStat->GetSTD();

						int numToIncrement = RS_MIN(config.maxReturnSamples, tReturns.size(0));
						if (numToIncrement > 0) {
							auto selectedReturns = tReturns.index_select(0, torch::randint(tReturns.size(0), { (int64_t)numToIncrement }));
							returnStat->Increment(TENSOR_TO_VEC<float>(selectedReturns));
						}
					}
					report["GAE/Avg Return"] = tReturns.abs().mean().item<float>();
					report["GAE/Returns STD (Iteration)"] = tReturns.std(/*unbiased*/ false).item<float>();
					report["GAE/Avg Advantage"] = tAdvantages.abs().mean().item<float>();
					report["GAE/Mean Advantage"] = tAdvantages.mean().item<float>();
					report["GAE/Advantage STD"] = tAdvantages.std(false).item<float>();
					report["GAE/Avg Val Target"] = tTargetVals.abs().mean().item<float>();

					RG_ASSERT(combinedTraj.playerCounts.size() == (size_t)trajectoryLength);

					torch::Tensor tPlayerCounts = torch::from_blob(
						combinedTraj.playerCounts.data(), {trajectoryLength},
						torch::TensorOptions().dtype(torch::kInt8));
					for (const GGL::ModeCriticStats& mode : GGL::ComputePerModeCriticStats(
						decomposedCritic ? headGAE.valPreds : tValPreds, tTargetVals, tAdvantages, tReturns, tPlayerCounts)) {
						const std::string label = TeamSizeLabel(mode.playerCount);
						if (mode.explainedVariance.has_value())
							report["Critic/" + label + "/Explained Variance"] = *mode.explainedVariance;
						report["Critic/" + label + "/Value Error"] = mode.valueError;
						report["GAE/" + label + "/Advantage STD"] = mode.advantageStd;
						report["GAE/" + label + "/Avg Return"] = mode.avgAbsReturn;
						report["GAE/" + label + "/Mean Advantage"] = mode.meanAdvantage;
					}

					torch::Tensor tCriticTargets = tTargetVals;
					torch::Tensor tCriticLossWeights;

					if (decomposedCritic) {
						tCriticTargets = ApplyPopArtUpdate(*popArtStats, *ppo->models["critic"], headGAE.headTargets, tSamplePresence);

						const torch::Tensor sigmas = popArtStats->GetSigmas();
						tCriticLossWeights = ComputeCriticLossWeights(
							tSamplePresence, tSampleWeights, sigmas, config.ppo.decomposedCritic.lossWeighting);

						const std::vector<std::string>& headNames = envSet->rewardHeadLayout->GetHeadNames();
						const std::vector<ValueHeadStats> headStats = ComputeValueHeadStats(
							tValPreds, headGAE.headTargets, tSamplePresence, tSampleWeights, sigmas, tPlayerCounts);

						const bool logDetailedHeadMetrics = config.ppo.decomposedCritic.logDetailedHeadMetrics;
						const torch::Tensor means = popArtStats->GetMeans();

						for (size_t headIdx = 0; headIdx < headNames.size(); headIdx++) {
							const std::string& name = headNames[headIdx];
							report["Value Heads/" + name + "/Sigma"] = sigmas[(int64_t)headIdx].item<float>();
							report["Value Heads/" + name + "/Contribution Share"] = headStats[headIdx].contributionShare;

							if (logDetailedHeadMetrics) {
								report["Value Heads/" + name + "/Mean"] = means[(int64_t)headIdx].item<float>();
								report["Value Heads/" + name + "/Loss"] = headStats[headIdx].normalizedLoss;
							}

							for (const HeadModeStats& mode : headStats[headIdx].modes) {
								const std::string modePrefix = "Value Heads/" + TeamSizeLabel(mode.playerCount) + "/" + name;
								if (mode.explainedVariance.has_value())
									report[modePrefix + "/Explained Variance"] = *mode.explainedVariance;

								if (!logDetailedHeadMetrics)
									continue;

								if (mode.targetStd.has_value())
									report[modePrefix + "/Target STD"] = *mode.targetStd;
								report[modePrefix + "/Contribution Share"] = mode.contributionShare;
							}
						}

						report["Value Heads/Heads At Sigma Floor"] = popArtStats->CountHeadsAtFloor();
					}

					// Set experience buffer. Move once to the training device so PPOLearner::Learn
					// can slice microbatches without per-microbatch CPU->GPU transfers.
					if (trainingDeviceIsCpu) {
						// CPU .to(device) is a no-op, so clone rollout views before ExperienceBuffer
						// can retain them beyond the local vector lifetime.
						experience.data.actions     = tActions.clone();
						experience.data.logProbs    = tLogProbs.clone();
						experience.data.actionMasks = tActionMasks.clone();
						experience.data.states      = tStates.clone();
					} else {
						experience.data.actions     = tActions.to(ppo->device, nonBlockingTransfer);
						experience.data.logProbs    = tLogProbs.to(ppo->device, nonBlockingTransfer);
						experience.data.actionMasks = tdActionMasks;
						experience.data.states      = tdStates;
					}
					experience.data.advantages   = tAdvantages.to(ppo->device, nonBlockingTransfer);
					experience.data.targetValues = tCriticTargets.to(ppo->device, nonBlockingTransfer);
					experience.data.criticLossWeights = tCriticLossWeights.defined()
						? tCriticLossWeights.to(ppo->device, nonBlockingTransfer)
						: torch::Tensor();
				}

				// Warm start temporarily lowers policy LR after checkpoint resume to avoid unstable first updates.
				if (warmStartIterationsLeft > 0 && !warmStartActive) {
					RG_LOG("[WarmStartPolicy] Applying reduced policy LR");
					ppo->SetLearningRates(
						config.ppo.policyLR * config.warmStartPolicyLRMult,
						config.ppo.criticLR);
					warmStartActive = true;
				}

				if (warmStartIterationsLeft == 0 && warmStartActive) {
					RG_LOG("[WarmStartPolicy] Restoring base learning rates");
					ppo->SetLearningRates(config.ppo.policyLR, config.ppo.criticLR);
					warmStartActive = false;
				}

				// Learn
				Timer learnTimer = {};
				ppo->Learn(experience, report, isFirstIteration);
				float learnElapsed = learnTimer.Elapsed();
				report[config.algorithm == LearnerAlgorithm::PPG ? "PPG Learn Time" : "PPO Learn Time"] = learnElapsed;

				if (warmStartIterationsLeft > 0)
					warmStartIterationsLeft--;

				// Set metrics
				float consumptionTime = consumptionTimer.Elapsed();
				const bool isPPG = config.algorithm == LearnerAlgorithm::PPG;
				const double iterationTime = collectionTime + consumptionTime;
				double overallStepsPerSecond = iterationTime > 0 ? stepsCollected / iterationTime : 0;

				SetStepsPerSecond(report, "Steps Per Second/Collection", stepsCollected, collectionTime);
				SetStepsPerSecond(report, "Steps Per Second/Consumption", stepsCollected, consumptionTime);

				if (isPPG) {
					ppgCycleSeconds += iterationTime;
					ppgCycleTimesteps += stepsCollected;

					SetStepsPerSecond(report, "Steps Per Second/PPG Policy Phase", stepsCollected, report["PPG/Policy Phase Time"]);
					if (report.Has("PPG/Aux Phase Time"))
						SetStepsPerSecond(report, "Steps Per Second/PPG Auxiliary Phase", report["PPG/Aux Buffer Timesteps"], report["PPG/Aux Phase Time"]);

					const bool auxPhaseCompleted = report.Has("PPG/Aux Phase Ran") && report["PPG/Aux Phase Ran"] != 0;
					if (auxPhaseCompleted) {
						if (ppgCycleSeconds > 0)
							ppgEffectiveStepsPerSecond = ppgCycleTimesteps / ppgCycleSeconds;
						ppgCycleSeconds = 0;
						ppgCycleTimesteps = 0;
					}

					if (ppgEffectiveStepsPerSecond > 0)
						overallStepsPerSecond = ppgEffectiveStepsPerSecond;
				}

				report["Steps Per Second/Overall"] = overallStepsPerSecond;
				report["Collection Time"] = collectionTime;
				report["Consumption Time"] = consumptionTime;
				report["Iteration Time"] = iterationTime;

				if (runtimeConfig.onIterationReport)
					runtimeConfig.onIterationReport(report);

				const CudaGraphStats modelGraphStatsAfter = ModelManager::GetCudaGraphStats();
				auto graphCounterDelta = [](uint64_t after, uint64_t before) {
					return after >= before ? after - before : after;
				};
				report["PolicyInference/CUDA Graph Captures"] = graphCounterDelta(
					modelGraphStatsAfter.captures, modelGraphStatsBefore.captures);
				report["PolicyInference/CUDA Graph Replays"] = graphCounterDelta(
					modelGraphStatsAfter.replays, modelGraphStatsBefore.replays);
				report["PolicyInference/CUDA Graph Persistent Input Replays"] = graphCounterDelta(
					modelGraphStatsAfter.persistentInputReplays,
					modelGraphStatsBefore.persistentInputReplays);
				report["PolicyInference/CUDA Graph Fallbacks"] = graphCounterDelta(
					modelGraphStatsAfter.fallbacks, modelGraphStatsBefore.fallbacks);
				report["PolicyInference/CUDA Graph Capture Failures"] = graphCounterDelta(
					modelGraphStatsAfter.captureFailures, modelGraphStatsBefore.captureFailures);
				report["PolicyInference/CUDA Graph Variant Limit Fallbacks"] = graphCounterDelta(
					modelGraphStatsAfter.variantLimitFallbacks,
					modelGraphStatsBefore.variantLimitFallbacks);
				report["PolicyInference/CUDA Graph Cache Entries"] = modelGraphStatsAfter.cacheEntries;
				report["PolicyInference/Learner Evaluated Rows Per Actual Row"] =
					learnerInferenceActualRows > 0
					? learnerInferenceEvaluatedRows / (double)learnerInferenceActualRows
					: 1;

				std::set<PolicyVersionKey> activeRecentVersions;
				std::set<int> activeStaticVersions;
				int activeRecentArenas = 0;
				int activeStaticArenas = 0;
				std::map<std::string, int> activeStaticArenasByOpponent;
				double activeRecentTimestepTotal = 0;
				uint64_t activeRecentTimestepMin = UINT64_MAX;
				uint64_t activeRecentTimestepMax = 0;
				for (const ArenaOpponentAssignment& assignment : arenaOpponentAssignments) {
					if (assignment.kind == ArenaOpponentKind::RECENT) {
						activeRecentArenas++;
						activeRecentVersions.insert(assignment.recentVersionKey);
						activeRecentTimestepTotal += assignment.recentVersionKey.timesteps;
						activeRecentTimestepMin = std::min(activeRecentTimestepMin, assignment.recentVersionKey.timesteps);
						activeRecentTimestepMax = std::max(activeRecentTimestepMax, assignment.recentVersionKey.timesteps);
					} else if (assignment.kind == ArenaOpponentKind::STATIC) {
						activeStaticArenas++;
						activeStaticVersions.insert(assignment.staticVersionIndex);
						if (staticOpponentMgr && assignment.staticVersionIndex >= 0
							&& assignment.staticVersionIndex < (int)staticOpponentMgr->versions.size()) {
							const PolicyVersion& version = staticOpponentMgr->versions[assignment.staticVersionIndex];
							activeStaticArenasByOpponent[MakeStaticMetricKeySuffix(version)]++;
						}
					}
				}

				report["RecentOpponents/Target Sample Fraction"] = recentSampleTracker.GetTargetFraction();
				report["RecentOpponents/Rolling Sample Fraction"] = recentSampleTracker.GetSampleFraction();
				report["RecentOpponents/Iteration Sample Fraction"] = stepsCollected > 0
					? recentStepsCollected / (double)stepsCollected
					: 0;
				report["RecentOpponents/Active Arenas"] = activeRecentArenas;
				report["RecentOpponents/Active Versions"] = activeRecentVersions.size();
				report["RecentOpponents/Selectable Versions"] = recentOpponentStack.GetSelectable().size();
				report["RecentOpponents/Draining Versions"] = recentOpponentStack.GetDraining().has_value() ? 1 : 0;
				report["RecentOpponents/Cohort Versions"] =
					recentOpponentStack.GetSelectable().size()
					+ (recentOpponentStack.GetDraining().has_value() ? 1 : 0);
				report["RecentOpponents/Stack Rotations"] = recentOpponentStack.GetRotationCount();
				report["RecentOpponents/Rotation Interval Timesteps"] =
					config.recentOpponentRotationTimesteps;
				report["RecentOpponents/Pending Rotations"] =
					recentOpponentStack.GetPendingRotationCount();
				report["RecentOpponents/Next Rotation Timestep"] =
					recentOpponentStack.GetNextRotationTimestep();
				report["RecentOpponents/Episodes Started"] = recentEpisodesStarted;
				report["RecentOpponents/Episodes Completed"] = recentEpisodesCompleted;
				report["RecentOpponents/Normal Episodes Started"] = normalEpisodesStarted;
				report["RecentOpponents/Ownership Changes Outside Reset"] = 0;
				report["RecentOpponents/Discarded Partial Trajectories At Reset"] = unexpectedOwnershipChanges;
				if (activeRecentArenas > 0) {
					report["RecentOpponents/Active Version Timestep Mean"] =
						activeRecentTimestepTotal / activeRecentArenas;
					report["RecentOpponents/Active Version Timestep Min"] = activeRecentTimestepMin;
					report["RecentOpponents/Active Version Timestep Max"] = activeRecentTimestepMax;
				}
				if (recentPolicyEnsemble) {
					const RecentPolicyEnsembleStats stats = recentPolicyEnsemble->GetStats();
					report["RecentOpponents/Ensemble Graph Captures"] =
						stats.captures - recentEnsembleStatsBefore.captures;
					report["RecentOpponents/Ensemble Graph Replays"] =
						stats.replays - recentEnsembleStatsBefore.replays;
					report["RecentOpponents/Ensemble Graph Fallbacks"] =
						stats.fallbacks - recentEnsembleStatsBefore.fallbacks;
					report["RecentOpponents/Ensemble Graph Capture Failures"] =
						stats.captureFailures - recentEnsembleStatsBefore.captureFailures;
					report["RecentOpponents/Ensemble Graph Evictions"] =
						stats.graphEvictions - recentEnsembleStatsBefore.graphEvictions;
					report["RecentOpponents/Ensemble Weight Copies"] =
						stats.weightCopies - recentEnsembleStatsBefore.weightCopies;
					report["RecentOpponents/Ensemble Graph Entries"] = stats.graphEntries;
					report["RecentOpponents/Ensemble Evaluated Rows Per Actual Row"] =
						recentInferenceActualRows > 0
						? recentInferencePaddedRows / (double)recentInferenceActualRows
						: 0;
				}
				report["StaticOpponents/Active Arenas"] = activeStaticArenas;
				report["StaticOpponents/Target Sample Fraction"] = staticSampleTracker.GetTargetFraction();
				report["StaticOpponents/Rolling Sample Fraction"] = staticSampleTracker.GetSampleFraction();
				report["StaticOpponents/Iteration Sample Fraction"] = stepsCollected > 0
					? staticStepsCollected / (double)stepsCollected
					: 0;
				report["StaticOpponents/Active Versions"] = activeStaticVersions.size();
				report["StaticOpponents/Selectable Versions"] = staticOpponentCohort
					? staticOpponentCohort->GetSelectable().size()
					: 0;
				report["StaticOpponents/Draining Versions"] =
					staticOpponentCohort && staticOpponentCohort->GetDraining().has_value() ? 1 : 0;
				report["StaticOpponents/Cohort Versions"] =
					staticOpponentCohort
					? staticOpponentCohort->GetSelectable().size()
						+ (staticOpponentCohort->GetDraining().has_value() ? 1 : 0)
					: 0;
				report["StaticOpponents/Cohort Rotations"] = staticOpponentCohort
					? staticOpponentCohort->GetRotationCount()
					: 0;
				report["StaticOpponents/Rotation Interval Timesteps"] =
					config.staticOpponentRotationTimesteps;
				report["StaticOpponents/Pending Rotations"] = staticOpponentCohort
					? staticOpponentCohort->GetPendingRotationCount()
					: 0;
				report["StaticOpponents/Next Rotation Timestep"] = staticOpponentCohort
					? staticOpponentCohort->GetNextRotationTimestep()
					: config.staticOpponentRotationTimesteps;
				report["StaticOpponents/Episodes Started"] = staticEpisodesStarted;
				report["StaticOpponents/Episodes Completed"] = staticEpisodesCompleted;
				report["StaticOpponents/Runtime Reuses"] = staticRuntimeReuses;
				report["StaticOpponents/Runtime Rebuilds"] = staticRuntimeRebuilds;
				report["StaticOpponents/Runtime Reuse Fraction"] =
					staticRuntimeReuses + staticRuntimeRebuilds > 0
					? staticRuntimeReuses / (double)(staticRuntimeReuses + staticRuntimeRebuilds)
					: 0;
				report["StaticOpponents/Inference Calls"] = staticInferenceCalls;
				report["StaticOpponents/Inference Rows"] = staticInferenceRows;
				report["StaticOpponents/Cadence Skipped Rows"] = staticCadenceSkippedRows;
				report["StaticOpponents/Cadence Saved Fraction"] =
					staticInferenceRows + staticCadenceSkippedRows > 0
					? staticCadenceSkippedRows / (double)(staticInferenceRows + staticCadenceSkippedRows)
					: 0;
				report["StaticOpponents/Inference Rows Per Call"] = staticInferenceCalls > 0
					? staticInferenceRows / (double)staticInferenceCalls
					: 0;
				report["StaticOpponents/CUDA Graph Requested Rows Per Actual Row"] =
					staticInferenceRows > 0
					? staticInferenceRequestedRows / (double)staticInferenceRows
					: 0;
				report["StaticOpponents/Observation Build Time"] = staticObservationBuildTime;
				report["StaticOpponents/Early Inference Time"] = earlyStaticInferenceTime;
				for (const auto& [opponent, activeArenas] : activeStaticArenasByOpponent)
					report["StaticOpponents/ActiveArenas/" + opponent] = activeArenas;
				if (staticOpponentMgr) {
					for (int versionIdx = 0; versionIdx < (int)staticVersionCount; versionIdx++) {
						if (staticEpisodesStartedByVersion[versionIdx] == 0
							&& staticLearnerSamplesByVersion[versionIdx] == 0
							&& staticInferenceCallsByVersion[versionIdx] == 0
							&& staticCadenceSkippedRowsByVersion[versionIdx] == 0)
							continue;

						const std::string suffix = MakeStaticMetricKeySuffix(
							staticOpponentMgr->versions[versionIdx]);
						report["StaticOpponents/EpisodeStarts/" + suffix] =
							staticEpisodesStartedByVersion[versionIdx];
						report["StaticOpponents/LearnerSamples/" + suffix] =
							staticLearnerSamplesByVersion[versionIdx];
						report["StaticOpponents/InferenceCalls/" + suffix] =
							staticInferenceCallsByVersion[versionIdx];
						report["StaticOpponents/InferenceRows/" + suffix] =
							staticInferenceRowsByVersion[versionIdx];
						report["StaticOpponents/CadenceSkippedRows/" + suffix] =
							staticCadenceSkippedRowsByVersion[versionIdx];
					}
				}

				uint64_t prevTimesteps = totalTimesteps;
				totalTimesteps += stepsCollected;
				report["Total Timesteps"] = totalTimesteps;
				totalIterations++;
				report["Total Iterations"] = totalIterations;

				if (versionMgr) {
					std::set<PolicyVersionKey> protectedRecentVersions = activeRecentVersions;
					for (const PolicyVersionKey& key : recentOpponentStack.GetProtectedVersions())
						protectedRecentVersions.insert(key);
					versionMgr->SetProtectedVersions(std::vector<PolicyVersionKey>(
						protectedRecentVersions.begin(), protectedRecentVersions.end()));
					versionMgr->OnIteration(ppo, report, totalTimesteps, prevTimesteps);
				}

				if (totalStaticDecidedGames > 0)
					report["StaticOpponents/DecidedGames"] = totalStaticDecidedGames;
				for (const auto& pair : staticOpponentWinLoss) {
					const int decidedGames = pair.second.wins + pair.second.losses;
					if (decidedGames <= 0)
						continue;

					const std::string baseKey = "StaticOpponents/";
					report[baseKey + "WinRate/" + pair.first] = pair.second.wins / (double)decidedGames;
					report[baseKey + "Games/" + pair.first] = decidedGames;
				}

				// RocketForge: target-driven exit. Use the same stop path as RequestStop.
				if (runtimeConfig.stopAtTotalTimesteps > 0 && totalTimesteps >= runtimeConfig.stopAtTotalTimesteps) {
					stopRequested.store(true, std::memory_order_relaxed);
				}

				if (stopRequested.load(std::memory_order_relaxed)) {
					if (!runtimeConfig.checkpointFolder.empty())
						Save(true);
					return;
				}

				if (!runtimeConfig.checkpointFolder.empty()) {
					if (totalTimesteps / config.tsPerSave > prevTimesteps / config.tsPerSave) {
						// Auto-save
						Save();
					}
				}

				report.Finish();

				SendMetrics(report);

				report.Display(
					{
						"Average Step Reward",
						"Episode Length",
						"",
						"GAE/Avg Advantage",
						"GAE/Avg Return",
						"GAE/Avg Val Target",
						"GAE/Returns STD",
						"Clipped Reward Portion",
						"",
						"LossAndEntropy/Policy Entropy",
						"LossAndEntropy/Policy Loss",
						"-LossAndEntropy/Policy Loss Abs",
						"-LossAndEntropy/Policy Relative Entropy Loss",
						"LossAndEntropy/Entropy Bonus",
						"-LossAndEntropy/Entropy Bonus to Policy Loss Ratio",
						"Scheduled/PPO/Entropy Scale",
						"LossAndEntropy/Critic Loss",
						"LossAndEntropy/Guiding Loss",
						"LossAndEntropy/Mean KL Divergence",
						"LossAndEntropy/SB3 Clip Fraction",
						"First Accuracy",
						"",
						"LossAndEntropy/Policy Update Magnitude",
						"LossAndEntropy/Critic Update Magnitude",
						"LossAndEntropy/Shared Head Update Magnitude",
						"",
						"Steps Per Second/Collection",
						"Steps Per Second/Consumption",
						"Steps Per Second/Overall",
						"-Steps Per Second/PPG Effective Overall",
						"-Steps Per Second/PPG Policy Phase",
						"-Steps Per Second/PPG Auxiliary Phase",
						"",
						"Collection Time",
						"-Inference Time",
						"-Inference GPU Time Per Call",
						"-Inference Calls",
						"-Env Step Time",
						"-Reward Time",
						"Consumption Time",
						"Iteration Time",
						"",
						"Collected Timesteps",
						"Total Timesteps",
						"Total Iterations",
						"",
						"PolicyInference/CUDA Graph Replays",
						"PolicyInference/CUDA Graph Persistent Input Replays",
						"PolicyInference/CUDA Graph Captures",
						"PolicyInference/CUDA Graph Fallbacks",
						"PolicyInference/CUDA Graph Capture Failures",
						"PolicyInference/CUDA Graph Variant Limit Fallbacks",
						"PolicyInference/CUDA Graph Cache Entries",
						"PolicyInference/Learner Evaluated Rows Per Actual Row",
						"",
						"RecentOpponents/Iteration Sample Fraction",
						"RecentOpponents/Active Arenas",
						"RecentOpponents/Active Versions",
						"RecentOpponents/Selectable Versions",
						"RecentOpponents/Draining Versions",
						"RecentOpponents/Ensemble Graph Captures",
						"RecentOpponents/Ensemble Graph Replays",
						"RecentOpponents/Ensemble Graph Fallbacks",
						"RecentOpponents/Ensemble Graph Capture Failures",
						"RecentOpponents/Ensemble Graph Evictions",
						"RecentOpponents/Ensemble Weight Copies",
						"RecentOpponents/Ensemble Evaluated Rows Per Actual Row",
						"",
						"SkillTest/NextRunTimestep",
						"SkillTest/Rating/1v1",
						"SkillTest/Rating/2v2",
						"SkillTest/Rating/3v3"
					}
				);

				if (runtimeConfig.stopAfterIterations > 0 && totalIterations >= runtimeConfig.stopAfterIterations)
					return;
			}
		}
		
	} catch (std::exception& e) {
		RG_ERR_CLOSE("Exception thrown during main learner loop: " << e.what());
	}
}

GGL::Learner::~Learner() {
	delete ppo;
	delete versionMgr;
	delete staticOpponentMgr;
	delete skillReferenceMgr;
	delete metricSender;
	if (envSet) {
		// Wait for any in-flight async arena steps before tearing down the envs.
		envSet->Sync();
		delete envSet;
	}
	delete returnStat;
	delete obsStat;

#ifdef RG_CUDA_SUPPORT
	// Release the CUDA caching allocator's reserved pool so system memory
	// returns to baseline after training ends (or between queue stages).
	if (torch::cuda::is_available())
		c10::cuda::CUDACachingAllocator::emptyCache();
#endif
}
