#include "ForgeTraining.h"
#include "ForgeTrainingInternal.h"
#include "StageQueue/StageQueueRunner.h"
#include "StageQueue/StageQueueSignals.h"
#include "StageRegistry.h"
#include "StaticOpponents/SkillReferenceSelection.h"
#include "TrainingStage.h"

#include <RocketSim/Engine.h>
#include <RLGymCPP/ThreadPool.h>
#include <GigaLearnCPP/Learner.h>
#include <GigaLearnInfer/AgentRegistry.h>
#include <GigaLearnInfer/AgentSetupSnapshot.h>

#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <pybind11/embed.h>

using namespace GGL; // GigaLearn
using namespace RLGC; // RLGymCPP
using namespace RocketForge;

// Protect the learner lifetime while external controls request a cooperative stop.
static std::mutex s_activeLearnerMutex;
static GGL::Learner* s_activeLearner = nullptr;
static std::once_flag s_trainingInitOnce;

std::unique_ptr<TrainingStage> RocketForgeTraining::CreateTrainingStage(
	const std::string& agentName,
	const std::string& stageName,
	const GlobalSettings& globals,
	const StageSettings& stage)
{
	std::unique_ptr<Agent> agent = AgentRegistry::Create(agentName);
	if (!agent)
		return nullptr;

	if (stage.agentSetupOverride.has_value())
		ApplyAgentSetupSnapshot(*agent, stage.agentSetupOverride.value());

	std::unique_ptr<TrainingStage> newStage = StageRegistry::Create(stageName, std::move(agent));
	if (!newStage)
		return nullptr;

	newStage->SetParameterOverrides(stage.parameterOverrides);
	newStage->learnerConfig.sendMetrics = globals.metricsEnabled;
	newStage->learnerConfig.warmStartPolicy = stage.warmStartPolicy;
	newStage->learnerConfig.warmStartIterations = stage.warmStartIterations;
	newStage->learnerConfig.trainAgainstStaticOpponents = stage.staticOpponentsEnabled;
	newStage->learnerConfig.runtime.staticOpponentAgentFactory = [](const std::string& sourceAgentName) {
		return AgentRegistry::Create(sourceAgentName);
	};
	newStage->learnerConfig.runtime.stopAtTotalTimesteps = stage.targetTotalTimesteps;

	// Reapplied so parameter overrides win over the queue entry's settings above.
	newStage->ApplyParameterOverrides();

	if (!globals.skillTestingEnabled)
		newStage->learnerConfig.skillTracker.enabled = false;

	return newStage;
}

void RocketForgeTraining::InitializeTraining()
{
	std::call_once(s_trainingInitOnce, []()
	{
#ifdef _WIN32
		// Windows needs an explicit embedded-Python stdlib root; Linux/Docker uses Python's compiled-in paths.
		std::filesystem::path pyPath = PY_LIB_PATH;
		static const std::wstring pyPathWide = pyPath.wstring();
		Py_SetPythonHome(pyPathWide.c_str());
#endif

		// Keep the embedded interpreter alive for the whole process lifetime.
		// Restarting it between runs breaks packages like wandb/pydantic-core.
		pybind11::initialize_interpreter();

		// Allow Docker/Linux builds to import python_scripts.metric_receiver from the build tree.
		pybind11::module_::import("sys")
			.attr("path")
			.attr("insert")(0, std::string(GIGALEARN_BUILD_DIR));

		// Release the GIL for the remainder of the process so the embedded interpreter can be driven from any worker thread.
		static pybind11::gil_scoped_release* s_gilRelease = new pybind11::gil_scoped_release();
		(void)s_gilRelease;

		RocketSim::Init(RocketSim::CollisionMeshesFolder());
	});
}

static void RunLearner(const RocketForge::TrainingStage* stage)
{
	for (RocketSim::EngineId engine : RocketForge::RequiredEngines(RocketForge::CaptureAgentSetupSnapshot(*stage->agent)))
		RocketSim::RequireEngine(engine);

	RG_LOG("Training simulator: " << RocketSim::GetEngineInfo(stage->learnerConfig.engine).displayName);
	RG_LOG("Ball predictor: RocketSim v" << stage->agent->GetBallPredictionEngineVersion());

	RocketForgeTraining::InitializeTraining();

	std::unique_ptr<GGL::Learner> learner = std::make_unique<GGL::Learner>(
		stage->environmentCreateFunc,
		stage->learnerConfig,
		stage->stepCallbackFunc);

	learner->onRemoteStopRequested = &RocketForgeTraining::RequestStop;

	{
		std::lock_guard lock(s_activeLearnerMutex);
		s_activeLearner = learner.get();

		// A stop or skip raised while the stage was being set up had no learner to reach.
		if (RocketForgeTraining::StageQueueSignalState::IsAnyRaised())
			learner->RequestStop();
	}

	try {
		learner->Start();
	} catch (...) {
		std::lock_guard lock(s_activeLearnerMutex);
		s_activeLearner = nullptr;
		throw;
	}

	{
		std::lock_guard lock(s_activeLearnerMutex);
		s_activeLearner = nullptr;
	}
}

static void SyncStaticOpponentsIfEnabled(const RocketForge::TrainingStage* stage)
{
	if (!stage || !stage->agent)
		return;

	if (!stage->learnerConfig.trainAgainstStaticOpponents)
		return;

	const std::string buildName = stage->learnerConfig.runtime.buildName.empty() ? GGL::DEFAULT_BUILD_NAME : stage->learnerConfig.runtime.buildName;
	RocketForgeTraining::SyncStaticOpponentSelection(stage->agent->agentName, buildName);
}

static void SyncSkillReferencesIfEnabled(const RocketForge::TrainingStage* stage)
{
	if (!stage || !stage->agent)
		return;

	if (!stage->learnerConfig.skillTracker.enabled)
		return;

	const std::string buildName = stage->learnerConfig.runtime.buildName.empty() ? GGL::DEFAULT_BUILD_NAME : stage->learnerConfig.runtime.buildName;
	RocketForgeTraining::SyncSkillReferenceSelection(stage->agent->agentName, buildName);
}

void RocketForgeTraining::RequestStop()
{
	StageQueueSignalState::RaiseStop();

	std::lock_guard lock(s_activeLearnerMutex);
	GGL::Learner* learner = s_activeLearner;
	if (learner)
		learner->RequestStop();
}

void RocketForgeTraining::LaunchContinueTraining(
	RocketForge::TrainingStage* stageToTrain,
	const RocketForge::TrainingStage* sourceStage,
	bool startFromScratch,
	bool pruneSourceCheckpoints,
	bool confirmedDelete)
{
	(void)sourceStage;
	(void)startFromScratch;
	(void)pruneSourceCheckpoints;
	(void)confirmedDelete;

	if (!stageToTrain)
		return;

	TrainingStage* mutableStage = stageToTrain;

	if (mutableStage->learnerConfig.runtime.segmentId.empty())
		RG_ERR_CLOSE("ContinueTraining requires a configured training segment.");

	mutableStage->learnerConfig.runtime.resetRunIdOnLoad = false;
	SyncStaticOpponentsIfEnabled(mutableStage);
	SyncSkillReferencesIfEnabled(mutableStage);
	RunLearner(mutableStage);
}

void RocketForgeTraining::LaunchNewStageTraining(
	RocketForge::TrainingStage* newStage,
	const RocketForge::TrainingStage* sourceStage,
	bool clearRunId,
	bool pruneSourceCheckpoints,
	bool confirmedDelete)
{
	(void)sourceStage;
	(void)pruneSourceCheckpoints;
	(void)confirmedDelete;

	TrainingStage* mutableNewStage = newStage;
	if (!mutableNewStage)
		return;

	if (mutableNewStage->learnerConfig.runtime.segmentId.empty())
		RG_ERR_CLOSE("StartNewStage requires a configured training segment.");

	mutableNewStage->learnerConfig.runtime.resetRunIdOnLoad = clearRunId;
	SyncStaticOpponentsIfEnabled(mutableNewStage);
	SyncSkillReferencesIfEnabled(mutableNewStage);
	RG_LOG("StartNewStage: Using preconfigured segment checkpoint root " << mutableNewStage->learnerConfig.runtime.checkpointFolder);
	RunLearner(mutableNewStage);
}

RocketForgeTraining::RunResult RocketForgeTraining::RunTraining(
	ForgeSettings& settings,
	const std::filesystem::path& settingsPath,
	const StageEvents& events)
{
	return RunTraining(settings, RunPersistence{ settingsPath }, events);
}

RocketForgeTraining::RunResult RocketForgeTraining::RunTraining(
	ForgeSettings& settings,
	const RunPersistence& persistence,
	const StageEvents& events)
{
	// The restriction is one-way for the process, so it is applied once, before the first stage sizes the thread pool.
	static std::once_flag performanceCoresOnce;
	if (settings.globals.performanceCoresOnly)
		std::call_once(performanceCoresOnce, [] { RLGC::RestrictProcessToPerformanceCores(); });

	return RunStageQueue(settings, persistence, events);
}

void RocketForgeTraining::RequestSkipStage()
{
	StageQueueSignalState::RaiseSkip();

	std::lock_guard lock(s_activeLearnerMutex);
	GGL::Learner* learner = s_activeLearner;
	if (learner)
		learner->RequestStop();
}
