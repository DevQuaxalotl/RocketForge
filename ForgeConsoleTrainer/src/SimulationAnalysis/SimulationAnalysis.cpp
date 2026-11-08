#include "SimulationAnalysis.h"

#include "Cli/CliResolve.h"

#include "AgentSetup.h"
#include "Inference.h"
#include "Options.h"
#include "SimulationState.h"
#include "SnapshotReplay.h"
#include "SummaryAnalysis.h"
#include "TraceOutput.h"

#include <ForgeTraining.h>
#include <GigaLearnInfer/AgentRegistry.h>
#include <GigaLearnInfer/AgentSetupSnapshot.h>
#include <GigaLearnInfer/FrameworkTorch.h>
#include <GigaLearnInfer/ModelManager.h>
#include <GigaLearnInfer/Models.h>
#include <TrainingStage.h>

#include <RLGymCPP/ActionParsers/ActionParser.h>
#include <RLGymCPP/Gamestates/GameState.h>
#include <RLGymCPP/ObsBuilders/ObsBuilder.h>
#include <RLGymCPP/StateSetters/StateSetter.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace RocketForge;
using namespace RocketForge::SimulationAnalysis;
using namespace RocketSim;
using namespace RLGC;

namespace
{
	std::vector<std::string> SplitCommaRow(const std::string& row)
	{
		std::vector<std::string> fields;
		size_t start = 0;
		while (start <= row.size())
		{
			const size_t end = row.find(',', start);
			fields.push_back(end == std::string::npos ? row.substr(start) : row.substr(start, end - start));
			if (end == std::string::npos)
				break;
			start = end + 1;
		}
		return fields;
	}

	std::map<int, int> LoadActionSequence(const std::filesystem::path& csvPath)
	{
		std::ifstream in(csvPath);
		if (!in.good())
			throw std::runtime_error("failed to open action sequence CSV: " + csvPath.string());

		std::string headerLine;
		if (!std::getline(in, headerLine))
			throw std::runtime_error("action sequence CSV is empty: " + csvPath.string());

		const std::vector<std::string> header = SplitCommaRow(headerLine);
		std::unordered_map<std::string, int> columns;
		for (int i = 0; i < (int)header.size(); i++)
			columns[header[i]] = i;

		auto getColumn = [&](const std::string& name) {
			const auto it = columns.find(name);
			if (it == columns.end())
				throw std::runtime_error("action sequence CSV missing column: " + name);
			return it->second;
		};

		const int tickCol = getColumn("physics_tick");
		const int inferCol = getColumn("infer_action");
		const int actionCol = getColumn("selected_action_id");

		std::map<int, int> sequence;
		std::string rowLine;
		while (std::getline(in, rowLine))
		{
			if (rowLine.empty())
				continue;

			const std::vector<std::string> row = SplitCommaRow(rowLine);
			if ((int)row.size() <= std::max({ tickCol, inferCol, actionCol }))
				continue;
			if (row[inferCol] != "1")
				continue;

			sequence[std::stoi(row[tickCol])] = std::stoi(row[actionCol]);
		}
		if (sequence.empty())
			throw std::runtime_error("action sequence CSV did not contain inference rows");
		return sequence;
	}

	Action ControlReplayAction(const ControlReplayFrame& frame, bool useLastInput)
	{
		return useLastInput ? frame.rlbotLastInput : frame.appliedAction;
	}

	GameState BuildPostStepState(
		const GameState& preStepState,
		Arena* arena,
		const std::vector<Action>& actions)
	{
		GameState previousState = preStepState;
		GameState postStepState = preStepState;
		postStepState.ResetBeforeStep();
		postStepState.UpdateFromArena(arena, actions, &previousState);

		// The snapshot owns all serialized values; do not retain pointers to this
		// helper's temporary previous-state copy.
		postStepState.prev = nullptr;
		for (Player& player : postStepState.players)
			player.prev = nullptr;
		return postStepState;
	}

	void ValidatePolicyReplaySeed(
		const SnapshotReplaySeed& seed,
		int tickSkip,
		int actionDelay,
		const char* mode)
	{
		const ControlReplayFrame& frame = seed.sourceFrame;
		if (!frame.inferAction || frame.policyTick != 0)
			throw std::runtime_error(std::string(mode) +
				" requires a snapshot row with infer_action=1 and policy_tick=0");
		if (frame.tickSkip != tickSkip || frame.actionDelay != actionDelay)
			throw std::runtime_error(std::string(mode) +
				" snapshot scheduler does not match the loaded agent setup (trace tick_skip=" +
				std::to_string(frame.tickSkip) + ", action_delay=" + std::to_string(frame.actionDelay) +
				"; setup tick_skip=" + std::to_string(tickSkip) + ", action_delay=" +
				std::to_string(actionDelay) + ")");
	}
}

int RocketForge::Cli::RunActionTraceCommand(const CliArgs& args)
{
	const SimulationAnalysisOptions options = ParseSimulationAnalysisOptions(args);

	std::vector<std::pair<std::string, std::string>> resolved = { { "agent", options.agentName } };
	if (!options.segmentId.empty()) {
		resolved.push_back({ "build", options.buildName });
		resolved.push_back({ "segment", options.segmentId });
	}
	resolved.push_back({ "setup", options.setupPath.string() });
	resolved.push_back({ "checkpoint", options.checkpointPath.string() });
	PrintResolved(std::cout, resolved);

	if (!std::filesystem::exists(options.setupPath))
		throw std::runtime_error("setup path does not exist: " + options.setupPath.string());
	if (!std::filesystem::exists(options.checkpointPath))
		throw std::runtime_error("checkpoint path does not exist: " + options.checkpointPath.string());

	if (!options.outPath.parent_path().empty())
		std::filesystem::create_directories(options.outPath.parent_path());

	std::unique_ptr<Agent> agent = AgentRegistry::Create(options.agentName);
	if (!agent)
		throw std::runtime_error("unknown agent: " + options.agentName);

	// Load the exact saved component names and network shape before creating
	// the obs builder, action parser, and policy model.
	AgentSetupSnapshot savedSetup = LoadAgentSetup(options.setupPath);
	ApplyAgentSetupSnapshot(*agent, savedSetup);

	std::unique_ptr<ObsBuilder> obsBuilder(agent->MakeObsBuilder());
	std::unique_ptr<ActionParser> actionParser(agent->MakeActionParser());
	if (!obsBuilder)
		throw std::runtime_error("failed to create obs builder");
	if (!actionParser)
		throw std::runtime_error("failed to create action parser");

	const int obsSize = agent->GetObsSize();
	const int numActions = actionParser->GetActionAmount();
	if (options.forcedActionId >= numActions)
		throw std::runtime_error("--force-action is outside the action space");
	std::map<int, int> actionSequence;
	if (!options.actionSequenceCsvPath.empty())
	{
		actionSequence = LoadActionSequence(options.actionSequenceCsvPath);
		for (const auto& [tick, actionId] : actionSequence)
		{
			if (actionId < 0 || actionId >= numActions)
				throw std::runtime_error("action sequence contains an action outside the action space");
		}
	}
	GGL::ModelSet models;
	GGL::ModelManager::MakeModels(
		false,
		obsSize,
		numActions,
		agent->sharedHead,
		agent->policy,
		{},
		torch::kCPU,
		models);
	models.Load(options.checkpointPath, false);

	std::optional<SnapshotReplaySeed> snapshotSeed;
	if (!options.snapshotCsvPath.empty())
		snapshotSeed = LoadSnapshotReplaySeed(
			options.snapshotCsvPath,
			options.snapshotTick,
			options.snapshotBotIndex,
			options.snapshotCarId,
			EngineFromVersion(agent->trainingEngineVersion));
	if (snapshotSeed && (options.controlReplayCsvPath.empty() || options.controlReplayFocusPolicy))
		ValidatePolicyReplaySeed(
			*snapshotSeed,
			agent->tickSkip,
			agent->actionDelay,
			options.controlReplayFocusPolicy ? "focus-policy replay" : "model snapshot replay");

	// Generated resets begin with blue car 0 as the traced car. Scenario setters
	// may designate another focus car after each reset.
	std::unique_ptr<Arena> arena;
	Car* car = nullptr;
	if (snapshotSeed)
		arena = BuildArenaFromSnapshot(*snapshotSeed, agent->carConfig, car);
	else
	{
		ArenaConfig arenaConfig = {};
		arenaConfig.engine = EngineFromVersion(agent->trainingEngineVersion);
		arena.reset(Arena::Create(agent->gameMode, arenaConfig));
		car = arena->AddCar(Team::BLUE, agent->carConfig);
		if (!options.stageResetName.empty())
		{
			for (int i = 1; i < options.teamSize; ++i)
				arena->AddCar(Team::BLUE, agent->carConfig);
			for (int i = 0; i < options.teamSize; ++i)
				arena->AddCar(Team::ORANGE, agent->carConfig);
		}
	}

	std::unique_ptr<TrainingStage> stageResetSource;
	std::unique_ptr<StateSetter> stageStateSetter;
	if (!options.stageResetName.empty())
	{
		stageResetSource = RocketForgeTraining::CreateTrainingStage(options.agentName, options.stageResetName, {}, {});
		if (!stageResetSource)
			throw std::runtime_error("failed to create stage for reset: " + options.stageResetName);
		stageStateSetter = stageResetSource->GetStateSetter();
		if (!stageStateSetter)
			throw std::runtime_error("stage returned null state setter: " + options.stageResetName);
	}

	std::optional<ControlReplayTrace> controlReplay;
	if (!options.controlReplayCsvPath.empty())
	{
		const int inputLookaheadTicks = options.controlReplayUseLastInput ? 1 : 0;
		const int metadataStartTick = snapshotSeed->sourcePhysicsTick;
		const int metadataEndTick = metadataStartTick + options.maxTicks;
		const int controlStartTick = metadataStartTick + inputLookaheadTicks - options.controlReplayInputDelayTicks;
		const int controlEndTick = metadataEndTick + inputLookaheadTicks - options.controlReplayInputDelayTicks;
		const int controlReplayStartTick = std::min(metadataStartTick, controlStartTick);
		const int controlReplayEndTick = std::max(metadataEndTick, controlEndTick);
		controlReplay = LoadControlReplayTrace(
			options.controlReplayCsvPath,
			snapshotSeed->tracedCarId,
			controlReplayStartTick,
			controlReplayEndTick - controlReplayStartTick);
	}

	std::ofstream csv(options.outPath);
	if (!csv.good())
		throw std::runtime_error("failed to open CSV output: " + options.outPath.string());
	csv << std::fixed << std::setprecision(6);
	WriteCsvHeader(csv);
	csv.flush();

	std::vector<TraceSample> samples;
	samples.reserve((size_t)options.episodes * (size_t)options.maxTicks);

	std::vector<Action> actionLookup;
	actionLookup.reserve((size_t)numActions);
	GameState emptyState;
	Player emptyPlayer;
	for (int actionId = 0; actionId < numActions; ++actionId)
		actionLookup.push_back(actionParser->ParseAction(actionId, emptyPlayer, emptyState));

	// Rollouts are stepped manually instead of using EnvSet so every physics
	// tick can be written to the trace.
	for (int episode = 0; episode < options.episodes; ++episode)
	{
		if (snapshotSeed)
			arena = BuildArenaFromSnapshot(*snapshotSeed, agent->carConfig, car);
		else if (stageStateSetter)
		{
			ResetStageProbe(*stageStateSetter, arena.get());
			if (stageStateSetter->trainingFocusCarId >= 0) {
				for (Car* candidate : arena->_cars) {
					if ((int)candidate->id == stageStateSetter->trainingFocusCarId) {
						car = candidate;
						break;
					}
				}
			}
		}
		else if (options.wallProbeMode)
			ResetWallProbe(arena.get(), car, episode, options.seed);
		else
			ResetKickoffProbe(arena.get(), car, episode, options.seed);

		actionParser->ResetRuntimeActionOverride();
		GameState state = snapshotSeed ? snapshotSeed->state : GameState(arena.get());
		state.lastArena = arena.get();
		GameState prevFullState;
		obsBuilder->Reset(state);

		Action appliedAction = snapshotSeed ? snapshotSeed->sourceFrame.appliedAction : Action{};
		int appliedActionId = snapshotSeed ? snapshotSeed->sourceFrame.appliedActionId : -1;
		int appliedBaseActionId = snapshotSeed ? snapshotSeed->sourceFrame.appliedBaseActionId : -1;
		int macroSegmentIndex = snapshotSeed ? snapshotSeed->sourceFrame.macroSegmentIndex : -1;
		int macroSegmentTick = snapshotSeed ? snapshotSeed->sourceFrame.macroSegmentTick : -1;
		int macroSegmentTicks = snapshotSeed ? snapshotSeed->sourceFrame.macroSegmentTicks : -1;
		int macroTicksLeft = snapshotSeed ? snapshotSeed->sourceFrame.macroTicksLeft : 0;
		bool lastAppliedJump = snapshotSeed
			? (snapshotSeed->hasPreviousAppliedAction
				? snapshotSeed->previousAppliedAction.jump > 0.5f
				: snapshotSeed->sourceFrame.rlbotLastInput.jump > 0.5f)
			: false;
		int physicsTick = snapshotSeed ? snapshotSeed->sourcePhysicsTick : 0;
		const int endPhysicsTick = physicsTick + options.maxTicks;
		int policyStep = snapshotSeed ? snapshotSeed->sourcePolicyStep : 0;
		std::unordered_map<uint32_t, ControlReplayFrame> lastMetadataReplayFrames;
		std::unordered_map<uint32_t, ControlReplayFrame> lastControlReplayFrames;
		std::optional<InferenceTrace> controlReplayInference;
		Action controlReplayModelSelectedAction = {};
		Action focusPolicyPrevAction = snapshotSeed ? snapshotSeed->sourceFrame.prevAction : Action{};
		Action pendingFocusPolicyAction = {};
		int pendingFocusPolicyActionId = -1;
		bool hasPendingFocusPolicyAction = false;
		int heldControlReplayTicks = 0;
		int heldMetadataTicks = 0;

		while (physicsTick < endPhysicsTick)
		{
			if (controlReplay)
			{
				const int inputLookaheadTicks = options.controlReplayUseLastInput ? 1 : 0;
				const int controlSourceTick = physicsTick + inputLookaheadTicks - options.controlReplayInputDelayTicks;
				const auto metadataTickIt = controlReplay->framesByTick.find(physicsTick);
				const auto controlTickIt = controlReplay->framesByTick.find(controlSourceTick);
				const ControlReplayFrame* metadataFrame = nullptr;
				const ControlReplayFrame* appliedFrame = nullptr;

				if (metadataTickIt != controlReplay->framesByTick.end())
				{
					for (const auto& [metadataCarId, frame] : metadataTickIt->second)
						lastMetadataReplayFrames[metadataCarId] = frame;
					const auto metadataIt = metadataTickIt->second.find(car->id);
					if (metadataIt != metadataTickIt->second.end())
						metadataFrame = &metadataIt->second;
				}

				if (controlTickIt != controlReplay->framesByTick.end())
				{
					for (Car* arenaCar : arena->_cars)
					{
						const auto frameIt = controlTickIt->second.find(arenaCar->id);
						if (frameIt == controlTickIt->second.end())
							continue;

						if (!options.controlReplayFocusPolicy || arenaCar != car)
							arenaCar->controls = (CarControls)ControlReplayAction(frameIt->second, options.controlReplayUseLastInput);
						lastControlReplayFrames[arenaCar->id] = frameIt->second;
						if (arenaCar == car)
							appliedFrame = &frameIt->second;
					}
				}

				if (!appliedFrame)
				{
					const auto lastFrameIt = lastControlReplayFrames.find(car->id);
					if (lastFrameIt == lastControlReplayFrames.end())
						throw std::runtime_error("control replay is missing the starting row for traced car_id " + std::to_string(car->id));
					appliedFrame = &lastFrameIt->second;
					heldControlReplayTicks++;
				}
				ControlReplayFrame heldMetadataFrame = {};
				if (!metadataFrame)
				{
					const auto lastMetadataIt = lastMetadataReplayFrames.find(car->id);
					if (lastMetadataIt == lastMetadataReplayFrames.end())
						throw std::runtime_error(
							"control replay is missing the starting metadata row for traced car_id " +
							std::to_string(car->id));
					heldMetadataFrame = lastMetadataIt->second;
					heldMetadataFrame.secondsElapsed +=
						(float)(physicsTick - heldMetadataFrame.physicsTick) / 120.f;
					heldMetadataFrame.physicsTick = physicsTick;
					heldMetadataFrame.inferAction = false;
					heldMetadataFrame.applyQueuedAction = false;
					heldMetadataFrame.ticksElapsed = 1;
					heldMetadataFrame.deltaTime = 1.f / 120.f;
					metadataFrame = &heldMetadataFrame;
					heldMetadataTicks++;
				}

				for (Car* arenaCar : arena->_cars)
				{
					if (options.controlReplayFocusPolicy && arenaCar == car)
						continue;
					if (controlTickIt != controlReplay->framesByTick.end() &&
						controlTickIt->second.find(arenaCar->id) != controlTickIt->second.end())
						continue;

					const auto lastFrameIt = lastControlReplayFrames.find(arenaCar->id);
					if (lastFrameIt != lastControlReplayFrames.end())
						arenaCar->controls = (CarControls)ControlReplayAction(lastFrameIt->second, options.controlReplayUseLastInput);
				}
				const ControlReplayFrame& outputFrame = *metadataFrame;
				if (outputFrame.applyQueuedAction && hasPendingFocusPolicyAction)
				{
					appliedAction = pendingFocusPolicyAction;
					appliedActionId = pendingFocusPolicyActionId;
					appliedBaseActionId = FindActionIdForControls(actionLookup, appliedAction);
					macroSegmentIndex = 0;
					macroSegmentTick = 0;
					macroSegmentTicks = agent->tickSkip - agent->actionDelay;
					macroTicksLeft = macroSegmentTicks;
					hasPendingFocusPolicyAction = false;
				}

				GameState inferenceState = state;
				for (Player& inferencePlayer : inferenceState.players)
				{
					if (options.controlReplayFocusPolicy && inferencePlayer.carId == car->id)
					{
						inferencePlayer.lastControls = (CarControls)appliedAction;
						inferencePlayer.prevAction = focusPolicyPrevAction;
						continue;
					}
					const auto lastFrameIt = lastMetadataReplayFrames.find(inferencePlayer.carId);
					if (lastFrameIt == lastMetadataReplayFrames.end())
						continue;

					inferencePlayer.lastControls = (CarControls)lastFrameIt->second.appliedAction;
					inferencePlayer.prevAction = lastFrameIt->second.prevAction;
				}

				const Player& player = FindPlayerByCarIdOrThrow(inferenceState, car->id);
				if (outputFrame.inferAction || !controlReplayInference)
				{
					controlReplayInference = InferActionTrace(
						models,
						*obsBuilder,
						*actionParser,
						player,
						inferenceState,
						obsSize,
						numActions,
						options.deterministic,
						options.temperature);
					controlReplayModelSelectedAction = actionParser->ParseAction(
						controlReplayInference->actionId,
						player,
						inferenceState);
					pendingFocusPolicyAction = controlReplayModelSelectedAction;
					pendingFocusPolicyActionId = controlReplayInference->actionId;
					hasPendingFocusPolicyAction = true;
					if (options.controlReplayFocusPolicy)
						focusPolicyPrevAction = controlReplayModelSelectedAction;
					if (agent->actionDelay == 0)
					{
						appliedAction = pendingFocusPolicyAction;
						appliedActionId = pendingFocusPolicyActionId;
						appliedBaseActionId = FindActionIdForControls(actionLookup, appliedAction);
						hasPendingFocusPolicyAction = false;
					}
				}
				const Action physicsAction = options.controlReplayFocusPolicy
					? appliedAction
					: ControlReplayAction(*appliedFrame, options.controlReplayUseLastInput);
				if (options.controlReplayFocusPolicy)
					car->controls = (CarControls)physicsAction;

				prevFullState = state;
				state.ResetBeforeStep();
				for (Car* arenaCar : arena->_cars)
					arenaCar->ResetStepContactState();

				const bool appliedJump = physicsAction.jump > 0.5f;
				const bool jumpEdge = appliedJump && !lastAppliedJump;
				lastAppliedJump = appliedJump;

				arena->Step(1);

				const CarState carState = car->GetState();
				const bool ballTouchedTick = carState.ballHitInfo.tickCountWhenHit == arena->tickCount - 1;
				std::vector<Action> fullStepActions;
				fullStepActions.reserve(arena->_cars.size());
				for (Car* arenaCar : arena->_cars)
				{
					const auto lastFrameIt = lastControlReplayFrames.find(arenaCar->id);
					fullStepActions.push_back(options.controlReplayFocusPolicy && arenaCar == car
						? physicsAction
						: lastFrameIt != lastControlReplayFrames.end()
						? ControlReplayAction(lastFrameIt->second, options.controlReplayUseLastInput)
						: Action{});
				}
				GameState postStepState = BuildPostStepState(prevFullState, arena.get(), fullStepActions);
				TraceSample sample = MakeTraceSample(
					episode,
					car->id,
					physicsTick,
					outputFrame.policyStep,
					outputFrame.policyTick,
					outputFrame.tickSkip,
					outputFrame.actionDelay,
					outputFrame.selectedActionId,
					outputFrame.selectedIsMacro,
					outputFrame.selectedMacroIndex,
					options.controlReplayFocusPolicy ? appliedActionId : appliedFrame->appliedActionId,
					options.controlReplayFocusPolicy ? appliedBaseActionId : appliedFrame->appliedBaseActionId,
					options.controlReplayFocusPolicy ? macroSegmentIndex : appliedFrame->macroSegmentIndex,
					options.controlReplayFocusPolicy ? macroSegmentTick : appliedFrame->macroSegmentTick,
					options.controlReplayFocusPolicy ? macroSegmentTicks : appliedFrame->macroSegmentTicks,
					options.controlReplayFocusPolicy ? macroTicksLeft : appliedFrame->macroTicksLeft,
					outputFrame.macroPlan,
					controlReplayInference->actionMask,
					controlReplayInference->obs,
					player,
					outputFrame.selectedAction,
					physicsAction,
					options.controlReplayFocusPolicy ? player.prevAction : appliedFrame->prevAction,
					carState,
					prevFullState,
					jumpEdge,
					ballTouchedTick);
				sample.source = options.controlReplayFocusPolicy
					? "rocketsim_focus_policy_replay"
					: "rocketsim_control_replay";
				sample.botIndex = outputFrame.botIndex;
				sample.botTeam = outputFrame.botTeam;
				sample.inferAction = outputFrame.inferAction;
				sample.applyQueuedAction = outputFrame.applyQueuedAction;
				sample.ticksElapsed = outputFrame.ticksElapsed;
				sample.secondsElapsed = outputFrame.secondsElapsed;
				sample.gameTimeRemaining = outputFrame.gameTimeRemaining;
				sample.deltaTime = outputFrame.deltaTime;
				sample.frameNum = outputFrame.frameNum;
				sample.rlbotLastInput = outputFrame.rlbotLastInput;
				sample.modelSelectedActionId = controlReplayInference->actionId;
				sample.modelSelectedAction = controlReplayModelSelectedAction;
				sample.modelAppliedActionId = appliedActionId;
				sample.modelAppliedAction = appliedAction;
				sample.postStateSnapshot = SerializeGameStateSnapshot(postStepState);
				WriteCsvRow(csv, sample);
				samples.push_back(sample);

				state = std::move(postStepState);
				physicsTick++;
				if (heldControlReplayTicks > 0 && physicsTick == endPhysicsTick)
					std::cout << "control replay held previous controls across "
						<< heldControlReplayTicks << " missing traced-car tick(s).\n";
				if (heldMetadataTicks > 0 && physicsTick == endPhysicsTick)
					std::cout << "control replay held scheduler metadata across "
						<< heldMetadataTicks << " missing traced-car tick(s).\n";
				continue;
			}

			const Player& player = FindPlayerByCarIdOrThrow(state, car->id);
			const int actionTicks = agent->tickSkip - agent->actionDelay;
			InferenceTrace inference = InferActionTrace(
				models,
				*obsBuilder,
				*actionParser,
				player,
				state,
				obsSize,
				numActions,
				options.deterministic,
				options.temperature);
			int selectedActionId = inference.actionId;
			if (options.forcedActionId >= 0)
				selectedActionId = options.forcedActionId;
			else if (!actionSequence.empty())
			{
				const auto sequenceIt = actionSequence.find(physicsTick);
				if (sequenceIt == actionSequence.end())
					throw std::runtime_error("action sequence is missing physics_tick " + std::to_string(physicsTick));
				selectedActionId = sequenceIt->second;
			}
			const Action modelSelectedAction = actionParser->ParseAction(inference.actionId, player, state);
			const Action selectedAction = actionParser->ParseAction(selectedActionId, player, state);
			if (agent->actionDelay != 0 && actionParser->IsMacroAction(selectedActionId, player, state))
				throw std::runtime_error("action macros require actionDelay == 0");
			std::vector<ActionMacro> selectedMacros;
			actionParser->ParseActionMacros(selectedActionId, player, state, actionTicks, selectedMacros);
			if (selectedMacros.empty())
				selectedMacros.push_back({ selectedAction, actionTicks });
			const bool selectedIsMacro = selectedMacros.size() != 1 || selectedMacros.front().ticks != actionTicks;
			const int selectedMacroIndex = selectedIsMacro ? selectedActionId : -1;
			const std::string selectedMacroPlan = BuildMacroPlanString(selectedMacros, actionLookup);
			const Action prevActionForObs = player.prevAction;
			const Player maskPlayer = player;

			int selectedMacroTicks = 0;
			for (const ActionMacro& macro : selectedMacros)
			{
				if (macro.ticks < 0)
					throw std::runtime_error("action parser returned a macro with negative ticks");
				selectedMacroTicks += macro.ticks;
			}
			if (selectedMacroTicks != actionTicks)
				throw std::runtime_error("action parser returned macros that do not match the action window");
			prevFullState = state;
			GameState traceStepState = prevFullState;
			state.ResetBeforeStep();
			for (Car* arenaCar : arena->_cars)
				arenaCar->ResetStepContactState();

			// Step one physics tick at a time so the CSV can reconstruct the
			// exact controller stream, including actionDelay and action macros.
			auto stepAndLog = [&](int policyTick) {
				const bool appliedJump = appliedAction.jump > 0.5f;
				const bool jumpEdge = appliedJump && !lastAppliedJump;
				lastAppliedJump = appliedJump;

				for (Car* arenaCar : arena->_cars)
					arenaCar->controls = arenaCar == car ? (CarControls)appliedAction : CarControls{};
				arena->Step(1);

				const CarState carState = car->GetState();
				const bool ballTouchedTick = carState.ballHitInfo.tickCountWhenHit == arena->tickCount - 1;
				const std::vector<Action> tickActions = BuildFullStepActions(arena.get(), car, appliedAction);
				traceStepState = BuildPostStepState(traceStepState, arena.get(), tickActions);
				TraceSample sample = MakeTraceSample(
					episode,
					car->id,
					physicsTick,
					policyStep,
					policyTick,
					agent->tickSkip,
					agent->actionDelay,
					selectedActionId,
					selectedIsMacro,
					selectedMacroIndex,
					appliedActionId,
					appliedBaseActionId,
					macroSegmentIndex,
					macroSegmentTick,
					macroSegmentTicks,
					macroTicksLeft,
					selectedMacroPlan,
					inference.actionMask,
					inference.obs,
					maskPlayer,
					selectedAction,
					appliedAction,
					prevActionForObs,
					carState,
					prevFullState,
					jumpEdge,
					ballTouchedTick);
				sample.modelSelectedActionId = inference.actionId;
				sample.modelSelectedAction = modelSelectedAction;
				sample.postStateSnapshot = SerializeGameStateSnapshot(traceStepState);
				WriteCsvRow(csv, sample);
				samples.push_back(sample);
				physicsTick++;
			};

			for (int i = 0; i < agent->actionDelay && physicsTick < endPhysicsTick; ++i)
				stepAndLog(i);

			appliedActionId = selectedActionId;
			int policyTick = agent->actionDelay;
			for (int segmentIdx = 0; segmentIdx < (int)selectedMacros.size(); segmentIdx++)
			{
				const ActionMacro& macro = selectedMacros[segmentIdx];
				appliedAction = macro.action;
				appliedBaseActionId = FindActionIdForControls(actionLookup, appliedAction);
				macroSegmentIndex = segmentIdx;
				macroSegmentTicks = macro.ticks;
				for (int i = 0; i < macro.ticks && physicsTick < endPhysicsTick; ++i)
				{
					macroSegmentTick = i;
					macroTicksLeft = macro.ticks - i;
					stepAndLog(policyTick++);
				}
			}

			std::vector<Action> fullStepActions = BuildFullStepActions(arena.get(), car, selectedMacros.back().action);
			state.UpdateFromArena(arena.get(), fullStepActions, &prevFullState);
			policyStep++;
		}
	}

	const std::filesystem::path summaryPath = SummaryPathForCsv(options.outPath);
	const std::vector<WallDashPairStats> wallDashPairSummary = BuildWallDashPairSummary(samples);
	const std::vector<SequenceStats> speedFlip4TickSummary = BuildSpeedFlipSummary(samples, 4, 4);
	const std::vector<SequenceStats> speedFlip2TickSummary = BuildSpeedFlipSummary(samples, 2, 4);
	WriteSummary(summaryPath, options, *agent, samples, wallDashPairSummary, speedFlip4TickSummary, speedFlip2TickSummary, actionLookup);
	models.Free();

	std::cout << TraceModeName(options) << " trace complete.\n"
		<< "  CSV: " << options.outPath << "\n"
		<< "  Summary: " << summaryPath << "\n"
		<< "  Samples: " << samples.size() << "\n"
		<< "  Wall-dash candidate pairs: " << wallDashPairSummary.size() << "\n"
		<< "  Speed-flip 4-tick sequences: " << speedFlip4TickSummary.size() << "\n"
		<< "  Speed-flip 2-tick sequences: " << speedFlip2TickSummary.size() << "\n";
	return 0;
}
