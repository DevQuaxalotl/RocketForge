#include "InferenceDebugRenderLoop.h"
#include <RocketSim/Engine.h>

#include <ForgeTraining.h>
#include <TrainingStage.h>
#include <Inference/InferenceModelSource.h>
#include <Inference/PersistedAdaptiveWeights.h>

#include <GigaLearnInfer/Agent.h>
#include <GigaLearnInfer/InferUnit.h>
#include <GigaLearnCPP/Util/RenderSender.h>

#include <ForgeDebugger.h>

#include <RLGymCPP/EnvSet/EnvSet.h>
#include <RLGymCPP/BasicTypes/Lists.h>

#include <StaticOpponents/StaticOpponentSelection.h>

#include "CriticProbe.h"
#include "DebugRenderFocus.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

using namespace RLGC;

namespace {
	// The pieces EnvSet takes raw pointers to but never frees, owned here so a discarded env doesn't leak them.
	struct OwnedEnvParts {
		std::vector<std::unique_ptr<RLGC::Reward>>            rewards;
		std::vector<std::unique_ptr<RLGC::TerminalCondition>> terminalConditions;
		std::unique_ptr<RLGC::ObsBuilder>                     obsBuilder;
		std::unique_ptr<RLGC::ActionParser>                   actionParser;
		std::unique_ptr<RLGC::StateSetter>                    stateSetter;
		const void*                                           userInfo = nullptr;
	};

	// One fully-loaded static opponent. Bundles the borrowed obs/action builders with the
	// InferUnit so they are freed together and outlive it, mirroring the primary policy's locals.
	struct LoadedStaticOpponent {
		RocketForge::ResolvedInferenceAgent resolved;
		std::unique_ptr<RLGC::ObsBuilder>   obsBuilder;
		std::unique_ptr<RLGC::ActionParser> actionParser;
		std::unique_ptr<GGL::InferUnit>     inferUnit;
		std::string                         displayName;
	};

	LoadedStaticOpponent LoadStaticOpponentInferUnit(
		const RocketForgeTraining::StaticOpponentSelectionEntry& entry, bool useGPU)
	{
		LoadedStaticOpponent loaded;
		const RocketForge::InferenceModelSource source = RocketForge::FromStaticOpponentEntry(entry);
		loaded.resolved = RocketForge::ResolveInferenceAgent(source);
		loaded.obsBuilder.reset(loaded.resolved.agent->MakeConfiguredObsBuilder());
		loaded.actionParser.reset(loaded.resolved.agent->MakeActionParser());
		loaded.inferUnit = RocketForge::MakeInferUnit(
			loaded.resolved, loaded.obsBuilder.get(), loaded.actionParser.get(), useGPU);
		loaded.displayName = entry.displayName.empty()
			? entry.checkpointPath.filename().string()
			: entry.displayName;
		return loaded;
	}
}

namespace RocketForge
{
	std::vector<RocketSim::EngineId> ResolveDebugEngines(const InferenceDebugRenderConfig& config)
	{
		const InferenceModelSource source = config.isImported
			? FromImportedModel(config.agentName, config.checkpointFolder)
			: FromSegmentCheckpoint(config.agentName, config.checkpointFolder);
		std::vector<RocketSim::EngineId> engines = RequiredEngines(CaptureAgentSetupSnapshot(*ResolveInferenceAgent(source).agent));
		if (!config.includeStaticOpponent)
			return engines;

		for (const auto& entry : RocketForgeTraining::LoadStaticOpponentSelection(config.agentName, config.buildName)) {
			const ResolvedInferenceAgent opponent = ResolveInferenceAgent(FromStaticOpponentEntry(entry));
			for (RocketSim::EngineId engine : RequiredEngines(CaptureAgentSetupSnapshot(*opponent.agent)))
				if (std::find(engines.begin(), engines.end(), engine) == engines.end())
					engines.push_back(engine);
		}

		return engines;
	}

	void RunInferenceDebugRender(const InferenceDebugRenderConfig& config)
	{
		for (RocketSim::EngineId engine : ResolveDebugEngines(config))
			RocketSim::RequireEngine(engine);

		if (config.pollControls)
			config.pollControls();

		// Ensure the embedded interpreter + RocketSim are initialized before we build the arena
		// or the RenderSender. Idempotent; the GUI runs this on a dedicated worker thread that
		// never went through the training path that would otherwise have initialized them.
		RocketForgeTraining::InitializeTraining();

		// 1. Build the stage (rewards / team sizes / state setters live here).
		std::unique_ptr<TrainingStage> stage =
			RocketForgeTraining::CreateTrainingStage(config.agentName, config.stageName, {}, {});
		if (!stage)
			throw std::runtime_error("InferenceDebugRender: stage '" + config.stageName
				+ "' is unavailable for agent '" + config.agentName + "'");

		// 2. Resolve an inference-ready agent (snapshot applied for segments; imported uses
		//    INFERENCE_POLICY.json via MakeInferenceModels).
		InferenceModelSource source = config.isImported
			? FromImportedModel(config.agentName, config.checkpointFolder)
			: FromSegmentCheckpoint(config.agentName, config.checkpointFolder);

		if (source.checkpointFolder.empty())
			throw std::runtime_error("InferenceDebugRender: no checkpoint to render for '"
				+ config.agentName + "'");

		ResolvedInferenceAgent resolved = ResolveInferenceAgent(source);

		// 3. Build obs/action + InferUnit from the resolved agent BEFORE moving the agent into the
		//    stage. The InferUnit keeps no Agent reference after construction, so the move is safe;
		//    obsBuilder/actionParser are borrowed by the InferUnit and outlive it (locals here).
		std::unique_ptr<ObsBuilder> obsBuilder(resolved.agent->MakeConfiguredObsBuilder());
		std::unique_ptr<ActionParser> actionParser(resolved.agent->MakeActionParser());

		const bool useGPU = ResolveInferenceUseGPU(config.useGPU);
		std::unique_ptr<GGL::InferUnit> inferUnitPtr =
			MakeInferUnit(resolved, obsBuilder.get(), actionParser.get(), useGPU);
		GGL::InferUnit& inferUnit = *inferUnitPtr;

		// Swap the snapshot-applied agent into the stage so CreateEnvironment builds the env
		// (obs/rewards/team sizes/car config) against the saved-weights setup.
		stage->agent = std::move(resolved.agent);
		const int tickSkip = stage->agent->tickSkip;

		// 4. Single-arena EnvSet from the canonical stage factory. saveRewards populates
		//    state.lastRewards / state.lastTrainingRewards for the collector.
		EnvSetConfig envCfg = {};
		envCfg.numArenas = 1;
		envCfg.tickSkip = tickSkip;
		envCfg.actionDelay = stage->agent->actionDelay;
		envCfg.saveRewards = true;
		envCfg.rewardSamplePlayerOverride = config.playerIndex;
		envCfg.envCreateCfg.numArenas = 1;
		envCfg.envCreateCfg.mode = stage->agent->gameMode;
		envCfg.envCreateCfg.teamSizeOverride = config.teamSizeOverride;
		envCfg.envCreateCfg.engine = RocketSim::EngineFromVersion(stage->agent->trainingEngineVersion);

		TrainingStage* stagePtr = stage.get();
		std::unique_ptr<OwnedEnvParts> createdParts;
		envCfg.envCreateFn = [stagePtr, &createdParts](const int index, const EnvCreateConfig& cfg) {
			// createdParts holds one env's parts, so a second arena would race on it and free the first arena's.
			RG_ASSERT(index == 0);

			EnvCreateResult result = stagePtr->CreateEnvironment(index, cfg);

			createdParts = std::make_unique<OwnedEnvParts>();
			for (WeightedReward& weighted : result.rewards)
				createdParts->rewards.emplace_back(weighted.reward);
			for (TerminalCondition* condition : result.terminalConditions)
				createdParts->terminalConditions.emplace_back(condition);
			createdParts->obsBuilder.reset(result.obsBuilder);
			createdParts->actionParser.reset(result.actionParser);
			createdParts->stateSetter.reset(result.stateSetter);
			createdParts->userInfo = result.userInfo;

			return result;
		};

		// Debug settings feed env creation (forced state setter / team-size override index),
		// mirroring TrainingWorker's stage->SetDebugSettings wiring. Must precede EnvSet, which
		// invokes CreateEnvironment. The pointed-to DebugSettings is owned by the caller.
		if (config.debugSettings)
			stage->SetDebugSettings(config.debugSettings);

		// Declared before envSet so the parts outlive the env that points at them.
		std::unique_ptr<OwnedEnvParts> ownedParts;
		std::unique_ptr<EnvSet> envSet;
		std::unique_ptr<CriticProbe> criticProbe;
		bool criticProbeAttempted = false;
		std::unique_ptr<ForgeDebug::RewardDebugCollector> collector;

		const std::vector<AdaptivePersistEntry> persistedAdaptiveWeights = LoadPersistedAdaptiveWeights(resolved.checkpointFolder);

		// Team size is fixed for an arena's lifetime, so re-rolling it needs a replacement env.
		// Everything that reads the env's reward list (collector, critic probe) is rebound to it.
		auto buildEnv = [&]() {
			std::unique_ptr<EnvSet> newEnv = std::make_unique<EnvSet>(envCfg);

			// The controller only evolves weights under the Learner, so the fresh EnvSet holds the
			// configured starting weights. Show the checkpoint's effective adaptive weights instead.
			int blue = 0, orange = 0;
			for (Car* car : newEnv->arenas[0]->_cars)
				(car->team == Team::BLUE ? blue : orange)++;

			ApplyPersistedAdaptiveWeights(persistedAdaptiveWeights, blue, orange, newEnv->rewards[0]);

			// Only the GUI reward view shows the critic, so console renders (showRewards off) never load it.
			// A failed load or evaluation hides it for the rest of the session rather than retrying each episode.
			if (config.showRewards) {
				if (!criticProbeAttempted) {
					criticProbeAttempted = true;

					CriticProbeLoadConfig probeConfig;
					probeConfig.agent = stage->agent.get();
					probeConfig.checkpointFolder = resolved.checkpointFolder;
					probeConfig.numActions = actionParser->GetActionAmount();
					probeConfig.popArtSigmaFloor = stage->learnerConfig.ppo.decomposedCritic.popArtSigmaFloor;
					probeConfig.rewards = &newEnv->rewards[0];
					probeConfig.useGPU = useGPU;
					criticProbe = CriticProbe::TryLoad(probeConfig);
				} else if (criticProbe) {
					criticProbe->RemapHeads(newEnv->rewards[0]);
				}
			}

			// The critic must see the exact observation the policy acted on, which the InferUnit keeps while capturing diagnostics.
			inferUnit.captureDiagnostics = criticProbe != nullptr;

			if (config.showRewards) {
				if (collector) {
					collector->Rebind(newEnv->rewards[0]);
				} else {
					// Apply the initial debug settings the same way DebugBridge::ApplyPendingSettings maps them onto the collector.
					collector = std::make_unique<ForgeDebug::RewardDebugCollector>();
					collector->Initialize(newEnv->rewards[0]);
					collector->SetPlayerIndex(config.playerIndex);

					if (config.debugSettings) {
						collector->SetNormalizationMode(config.debugSettings->normMode);
						collector->SetRewardDisplayMode(config.debugSettings->displayMode);
						collector->SetForceDebugAll(config.debugSettings->forceDebugAll);
						if (config.debugSettings->hasRewardSelection)
							collector->SetDebugSelection(config.debugSettings->debugEnabledRewards);
					}
				}
			}

			// Retire the old env first, since its components are what the old parts own.
			envSet = std::move(newEnv);
			if (ownedParts)
				stage->ReleaseRewardCache(ownedParts->userInfo);
			ownedParts = std::move(createdParts);
		};

		buildEnv();

		// Only the weighted default re-rolls; an explicit override (fixed or picked live in the GUI) keeps the same arena.
		auto shouldRerollTeamSize = [&]() {
			return config.teamSizeOverride <= 0
				&& stage->GetWeightedTeamSizes().size() > 1
				&& !(config.debugSettings && config.debugSettings->teamSizeOverrideIndex >= 0);
		};

		std::atomic<float> localTimeScale{ DEFAULT_GAME_SPEED };

		std::unique_ptr<GGL::RenderSender> renderSender;
		if (config.renderScene)
			renderSender = std::make_unique<GGL::RenderSender>();

		std::atomic<float>& gameSpeed = config.timeScale ? *config.timeScale : localTimeScale;
		if (renderSender)
			renderSender->SetRuntimeTimeScale(&gameSpeed);

		// Recomputed from live settings before each step below.
		int maxEpisodeLength = 1 << 30;

		const bool deterministic = stage->learnerConfig.ppo.deterministic;
		const float temperature = stage->learnerConfig.ppo.policyTemperature;

		// Optional static opponent. Its team is chosen per episode: opposite the setter's focus car, else ORANGE.
		// An empty selection key re-picks a random allowlist entry each episode;
		// a non-empty key pins one entry, loaded once here.
		std::vector<RocketForgeTraining::StaticOpponentSelectionEntry> staticAllowlist;
		bool staticRandomEachEpisode = false;
		std::optional<LoadedStaticOpponent> opponent;
		std::mt19937 opponentRng{ std::random_device{}() };

		if (config.includeStaticOpponent) {
			staticAllowlist = RocketForgeTraining::LoadStaticOpponentSelection(config.agentName, config.buildName);
			if (staticAllowlist.empty())
				throw std::runtime_error("InferenceDebugRender: static opponent requested but Build '"
					+ config.buildName + "' has no static opponents selected");

			staticRandomEachEpisode = config.staticOpponentSelectionKey.empty();
			if (!staticRandomEachEpisode) {
				const auto pinned = std::find_if(staticAllowlist.begin(), staticAllowlist.end(),
					[&](const RocketForgeTraining::StaticOpponentSelectionEntry& e) {
						return RocketForgeTraining::MakeStaticOpponentSelectionKey(e)
							== config.staticOpponentSelectionKey;
					});
				if (pinned == staticAllowlist.end())
					throw std::runtime_error("InferenceDebugRender: static opponent selection '"
						+ config.staticOpponentSelectionKey + "' is not in the Build's allowlist");
				opponent = LoadStaticOpponentInferUnit(*pinned, useGPU);
			}
		}

		int renderEpisodeStep = 0;
		const bool asyncFirstHalf = true;
		const bool asyncSecondHalf = false;

		int focusIndex = config.playerIndex;
		Team staticTeam = Team::ORANGE;
		bool paused = false;
		bool firstEpisode = true;

		// One owner for the focused player: it drives the viewer camera and reward sampling together.
		auto setFocus = [&](int requestedIndex) {
			const GameState& gs = envSet->state.gameStates[0];
			if (gs.players.empty())
				return;

			focusIndex = std::clamp(requestedIndex, 0, (int)gs.players.size() - 1);
			envSet->config.rewardSamplePlayerOverride = focusIndex;
			if (collector)
				collector->SetPlayerIndex(focusIndex);
		};

		auto applyViewerCommands = [&]() {
			if (!renderSender)
				return;

			for (const std::string& datagram : renderSender->PollCommands()) {
				const std::optional<ViewerCommand> command = ParseViewerCommand(datagram);
				if (!command)
					continue;

				switch (command->type) {
				case ViewerCommand::Type::FocusSet:
					setFocus(command->value);
					break;
				case ViewerCommand::Type::SpeedStep:
					gameSpeed.store(StepGameSpeed(gameSpeed.load(), command->value));
					break;
				case ViewerCommand::Type::SpeedReset:
					gameSpeed.store(DEFAULT_GAME_SPEED);
					break;
				case ViewerCommand::Type::PauseToggle:
					paused = !paused;
					break;
				}
			}
		};

		auto applyGuiSettings = [&]() {
			if (!collector || !config.debugRenderer)
				return;

			const int prevPlayerIdx = collector->GetPlayerIndex();
			config.debugRenderer->ApplyPendingSettings(*collector, envSet->state.gameStates[0]);
			if (collector->GetPlayerIndex() != prevPlayerIdx)
				setFocus(collector->GetPlayerIndex());
		};

		std::string contextLabel;

		// Discard any stale datagram left over from a previous render session on this process-lifetime
		// socket (e.g. a cached pause_toggle), then clamp the starting focus against the freshly-reset arena.
		if (renderSender)
			renderSender->PollCommands();
		setFocus(config.playerIndex);

		while (!config.shouldStop || !config.shouldStop()) {
			if (config.pollControls)
				config.pollControls();

			if (config.consumePlayerIndex) {
				const int index = config.consumePlayerIndex();
				if (index >= 0)
					setFocus(index);
			}

			const float duration = config.debugSettings && config.debugSettings->maxEpisodeTimeEnabled
				? config.debugSettings->maxEpisodeTime
				: stage->learnerConfig.ppo.maxEpisodeDuration;
			maxEpisodeLength = duration > 0 ? std::max(1, int(duration * (120.f / tickSkip))) : 1 << 30;

			applyViewerCommands();

			if (paused) {
				applyGuiSettings();

				const GameState& pausedGs = envSet->state.gameStates[0];
				const GGL::RenderViewState pausedView{ focusIndex, true };

				if (collector && !pausedGs.players.empty() && focusIndex < (int)pausedGs.players.size()) {
					// The collector records nothing while paused, so its snapshot still names the pre-pause player;
					// patch in the current focus so the GUI follows focus changes made while paused.
					ForgeDebug::RewardDebugSnapshot pausedSnapshot = collector->GetSnapshot();

					// The cached estimate belongs to the player the snapshot was recorded for.
					if (pausedSnapshot.playerIndex != focusIndex)
						pausedSnapshot.critic.reset();
					pausedSnapshot.playerIndex = focusIndex;
					pausedSnapshot.playerCarId = (int)pausedGs.players[focusIndex].carId;
					pausedSnapshot.playerTeam = (int)pausedGs.players[focusIndex].team;

					if (renderSender)
						renderSender->Send(pausedGs, &pausedSnapshot, contextLabel, pausedView);
					if (config.debugRenderer)
						config.debugRenderer->Render(pausedSnapshot);
				} else {
					if (renderSender)
						renderSender->Send(pausedGs, nullptr, contextLabel, pausedView);
				}

				continue;
			}

			const bool wasTerminal = envSet->state.terminals[0];
			if (wasTerminal && shouldRerollTeamSize())
				buildEnv();

			envSet->Reset();
			if (collector && wasTerminal) {
				collector->ResetEpisodeTotals();
				collector->BeginEpisode();
			}

			// Random mode loads a fresh opponent at each episode boundary (and on the first
			// iteration, when none is loaded yet), freeing the previous one â€” one model resident
			// at a time. Pinned mode loaded once before the loop.
			if (config.includeStaticOpponent && staticRandomEachEpisode && (wasTerminal || !opponent)) {
				std::uniform_int_distribution<size_t> pick(0, staticAllowlist.size() - 1);
				opponent = LoadStaticOpponentInferUnit(staticAllowlist[pick(opponentRng)], useGPU);
			}

			const bool useStaticOpponent = opponent.has_value();

			GameState& gs = envSet->state.gameStates[0];

			// Let a GUI renderer apply pending setting changes (incl. player index) before the
			// episode's auto-focus decision runs, so auto-focus stays the last word on a reset frame.
			// NOTE: Applied once per frame (unlike the Learner's twice), so a setting change takes
			// effect on the next physics step: one-frame latency, not a parity bug.
			applyGuiSettings();

			if (wasTerminal || firstEpisode) {
				firstEpisode = false;

				const int focusCarIndex = FindPlayerIndexByCarId(gs, envSet->stateSetters[0]->trainingFocusCarId);
				staticTeam = ChooseStaticTeam(gs, focusCarIndex);

				const std::optional<Team> activeStaticTeam = useStaticOpponent ? std::optional<Team>(staticTeam) : std::nullopt;
				setFocus(PickEpisodeFocusIndex(gs, focusCarIndex, activeStaticTeam, focusIndex));

				contextLabel = useStaticOpponent
					? ("Static opponent (" + std::string(staticTeam == Team::BLUE ? "BLUE" : "ORANGE") + "): " + opponent->displayName)
					: std::string();
			}

			// Advance the action-delay window, then infer one action index per player from the
			// post-first-half state. InferUnit builds obs from gameStates, so the first half
			// must be synced before it is read.

			// Each player's action index must be decoded by the parser of the policy that produced
			// it, since a static opponent may use a different action table than the debug policy.
			// The primary-policy substitution assumes a stateless parser: it routes through the loop-local
			// parser instance rather than the env's own, so any per-episode override state a parser
			// carried would not survive here (no current agent's parser has such state).
			std::vector<ActionParser*> parsersByPlayer;
			if (useStaticOpponent) {
				parsersByPlayer.reserve(gs.players.size());
				for (const Player& player : gs.players)
					parsersByPlayer.push_back(player.team == staticTeam
						? opponent->actionParser.get()
						: actionParser.get());
				envSet->StepFirstHalf(asyncFirstHalf, parsersByPlayer);
			} else {
				envSet->StepFirstHalf(asyncFirstHalf);
			}
			envSet->Sync();

			IList actionIndices;
			actionIndices.reserve(gs.players.size());
			std::optional<std::vector<float>> focusObs;
			for (int playerIdx = 0; playerIdx < (int)gs.players.size(); playerIdx++) {
				const Player& player = gs.players[playerIdx];
				const bool driveWithStaticOpponent = useStaticOpponent && player.team == staticTeam;
				GGL::InferUnit& unit = driveWithStaticOpponent ? *opponent->inferUnit : inferUnit;
				actionIndices.push_back(unit.InferActionIndex(player, gs, deterministic, temperature));

				if (criticProbe && playerIdx == focusIndex && !driveWithStaticOpponent)
					focusObs = inferUnit.lastObs;
			}

			if (useStaticOpponent)
				envSet->StepSecondHalf(actionIndices, parsersByPlayer, asyncSecondHalf);
			else
				envSet->StepSecondHalf(actionIndices, asyncSecondHalf);

			// Same object as gs: StepSecondHalf mutates gameStates[0] in place
			// rather than producing a distinct post-step snapshot.
			GameState& postGs = envSet->state.gameStates[0];

			const GGL::RenderViewState view{ focusIndex, false };

			if (collector) {
				// Defensive: focus is clamped whenever it changes, but must never index past this arena's players.
				const int sampleIdx = std::min(collector->GetPlayerIndex(), (int)postGs.players.size() - 1);

				// Reflect any adaptive weight changes in the debug view.
				collector->SyncWeights(envSet->rewards[0]);

				// A focus player driven by the static opponent has no observation from this policy to value.
				std::optional<ForgeDebug::CriticDebugData> criticEstimate;
				if (focusObs) {
					try {
						criticEstimate = criticProbe->Evaluate(*focusObs, envSet->rewards[0]);
					} catch (const std::exception& e) {
						RG_LOG("Critic display: evaluation failed, critic hidden: " << e.what());
						criticProbe.reset();
						inferUnit.captureDiagnostics = false;
					}
				}
				collector->SetCriticEstimate(criticEstimate);

				collector->RecordStep(
					envSet->state.lastRewards[0],
					envSet->state.lastTrainingRewards,
					postGs.deltaTime, postGs.players[sampleIdx], postGs);

				const ForgeDebug::RewardDebugSnapshot& snapshot = collector->GetSnapshot();
				if (renderSender)
					renderSender->Send(postGs, &snapshot, contextLabel, view);
				if (config.debugRenderer)
					config.debugRenderer->Render(snapshot);
			} else {
				if (renderSender)
					renderSender->Send(postGs, nullptr, contextLabel, view);
			}

			// Enforce max episode length; the trajectory path that normally does this is absent.
			if (envSet->state.terminals[0]) {
				renderEpisodeStep = 0;
			} else if (++renderEpisodeStep >= maxEpisodeLength) {
				envSet->state.terminals[0] = RLGC::TerminalType::TRUNCATED;
				renderEpisodeStep = 0;
			}
		}
	}
}
