#pragma once

#include <atomic>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include <RocketSim/Engine.h>

namespace ForgeDebug
{
	class IDebugRenderer;
	struct DebugSettings;
}

namespace RocketForge
{
	/** Everything the inference render loop needs to run one fixed-policy rendering session. */
	struct InferenceDebugRenderConfig
	{
		std::string agentName;

		/** Reward set / team sizes / state setters. */
		std::string stageName;

		std::filesystem::path checkpointFolder;

		/** Build whose static_opponents.json allowlist supplies the opponent (when enabled). */
		std::string buildName;

		/** Drive the static-opponent team (opposite the setter's focus car, else ORANGE) with a model from the Build allowlist. */
		bool includeStaticOpponent = false;

		/** Empty picks a random allowlist entry each episode; otherwise pins this selection key. */
		std::string staticOpponentSelectionKey;

		/** FromImportedModel vs FromSegmentCheckpoint. */
		bool isImported = false;

		/** Build the RewardDebugCollector and send snapshots. */
		bool showRewards = false;

		/** Players per team in the render arena. */
		int teamSizeOverride = 1;

		/** Player the camera and reward sampling start on; each episode's auto-focus may replace it. */
		int playerIndex = 0;

		/** Optional GUI renderer (e.g. DebugBridge). Null in headless/console use. */
		ForgeDebug::IDebugRenderer* debugRenderer = nullptr;

		/** Optional debug settings (norm/display mode, forceDebugAll, maxEpisodeTime). May be null. */
		const ForgeDebug::DebugSettings* debugSettings = nullptr;

		/** Live render time scale (1 = realtime), shared with the caller: viewer speed keys write it. Null uses a loop-local scale. */
		std::atomic<float>* timeScale = nullptr;

		/** Console/test callers may omit the 3D viewer or request CPU inference. **/
		bool renderScene = true;
		bool useGPU = true;

		/** Applies pending GUI game controls each frame, even while paused or without reward collection. **/
		std::function<void()> pollControls;

		/** Returns a newly requested player index, or -1 when unchanged. **/
		std::function<int()> consumePlayerIndex;

		/** Polled each frame; loop exits promptly when it returns true. */
		std::function<bool()> shouldStop;
	};

	/**
	 * Engines the render needs: each model's training engine and predictor, from its checkpoint setup or the current agent for imported models.
	 * Includes the Build's static opponents when the render plays one.
	 */
	std::vector<RocketSim::EngineId> ResolveDebugEngines(const InferenceDebugRenderConfig& config);

	/**
	 * Runs an inference-only render session until config.shouldStop() returns true. Loads the
	 * model via InferenceModelSource, builds a single RocketSim arena from the stage, and drives
	 * RenderSender (+ RewardDebugCollector when showRewards). Throws std::runtime_error on setup
	 * failure (unknown agent/stage, no checkpoint to load).
	 */
	void RunInferenceDebugRender(const InferenceDebugRenderConfig& config);
}
