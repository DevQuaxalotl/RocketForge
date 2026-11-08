#pragma once
#include "Report.h"
#include <pybind11/pybind11.h>
#include <RLGymCPP/Gamestates/GameState.h>
#include <RLGymCPP/BasicTypes/Action.h>
#include <GigaLearnCPP/Util/Timer.h>
#include <ForgeDebugger.h>
#include <atomic>
#include <string>
#include <vector>

namespace GGL {
	/** Viewer-facing state that is not part of the game state itself. */
	struct RenderViewState {
		/** Player index the viewer camera should follow; negative leaves the viewer's own choice. */
		int spectateIndex = -1;

		bool paused = false;
	};

	struct GGL_TRAIN_IMEXPORT RenderSender {
		pybind11::module pyMod;

		float timeScale;
		double adaptiveRenderDelay = -1;
		Timer renderTimer = {};

		RenderSender(float timeScale = 1);

		RG_NO_COPY(RenderSender);

		void Send(
			const RLGC::GameState& state,
			const ForgeDebug::RewardDebugSnapshot* debugSnapshot = nullptr,
			const std::string& contextLabel = {},
			const RenderViewState& view = {});
		void SetRuntimeTimeScale(std::atomic<float>* runtimeScale);

		/** Drains control datagrams the viewer sent back since the last call. Never blocks. */
		std::vector<std::string> PollCommands();

		~RenderSender();

	private:
		std::atomic<float>* _runtimeTimeScale = nullptr;
	};
}
