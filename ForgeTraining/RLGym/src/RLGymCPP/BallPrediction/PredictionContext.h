#pragma once

#include <RocketSim/Arena.h>

#include <cstdint>
#include <memory>

namespace RLGC::BallPrediction {
	/**
	 * One model's ball predictor, which may run a different engine than the match it observes.
	 * Virtual so a context made in one module predicts with that module's sim code, whichever module calls it.
	 */
	class Context {
	public:
		virtual ~Context() = default;

		virtual RocketSim::EngineId Engine() const = 0;

		/** Ball state `seconds` after `ball`, seen at arena tick `tick`; reuses its buffer while the ball follows the prediction. **/
		virtual RocketSim::BallState Predict(const RocketSim::BallState& ball, uint64_t tick, float seconds) = 0;
	};

	/**
	 * Shares one context per (arena, mode, engine); `arena` may be null for packet-driven play, which gets its own context.
	 * Dropshot contexts read the tiles of `arena` before each prediction, so it must outlive the context's use on it.
	 * Throws RocketSim::EngineUnavailableError when `engine` isn't built.
	 */
	std::shared_ptr<Context> Acquire(const RocketSim::Arena* arena, RocketSim::GameMode mode, RocketSim::EngineId engine);
}
