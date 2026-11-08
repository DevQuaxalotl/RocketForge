#pragma once

#include "../ObsBuilders/ObsBuilder.h"
#include "PredictionContext.h"

namespace RLGC {
    /** Applies one model's predictor without changing the arena or other models' observations. **/
    class ProfiledObsBuilder final : public ObsBuilder {
    public:
        ProfiledObsBuilder(ObsBuilder* builder, RocketSim::EngineId engine) : _inner(builder), _engine(engine) {}

        void Reset(const GameState& state) override {
            ScopedPredictionContext scope(state, Prepare(state));
            _inner->Reset(state);
        }

        FList BuildObs(const Player& player, const GameState& state) override {
            ScopedPredictionContext scope(state, Prepare(state));
            return _inner->BuildObs(player, state);
        }

        void BuildObsInto(const Player& player, const GameState& state, FList& out) override {
            ScopedPredictionContext scope(state, Prepare(state));
            _inner->BuildObsInto(player, state, out);
        }

    private:
        /** Points `state` at a context until the build returns or throws, then restores the one it had. **/
        class ScopedPredictionContext {
        public:
            ScopedPredictionContext(const GameState& state, BallPrediction::Context* context)
                : _state(state), _previous(state.predictionContext) {
                state.predictionContext = context;
            }
            ~ScopedPredictionContext() { _state.predictionContext = _previous; }

            ScopedPredictionContext(const ScopedPredictionContext&) = delete;
            ScopedPredictionContext& operator=(const ScopedPredictionContext&) = delete;

        private:
            const GameState& _state;
            BallPrediction::Context* _previous;
        };

        BallPrediction::Context* Prepare(const GameState& state) {
            if (!_context || _arena != state.lastArena || _mode != state.gameMode) {
                _arena = state.lastArena;
                _mode = state.gameMode;
                _context = BallPrediction::Acquire(_arena, _mode, _engine);
            }
            return _context.get();
        }

        std::unique_ptr<ObsBuilder> _inner;
        RocketSim::EngineId _engine;
        std::shared_ptr<BallPrediction::Context> _context;
        const RocketSim::Arena* _arena = nullptr;
        RocketSim::GameMode _mode = RocketSim::GameMode::SOCCAR;
    };
}
