#include "PredictionContext.h"

#include <RocketSim/BallPredTracker.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>

using namespace RocketSim;

namespace RLGC::BallPrediction {
	namespace {
		constexpr float PREDICTION_TICK_RATE = 120.f;
		constexpr float MAX_PREDICTION_SECONDS = 60.f;

		/** The ball fields that seed a prediction, compared bit for bit to spot a ball set within one tick. **/
		struct SeedKey {
			float values[18];

			explicit SeedKey(const BallState& ball) {
				for (int i = 0; i < 3; i++) {
					values[i] = ball.pos[i];
					values[3 + i] = ball.vel[i];
					values[6 + i] = ball.angVel[i];
					for (int j = 0; j < 3; j++)
						values[9 + i * 3 + j] = ball.rotMat[i][j];
				}
			}

			bool operator==(const SeedKey& other) const { return std::memcmp(values, other.values, sizeof(values)) == 0; }
		};

		/**
		 * Soccar and Hoops predict from and return only the physical ball state, which is what their models were always given.
		 * Other modes keep their mode state (Dropshot charge, Heatseeker target), since the prediction depends on it.
		 */
		BallState PredictionView(const BallState& ball, GameMode mode) {
			if (mode != GameMode::SOCCAR && mode != GameMode::HOOPS)
				return ball;

			BallState view;
			view.pos = ball.pos;
			view.vel = ball.vel;
			view.angVel = ball.angVel;
			view.rotMat = ball.rotMat;
			return view;
		}

		/** Each engine keeps the prediction horizon its models were trained against. **/
		size_t PredictionTickCount(EngineId engine, float seconds) {
			if (engine == EngineId::V3)
				return std::max<size_t>(2, static_cast<size_t>(std::ceil(seconds * PREDICTION_TICK_RATE)) + 1);
			return std::max<size_t>(2, static_cast<size_t>(seconds / (1.f / PREDICTION_TICK_RATE)));
		}

		/** Predicts on a ball-only arena of its own engine, fed the observed ball rather than stepped with the match. **/
		class TrackerContext final : public Context {
		public:
			TrackerContext(const Arena* arena, GameMode mode, EngineId engine)
				: _mode(mode), _engine(engine), _dropshotTilesSource(mode == GameMode::DROPSHOT ? arena : nullptr) {}

			EngineId Engine() const override { return _engine; }

			BallState Predict(const BallState& ball, uint64_t tick, float seconds) override {
				if (!std::isfinite(seconds) || seconds < 0.f || seconds > MAX_PREDICTION_SECONDS)
					throw std::invalid_argument("Ball prediction requires a time from 0 to 60 seconds");

				const BallState seed = PredictionView(ball, _mode);
				if (seconds == 0.f)
					return seed;

				const SeedKey seedKey(seed);
				const size_t tickCount = PredictionTickCount(_engine, seconds);

				std::lock_guard lock(_mutex);
				const bool tilesChanged = RefreshDropshotTiles();

				if (!_tracker) {
					// The tracker copies its prediction arena's config and mode from this throwaway seed arena.
					std::unique_ptr<Arena> seedArena(Arena::Create(_mode, MakeArenaConfig(), PREDICTION_TICK_RATE));
					if (_dropshotTilesSource)
						seedArena->SetDropshotTilesState(_sentTiles);
					seedArena->ball->SetState(seed);
					_tracker = std::make_unique<BallPredTracker>(seedArena.get(), tickCount);
				} else {
					if (tickCount > _tracker->numPredTicks)
						_tracker->SetNumPredTicks(tickCount);

					const bool changedAtSameTick = tick == _lastTick && !(seedKey == _lastSeed);
					const bool tickCounterJumped = tick < _lastTick || tick - _lastTick > static_cast<uint64_t>(std::numeric_limits<int>::max());

					if (tilesChanged) {
						_tracker->ballPredArena->SetDropshotTilesState(_sentTiles);
						_tracker->ForceUpdateAllPred(seed);
					} else if (tickCounterJumped || changedAtSameTick) {
						_tracker->ForceUpdateAllPred(seed);
					} else if (tick != _lastTick) {
						// Pass a relative tick delta: absolute run counters need not fit the tracker's int.
						_tracker->lastUpdateTickCount = 0;
						_tracker->UpdatePredManual(seed, static_cast<int>(tick - _lastTick));
					}
				}

				_lastTick = tick;
				_lastSeed = seedKey;
				return PredictionView(_tracker->GetBallStateForTime(seconds), _mode);
			}

		private:
			ArenaConfig MakeArenaConfig() const {
				ArenaConfig config = {};
				config.engine = _engine;
				return config;
			}

			/** Copies the match's Dropshot tiles; returns true when they differ from those last predicted on. **/
			bool RefreshDropshotTiles() {
				if (!_dropshotTilesSource)
					return false;

				const DropshotTilesState tiles = _dropshotTilesSource->GetDropshotTilesState();
				if (std::memcmp(&tiles, &_sentTiles, sizeof(DropshotTilesState)) == 0)
					return false;

				_sentTiles = tiles;
				return true;
			}

			const GameMode _mode;
			const EngineId _engine;

			/** The match arena in Dropshot, null otherwise. Not owned. **/
			const Arena* const _dropshotTilesSource;

			/** Shared contexts can be reached from several observation threads. **/
			std::mutex _mutex;

			std::unique_ptr<BallPredTracker> _tracker;
			uint64_t _lastTick = 0;
			SeedKey _lastSeed{BallState()};
			DropshotTilesState _sentTiles{};
		};
	}

	std::shared_ptr<Context> Acquire(const Arena* arena, GameMode mode, EngineId engine) {
		RequireEngine(engine);

		if (!arena)
			return std::make_shared<TrackerContext>(arena, mode, engine);

		using Key = std::tuple<const Arena*, GameMode, EngineId>;
		static std::mutex mutex;
		static std::map<Key, std::weak_ptr<Context>> contexts;
		static size_t acquisitions = 0;

		std::lock_guard lock(mutex);

		// Arenas come and go with environments, so drop the expired entries now and then.
		if (++acquisitions % 256 == 0) {
			for (auto it = contexts.begin(); it != contexts.end();) {
				if (it->second.expired())
					it = contexts.erase(it);
				else
					++it;
			}
		}

		std::weak_ptr<Context>& weak = contexts[Key(arena, mode, engine)];
		std::shared_ptr<Context> context = weak.lock();
		if (!context) {
			context = std::make_shared<TrackerContext>(arena, mode, engine);
			weak = context;
		}

		return context;
	}
}
