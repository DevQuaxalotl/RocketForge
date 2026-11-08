#pragma once

#include "rocketsim_ffi.h"
#include <memory>
#include <stdexcept>
#include <vector>

/** Generic C++ ownership wrapper. No RocketForge or v2 types are required. **/
namespace rocketsim_v3 {
    enum class GameMode : uint32_t { Soccar, Hoops, Heatseeker, Snowday, Dropshot, TheVoid };
    enum class MemoryMode : uint32_t { Heavy, Balanced, Light };

    inline void Check(int status) {
        if (status != 0)
            throw std::runtime_error(rsf_last_error());
    }

    inline void Initialize(const char* meshPath, bool silent = false) {
        Check(rsf_init(meshPath, silent));
    }

    /** Move-only arena owner. Calls on one arena must not overlap across threads. **/
    class Arena {
        struct Deleter { void operator()(RsfArena* arena) const { rsf_arena_free(arena); } };
        std::unique_ptr<RsfArena, Deleter> handle;

        explicit Arena(RsfArena* arena) : handle(arena) {
            if (!arena)
                throw std::runtime_error(rsf_last_error());
        }

    public:
        explicit Arena(
            GameMode mode = GameMode::Soccar,
            int64_t seed = -1,
            MemoryMode memory = MemoryMode::Heavy,
            bool noBallRotation = false)
            : Arena(rsf_arena_new((uint32_t)mode, seed, noBallRotation, (uint32_t)memory)) {}

        Arena(Arena&&) noexcept = default;
        Arena& operator=(Arena&&) noexcept = default;
        Arena(const Arena&) = delete;
        Arena& operator=(const Arena&) = delete;

        /** Borrowed handle; ownership stays with this object. **/
        RsfArena* Get() const { return handle.get(); }
        Arena Clone() const { return Arena(rsf_arena_clone(Get())); }

        uint32_t AddCar(uint32_t team, const RsfCarConfig* config = nullptr) {
            const uint32_t index = rsf_arena_add_car(Get(), team, config);
            if (index == UINT32_MAX)
                throw std::runtime_error(rsf_last_error());

            return index;
        }

        void Step(uint32_t ticks = 1) { Check(rsf_arena_step(Get(), ticks)); }
        void ResetKickoff(int64_t seed = -1) { Check(rsf_arena_reset_kickoff(Get(), seed)); }

        RsfCarState GetCarState(uint32_t index) const {
            RsfCarState state{};
            Check(rsf_arena_get_car_state(Get(), index, &state));
            return state;
        }

        void SetCarState(uint32_t index, const RsfCarState& state) {
            Check(rsf_arena_set_car_state(Get(), index, &state));
        }

        void SetControls(const std::vector<RsfCarControls>& controls) {
            if (controls.size() > UINT32_MAX)
                throw std::length_error("Too many controls");

            Check(rsf_arena_set_controls(Get(), controls.data(), (uint32_t)controls.size()));
        }

        RsfBallState GetBallState() const {
            RsfBallState state{};
            Check(rsf_arena_get_ball_state(Get(), &state));
            return state;
        }

        void SetBallState(const RsfBallState& state) { Check(rsf_arena_set_ball_state(Get(), &state)); }

        /** Steps with every car's controls and fills the caller-sized buffers in `out`. **/
        void StepSnapshot(const std::vector<RsfCarControls>& controls, uint32_t ticks, RsfStepOutput& out) {
            if (controls.size() != out.numCars) throw std::length_error("Controls count must equal out.numCars");
            Check(rsf_arena_step_snapshot(Get(), controls.data(), ticks, &out));
        }
        void SetCarStatePreserving(uint32_t index, const RsfCarState& mapped) {
            Check(rsf_arena_set_car_state_preserving(Get(), index, &mapped));
        }
        void SetBallStatePreserving(const RsfBallState& state) {
            Check(rsf_arena_set_ball_state_preserving(Get(), &state));
        }
        std::vector<uint8_t> GetTiles() const {
            const uint32_t count = rsf_arena_num_tiles(Get());
            if (count == UINT32_MAX) throw std::runtime_error(rsf_last_error());
            std::vector<uint8_t> tiles(count);
            Check(rsf_arena_get_tiles(Get(), tiles.data(), count));
            return tiles;
        }
        void SetTiles(const std::vector<uint8_t>& tiles) {
            if (tiles.size() > UINT32_MAX) throw std::length_error("Too many tiles");
            Check(rsf_arena_set_tiles(Get(), tiles.data(), (uint32_t)tiles.size()));
        }
        std::vector<RsfEvent> GetEvents() const {
            uint32_t count = 0;
            if (rsf_arena_get_events(Get(), nullptr, 0, &count) == UINT32_MAX)
                throw std::runtime_error(rsf_last_error());

            std::vector<RsfEvent> events(count);
            if (rsf_arena_get_events(Get(), events.data(), count, nullptr) == UINT32_MAX)
                throw std::runtime_error(rsf_last_error());

            return events;
        }
    };

    /** Move-only ball predictor owner. Calls on one predictor must not overlap across threads. **/
    class BallPredictor {
        struct Deleter { void operator()(RsfBallPredictor* p) const { rsf_ball_predictor_free(p); } };
        std::unique_ptr<RsfBallPredictor, Deleter> handle;
    public:
        explicit BallPredictor(GameMode mode = GameMode::Soccar, bool noBallRotation = false)
            : handle(rsf_ball_predictor_new((uint32_t)mode, noBallRotation)) {
            if (!handle) throw std::runtime_error(rsf_last_error());
        }

        /** Borrowed handle; ownership stays with this object. **/
        RsfBallPredictor* Get() const { return handle.get(); }
        void SetTiles(const std::vector<uint8_t>& tiles) {
            if (tiles.size() > UINT32_MAX) throw std::length_error("Too many tiles");
            Check(rsf_ball_predictor_set_tiles(Get(), tiles.data(), (uint32_t)tiles.size()));
        }

        /** Fills `out` with one state per tick and returns the first scoring index, or -1. **/
        int32_t Predict(const RsfBallState& start, std::vector<RsfBallState>& out) {
            if (out.size() > INT32_MAX) throw std::length_error("Too many prediction ticks");
            const int32_t scored = rsf_ball_predict(Get(), &start, (uint32_t)out.size(), out.data());
            if (scored == INT32_MIN) throw std::runtime_error(rsf_last_error());
            return scored;
        }
    };
}
