#pragma once
#include <RLGymCPP/Gamestates/GameState.h>
#include <unordered_map>
#include <typeindex>
#include <memory>

namespace RocketForge {

    /** A single lazily-computed value tagged with the tick it was produced for. **/
    template<typename T>
    struct CachedValue {
        T value = {};
        uint64_t computedTick = UINT64_MAX;

        bool IsValid(uint64_t curTick) const { return computedTick == curTick; }
        void MarkComputed(uint64_t curTick) { computedTick = curTick; }
    };

    /**
     * Per-arena scratch space shared between rewards within a single frame.
     * Rewards/helpers fetch a typed slot by tag, then manage their own
     * compute/validity. Slots are allocated once on first use and reused for the
     * arena's lifetime. Accessed via GameState::userInfo; owned by TrainingStage.
     **/
    class RewardCacheBase {
    public:
        virtual ~RewardCacheBase() = default;

        /**
         * Returns the cached slot for the given Tag, creating it on first access.
         * Tag must define `using ValueType = ...`. The tag type identifies the
         * quantity, allowing two distinct quantities of the same ValueType to
         * coexist and letting different rewards share one slot.
         **/
        template<typename Tag>
        CachedValue<typename Tag::ValueType>& GetSlot() {
            const std::type_index key(typeid(Tag));

            auto it = _slots.find(key);
            if (it == _slots.end())
                it = _slots.emplace(key, std::make_unique<Slot<typename Tag::ValueType>>()).first;

            return static_cast<Slot<typename Tag::ValueType>*>(it->second.get())->cached;
        }

    private:
        struct SlotBase { virtual ~SlotBase() = default; };

        template<typename T>
        struct Slot : SlotBase { CachedValue<T> cached; };

        std::unordered_map<std::type_index, std::unique_ptr<SlotBase>> _slots;
    };

    /**
     * Generic helper for the common case: a per-player array computed once per
     * frame. Returns the value for `player`. Falls back to computing only the
     * requested player when no cache is present (RLBot live play, analysis tools).
     *
     * computeElem must be callable as: ValueType::value_type(const Player&, const GameState&)
     **/
    template<typename Tag, typename ComputeElemFn>
    typename Tag::ValueType::value_type GetCachedPerPlayer(
        const RLGC::Player& player, const RLGC::GameState& state, ComputeElemFn&& computeElem)
    {
        RewardCacheBase* cache = static_cast<RewardCacheBase*>(state.userInfo);
        if (!cache)
            return computeElem(player, state);

        CachedValue<typename Tag::ValueType>& slot = cache->GetSlot<Tag>();
        if (!slot.IsValid(state.lastTickCount)) {
            slot.value.resize(state.players.size());
            for (size_t i = 0; i < state.players.size(); i++)
                slot.value[i] = computeElem(state.players[i], state);
            slot.MarkComputed(state.lastTickCount);
        }

        RG_ASSERT(player.index < slot.value.size());
        return slot.value[player.index];
    }
}
