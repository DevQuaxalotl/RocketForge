#pragma once
#include "../Gamestates/GameState.h"
#include "AdaptiveWeightSettings.h"

#include <string>

// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/reward_function.py
namespace RLGC {

	enum class RewardCategory : uint8_t {
		CONTINUOUS,  // Produces values every frame (velocity, distance, etc.)
		EVENT        // Produces 0 most frames, spikes on events (goal, demo, etc.)
	};

	struct RewardValueRange {
		float minValue = 0.f;
		float maxValue = 1.f;
	};

	struct DebugLine {
		Vec start;
		Vec end;
		Vec color; // RGB 0-1
	};

	/** Demangled class name with namespaces stripped; the default name of a reward class. */
	std::string GetRewardTypeName(const std::type_info& type);

	class Reward {
	private:
		std::string _cachedName = {};

	public:
		bool debugEnabled = false;

		virtual void Reset(const GameState& initialState) {}

		virtual void PreStep(const GameState& state) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			throw std::runtime_error("GetReward() is unimplemented");
			return 0;
		}

		/** Returns one reward per player in state order. */
		virtual std::vector<float> GetAllRewards(const GameState& state, bool isFinal);

		/**
		 * Writes one reward per player in state order, replacing every value in out.
		 * The caller may reuse out between rewards and steps; do not retain references to it.
		 * The default dispatches to GetAllRewards, so subclasses that override only that keep working.
		 **/
		virtual void GetAllRewardsInto(const GameState& state, bool isFinal, std::vector<float>& out);

		virtual std::string GetName();

		/** Returns whether this reward is a continuous signal or a sparse event spike */
		virtual RewardCategory GetCategory() const { return RewardCategory::CONTINUOUS; }

		/** Returns the expected output range of this reward, used for normalization in debug visualization */
		virtual RewardValueRange GetValueRange() const { return { 0.f, 1.f }; }

		/** Returns 3D debug lines to visualize this reward's internal state. Only called when debugEnabled is true. */
		virtual std::vector<DebugLine> GetDebugLines(const Player& player, const GameState& state) const { return {}; }

		bool IsDebugEnabled() const { return debugEnabled; }
		void SetDebugEnabled(bool enabled) { debugEnabled = enabled; }

		virtual ~Reward() {};
	};

	struct WeightedReward {
		Reward* reward;

		/**
		 * Effective weight applied during stepping. 
		 * When adaptiveSettings.enabled, the controller overwrites this each iteration;
		 * otherwise it stays fixed.
		 */
		float weight;

		/** enabled == false means no adaptive control. */
		AdaptiveWeightSettings adaptiveSettings = {};

		WeightedReward(Reward* reward, float scale)
			: reward(reward), weight(scale) {}
		WeightedReward(Reward* reward, int scale)
			: reward(reward), weight((float)scale) {}
		WeightedReward(Reward* reward, float scale, const AdaptiveWeightSettings& adaptive)
			: reward(reward), weight(scale), adaptiveSettings(adaptive) {}
	};
}
