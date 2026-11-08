#pragma once
#include "ActionParser.h"

#include <array>

namespace RLGC {

	// Actions match DiscreteAction in RLGymPPO_CPP and lookup_act.py in Python RLGym, but also has action masking
	class DefaultAction : public ActionParser {
	public:

		std::vector<Action> actions;
		std::vector<uint8_t> groundMask, airMask, jumpMask, boostMask;
		std::array<std::vector<uint8_t>, 8> cachedMasks;

		DefaultAction();

		virtual Action ParseAction(int index, const Player& player, const GameState& state) override {
			return actions[index];
		}

		virtual int GetActionAmount() override {
			return actions.size();
		}

		virtual bool UsesActionMask() override {
			return true;
		}

		virtual std::vector<uint8_t> GetActionMask(const Player& player, const GameState& state) override;
		virtual void GetActionMaskInto(const Player& player, const GameState& state, std::vector<uint8_t>& out) override;

	private:
		void BuildCachedMasks();
	};
}
