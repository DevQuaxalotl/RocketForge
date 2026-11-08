#pragma once
#include "../Gamestates/GameState.h"
#include "../BasicTypes/Action.h"
#include "../BasicTypes/Lists.h"

#include <string>
#include <typeinfo>

// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/obs_builders/obs_builder.py
namespace RLGC {
	/**
	 * A control macro emitted by a policy action.
	 * ticks is measured in physics ticks within the current action window.
	 **/
	struct ActionMacro {
		Action action = {};
		int ticks = 0;
	};

	// TODO: Only designed for discrete actions currently 
	class ActionParser {
	public:
		virtual ~ActionParser() = default;

		virtual Action ParseAction(int actionIdx, const Player& player, const GameState& state) = 0;
		virtual void ParseActionMacros(int actionIdx, const Player& player, const GameState& state, int tickCount, std::vector<ActionMacro>& out) {
			out.clear();
			out.push_back({ ParseAction(actionIdx, player, state), tickCount });
		}
		virtual bool IsMacroAction(int, const Player&, const GameState&) {
			return false;
		}

		/** Returns whether this parser has a stateful runtime action override. **/
		virtual bool HasRuntimeActionOverride() const {
			return false;
		}

		/**
		 * Returns whether this parser needs its policy action reconsidered on
		 * every physics tick for the supplied player and state.
		 **/
		virtual bool RequiresPerTickActionOverride(const Player&, const GameState&) {
			return false;
		}

		/**
		 * Applies a stateful runtime override after policy action parsing.
		 * ticksElapsed is the number of physics ticks since the previous call.
		 **/
		virtual Action OverrideRuntimeAction(
			const Action& policyAction,
			const Player&,
			const GameState&,
			int) {
			return policyAction;
		}

		/** Clears state retained by the runtime action override. **/
		virtual void ResetRuntimeActionOverride() {
		}

		virtual int GetActionAmount() = 0;

		virtual std::string GetPolicyCompatibilityKey() {
			return typeid(*this).name();
		}

		// Returns true or false for each action, depending on if it is available in the current situation
		// Not using std::vector<bool> because it has major issues (see https://isocpp.org/blog/2012/11/on-vectorbool)
		virtual std::vector<uint8_t> GetActionMask(const Player& player, const GameState& state) {
			return std::vector<uint8_t>(GetActionAmount(), true);
		}

		virtual bool UsesActionMask() {
			return false;
		}

		virtual void GetActionMaskInto(const Player& player, const GameState& state, std::vector<uint8_t>& out) {
			out.assign(GetActionAmount(), true);
		}
	};
}