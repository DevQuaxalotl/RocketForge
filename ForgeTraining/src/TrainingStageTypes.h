#pragma once

#include <GigaLearnCPP/Util/Report.h>

#include <RLGymCPP/Gamestates/GameState.h>
#include <RLGymCPP/StateSetters/StateSetter.h>

#include <functional>
#include <string>
#include <vector>

namespace GGL
{
	class Learner;
}

namespace RocketForge
{
	/** One matchup a stage trains (teamSizeA vs teamSizeB) and its relative chance of being picked for an arena. */
	struct WeightedTeamSize {
		int teamSizeA;
		int teamSizeB;
		float weight;

		WeightedTeamSize(int teamSize, float weight)
			: teamSizeA(teamSize), teamSizeB(teamSize), weight(weight) {}

		WeightedTeamSize(int teamSizeA, int teamSizeB, float weight)
			: teamSizeA(teamSizeA), teamSizeB(teamSizeB), weight(weight) {}
	};

	/** Pairs a state setter with selection metadata before it is wrapped in CombinedState. */
	struct WeightedStateSetter {
		RLGC::StateSetter* stateSetter = nullptr;
		float weight = 0.f;

		WeightedStateSetter(RLGC::StateSetter* stateSetter, float weight)
			: stateSetter(stateSetter), weight(weight) {}
	};

	/** User-facing state setter option shown by the debug render force-setter dropdown. */
	struct DebugStateSetterOption {
		int index = 0;

		/** The setter's class name, without namespace. */
		std::string name;
		std::string label;
		float weight = 0.f;
	};

	typedef std::function<void(class GGL::Learner*, const std::vector<RLGC::GameState>& states, GGL::Report& report)> StepCallbackFn;
}
