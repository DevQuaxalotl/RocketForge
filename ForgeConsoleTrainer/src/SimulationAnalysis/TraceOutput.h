#pragma once

#include "SimulationAnalysisTypes.h"

#include <RLGymCPP/ActionParsers/ActionParser.h>

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace RocketForge::SimulationAnalysis
{
	/** Writes the stable action-trace CSV header. **/
	void WriteCsvHeader(std::ostream& out);
	/** Writes one action-trace CSV row. **/
	void WriteCsvRow(std::ostream& out, const TraceSample& sample);
	/** Finds the lookup-table action id matching a control vector, or -1 when none matches. **/
	int FindActionIdForControls(const std::vector<RLGC::Action>& actions, const RLGC::Action& action);
	/** Formats a macro plan as base-action/tick segments for CSV output. **/
	std::string BuildMacroPlanString(
		const std::vector<RLGC::ActionMacro>& macros,
		const std::vector<RLGC::Action>& actionLookup);
	/** Converts one physics tick of RocketSim state into the stable CSV sample shape. **/
	TraceSample MakeTraceSample(
		int episode,
		uint32_t carId,
		int physicsTick,
		int policyStep,
		int policyTick,
		int tickSkip,
		int actionDelay,
		int selectedActionId,
		bool selectedIsMacro,
		int selectedMacroIndex,
		int appliedActionId,
		int appliedBaseActionId,
		int macroSegmentIndex,
		int macroSegmentTick,
		int macroSegmentTicks,
		int macroTicksLeft,
		const std::string& macroPlan,
		const std::vector<uint8_t>& actionMask,
		const RLGC::FList& obs,
		const RLGC::Player& maskPlayer,
		const RLGC::Action& selectedAction,
		const RLGC::Action& appliedAction,
		const RLGC::Action& prevAction,
		const RocketSim::CarState& carState,
		const RLGC::GameState& obsState,
		bool jumpEdge,
		bool ballTouchedTick);
}