#include "Inference.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace RocketForge::SimulationAnalysis
{
	using namespace RLGC;

	InferenceTrace InferActionTrace(
		GGL::ModelSet& models,
		ObsBuilder& obsBuilder,
		ActionParser& actionParser,
		const Player& player,
		const GameState& state,
		int obsSize,
		int numActions,
		bool deterministic,
		float temperature)
	{
		const bool debugTrace = std::getenv("ROCKETFORGE_ACTION_TRACE_DEBUG") != nullptr;
		// Rebuild the one-player inference path used by training, including the
		// current action mask, so traced action ids match policy decisions.
		if (debugTrace)
			std::cerr << "action-trace: infer build obs\n";
		FList obs;
		obsBuilder.BuildObsInto(player, state, obs);
		if (debugTrace)
			std::cerr << "action-trace: infer obs built size " << obs.size() << "\n";
		if ((int)obs.size() != obsSize)
			throw std::runtime_error("obs builder produced unexpected size");

		std::vector<uint8_t> actionMask;
		if (debugTrace)
			std::cerr << "action-trace: infer build mask\n";
		if (actionParser.UsesActionMask())
			actionParser.GetActionMaskInto(player, state, actionMask);
		else
			actionMask.assign(numActions, true);
		if (debugTrace)
			std::cerr << "action-trace: infer mask built size " << actionMask.size() << "\n";

		if (debugTrace)
			std::cerr << "action-trace: infer tensors\n";
		torch::Tensor tObs = torch::tensor(obs).reshape({ 1, obsSize });
		torch::Tensor tMasks = torch::tensor(actionMask).reshape({ 1, numActions });
		torch::Tensor tActions;
		if (debugTrace)
			std::cerr << "action-trace: infer model\n";
		GGL::ModelManager::InferActionsFromModels(
			models,
			tObs,
			tMasks,
			deterministic,
			temperature,
			false,
			&tActions,
			nullptr);
		if (debugTrace)
			std::cerr << "action-trace: infer model complete\n";

		std::vector<int> actionIds = GGL::TENSOR_TO_VEC<int>(tActions);
		if (actionIds.empty())
			throw std::runtime_error("inference returned no actions");
		return { actionIds[0], std::move(obs), std::move(actionMask) };
	}
}