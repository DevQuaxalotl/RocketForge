#include "GigaLearnInfer/InferUnit.h"

#include <GigaLearnInfer/Models.h>
#include <GigaLearnInfer/ModelManager.h>
#include <GigaLearnInfer/Agent.h>
#include <stdexcept>

GGL::InferUnit::InferUnit(
	RLGC::ObsBuilder* obsBuilder, int obsSize, RLGC::ActionParser* actionParser,
	PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig, 
	std::filesystem::path modelsFolder, bool useGPU) : 
	obsBuilder(obsBuilder), obsSize(obsSize), actionParser(actionParser), useGPU(useGPU) {

	this->models = new ModelSet();

	try {
		ModelManager::MakeModels(
			false, obsSize, actionParser->GetActionAmount(),
			sharedHeadConfig, policyConfig, {},
			useGPU ? torch::kCUDA : torch::kCPU,
			*this->models
		);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("InferUnit: Exception when trying to construct models: " << e.what());
	}

	try {
		this->models->Load(modelsFolder, false);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("InferUnit: Exception when trying to load models: " << e.what());
	}
}

GGL::InferUnit::InferUnit(
	RLGC::ObsBuilder* obsBuilder, int obsSize, RLGC::ActionParser* actionParser,
	const Agent& agent, std::filesystem::path modelsFolder, bool useGPU) :
	obsBuilder(obsBuilder), obsSize(obsSize), actionParser(actionParser), useGPU(useGPU) {

	this->models = new ModelSet();

	try {
		torch::Device device = useGPU ? torch::kCUDA : torch::kCPU;
		agent.MakeInferenceModels(obsSize, actionParser->GetActionAmount(), device, *this->models);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("InferUnit: Exception when trying to construct models: " << e.what());
	}

	try {
		this->models->Load(modelsFolder, false);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("InferUnit: Exception when trying to load models: " << e.what());
	}
}

GGL::InferUnit::~InferUnit() {
	if (models) {
		models->Free();
		delete models;
		models = nullptr;
	}
}

RLGC::Action GGL::InferUnit::InferAction(const RLGC::Player& player, const RLGC::GameState& state, bool deterministic, float temperature) {
	return BatchInferActions({ player }, { state }, deterministic, temperature)[0];
}

int GGL::InferUnit::InferActionIndex(const RLGC::Player& player, const RLGC::GameState& state, bool deterministic, float temperature) {
	return InferRows({ &player }, { &state }, deterministic, temperature)[0];
}

std::vector<int> GGL::InferUnit::BatchInferActionIndices(const std::vector<RLGC::Player>& players, const std::vector<RLGC::GameState>& states, bool deterministic, float temperature) {
	if (players.size() != states.size())
		throw std::invalid_argument("Inference players and states must have equal lengths");

	std::vector<const RLGC::Player*> playerRows;
	std::vector<const RLGC::GameState*> stateRows;
	playerRows.reserve(players.size());
	stateRows.reserve(states.size());

	for (size_t i = 0; i < players.size(); ++i) {
		playerRows.push_back(&players[i]);
		stateRows.push_back(&states[i]);
	}

	return InferRows(playerRows, stateRows, deterministic, temperature);
}

std::vector<int> GGL::InferUnit::BatchInferActionIndices(const std::vector<unsigned>& indices, const RLGC::GameState& state, bool deterministic, float temperature) {
	std::vector<const RLGC::Player*> players;
	players.reserve(indices.size());

	for (unsigned index : indices) {
		if (index >= state.players.size())
			throw std::out_of_range("Inference player index is outside the packet");
		players.push_back(&state.players[index]);
	}

	return InferRows(players, std::vector<const RLGC::GameState*>(indices.size(), &state), deterministic, temperature);
}

std::vector<int> GGL::InferUnit::InferRows(const std::vector<const RLGC::Player*>& players, const std::vector<const RLGC::GameState*>& states, bool deterministic, float temperature) {
	if (players.empty()) {
		lastBatchSize = 0;
		lastObs.clear();
		lastActionMasks.clear();
		lastActionProbs.clear();
		return {};
	}

	int batchSize = players.size();
	std::vector<float> allObs;
	std::vector<uint8_t> allActionMasks;
	allObs.reserve((size_t)batchSize * obsSize);
	allActionMasks.reserve((size_t)batchSize * actionParser->GetActionAmount());
	for (int i = 0; i < batchSize; i++) {
		FList curObs = obsBuilder->BuildObs(*players[i], *states[i]);
		if (curObs.size() != obsSize) {
			RG_ERR_CLOSE(
				"InferUnit: Obs builder produced an obs that differs from the provided size (expected: " << obsSize << ", got: " << curObs.size() << ")\n" <<
				"Make sure you provided the correct obs size to the InferUnit constructor.\n" <<
				"Also, make sure there aren't an incorrect number of players (there are " << states[i]->players.size() << " in this state)"
			);
		}
		allObs += curObs;

		allActionMasks += actionParser->GetActionMask(*players[i], *states[i]);
	}
	return BatchInferActionIndicesFromObs(allObs, allActionMasks, batchSize, deterministic, temperature);
}

std::vector<int> GGL::InferUnit::BatchInferActionIndicesFromObs(
	const std::vector<float>& obs, const std::vector<uint8_t>& actionMasks, int batchSize, bool deterministic, float temperature) {
	RG_ASSERT(batchSize > 0);
	RG_ASSERT(obs.size() == (size_t)batchSize * obsSize);
	RG_ASSERT(actionMasks.size() == (size_t)batchSize * actionParser->GetActionAmount());

	if (captureDiagnostics) {
		lastBatchSize = batchSize;
		lastObs = obs;
		lastActionMasks = actionMasks;
	}

	std::vector<int> results = {};

	try {
		RG_NO_GRAD;

		auto device = useGPU ? torch::kCUDA : torch::kCPU;

		auto tObs = torch::tensor(obs).reshape({(int64_t)batchSize, obsSize});
		auto tActionMasks = torch::tensor(actionMasks).reshape({(int64_t)batchSize, this->actionParser->GetActionAmount()});

		tObs = tObs.to(device);
		tActionMasks = tActionMasks.to(device);

		torch::Tensor tActions;
		if (captureDiagnostics) {
			torch::Tensor tProbs = ModelManager::InferPolicyProbsFromModels(*models, tObs, tActionMasks, temperature, false);
			lastActionProbs = TENSOR_TO_VEC<float>(tProbs.flatten());

			tActions = deterministic
				? tProbs.argmax(1)
				: torch::multinomial(tProbs, 1, true).flatten();
		} else {
			torch::Tensor tLogProbs;
			ModelManager::InferActionsFromModels(*models, tObs, tActionMasks, deterministic, temperature, false, &tActions, &tLogProbs);
		}

		auto actionIndices = TENSOR_TO_VEC<int>(tActions);
		results.insert(results.end(), actionIndices.begin(), actionIndices.end());

	} catch (std::exception& e) {
		RG_ERR_CLOSE("InferUnit: Exception when inferring model: " << e.what());
	}

	return results;
}

std::vector<RLGC::Action> GGL::InferUnit::BatchInferActions(const std::vector<RLGC::Player>& players, const std::vector<RLGC::GameState>& states, bool deterministic, float temperature) {
	std::vector<int> actionIndices = BatchInferActionIndices(players, states, deterministic, temperature);
	std::vector<RLGC::Action> results = {};
	results.reserve(actionIndices.size());

	for (int i = 0; i < (int)actionIndices.size(); i++)
		results.push_back(actionParser->ParseAction(actionIndices[i], players[i], states[i]));

	return results;
}