#include "GigaLearnInfer/Models.h"
#include "GigaLearnInfer/ModelManager.h"

#include <torch/csrc/api/include/torch/serialize.h>
#include <torch/csrc/api/include/torch/nn/utils/convert_parameters.h>
#include <torch/nn/modules/normalization.h>

#include <iterator>

GGL::Model::Model(
	const char* modelName,
	ModelConfig config,
	torch::Device device) : 
	modelName(modelName), device(device), seq({})
	, seqHalf({})
	, config(config) {

	if (!config.IsValid())
		RG_ERR_CLOSE("Failed to create model \"" << modelName << "\" with invalid config");

	int lastSize = config.numInputs;
	for (int i = 0; i < config.layerSizes.size(); i++) {
		seq->push_back(torch::nn::Linear(lastSize, config.layerSizes[i]));
		if (config.addLayerNorm)
			seq->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({(int64_t)config.layerSizes[i]})));
		lastSize = config.layerSizes[i];
		AddActivationFunc(seq, config.activationType);
	}
	
	if (config.addOutputLayer) {
		seq->push_back(torch::nn::Linear(lastSize, config.numOutputs));
	} else {
		config.numOutputs = config.layerSizes.back();
	}

	register_module("seq", seq);
	seq->to(device);
}

GGL::Model::~Model() {
	// Evict any CUDA graphs captured from this model so a later allocation at the same
	// address can never replay a graph that references this model's freed weights.
	ModelManager::ReleaseCudaGraphs(this);
}

torch::Tensor GGL::Model::Forward(torch::Tensor input, bool halfPrec) {

	if (torch::GradMode::is_enabled())
		halfPrec = false;

	if (halfPrec) {

		if (_seqHalfOutdated) {
			_seqHalfOutdated = false;

			if (seqHalf->size() == 0) {
				for (auto& mod : *seq)
					seqHalf->push_back(mod.clone());
				seqHalf->to(RG_HALFPERC_TYPE, true);
			} else {
				auto fromParams = seq->parameters();
				auto toParams = seqHalf->parameters();
				for (int i = 0; i < fromParams.size(); i++) {
					auto scaledParams = fromParams[i].to(RG_HALFPERC_TYPE, true);
					toParams[i].copy_(scaledParams, true);
				}
			}
		}
		
		auto halfInput = input.to(RG_HALFPERC_TYPE);
		auto halfOutput = seqHalf->forward(halfInput);
		return halfOutput.to(torch::kFloat);
	} else {
		return seq->forward(input);
	}
}

torch::Tensor GGL::Model::ForwardFeatures(torch::Tensor input, bool halfPrec) {
	if (!config.addOutputLayer)
		return Forward(input, halfPrec);

	if (seq->size() <= 1)
		return input;

	// Features are used by training heads, so keep this path in full precision.
	auto iterator = seq->begin();
	auto value = iterator->any_forward(std::move(input));
	for (++iterator; std::next(iterator) != seq->end(); ++iterator)
		value = iterator->any_forward(std::move(value));

	if (auto* result = value.try_get<torch::Tensor>())
		return std::move(*result);

	RG_ERR_CLOSE("Model::ForwardFeatures(): expected tensor features from model \"" << modelName << "\"");
	return {};
}

int GGL::Model::GetFeatureSize() const {
	if (config.addOutputLayer) {
		if (!config.layerSizes.empty())
			return config.layerSizes.back();
		return config.numInputs;
	}

	return config.numOutputs;
}

// Get sizes of all parameters in a sequence
std::vector<uint64_t> GetSeqSizes(torch::nn::Sequential& seq) {
	std::vector<uint64_t> result = {};

	for (int i = 0; i < seq->size(); i++)
		for (auto param : seq[i]->parameters())
			result.push_back(param.numel());

	return result;
}

void GGL::Model::Save(std::filesystem::path folder) {
	std::filesystem::path path = GetSavePath(folder);
	auto streamOut = std::ofstream(path, std::ios::binary);
	torch::save(seq, streamOut);
}

void GGL::Model::Load(std::filesystem::path folder, bool allowNotExist) {
	std::filesystem::path path = GetSavePath(folder);

	if (!std::filesystem::exists(path)) {
		if (allowNotExist) {
			RG_LOG("Warning: Model \"" << modelName << "\" does not exist in " << folder << " and will be reset");
			return;
		} else {
			RG_ERR_CLOSE("Model \"" << modelName << "\" does not exist in " << folder);
		}
	}

	auto streamIn = std::ifstream(path, std::ios::binary);
	streamIn >> std::noskipws;

	if (!streamIn.good())
		RG_ERR_CLOSE("Failed to load from " << path << ", file does not exist or can't be accessed");

	auto sizesBefore = GetSeqSizes(seq);

	try {
		torch::load(this->seq, streamIn, device);
	} catch (std::exception& e) {
		RG_ERR_CLOSE(
			"Failed to load model \"" << modelName << ", checkpoint may be corrupt or of different model arch.\n" <<
			"Exception: " << e.what()
		);
	}

	// Torch will happily load in a model of a totally different size, then we will crash when we try to use it
	// So we need to manually check if it is the same size
	auto sizesAfter = GetSeqSizes(seq);
	if (!std::equal(sizesBefore.begin(), sizesBefore.end(), sizesAfter.begin(), sizesAfter.end())) {
		std::stringstream stream;
		stream << "Saved model has different size than current model, cannot load model from " << path << ":\n";

		for (int i = 0; i < 2; i++) {
			stream << " > " << (i ? "Saved model:   [ " : "Current model: [ ");
			for (uint64_t size : (i ? sizesAfter : sizesBefore))
				stream << size << ' ';

			stream << " ]";
			if (i == 0)
				stream << ",\n";
		}

		RG_ERR_CLOSE(stream.str());
	}

}

torch::Tensor GGL::Model::CopyParams() const {
	return torch::nn::utils::parameters_to_vector(parameters()).cpu();
}
