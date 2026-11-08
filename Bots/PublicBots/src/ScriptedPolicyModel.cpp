#include "ScriptedPolicyModel.h"

#include <stdexcept>

namespace
{
	GGL::ModelConfig MakeScriptedConfig(int obsSize, int numActions)
	{
		GGL::PartialModelConfig partial = {};
		partial.layerSizes = { 1 };
		GGL::ModelConfig config(partial);
		config.numInputs = obsSize;
		config.numOutputs = numActions;
		return config;
	}
}

GGL::ScriptedPolicyModel::ScriptedPolicyModel(
	std::string artifactName,
	int obsSize,
	int numActions,
	torch::Device device,
	ScriptedPolicyInput inputLayout,
	int attentionEntities)
	: Model(),
	artifactName(std::move(artifactName)),
	inputLayout(inputLayout),
	attentionEntities(attentionEntities)
{
	modelName = "policy";
	this->device = device;
	config = MakeScriptedConfig(obsSize, numActions);

	if (inputLayout == ScriptedPolicyInput::PackedAttention) {
		const int expectedObsSize = 32 + attentionEntities * 24 + attentionEntities;
		if (attentionEntities <= 0 || obsSize != expectedObsSize)
			throw std::invalid_argument("ScriptedPolicyModel: invalid packed-attention dimensions");
	}
}

torch::Tensor GGL::ScriptedPolicyModel::Forward(torch::Tensor input, bool)
{
	if (!loaded)
		throw std::runtime_error("ScriptedPolicyModel: policy has not been loaded");
	if (input.dim() != 2 || input.size(1) != config.numInputs)
		throw std::runtime_error("ScriptedPolicyModel: policy input shape does not match the registered observation size");

	const int64_t batchSize = input.size(0);
	torch::Tensor logits;
	if (inputLayout == ScriptedPolicyInput::Flat) {
		logits = module.forward({ input }).toTensor();
	} else {
		torch::Tensor q = input.slice(1, 0, 32).reshape({ batchSize, 1, 32 });
		const int64_t kvEnd = 32 + (int64_t)attentionEntities * 24;
		torch::Tensor kv = input.slice(1, 32, kvEnd).reshape({ batchSize, attentionEntities, 24 });
		torch::Tensor mask = input.slice(1, kvEnd).reshape({ batchSize, attentionEntities });
		logits = module.forward({
			c10::ivalue::Tuple::create({
				std::move(q),
				std::move(kv),
				std::move(mask)
			})
		}).toTensor();
	}

	if (logits.dim() != 2
		|| logits.size(0) != batchSize
		|| logits.size(1) != config.numOutputs) {
		throw std::runtime_error(
			"ScriptedPolicyModel: policy output shape does not match the registered action count");
	}
	return logits;
}

torch::Tensor GGL::ScriptedPolicyModel::ForwardFeatures(torch::Tensor input, bool halfPrec)
{
	return Forward(std::move(input), halfPrec);
}

void GGL::ScriptedPolicyModel::Save(std::filesystem::path)
{
	throw std::runtime_error("ScriptedPolicyModel: inference-only policies cannot be saved as trainable models");
}

void GGL::ScriptedPolicyModel::Load(std::filesystem::path folder, bool allowNotExist)
{
	const std::filesystem::path path = folder / artifactName;
	if (!std::filesystem::exists(path)) {
		if (allowNotExist)
			return;
		throw std::runtime_error("ScriptedPolicyModel: artifact not found: " + path.string());
	}

	module = torch::jit::load(path.string(), device);
	module.eval();
	loaded = true;
}

torch::Tensor GGL::ScriptedPolicyModel::CopyParams() const
{
	throw std::runtime_error("ScriptedPolicyModel: inference-only policies do not expose trainable parameters");
}
