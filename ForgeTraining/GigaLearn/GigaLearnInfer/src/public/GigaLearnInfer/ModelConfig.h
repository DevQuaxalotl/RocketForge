#pragma once

#include <GigaLearnInfer/Framework.h>

namespace GGL {
	enum class ModelOptimType {
		ADAM,
		ADAMW,
		ADAGRAD,
		RMSPROP,
		MAGSGD
	};

	enum class ModelActivationType {
		RELU,
		LEAKY_RELU,
		SIGMOID,
		TANH
	};

	// Doesn't include inputs or outputs
	struct PartialModelConfig {
		std::vector<int> layerSizes = {};
		ModelActivationType activationType = ModelActivationType::RELU;
		ModelOptimType optimType = ModelOptimType::ADAM;
		bool addLayerNorm = true;
		bool addOutputLayer = true;

		bool IsValid() const {
			return !layerSizes.empty();
		}
	};

	struct ModelConfig : PartialModelConfig {
		int numInputs = -1;
		int numOutputs = -1;

		bool IsValid() const {
			return numInputs > 0 &&
				((addOutputLayer && numOutputs > 0) || (!addOutputLayer && PartialModelConfig::IsValid()));
		}

		ModelConfig(const PartialModelConfig& partialConfig) : PartialModelConfig(partialConfig) {}
	};
}