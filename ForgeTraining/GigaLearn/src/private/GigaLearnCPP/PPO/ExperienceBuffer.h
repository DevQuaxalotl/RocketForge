#pragma once
#include "GigaLearnInfer/FrameworkTorch.h"

namespace GGL {

	struct ExperienceTensors {
		torch::Tensor
			states, actions, logProbs, targetValues, actionMasks, criticLossWeights, advantages;

		auto begin() { return &states; }
		auto end() { return &advantages + 1; }
		auto begin() const { return &states; }
		auto end() const { return &advantages + 1; }
	};

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/experience_buffer.py
	class ExperienceBuffer {
	public:

		torch::Device device;
		int seed;

		ExperienceTensors data;

		std::default_random_engine rng;

		GGL_TRAIN_IMEXPORT ExperienceBuffer(int seed, torch::Device device);

		GGL_TRAIN_IMEXPORT ExperienceTensors _GetSamples(const torch::Tensor& indices) const;

		// Not const because it uses our random engine
		GGL_TRAIN_IMEXPORT std::vector<ExperienceTensors> GetAllBatchesShuffled(int64_t batchSize, bool overbatching);
	};
}