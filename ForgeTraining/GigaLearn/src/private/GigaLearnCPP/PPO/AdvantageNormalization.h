#pragma once

#include <GigaLearnInfer/FrameworkTorch.h>

namespace GGL {
	/**
	 * Returns (advantages - mean) / (populationStd + epsilon).
	 * Inputs with fewer than 2 elements have no meaningful spread, so the input itself is returned without normalizing.
	 */
	torch::Tensor NormalizeAdvantages(const torch::Tensor& advantages, float epsilon);
}
