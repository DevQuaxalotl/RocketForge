#include "AdvantageNormalization.h"

torch::Tensor GGL::NormalizeAdvantages(const torch::Tensor& advantages, float epsilon)
{
	if (advantages.numel() < 2)
		return advantages;

	const torch::Tensor centered = advantages - advantages.mean();

	return centered / (advantages.std(/*unbiased*/ false) + epsilon);
}
