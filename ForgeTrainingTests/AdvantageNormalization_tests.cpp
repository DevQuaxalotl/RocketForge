#include <gtest/gtest.h>

#include <GigaLearnCPP/PPO/AdvantageNormalization.h>

#include <cmath>
#include <vector>

namespace {
	constexpr float ADVANTAGE_NORM_EPSILON = 1e-8f;

	bool Near(float a, float b, float tolerance = 1e-5f) {
		return std::fabs(a - b) <= tolerance;
	}
}

TEST(AdvantageNormalization, CentersAndScalesToUnitStd) {
	// mean = 2.5, population std = sqrt(1.25)
	torch::Tensor advantages = torch::tensor(std::vector<float>{ 1.f, 2.f, 3.f, 4.f });
	torch::Tensor normalized = GGL::NormalizeAdvantages(advantages, ADVANTAGE_NORM_EPSILON);
	const float populationStd = std::sqrt(1.25f);

	ASSERT_TRUE(Near(normalized[0].item<float>(), -1.5f / populationStd));
	ASSERT_TRUE(Near(normalized[1].item<float>(), -0.5f / populationStd));
	ASSERT_TRUE(Near(normalized[2].item<float>(), 0.5f / populationStd));
	ASSERT_TRUE(Near(normalized[3].item<float>(), 1.5f / populationStd));
	ASSERT_TRUE(Near(normalized.mean().item<float>(), 0.f));
	ASSERT_TRUE(Near(normalized.std(/*unbiased*/ false).item<float>(), 1.f));

	// The input is left untouched.
	ASSERT_TRUE(Near(advantages[0].item<float>(), 1.f));
}

TEST(AdvantageNormalization, ConstantInputBecomesFiniteZeros) {
	torch::Tensor advantages = torch::full({ 5 }, 3.f);
	torch::Tensor normalized = GGL::NormalizeAdvantages(advantages, ADVANTAGE_NORM_EPSILON);

	ASSERT_TRUE(torch::isfinite(normalized).all().item<bool>());
	ASSERT_TRUE(Near(normalized.abs().max().item<float>(), 0.f));
}

TEST(AdvantageNormalization, FewerThanTwoSamplesPassThrough) {
	torch::Tensor single = torch::tensor(std::vector<float>{ 0.7f });
	torch::Tensor normalizedSingle = GGL::NormalizeAdvantages(single, ADVANTAGE_NORM_EPSILON);
	ASSERT_EQ(normalizedSingle.numel(), (int64_t)1);
	ASSERT_TRUE(Near(normalizedSingle[0].item<float>(), 0.7f));

	torch::Tensor empty = torch::empty({ 0 });
	ASSERT_EQ(GGL::NormalizeAdvantages(empty, ADVANTAGE_NORM_EPSILON).numel(), (int64_t)0);
}

TEST(AdvantageNormalization, WholeBatchStatsDifferFromPerChunkStats) {
	// Microbatches must slice one whole-batch normalization; normalizing each chunk on its own gives different values.
	torch::Tensor advantages = torch::tensor(std::vector<float>{ 0.1f, -0.4f, 0.9f, 0.0f, 2.0f, -1.2f, 0.3f, 0.5f, -0.7f, 1.1f });
	torch::Tensor wholeBatch = GGL::NormalizeAdvantages(advantages, ADVANTAGE_NORM_EPSILON);

	const std::vector<std::pair<int64_t, int64_t>> chunkBounds = { { 0, 3 }, { 3, 7 }, { 7, 10 } };
	bool anyChunkDiffers = false;
	for (const auto& [start, stop] : chunkBounds) {
		torch::Tensor perChunk = GGL::NormalizeAdvantages(advantages.slice(0, start, stop), ADVANTAGE_NORM_EPSILON);
		if (!torch::allclose(perChunk, wholeBatch.slice(0, start, stop)))
			anyChunkDiffers = true;
	}

	ASSERT_TRUE(anyChunkDiffers);
}
