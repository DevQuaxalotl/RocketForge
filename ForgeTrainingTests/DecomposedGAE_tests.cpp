#include <gtest/gtest.h>

#include <GigaLearnCPP/PPO/DecomposedGAE.h>
#include <GigaLearnCPP/PPO/GAE.h>
#include <RLGymCPP/TerminalConditions/TerminalCondition.h>

#include <vector>

using namespace GGL;

namespace {
	bool AllClose(const torch::Tensor& a, const torch::Tensor& b, float tolerance = 1e-4f) {
		return (a - b).abs().max().item<float>() <= tolerance;
	}

	// Three episodes: arena A ends normally, arena B truncates, arena A ends normally.
	// Each episode keeps its arena's weights, matching how trajectories are collected.
	struct MixedBatch {
		torch::Tensor headRewards, terminals, headValPreds, headTruncValPreds, sampleWeights, truncWeights;

		MixedBatch() {
			torch::manual_seed(11);
			constexpr int8_t NOT_TERMINAL = (int8_t)RLGC::TerminalType::NOT_TERMINAL;
			constexpr int8_t NORMAL = (int8_t)RLGC::TerminalType::NORMAL;
			constexpr int8_t TRUNCATED = (int8_t)RLGC::TerminalType::TRUNCATED;

			headRewards = torch::randn({ 7, 3 });
			terminals = torch::tensor(std::vector<int8_t>{
				NOT_TERMINAL, NOT_TERMINAL, NORMAL,
				NOT_TERMINAL, TRUNCATED,
				NOT_TERMINAL, NORMAL });
			headValPreds = torch::randn({ 7, 3 }) * 4.f;
			headTruncValPreds = torch::randn({ 1, 3 }) * 4.f;

			const torch::Tensor weightsA = torch::tensor(std::vector<float>{ 2.f, 0.5f, 0.f });
			const torch::Tensor weightsB = torch::tensor(std::vector<float>{ 7.f, 0.f, 3.f });
			sampleWeights = torch::stack({ weightsA, weightsA, weightsA, weightsB, weightsB, weightsA, weightsA });
			truncWeights = weightsB.unsqueeze(0);
		}
	};
}

TEST(DecomposedGAE, CombinedMatchesSingleStreamGAE) {
	MixedBatch batch;

	DecomposedGAEResult decomposed = ComputeDecomposedGAE(
		batch.headRewards, batch.terminals, batch.headValPreds, batch.headTruncValPreds, batch.sampleWeights, 0.99f, 0.95f);

	torch::Tensor advantages, targetValues, returns;
	float rewardClipPortion = 0.f;
	GAE::Compute(
		(batch.sampleWeights * batch.headRewards).sum(1), batch.terminals,
		(batch.sampleWeights * batch.headValPreds).sum(1), (batch.truncWeights * batch.headTruncValPreds).sum(1),
		advantages, targetValues, returns, rewardClipPortion,
		0.99f, 0.95f, 0.f, 0.f);

	ASSERT_TRUE(AllClose(decomposed.advantages, advantages));
	ASSERT_TRUE(AllClose(decomposed.targetValues, targetValues));
	ASSERT_TRUE(AllClose(decomposed.returns, returns));
	ASSERT_TRUE(AllClose(decomposed.valPreds, (batch.sampleWeights * batch.headValPreds).sum(1)));
}

TEST(DecomposedGAE, HeadColumnsAreUnweightedPerHeadGAE) {
	MixedBatch batch;

	DecomposedGAEResult decomposed = ComputeDecomposedGAE(
		batch.headRewards, batch.terminals, batch.headValPreds, batch.headTruncValPreds, batch.sampleWeights, 0.99f, 0.95f);

	// Head 2 has weight 0 in arena A, but its own column must still be its full unweighted GAE.
	torch::Tensor advantages, targetValues, returns;
	float rewardClipPortion = 0.f;
	GAE::Compute(
		batch.headRewards.select(1, 2).contiguous(), batch.terminals,
		batch.headValPreds.select(1, 2).contiguous(), batch.headTruncValPreds.select(1, 2).contiguous(),
		advantages, targetValues, returns, rewardClipPortion,
		0.99f, 0.95f, 0.f, 0.f);

	ASSERT_TRUE(AllClose(decomposed.headAdvantages.select(1, 2), advantages));
	ASSERT_TRUE(AllClose(decomposed.headTargets.select(1, 2), targetValues));
}

TEST(DecomposedGAE, AcceptsBatchWithoutTruncations) {
	torch::Tensor terminals = torch::tensor(std::vector<int8_t>{
		(int8_t)RLGC::TerminalType::NOT_TERMINAL, (int8_t)RLGC::TerminalType::NORMAL });

	DecomposedGAEResult decomposed = ComputeDecomposedGAE(
		torch::ones({ 2, 2 }), terminals, torch::zeros({ 2, 2 }), torch::Tensor(), torch::ones({ 2, 2 }), 1.f, 1.f);

	ASSERT_EQ(decomposed.headAdvantages.size(0), (int64_t)2);
	ASSERT_EQ(decomposed.headAdvantages.size(1), (int64_t)2);
	ASSERT_TRUE(AllClose(decomposed.advantages, torch::tensor(std::vector<float>{ 4.f, 2.f })));
}
