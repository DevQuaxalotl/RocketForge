#include <gtest/gtest.h>

#include <GigaLearnCPP/PPO/GAE.h>
#include <RLGymCPP/TerminalConditions/TerminalCondition.h>

#include <cmath>
#include <vector>

TEST(GAE, NormalTerminalDoesNotRequireNextValuePrediction) {
	torch::Tensor rewards = torch::tensor(std::vector<float>{ 3.f });
	torch::Tensor terminals = torch::tensor(std::vector<int8_t>{ (int8_t)RLGC::TerminalType::NORMAL });
	torch::Tensor valPreds = torch::tensor(std::vector<float>{ 5.f });

	torch::Tensor advantages;
	torch::Tensor targetValues;
	torch::Tensor returns;
	float rewClipPortion = 0.f;

	GGL::GAE::Compute(
		rewards, terminals, valPreds, torch::Tensor(),
		advantages, targetValues, returns, rewClipPortion,
		1.f, 1.f, 0.f, 0.f
	);

	ASSERT_TRUE(std::abs(advantages[0].item<float>() + 2.f) < 0.0001f);
	ASSERT_TRUE(std::abs(targetValues[0].item<float>() - 3.f) < 0.0001f);
	ASSERT_TRUE(std::abs(returns[0].item<float>() - 3.f) < 0.0001f);
}

TEST(GAE, PropagatesWithinEpisodesAndStopsAtNormalTerminals) {
	torch::Tensor rewards = torch::tensor(std::vector<float>{ 1.f, 2.f, 3.f, 4.f });
	torch::Tensor terminals = torch::tensor(std::vector<int8_t>{
		(int8_t)RLGC::TerminalType::NOT_TERMINAL,
		(int8_t)RLGC::TerminalType::NORMAL,
		(int8_t)RLGC::TerminalType::NOT_TERMINAL,
		(int8_t)RLGC::TerminalType::NORMAL
	});
	torch::Tensor valPreds = torch::zeros({ 4 });

	torch::Tensor advantages;
	torch::Tensor targetValues;
	torch::Tensor returns;
	float rewClipPortion = 0.f;

	GGL::GAE::Compute(
		rewards, terminals, valPreds, torch::Tensor(),
		advantages, targetValues, returns, rewClipPortion,
		1.f, 1.f, 1.f, 0.f
	);

	ASSERT_TRUE(std::abs(advantages[0].item<float>() - 3.f) < 0.0001f);
	ASSERT_TRUE(std::abs(advantages[1].item<float>() - 2.f) < 0.0001f);
	ASSERT_TRUE(std::abs(advantages[2].item<float>() - 7.f) < 0.0001f);
	ASSERT_TRUE(std::abs(advantages[3].item<float>() - 4.f) < 0.0001f);
	ASSERT_TRUE(std::abs(returns[0].item<float>() - 3.f) < 0.0001f);
	ASSERT_TRUE(std::abs(returns[2].item<float>() - 7.f) < 0.0001f);
}

TEST(GAE, PropagatesTruncationBootstrapWithinEpisodeOnly) {
	torch::Tensor rewards = torch::zeros({ 4 });
	torch::Tensor terminals = torch::tensor(std::vector<int8_t>{
		(int8_t)RLGC::TerminalType::NOT_TERMINAL,
		(int8_t)RLGC::TerminalType::TRUNCATED,
		(int8_t)RLGC::TerminalType::NOT_TERMINAL,
		(int8_t)RLGC::TerminalType::TRUNCATED
	});
	torch::Tensor valPreds = torch::zeros({ 4 });
	torch::Tensor truncValPreds = torch::tensor(std::vector<float>{ 10.f, 20.f });

	torch::Tensor advantages;
	torch::Tensor targetValues;
	torch::Tensor returns;
	float rewClipPortion = 0.f;

	GGL::GAE::Compute(
		rewards, terminals, valPreds, truncValPreds,
		advantages, targetValues, returns, rewClipPortion,
		1.f, 1.f, 1.f, 0.f
	);

	ASSERT_TRUE(std::abs(advantages[0].item<float>() - 10.f) < 0.0001f);
	ASSERT_TRUE(std::abs(advantages[1].item<float>() - 10.f) < 0.0001f);
	ASSERT_TRUE(std::abs(advantages[2].item<float>() - 20.f) < 0.0001f);
	ASSERT_TRUE(std::abs(advantages[3].item<float>() - 20.f) < 0.0001f);
	ASSERT_TRUE(std::abs(returns[0].item<float>()) < 0.0001f);
	ASSERT_TRUE(std::abs(returns[2].item<float>()) < 0.0001f);
}

TEST(GAE, UnstandardizedRewardsReportNoClipPortion) {
	torch::Tensor rewards = torch::tensor(std::vector<float>{ 3.f });
	torch::Tensor terminals = torch::tensor(std::vector<int8_t>{ (int8_t)RLGC::TerminalType::NORMAL });
	torch::Tensor valPreds = torch::zeros({ 1 });

	torch::Tensor advantages;
	torch::Tensor targetValues;
	torch::Tensor returns;
	float rewClipPortion = 0.f;

	GGL::GAE::Compute(
		rewards, terminals, valPreds, torch::Tensor(),
		advantages, targetValues, returns, rewClipPortion,
		1.f, 1.f, 0.f, 1.f
	);

	ASSERT_TRUE(std::abs(rewClipPortion) < 0.0001f);
}

TEST(GAE, AllowsExplicitBootstrapValueForIncompleteFinalStep) {
	torch::Tensor rewards = torch::tensor(std::vector<float>{ 1.f });
	torch::Tensor terminals = torch::tensor(std::vector<int8_t>{ (int8_t)RLGC::TerminalType::NOT_TERMINAL });
	torch::Tensor valPreds = torch::tensor(std::vector<float>{ 2.f, 5.f });

	torch::Tensor advantages;
	torch::Tensor targetValues;
	torch::Tensor returns;
	float rewClipPortion = 0.f;

	GGL::GAE::Compute(
		rewards, terminals, valPreds, torch::Tensor(),
		advantages, targetValues, returns, rewClipPortion,
		1.f, 1.f, 1.f, 0.f
	);

	ASSERT_TRUE(std::abs(advantages[0].item<float>() - 4.f) < 0.0001f);
	ASSERT_TRUE(std::abs(targetValues[0].item<float>() - 6.f) < 0.0001f);
}
