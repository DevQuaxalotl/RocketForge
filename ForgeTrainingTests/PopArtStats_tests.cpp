#include <gtest/gtest.h>

#include <GigaLearnCPP/PPO/PopArtStats.h>

#include <torch/nn/modules/linear.h>

#include <cmath>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using namespace GGL;

namespace {
	bool Near(float a, float b, float tolerance = 1e-5f) {
		return std::fabs(a - b) <= tolerance;
	}

	torch::Tensor Ones(int64_t samples, int64_t heads) {
		return torch::ones({ samples, heads });
	}
}

TEST(PopArtStats, UninitializedHeadsUseUnitScale) {
	PopArtStats stats({ "A", "B" }, 0.02f, 0.01f);
	torch::Tensor normalized = torch::tensor(std::vector<float>{ 0.5f, -2.f }).reshape({ 1, 2 });

	torch::Tensor raw = stats.Denormalize(normalized);

	ASSERT_TRUE(Near(raw[0][0].item<float>(), 0.5f));
	ASSERT_TRUE(Near(raw[0][1].item<float>(), -2.f));
	ASSERT_EQ(stats.CountHeadsAtFloor(), 0);
}

TEST(PopArtStats, FirstBatchSetsStatsDirectlyAndPreservesNothing) {
	PopArtStats stats({ "A", "B" }, 0.02f, 0.01f);
	torch::Tensor targets = torch::tensor(std::vector<float>{ 1.f, 10.f, 3.f, 30.f }).reshape({ 2, 2 });

	PopArtUpdateResult update = stats.Update(targets, Ones(2, 2));

	ASSERT_TRUE(Near(stats.GetMeans()[0].item<float>(), 2.f));
	ASSERT_TRUE(Near(stats.GetMeans()[1].item<float>(), 20.f));
	ASSERT_TRUE(Near(stats.GetSigmas()[0].item<float>(), 1.f));
	ASSERT_TRUE(Near(stats.GetSigmas()[1].item<float>(), 10.f));
	ASSERT_FALSE(update.preserveMask[0].item<bool>());
	ASSERT_FALSE(update.preserveMask[1].item<bool>());
}

TEST(PopArtStats, LaterBatchesMoveByBeta) {
	PopArtStats stats({ "A" }, 0.5f, 0.01f);
	stats.Update(torch::tensor(std::vector<float>{ 0.f, 2.f }).reshape({ 2, 1 }), Ones(2, 1));

	// First batch: mean 1, meanSq 2. Second batch: mean 4, meanSq 16. Halfway: mean 2.5, meanSq 9.
	PopArtUpdateResult update = stats.Update(torch::tensor(std::vector<float>{ 4.f, 4.f }).reshape({ 2, 1 }), Ones(2, 1));

	ASSERT_TRUE(update.preserveMask[0].item<bool>());
	ASSERT_TRUE(Near(stats.GetMeans()[0].item<float>(), 2.5f));
	ASSERT_TRUE(Near(stats.GetSigmas()[0].item<float>(), std::sqrt(9.f - 6.25f)));
}

TEST(PopArtStats, AbsentCellsAreIgnored) {
	PopArtStats stats({ "A" }, 0.02f, 0.01f);
	torch::Tensor targets = torch::tensor(std::vector<float>{ 2.f, 4.f, 1000.f }).reshape({ 3, 1 });
	torch::Tensor presence = torch::tensor(std::vector<float>{ 1.f, 1.f, 0.f }).reshape({ 3, 1 });

	stats.Update(targets, presence);

	ASSERT_TRUE(Near(stats.GetMeans()[0].item<float>(), 3.f));
	ASSERT_TRUE(Near(stats.GetSigmas()[0].item<float>(), 1.f));
}

TEST(PopArtStats, HeadWithNoPresentCellsIsUnchanged) {
	PopArtStats stats({ "A", "B" }, 0.5f, 0.01f);
	stats.Update(torch::tensor(std::vector<float>{ 1.f, 5.f, 3.f, 5.f }).reshape({ 2, 2 }), Ones(2, 2));
	const float meanBefore = stats.GetMeans()[1].item<float>();

	torch::Tensor presence = torch::tensor(std::vector<float>{ 1.f, 0.f, 1.f, 0.f }).reshape({ 2, 2 });
	stats.Update(torch::tensor(std::vector<float>{ 9.f, 99.f, 9.f, 99.f }).reshape({ 2, 2 }), presence);

	ASSERT_TRUE(Near(stats.GetMeans()[1].item<float>(), meanBefore));
	ASSERT_FALSE(std::isnan(stats.GetSigmas()[1].item<float>()));
}

TEST(PopArtStats, SigmaIsFlooredAndCounted) {
	PopArtStats stats({ "Constant", "Untouched" }, 0.02f, 0.01f);
	torch::Tensor presence = torch::tensor(std::vector<float>{ 1.f, 0.f, 1.f, 0.f }).reshape({ 2, 2 });

	stats.Update(torch::tensor(std::vector<float>{ 5.f, 0.f, 5.f, 0.f }).reshape({ 2, 2 }), presence);

	ASSERT_TRUE(Near(stats.GetSigmas()[0].item<float>(), 0.01f));
	ASSERT_EQ(stats.CountHeadsAtFloor(), 1);
}

TEST(PopArtStats, PreserveOutputsKeepsRawPredictions) {
	torch::manual_seed(7);
	torch::nn::Linear outputLayer(4, 2);
	torch::Tensor inputs = torch::randn({ 16, 4 });
	PopArtStats stats({ "A", "B" }, 0.3f, 0.01f);
	stats.Update(torch::randn({ 32, 2 }) * 5.f + 3.f, Ones(32, 2));

	torch::Tensor before = stats.Denormalize(outputLayer->forward(inputs)).detach();
	PopArtUpdateResult update = stats.Update(torch::randn({ 32, 2 }) * 40.f - 20.f, Ones(32, 2));
	stats.PreserveOutputs(outputLayer->weight, outputLayer->bias, update);
	torch::Tensor after = stats.Denormalize(outputLayer->forward(inputs)).detach();

	const float maxError = (after - before).abs().max().item<float>();
	const float scale = before.abs().max().item<float>() + 1.f;
	ASSERT_TRUE(maxError <= 1e-4f * scale);
}

TEST(PopArtStats, FirstInitializationLeavesRowsAndPredictsTheMean) {
	torch::nn::Linear outputLayer(3, 1);
	{
		torch::NoGradGuard noGrad;
		outputLayer->weight.zero_();
		outputLayer->bias.zero_();
	}
	PopArtStats stats({ "A" }, 0.02f, 0.01f);

	PopArtUpdateResult update = stats.Update(torch::tensor(std::vector<float>{ 6.f, 8.f }).reshape({ 2, 1 }), Ones(2, 1));
	stats.PreserveOutputs(outputLayer->weight, outputLayer->bias, update);

	torch::Tensor predictions = stats.Denormalize(outputLayer->forward(torch::randn({ 5, 3 }))).detach();
	ASSERT_TRUE(Near(predictions.min().item<float>(), 7.f));
	ASSERT_TRUE(Near(predictions.max().item<float>(), 7.f));
}

TEST(PopArtStats, NormalizeInvertsDenormalize) {
	PopArtStats stats({ "A", "B" }, 0.02f, 0.01f);
	stats.Update(torch::randn({ 20, 2 }) * 3.f + 1.f, Ones(20, 2));
	torch::Tensor raw = torch::randn({ 6, 2 }) * 10.f;

	torch::Tensor roundTrip = stats.Denormalize(stats.Normalize(raw));

	ASSERT_TRUE((roundTrip - raw).abs().max().item<float>() < 1e-4f);
}

TEST(PopArtStats, HeadStatesRoundTripThroughJson) {
	PopArtStats stats({ "A", "B" }, 0.02f, 0.01f);
	stats.Update(torch::tensor(std::vector<float>{ 1.f, 2.f, 3.f, 6.f }).reshape({ 2, 2 }), Ones(2, 2));

	nlohmann::json j;
	std::vector<PopArtHeadState> saved = stats.GetHeads();
	JsonArchive saver = JsonArchive::MakeSaver(j);
	saver.Field("value_heads", saved);

	std::vector<PopArtHeadState> loaded = {};
	JsonArchive loader = JsonArchive::MakeLoader(j);
	loader.Field("value_heads", loaded);

	ASSERT_TRUE(loader.GetErrors().empty());
	ASSERT_EQ(loaded.size(), (size_t)2);
	ASSERT_EQ(loaded[1].name, std::string("B"));
	ASSERT_TRUE(loaded[1].initialized);
	ASSERT_EQ(loaded[1].mean, saved[1].mean);
	ASSERT_EQ(loaded[1].meanSq, saved[1].meanSq);
}

TEST(PopArtStats, RemapFromKeepsStatsByName) {
	PopArtStats saved({ "A", "B" }, 0.02f, 0.01f);
	saved.Update(torch::tensor(std::vector<float>{ 1.f, 10.f, 3.f, 30.f }).reshape({ 2, 2 }), Ones(2, 2));

	PopArtStats remapped({ "B", "C" }, 0.02f, 0.01f);
	remapped.RemapFrom(saved.GetHeads());

	ASSERT_EQ(remapped.GetNumHeads(), 2);
	ASSERT_EQ(remapped.GetHeads()[0].name, std::string("B"));
	ASSERT_TRUE(remapped.GetHeads()[0].initialized);
	ASSERT_EQ(remapped.GetHeads()[0].mean, saved.GetHeads()[1].mean);
	ASSERT_EQ(remapped.GetHeads()[0].meanSq, saved.GetHeads()[1].meanSq);
	ASSERT_EQ(remapped.GetHeads()[1].name, std::string("C"));
	ASSERT_FALSE(remapped.GetHeads()[1].initialized);
}

TEST(PopArtStats, RemapFromResetsHeadsNotInCheckpoint) {
	PopArtStats saved({ "A" }, 0.02f, 0.01f);
	saved.Update(torch::tensor(std::vector<float>{ 2.f, 4.f }).reshape({ 2, 1 }), Ones(2, 1));

	// "C" has stats of its own before the remap; the checkpoint has none for it, so they must not survive.
	PopArtStats remapped({ "A", "C" }, 0.02f, 0.01f);
	remapped.Update(torch::tensor(std::vector<float>{ 7.f, 7.f, 9.f, 9.f }).reshape({ 2, 2 }), Ones(2, 2));
	remapped.RemapFrom(saved.GetHeads());

	ASSERT_EQ(remapped.GetHeads()[0].mean, saved.GetHeads()[0].mean);
	ASSERT_FALSE(remapped.GetHeads()[1].initialized);
	ASSERT_EQ(remapped.GetHeads()[1].mean, 0.0);
	ASSERT_EQ(remapped.GetHeads()[1].meanSq, 0.0);
	ASSERT_EQ(remapped.GetHeads()[1].name, std::string("C"));
}
