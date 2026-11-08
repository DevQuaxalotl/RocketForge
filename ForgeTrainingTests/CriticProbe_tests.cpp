#include <gtest/gtest.h>

#include "TestSupport.h"

#include <DebugRender/CriticProbe.h>

#include <GigaLearnInfer/Agent.h>
#include <GigaLearnInfer/ModelManager.h>
#include <GigaLearnInfer/Models.h>
#include <RLGymCPP/Rewards/Reward.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace RocketForge;

namespace fs = std::filesystem;

namespace {
	constexpr int OBS_SIZE = 5;
	constexpr int NUM_ACTIONS = 3;

	bool Near(float a, float b) { return std::abs(a - b) < 1e-4f; }

	class TinyAgent : public Agent {
	public:
		TinyAgent() {
			sharedHead.layerSizes = {};
			policy.layerSizes = { 8 };
			critic.layerSizes = { 8 };
		}
		int GetObsSize() const override { return OBS_SIZE; }
	};

	class NamedReward : public RLGC::Reward {
	public:
		explicit NamedReward(std::string name) : _name(std::move(name)) {}
		std::string GetName() override { return _name; }
		float GetReward(const RLGC::Player&, const RLGC::GameState&, bool) override { return 0.f; }
	private:
		std::string _name;
	};

	struct TempCheckpoint {
		fs::path folder;

		explicit TempCheckpoint(const std::string& name) : folder(RocketForgeTests::TestTempDir(name)) {
			fs::remove_all(folder);
			fs::create_directories(folder);
		}
		~TempCheckpoint() { fs::remove_all(folder); }

		void WriteStats(const std::string& json) const {
			std::ofstream out(folder / "RUNNING_STATS.json");
			out << json;
		}

		void SaveModels(const TinyAgent& agent, int numCriticOutputs) const {
			GGL::ModelSet models;
			const bool makeCritic = true;
			GGL::ModelManager::MakeModels(
				makeCritic, OBS_SIZE, NUM_ACTIONS, agent.sharedHead, agent.policy, agent.critic,
				torch::Device(torch::kCPU), models, numCriticOutputs);
			models.Save(folder);
			models.Free();
		}

		std::vector<float> RawCriticOutputs(const TinyAgent& agent, int numCriticOutputs, const std::vector<float>& obs) const {
			GGL::ModelSet models;
			const bool makeCritic = true;
			GGL::ModelManager::MakeModels(
				makeCritic, OBS_SIZE, NUM_ACTIONS, agent.sharedHead, agent.policy, agent.critic,
				torch::Device(torch::kCPU), models, numCriticOutputs);
			const bool allowNotExist = false;
			models.Load(folder, allowNotExist);

			const bool useHalfPrecision = false;
			torch::Tensor outputs = GGL::ModelManager::InferCriticHeads(
				torch::tensor(obs).reshape({ 1, OBS_SIZE }), models, useHalfPrecision).flatten();
			std::vector<float> values = GGL::TENSOR_TO_VEC<float>(outputs);
			models.Free();
			return values;
		}
	};

	const char* TWO_HEAD_STATS = R"({
		"total_timesteps": 0, "total_iterations": 0,
		"value_heads": [
			{ "name": "touch", "initialized": true, "mean": 100.0, "mean_sq": 10004.0 },
			{ "name": "gone",  "initialized": false, "mean": 0.0, "mean_sq": 0.0 }
		]
	})";

	CriticProbeLoadConfig MakeConfig(const TinyAgent& agent, const fs::path& folder, const std::vector<RLGC::WeightedReward>& rewards) {
		CriticProbeLoadConfig config;
		config.agent = &agent;
		config.checkpointFolder = folder;
		config.numActions = NUM_ACTIONS;
		config.popArtSigmaFloor = 0.01f;
		config.rewards = &rewards;
		config.useGPU = false;
		return config;
	}
}

TEST(CriticProbe, LoadStatsReadsValueHeadsAndReturnStat) {
	TempCheckpoint checkpoint("stats");
	checkpoint.WriteStats(R"({
		"value_heads": [ { "name": "a", "initialized": true, "mean": 1.5, "mean_sq": 4.0 } ],
		"return_stat": { "mean": 0.0, "var": 18.0, "count": 3 }
	})");

	std::optional<CriticCheckpointStats> stats = LoadCriticCheckpointStats(checkpoint.folder);

	ASSERT_TRUE(stats.has_value());
	ASSERT_EQ(stats->valueHeads.size(), size_t(1));
	ASSERT_EQ(stats->valueHeads[0].name, std::string("a"));
	ASSERT_TRUE(stats->valueHeads[0].initialized);
	ASSERT_EQ(stats->valueHeads[0].mean, 1.5);
	ASSERT_EQ(stats->valueHeads[0].meanSq, 4.0);
	ASSERT_TRUE(Near(stats->returnStd, 3.f));
}

TEST(CriticProbe, LoadStatsWithoutReturnStatUsesUnitScale) {
	TempCheckpoint checkpoint("no_return_stat");
	checkpoint.WriteStats(R"({ "total_timesteps": 10 })");

	std::optional<CriticCheckpointStats> stats = LoadCriticCheckpointStats(checkpoint.folder);

	ASSERT_TRUE(stats.has_value());
	ASSERT_TRUE(stats->valueHeads.empty());
	ASSERT_TRUE(Near(stats->returnStd, 1.f));
}

TEST(CriticProbe, LoadStatsRejectsMissingOrUnparsableFile) {
	TempCheckpoint checkpoint("bad_stats");
	ASSERT_FALSE(LoadCriticCheckpointStats(checkpoint.folder).has_value());

	checkpoint.WriteStats("{ not json");
	ASSERT_FALSE(LoadCriticCheckpointStats(checkpoint.folder).has_value());
}

TEST(CriticProbe, TryLoadWithoutCriticFileReturnsNull) {
	TinyAgent agent;
	TempCheckpoint checkpoint("no_critic");
	checkpoint.WriteStats(TWO_HEAD_STATS);
	std::vector<RLGC::WeightedReward> rewards;

	ASSERT_TRUE(CriticProbe::TryLoad(MakeConfig(agent, checkpoint.folder, rewards)) == nullptr);
}

TEST(CriticProbe, TryLoadRejectsHeadCountMismatch) {
	TinyAgent agent;
	TempCheckpoint checkpoint("mismatch");
	checkpoint.SaveModels(agent, 3);
	checkpoint.WriteStats(TWO_HEAD_STATS);
	std::vector<RLGC::WeightedReward> rewards;

	ASSERT_TRUE(CriticProbe::TryLoad(MakeConfig(agent, checkpoint.folder, rewards)) == nullptr);
}

TEST(CriticProbe, EvaluateDenormalizesAndWeightsHeads) {
	TinyAgent agent;
	TempCheckpoint checkpoint("decomposed");
	checkpoint.SaveModels(agent, 2);
	checkpoint.WriteStats(TWO_HEAD_STATS);

	NamedReward touch("touch");
	std::vector<RLGC::WeightedReward> rewards;
	rewards.emplace_back(&touch, 2.f);

	std::unique_ptr<CriticProbe> probe = CriticProbe::TryLoad(MakeConfig(agent, checkpoint.folder, rewards));
	ASSERT_TRUE(probe != nullptr);
	ASSERT_TRUE(probe->IsDecomposed());

	const std::vector<float> obs = { 0.1f, -0.2f, 0.3f, 0.f, 1.f };
	const std::vector<float> normalized = checkpoint.RawCriticOutputs(agent, 2, obs);
	const ForgeDebug::CriticDebugData estimate = probe->Evaluate(obs, rewards);

	// touch: mean 100, meanSq 10004 -> sigma 2; gone: uninitialized -> identity, no matching reward
	ASSERT_EQ(estimate.heads.size(), size_t(2));
	ASSERT_TRUE(Near(estimate.heads[0].raw, normalized[0] * 2.f + 100.f));
	ASSERT_TRUE(Near(estimate.heads[0].contribution, 2.f * estimate.heads[0].raw));
	ASSERT_FALSE(estimate.heads[1].weight.has_value());
	ASSERT_TRUE(Near(estimate.heads[1].raw, normalized[1]));
	ASSERT_TRUE(Near(estimate.total, estimate.heads[0].contribution));

	// Weights are read from the list passed to Evaluate, so adaptive weight changes show up.
	rewards[0].weight = 3.f;
	ASSERT_TRUE(Near(probe->Evaluate(obs, rewards).total, 3.f * estimate.heads[0].raw));
}

TEST(CriticProbe, EvaluateSingleCriticScalesByReturnStd) {
	TinyAgent agent;
	TempCheckpoint checkpoint("single");
	checkpoint.SaveModels(agent, 1);
	checkpoint.WriteStats(R"({ "return_stat": { "mean": 0.0, "var": 18.0, "count": 3 } })");
	std::vector<RLGC::WeightedReward> rewards;

	std::unique_ptr<CriticProbe> probe = CriticProbe::TryLoad(MakeConfig(agent, checkpoint.folder, rewards));
	ASSERT_TRUE(probe != nullptr);
	ASSERT_FALSE(probe->IsDecomposed());

	const std::vector<float> obs = { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f };
	const ForgeDebug::CriticDebugData estimate = probe->Evaluate(obs, rewards);

	ASSERT_FALSE(estimate.isDecomposed);
	ASSERT_TRUE(Near(estimate.total, checkpoint.RawCriticOutputs(agent, 1, obs)[0] * 3.f));
}
