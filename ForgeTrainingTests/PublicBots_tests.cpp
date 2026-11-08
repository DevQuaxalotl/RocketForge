#include <gtest/gtest.h>

#include "TestSupport.h"

#include "Settings/ForgeSettings.h"
#include <GigaLearnCPP/TrainingStorage.h>
#include <GigaLearnInfer/Agent.h>
#include <GigaLearnInfer/AgentRegistry.h>
#include <GigaLearnInfer/ModelManager.h>
#include <GigaLearnInfer/Models.h>
#include <RLGymCPP/ActionParsers/ActionParser.h>
#include <RLGymCPP/Gamestates/GameState.h>
#include <torch/script.h>
#include <torch/cuda.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace
{
	struct PublicBotExpectation
	{
		std::string name;
		int obsSize;
		int numActions;
	};

	const std::vector<PublicBotExpectation> PUBLIC_BOTS = {
		{ "Necto", 1057, 72 },
		{ "Nexto", 1057, 90 },
		{ "Element", 107, 1944 }
	};

	std::filesystem::path ModelFolder(const std::string& name)
	{
		return std::filesystem::path(FORGE_SOURCE_DIR)
			/ "Bots" / "PublicBots" / name / "Models" / "Imported Models" / "RLBotPack";
	}

	torch::Tensor PackAttentionInput(
		const torch::Tensor& query,
		const torch::Tensor& keyValue,
		const torch::Tensor& mask)
	{
		return torch::cat({
			query.flatten(1),
			keyValue.flatten(1),
			mask.flatten(1)
		}, 1);
	}

	torch::jit::IValue MakeAttentionInput(
		torch::Tensor query,
		torch::Tensor keyValue,
		torch::Tensor mask)
	{
		return c10::ivalue::Tuple::create({
			std::move(query),
			std::move(keyValue),
			std::move(mask)
		});
	}

	torch::Tensor ExtractOriginalNectoLogits(const torch::jit::IValue& output)
	{
		const auto& heads = output.toTuple()->elements().at(0).toTuple()->elements();
		torch::Tensor joint = heads.at(0).toTensor().squeeze(1);
		for (int i = 1; i < (int)heads.size(); i++) {
			torch::Tensor head = heads.at(i).toTensor().squeeze(1);
			joint = (joint.unsqueeze(-1) + head.unsqueeze(-2)).flatten(1);
		}
		return joint;
	}

	torch::Tensor ExtractOriginalNextoLogits(const torch::jit::IValue& output)
	{
		return output.toTuple()->elements().at(0).toTensor().squeeze(1);
	}

	void AssertAction(const RLGC::Action& actual, const RLGC::Action& expected)
	{
		for (int i = 0; i < (int)RLGC::Action::ELEM_AMOUNT; i++)
			ASSERT_TRUE(std::abs(actual[i] - expected[i]) <= 1e-6f);
	}

	RLGC::GameState MakeKickoffState()
	{
		RLGC::GameState state = {};
		state.gameMode = RocketSim::GameMode::SOCCAR;
		state.ball.pos = { 0, 0, 93 };
		state.players.resize(4);

		state.players[0].carId = 10;
		state.players[0].team = RocketSim::Team::BLUE;
		state.players[0].pos = { -2048, -2560, 17 };
		state.players[0].isOnGround = true;

		state.players[1].carId = 11;
		state.players[1].team = RocketSim::Team::BLUE;
		state.players[1].pos = { 2048, -2560, 17 };
		state.players[1].isOnGround = true;

		state.players[2].carId = 20;
		state.players[2].team = RocketSim::Team::ORANGE;
		state.players[2].pos = { -2048, 2560, 17 };
		state.players[2].isOnGround = true;

		state.players[3].carId = 21;
		state.players[3].team = RocketSim::Team::ORANGE;
		state.players[3].pos = { 2048, 2560, 17 };
		state.players[3].isOnGround = true;
		return state;
	}
}

TEST(PublicBots, RegisterInferenceOnlyAgents)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	for (const PublicBotExpectation& expected : PUBLIC_BOTS) {
		std::unique_ptr<Agent> agent = AgentRegistry::Create(expected.name);
		ASSERT_TRUE(agent != nullptr);
		ASSERT_EQ(agent->agentName, expected.name);
		ASSERT_TRUE(agent->inferenceOnly);
		ASSERT_TRUE(agent->inferenceDeterministic);
		ASSERT_EQ(agent->tickSkip, 8);
		ASSERT_EQ(agent->actionDelay, 7);
		ASSERT_EQ(agent->GetObsSize(), expected.obsSize);

		std::unique_ptr<RLGC::ActionParser> actionParser(agent->MakeActionParser());
		ASSERT_EQ(actionParser->GetActionAmount(), expected.numActions);
		ASSERT_TRUE(actionParser->HasRuntimeActionOverride());
		ASSERT_TRUE(GGL::TrainingStorage::HasDirectModelFiles(agent->modelPath));
	}
}

TEST(PublicBots, NectoAndNextoRecordedKickoffMatchesPublishedTicks)
{
	for (const std::string& name : { "Necto", "Nexto" }) {
		std::unique_ptr<Agent> agent = AgentRegistry::Create(name);
		std::unique_ptr<RLGC::ActionParser> parser(agent->MakeActionParser());
		RLGC::GameState state = MakeKickoffState();
		// NOTE: The published wrapper's tie-break chooses this side despite its
		// source comment saying "Left goes".
		const RLGC::Player& taker = state.players[1];
		const RLGC::Player& nonTaker = state.players[0];
		const RLGC::Action policyAction = {};

		ASSERT_TRUE(parser->RequiresPerTickActionOverride(taker, state));
		ASSERT_FALSE(parser->RequiresPerTickActionOverride(nonTaker, state));

		parser->ResetRuntimeActionOverride();
		RLGC::GameState boundaryState = MakeKickoffState();
		boundaryState.ball.pos.z = 17;
		boundaryState.players.resize(2);
		boundaryState.players[0].pos = { 0, -1010, 17 };
		boundaryState.players[1].team = RocketSim::Team::ORANGE;
		boundaryState.players[1].pos = { 0, 1000, 17 };
		ASSERT_TRUE(parser->RequiresPerTickActionOverride(
			boundaryState.players[0],
			boundaryState));
		parser->ResetRuntimeActionOverride();

		RLGC::GameState toleranceState = MakeKickoffState();
		toleranceState.players.resize(3);
		toleranceState.players[0].pos = { 0, -1009, 17 };
		toleranceState.players[1].pos = { 1018, 0, 17 };
		toleranceState.players[2].team = RocketSim::Team::ORANGE;
		toleranceState.players[2].pos = { 0, 1000, 17 };
		ASSERT_FALSE(parser->RequiresPerTickActionOverride(
			toleranceState.players[0],
			toleranceState));
		parser->ResetRuntimeActionOverride();

		std::vector<RLGC::Action> outputs;
		for (int tick = 0; tick <= 168; tick++) {
			outputs.push_back(parser->OverrideRuntimeAction(
				policyAction,
				taker,
				state,
				tick == 0 ? 0 : 1));
		}

		AssertAction(outputs[0], { 1, 0, 0, 0, 0, 0, 1, 0 });
		AssertAction(outputs[43], { 1, 0, 0, 0, 0, 0, 1, 0 });
		AssertAction(outputs[44], { 1, -1, 0, 0, 0, 0, 1, 0 });
		AssertAction(outputs[60], { 1, 0, 0, 0, 0, 1, 1, 0 });
		AssertAction(outputs[68], { 1, 0, 0, 0, 0, 0, 1, 0 });
		AssertAction(outputs[72], { 1, 0, -0.7f, 0.8f, 0, 1, 1, 0 });
		AssertAction(outputs[76], { 1, 0, 1, 0, 0, 0, 1, 0 });
		AssertAction(outputs[128], { 1, 0, 0.5f, 0, 1, 0, 0, 0 });
		AssertAction(outputs[167], { 1, 0, 0.5f, 0, 1, 0, 0, 0 });
		AssertAction(outputs[168], policyAction);

		state.ball.pos.y = 1;
		ASSERT_FALSE(parser->RequiresPerTickActionOverride(taker, state));
		state.ball.pos.y = 0;
		ASSERT_TRUE(parser->RequiresPerTickActionOverride(taker, state));
		AssertAction(
			parser->OverrideRuntimeAction(policyAction, taker, state, 1),
			{ 1, 0, 0, 0, 0, 0, 1, 0 });

		for (int tick = 0; tick < 12; tick++)
			parser->OverrideRuntimeAction(policyAction, taker, state, 1);
		parser->ResetRuntimeActionOverride();
		ASSERT_TRUE(parser->RequiresPerTickActionOverride(taker, state));
		AssertAction(
			parser->OverrideRuntimeAction(policyAction, taker, state, 1),
			{ 1, 0, 0, 0, 0, 0, 1, 0 });
	}
}

TEST(PublicBots, ElementSpeedflipMatchesPublishedStateMachine)
{
	std::unique_ptr<Agent> agent = AgentRegistry::Create("Element");
	std::unique_ptr<RLGC::ActionParser> parser(agent->MakeActionParser());
	RLGC::GameState state = MakeKickoffState();
	state.players.resize(2);
	state.players[0].pos = { 2048, -2560, 17 };
	state.players[1].team = RocketSim::Team::ORANGE;
	state.players[1].pos = { 2048, 2560, 17 };
	RLGC::Player& player = state.players[0];
	const RLGC::Action policyAction = {};

	AssertAction(
		parser->OverrideRuntimeAction(policyAction, player, state, 0),
		{ 1, -1, 0, 0, 0, 0, 1, 0 });

	player.rotMat.forward = { 0, 1, 0 };
	AssertAction(
		parser->OverrideRuntimeAction(policyAction, player, state, 1),
		{ 1, 0, 0, 0, 0, 0, 1, 0 });

	player.pos.x += 300;
	AssertAction(
		parser->OverrideRuntimeAction(policyAction, player, state, 1),
		{ 1, 0, 0, 0, 0, 1, 1, 0 });
	AssertAction(
		parser->OverrideRuntimeAction(policyAction, player, state, 1),
		{ 1, 0, 0, 0, 1, 0, 1, 0 });

	player.isOnGround = false;
	for (int tick = 0; tick < 6; tick++) {
		AssertAction(
			parser->OverrideRuntimeAction(policyAction, player, state, 1),
			{ 1, 0, 0, 0, 1, 0, 1, 0 });
	}
	AssertAction(
		parser->OverrideRuntimeAction(policyAction, player, state, 1),
		{ 1, 0, -1, 0, 1, 1, 1, 0 });
	AssertAction(
		parser->OverrideRuntimeAction(policyAction, player, state, 1),
		{ 1, 0, 1, 0.4f, 1, 0, 1, 0 });

	player.isOnGround = true;
	AssertAction(
		parser->OverrideRuntimeAction(policyAction, player, state, 1),
		{ 1, 0, 0, 0, 0, 0, 1, 0 });
	ASSERT_FALSE(parser->RequiresPerTickActionOverride(player, state));

	parser->ResetRuntimeActionOverride();
	AssertAction(
		parser->OverrideRuntimeAction(policyAction, player, state, 1),
		{ 1, -1, 0, 0, 0, 0, 1, 0 });
}

TEST(PublicBots, PaddedAttentionPoliciesMatchPublishedVariablePlayerOutputs)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	torch::NoGradGuard noGrad;
	torch::manual_seed(0x4E455854);

	for (const std::string& name : { "Necto", "Nexto" }) {
		std::unique_ptr<Agent> agent = AgentRegistry::Create(name);
		GGL::ModelSet converted = {};
		GGL::ModelSet convertedCuda = {};
		std::unique_ptr<RLGC::ActionParser> actionParser(agent->MakeActionParser());
		agent->MakeInferenceModels(
			agent->GetObsSize(),
			actionParser->GetActionAmount(),
			torch::kCPU,
			converted);

		try {
			converted.Load(agent->modelPath, false);
			torch::jit::script::Module original = torch::jit::load(
				(ModelFolder(name) / (
					name == "Necto"
						? "necto-model-original.pt"
						: "nexto-model-original.pt")).string(),
				torch::kCPU);
			original.eval();

			if (torch::cuda::is_available()) {
				agent->MakeInferenceModels(
					agent->GetObsSize(),
					actionParser->GetActionAmount(),
					torch::kCUDA,
					convertedCuda);
				convertedCuda.Load(agent->modelPath, false);
			}

			float maxCpuError = 0;
			float maxCudaError = 0;
			for (int playerCount : { 2, 4, 6 }) {
				const int batchSize = 32;
				torch::Tensor query = torch::randn({ batchSize, 1, 32 });
				torch::Tensor variableKeyValue =
					torch::randn({ batchSize, playerCount + 35, 24 });
				torch::Tensor variableMask =
					torch::zeros({ batchSize, playerCount + 35 });
				torch::Tensor expected = name == "Necto"
					? ExtractOriginalNectoLogits(original.forward({
						MakeAttentionInput(query, variableKeyValue, variableMask) }))
					: ExtractOriginalNextoLogits(original.forward({
						MakeAttentionInput(query, variableKeyValue, variableMask) }));

				torch::Tensor fixedKeyValue = torch::zeros({ batchSize, 41, 24 });
				torch::Tensor fixedMask = torch::zeros({ batchSize, 41 });
				if (name == "Necto") {
					fixedKeyValue.select(1, 0).copy_(variableKeyValue.select(1, 0));
					fixedKeyValue.slice(1, 1, 1 + playerCount).copy_(
						variableKeyValue.slice(1, 1, 1 + playerCount));
					fixedKeyValue.slice(1, 7).copy_(
						variableKeyValue.slice(1, 1 + playerCount));
					fixedMask.slice(1, 1 + playerCount, 7).fill_(1);
				} else {
					fixedKeyValue.slice(1, 0, playerCount).copy_(
						variableKeyValue.slice(1, 0, playerCount));
					fixedKeyValue.select(1, 6).copy_(
						variableKeyValue.select(1, playerCount));
					fixedKeyValue.slice(1, 7).copy_(
						variableKeyValue.slice(1, playerCount + 1));
					fixedMask.slice(1, playerCount, 6).fill_(1);
				}

				torch::Tensor packed = PackAttentionInput(query, fixedKeyValue, fixedMask);
				torch::Tensor actual = converted["policy"]->Forward(packed, false);
				ASSERT_TRUE(torch::equal(expected.argmax(1), actual.argmax(1)));
				maxCpuError = std::max(
					maxCpuError,
					(expected - actual).abs().max().item<float>());

				if (torch::cuda::is_available()) {
					torch::Tensor actualCuda = convertedCuda["policy"]->Forward(
						packed.to(torch::kCUDA),
						false).to(torch::kCPU);
					ASSERT_TRUE(torch::equal(
						expected.argmax(1),
						actualCuda.argmax(1)));
					maxCudaError = std::max(
						maxCudaError,
						(expected - actualCuda).abs().max().item<float>());
				}
			}

			if (maxCpuError > 5e-4f) {
				FAIL() << name << " CPU conversion max logit error: " << maxCpuError;
			}
			if (maxCudaError > 2e-3f) {
				FAIL() << name << " CUDA conversion max logit error: " << maxCudaError;
			}
			convertedCuda.Free();
		} catch (...) {
			convertedCuda.Free();
			converted.Free();
			throw;
		}
		converted.Free();
	}
}

TEST(PublicBots, RejectDirectLearnerTraining)
{
	RocketForgeTraining::ForgeSettings settings = {};
	settings.schemaVersion = RocketForgeTraining::kForgeSettingsSchemaVersion;
	settings.agentName = "Nexto";

	const std::vector<std::string> errors =
		RocketForgeTraining::ValidateForgeSettings(settings);
	const bool foundInferenceOnlyError = std::any_of(
		errors.begin(),
		errors.end(),
		[](const std::string& error) {
			return error.find("inference-only") != std::string::npos;
		});
	ASSERT_TRUE(foundInferenceOnlyError);
}

TEST(PublicBots, LoadPublishedPoliciesAndInferOnCpu)
{
	for (const PublicBotExpectation& expected : PUBLIC_BOTS) {
		std::unique_ptr<Agent> agent = AgentRegistry::Create(expected.name);
		GGL::ModelSet models = {};
		agent->MakeInferenceModels(
			expected.obsSize,
			expected.numActions,
			torch::kCPU,
			models);

		try {
			models.Load(agent->modelPath, false);
			torch::NoGradGuard noGrad;
			torch::Tensor logits = models["policy"]->Forward(
				torch::zeros({ 2, expected.obsSize }),
				false);
			ASSERT_EQ(logits.dim(), int64_t(2));
			ASSERT_EQ(logits.size(0), int64_t(2));
			ASSERT_EQ(logits.size(1), int64_t(expected.numActions));
			ASSERT_TRUE(torch::isfinite(logits).all().item<bool>());
		} catch (...) {
			models.Free();
			throw;
		}
		models.Free();
	}
}

TEST(PublicBots, LoadPublishedPoliciesAndInferOnCuda)
{
	ROCKETFORGE_REQUIRE_CUDA();

	for (const PublicBotExpectation& expected : PUBLIC_BOTS) {
		std::unique_ptr<Agent> agent = AgentRegistry::Create(expected.name);
		GGL::ModelSet models = {};
		agent->MakeInferenceModels(
			expected.obsSize,
			expected.numActions,
			torch::kCUDA,
			models);

		try {
			models.Load(agent->modelPath, false);
			torch::NoGradGuard noGrad;
			torch::Tensor logits = models["policy"]->Forward(
				torch::zeros(
					{ 2, expected.obsSize },
					torch::TensorOptions().device(torch::kCUDA)),
				false);
			ASSERT_EQ(logits.size(1), int64_t(expected.numActions));
			ASSERT_TRUE(torch::isfinite(logits).all().item<bool>());
		} catch (...) {
			models.Free();
			throw;
		}
		models.Free();
	}
}

TEST(PublicBots, CudaGraphsMatchEagerInference)
{
#ifndef RG_CUDA_SUPPORT
	GTEST_SKIP() << "Built without CUDA support";
#else
	ROCKETFORGE_REQUIRE_CUDA();

	torch::NoGradGuard noGrad;
	const torch::Device device = torch::Device(torch::kCUDA);
	for (const PublicBotExpectation& expected : PUBLIC_BOTS) {
		std::unique_ptr<Agent> agent = AgentRegistry::Create(expected.name);
		GGL::ModelSet models = {};
		agent->MakeInferenceModels(
			expected.obsSize,
			expected.numActions,
			device,
			models);

		try {
			models.Load(agent->modelPath, false);
			torch::Tensor obs = torch::randn({ 16, expected.obsSize }, device);
			if (expected.obsSize == 1057)
				obs.slice(1, expected.obsSize - 41).zero_();
			torch::Tensor actionMasks = torch::ones(
				{ 16, expected.numActions },
				torch::TensorOptions().dtype(torch::kUInt8).device(device));
			torch::Tensor eagerActions;
			torch::Tensor graphActions;

			GGL::ModelManager::InferActionsFromModels(
				models,
				obs,
				actionMasks,
				true,
				1.f,
				false,
				&eagerActions,
				nullptr,
				0.f,
				false);
			GGL::ModelManager::ResetCudaGraphStats();
			GGL::ModelManager::InferActionsFromModels(
				models,
				obs,
				actionMasks,
				true,
				1.f,
				false,
				&graphActions,
				nullptr,
				0.f,
				true);
			GGL::ModelManager::InferActionsFromModels(
				models,
				obs,
				actionMasks,
				true,
				1.f,
				false,
				&graphActions,
				nullptr,
				0.f,
				true);

			const GGL::CudaGraphStats stats = GGL::ModelManager::GetCudaGraphStats();
			ASSERT_TRUE(torch::equal(eagerActions.cpu(), graphActions.cpu()));
			ASSERT_TRUE(stats.captures >= 1);
			ASSERT_TRUE(stats.replays >= 1);
			ASSERT_EQ(stats.captureFailures, (uint64_t)0);
			GGL::ModelManager::ClearCudaGraphCache();
		} catch (...) {
			GGL::ModelManager::ClearCudaGraphCache();
			models.Free();
			throw;
		}
		models.Free();
	}
#endif
}

TEST(AgentSimulation, DefaultsAndPins) {
	EXPECT_EQ(Agent{}.trainingEngineVersion, 3);
	EXPECT_EQ(AgentRegistry::Create("example-agent")->trainingEngineVersion, 2);

	// PublicBots agents are the registered inference-only ones.
	int publicAgents = 0;
	for (const std::string& name : AgentRegistry::GetAvailableAgents()) {
		const std::unique_ptr<Agent> agent = AgentRegistry::Create(name);
		if (!agent->inferenceOnly)
			continue;
		publicAgents++;
		EXPECT_EQ(agent->trainingEngineVersion, 2) << name;
	}
	EXPECT_GE(publicAgents, static_cast<int>(PUBLIC_BOTS.size()));
}
