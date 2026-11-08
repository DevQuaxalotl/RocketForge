#include <RocketForge/ParityProbe/AdapterRegistry.h>
#include <RocketForge/ParityProbe/Protocol.h>

#include <GigaLearnInfer/AgentRegistry.h>
#include <GigaLearnInfer/AgentSetupSnapshot.h>
#include <GigaLearnInfer/ModelManager.h>
#include <GigaLearnInfer/Models.h>

#include <RocketSim/Includes.h>

#include <nlohmann/json.hpp>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
	using RocketForge::ParityProbe::FrameAdapter;
	using RocketForge::ParityProbe::FrameAdapterRegistry;
	using RocketForge::ParityProbe::FrameFixture;
	using RocketForge::ParityProbe::FrameHookContext;
	using RocketForge::ParityProbe::FrameResult;
	using RocketForge::ParityProbe::FixtureDocument;
	using RocketForge::ParityProbe::SequenceFixture;

	struct Options
	{
		std::string agentName;
		std::filesystem::path checkpointPath;
		std::filesystem::path fixturePath;
		std::filesystem::path outputPath;
		std::filesystem::path setupPath;
		std::string producer = "rocketforge";
		std::string adapterName;
		std::string deviceName = "cpu";
		bool listAgents = false;
		bool listAdapters = false;
		bool showHelp = false;
	};

	void PrintUsage(std::ostream& out)
	{
		out
			<< "RocketForgeParityProbe --agent <name> --checkpoint <folder> --fixture <file> [options]\n"
			<< "\n"
			<< "Options:\n"
			<< "  --output <file>    Write JSONL to a file instead of stdout\n"
			<< "  --setup <file>     Apply a saved agent_setup before creating runtime objects\n"
			<< "  --producer <name>  Set the result producer label (default: rocketforge)\n"
			<< "  --adapter <name>   Run an optionally compiled frame adapter\n"
			<< "  --device <name>    Run inference on cpu or cuda (default: cpu)\n"
			<< "  --list-agents      List registered agents and exit\n"
			<< "  --list-adapters    List optional frame adapters and exit\n"
			<< "  --help             Show this text\n";
	}

	Options ParseOptions(int argc, char* argv[])
	{
		Options options;
		for (int index = 1; index < argc; ++index)
		{
			const std::string argument = argv[index];
			auto requireValue = [&]() -> std::string
			{
				if (++index >= argc)
					throw std::runtime_error("missing value after " + argument);
				return argv[index];
			};

			if (argument == "--agent")
				options.agentName = requireValue();
			else if (argument == "--checkpoint")
				options.checkpointPath = requireValue();
			else if (argument == "--fixture")
				options.fixturePath = requireValue();
			else if (argument == "--output")
				options.outputPath = requireValue();
			else if (argument == "--setup")
				options.setupPath = requireValue();
			else if (argument == "--producer")
				options.producer = requireValue();
			else if (argument == "--adapter")
				options.adapterName = requireValue();
			else if (argument == "--device")
				options.deviceName = requireValue();
			else if (argument == "--list-agents")
				options.listAgents = true;
			else if (argument == "--list-adapters")
				options.listAdapters = true;
			else if (argument == "--help" || argument == "-h")
				options.showHelp = true;
			else
				throw std::runtime_error("unknown argument: " + argument);
		}
		return options;
	}

	RocketForge::AgentSetupSnapshot LoadAgentSetup(const std::filesystem::path& path)
	{
		std::ifstream input(path);
		if (!input.good())
			throw std::runtime_error("failed to open agent setup: " + path.string());
		nlohmann::json root = nlohmann::json::parse(input, nullptr, false);
		if (root.is_discarded() || !root.is_object())
			throw std::runtime_error("agent setup is not valid JSON");
		const nlohmann::json& setupJson = root.contains("agent_setup") ? root.at("agent_setup") : root;
		std::optional<RocketForge::AgentSetupSnapshot> setup = RocketForge::AgentSetupSnapshotFromJson(setupJson);
		if (!setup)
			throw std::runtime_error("agent setup does not contain a supported agent_setup object");
		return *setup;
	}

	RLGC::Player& FindTargetPlayer(RLGC::GameState& state, uint32_t targetCarId)
	{
		auto iterator = std::find_if(
			state.players.begin(),
			state.players.end(),
			[&](const RLGC::Player& player) { return player.carId == targetCarId; });
		if (iterator == state.players.end())
			throw std::runtime_error("target car disappeared after parity frame preparation");
		return *iterator;
	}

	void LinkPreviousState(RLGC::GameState& state, RLGC::GameState* previous)
	{
		// NOTE: RocketForge exposes exactly one previous frame. UpdateFromArena()
		// clears the predecessor's links before connecting the current state.
		if (previous)
		{
			previous->prev = nullptr;
			for (RLGC::Player& player : previous->players)
				player.prev = nullptr;
		}
		state.prev = previous;
		for (RLGC::Player& player : state.players)
		{
			player.prev = nullptr;
			if (!previous)
				continue;
			auto iterator = std::find_if(
				previous->players.begin(),
				previous->players.end(),
				[&](const RLGC::Player& candidate) { return candidate.carId == player.carId; });
			if (iterator != previous->players.end())
				player.prev = &*iterator;
		}
	}

	std::vector<float> TensorToFloatVector(const torch::Tensor& tensor)
	{
		torch::Tensor contiguous = tensor.detach().to(torch::kCPU, torch::kFloat32).contiguous().flatten();
		const float* begin = contiguous.data_ptr<float>();
		return { begin, begin + contiguous.numel() };
	}

	FrameResult EvaluateFrame(
		Agent& agent,
		GGL::ModelSet& models,
		RLGC::ObsBuilder& obsBuilder,
		RLGC::ActionParser& actionParser,
		int obsSize,
		int numActions,
		const torch::Device& device,
		FrameAdapter* adapter,
		const nlohmann::json& fixtureId,
		const nlohmann::json& sequenceId,
		const FrameFixture& fixture,
		RLGC::GameState& state)
	{
		FrameHookContext hookContext{
			agent,
			obsBuilder,
			actionParser,
			fixture.source,
			state,
			fixture.targetCarId
		};
		if (adapter)
			adapter->BeforeReset(hookContext);
		LinkPreviousState(state, state.prev);
		if (fixture.resetObsBuilder)
			obsBuilder.Reset(state);
		if (adapter)
			adapter->BeforeBuildObs(hookContext);
		LinkPreviousState(state, state.prev);

		RLGC::Player& player = FindTargetPlayer(state, fixture.targetCarId);
		// NOTE: Seed immediately before the single observation build so adapters and
		// reset hooks cannot consume the ordering RNG used by shuffled observations.
		::Math::GetRandEngine().seed(fixture.obsRngSeed);
		std::vector<float> observation;
		obsBuilder.BuildObsInto(player, state, observation);
		if (static_cast<int>(observation.size()) != obsSize)
			throw std::runtime_error("observation builder produced a size that differs from the cached agent dimension");

		std::vector<uint8_t> actionMask;
		if (actionParser.UsesActionMask())
			actionParser.GetActionMaskInto(player, state, actionMask);
		else
			actionMask.assign(numActions, uint8_t{ 1 });
		if (static_cast<int>(actionMask.size()) != numActions)
			throw std::runtime_error("action parser produced a mask with an unexpected size");
		for (uint8_t& enabled : actionMask)
			enabled = enabled != 0 ? uint8_t{ 1 } : uint8_t{ 0 };
		if (std::none_of(actionMask.begin(), actionMask.end(), [](uint8_t enabled) { return enabled != 0; }))
			throw std::runtime_error("action parser disabled every policy action");

		torch::Tensor obsTensor = torch::tensor(
			observation,
			torch::TensorOptions().dtype(torch::kFloat32).device(device)).reshape({ 1, obsSize });
		torch::Tensor featuresTensor = obsTensor;
		if (models["shared_head"])
			featuresTensor = models["shared_head"]->Forward(obsTensor, false);
		torch::Tensor rawLogitsTensor = models["policy"]->Forward(featuresTensor, false);
		torch::Tensor maskTensor = torch::tensor(
			actionMask,
			torch::TensorOptions().dtype(torch::kUInt8).device(device)).reshape({ 1, numActions }).to(torch::kBool);
		constexpr float ACTION_DISABLED_LOGIT = -1e10f;
		torch::Tensor maskedLogitsTensor =
			torch::nan_to_num(rawLogitsTensor, 0.0, 50.0, -50.0) +
			ACTION_DISABLED_LOGIT * maskTensor.logical_not();

		const int actionId = maskedLogitsTensor.argmax(1).item<int>();
		std::vector<float> maskedLogits = TensorToFloatVector(maskedLogitsTensor);
		float largest = -std::numeric_limits<float>::infinity();
		float secondLargest = -std::numeric_limits<float>::infinity();
		for (size_t index = 0; index < maskedLogits.size(); ++index)
		{
			if (!actionMask[index])
				continue;
			const float value = maskedLogits[index];
			if (value > largest)
			{
				secondLargest = largest;
				largest = value;
			}
			else if (value > secondLargest)
			{
				secondLargest = value;
			}
		}
		const float margin = std::isfinite(secondLargest) ? largest - secondLargest : 0.f;
		if (agent.tickSkip <= 0)
			throw std::runtime_error("agent tickSkip must be positive");
		if (agent.actionDelay < 0 || agent.actionDelay > agent.tickSkip)
			throw std::runtime_error("agent actionDelay must be between zero and tickSkip");
		const int actionTicks = agent.tickSkip - agent.actionDelay;
		if (agent.actionDelay != 0 && actionParser.IsMacroAction(actionId, player, state))
			throw std::runtime_error("action macros require actionDelay == 0");
		std::vector<RLGC::ActionMacro> macros;
		actionParser.ParseActionMacros(actionId, player, state, actionTicks, macros);
		if (macros.empty())
			macros.push_back({ actionParser.ParseAction(actionId, player, state), actionTicks });
		int plannedTicks = 0;
		for (const RLGC::ActionMacro& macro : macros)
		{
			if (macro.ticks < 0)
				throw std::runtime_error("action parser produced a macro with negative ticks");
			plannedTicks += macro.ticks;
		}
		if (plannedTicks != actionTicks)
			throw std::runtime_error("action parser macro ticks do not fill the action window");
		const RLGC::Action parsedAction = macros.front().action;

		FrameResult result;
		result.fixtureId = fixtureId;
		result.sequenceId = sequenceId;
		result.frameId = fixture.frameId;
		result.targetCarId = fixture.targetCarId;
		result.obsRngSeed = fixture.obsRngSeed;
		result.observation = std::move(observation);
		result.actionMask = std::move(actionMask);
		result.sharedFeatures = TensorToFloatVector(featuresTensor);
		result.rawLogits = TensorToFloatVector(rawLogitsTensor);
		result.maskedLogits = std::move(maskedLogits);
		result.argmaxActionId = actionId;
		result.top1Margin = margin;
		result.parsedAction = parsedAction;
		result.actionPlan.reserve(macros.size());
		for (const RLGC::ActionMacro& macro : macros)
			result.actionPlan.push_back({ macro.action, macro.ticks });
		result.tickSkip = agent.tickSkip;
		result.actionDelay = agent.actionDelay;
		result.actionTicks = actionTicks;
		result.queuedAction = macros.front().action;
		result.appliedAction = fixture.runtime.appliedAction.value_or(player.prevAction);
		return result;
	}

	void RunProbe(const Options& options)
	{
		if (options.agentName.empty() || options.checkpointPath.empty() || options.fixturePath.empty())
			throw std::runtime_error("--agent, --checkpoint, and --fixture are required");
		if (options.producer.empty())
			throw std::runtime_error("--producer cannot be empty");
		if (options.deviceName != "cpu" && options.deviceName != "cuda")
			throw std::runtime_error("--device must be cpu or cuda");
		if (options.deviceName == "cuda" && !torch::cuda::is_available())
			throw std::runtime_error("--device cuda requested, but CUDA is unavailable");
		if (!std::filesystem::is_directory(options.checkpointPath))
			throw std::runtime_error("checkpoint path is not a directory: " + options.checkpointPath.string());
		const torch::Device device =
			options.deviceName == "cuda" ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

		std::unique_ptr<Agent> agent = AgentRegistry::Create(options.agentName);
		if (!options.setupPath.empty())
			RocketForge::ApplyAgentSetupSnapshot(*agent, LoadAgentSetup(options.setupPath));
		std::unique_ptr<RLGC::ObsBuilder> obsBuilder(agent->MakeObsBuilder());
		std::unique_ptr<RLGC::ActionParser> actionParser(agent->MakeActionParser());
		if (!obsBuilder || !actionParser)
			throw std::runtime_error("agent did not create both inference runtime components");
		const int obsSize = agent->GetObsSize();
		const int numActions = actionParser->GetActionAmount();
		if (obsSize <= 0 || numActions <= 0)
			throw std::runtime_error("agent inference dimensions must be positive");

		GGL::ModelSet models;
		agent->MakeInferenceModels(
			obsSize,
			numActions,
			device,
			models);
		models.Load(options.checkpointPath, false);
		for (GGL::Model* model : models)
			model->eval();

		std::unique_ptr<FrameAdapter> adapter = FrameAdapterRegistry::Create(options.adapterName);
		FixtureDocument fixture = RocketForge::ParityProbe::LoadFixtureDocument(options.fixturePath);

		std::ofstream outputFile;
		std::ostream* output = &std::cout;
		if (!options.outputPath.empty())
		{
			if (!options.outputPath.parent_path().empty())
				std::filesystem::create_directories(options.outputPath.parent_path());
			outputFile.open(options.outputPath);
			if (!outputFile.good())
				throw std::runtime_error("failed to open parity output: " + options.outputPath.string());
			output = &outputFile;
		}

		torch::set_num_threads(1);
		torch::manual_seed(0);
		c10::InferenceMode inferenceMode;
		for (const SequenceFixture& sequence : fixture.sequences)
		{
			std::vector<std::unique_ptr<RLGC::GameState>> stateHistory;
			stateHistory.reserve(sequence.frames.size());
			for (const FrameFixture& frame : sequence.frames)
			{
				auto state = std::make_unique<RLGC::GameState>(frame.state);
				LinkPreviousState(*state, stateHistory.empty() ? nullptr : stateHistory.back().get());
				FrameResult result = EvaluateFrame(
					*agent,
					models,
					*obsBuilder,
					*actionParser,
					obsSize,
					numActions,
					device,
					adapter.get(),
					fixture.fixtureId,
					sequence.sequenceId,
					frame,
					*state);
				*output << RocketForge::ParityProbe::FrameResultToJson(result, options.producer).dump() << '\n';
				stateHistory.push_back(std::move(state));
			}
		}
		output->flush();
		models.Free();
	}
}

int main(int argc, char* argv[])
{
	try
	{
		const Options options = ParseOptions(argc, argv);
		if (options.showHelp)
		{
			PrintUsage(std::cout);
			return 0;
		}
		if (options.listAgents)
		{
			for (const std::string& name : AgentRegistry::GetAvailableAgents())
				std::cout << name << '\n';
			return 0;
		}
		if (options.listAdapters)
		{
			for (const std::string& name : FrameAdapterRegistry::GetAvailableAdapters())
				std::cout << name << '\n';
			return 0;
		}
		RunProbe(options);
		return 0;
	}
	catch (const std::exception& error)
	{
		std::cerr << "parity-probe: " << error.what() << '\n';
		return 1;
	}
}
