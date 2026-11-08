#include "RunRLBot.h"

#include "RLBotClient.h"
#include "RLBotDevice.h"
#include "RLBotCapture.h"

#include <GigaLearnInfer/AgentSetupSnapshot.h>

#include <GigaLearnInfer/AgentRegistry.h>
#include <GigaLearnInfer/Agent.h>
#include <RLGymCPP/Framework.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <fstream>
#include <optional>
#include <utility>

#ifndef RLBOT_USE_GPU
#define RLBOT_USE_GPU 1
#endif

#ifndef RLBOT_DEFAULT_AGENT
#define RLBOT_DEFAULT_AGENT "example-agent"
#endif

using namespace std;

namespace
{
    filesystem::path ResolveExecutableDirectory(const char* executablePath) {
        std::error_code error;
        if (!executablePath || executablePath[0] == '\0')
            return filesystem::current_path(error);

        filesystem::path path = executablePath;
        if (path.is_relative())
            path = filesystem::absolute(path, error);
        if (error)
            return filesystem::current_path();
        return path.parent_path();
    }

    filesystem::path ResolveCheckpointDirectory(const filesystem::path& basePath) {
        if (basePath.empty() || !filesystem::exists(basePath))
            return {};

        if (filesystem::exists(basePath / "POLICY.lt"))
            return basePath;

        int64_t highestStep = -1;
        filesystem::path highestPath = {};
        for (const auto& entry : filesystem::directory_iterator(basePath)) {
            if (!entry.is_directory())
                continue;

            const string name = entry.path().filename().string();
            if (name.empty() || !all_of(name.begin(), name.end(), [](unsigned char c) { return std::isdigit(c) != 0; }))
                continue;

            int64_t step = 0;
            const auto parsed = std::from_chars(name.data(), name.data() + name.size(), step);
            if (parsed.ec != std::errc() || parsed.ptr != name.data() + name.size())
                continue;
            if (step > highestStep && filesystem::exists(entry.path() / "POLICY.lt")) {
                highestStep = step;
                highestPath = entry.path();
            }
        }

        return highestPath;
    }

	filesystem::path ResolveModelDirectory(
		const Agent* agent,
		const filesystem::path& overridePath,
		const filesystem::path& executableDirectory,
		RLBotLaunchMode mode) {
		// Resolve models in priority order: explicit CLI override, environment override,
		// agent metadata, then legacy locations.
        if (!overridePath.empty()) {
            filesystem::path resolved = ResolveCheckpointDirectory(overridePath);
            if (!resolved.empty())
                return resolved;
            RG_LOG("RLBot: --model-dir \"" << overridePath
                << "\" set but no usable model was found there, falling back.");
        }

        if (const char* envModelDir = getenv("ROCKETFORGE_MODEL_DIR")) {
            filesystem::path envPath = envModelDir;
            filesystem::path resolved = ResolveCheckpointDirectory(envPath);
            if (!resolved.empty())
                return resolved;
            RG_LOG("RLBot: ROCKETFORGE_MODEL_DIR \"" << envPath
                << "\" set but no usable model was found there, falling back.");
        }

		if (mode == RLBotLaunchMode::PORTABLE) {
			filesystem::path resolved = ResolveCheckpointDirectory(executableDirectory);
			if (!resolved.empty())
				return resolved;
		}

		if (agent && !agent->modelPath.empty()) {
			filesystem::path resolved = ResolveCheckpointDirectory(agent->modelPath);
			if (!resolved.empty())
				return resolved;
            RG_LOG("RLBot: agent->modelPath \"" << agent->modelPath
                << "\" set but no usable model was found there, falling back.");
        }

		if (mode == RLBotLaunchMode::PORTABLE) {
			filesystem::path resolved = ResolveCheckpointDirectory(filesystem::current_path());
			if (!resolved.empty())
				return resolved;
		}
		return ResolveCheckpointDirectory(filesystem::current_path().parent_path());
    }

    filesystem::path ResolveCollisionMeshesDirectory(const filesystem::path& executableDirectory) {
        const filesystem::path candidates[] = {
            executableDirectory / "collision_meshes",
            filesystem::current_path() / "collision_meshes",
            RocketSim::CollisionMeshesFolder(),
        };
        for (const filesystem::path& candidate : candidates) {
            if (filesystem::exists(candidate) && filesystem::is_directory(candidate))
                return candidate;
        }
        return {};
    }

    bool ApplyAgentSetupOverride(
        Agent& agent,
        const filesystem::path& setupPath,
        bool required) {
        if (setupPath.empty()) {
            if (required)
                fprintf(stderr, "Packaged runtime is missing agent_setup.json\n");
            return !required;
        }

        ifstream fIn(setupPath);
        if (!fIn.good()) {
            fprintf(stderr, "Failed to open agent setup file: %s\n", setupPath.string().c_str());
            return false;
        }

        nlohmann::json parsed = nlohmann::json::parse(fIn, nullptr, false);
        if (!parsed.is_object()) {
            fprintf(stderr, "Failed to parse agent setup file: %s\n", setupPath.string().c_str());
            return false;
        }

        optional<RocketForge::AgentSetupSnapshot> snapshot =
            RocketForge::AgentSetupSnapshotFromJson(parsed);
        if (!snapshot.has_value()) {
            fprintf(stderr, "Agent setup file did not contain a valid setup: %s\n", setupPath.string().c_str());
            return false;
        }

        RocketForge::ApplyAgentSetupSnapshot(agent, snapshot.value());
        return true;
    }
}

int RunRLBot(
    int argc,
    char* argv[],
    RLBotLaunchMode mode,
    RLBotClientExtensionFactory extensionFactory)
{
    const filesystem::path executableDirectory = ResolveExecutableDirectory(argc > 0 ? argv[0] : nullptr);
    if (argc > 1)
    {
        string arg = argv[1];

        if (arg == "--help") {
            cout << "RLBot [agent] [--device gpu|cpu] [--hivemind on|off]\n"
                 << "  --model-dir CHECKPOINT --agent-setup SETUP.json\n"
                 << "  --capture-dir DIRECTORY   Record live callback inputs and decisions (off by default)\n"
                 << "  --replay FILE --capture-dir DIRECTORY --model-dir CHECKPOINT\n"
                 << "                            Replay offline; --hivemind off tests separate cars\n"
                 << "  --allow-incomplete        Replay only a verified interrupted prefix\n";
            return EXIT_SUCCESS;
        }

        if (arg == "--list-agents")
        {
            vector<string> agents = AgentRegistry::GetAvailableAgents();

            cout << "Available Agents:\n";
            for (const auto& name : agents)
            {
                cout << "  - " << name << "\n";
            }

            return 0;
        }
    }

    // Portable launchers may supply options without an explicit compiled-in agent name.
    const bool hasAgentArgument = argc > 1 && string(argv[1]).rfind("--", 0) != 0;
    string selectedAgent = hasAgentArgument ? argv[1] : RLBOT_DEFAULT_AGENT;
    bool hivemind = true;
    bool useGPU = RLBOT_USE_GPU != 0;
    filesystem::path modelDirOverride = {};
    filesystem::path agentSetupOverride = {};
    filesystem::path captureDirectory, replayPath;
    bool allowIncomplete = false;
    nlohmann::json replayHeader;

    for (int i = hasAgentArgument ? 2 : 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--capture-dir" || arg == "--replay") {
            if (i + 1 >= argc || string(argv[i + 1]).rfind("--", 0) == 0) {
                fprintf(stderr, "%s requires a path\n", arg.c_str());
                return EXIT_FAILURE;
            }
            (arg == "--capture-dir" ? captureDirectory : replayPath) = argv[++i];
        } else if (arg == "--allow-incomplete") {
            allowIncomplete = true;
        } else if (arg == "--device") {
            if (i + 1 >= argc) {
                fprintf(stderr, "--device requires gpu or cpu\n");
                return EXIT_FAILURE;
            }
            const string value = argv[++i];
            if (value != "gpu" && value != "cpu") {
                fprintf(stderr, "--device must be gpu or cpu\n");
                return EXIT_FAILURE;
            }
            useGPU = value == "gpu";
        } else if (arg == "--hivemind") {
            if (i + 1 >= argc) {
                fprintf(stderr, "--hivemind requires on or off\n");
                return EXIT_FAILURE;
            }
            const string value = argv[++i];
            if (value != "on" && value != "off") {
                fprintf(stderr, "--hivemind must be on or off\n");
                return EXIT_FAILURE;
            }
            hivemind = value == "on";
        } else if (arg == "--model-dir" && i + 1 < argc) {
            modelDirOverride = argv[++i];
        } else if (arg == "--agent-setup" && i + 1 < argc) {
            agentSetupOverride = argv[++i];
        }
    }

    if ((!replayPath.empty() && (captureDirectory.empty() || modelDirOverride.empty())) ||
        (allowIncomplete && replayPath.empty())) {
        fprintf(stderr, "Replay requires --replay FILE --model-dir CHECKPOINT --capture-dir OUTPUT; --allow-incomplete is replay-only\n");
        return EXIT_FAILURE;
    }

    try {
        if (!replayPath.empty()) {
            replayHeader = ReadRLBotCapture(replayPath, allowIncomplete);
            if (!hasAgentArgument)
                selectedAgent = replayHeader.at("agent").get<string>();
            else if (selectedAgent != replayHeader.at("agent").get<string>())
                throw runtime_error("selected agent differs from the recording");
        }

        ValidateRLBotDevice(useGPU);
    } catch (const std::exception& error) {
        fprintf(stderr, "RLBot %s initialization failed: %s\n", useGPU ? "GPU" : "CPU", error.what());
        return EXIT_FAILURE;
    }

    RG_LOG("RLBot inference device: " << (useGPU ? "GPU (CUDA)" : "CPU"));

    if (mode == RLBotLaunchMode::PORTABLE && agentSetupOverride.empty() && replayPath.empty()) {
        const filesystem::path portableAgentSetup = executableDirectory / "agent_setup.json";
        if (filesystem::exists(portableAgentSetup))
            agentSetupOverride = portableAgentSetup;
    }

    unique_ptr<Agent> agent;
    try {
        agent = AgentRegistry::Create(selectedAgent);
    }
    catch (exception const& ex) {
        fprintf(stderr, "Failed to construct agent '%s': %s\n", selectedAgent.c_str(), ex.what());
        return EXIT_FAILURE;
    }

    try {
        if (!replayPath.empty() && agentSetupOverride.empty()) {
            const std::optional<RocketForge::AgentSetupSnapshot> setup = RocketForge::AgentSetupSnapshotFromJson(replayHeader.at("setup"));
            if (!setup) throw runtime_error("invalid recorded agent setup");
            RocketForge::ApplyAgentSetupSnapshot(*agent, *setup);
        }
    } catch (const exception& error) {
        fprintf(stderr, "Failed to apply recorded setup: %s\n", error.what());
        return EXIT_FAILURE;
    }

    if (!ApplyAgentSetupOverride(*agent, agentSetupOverride, mode == RLBotLaunchMode::PORTABLE && replayPath.empty()))
        return EXIT_FAILURE;

    // Fail before connecting rather than on the first packet, when the match is already waiting on this bot.
    try {
        for (RocketSim::EngineId engine : RocketForge::RequiredEngines(RocketForge::CaptureAgentSetupSnapshot(*agent)))
            RocketSim::RequireEngine(engine);
    } catch (const exception& error) {
        fprintf(stderr, "This bot can't run in this RLBot build: %s\n", error.what());
        return EXIT_FAILURE;
    }

    filesystem::path const modelsFolder = replayPath.empty()
        ? ResolveModelDirectory(agent.get(), modelDirOverride, executableDirectory, mode)
        : ResolveCheckpointDirectory(modelDirOverride);
    if (modelsFolder.empty()) {
        RG_LOG("Failed to find model directory for agent " << selectedAgent << ". Expected: "
            << (filesystem::current_path().parent_path() / "Models"));
        return EXIT_FAILURE;
    }

	if (RocketSim::GetStage() == RocketSim::RocketSimStage::UNINITIALIZED) {
		if (mode == RLBotLaunchMode::PORTABLE) {
			const filesystem::path collisionMeshes = ResolveCollisionMeshesDirectory(executableDirectory);
			if (collisionMeshes.empty()) {
				fprintf(stderr, "Failed to find the collision_meshes folder beside the RLBot runtime.\n");
				return EXIT_FAILURE;
			}
			RocketSim::Init(collisionMeshes, true);
		} else {
			RocketSim::Init(RocketSim::CollisionMeshesFolder(), true);
		}
	}

    auto const serverHost = []() -> char const* {
        auto const env = getenv("RLBOT_SERVER_IP");
        return env ? env : "127.0.0.1";
        }();

    auto const serverPort = []() -> char const* {
        auto const env = getenv("RLBOT_SERVER_PORT");
        return env ? env : "23234";
        }();

    RG_LOG("RLBot team batching: " << (hivemind ? "on" : "off"));

    auto ctx = make_shared<SharedBotContext>();
    ctx->agent = shared_ptr<const Agent>(move(agent));
    ctx->modelsFolder = modelsFolder;
    ctx->useGPU = useGPU;
    ctx->extensionFactory = std::move(extensionFactory);
    ctx->captureDirectory = captureDirectory;

    try {
        if (!captureDirectory.empty()) {
            filesystem::create_directories(captureDirectory);
            ctx->captureChecksums = RLBotCheckpointChecksums(modelsFolder);
        }

        if (!replayPath.empty()) {
            ReplayRLBotCapture(replayPath, ctx, !hivemind, allowIncomplete);
            return EXIT_SUCCESS;
        }
    } catch (const exception& error) {
        fprintf(stderr, "RLBot recording/replay failed: %s\n", error.what());
        return EXIT_FAILURE;
    }

    const char* agentId = getenv("RLBOT_AGENT_ID");
    if (!agentId || strlen(agentId) == 0) {
        fprintf(stderr, "Missing environment variable RLBOT_AGENT_ID\n");
        return EXIT_FAILURE;
    }

    const bool requestCppInterfaceBallPrediction = false;

    RLBotBotManager manager(ctx, hivemind);
    if (!manager.connect(serverHost, serverPort, agentId, requestCppInterfaceBallPrediction)) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
