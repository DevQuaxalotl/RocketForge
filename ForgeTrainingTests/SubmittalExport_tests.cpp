#include <gtest/gtest.h>

#include "TestSupport.h"

#include "Storage/BuildStorage.h"
#include "TrainingSetup/SubmittalExport/SubmittalExport.h"
#include "TrainingSetup/SubmittalExport/SubmittalExportInternal.h"
#include "TrainingSetup/TrainingSetupSnapshot.h"

#include <GigaLearnInfer/Agent.h>
#include <GigaLearnInfer/AgentRegistry.h>
#include <GigaLearnInfer/AgentSetupSnapshot.h>
#include <RocketSim/Engine.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

// An export copies the RLBot cpp-interface sources, which a clone without submodules doesn't have.
#define ROCKETFORGE_REQUIRE_RLBOT_CPP_INTERFACE()                                                                  \
	do {                                                                                                            \
		const std::filesystem::path _interfaceRoot = std::filesystem::path(FORGE_SOURCE_DIR) / "RLBot/inc/cpp-interface"; \
		if (!std::filesystem::exists(_interfaceRoot / "CMakeLists.txt"))                                            \
			GTEST_SKIP() << "The RLBot cpp-interface submodule is empty (" << _interfaceRoot                         \
						 << "). Run \"git submodule update --init RLBot/inc/cpp-interface\".";                    \
	} while (0)

namespace
{
	namespace fs = std::filesystem;

	struct TempFolder
	{
		fs::path path;
		~TempFolder()
		{
			std::error_code error;
			fs::remove_all(path, error);
		}
	};

	void WriteTextFile(const fs::path& path, const std::string& contents)
	{
		fs::create_directories(path.parent_path());
		std::ofstream output(path, std::ios::binary);
		output << contents;
	}

	std::string ReadTextFile(const fs::path& path)
	{
		std::ifstream input(path, std::ios::binary);
		std::ostringstream contents;
		contents << input.rdbuf();
		return contents.str();
	}

	bool TreeContainsFileNamed(const fs::path& root, const std::string& fileName)
	{
		for (const fs::directory_entry& entry : fs::recursive_directory_iterator(root)) {
			if (entry.is_regular_file() && entry.path().filename() == fileName)
				return true;
		}
		return false;
	}

	/** A registered bot under `root` with one checkpointed segment that trains and predicts with the given engines. **/
	struct EngineTestBot
	{
		std::string agentName;
		std::string buildName = "Engine Build";
		std::string segmentId = "v1.00";
	};

	EngineTestBot MakeEngineTestBot(const fs::path& root, const std::string& agentName, int trainingEngine, int predictionEngine)
	{
		using namespace RocketForgeTraining;

		const fs::path botRoot = root / "EngineBot";
		const fs::path modelsRoot = botRoot / "Models";
		WriteTextFile(
			botRoot / "CMakeLists.txt",
			"file(GLOB_RECURSE FILES_SRC \"src/*.cpp\" \"src/*.h\")\n"
			"add_library(EngineTestAgent OBJECT ${FILES_SRC})\n"
			"target_link_libraries(EngineTestAgent PUBLIC GigaLearnInfer)\n"
			"target_sources(Agents INTERFACE $<TARGET_OBJECTS:EngineTestAgent>)\n");
		WriteTextFile(
			botRoot / "src" / "EngineTestAgent.cpp",
			"#include <GigaLearnInfer/AgentRegistry.h>\n"
			"REGISTER_AGENT(\"" + agentName + "\", Agent)\n");

		AgentRegistry::Register(agentName, [agentName, modelsRoot]() {
			auto agent = std::make_unique<Agent>();
			agent->agentName = agentName;
			agent->modelPath = modelsRoot;
			return agent;
		});

		EngineTestBot bot{ agentName };
		BuildStorage storage(modelsRoot, bot.buildName);
		storage.EnsureBuildExists();
		TrainingSegmentMetadata segment = {};
		segment.segmentId = bot.segmentId;
		segment.folderName = storage.MakeSegmentFolderName(segment.segmentId, "engine-stage");
		segment.buildName = bot.buildName;
		segment.stageName = "engine-stage";
		segment.status = "active";
		storage.SaveSegment(segment);
		WriteTextFile(storage.SegmentCheckpointsRoot(segment.folderName) / "1000" / "POLICY.lt", "policy");

		Agent savedAgent;
		savedAgent.agentName = agentName;
		savedAgent.trainingEngineVersion = trainingEngine;
		savedAgent.ballPredictionEngineVersion = predictionEngine;
		savedAgent.sharedHead.layerSizes = {};
		RocketForge::TrainingSetupSnapshot setup = {};
		setup.agentName = agentName;
		setup.stageName = segment.stageName;
		setup.versionId = segment.segmentId;
		setup.agentSetup = RocketForge::CaptureAgentSetupSnapshot(savedAgent);
		GGL::JsonSaveToFile(setup, storage.SegmentRoot(segment.folderName) / RocketForge::TrainingSetupSnapshot::FILE_NAME);
		return bot;
	}

	fs::path ExportEngineTestBot(const EngineTestBot& bot, const fs::path& outputParent)
	{
		const bool useGPU = false;
		const bool overwrite = false;
		return RocketForgeTraining::ExportVersionForSubmittal(bot.agentName, bot.buildName, bot.segmentId, outputParent, useGPU, overwrite)
			.packageRoot;
	}
}

TEST(SubmittalExport, NormalizesLinuxContainerLineEndings)
{
	using RocketForgeTraining::SubmittalExportInternal::NormalizeLineEndingsToLf;
	ASSERT_EQ(NormalizeLineEndingsToLf("first\r\nsecond\rthird\nfourth"),
		"first\nsecond\nthird\nfourth");
}

TEST(SubmittalExport, WritesPortableBobPackage)
{
	ROCKETFORGE_REQUIRE_COLLISION_MESHES();
	ROCKETFORGE_REQUIRE_RLBOT_CPP_INTERFACE();

	using namespace RocketForgeTraining;

	TempFolder temp{ RocketForgeTests::TestTempDir() };
	const fs::path botRoot = temp.path / "TestBot";
	const fs::path modelsRoot = botRoot / "Models";
	const fs::path outputParent = temp.path / "Exports";

	const std::string agentName = "submittal-package-test-agent";
	const std::string selectedObsName = "submittal-selected-obs";
	const std::string selectedActionName = "submittal-selected-action";
	WriteTextFile(
		botRoot / "CMakeLists.txt",
		"file(GLOB_RECURSE FILES_SRC \"src/*.cpp\" \"src/*.h\")\n"
		"add_library(SubmittalPackageTestAgent OBJECT ${FILES_SRC})\n"
		"target_link_libraries(SubmittalPackageTestAgent PUBLIC GigaLearnInfer)\n"
		"target_sources(Agents INTERFACE $<TARGET_OBJECTS:SubmittalPackageTestAgent>)\n");
	WriteTextFile(
		botRoot / "src" / "SubmittalPackageTestAgent.h",
		"#pragma once\n"
		"#include <GigaLearnInfer/Agent.h>\n"
		"class SubmittalPackageTestAgent : public Agent {\n"
		"public:\n"
		"    std::unique_ptr<Agent> Clone() const override {\n"
		"        return std::make_unique<SubmittalPackageTestAgent>(*this);\n"
		"    }\n"
		"};\n");
	WriteTextFile(
		botRoot / "src" / "SubmittalPackageTestAgent.cpp",
		"#include \"SubmittalPackageTestAgent.h\"\n"
		"#include <GigaLearnInfer/AgentRegistry.h>\n"
		"REGISTER_AGENT(\"submittal-package-test-agent\", SubmittalPackageTestAgent)\n");
	WriteTextFile(
		botRoot / "src" / "SubmittalSelectedObs.h",
		"#pragma once\n"
		"#include <RLGymCPP/ObsBuilders/AdvancedObs.h>\n"
		"class SubmittalSelectedObs : public RLGC::AdvancedObs {\n"
		"public:\n"
		"    RLGC::FList BuildObs(const RLGC::Player&, const RLGC::GameState&) override {\n"
		"        return {};\n"
		"    }\n"
		"};\n");
	WriteTextFile(
		botRoot / "src" / "SubmittalSelectedObs.cpp",
		"#include \"SubmittalSelectedObs.h\"\n"
		"#include <RLGymCPP/ActionParsers/ActionParser.h>\n"
		"#include <GigaLearnInfer/AgentComponentRegistry.h>\n"
		"REGISTER_OBS_BUILDER(\"submittal-selected-obs\", [](const Agent&) {\n"
		"    return std::make_unique<SubmittalSelectedObs>();\n"
		"})\n");
	WriteTextFile(
		botRoot / "src" / "SubmittalSelectedAction.h",
		"#pragma once\n"
		"#include <RLGymCPP/ActionParsers/ActionParser.h>\n"
		"class SubmittalSelectedAction : public RLGC::ActionParser {\n"
		"public:\n"
		"    RLGC::Action ParseAction(int, const RLGC::Player&, const RLGC::GameState&) override {\n"
		"        return {};\n"
		"    }\n"
		"    int GetActionAmount() override { return 1; }\n"
		"};\n");
	WriteTextFile(
		botRoot / "src" / "SubmittalSelectedAction.cpp",
		"#include \"SubmittalSelectedAction.h\"\n"
		"#include <RLGymCPP/ObsBuilders/ObsBuilder.h>\n"
		"#include <GigaLearnInfer/AgentComponentRegistry.h>\n"
		"REGISTER_ACTION_PARSER(\"submittal-selected-action\", [](const Agent&) {\n"
		"    return std::make_unique<SubmittalSelectedAction>();\n"
		"})\n");
	WriteTextFile(
		botRoot / "src" / "SubmittalUnusedObs.h",
		"#pragma once\n"
		"#include <RLGymCPP/ObsBuilders/ObsBuilder.h>\n"
		"class SubmittalUnusedObs : public RLGC::ObsBuilder {\n"
		"public:\n"
		"    RLGC::FList BuildObs(const RLGC::Player&, const RLGC::GameState&) override { return {}; }\n"
		"};\n");
	WriteTextFile(
		botRoot / "src" / "SubmittalUnusedObs.cpp",
		"#include \"SubmittalUnusedObs.h\"\n"
		"#include <RLGymCPP/ActionParsers/ActionParser.h>\n"
		"#include <GigaLearnInfer/AgentComponentRegistry.h>\n"
		"REGISTER_OBS_BUILDER(\"submittal-unused-obs\", [](const Agent&) {\n"
		"    return std::make_unique<SubmittalUnusedObs>();\n"
		"})\n");
	WriteTextFile(
		botRoot / "src" / "SubmittalUnusedAction.h",
		"#pragma once\n"
		"#include <RLGymCPP/ActionParsers/ActionParser.h>\n"
		"class SubmittalUnusedAction : public RLGC::ActionParser {\n"
		"public:\n"
		"    RLGC::Action ParseAction(int, const RLGC::Player&, const RLGC::GameState&) override { return {}; }\n"
		"    int GetActionAmount() override { return 2; }\n"
		"};\n");
	WriteTextFile(
		botRoot / "src" / "SubmittalUnusedAction.cpp",
		"#include \"SubmittalUnusedAction.h\"\n"
		"#include <RLGymCPP/ObsBuilders/ObsBuilder.h>\n"
		"#include <GigaLearnInfer/AgentComponentRegistry.h>\n"
		"REGISTER_ACTION_PARSER(\"submittal-unused-action\", [](const Agent&) {\n"
		"    return std::make_unique<SubmittalUnusedAction>();\n"
		"})\n");
	WriteTextFile(botRoot / ".git" / "config", "must not be copied");
	WriteTextFile(botRoot / ".gitattributes", "must not be copied");
	WriteTextFile(botRoot / ".gitignore", "must not be copied");
	WriteTextFile(botRoot / "CMakeSettings.json", "must not be copied");
	WriteTextFile(botRoot / "README.md", "must not be copied");
	WriteTextFile(botRoot / "out" / "stale.obj", "must not be copied");
	WriteTextFile(
		botRoot / "TestBotTrainingPlan" / "src" / "TestBotTrainingPlan.cpp",
		"must not be copied");
	WriteTextFile(
		botRoot / "rlbot_exports" / "historical.bot.toml",
		"[settings]\nname = \"Historical Export\"\n");
	WriteTextFile(
		botRoot / "forge_debug_settings.json",
		"{\"debug\": true, \"segmentId\": \"unrelated\"}\n");
	WriteTextFile(botRoot / "rlbot" / "bot.toml",
		"[settings]\n"
		"agent_id = \"Test/Old\"\n"
		"name = \"Old Name\"\n"
		"loadout_file = \"loadout.toml\"\n"
		"logo_file = \"logo.jpg\"\n"
		"run_command = \"cmd /C old.bat\"\n\n"
		"[settings.environment]\n"
		"Path = \"C:\\\\old-libtorch\\\\lib\"\n"
		"LD_LIBRARY_PATH = \"/old-libtorch/lib\"\n"
		"CUSTOM_ENV = \"preserved\"\n\n"
		"[details]\n"
		"developer = \"Test\"\n"
		"language = \"C++\"\n"
		"tags = []\n");
	WriteTextFile(botRoot / "rlbot" / "loadout.toml", "[blue_loadout]\ncar_id = 23\n");
	WriteTextFile(botRoot / "rlbot" / "logo.jpg", "logo");
	WriteTextFile(botRoot / "rlbot" / "RLBot.exe", "must not be copied");

	AgentRegistry::Register(agentName, [agentName, modelsRoot]() {
		auto agent = std::make_unique<Agent>();
		agent->agentName = agentName;
		agent->modelPath = modelsRoot;
		return agent;
	});

	const std::string buildName = "Bob Build";
	BuildStorage storage(modelsRoot, buildName);
	storage.EnsureBuildExists();
	TrainingSegmentMetadata segment = {};
	segment.segmentId = "v1.23";
	segment.folderName = storage.MakeSegmentFolderName(segment.segmentId, "test-stage");
	segment.buildName = buildName;
	segment.stageName = "test-stage";
	segment.status = "active";
	storage.SaveSegment(segment);

	const fs::path checkpoint = storage.SegmentCheckpointsRoot(segment.folderName) / "1234567";
	WriteTextFile(checkpoint / "POLICY.lt", "policy");
	WriteTextFile(checkpoint / "SHARED_HEAD.lt", "shared-head");
	WriteTextFile(checkpoint / "CRITIC.lt", "not needed for inference");

	Agent savedAgent;
	savedAgent.agentName = agentName;
	savedAgent.obsBuilderName = selectedObsName;
	savedAgent.actionParserName = selectedActionName;
	savedAgent.tickSkip = 4;
	savedAgent.actionDelay = 3;
	savedAgent.trainingEngineVersion = 2;
	RocketForge::TrainingSetupSnapshot setup = {};
	setup.agentName = agentName;
	setup.stageName = segment.stageName;
	setup.versionId = segment.segmentId;
	setup.agentSetup = RocketForge::CaptureAgentSetupSnapshot(savedAgent);
	ASSERT_TRUE(GGL::JsonSaveToFile(
		setup,
		storage.SegmentRoot(segment.folderName) / RocketForge::TrainingSetupSnapshot::FILE_NAME));

	SubmittalBotMetadata metadata = GetDefaultSubmittalBotMetadata(
		agentName,
		buildName,
		segment.segmentId);
	ASSERT_EQ(metadata.name, std::string("submittal-package-test-agent Bob Build v1.23 1M"));
	ASSERT_EQ(metadata.developer, std::string("Test"));
	ASSERT_TRUE(metadata.hivemind);
	ASSERT_FALSE(metadata.useGPU);
	metadata.name = "Kilobyte";
	metadata.agentId = "TestDeveloper/Kilobyte";
	metadata.description = "A custom submittal export";
	metadata.funFact = "Named during export";
	metadata.sourceLink = "https://example.com/kilobyte";
	metadata.developer = "Test Developer";
	metadata.language = "C++";
	metadata.tags = { "1v1", "teamplay" };
	const SubmittalExportResult result = ExportVersionForSubmittal(
		agentName,
		buildName,
		segment.segmentId,
		outputParent,
		metadata);
	const fs::path packageRoot = outputParent /
		MakeSubmittalExportFolderName(metadata.name);

	ASSERT_TRUE(result.packageRoot == packageRoot);
	ASSERT_EQ(result.projectName, std::string("Kilobyte"));
	ASSERT_EQ(result.latestTimesteps, uint64_t(1234567));
	ASSERT_TRUE(fs::exists(packageRoot / "bob.toml"));
	ASSERT_TRUE(fs::exists(packageRoot / "cpp.Dockerfile"));
	ASSERT_TRUE(ReadTextFile(packageRoot / "cpp.Dockerfile").find('\r') == std::string::npos);
	ASSERT_TRUE(fs::exists(packageRoot / "CMakeLists.txt"));
	ASSERT_TRUE(fs::exists(packageRoot / ".dockerignore"));
	// Templates are written with LF, whatever line endings git checked them out with.
	ASSERT_EQ(ReadTextFile(packageRoot / "README.md"), SubmittalExportInternal::NormalizeLineEndingsToLf(ReadTextFile(
		fs::path(FORGE_SOURCE_DIR) / "ForgeTraining/templates/submittal/README.md")));
	ASSERT_FALSE(TreeContainsFileNamed(packageRoot / "Bots", "README.md"));
	ASSERT_FALSE(TreeContainsFileNamed(packageRoot, ".gitattributes"));
	ASSERT_TRUE(fs::exists(packageRoot / ".gitignore"));
	ASSERT_FALSE(TreeContainsFileNamed(packageRoot / "Bots", ".gitignore"));
	ASSERT_NE(ReadTextFile(packageRoot / ".dockerignore").find("!build-support/**"), std::string::npos);
	ASSERT_FALSE(TreeContainsFileNamed(packageRoot, ".clang-format"));
	ASSERT_FALSE(TreeContainsFileNamed(packageRoot, "CMakeSettings.json"));
	ASSERT_FALSE(fs::exists(packageRoot / "Bots" / "TestBot" / "src" / "SubmittalPackageTestAgent.cpp"));
	ASSERT_FALSE(fs::exists(packageRoot / "Bots" / "TestBot" / "src" / "SubmittalPackageTestAgent.h"));
	ASSERT_TRUE(fs::exists(packageRoot / "Bots" / "ExportedRuntimeRegistration.cpp"));
	ASSERT_TRUE(fs::exists(packageRoot / "Bots" / "TestBot" / "src" / "SubmittalSelectedObs.cpp"));
	ASSERT_TRUE(fs::exists(packageRoot / "Bots" / "TestBot" / "src" / "SubmittalSelectedObs.h"));
	ASSERT_TRUE(fs::exists(packageRoot / "Bots" / "TestBot" / "src" / "SubmittalSelectedAction.cpp"));
	ASSERT_TRUE(fs::exists(packageRoot / "Bots" / "TestBot" / "src" / "SubmittalSelectedAction.h"));
	ASSERT_FALSE(fs::exists(packageRoot / "Bots" / "TestBot" / "src" / "SubmittalUnusedObs.cpp"));
	ASSERT_FALSE(fs::exists(packageRoot / "Bots" / "TestBot" / "src" / "SubmittalUnusedObs.h"));
	ASSERT_FALSE(fs::exists(packageRoot / "Bots" / "TestBot" / "src" / "SubmittalUnusedAction.cpp"));
	ASSERT_FALSE(fs::exists(packageRoot / "Bots" / "TestBot" / "src" / "SubmittalUnusedAction.h"));
	ASSERT_FALSE(fs::exists(packageRoot / "Bots" / "TestBot" / "rlbot_exports"));
	ASSERT_FALSE(fs::exists(packageRoot / "Bots" / "TestBot" / "forge_debug_settings.json"));
	ASSERT_TRUE(fs::exists(packageRoot / "RocketForgeRLBot" / "inc" / "cpp-interface" /
		"library" / "include" / "rlbot" / "Bot.h"));
	ASSERT_TRUE(fs::exists(packageRoot / "RocketForgeRLBot" / "inc" / "cpp-interface" /
		"library" / "TracyHelper.h"));
	ASSERT_TRUE(fs::exists(packageRoot / "RocketForgeRLBot" / "src" / "Runtime" / "RLBotClient.cpp"));
	ASSERT_TRUE(fs::exists(packageRoot / "RocketForgeRLBot" / "src" / "Runtime" / "RLBotCapture.cpp"));
	ASSERT_EQ(ReadTextFile(packageRoot / "scripts" / "rlbot_capture.py"),
		ReadTextFile(fs::path(FORGE_SOURCE_DIR) / "tools" / "rlbot" / "capture" / "capture.py"));
	ASSERT_TRUE(fs::exists(packageRoot / "RocketForgeRLBot" / "src" / "Runtime" / "RLBotClientExtension.h"));
	ASSERT_TRUE(fs::exists(packageRoot / "RocketForgeRLBot" / "src" / "Runtime" / "RLBotLauncher.cpp"));
	ASSERT_TRUE(fs::exists(packageRoot / "RocketForgeRLBot" / "src" / "Runtime" / "PortableMain.cpp"));
	ASSERT_TRUE(fs::exists(packageRoot / "RocketForgeRLBot" / "src" / "Runtime" / "RLBotPolicyRuntime.cpp"));
	ASSERT_TRUE(fs::exists(packageRoot / "RocketForgeRLBot" / "src" / "Runtime" / "RunRLBot.cpp"));
	ASSERT_TRUE(fs::exists(packageRoot / "rlbot" / "POLICY.lt"));
	ASSERT_TRUE(fs::exists(packageRoot / "rlbot" / "SHARED_HEAD.lt"));
	ASSERT_TRUE(fs::exists(packageRoot / "rlbot" / "agent_setup.json"));
	ASSERT_TRUE(fs::exists(packageRoot / "rlbot" / "logo.jpg"));
	ASSERT_FALSE(fs::exists(packageRoot / "rlbot" / "run.bat"));
	ASSERT_FALSE(fs::exists(packageRoot / "rlbot" / "run.sh"));
	ASSERT_TRUE(fs::exists(packageRoot / RocketSim::CollisionMeshesRelativePath() / "soccar" / "mesh_0.cmf"));
	ASSERT_FALSE(fs::exists(packageRoot / "rlbot" / "CRITIC.lt"));
	ASSERT_FALSE(fs::exists(packageRoot / "Bots" / "TestBot" / ".git"));
	ASSERT_FALSE(fs::exists(packageRoot / "Bots" / "TestBot" / "out"));
	ASSERT_FALSE(fs::exists(packageRoot / "Bots" / "TestBot" / "Models"));
	ASSERT_FALSE(fs::exists(packageRoot / "Bots" / "TestBot" / "rlbot"));
	ASSERT_FALSE(fs::exists(packageRoot / "Bots" / "TestBot" / "TestBotTrainingPlan"));
	const fs::path rlgymRoot = packageRoot / "ForgeTraining" / "RLGym";
	ASSERT_FALSE(fs::exists(rlgymRoot / "src" / "RLGymCPP" / "EnvSet"));
	ASSERT_FALSE(fs::exists(rlgymRoot / "src" / "RLGymCPP" / "Rewards"));
	ASSERT_FALSE(fs::exists(rlgymRoot / "src" / "RLGymCPP" / "StateSetters"));
	ASSERT_FALSE(fs::exists(rlgymRoot / "src" / "RLGymCPP" / "TerminalConditions"));
	ASSERT_FALSE(fs::exists(rlgymRoot / "src" / "RLGymCPP" / "ThreadPool.cpp"));
	ASSERT_FALSE(fs::exists(rlgymRoot / "src" / "RLGymCPP" / "ThreadPool.h"));
	ASSERT_FALSE(fs::exists(rlgymRoot / "thread_pool"));
	ASSERT_FALSE(fs::exists(rlgymRoot / "src" / "RLGymCPP" / "ActionParsers" / "DefaultAction.cpp"));
	ASSERT_FALSE(fs::exists(rlgymRoot / "src" / "RLGymCPP" / "ActionParsers" / "DefaultAction.h"));
	ASSERT_TRUE(fs::exists(rlgymRoot / "src" / "RLGymCPP" / "ObsBuilders" / "AdvancedObs.cpp"));
	ASSERT_TRUE(fs::exists(rlgymRoot / "src" / "RLGymCPP" / "ObsBuilders" / "AdvancedObs.h"));
	ASSERT_TRUE(ReadTextFile(rlgymRoot / "CMakeLists.txt").find(
		"src/RLGymCPP/ObsBuilders/AdvancedObs.cpp") != std::string::npos);
	ASSERT_FALSE(fs::exists(rlgymRoot / "src" / "RLGymCPP" / "ObsBuilders" / "DefaultObs.cpp"));
	ASSERT_FALSE(fs::exists(rlgymRoot / "src" / "RLGymCPP" / "ObsBuilders" / "DefaultObs.h"));
	ASSERT_FALSE(fs::exists(rlgymRoot / "src" / "RLGymCPP" / "ObsBuilders" / "DefaultObsPadded.cpp"));
	ASSERT_FALSE(fs::exists(rlgymRoot / "src" / "RLGymCPP" / "ObsBuilders" / "DefaultObsPadded.h"));
	ASSERT_TRUE(ReadTextFile(rlgymRoot / "CMakeLists.txt").find("add_subdirectory(Sim)") != std::string::npos);
	const fs::path simRoot = rlgymRoot / "Sim";
	ASSERT_TRUE(fs::exists(simRoot / "CMakeLists.txt"));
	ASSERT_TRUE(fs::exists(simRoot / "EngineConfig.h.in"));
	ASSERT_TRUE(fs::exists(simRoot / "include" / "RocketSim" / "Engine.h"));
	ASSERT_TRUE(fs::exists(simRoot / "src" / "Engine.cpp"));
	ASSERT_FALSE(fs::exists(simRoot / "src" / "PhysState.cpp"));
	const fs::path rocketSimRoot = rlgymRoot / "RocketSim";
	ASSERT_FALSE(fs::exists(rocketSimRoot / "src" / "BulletLink.cpp"));
	ASSERT_FALSE(fs::exists(rocketSimRoot / "src" / "Sim" / "PhysState" / "PhysState.cpp"));
	const fs::path bulletRoot = rlgymRoot / "RocketSim" / "libsrc" / "bullet3-3.24";
	ASSERT_FALSE(fs::exists(bulletRoot / "BulletCollision" / "BroadphaseCollision" / "btAxisSweep3.cpp"));
	ASSERT_FALSE(fs::exists(bulletRoot / "BulletCollision" / "BroadphaseCollision" / "btAxisSweep3.h"));
	ASSERT_FALSE(fs::exists(bulletRoot / "BulletCollision" / "BroadphaseCollision" / "btAxisSweep3Internal.h"));
	ASSERT_FALSE(fs::exists(bulletRoot / "BulletCollision" / "BroadphaseCollision" / "btBroadphaseProxy.cpp"));
	ASSERT_FALSE(fs::exists(bulletRoot / "BulletCollision" / "BroadphaseCollision" / "btSimpleBroadphase.cpp"));
	ASSERT_FALSE(fs::exists(bulletRoot / "BulletCollision" / "BroadphaseCollision" / "btSimpleBroadphase.h"));
	ASSERT_FALSE(fs::exists(bulletRoot / "BulletCollision" / "CollisionDispatch" / "btGhostObject.cpp"));
	ASSERT_FALSE(fs::exists(bulletRoot / "BulletCollision" / "CollisionDispatch" / "btGhostObject.h"));
	ASSERT_FALSE(fs::exists(bulletRoot / "BulletCollision" / "CollisionDispatch" / "btSphereBoxCollisionAlgorithm.cpp"));
	ASSERT_FALSE(fs::exists(bulletRoot / "BulletCollision" / "CollisionDispatch" / "btSphereBoxCollisionAlgorithm.h"));
	ASSERT_FALSE(fs::exists(bulletRoot / "BulletCollision" / "CollisionShapes" / "btEmptyShape.cpp"));
	ASSERT_FALSE(fs::exists(bulletRoot / "BulletCollision" / "CollisionShapes" / "btEmptyShape.h"));
	ASSERT_FALSE(fs::exists(bulletRoot / "BulletCollision" / "CollisionShapes" / "btShapeHull.cpp"));
	ASSERT_FALSE(fs::exists(bulletRoot / "BulletCollision" / "CollisionShapes" / "btShapeHull.h"));
	ASSERT_FALSE(fs::exists(bulletRoot / "BulletCollision" / "CollisionShapes" / "btTetrahedronShape.cpp"));
	ASSERT_FALSE(fs::exists(bulletRoot / "BulletCollision" / "CollisionShapes" / "btTetrahedronShape.h"));
	ASSERT_FALSE(fs::exists(bulletRoot / "BulletCollision" / "CollisionShapes" / "btTriangleBuffer.cpp"));
	ASSERT_FALSE(fs::exists(bulletRoot / "BulletCollision" / "CollisionShapes" / "btTriangleBuffer.h"));
	ASSERT_FALSE(fs::exists(bulletRoot / "LinearMath" / "btConvexHull.cpp"));
	ASSERT_FALSE(fs::exists(bulletRoot / "LinearMath" / "btConvexHull.h"));
	ASSERT_FALSE(fs::exists(bulletRoot / "LinearMath" / "btPolarDecomposition.cpp"));
	ASSERT_FALSE(fs::exists(bulletRoot / "LinearMath" / "btPolarDecomposition.h"));
	ASSERT_FALSE(fs::exists(bulletRoot / "LinearMath" / "btQuickprof.cpp"));
	ASSERT_FALSE(fs::exists(bulletRoot / "LinearMath" / "btReducedVector.cpp"));
	ASSERT_FALSE(fs::exists(bulletRoot / "LinearMath" / "btReducedVector.h"));
	ASSERT_FALSE(fs::exists(bulletRoot / "LinearMath" / "btSerializer.cpp"));
	ASSERT_FALSE(fs::exists(bulletRoot / "LinearMath" / "btSerializer64.cpp"));
	ASSERT_FALSE(fs::exists(bulletRoot / "LinearMath" / "btThreads.cpp"));
	ASSERT_FALSE(fs::exists(bulletRoot / "LinearMath" / "TaskScheduler" / "btTaskScheduler.cpp"));
	ASSERT_FALSE(fs::exists(bulletRoot / "LinearMath" / "TaskScheduler" / "btThreadSupportInterface.h"));
	ASSERT_FALSE(fs::exists(bulletRoot / "LinearMath" / "TaskScheduler" / "btThreadSupportPosix.cpp"));
	ASSERT_FALSE(fs::exists(bulletRoot / "LinearMath" / "TaskScheduler" / "btThreadSupportWin32.cpp"));
	const fs::path inferRoot = packageRoot / "ForgeTraining" / "GigaLearn" / "GigaLearnInfer";
	ASSERT_TRUE(fs::exists(inferRoot / "src" / "public" / "GigaLearnInfer" / "JsonArchive.h"));
	ASSERT_FALSE(fs::exists(inferRoot / "src" / "private" / "MagSGD.cpp"));
	ASSERT_FALSE(fs::exists(inferRoot / "src" / "public" / "GigaLearnInfer" / "MagSGD.h"));
	ASSERT_TRUE(fs::exists(inferRoot / "src" / "private" / "ModelManager.cpp"));
	ASSERT_TRUE(fs::exists(inferRoot / "src" / "public" / "GigaLearnInfer" / "ModelManager.h"));
	ASSERT_FALSE(TreeContainsFileNamed(packageRoot, "ModelOptimizer.cpp"));
	ASSERT_FALSE(TreeContainsFileNamed(packageRoot, "DefaultAgentComponents.cpp"));
	ASSERT_FALSE(fs::exists(packageRoot / "RocketForgeRLBot" / "src" / "Development"));
	ASSERT_FALSE(fs::exists(packageRoot / "RocketForgeRLBot" / "inc" / "cpp-interface" / "benchmark"));
	ASSERT_FALSE(fs::exists(packageRoot / "RocketForgeRLBot" / "inc" / "cpp-interface" / "examples"));
	ASSERT_TRUE(fs::exists(packageRoot / "RocketForgeRLBot" / "cmake" / "PrepareCppInterface.cmake"));
	ASSERT_TRUE(fs::exists(packageRoot / "RocketForgeRLBot" / "patches" / "cpp-interface.patch"));
	ASSERT_FALSE(fs::exists(packageRoot / "RocketForgeRLBot" / "cmake" / "UpdateRlBotExes.cmake"));
	ASSERT_FALSE(fs::exists(packageRoot / "RocketForgeRLBot" / "run.bat.in"));

	const std::string botToml = ReadTextFile(packageRoot / "rlbot" / "bot.toml");
	ASSERT_TRUE(botToml.find("agent_id = \"TestDeveloper/Kilobyte\"") != std::string::npos);
	ASSERT_TRUE(botToml.find("name = \"Kilobyte\"") != std::string::npos);
	ASSERT_TRUE(botToml.find("description = \"A custom submittal export\"") != std::string::npos);
	ASSERT_TRUE(botToml.find("fun_fact = \"Named during export\"") != std::string::npos);
	ASSERT_TRUE(botToml.find("source_link = \"https://example.com/kilobyte\"") != std::string::npos);
	ASSERT_TRUE(botToml.find("developer = \"Test Developer\"") != std::string::npos);
	ASSERT_TRUE(botToml.find("tags = [\"1v1\", \"teamplay\"]") != std::string::npos);
	ASSERT_TRUE(botToml.find("run_command = \"RLBot.exe --hivemind on\"") != std::string::npos);
	ASSERT_TRUE(botToml.find("run_command_linux = \"./RLBot --hivemind on\"") != std::string::npos);
	ASSERT_TRUE(botToml.find("[settings.environment]") != std::string::npos);
	ASSERT_TRUE(botToml.find("Path =") == std::string::npos);
	ASSERT_TRUE(botToml.find("PATH =") == std::string::npos);
	ASSERT_TRUE(botToml.find("LD_LIBRARY_PATH =") == std::string::npos);
	ASSERT_TRUE(botToml.find("CUSTOM_ENV = \"preserved\"") != std::string::npos);
	ASSERT_TRUE(botToml.find("old-libtorch") == std::string::npos);
	const fs::path torchOnlyBotToml = temp.path / "torch-only.bot.toml";
	WriteTextFile(
		torchOnlyBotToml,
		"[settings]\nname = \"Old Name\"\n\n"
		"[settings.environment]\n"
		"Path = \"C:\\\\old-libtorch\\\\lib\"\n"
		"LD_LIBRARY_PATH = \"/old-libtorch/lib\"\n\n"
		"[details]\ndeveloper = \"Test\"\n");
	const std::string torchOnlyPortable = SubmittalExportInternal::MakePortableBotToml(
		torchOnlyBotToml,
		metadata,
		false);
	ASSERT_TRUE(torchOnlyPortable.find("[settings.environment]") == std::string::npos);
	ASSERT_TRUE(ReadTextFile(packageRoot / "bob.toml").find(
		"project_name = \"Kilobyte\"") != std::string::npos);

	const std::string rootCMake = ReadTextFile(packageRoot / "CMakeLists.txt");
	ASSERT_TRUE(rootCMake.find("RLBOT_DEFAULT_AGENT=\"submittal-package-test-agent\"") != std::string::npos);
	ASSERT_TRUE(rootCMake.find("target_compile_definitions(RLBotCore PRIVATE") != std::string::npos);
	ASSERT_TRUE(rootCMake.find("_GLIBCXX_USE_CXX11_ABI=0") != std::string::npos);
	ASSERT_TRUE(rootCMake.find("add_subdirectory(ForgeTraining/GigaLearn/GigaLearnInfer)") != std::string::npos);
	ASSERT_TRUE(rootCMake.find("add_subdirectory(RocketForgeRLBot)") != std::string::npos);
	const std::string runtimeRegistration =
		ReadTextFile(packageRoot / "Bots" / "ExportedRuntimeRegistration.cpp");
	ASSERT_TRUE(runtimeRegistration.find(
		"REGISTER_AGENT(\"submittal-package-test-agent\", Agent)") != std::string::npos);
	const std::string interfaceCMake = ReadTextFile(
		packageRoot / "RocketForgeRLBot" / "inc" / "cpp-interface" / "CMakeLists.txt");
	ASSERT_TRUE(interfaceCMake.find("add_subdirectory(examples/") == std::string::npos);
	const std::string interfaceLibraryCMake = ReadTextFile(
		packageRoot / "RocketForgeRLBot" / "inc" / "cpp-interface" / "library" / "CMakeLists.txt");
	ASSERT_TRUE(interfaceLibraryCMake.find("add_library(${PROJECT_NAME}-static STATIC)") != std::string::npos);
	ASSERT_TRUE(interfaceLibraryCMake.find("add_library(${PROJECT_NAME} SHARED)") == std::string::npos);
	ASSERT_TRUE(ReadTextFile(rlgymRoot / "src" / "RLGymCPP" / "CommonValues.h").find(
		"#include \"Framework.h\"") != std::string::npos);
	const std::string runtimeSetup = ReadTextFile(packageRoot / "rlbot" / "agent_setup.json");
	ASSERT_TRUE(runtimeSetup.find("\"critic\"") == std::string::npos);
	ASSERT_TRUE(runtimeSetup.find("\"optim_type\"") == std::string::npos);
	const std::string runtimeModelConfig = ReadTextFile(
		inferRoot / "src" / "public" / "GigaLearnInfer" / "ModelConfig.h");
	ASSERT_TRUE(runtimeModelConfig.find("ModelOptimType") != std::string::npos);
	ASSERT_TRUE(runtimeModelConfig.find("optimType") != std::string::npos);
	const std::string runtimeAgent = ReadTextFile(
		inferRoot / "src" / "public" / "GigaLearnInfer" / "Agent.h");
	ASSERT_TRUE(runtimeAgent.find("critic") != std::string::npos);
	ASSERT_TRUE(runtimeAgent.find("MakeInferenceModels") != std::string::npos);
	ASSERT_TRUE(runtimeAgent.find("inferenceOnly") != std::string::npos);
	const std::string runtimeSnapshot = ReadTextFile(
		inferRoot / "src" / "private" / "AgentSetupSnapshot.cpp");
	ASSERT_TRUE(runtimeSnapshot.find("critic") != std::string::npos);
	ASSERT_TRUE(runtimeSnapshot.find("optim") != std::string::npos);
	ASSERT_TRUE(runtimeSnapshot.find("DiffAgentSetupSnapshots") != std::string::npos);
	ASSERT_TRUE(runtimeSnapshot.find("AgentSetupSnapshotToJson") != std::string::npos);
	const std::string runtimeModels = ReadTextFile(
		inferRoot / "src" / "public" / "GigaLearnInfer" / "Models.h");
	ASSERT_TRUE(runtimeModels.find("Save(") != std::string::npos);
	ASSERT_TRUE(runtimeModels.find("Clone") != std::string::npos);
	ASSERT_TRUE(runtimeModels.find("CopyParams") != std::string::npos);
	ASSERT_TRUE(runtimeModels.find("torch::optim") == std::string::npos);
	ASSERT_TRUE(runtimeModels.find("MagSGD") == std::string::npos);
	ASSERT_TRUE(runtimeModels.find("ModelOptimizer") == std::string::npos);
	ASSERT_TRUE(runtimeModels.find("SetOptimLR") == std::string::npos);
	ASSERT_TRUE(runtimeModels.find("StepOptim") == std::string::npos);
	ASSERT_TRUE(runtimeModels.find("saveOptim") == std::string::npos);
	ASSERT_TRUE(runtimeModels.find("loadOptim") == std::string::npos);
	ASSERT_TRUE(runtimeModels.find("GIGALEARN_INFERENCE_ONLY") == std::string::npos);
	const std::string runtimeModelsCpp = ReadTextFile(
		inferRoot / "src" / "private" / "Models.cpp");
	ASSERT_TRUE(runtimeModelsCpp.find("torch::optim") == std::string::npos);
	ASSERT_TRUE(runtimeModelsCpp.find("ModelOptimizer") == std::string::npos);
	ASSERT_TRUE(runtimeModelsCpp.find("GIGALEARN_INFERENCE_ONLY") == std::string::npos);
	const std::string runtimeGigaCMake = ReadTextFile(inferRoot / "CMakeLists.txt");
	ASSERT_TRUE(runtimeGigaCMake.find("GIGALEARN_INFERENCE_ONLY") == std::string::npos);
	const fs::path sourceInferRoot = fs::path(FORGE_SOURCE_DIR) /
		"ForgeTraining" / "GigaLearn" / "GigaLearnInfer";
	ASSERT_EQ(
		ReadTextFile(inferRoot / "src" / "private" / "Agent.cpp"),
		ReadTextFile(sourceInferRoot / "src" / "private" / "Agent.cpp"));
	ASSERT_EQ(runtimeModels, ReadTextFile(
		sourceInferRoot / "src" / "public" / "GigaLearnInfer" / "Models.h"));
	const std::string runtimeInfer = ReadTextFile(
		inferRoot / "src" / "public" / "GigaLearnInfer" / "InferUnit.h");
	ASSERT_TRUE(runtimeInfer.find("captureDiagnostics") != std::string::npos);
	ASSERT_TRUE(runtimeInfer.find("BatchInfer") != std::string::npos);
	const std::string runtimePolicy = ReadTextFile(
		packageRoot / "RocketForgeRLBot" / "src" / "Runtime" / "RLBotPolicyRuntime.h");
	ASSERT_TRUE(runtimePolicy.find("Diagnostics") == std::string::npos);
	ASSERT_TRUE(runtimePolicy.find("actionProbabilities") == std::string::npos);
	const fs::path sourceRLBotRuntime = fs::path(FORGE_SOURCE_DIR) / "RLBot" / "src" / "Runtime";
	const std::string runtimeClient = ReadTextFile(
		packageRoot / "RocketForgeRLBot" / "src" / "Runtime" / "RLBotClient.cpp");
	ASSERT_EQ(runtimeClient, ReadTextFile(sourceRLBotRuntime / "RLBotClient.cpp"));
	ASSERT_EQ(
		ReadTextFile(packageRoot / "RocketForgeRLBot" / "src" / "Runtime" / "RunRLBot.cpp"),
		ReadTextFile(sourceRLBotRuntime / "RunRLBot.cpp"));
	ASSERT_EQ(
		ReadTextFile(packageRoot / "RocketForgeRLBot" / "src" / "Runtime" / "RLBotLauncher.cpp"),
		ReadTextFile(sourceRLBotRuntime / "RLBotLauncher.cpp"));
	const std::string runtimeCMake = ReadTextFile(packageRoot / "RocketForgeRLBot" / "CMakeLists.txt");
	ASSERT_TRUE(runtimeCMake.find("add_executable(RLBotCore") != std::string::npos);
	ASSERT_TRUE(runtimeCMake.find("add_executable(RLBot src/Runtime/RLBotLauncher.cpp)") != std::string::npos);
	ASSERT_TRUE(runtimeCMake.find("target_link_libraries(RLBot PRIVATE shell32)") != std::string::npos);
	ASSERT_EQ(ReadTextFile(packageRoot / "rlbot/device.txt"), "cpu\n");
	ASSERT_TRUE(fs::exists(packageRoot / "scripts/build_bob.py"));
	ASSERT_TRUE(fs::exists(packageRoot / "cmake/PortableTorch.cmake"));
	const std::string dockerfile = ReadTextFile(packageRoot / "cpp.Dockerfile");
	ASSERT_TRUE(dockerfile.find("x86_64-pc-windows-msvc") != std::string::npos);
	ASSERT_TRUE(dockerfile.find(
		"botpack_x86_64-linux.tar.xz") != std::string::npos);
	ASSERT_TRUE(dockerfile.find("botpack_x86_64-windows.tar.xz") != std::string::npos);
	ASSERT_TRUE(dockerfile.find(
		"/tmp/pytorch-linux/torch-archive/torch") != std::string::npos);
	ASSERT_TRUE(dockerfile.find("-DLIBTORCH_ROOT=/tmp/libtorch-sdk") != std::string::npos);
	ASSERT_TRUE(dockerfile.find("download.pytorch.org/libtorch") == std::string::npos);
	ASSERT_TRUE(dockerfile.find("copy-pe-dll-closure") == std::string::npos);
	ASSERT_TRUE(dockerfile.find("RLBotCore") != std::string::npos);
	ASSERT_TRUE(dockerfile.find("000-runtime") != std::string::npos);
	ASSERT_TRUE(dockerfile.find("\"--sort=name\"") != std::string::npos);

	bool rejectedExistingPackage = false;
	try {
		ExportVersionForSubmittal(agentName, buildName, segment.segmentId, outputParent, metadata);
	} catch (const std::exception&) {
		rejectedExistingPackage = true;
	}
	ASSERT_TRUE(rejectedExistingPackage);

	// An explicit destination lets build checks compile the exact generated package.
	if (const char* artifactPath = std::getenv("ROCKETFORGE_SUBMITTAL_TEST_PACKAGE_DIR")) {
		const fs::path artifactRoot = artifactPath;
		ASSERT_TRUE(artifactRoot.is_absolute());
		ASSERT_FALSE(fs::exists(artifactRoot));
		fs::copy(packageRoot, artifactRoot, fs::copy_options::recursive);
	}

	metadata.hivemind = false;
	metadata.useGPU = true;
	WriteTextFile(packageRoot / "stale.txt", "old package");
	const bool overwrite = true;
	ExportVersionForSubmittal(agentName, buildName, segment.segmentId, outputParent, metadata, overwrite);
	ASSERT_FALSE(fs::exists(packageRoot / "stale.txt"));
	ASSERT_EQ(ReadTextFile(packageRoot / "rlbot/device.txt"), "gpu\n");
	ASSERT_NE(ReadTextFile(packageRoot / "rlbot" / "bot.toml").find("hivemind = false"), std::string::npos);
	ASSERT_NE(ReadTextFile(packageRoot / "rlbot" / "bot.toml").find("RLBot.exe --hivemind off"), std::string::npos);

	// Policy-only agents do not produce a SHARED_HEAD.lt checkpoint artifact.
	savedAgent.sharedHead.layerSizes = {};
	setup.agentSetup = RocketForge::CaptureAgentSetupSnapshot(savedAgent);
	ASSERT_TRUE(GGL::JsonSaveToFile(
		setup,
		storage.SegmentRoot(segment.folderName) / RocketForge::TrainingSetupSnapshot::FILE_NAME));
	ASSERT_TRUE(fs::remove(checkpoint / "SHARED_HEAD.lt"));

	ExportVersionForSubmittal(agentName, buildName, segment.segmentId, outputParent, metadata, true);
	ASSERT_TRUE(fs::exists(packageRoot / "rlbot" / "POLICY.lt"));
	ASSERT_FALSE(fs::exists(packageRoot / "rlbot" / "SHARED_HEAD.lt"));
}

TEST(SubmittalExport, CommentNamingRegistrationMacroIsNotARegistration)
{
	TempFolder temp{ RocketForgeTests::TestTempDir() };
	WriteTextFile(
		temp.path / "src" / "CommentedAgent.cpp",
		"#include <GigaLearnInfer/AgentRegistry.h>\n"
		"REGISTER_AGENT(\"commented-agent\", Agent)\n"
		"// A custom builder needs REGISTER_OBS_BUILDER in its own source file.\n"
		"const char* obsName = \"AdvancedObs\";\n");

	const RocketForgeTraining::SubmittalExportInternal::BotSourceExport exported =
		RocketForgeTraining::SubmittalExportInternal::ResolveBotSourceExport(temp.path, "commented-agent", "AdvancedObs", "");
	EXPECT_EQ(exported.rlgymFiles.count("src/RLGymCPP/ObsBuilders/AdvancedObs.cpp"), size_t(1));
}

TEST(SubmittalExport, AdvancedObsPaddedShipsItsBaseBuilder)
{
	TempFolder temp{ RocketForgeTests::TestTempDir() };
	WriteTextFile(
		temp.path / "src" / "PaddedAgent.cpp",
		"#include <GigaLearnInfer/AgentRegistry.h>\n"
		"REGISTER_AGENT(\"padded-agent\", Agent)\n");

	const RocketForgeTraining::SubmittalExportInternal::BotSourceExport exported =
		RocketForgeTraining::SubmittalExportInternal::ResolveBotSourceExport(temp.path, "padded-agent", "AdvancedObsPadded", "");
	EXPECT_EQ(exported.rlgymFiles.count("src/RLGymCPP/ObsBuilders/AdvancedObsPadded.cpp"), size_t(1));
	EXPECT_EQ(exported.rlgymFiles.count("src/RLGymCPP/ObsBuilders/AdvancedObsPadded.h"), size_t(1));
	EXPECT_EQ(exported.rlgymFiles.count("src/RLGymCPP/ObsBuilders/AdvancedObs.cpp"), size_t(1));
	EXPECT_EQ(exported.rlgymFiles.count("src/RLGymCPP/ObsBuilders/AdvancedObs.h"), size_t(1));
	EXPECT_NE(exported.registrationSource.find("RLGC::AdvancedObsPadded>(agent.maxTeamSize)"), std::string::npos);
}

TEST(SubmittalExport, PredictionEngineAloneNeedsV3)
{
	Agent agent;
	agent.trainingEngineVersion = 2;
	agent.ballPredictionEngineVersion = 0;
	EXPECT_FALSE(RocketForgeTraining::PackageNeedsV3(agent));

	agent.ballPredictionEngineVersion = 3;
	EXPECT_TRUE(RocketForgeTraining::PackageNeedsV3(agent));

	agent.trainingEngineVersion = 3;
	agent.ballPredictionEngineVersion = 2;
	EXPECT_TRUE(RocketForgeTraining::PackageNeedsV3(agent));
}

TEST(SubmittalExport, V2PackageHasNoRust)
{
	ROCKETFORGE_REQUIRE_COLLISION_MESHES();
	ROCKETFORGE_REQUIRE_RLBOT_CPP_INTERFACE();

	TempFolder temp{ RocketForgeTests::TestTempDir() };
	const EngineTestBot bot = MakeEngineTestBot(temp.path, "submittal-v2-engine-agent", 2, 0);
	const fs::path packageRoot = ExportEngineTestBot(bot, temp.path / "Exports");

	EXPECT_FALSE(fs::exists(packageRoot / "ForgeTraining" / "RLGym" / "RocketSimV3"));
	EXPECT_FALSE(fs::exists(packageRoot / "cmake" / "DetectRocketSimV3.cmake"));
	EXPECT_NE(ReadTextFile(packageRoot / "CMakeLists.txt").find("ROCKETFORGE_ENABLE_V3 OFF"), std::string::npos);
	EXPECT_EQ(ReadTextFile(packageRoot / "CMakeLists.txt").find("ROCKETFORGE_ROCKETSIM_VERSION"), std::string::npos);
}

TEST(SubmittalExport, V3PackageEnablesV3)
{
	ROCKETFORGE_REQUIRE_COLLISION_MESHES();
	ROCKETFORGE_REQUIRE_RLBOT_CPP_INTERFACE();
	ROCKETFORGE_REQUIRE_ENGINE(RocketSim::EngineId::V3);

	std::vector<std::vector<std::string>> cargoCalls;
	RocketForgeTraining::SubmittalExportInternal::Testing::ScopedCargoRunner fakeCargo(
		[&](const std::vector<std::string>& arguments, const fs::path& workingDirectory, const std::string&) {
			cargoCalls.push_back(arguments);
			if (arguments.front() == "vendor")
				fs::create_directories(workingDirectory / "vendor" / "glam");
		});

	TempFolder temp{ RocketForgeTests::TestTempDir() };
	const EngineTestBot bot = MakeEngineTestBot(temp.path, "submittal-v3-engine-agent", 3, 0);
	const fs::path packageRoot = ExportEngineTestBot(bot, temp.path / "Exports");

	ASSERT_EQ(cargoCalls.size(), size_t(2));
	EXPECT_EQ(cargoCalls[0].front(), "metadata");
	EXPECT_EQ(cargoCalls[1].front(), "vendor");
	EXPECT_TRUE(fs::exists(packageRoot / "ForgeTraining" / "RLGym" / "RocketSimV3" / "bindings" / "Cargo.toml"));
	EXPECT_NE(ReadTextFile(packageRoot / "CMakeLists.txt").find("ROCKETFORGE_ENABLE_V3 ON"), std::string::npos);
}

TEST(SubmittalExport, V3SourcesCopyWithoutCargo)
{
	const fs::path projectRoot = FORGE_SOURCE_DIR;
	const fs::path v3Source = projectRoot / "ForgeTraining" / "RLGym" / "RocketSimV3";
	if (!fs::exists(v3Source / "upstream" / "rocketsim" / "Cargo.toml"))
		GTEST_SKIP() << "The RocketSim v3 submodule isn't checked out (ForgeTraining/RLGym/RocketSimV3/upstream).";

	std::vector<fs::path> cargoDirectories;
	RocketForgeTraining::SubmittalExportInternal::Testing::ScopedCargoRunner fakeCargo(
		[&](const std::vector<std::string>&, const fs::path& workingDirectory, const std::string&) {
			cargoDirectories.push_back(workingDirectory);
		});

	TempFolder temp{ RocketForgeTests::TestTempDir() };
	RocketForgeTraining::SubmittalExportInternal::CopyRocketSimV3(projectRoot, temp.path);

	const fs::path v3Root = temp.path / "ForgeTraining" / "RLGym" / "RocketSimV3";
	const fs::path bindings = v3Root / "bindings";
	EXPECT_TRUE(fs::exists(v3Root / "CMakeLists.txt"));
	EXPECT_TRUE(fs::exists(temp.path / "cmake" / "DetectRocketSimV3.cmake"));
	EXPECT_TRUE(fs::exists(bindings / "Cargo.lock"));
	EXPECT_TRUE(fs::exists(bindings / "rust-toolchain.toml"));
	EXPECT_TRUE(fs::exists(bindings / "include" / "rocketsim_ffi.h"));
	EXPECT_TRUE(fs::exists(bindings / "src" / "lib.rs"));
	EXPECT_FALSE(fs::exists(bindings / "target"));
	EXPECT_FALSE(fs::exists(bindings / "tests"));
	EXPECT_TRUE(fs::exists(v3Root / "upstream" / "Cargo.toml"));
	EXPECT_TRUE(fs::exists(v3Root / "upstream" / "rocketsim" / "Cargo.toml"));
	EXPECT_TRUE(fs::exists(v3Root / "upstream" / "rocketsim" / "src" / "lib.rs"));
	EXPECT_FALSE(fs::exists(v3Root / "upstream" / "rocketsim_vis"));
	EXPECT_FALSE(fs::exists(v3Root / "upstream" / "rocketsim_test"));
	EXPECT_NE(ReadTextFile(bindings / ".cargo" / "config.toml").find("directory = \"vendor\""), std::string::npos);

	// The header check is the only use of cbindgen, and the package ships the checked header.
	EXPECT_FALSE(fs::exists(bindings / "build.rs"));
	EXPECT_EQ(ReadTextFile(bindings / "Cargo.toml").find("cbindgen"), std::string::npos);

	ASSERT_EQ(cargoDirectories.size(), size_t(2));
	EXPECT_TRUE(cargoDirectories[0] == bindings && cargoDirectories[1] == bindings);
}

TEST(SubmittalExport, V3PackageVendorsRealCrates)
{
	ROCKETFORGE_REQUIRE_ENGINE(RocketSim::EngineId::V3);

	using RocketForgeTraining::SubmittalExportInternal::RunConfiguredCargo;

	// Only this probe may skip the test: once the source's locked crates are cached, every failure in the export is real.
	const fs::path sourceBindings = fs::path(FORGE_SOURCE_DIR) / "ForgeTraining" / "RLGym" / "RocketSimV3" / "bindings";
	try {
		RunConfiguredCargo({ "fetch", "--offline", "--locked", "--manifest-path", "Cargo.toml" }, sourceBindings, "find the locked crates offline");
	} catch (const std::exception& error) {
		GTEST_SKIP() << "Vendoring RocketSim v3's crates needs network or a warm cargo cache: " << error.what();
	}

	RocketForgeTraining::SubmittalExportInternal::Testing::ScopedCargoRunner offlineCargo(
		[](const std::vector<std::string>& arguments, const fs::path& workingDirectory, const std::string& purpose) {
			std::vector<std::string> offline = arguments;
			offline.push_back("--offline");
			RunConfiguredCargo(offline, workingDirectory, purpose);
		});

	TempFolder temp{ RocketForgeTests::TestTempDir() };
	RocketForgeTraining::SubmittalExportInternal::CopyRocketSimV3(fs::path(FORGE_SOURCE_DIR), temp.path);

	const fs::path bindings = temp.path / "ForgeTraining" / "RLGym" / "RocketSimV3" / "bindings";
	EXPECT_TRUE(fs::is_directory(bindings / "vendor" / "glam"));
	EXPECT_EQ(ReadTextFile(bindings / "Cargo.lock").find("cbindgen"), std::string::npos);
	EXPECT_FALSE(fs::exists(bindings / "vendor" / "cbindgen"));
}

TEST(SubmittalExport, RemovesEveryBuildDependenciesTable)
{
	const std::string manifest =
		"[package]\r\n"
		"name = \"kept\"\n\n"
		"[build-dependencies]\n"
		"cbindgen = \"0.29\"\n\n"
		"[dependencies]\n"
		"glam = \"=0.33.5\"\n\n"
		"[build-dependencies.cc]\n"
		"version = \"1\"\n\n"
		"[target.'cfg(windows)'.build-dependencies]\n"
		"winres = \"0.1\"\n\n"
		"[target.\"cfg(unix)\".dependencies]\n"
		"libc = \"0.2\"\n\n"
		"  [build-dependencies] # trailing comment\n"
		"bindgen = \"0.70\"\n\n"
		"[profile.release]\n"
		"lto = true\n";

	const std::string trimmed = RocketForgeTraining::SubmittalExportInternal::RemoveBuildDependencies(manifest);
	for (const char* removed : { "cbindgen", "[build-dependencies", "winres", "bindgen", "version = \"1\"" })
		EXPECT_EQ(trimmed.find(removed), std::string::npos) << removed;
	for (const char* kept : { "name = \"kept\"", "glam = \"=0.33.5\"", "[target.\"cfg(unix)\".dependencies]", "libc", "lto = true" })
		EXPECT_NE(trimmed.find(kept), std::string::npos) << kept;
	EXPECT_EQ(trimmed.find('\r'), std::string::npos);
}

TEST(SubmittalExport, V3PackageWithoutV3FailsBeforeWriting)
{
	TempFolder temp{ RocketForgeTests::TestTempDir() };
	const EngineTestBot bot = MakeEngineTestBot(temp.path, "submittal-missing-v3-agent", 2, 3);
	const fs::path outputParent = temp.path / "Exports";

	RocketSim::Testing::ScopedEngineUnavailable missingV3(RocketSim::EngineId::V3, "RocketSim v3 is missing for this test.");
	try {
		ExportEngineTestBot(bot, outputParent);
		FAIL() << "Exporting a v3 bot without v3 must throw.";
	} catch (const RocketSim::EngineUnavailableError& error) {
		EXPECT_EQ(std::string(error.what()), "RocketSim v3 is missing for this test.");
	}
	EXPECT_FALSE(fs::exists(outputParent));
}

TEST(SubmittalExport, SegmentSetupPicksEngine)
{
	ROCKETFORGE_REQUIRE_COLLISION_MESHES();
	ROCKETFORGE_REQUIRE_RLBOT_CPP_INTERFACE();

	// Keeps a wrongly chosen v3 export from running real cargo.
	RocketForgeTraining::SubmittalExportInternal::Testing::ScopedCargoRunner fakeCargo(
		[](const std::vector<std::string>& arguments, const fs::path& workingDirectory, const std::string&) {
			if (arguments.front() == "vendor")
				fs::create_directories(workingDirectory / "vendor" / "glam");
		});

	TempFolder temp{ RocketForgeTests::TestTempDir() };
	const EngineTestBot bot = MakeEngineTestBot(temp.path, "submittal-legacy-engine-agent", 3, 0);
	RocketForgeTraining::BuildStorage storage(temp.path / "EngineBot" / "Models", bot.buildName);
	const fs::path setupPath = storage.SegmentRoot(storage.LoadSegmentById(bot.segmentId)->folderName)
		/ RocketForge::TrainingSetupSnapshot::FILE_NAME;

	// A setup that exists but can't be read must not fall back to the compiled agent's engine.
	WriteTextFile(setupPath, "broken");
	EXPECT_THROW(ExportEngineTestBot(bot, temp.path / "B"), std::runtime_error);
	EXPECT_THROW(
		RocketForgeTraining::GetDefaultSubmittalBotMetadata(bot.agentName, bot.buildName, bot.segmentId),
		std::runtime_error);

	// A segment saved before setups were recorded trained on v2, though the agent now defaults to v3.
	fs::remove(setupPath);
	WriteTextFile(setupPath.parent_path() / "checkpoints" / "1000" / "SHARED_HEAD.lt", "shared head");
	const fs::path packageRoot = ExportEngineTestBot(bot, temp.path / "L");
	const nlohmann::json runtimeSetup = nlohmann::json::parse(ReadTextFile(packageRoot / "rlbot" / "agent_setup.json"));
	EXPECT_EQ(runtimeSetup.at("training_engine_version"), 2);
	EXPECT_EQ(runtimeSetup.at("ball_prediction_engine_version"), 2);
	EXPECT_FALSE(fs::exists(packageRoot / "ForgeTraining" / "RLGym" / "RocketSimV3"));
}

TEST(SubmittalExport, KeepsHivemindAndPortableCommandsConsistent)
{
	using namespace RocketForgeTraining;
	using SubmittalExportInternal::MakePortableBotToml;
	using SubmittalExportInternal::ReadSubmittalBotMetadata;

	TempFolder temp{ RocketForgeTests::TestTempDir() };
	SubmittalBotMetadata metadata;
	metadata.name = "Public Test Bot";
	metadata.agentId = "Tests/PublicBot";
	ASSERT_TRUE(metadata.hivemind);
	ASSERT_FALSE(metadata.useGPU);

	const fs::path source = temp.path / "bot.toml";
	ASSERT_TRUE(ReadSubmittalBotMetadata(source, metadata.name).hivemind);

	const bool hasLoadout = false;
	for (const std::string input : {
		std::string{},
		std::string("[settings]\nname = \"Old\"\n[details]\ndeveloper = \"Test\"\n"),
		std::string("[settings]\nhivemind = true\nrun_command = \"old.exe --hivemind off\"\nrun_command_linux = \"old\"\n[details]\n"),
		std::string("[settings]\nhivemind = false\n[settings.environment]\nCUSTOM_ENV = \"kept\"\n") }) {
		for (bool enabled : { false, true }) {
			SCOPED_TRACE(input + (enabled ? " enabled" : " disabled"));
			WriteTextFile(source, input);
			ASSERT_EQ(ReadSubmittalBotMetadata(source, metadata.name).hivemind, input.find("hivemind = false") == std::string::npos);

			metadata.hivemind = enabled;
			const std::string output = MakePortableBotToml(input.empty() ? fs::path{} : source, metadata, hasLoadout);
			const std::string value = enabled ? "true" : "false";
			const std::string option = enabled ? "on" : "off";
			const size_t setting = output.find("hivemind = " + value);
			ASSERT_NE(setting, std::string::npos);
			ASSERT_EQ(output.find("hivemind = ", setting + 1), std::string::npos);
			ASSERT_NE(output.find("run_command = \"RLBot.exe --hivemind " + option + "\""), std::string::npos);
			ASSERT_NE(output.find("run_command_linux = \"./RLBot --hivemind " + option + "\""), std::string::npos);

			WriteTextFile(source, output);
			ASSERT_EQ(ReadSubmittalBotMetadata(source, metadata.name).hivemind, enabled);
		}
	}

	WriteTextFile(source, "[settings]\nhivemind = true # grouped\n[details]\nhivemind = false\n");
	ASSERT_TRUE(ReadSubmittalBotMetadata(source, metadata.name).hivemind);
}

TEST(SubmittalExport, DefaultsToCpuWhateverTheLocalDeviceSelection)
{
	using RocketForgeTraining::SubmittalExportInternal::ReadSubmittalBotMetadata;

	const fs::path directory = RocketForgeTests::TestTempDir("submittal-device");
	fs::create_directories(directory);
	const fs::path source = directory / "bot.toml";
	ASSERT_FALSE(ReadSubmittalBotMetadata(source, "Test").useGPU);

	for (const std::string device : {"cpu", "gpu", "auto"}) {
		SCOPED_TRACE(device);
		WriteTextFile(directory / "device.txt", device + "\n");
		ASSERT_FALSE(ReadSubmittalBotMetadata(source, "Test").useGPU);
	}
}
