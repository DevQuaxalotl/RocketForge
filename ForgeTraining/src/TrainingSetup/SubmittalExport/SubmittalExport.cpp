#include "SubmittalExport.h"
#include "SubmittalExportInternal.h"

#include "AgentModelPaths.h"
#include "Storage/BuildStorage.h"
#include "TrainingSetup/TrainingSetupSnapshot.h"

#include <GigaLearnCPP/TrainingStorage.h>
#include <GigaLearnInfer/AgentRegistry.h>
#include <GigaLearnInfer/AgentSetupSnapshot.h>
#include <RocketSim/Arena.h>

#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <system_error>

namespace fs = std::filesystem;

namespace RocketForgeTraining
{
	using namespace SubmittalExportInternal;

	namespace
	{
		constexpr const char* BOB_DOCKERFILE_RELATIVE_PATH = "ForgeTraining/templates/submittal/cpp.Dockerfile";
		constexpr const char* AGENT_SETUP_FILE_NAME = "agent_setup.json";

		// Applies the segment's saved setup to `agent`;
		// segments saved without one keep the agent's network but train on their recorded engine.
		void ApplySegmentAgentSetup(Agent& agent, const BuildStorage& storage, const TrainingSegmentMetadata& segment)
		{
			const fs::path setupPath = storage.SegmentRoot(segment.folderName) / RocketForge::TrainingSetupSnapshot::FILE_NAME;
			const std::optional<RocketForge::TrainingSetupSnapshot> saved = RocketForge::LoadExistingTrainingSetupSnapshot(setupPath);
			RocketForge::ApplyAgentSetupSnapshot(
				agent,
				RocketForge::ResolveSavedAgentSetup(saved, RocketForge::CaptureAgentSetupSnapshot(agent)));
		}

		fs::path FindCheckpointModel(const fs::path& checkpointRoot, const std::string& fileName)
		{
			const std::string expected = ToLower(fileName);

			for (const fs::directory_entry& entry : fs::directory_iterator(checkpointRoot)) {
				if (entry.is_regular_file() && ToLower(entry.path().filename().string()) == expected)
					return entry.path();
			}

			return {};
		}
	}

	std::string MakeSubmittalExportFolderName(const std::string& botName)
	{
		return BuildStorage::SanitizeFolderPart(botName) + "_Submittal";
	}

	std::string MakeSubmittalExportFolderName(const std::string& agentName, const std::string& buildName, const std::string& versionId)
	{
		std::string name = agentName;
		if (!buildName.empty())
			name += "_" + buildName;
		if (!versionId.empty())
			name += "_" + versionId;

		return BuildStorage::SanitizeFolderPart(name) + "_Submittal";
	}

	std::string MakeSubmittalAgentId(const std::string& botName)
	{
		return MakeGeneratedAgentId(botName);
	}

	SubmittalBotMetadata GetDefaultSubmittalBotMetadata(
		const std::string& agentName,
		const std::string& buildName,
		const std::string& versionId)
	{
		if (agentName.empty() || buildName.empty() || versionId.empty())
			throw std::runtime_error("Agent, build, and segment are required for submittal export.");

		std::unique_ptr<Agent> agent = AgentRegistry::Create(agentName);
		if (!agent)
			throw std::runtime_error("Unable to create agent '" + agentName + "'.");

		const fs::path modelsRoot = ResolveAgentModelsFolder(agent.get());
		BuildStorage storage(modelsRoot, buildName);
		const std::optional<TrainingSegmentMetadata> segment = storage.LoadSegmentById(versionId);
		if (!segment.has_value())
			throw std::runtime_error("Unable to find segment '" + versionId + "' in build '" + buildName + "'.");

		const std::set<int64_t> checkpoints = GGL::TrainingStorage::FindNumberedDirs(storage.SegmentCheckpointsRoot(segment->folderName));
		if (checkpoints.empty())
			throw std::runtime_error("The selected segment has no full checkpoints to export.");

		const uint64_t latestTimesteps = static_cast<uint64_t>(*checkpoints.rbegin());

		ApplySegmentAgentSetup(*agent, storage, segment.value());

		const std::string defaultProjectName = MakeProjectName(
			agent->agentName.empty() ? agentName : agent->agentName,
			buildName,
			versionId,
			latestTimesteps);

		return ReadSubmittalBotMetadata(modelsRoot.parent_path() / "rlbot" / "bot.toml", defaultProjectName);
	}

	static SubmittalExportResult ExportVersionForSubmittalImpl(
		const std::string& agentName,
		const std::string& buildName,
		const std::string& versionId,
		const fs::path& outputParent,
		const SubmittalBotMetadata* requestedMetadata,
		bool useGPU,
		bool overwrite)
	{
		if (agentName.empty() || buildName.empty() || versionId.empty())
			throw std::runtime_error("Agent, build, and segment are required for submittal export.");
		if (outputParent.empty())
			throw std::runtime_error("Choose an output folder for the submittal export.");

		const fs::path absoluteOutputParent = fs::absolute(outputParent).lexically_normal();

		std::unique_ptr<Agent> agent = AgentRegistry::Create(agentName);
		if (!agent)
			throw std::runtime_error("Unable to create agent '" + agentName + "'.");

		const fs::path modelsRoot = ResolveAgentModelsFolder(agent.get());
		BuildStorage storage(modelsRoot, buildName);
		const std::optional<TrainingSegmentMetadata> segment = storage.LoadSegmentById(versionId);
		if (!segment.has_value())
			throw std::runtime_error("Unable to find segment '" + versionId + "' in build '" + buildName + "'.");

		const fs::path checkpointsRoot = storage.SegmentCheckpointsRoot(segment->folderName);
		const std::set<int64_t> checkpoints = GGL::TrainingStorage::FindNumberedDirs(checkpointsRoot);
		if (checkpoints.empty())
			throw std::runtime_error("The selected segment has no full checkpoints to export.");

		const uint64_t latestTimesteps = static_cast<uint64_t>(*checkpoints.rbegin());
		const fs::path checkpointRoot = checkpointsRoot / std::to_string(latestTimesteps);

		ApplySegmentAgentSetup(*agent, storage, segment.value());

		const bool packageNeedsV3 = PackageNeedsV3(*agent);
		if (packageNeedsV3)
			RequireV3PackageTools();

		fs::create_directories(absoluteOutputParent);
		if (!fs::is_directory(absoluteOutputParent))
			throw std::runtime_error("Submittal output is not a directory: " + absoluteOutputParent.string());

		const fs::path policyPath = FindCheckpointModel(checkpointRoot, "POLICY.lt");
		const fs::path sharedHeadPath = FindCheckpointModel(checkpointRoot, "SHARED_HEAD.lt");
		if (policyPath.empty())
			throw std::runtime_error("The latest checkpoint must contain POLICY.lt for Bob export: " + checkpointRoot.string());
		if (agent->sharedHead.IsValid() && sharedHeadPath.empty())
			throw std::runtime_error(
				"The saved agent setup uses a shared head, but the latest checkpoint does not contain "
				"SHARED_HEAD.lt for Bob export: " + checkpointRoot.string());

		const std::string defaultProjectName = MakeProjectName(
			agent->agentName.empty() ? agentName : agent->agentName,
			buildName,
			versionId,
			latestTimesteps);

		const fs::path sourceRLBotRoot = modelsRoot.parent_path() / "rlbot";
		SubmittalBotMetadata metadata = requestedMetadata
			? *requestedMetadata
			: ReadSubmittalBotMetadata(sourceRLBotRoot / "bot.toml", defaultProjectName);
		metadata.useGPU = useGPU;

		metadata.name = Trim(metadata.name);
		metadata.agentId = Trim(metadata.agentId);
		if (metadata.name.empty())
			throw std::runtime_error("Bot name is required for submittal export.");
		if (metadata.agentId.empty())
			metadata.agentId = MakeGeneratedAgentId(metadata.name);

		const std::string projectName = metadata.name;
		const std::string folderName = requestedMetadata
			? MakeSubmittalExportFolderName(projectName)
			: MakeSubmittalExportFolderName(agentName, buildName, versionId);
		const fs::path packageRoot = absoluteOutputParent / folderName;
		if (fs::exists(packageRoot) && !overwrite)
			throw std::runtime_error("Submittal export already exists: " + packageRoot.string());

		const fs::path temporaryRoot = MakeScratchRoot(absoluteOutputParent, ".rfe-");
		try {
			fs::create_directories(temporaryRoot);

			const fs::path projectRoot = fs::path(PROJECT_ROOT);
			const fs::path botSourceRoot = ResolveBotSourceRoot(agentName, modelsRoot);
			const fs::path copiedBotRoot = temporaryRoot / "Bots" / botSourceRoot.filename();
			const BotSourceExport botSourceExport = ResolveBotSourceExport(
				botSourceRoot,
				agentName,
				agent->obsBuilderName,
				agent->actionParserName);

			CopyExplicitFiles(botSourceRoot, copiedBotRoot, botSourceExport.botFiles);
			WriteTextFile(temporaryRoot / "Bots" / "ExportedRuntimeRegistration.cpp", botSourceExport.registrationSource);
			WriteTextFile(temporaryRoot / "Bots" / "CMakeLists.txt", MakeBotsCMake(botSourceRoot.filename(), botSourceExport.botFiles));

			CopyMinimalRLGym(projectRoot, temporaryRoot / "ForgeTraining" / "RLGym", botSourceExport.rlgymFiles);
			if (packageNeedsV3)
				CopyRocketSimV3(projectRoot, temporaryRoot);
			CopyMinimalGigaLearn(projectRoot, temporaryRoot / "ForgeTraining" / "GigaLearn");
			CopyMinimalRLBot(projectRoot, temporaryRoot / "RocketForgeRLBot");
			const fs::path collisionMeshes = RocketSim::CollisionMeshesRelativePath();
			CopyRuntimeAssetTree(projectRoot / collisionMeshes, temporaryRoot / collisionMeshes);
			WriteTextFile(
				temporaryRoot / "cpp.Dockerfile",
				NormalizeLineEndingsToLf(ReadTextFile(projectRoot / BOB_DOCKERFILE_RELATIVE_PATH)));

			const fs::path templates = projectRoot / "ForgeTraining/templates/submittal";
			for (const fs::path& relative : {
				fs::path("scripts/build_bob.py"), fs::path("cmake/ReadDevice.cmake"),
				fs::path("cmake/PortableTorch.cmake"), fs::path("README.md"),
				fs::path("build-support/libtorch/README.md"), fs::path(".gitignore") }) {
				WriteTextFile(temporaryRoot / relative, NormalizeLineEndingsToLf(ReadTextFile(templates / relative)));
			}

			const fs::path rlbotRoot = temporaryRoot / "rlbot";
			WriteTextFile(rlbotRoot / "device.txt", metadata.useGPU ? "gpu\n" : "cpu\n");
			const bool hasLoadout = CopySelectedRLBotMetadata(sourceRLBotRoot, rlbotRoot);
			CopyFile(policyPath, rlbotRoot / "POLICY.lt");
			if (agent->sharedHead.IsValid())
				CopyFile(sharedHeadPath, rlbotRoot / "SHARED_HEAD.lt");

			RocketForge::AgentSetupSnapshot runtimeSetup = RocketForge::CaptureAgentSetupSnapshot(*agent);
			if (runtimeSetup.agentName.empty())
				runtimeSetup.agentName = agentName;

			WriteTextFile(rlbotRoot / AGENT_SETUP_FILE_NAME, MakeRuntimeAgentSetupJson(runtimeSetup).dump(4));
			WriteTextFile(rlbotRoot / "bot.toml", MakePortableBotToml(sourceRLBotRoot / "bot.toml", metadata, hasLoadout));

			WriteTextFile(temporaryRoot / "CMakeLists.txt", MakeRootCMake(agentName, projectName, packageNeedsV3));
			WriteTextFile(temporaryRoot / "bob.toml", MakeBobToml(projectName));
			WriteTextFile(
				temporaryRoot / ".dockerignore",
				".git\n.vs\nbob_build\nbuild-*\n!build-support/\n!build-support/**\nout\n**/Models\n**/*.exe\n**/*.dll\n");

			CommitPackage(temporaryRoot, packageRoot, overwrite);
		} catch (...) {
			std::error_code cleanupError;
			fs::remove_all(temporaryRoot, cleanupError);
			throw;
		}

		return SubmittalExportResult{ packageRoot, projectName, latestTimesteps };
	}

	SubmittalExportResult ExportVersionForSubmittal(
		const std::string& agentName,
		const std::string& buildName,
		const std::string& versionId,
		const fs::path& outputParent,
		bool useGPU,
		bool overwrite)
	{
		return ExportVersionForSubmittalImpl(agentName, buildName, versionId, outputParent, nullptr, useGPU, overwrite);
	}

	SubmittalExportResult ExportVersionForSubmittal(
		const std::string& agentName,
		const std::string& buildName,
		const std::string& versionId,
		const fs::path& outputParent,
		const SubmittalBotMetadata& metadata,
		bool overwrite)
	{
		return ExportVersionForSubmittalImpl(agentName, buildName, versionId, outputParent, &metadata, metadata.useGPU, overwrite);
	}
}
