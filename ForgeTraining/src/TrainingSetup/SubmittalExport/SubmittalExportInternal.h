#pragma once

#include "SubmittalExport.h"

#include <GigaLearnInfer/AgentSetupSnapshot.h>

#include <filesystem>
#include <functional>
#include <nlohmann/json.hpp>
#include <set>
#include <string>
#include <vector>

class Agent;

namespace RocketForgeTraining
{
	/** True when the package must build RocketSim v3: the agent trains (and so warms up) or predicts with it. */
	bool PackageNeedsV3(const Agent& agent);
}

namespace RocketForgeTraining::SubmittalExportInternal
{
	struct BotSourceExport
	{
		std::set<std::filesystem::path> botFiles;
		std::set<std::filesystem::path> rlgymFiles;
		std::string registrationSource;
	};

	std::string ToLower(std::string text);
	std::string Trim(std::string text);
	std::string EscapeTomlString(const std::string& text);
	std::string EscapeCMakeString(const std::string& text);

	/** Converts CRLF and standalone CR line endings to LF for Linux container inputs. */
	std::string NormalizeLineEndingsToLf(const std::string& text);

	std::string ReadTextFile(const std::filesystem::path& path);
	void WriteTextFile(const std::filesystem::path& path, const std::string& contents);
	void CopyFile(const std::filesystem::path& source, const std::filesystem::path& destination);
	bool IsPathInside(const std::filesystem::path& path, const std::filesystem::path& parent);
	void CopyRuntimeAssetTree(const std::filesystem::path& sourceRoot, const std::filesystem::path& destinationRoot);
	bool IsCppSourceFile(const std::filesystem::path& path);
	bool IsCppFile(const std::filesystem::path& path);
	void CopyExplicitFiles(
		const std::filesystem::path& sourceRoot,
		const std::filesystem::path& destinationRoot,
		const std::set<std::filesystem::path>& relativePaths);
	void CopyCodeTree(
		const std::filesystem::path& sourceRoot,
		const std::filesystem::path& destinationRoot,
		const std::set<std::filesystem::path>& excludedRelativeRoots = {});
	std::filesystem::path MakeScratchRoot(const std::filesystem::path& outputParent, const std::string& prefix);
	void CommitPackage(const std::filesystem::path& temporaryRoot, const std::filesystem::path& packageRoot, bool overwrite);

	std::filesystem::path ResolveBotSourceRoot(const std::string& agentName, const std::filesystem::path& modelsRoot);
	BotSourceExport ResolveBotSourceExport(
		const std::filesystem::path& botSourceRoot,
		const std::string& agentName,
		const std::string& obsBuilderName,
		const std::string& actionParserName);
	std::string MakeBotsCMake(const std::filesystem::path& botFolderName, const std::set<std::filesystem::path>& botFiles);

	std::string MakeProjectName(
		const std::string& agentName,
		const std::string& buildName,
		const std::string& versionId,
		uint64_t timesteps);
	std::string MakeGeneratedAgentId(const std::string& botName);
	SubmittalBotMetadata ReadSubmittalBotMetadata(const std::filesystem::path& sourcePath, const std::string& defaultProjectName);
	std::string MakePortableBotToml(const std::filesystem::path& sourcePath, const SubmittalBotMetadata& metadata, bool hasLoadout);
	bool CopySelectedRLBotMetadata(const std::filesystem::path& sourceRoot, const std::filesystem::path& destinationRoot);
	nlohmann::json MakeRuntimeAgentSetupJson(const RocketForge::AgentSetupSnapshot& snapshot);
	std::string MakeRootCMake(const std::string& agentName, const std::string& projectName, bool enableV3);
	std::string MakeBobToml(const std::string& projectName);

	void CopyMinimalRLGym(
		const std::filesystem::path& projectRoot,
		const std::filesystem::path& destinationRoot,
		const std::set<std::filesystem::path>& selectedComponentFiles);

	/** Runs cargo with `arguments` from `workingDirectory`; throws, naming `purpose` and cargo's output, when cargo fails. */
	using CargoRunner = std::function<void(
		const std::vector<std::string>& arguments,
		const std::filesystem::path& workingDirectory,
		const std::string& purpose)>;

	/** The CargoRunner exports use: the cargo this build was configured with, from `workingDirectory` so rustup applies its pin. */
	void RunConfiguredCargo(
		const std::vector<std::string>& arguments,
		const std::filesystem::path& workingDirectory,
		const std::string& purpose);

	namespace Testing
	{
		/** Test-only: routes the export's cargo calls through `runner` for the guard's lifetime. */
		class ScopedCargoRunner
		{
		public:
			explicit ScopedCargoRunner(CargoRunner runner);
			~ScopedCargoRunner();

			ScopedCargoRunner(const ScopedCargoRunner&) = delete;
			ScopedCargoRunner& operator=(const ScopedCargoRunner&) = delete;

		private:
			CargoRunner _previous;
		};
	}

	/** Drops every build-dependencies table from a Cargo manifest. */
	std::string RemoveBuildDependencies(const std::string& manifest);

	/** Throws, naming the fix, unless this build can export RocketSim v3: v3 built in and its cargo still installed. */
	void RequireV3PackageTools();

	/** Copies the v3 bindings and upstream crate into `packageRoot`, and vendors their crates.io dependencies. */
	void CopyRocketSimV3(const std::filesystem::path& projectRoot, const std::filesystem::path& packageRoot);

	void CopyMinimalGigaLearn(const std::filesystem::path& projectRoot, const std::filesystem::path& destinationRoot);
	void CopyMinimalRLBot(const std::filesystem::path& projectRoot, const std::filesystem::path& destinationRoot);
}
