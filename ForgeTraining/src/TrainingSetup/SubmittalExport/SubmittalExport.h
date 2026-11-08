#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace RocketForgeTraining
{
	/** User-facing metadata written into the exported RLBot bot.toml. */
	struct SubmittalBotMetadata
	{
		std::string name;
		std::string agentId;
		std::string description;
		std::string funFact;
		std::string sourceLink;
		std::string developer;
		std::string language = "C++";
		std::vector<std::string> tags;

		/** Group matching teammates into one RLBot inference process. */
		bool hivemind = true;

		/**
		 * Select the Windows CUDA submission build; CPU also supports Linux.
		 * Off by default, independent of the agent's local rlbot/device.txt, because the RLBot bot pack builds CPU only.
		 */
		bool useGPU = false;
	};

	/** Summary of a Bob-compatible bot source package written to disk. */
	struct SubmittalExportResult
	{
		std::filesystem::path packageRoot;
		std::string projectName;
		uint64_t latestTimesteps = 0;
	};

	/** Loads the metadata defaults shown before exporting a saved segment. */
	SubmittalBotMetadata GetDefaultSubmittalBotMetadata(
		const std::string& agentName,
		const std::string& buildName,
		const std::string& versionId);

	/** Returns a stable RLBot agent id derived from a display name. */
	std::string MakeSubmittalAgentId(const std::string& botName);

	/** Returns the export child-folder name for the chosen bot display name. */
	std::string MakeSubmittalExportFolderName(const std::string& botName);

	/** Returns the legacy generated export child-folder name. */
	std::string MakeSubmittalExportFolderName(const std::string& agentName, const std::string& buildName, const std::string& versionId);

	/**
	 * Packages one saved segment for Bob the Bot Builder.
	 * The result contains the selected model, portable runtime assets,
	 * and only the RocketForge source needed to compile the bot for RLBot.
	 * Metadata comes from the agent's rlbot/bot.toml.
	 */
	SubmittalExportResult ExportVersionForSubmittal(
		const std::string& agentName,
		const std::string& buildName,
		const std::string& versionId,
		const std::filesystem::path& outputParent,
		bool useGPU,
		bool overwrite);

	/** Packages a saved segment using explicitly chosen bot.toml metadata. */
	SubmittalExportResult ExportVersionForSubmittal(
		const std::string& agentName,
		const std::string& buildName,
		const std::string& versionId,
		const std::filesystem::path& outputParent,
		const SubmittalBotMetadata& metadata,
		bool overwrite = false);
}
