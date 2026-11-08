#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace RocketForgeTraining
{
	/** The model a debug render plays, resolved from a debug selection. */
	struct DebugRenderTarget
	{
		std::filesystem::path checkpointFolder;

		/** Stage the segment was trained on; empty for imported models. */
		std::string trainedStageName;

		bool isImported = false;

		/** Non-empty when the selection can't be rendered; written for the user. */
		std::string error;
	};

	/**
	 * Resolves a debug selection the way forge_debug_settings.json encodes it:
	 * an empty buildName makes segmentId an imported model's folder name.
	 * Zero checkpointTimesteps picks the segment's newest checkpoint (autosaves count).
	 */
	DebugRenderTarget ResolveDebugRenderTarget(
		const std::string& agentName,
		const std::string& buildName,
		const std::string& segmentId,
		uint64_t checkpointTimesteps = 0);
}
