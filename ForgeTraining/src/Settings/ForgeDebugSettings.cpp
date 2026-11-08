#include "ForgeDebugSettings.h"
#include "AgentModelPaths.h"

#include <filesystem>

namespace fs = std::filesystem;

// TODO: Unknown field preservation. Unknown JSON fields are currently dropped on load
// and not round-tripped on save. Acceptable while there is one schema version, but must
// be added before kForgeDebugSettingsSchemaVersion evolves to v2. (Mirrors the same note
// in ForgeSettingsJson.cpp.)

namespace RocketForgeTraining {

	void ForgeDebugSettings::Serialize(GGL::JsonArchive& ar) {
		ar.Field("schemaVersion", schemaVersion);
		ar.Field("buildName", buildName);
		ar.Field("segmentId", segmentId);
		ar.Field("renderStageName", renderStageName);
		ar.Field("showRewards", showRewards);
		ar.Field("debug", debug);
	}

	ForgeDebugSettingsLoadResult LoadForgeDebugSettings(const fs::path& path) {
		return GGL::JsonLoadFromFileWithSchema<ForgeDebugSettings>(path, "schemaVersion", kForgeDebugSettingsSchemaVersion);
	}

	fs::path GetDefaultForgeDebugSettingsPath(const std::string& agentName) {
		fs::path modelsRoot = ResolveAgentModelsFolder(agentName);
		// modelsRoot is .../Bots/Agents/<agent>/Models — the settings file lives one level above.
		return modelsRoot.parent_path() / "forge_debug_settings.json";
	}
}

// ForgeDebug::DebugSettings::Serialize is defined here rather than in ForgeDebugger (its own
// module) because ForgeDebugger does not link GigaLearnInfer and so cannot see JsonArchive.h.
// This file already includes it via ForgeDebugSettings.h and
// ForgeDebugSettings directly embeds a ForgeDebug::DebugSettings member, making this the
// topically natural home for the definition.
namespace ForgeDebug {

	void DebugSettings::Serialize(GGL::JsonArchive& ar) {
		ar.EnumIntField("normMode", normMode);
		ar.EnumIntField("displayMode", displayMode);
		ar.Field("playerIndex", playerIndex);
		ar.Field("forceDebugAll", forceDebugAll);
		ar.Field("gameSpeed", gameSpeed);
		ar.Field("teamSizeOverrideIndex", teamSizeOverrideIndex);
		ar.Field("forceSingleStateSetter", forceSingleStateSetter);
		ar.Field("forceStateSetterIndex", forceStateSetterIndex);
		ar.Field("maxEpisodeTimeEnabled", maxEpisodeTimeEnabled);
		ar.Field("maxEpisodeTime", maxEpisodeTime);
		ar.Field("includeStaticOpponent", includeStaticOpponent);
		ar.Field("staticOpponentSelectionKey", staticOpponentSelectionKey);
		ar.Field("debugEnabledRewards", debugEnabledRewards);
		ar.Field("hasRewardSelection", hasRewardSelection);
		ar.Field("showCriticHeads", showCriticHeads);
		ar.Field("debugEnabledCriticHeads", debugEnabledCriticHeads);
		ar.Field("hasCriticHeadSelection", hasCriticHeadSelection);
	}
}
