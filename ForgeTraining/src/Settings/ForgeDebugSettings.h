#pragma once

#include <DebugSettings.h>            // ForgeDebug::DebugSettings
#include <GigaLearnInfer/JsonArchive.h>

#include <filesystem>
#include <string>
#include <vector>

namespace RocketForgeTraining {

    /** The schema version this binary writes/expects for forge_debug_settings.json. */
    constexpr int kForgeDebugSettingsSchemaVersion = 1;

    /**
     * Per-agent debug/render setup, independent of the training queue.
     * Stored at <agentRoot>/forge_debug_settings.json (one file per agent).
     * Selection (build/segment) is sticky so the Debug tab restores it on launch.
     */
    struct ForgeDebugSettings {
        int         schemaVersion = kForgeDebugSettingsSchemaVersion;
        std::string buildName;                 // sticky selection
        std::string segmentId;                 // sticky selection
        std::string renderStageName;           // GUI Debug tab stage override; empty = segment's trained stage
        bool        showRewards = false;
        ForgeDebug::DebugSettings debug;       // viz + game-control options

        /** Serializes all fields to/from JSON via the shared JsonArchive. */
        void Serialize(GGL::JsonArchive& ar);
    };

    /** Result of loading the debug settings file: ok with 'value' populated, or 'errors'. */
    using ForgeDebugSettingsLoadResult = GGL::JsonFileLoadResult<ForgeDebugSettings>;

    /** Loads from an explicit path. Missing file => ok=false with a descriptive error.
        Path-parameterized so a future presets feature can point at alternate files. */
    ForgeDebugSettingsLoadResult LoadForgeDebugSettings(const std::filesystem::path& path);

    /** <agentRoot>/forge_debug_settings.json — mirrors GetDefaultForgeSettingsPath. */
    std::filesystem::path GetDefaultForgeDebugSettingsPath(const std::string& agentName);
}
