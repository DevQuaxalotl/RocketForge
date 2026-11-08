#include "VersionManagement.h"
#include "AgentModelPaths.h"
#include "Storage/BuildStorage.h"
#include "TrainingSetup/TrainingSetupSnapshot.h"
#include "TrainingSetup/SubmittalExport/SubmittalExportInternal.h"
#include "TrainingStage.h"

#include <GigaLearnInfer/AgentRegistry.h>
#include <GigaLearnCPP/TrainingStorage.h>
#include <GigaLearnCPP/SegmentVersioning.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>

using namespace GGL;
using namespace RocketForge;

static constexpr const char* RLBOT_AGENT_SETUP_FILE_NAME = "agent_setup.json";
static constexpr const char* RLBOT_MODEL_SNAPSHOT_FOLDER_NAME = "model_snapshot";

static std::optional<double> ReadCheckpointSkillRating(const std::filesystem::path& checkpointFolder)
{
    std::filesystem::path statsPath = checkpointFolder / "RUNNING_STATS.json";
    if (!std::filesystem::exists(statsPath))
        return std::nullopt;

    try
    {
        std::ifstream fIn(statsPath);
        if (!fIn.good())
            return std::nullopt;

        nlohmann::json j = nlohmann::json::parse(fIn, nullptr, false);
        if (!j.is_object() || !j.contains("skill_ratings"))
            return std::nullopt;

        const nlohmann::json& ratings = j["skill_ratings"];
        if (!ratings.is_object())
            return std::nullopt;

        GGL::SkillTrackerConfig defaultSkillConfig = {};
        double total = 0.0;
        int count = 0;

        for (auto& pair : ratings.items()) {
            if (pair.value().is_object()) {
                const double mu = pair.value().value("mu", (double)defaultSkillConfig.initialMu);
                const double sigma = pair.value().value("sigma", (double)defaultSkillConfig.initialSigma);
                const bool looksUntrained =
                    std::abs(mu - defaultSkillConfig.initialMu) < 0.001 &&
                    std::abs(sigma - defaultSkillConfig.initialSigma) < 0.001;
                if (looksUntrained)
                    continue;

                total += mu;
                count++;
            } else if (pair.value().is_number()) {
                total += pair.value().get<double>();
                count++;
            }
        }

        if (count == 0)
            return std::nullopt;

        return total / count;
    }
    catch (...)
    {
        return std::nullopt;
    }
}

static std::string MakeExportBotName(
    const RocketForge::TrainingStage* stage,
    const std::string& buildName,
    const std::string& versionId,
    uint64_t timesteps)
{
    std::string name = stage->agent->agentName;
    if (!buildName.empty())
        name += buildName;

    constexpr uint64_t MILLION = 1'000'000;
    constexpr uint64_t BILLION = 1'000'000'000;
    const uint64_t unit = timesteps < BILLION ? MILLION : BILLION;
    const char suffix = timesteps < BILLION ? 'M' : 'B';
    const uint64_t rounded = (timesteps + (unit / 2)) / unit;

    return name + " " + versionId + " " + std::to_string(rounded) + suffix;
}

static std::string MakeStableExportBotName(
    const RocketForge::TrainingStage* stage,
    const std::string& buildName,
    const std::string& versionId)
{
    std::string name = stage->agent->agentName;
    if (!buildName.empty())
        name += buildName;

    return name + " " + versionId;
}

static std::string SanitizeRLBotIdPart(const std::string& text)
{
    std::string result = {};
    bool lastWasSeparator = false;
    for (char c : text)
    {
        unsigned char uc = static_cast<unsigned char>(c);
        if (std::isalnum(uc))
        {
            result.push_back((char)std::tolower(uc));
            lastWasSeparator = false;
        }
        else if (!lastWasSeparator)
        {
            result.push_back('-');
            lastWasSeparator = true;
        }
    }

    while (!result.empty() && result.front() == '-')
        result.erase(result.begin());
    while (!result.empty() && result.back() == '-')
        result.pop_back();

    return result.empty() ? "bot" : result;
}

static std::string EscapeTomlString(const std::string& text)
{
    std::string result = {};
    for (char c : text) {
        switch (c) {
            case '\\': result += "\\\\"; break;
            case '"': result += "\\\""; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default: result.push_back(c); break;
        }
    }

    return result;
}

static std::string MakeRLBotConfigPathString(const std::filesystem::path& path, const std::filesystem::path& base)
{
    std::error_code error = {};
    std::filesystem::path relative = std::filesystem::relative(path, base, error);
    if (error)
        relative = path;

    return relative.generic_string();
}

static std::string MakeRLBotRunBatPathString(const std::filesystem::path& path, const std::filesystem::path& base)
{
    std::error_code error = {};
    std::filesystem::path relative = std::filesystem::relative(path, base, error);
    if (error || relative.empty())
        relative = path;

    relative.make_preferred();
    return relative.string();
}

static bool RLBotHivemindEnabled(const std::filesystem::path& botTomlPath)
{
    return RocketForgeTraining::SubmittalExportInternal::ReadSubmittalBotMetadata(botTomlPath, {}).hivemind;
}

static void UpdateExportedBotToml(
    const std::filesystem::path& botTomlPath,
    const RocketForge::TrainingStage* stage,
    const std::string& buildName,
    const std::string& versionId,
    uint64_t timesteps)
{
    if (!std::filesystem::exists(botTomlPath))
        return;

    std::ifstream fIn(botTomlPath);
    if (!fIn.good())
        RG_ERR_CLOSE("UpdateExportedBotToml: Failed to open " << botTomlPath);

    std::stringstream buffer;
    buffer << fIn.rdbuf();
    std::string contents = buffer.str();

    const std::string botName = MakeExportBotName(stage, buildName, versionId, timesteps);
    const std::string agentId = "RocketForge/" + SanitizeRLBotIdPart(botName);

    const bool hivemind = RLBotHivemindEnabled(botTomlPath);
    bool inSettings = false;
    bool replacedAgentId = false;
    bool replacedName = false;
    std::stringstream input(contents);
    std::ostringstream output;
    std::string line = {};
    while (std::getline(input, line))
    {
        std::string trimmed = line;
        trimmed.erase(0, trimmed.find_first_not_of(" \t"));

        const std::string section = RocketForgeTraining::SubmittalExportInternal::Trim(line);
        if (section.starts_with("[") && section.ends_with("]"))
            inSettings = section == "[settings]";

        // Normalize the setting beside the section header, preserving an explicit opt-out.
        if (inSettings && RocketForgeTraining::SubmittalExportInternal::Trim(section.substr(0, section.find('='))) == "hivemind")
            continue;

        if (trimmed.rfind("agent_id", 0) == 0)
        {
            line = "agent_id = \"" + agentId + "\"";
            replacedAgentId = true;
        }
        else if (trimmed.rfind("name", 0) == 0)
        {
            line = "name = \"" + botName + "\"";
            replacedName = true;
        }

        output << line;
        if (section == "[settings]")
            output << "\nhivemind = " << (hivemind ? "true" : "false");
        if (!input.eof())
            output << '\n';
    }

    contents = output.str();

    if (!replacedAgentId || !replacedName)
        RG_ERR_CLOSE("UpdateExportedBotToml: Expected agent_id and name entries in " << botTomlPath);

    std::ofstream fOut(botTomlPath);
    if (!fOut.good())
        RG_ERR_CLOSE("UpdateExportedBotToml: Failed to write " << botTomlPath);
    fOut << contents;
}

static std::filesystem::path ResolveRLBotDiscoveryRoot(const RocketForge::TrainingStage* stage)
{
    if (!stage || !stage->agent)
        return {};

    std::filesystem::path modelsRoot = RocketForgeTraining::ResolveAgentModelsFolder(stage->agent.get());
    if (modelsRoot.empty())
        return {};

    return modelsRoot.parent_path() / "rlbot_exports";
}

static std::filesystem::path ResolveRLBotDiscoveryConfigPath(
    const RocketForge::TrainingStage* stage,
    const std::string& buildName,
    const std::string& versionId)
{
    std::filesystem::path discoveryRoot = ResolveRLBotDiscoveryRoot(stage);
    if (discoveryRoot.empty())
        return {};

    const std::string fileStem = SanitizeRLBotIdPart(MakeStableExportBotName(stage, buildName, versionId));
    return discoveryRoot / (fileStem + ".bot.toml");
}

static void WriteRLBotDiscoveryBotToml(
    const RocketForge::TrainingStage* stage,
    const std::string& buildName,
    const std::string& versionId,
    uint64_t timesteps,
    const std::filesystem::path& exportRoot)
{
    namespace fs = std::filesystem;

    if (!stage || !stage->agent)
        return;

    fs::path discoveryConfig = ResolveRLBotDiscoveryConfigPath(stage, buildName, versionId);
    if (discoveryConfig.empty())
        return;

    const fs::path discoveryRoot = discoveryConfig.parent_path();
    const std::string botName = MakeExportBotName(stage, buildName, versionId, timesteps);
    const std::string agentId = "RocketForge/" + SanitizeRLBotIdPart(botName);

    fs::create_directories(discoveryRoot);
    const std::string rootDir = MakeRLBotConfigPathString(exportRoot, discoveryRoot);
    const std::string loadoutFile = MakeRLBotConfigPathString(exportRoot / "loadout.toml", discoveryRoot);

    std::ofstream fOut(discoveryConfig);
    if (!fOut.good())
        RG_ERR_CLOSE("WriteRLBotDiscoveryBotToml: Failed to write " << discoveryConfig);

    fOut <<
        "[settings]\n"
        "hivemind = " << (RLBotHivemindEnabled(exportRoot / "bot.toml") ? "true" : "false") << "\n"
        "agent_id = \"" << EscapeTomlString(agentId) << "\"\n"
        "name = \"" << EscapeTomlString(botName) << "\"\n"
        "loadout_file = \"" << EscapeTomlString(loadoutFile) << "\"\n"
        "root_dir = \"" << EscapeTomlString(rootDir) << "\"\n"
        "run_command = \"cmd /C run.bat\"\n"
        "\n"
        "[details]\n"
        "description = \"\"\n"
        "developer = \"\"\n"
        "language = \"C++\"\n"
        "tags = []\n";
}

static void RemoveRLBotDiscoveryBotToml(const RocketForge::TrainingStage* stage, const std::string& buildName, const std::string& versionId)
{
    namespace fs = std::filesystem;

    fs::path discoveryConfig = ResolveRLBotDiscoveryConfigPath(stage, buildName, versionId);
    if (discoveryConfig.empty())
        return;

    std::error_code error = {};
    fs::remove(discoveryConfig, error);
    if (error)
        RG_LOG("Warning: failed to remove RLBot discovery config " << discoveryConfig << ": " << error.message());

    const fs::path discoveryRoot = discoveryConfig.parent_path();
    error.clear();
    if (fs::exists(discoveryRoot, error) && fs::is_empty(discoveryRoot, error))
        fs::remove(discoveryRoot, error);
}

static std::filesystem::path ResolveRLBotTemplateRoot(const RocketForge::TrainingStage* stage)
{
    if (!stage || !stage->agent)
        return {};

    const std::filesystem::path modelsRoot = RocketForgeTraining::ResolveAgentModelsFolder(stage->agent.get());
    return modelsRoot.parent_path() / "rlbot";
}

static void CopyRLBotTemplateFiles(
    const RocketForge::TrainingStage* stage,
    const std::string& buildName,
    const std::string& versionId,
    uint64_t timesteps,
    const std::filesystem::path& exportRoot)
{
    namespace fs = std::filesystem;

    fs::path sourceRLBot = ResolveRLBotTemplateRoot(stage);
    if (!fs::exists(sourceRLBot / "RLBot.exe")) {
        throw std::runtime_error(
            "RLBot export executable is missing for " + stage->agent->agentName +
            ". Build the build-rlbot-all-agents preset (target RLBotPackageAllAgents) before exporting: " +
            (sourceRLBot / "RLBot.exe").string());
    }

    if (fs::exists(sourceRLBot / "bot.toml"))
        fs::copy_file(sourceRLBot / "bot.toml", exportRoot / "bot.toml", fs::copy_options::overwrite_existing);
    if (fs::exists(sourceRLBot / "device.txt"))
        fs::copy_file(sourceRLBot / "device.txt", exportRoot / "device.txt", fs::copy_options::overwrite_existing);
    if (fs::exists(sourceRLBot / "loadout.toml"))
        fs::copy_file(sourceRLBot / "loadout.toml", exportRoot / "loadout.toml", fs::copy_options::overwrite_existing);
    if (fs::exists(sourceRLBot / "logo.png"))
        fs::copy_file(sourceRLBot / "logo.png", exportRoot / "logo.png", fs::copy_options::overwrite_existing);
    fs::copy_file(sourceRLBot / "RLBot.exe", exportRoot / "RLBot.exe", fs::copy_options::overwrite_existing);

    UpdateExportedBotToml(exportRoot / "bot.toml", stage, buildName, versionId, timesteps);
}

// The segment's saved agent setup; segments saved without one keep `current`'s network but train on their recorded engine.
static RocketForge::AgentSetupSnapshot LoadSegmentAgentSetup(
    const RocketForgeTraining::BuildStorage& buildStorage,
    const RocketForgeTraining::TrainingSegmentMetadata& segment,
    const Agent& current)
{
    const std::filesystem::path trainingSetupPath =
        buildStorage.SegmentRoot(segment.folderName) / RocketForge::TrainingSetupSnapshot::FILE_NAME;
    const std::optional<RocketForge::TrainingSetupSnapshot> snapshot = RocketForge::LoadExistingTrainingSetupSnapshot(trainingSetupPath);

    return RocketForge::ResolveSavedAgentSetup(snapshot, RocketForge::CaptureAgentSetupSnapshot(current));
}

class StoredSegmentExportStage final : public RocketForge::TrainingStage
{
public:
    StoredSegmentExportStage(std::unique_ptr<Agent> agent, std::string storedStageName)
        : TrainingStage(std::move(agent))
    {
        stageName = std::move(storedStageName);
    }
};

static std::unique_ptr<RocketForge::TrainingStage> CreateStoredSegmentExportStage(
    const std::string& agentName,
    const std::string& buildName,
    const std::string& versionId)
{
    std::unique_ptr<Agent> agent = AgentRegistry::Create(agentName);
    if (!agent)
        throw std::runtime_error("RLBot export agent is not registered: " + agentName);

    RocketForgeTraining::BuildStorage buildStorage(RocketForgeTraining::ResolveAgentModelsFolder(agent.get()), buildName);
    std::optional<RocketForgeTraining::TrainingSegmentMetadata> segment = buildStorage.LoadSegmentById(versionId);
    if (!segment.has_value())
        throw std::runtime_error("RLBot export segment was not found: " + agentName + "/" + buildName + "/" + versionId);

    RocketForge::ApplyAgentSetupSnapshot(*agent, LoadSegmentAgentSetup(buildStorage, segment.value(), *agent));

    auto stage = RocketForge::TrainingStage::Create<StoredSegmentExportStage>(std::move(agent), segment->stageName);
    stage->learnerConfig.runtime.buildName = buildName;

    return stage;
}

static void WriteRLBotAgentSetupFile(const std::filesystem::path& exportRoot, const RocketForge::AgentSetupSnapshot& agentSetup)
{
    const std::filesystem::path setupPath = exportRoot / RLBOT_AGENT_SETUP_FILE_NAME;
    std::ofstream fOut(setupPath);
    if (!fOut.good())
        RG_ERR_CLOSE("WriteRLBotAgentSetupFile: Failed to write " << setupPath);

    fOut << RocketForge::AgentSetupSnapshotToJson(agentSetup).dump(4);
}

static bool IsRLBotInferenceFile(const std::filesystem::path& file)
{
    const std::string name = file.filename().string();
    const bool isOptimizerState = name.ends_with("_OPTIM.lt");
    const bool isCritic = name.starts_with("CRITIC");

    return !isOptimizerState && !isCritic;
}

static void RemoveRLBotModelSnapshot(const std::filesystem::path& exportRoot)
{
    std::error_code error = {};
    std::filesystem::remove_all(exportRoot / RLBOT_MODEL_SNAPSHOT_FOLDER_NAME, error);
}

// Autosaves are transient, pruned as training continues and wiped when a stage completes,
// so an export from one gets its own copy of the inference files.
static std::filesystem::path SnapshotAutosaveForRLBot(
    const std::filesystem::path& autosaveFolder,
    uint64_t timesteps,
    const std::filesystem::path& exportRoot)
{
    namespace fs = std::filesystem;

    RemoveRLBotModelSnapshot(exportRoot);

    const fs::path snapshotFolder = exportRoot / RLBOT_MODEL_SNAPSHOT_FOLDER_NAME / std::to_string(timesteps);
    std::error_code error = {};
    fs::create_directories(snapshotFolder, error);
    if (error)
        RG_ERR_CLOSE("SnapshotAutosaveForRLBot: Failed to create " << snapshotFolder << ": " << error.message());

    for (const fs::directory_entry& entry : fs::directory_iterator(autosaveFolder)) {
        if (!entry.is_regular_file() || !IsRLBotInferenceFile(entry.path()))
            continue;

        fs::copy_file(entry.path(), snapshotFolder / entry.path().filename(), fs::copy_options::overwrite_existing, error);
        if (error)
            RG_ERR_CLOSE("SnapshotAutosaveForRLBot: Failed to copy " << entry.path() << ": " << error.message());
    }

    return snapshotFolder;
}

static void WriteRLBotRunBat(
    const RocketForge::TrainingStage* stage,
    const std::filesystem::path& exportRoot,
    const std::filesystem::path& modelRoot)
{
    const char* hivemind = RLBotHivemindEnabled(exportRoot / "bot.toml") ? "on" : "off";
    if (!std::filesystem::exists(exportRoot / "device.txt"))
        RocketForgeTraining::SubmittalExportInternal::WriteTextFile(exportRoot / "device.txt", "gpu\n");

    std::ofstream runBat(exportRoot / "run.bat");
    const std::string modelRootString = MakeRLBotRunBatPathString(modelRoot, exportRoot);
    runBat <<
        "@echo off\n"
        "setlocal\n"
        "set \"RLBOT_EXE=%~dp0RLBot.exe\"\n"
        "set \"MODEL_ROOT=" << modelRootString << "\"\n"
        "set \"TORCH_LIB=%LOCALAPPDATA%\\libtorch\\lib\"\n"
        "set \"PATH=%~dp0;%TORCH_LIB%;%PATH%\"\n"
        "cd /d \"%~dp0\"\n"
        "set \"RLBOT_DEVICE=gpu\"\n"
        "if exist \"%~dp0device.txt\" set /p \"RLBOT_DEVICE=\" < \"%~dp0device.txt\"\n";

    runBat <<
        "set \"AGENT_SETUP=%~dp0" << RLBOT_AGENT_SETUP_FILE_NAME << "\"\n"
        "\"%RLBOT_EXE%\" \"" << stage->agent->agentName
        << "\" --model-dir \"%MODEL_ROOT%\" --agent-setup \"%AGENT_SETUP%\" --hivemind " << hivemind << " --device \"%RLBOT_DEVICE%\" %*\n";
}

static void UpdateRLBotRunBatModelRoot(const std::filesystem::path& runBatPath, const std::filesystem::path& modelRoot)
{
    if (!std::filesystem::exists(runBatPath))
        return;

    std::ifstream fIn(runBatPath);
    if (!fIn.good())
        RG_ERR_CLOSE("UpdateRLBotRunBatModelRoot: Failed to open " << runBatPath);

    std::stringstream buffer;
    buffer << fIn.rdbuf();
    std::string contents = buffer.str();

    bool replacedModelRoot = false;
    std::stringstream input(contents);
    std::ostringstream output;
    std::string line = {};
    while (std::getline(input, line))
    {
        std::string trimmed = line;
        trimmed.erase(0, trimmed.find_first_not_of(" \t"));

        if (trimmed.rfind("set \"MODEL_ROOT=", 0) == 0)
        {
            line = "set \"MODEL_ROOT=" + MakeRLBotRunBatPathString(modelRoot, runBatPath.parent_path()) + "\"";
            replacedModelRoot = true;
        }

        output << line;
        if (!input.eof())
            output << '\n';
    }

    if (!replacedModelRoot) {
        RG_LOG("Warning: expected MODEL_ROOT entry in RLBot run script " << runBatPath);
        return;
    }

    std::ofstream fOut(runBatPath);
    if (!fOut.good())
        RG_ERR_CLOSE("UpdateRLBotRunBatModelRoot: Failed to write " << runBatPath);
    fOut << output.str();
}

static std::vector<RocketForgeTraining::BuildSummary> BuildListForAgent(const std::string& agentName)
{
    std::filesystem::path agentModelsFolder = RocketForgeTraining::ResolveAgentModelsFolder(agentName);
    RocketForgeTraining::BuildStorage registryProbe(agentModelsFolder, GGL::DEFAULT_BUILD_NAME);
    std::vector<RocketForgeTraining::BuildSummary> builds = registryProbe.ListBuilds();
    std::set<std::string> knownFolders = {};

    builds.erase(
        std::remove_if(builds.begin(), builds.end(), [&](const RocketForgeTraining::BuildSummary& build) {
            const std::string buildName = build.buildName.empty() ? build.folderName : build.buildName;
            RocketForgeTraining::BuildStorage storage(agentModelsFolder, buildName);
            if (!std::filesystem::exists(storage.BuildMetadataPath()))
                return true;

            const std::string folderName = build.folderName.empty()
                ? RocketForgeTraining::BuildStorage::SanitizeFolderPart(buildName)
                : build.folderName;
            knownFolders.insert(folderName);
            return false;
        }),
        builds.end());

    if (!std::filesystem::exists(agentModelsFolder) || !std::filesystem::is_directory(agentModelsFolder))
        return builds;

    std::vector<RocketForgeTraining::BuildSummary> discovered = {};
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(agentModelsFolder)) {
        if (!entry.is_directory())
            continue;

        const std::string folderName = entry.path().filename().string();
        if (folderName.empty() || knownFolders.count(folderName) > 0)
            continue;

        RocketForgeTraining::BuildStorage storage(agentModelsFolder, folderName);
        if (!std::filesystem::exists(storage.BuildMetadataPath()))
            continue;

        RocketForgeTraining::BuildMetadata metadata = storage.LoadBuild();
        RocketForgeTraining::BuildSummary summary = {};
        summary.buildName = metadata.buildName.empty() ? folderName : metadata.buildName;
        summary.folderName = metadata.folderName.empty() ? folderName : metadata.folderName;
        summary.createdAt = metadata.createdAt;
        summary.updatedAt = metadata.updatedAt;
        summary.active = false;
        discovered.push_back(summary);
        knownFolders.insert(folderName);
    }

    std::sort(discovered.begin(), discovered.end(), [](const RocketForgeTraining::BuildSummary& a, const RocketForgeTraining::BuildSummary& b) {
        return a.buildName < b.buildName;
    });
    builds.insert(builds.end(), discovered.begin(), discovered.end());

    return builds;
}

static std::vector<RocketForgeTraining::BuildSummary> FilterBuilds(const std::string& agentName, const std::string& buildName)
{
    if (buildName.empty())
        return BuildListForAgent(agentName);

    RocketForgeTraining::BuildStorage storage(RocketForgeTraining::ResolveAgentModelsFolder(agentName), buildName);
    if (!std::filesystem::exists(storage.BuildMetadataPath()))
        return {};

    RocketForgeTraining::BuildSummary summary = {};
    summary.buildName = buildName;
    summary.folderName = RocketForgeTraining::BuildStorage::SanitizeFolderPart(buildName);

    const RocketForgeTraining::BuildMetadata metadata = storage.LoadBuild();
    summary.buildName = metadata.buildName.empty() ? buildName : metadata.buildName;
    summary.folderName = metadata.folderName.empty() ? summary.folderName : metadata.folderName;
    summary.createdAt = metadata.createdAt;
    summary.updatedAt = metadata.updatedAt;
    summary.active = true;

    return { summary };
}

static GGL::SegmentVersionSummary MakeSegmentSummary(
    const RocketForgeTraining::BuildStorage& buildStorage,
    const RocketForgeTraining::TrainingSegmentMetadata& segment)
{
    GGL::SegmentVersionSummary summary = {};
    summary.id = segment.segmentId;
    summary.type = segment.status;

    GGL::SegmentVersionNumber parsed = {};
    if (GGL::TryParseSegmentVersionId(segment.segmentId, parsed)) {
        summary.major = parsed.major;
        summary.minor = parsed.minor;
    }

    summary.notes = segment.notes;
    summary.createdAt = segment.createdAt;
    summary.updatedAt = segment.updatedAt;
    summary.lastTrainedAt = segment.lastTrainedAt;
    summary.runName = segment.wandbRunName;
    summary.rootPath = buildStorage.SegmentRoot(segment.folderName).string();
    summary.buildName = segment.buildName;
    summary.sourceVersionId = segment.parentSegmentId;
    summary.branchName = segment.branchName.empty() ? "main" : segment.branchName;
    summary.stageName = segment.stageName;
    summary.available = segment.status != "inactive";
    summary.eligibleBenchmark = segment.eligibleBenchmark;

    const std::filesystem::path rlbotRoot = buildStorage.SegmentRoot(segment.folderName) / "rlbot";
    summary.eligibleRLBot = segment.eligibleRLBot
        && std::filesystem::exists(rlbotRoot / "bot.toml")
        && std::filesystem::exists(rlbotRoot / "run.bat")
        && std::filesystem::exists(rlbotRoot / "RLBot.exe");
    summary.pinned = segment.pinned;
    summary.totalTrainingSeconds = segment.totalTrainingSeconds;

    const std::filesystem::path checkpointRoot = buildStorage.SegmentCheckpointsRoot(segment.folderName);
    std::set<int64_t> modelFolders = GGL::TrainingStorage::FindNumberedDirs(checkpointRoot);
    summary.modelCount = modelFolders.size();

    const std::optional<RocketForgeTraining::ResolvedCheckpoint> latest = buildStorage.LatestSegmentCheckpoint(segment.folderName);
    summary.latestTimesteps = latest.has_value() ? latest->timesteps : segment.endTimestep;
    summary.latestIsAutosave = latest.has_value() && latest->isAutosave;

    summary.hasSkillRating = false;
    summary.skillRating = 0.0;

    for (int64_t timesteps : modelFolders) {
        std::optional<double> rating = ReadCheckpointSkillRating(checkpointRoot / std::to_string(timesteps));
        if (!rating.has_value())
            continue;

        if (!summary.hasSkillRating || rating.value() > summary.skillRating) {
            summary.skillRating = rating.value();
            summary.hasSkillRating = true;
        }
    }

    return summary;
}

struct SegmentLookupResult
{
    std::string buildName;
    RocketForgeTraining::TrainingSegmentMetadata segment;
};

static std::optional<SegmentLookupResult> FindSegmentForStage(const RocketForge::TrainingStage* stage, const std::string& segmentId)
{
    if (!stage || segmentId.empty())
        return std::nullopt;

    const std::string agentName = stage->agent ? stage->agent->agentName : std::string();
    const std::string buildFilter = stage->learnerConfig.runtime.buildName;

    for (const RocketForgeTraining::BuildSummary& build : FilterBuilds(agentName, buildFilter)) {
        RocketForgeTraining::BuildStorage buildStorage(RocketForgeTraining::ResolveAgentModelsFolder(agentName), build.buildName);
        std::optional<RocketForgeTraining::TrainingSegmentMetadata> segment = buildStorage.LoadSegmentById(segmentId);
        if (!segment.has_value())
            continue;
        if (segment->stageName != stage->GetStageName())
            continue;

        return SegmentLookupResult{ build.buildName, segment.value() };
    }

    return std::nullopt;
}

static void SaveSegmentForStage(const RocketForge::TrainingStage* stage, const SegmentLookupResult& lookup)
{
    RocketForgeTraining::BuildStorage buildStorage(RocketForgeTraining::ResolveAgentModelsFolder(stage->agent.get()), lookup.buildName);
    buildStorage.SaveSegment(lookup.segment);
}

static void SetSegmentRLBotExport(
    const RocketForge::TrainingStage* stage,
    const SegmentLookupResult& lookup,
    bool exported,
    std::optional<uint64_t> latestTimesteps = std::nullopt)
{
    namespace fs = std::filesystem;

    RocketForgeTraining::BuildStorage buildStorage(RocketForgeTraining::ResolveAgentModelsFolder(stage->agent.get()), lookup.buildName);
    const fs::path exportsPath = buildStorage.RLBotExportsPath();
    nlohmann::json j = {
        { "schema_version", 1 },
        { "exports", nlohmann::json::array() }
    };

    if (fs::exists(exportsPath)) {
        std::ifstream fIn(exportsPath);
        if (fIn.good()) {
            nlohmann::json parsed = nlohmann::json::parse(fIn, nullptr, false);
            if (parsed.is_object())
                j = parsed;
        }
    }

    if (!j.contains("exports") || !j["exports"].is_array())
        j["exports"] = nlohmann::json::array();

    std::string previousExportedAt = {};
    for (auto it = j["exports"].begin(); it != j["exports"].end();) {
        if (it->value("segment_id", std::string()) == lookup.segment.segmentId) {
            previousExportedAt = it->value("exported_at", std::string());
            it = j["exports"].erase(it);
        } else {
            ++it;
        }
    }

    if (exported) {
        const fs::path exportRoot = buildStorage.SegmentRoot(lookup.segment.folderName) / "rlbot";
        j["exports"].push_back({
            { "agent", stage->agent ? stage->agent->agentName : std::string() },
            { "build_name", lookup.buildName },
            { "segment_id", lookup.segment.segmentId },
            { "stage_name", lookup.segment.stageName },
            { "segment_revision", lookup.segment.segmentId },
            { "latest_timesteps", latestTimesteps.value_or(lookup.segment.endTimestep) },
            { "segment_path", buildStorage.MakeRelativeToBuild(buildStorage.SegmentRoot(lookup.segment.folderName)) },
            { "rlbot_path", buildStorage.MakeRelativeToBuild(exportRoot) },
            { "exported_at", previousExportedAt.empty() ? RocketForgeTraining::BuildStorage::MakeTimestamp() : previousExportedAt }
        });
    }

    fs::create_directories(exportsPath.parent_path());
    std::ofstream fOut(exportsPath);
    if (!fOut.good())
        RG_ERR_CLOSE("Failed to write RLBot exports registry: " << exportsPath);
    fOut << j.dump(4);
}

std::vector<RocketForgeTraining::BuildSummary> RocketForgeTraining::ListAgentBuilds(const std::string& agentName)
{
    return BuildListForAgent(agentName);
}

// Reuse the segments returned by registry reconciliation
// rather than opening every historical segment again for each public listing helper.
static std::map<std::string, std::vector<RocketForgeTraining::TrainingSegmentMetadata>>
LoadAgentSegments(const std::string& agentName, const std::string& buildName)
{
    using namespace RocketForgeTraining;
    const std::filesystem::path root = ResolveAgentModelsFolder(agentName);
    if (!buildName.empty())
        return { { buildName, BuildStorage(root, buildName).ListSegments() } };

    std::map<std::string, std::vector<TrainingSegmentMetadata>> segments;
    BuildStorage(root, GGL::DEFAULT_BUILD_NAME).ListBuilds(&segments);

    return segments;
}

std::vector<std::string> RocketForgeTraining::ListBuildTrainingStages(const std::string& agentName, const std::string& buildName)
{
    std::set<std::string> stages = {};
    for (const auto& [name, segments] : LoadAgentSegments(agentName, buildName)) {
        for (const TrainingSegmentMetadata& segment : segments) {
            if (segment.status != "inactive" && !segment.stageName.empty())
                stages.insert(segment.stageName);
        }
    }

    return { stages.begin(), stages.end() };
}

std::vector<std::string> RocketForgeTraining::ListBuildBranches(
    const std::string& agentName,
    const std::string& buildName,
    bool includeInactive)
{
    std::set<std::string> branches = {};
    for (const auto& [name, segments] : LoadAgentSegments(agentName, buildName)) {
        for (const TrainingSegmentMetadata& segment : segments) {
            if (!includeInactive && segment.status == "inactive")
                continue;

            branches.insert(segment.branchName.empty() ? "main" : segment.branchName);
        }
    }

    if (branches.empty())
        branches.insert("main");

    return { branches.begin(), branches.end() };
}

std::optional<GGL::SegmentVersionSummary> RocketForgeTraining::LatestBuildVersion(
    const std::string& agentName,
    const std::string& buildName)
{
    if (agentName.empty() || buildName.empty())
        return std::nullopt;

    BuildStorage buildStorage(ResolveAgentModelsFolder(agentName), buildName);
    BuildMetadata build = buildStorage.LoadBuild();
    if (build.activeSegmentId.empty())
        return std::nullopt;

    std::optional<TrainingSegmentMetadata> segment = buildStorage.LoadSegmentById(build.activeSegmentId);
    if (!segment.has_value())
        return std::nullopt;

    return MakeSegmentSummary(buildStorage, segment.value());
}

std::vector<GGL::SegmentVersionSummary> RocketForgeTraining::ListBuildVersions(
    const std::string& agentName,
    const std::string& buildName,
    const std::string& stageFilter,
    bool includeInactive)
{
    std::vector<GGL::SegmentVersionSummary> versions = {};

    for (const auto& [name, segments] : LoadAgentSegments(agentName, buildName)) {
        BuildStorage buildStorage(ResolveAgentModelsFolder(agentName), name);
        for (const TrainingSegmentMetadata& segment : segments) {
            if (!includeInactive && segment.status == "inactive")
                continue;
            if (!stageFilter.empty() && segment.stageName != stageFilter)
                continue;

            versions.push_back(MakeSegmentSummary(buildStorage, segment));
        }
    }

    return versions;
}

std::vector<GGL::SegmentVersionSummary> RocketForgeTraining::ListVersions(const TrainingStage* stage, bool includeInactive)
{
    if (!stage)
        return {};

    const std::string agentName = stage->agent ? stage->agent->agentName : std::string();
    const std::string allBuilds = {};
    return ListBuildVersions(agentName, allBuilds, stage->GetStageName(), includeInactive);
}

void RocketForgeTraining::SetVersionNotes(const TrainingStage* stage, const std::string& versionId, const std::string& notes)
{
    if (!stage)
        return;

    std::optional<SegmentLookupResult> lookup = FindSegmentForStage(stage, versionId);
    if (!lookup.has_value())
        RG_ERR_CLOSE("SetVersionNotes: Segment not found: " << versionId);

    lookup->segment.notes = notes;
    lookup->segment.updatedAt = BuildStorage::MakeTimestamp();
    SaveSegmentForStage(stage, lookup.value());
}

void RocketForgeTraining::SetVersionEligibility(
    const TrainingStage* stage,
    const std::string& versionId,
    std::optional<bool> benchmark,
    std::optional<bool> rlbot,
    std::optional<bool> pinned)
{
    if (!stage)
        return;

    std::optional<SegmentLookupResult> lookup = FindSegmentForStage(stage, versionId);
    if (!lookup.has_value())
        RG_ERR_CLOSE("SetVersionEligibility: Segment not found: " << versionId);

    if (benchmark.has_value())
        lookup->segment.eligibleBenchmark = benchmark.value();
    if (rlbot.has_value())
        lookup->segment.eligibleRLBot = rlbot.value();
    if (pinned.has_value())
        lookup->segment.pinned = pinned.value();

    lookup->segment.updatedAt = BuildStorage::MakeTimestamp();
    SaveSegmentForStage(stage, lookup.value());
}

void RocketForgeTraining::SetVersionActive(const TrainingStage* stage, const std::string& versionId, bool active)
{
    if (!stage)
        return;

    std::optional<SegmentLookupResult> lookup = FindSegmentForStage(stage, versionId);
    if (!lookup.has_value())
        RG_ERR_CLOSE("SetVersionActive: Segment not found: " << versionId);

    lookup->segment.status = active ? "active" : "inactive";
    lookup->segment.updatedAt = BuildStorage::MakeTimestamp();
    SaveSegmentForStage(stage, lookup.value());
}

void RocketForgeTraining::SetVersionBranch(const TrainingStage* stage, const std::string& versionId, const std::string& branchName)
{
    if (!stage)
        return;

    std::optional<SegmentLookupResult> lookup = FindSegmentForStage(stage, versionId);
    if (!lookup.has_value())
        RG_ERR_CLOSE("SetVersionBranch: Segment not found: " << versionId);

    lookup->segment.branchName = branchName.empty() ? "main" : branchName;
    lookup->segment.updatedAt = BuildStorage::MakeTimestamp();
    SaveSegmentForStage(stage, lookup.value());
}

void RocketForgeTraining::ExportVersionToRLBot(const TrainingStage* stage, const std::string& versionId)
{
    namespace fs = std::filesystem;

    if (!stage)
        return;

    std::optional<SegmentLookupResult> lookup = FindSegmentForStage(stage, versionId);
    if (!lookup.has_value())
        RG_ERR_CLOSE("ExportVersionToRLBot: Segment not found: " << versionId);

    BuildStorage buildStorage(ResolveAgentModelsFolder(stage->agent.get()), lookup->buildName);

    const std::optional<RocketForgeTraining::ResolvedCheckpoint> latest = buildStorage.LatestSegmentCheckpoint(lookup->segment.folderName);
    if (!latest.has_value())
        RG_ERR_CLOSE("ExportVersionToRLBot: No checkpoints found for " << versionId);

    const uint64_t latestTimesteps = latest->timesteps;
    const fs::path exportRoot = buildStorage.SegmentRoot(lookup->segment.folderName) / "rlbot";
    fs::create_directories(exportRoot);

    fs::path latestCheckpoint = latest->folder;
    if (latest->isAutosave)
        latestCheckpoint = SnapshotAutosaveForRLBot(latest->folder, latestTimesteps, exportRoot);
    else
        RemoveRLBotModelSnapshot(exportRoot);

    const RocketForge::AgentSetupSnapshot agentSetup = LoadSegmentAgentSetup(buildStorage, lookup->segment, *stage->agent);

    CopyRLBotTemplateFiles(stage, lookup->buildName, versionId, latestTimesteps, exportRoot);
    WriteRLBotAgentSetupFile(exportRoot, agentSetup);
    WriteRLBotRunBat(stage, exportRoot, latestCheckpoint);
    WriteRLBotDiscoveryBotToml(stage, lookup->buildName, versionId, latestTimesteps, exportRoot);

    buildStorage.RefreshSegmentCheckpoints(lookup->segment);
    lookup->segment.eligibleRLBot = true;
    lookup->segment.updatedAt = BuildStorage::MakeTimestamp();
    buildStorage.SaveSegment(lookup->segment);

    const bool exported = true;
    SetSegmentRLBotExport(stage, lookup.value(), exported, latestTimesteps);
}

void RocketForgeTraining::ExportVersionToRLBot(const std::string& agentName, const std::string& buildName, const std::string& versionId)
{
    std::unique_ptr<TrainingStage> stage = CreateStoredSegmentExportStage(agentName, buildName, versionId);
    ExportVersionToRLBot(stage.get(), versionId);
}

void RocketForgeTraining::RefreshRLBotExportAfterCheckpoint(
    const TrainingStage* stage,
    const std::string& versionId,
    uint64_t latestTimesteps,
    const std::filesystem::path& latestCheckpoint,
    const std::string& segmentFolderName)
{
    namespace fs = std::filesystem;

    if (!stage || latestTimesteps == 0 || latestCheckpoint.empty())
        return;

    std::optional<SegmentLookupResult> lookup;
    if (!segmentFolderName.empty() && !stage->learnerConfig.runtime.buildName.empty()) {
        BuildStorage storage(ResolveAgentModelsFolder(stage->agent.get()), stage->learnerConfig.runtime.buildName);
        const std::optional<TrainingSegmentMetadata> segment = storage.LoadSegmentByFolderName(segmentFolderName);
        if (segment && segment->segmentId == versionId)
            lookup = SegmentLookupResult{ stage->learnerConfig.runtime.buildName, *segment };
    }

    if (!lookup)
        lookup = FindSegmentForStage(stage, versionId);
    if (!lookup.has_value())
        return;
    if (!lookup->segment.eligibleRLBot)
        return;

    BuildStorage buildStorage(ResolveAgentModelsFolder(stage->agent.get()), lookup->buildName);
    const fs::path exportRoot = buildStorage.SegmentRoot(lookup->segment.folderName) / "rlbot";
    if (!fs::exists(exportRoot))
        return;

    UpdateExportedBotToml(exportRoot / "bot.toml", stage, lookup->buildName, versionId, latestTimesteps);
    UpdateRLBotRunBatModelRoot(exportRoot / "run.bat", latestCheckpoint);
    RemoveRLBotModelSnapshot(exportRoot);
    WriteRLBotDiscoveryBotToml(stage, lookup->buildName, versionId, latestTimesteps, exportRoot);

    const bool exported = true;
    SetSegmentRLBotExport(stage, lookup.value(), exported, latestTimesteps);
}

void RocketForgeTraining::RemoveVersionFromRLBot(const TrainingStage* stage, const std::string& versionId)
{
    namespace fs = std::filesystem;

    if (!stage)
        return;

    std::optional<SegmentLookupResult> lookup = FindSegmentForStage(stage, versionId);
    if (!lookup.has_value())
        RG_ERR_CLOSE("RemoveVersionFromRLBot: Segment not found: " << versionId);

    BuildStorage buildStorage(ResolveAgentModelsFolder(stage->agent.get()), lookup->buildName);
    fs::path exportRoot = buildStorage.SegmentRoot(lookup->segment.folderName) / "rlbot";
    if (fs::exists(exportRoot))
        fs::remove_all(exportRoot);
    RemoveRLBotDiscoveryBotToml(stage, lookup->buildName, versionId);

    lookup->segment.eligibleRLBot = false;
    lookup->segment.updatedAt = BuildStorage::MakeTimestamp();
    SaveSegmentForStage(stage, lookup.value());

    const bool exported = false;
    SetSegmentRLBotExport(stage, lookup.value(), exported);
}

void RocketForgeTraining::RemoveVersionFromRLBot(const std::string& agentName, const std::string& buildName, const std::string& versionId)
{
    std::unique_ptr<TrainingStage> stage = CreateStoredSegmentExportStage(agentName, buildName, versionId);
    RemoveVersionFromRLBot(stage.get(), versionId);
}

void RocketForgeTraining::DeleteVersion(const TrainingStage* stage, const std::string& versionId)
{
    namespace fs = std::filesystem;

    if (!stage || versionId.empty())
        return;

    std::optional<SegmentLookupResult> lookup = FindSegmentForStage(stage, versionId);
    if (!lookup.has_value())
        RG_ERR_CLOSE("DeleteVersion: Segment not found: " << versionId);

    BuildStorage buildStorage(ResolveAgentModelsFolder(stage->agent.get()), lookup->buildName);
    fs::path segmentRoot = buildStorage.SegmentRoot(lookup->segment.folderName);
    RemoveRLBotDiscoveryBotToml(stage, lookup->buildName, versionId);
    if (fs::exists(segmentRoot))
        fs::remove_all(segmentRoot);

    const bool exported = false;
    SetSegmentRLBotExport(stage, lookup.value(), exported);
}

void RocketForgeTraining::PruneVersionModels(const TrainingStage* stage, const std::string& versionId)
{
    namespace fs = std::filesystem;

    if (!stage || versionId.empty())
        return;

    std::optional<SegmentLookupResult> lookup = FindSegmentForStage(stage, versionId);
    if (!lookup.has_value())
        RG_ERR_CLOSE("PruneVersionModels: Segment not found: " << versionId);

    BuildStorage buildStorage(ResolveAgentModelsFolder(stage->agent.get()), lookup->buildName);
    fs::path checkpointRoot = buildStorage.SegmentCheckpointsRoot(lookup->segment.folderName);
    const size_t checkpointsToKeep = 1;
    std::vector<fs::path> toPrune = GGL::TrainingStorage::GetCheckpointFoldersToPrune(checkpointRoot, checkpointsToKeep);
    for (const fs::path& path : toPrune) {
        if (fs::exists(path))
            fs::remove_all(path);
    }

    buildStorage.RefreshSegmentCheckpoints(lookup->segment);
    buildStorage.SaveSegment(lookup->segment);
}
