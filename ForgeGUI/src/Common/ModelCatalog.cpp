#include "ModelCatalog.h"
#include <Utils/FileRevision.h>
#include <AgentModelPaths.h>
#include <TrainingSetup/TrainingSetupSnapshot.h>
#include <GigaLearnCPP/SkillTrackerConfig.h>
#include <GigaLearnCPP/TrainingStorage.h>
#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileSystemWatcher>
#include <QSaveFile>
#include <QTimer>
#include <QElapsedTimer>
#include <QLoggingCategory>
#include <algorithm>
#include <cmath>
#include <future>
#include <set>
#include <tuple>
#include <utility>

using nlohmann::json;
namespace fs = std::filesystem;
using namespace RocketForgeTraining;

namespace GGL {
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(SegmentVersionSummary, id, type, major, minor, notes,
    createdAt, updatedAt, lastTrainedAt, runName, rootPath, buildName, sourceVersionId, branchName,
    stageName, available, pinned, eligibleBenchmark, eligibleRLBot, latestTimesteps, latestIsAutosave, modelCount,
    totalTrainingSeconds, skillRating, hasSkillRating)
}
namespace RocketForgeTraining {
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(SegmentCheckpoint, timesteps, path)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(TrainingSegmentMetadata, schemaVersion, segmentId,
    folderName, buildName, parentSegmentId, branchName, status, stageName, setupHash, createdAt,
    updatedAt, lastTrainedAt, startTimestep, endTimestep, totalTrainingSeconds, sourceCheckpointPath, skillStartRatings,
    wandbProject, wandbGroup, wandbRunName, wandbRunId, notes, eligibleBenchmark,
    eligibleRLBot, pinned, checkpoints)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(BuildSummary, buildName, folderName, createdAt, updatedAt, active)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(AgentVersionCatalog, agentName, builds, segmentsByBuild, summariesByBuild)
}
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(CatalogModel, version, sourceLabel, displayName, checkpointPath, setupPath, agentSetup)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ModelCatalogSnapshot, storage, activeSegments, models, root)

Q_LOGGING_CATEGORY(catalogLog, "rocketforge.catalog", QtWarningMsg)

namespace {
    constexpr int CACHE_VERSION = 3;

    json Read(const fs::path& path)
    {
        QFile file(QString::fromStdWString(path.wstring()));
        if (!file.open(QIODevice::ReadOnly)) throw std::runtime_error("Cannot read " + path.string());

        return json::parse(file.readAll().toStdString());
    }

    void Save(const fs::path& path, const json& value)
    {
        QDir().mkpath(QString::fromStdWString(path.parent_path().wstring()));
        QSaveFile file(QString::fromStdWString(path.wstring()));
        const std::string bytes = value.dump();

        // A derived cache is expendable. Failure never makes model storage unusable.
        if (file.open(QIODevice::WriteOnly) && file.write(bytes.data(), bytes.size()) == (qint64)bytes.size())
            file.commit();
    }

    std::string Text(const json& value, const char* key, const std::string& fallback = {})
    {
        return value.contains(key) && value[key].is_string() ? value[key].get<std::string>() : fallback;
    }

    struct ReadIndex {
        fs::path root;
        json entries = json::object();
        std::map<std::string, std::string> observed;
        std::set<std::string> touched;
        QStringList directories;
        bool changed = false;
        int jsonReads = 0, directoryReads = 0;

        explicit ReadIndex(fs::path path) : root(std::move(path)) {
            if (root.empty()) return;

            try {
                const json saved = Read(root / ".rocketforge" / "catalog-files.json");
                const bool matchesCache = saved.value("version", 0) == CACHE_VERSION && saved.value("root", std::string()) == root.string();
                if (matchesCache && saved.at("entries").is_object())
                    entries = saved.at("entries");
            } catch (...) {}
        }

        void Begin() {
            observed.clear();
            touched.clear();
            directories.clear();
            changed = false;
            jsonReads = directoryReads = 0;
        }

        std::string Stamp(const fs::path& path) {
            const std::string key = path.lexically_normal().string();
            const auto found = observed.find(key);
            if (found != observed.end()) return found->second;

            return observed.emplace(key, RocketForge::FileRevision(path)).first->second;
        }

        json Get(const fs::path& path, bool directory = false) {
            const std::string key = (directory ? "dir:" : "json:") + path.lexically_normal().string();
            touched.insert(key);
            const std::string revision = Stamp(path);
            if (!revision.empty()) directories << QString::fromStdWString((directory ? path : path.parent_path()).wstring());

            auto old = entries.find(key);
            if (old != entries.end() && old->value("revision", std::string()) == revision && old->contains("value"))
                return old->at("value");

            json value = directory ? json::array() : json::object();
            if (!revision.empty()) {
                if (directory) {
                    ++directoryReads;
                    for (const fs::directory_entry& entry : fs::directory_iterator(path))
                        if (entry.is_directory()) value.push_back(entry.path().filename().string());
                    std::sort(value.begin(), value.end());
                } else {
                    ++jsonReads;
                    value = Read(path);
                }

                if (RocketForge::FileRevision(path) != revision)
                    throw std::runtime_error("Model files changed during refresh; retrying");
            }

            entries[key] = {{"revision", revision}, {"value", value}};
            changed = true;

            return value;
        }

        std::vector<fs::path> Dirs(const fs::path& path) {
            const bool isDirectory = true;
            std::vector<fs::path> result;
            for (const json& name : Get(path, isDirectory)) {
                const std::string folder = name.get<std::string>();

                // Cache contents cannot escape their recorded root through a folder entry.
                if (fs::path(folder).filename() != fs::path(folder) || folder == "." || folder == "..")
                    throw std::runtime_error("Invalid catalog directory entry");
                result.push_back(path / folder);
            }

            return result;
        }

        std::map<uint64_t, fs::path> Checkpoints(const fs::path& path) {
            std::map<uint64_t, fs::path> result;
            for (const fs::path& dir : Dirs(path)) {
                const std::string name = dir.filename().string();
                if (!GGL::TrainingStorage::IsAllDigits(name)) continue;
                try { result.emplace(std::stoull(name), dir); } catch (...) {}
            }

            return result;
        }

        void Commit() {
            for (const auto& [path, stamp] : observed)
                if (RocketForge::FileRevision(path) != stamp)
                    throw std::runtime_error("Model files changed during refresh; retrying");

            for (auto it = entries.begin(); it != entries.end();) {
                if (!touched.contains(it.key())) {
                    it = entries.erase(it);
                    changed = true;
                } else {
                    ++it;
                }
            }

            if (changed && fs::is_directory(root)) {
                const json cache = {{"version", CACHE_VERSION}, {"root", root.string()}, {"entries", entries}};
                Save(root / ".rocketforge" / "catalog-files.json", cache);
            }
        }
    };

    std::optional<double> SkillRating(const json& stats)
    {
        if (!stats.contains("skill_ratings") || !stats["skill_ratings"].is_object()) return {};

        GGL::SkillTrackerConfig defaults;
        double total = 0;
        int count = 0;
        for (const json& rating : stats["skill_ratings"]) {
            if (rating.is_object()) {
                const double mu = rating.value("mu", (double)defaults.initialMu);
                const double sigma = rating.value("sigma", (double)defaults.initialSigma);
                if (std::abs(mu - defaults.initialMu) < 0.001 && std::abs(sigma - defaults.initialSigma) < 0.001) continue;
                total += mu;
                ++count;
            } else if (rating.is_number()) {
                total += rating.get<double>();
                ++count;
            }
        }

        return count ? std::optional<double>(total / count) : std::nullopt;
    }

    CatalogModel Summarize(ReadIndex& index, const fs::path& segmentRoot, const fs::path& buildRoot, TrainingSegmentMetadata& segment)
    {
        CatalogModel model;
        GGL::SegmentVersionSummary& value = model.version;
        value.id = segment.segmentId;
        value.type = segment.status;
        GGL::SegmentVersionNumber parsed;
        if (GGL::TryParseSegmentVersionId(value.id, parsed)) {
            value.major = parsed.major;
            value.minor = parsed.minor;
        }

        value.notes = segment.notes;
        value.createdAt = segment.createdAt;
        value.updatedAt = segment.updatedAt;
        value.lastTrainedAt = segment.lastTrainedAt;
        value.runName = segment.wandbRunName;
        value.rootPath = segmentRoot.string();
        value.buildName = segment.buildName;
        value.sourceVersionId = segment.parentSegmentId;
        value.branchName = segment.branchName.empty() ? "main" : segment.branchName;
        value.stageName = segment.stageName;
        value.available = segment.status != "inactive";
        value.pinned = segment.pinned;
        value.eligibleBenchmark = segment.eligibleBenchmark;
        value.totalTrainingSeconds = segment.totalTrainingSeconds;

        const fs::path root(value.rootPath);
        value.eligibleRLBot = segment.eligibleRLBot && !index.Stamp(root / "rlbot" / "bot.toml").empty()
            && !index.Stamp(root / "rlbot" / "run.bat").empty() && !index.Stamp(root / "rlbot" / "RLBot.exe").empty();

        const std::map<uint64_t, fs::path> checkpoints = index.Checkpoints(root / "checkpoints");
        segment.checkpoints.clear();
        for (const auto& [time, checkpoint] : checkpoints) {
            segment.checkpoints.push_back({time, checkpoint.lexically_relative(buildRoot).generic_string()});
            const std::optional<double> rating = SkillRating(index.Get(checkpoint / "RUNNING_STATS.json"));
            if (rating && (!value.hasSkillRating || *rating > value.skillRating)) {
                value.hasSkillRating = true;
                value.skillRating = *rating;
            }
        }

        value.modelCount = checkpoints.size();
        segment.endTimestep = checkpoints.empty() ? segment.endTimestep : checkpoints.rbegin()->first;
        value.latestTimesteps = segment.endTimestep;
        if (!checkpoints.empty()) model.checkpointPath = checkpoints.rbegin()->second.string();

        const std::map<uint64_t, fs::path> autosaves = index.Checkpoints(root / "temp_checkpoints");
        if (!autosaves.empty() && (checkpoints.empty() || autosaves.rbegin()->first > checkpoints.rbegin()->first)) {
            value.latestTimesteps = autosaves.rbegin()->first;
            value.latestIsAutosave = true;
            model.checkpointPath = autosaves.rbegin()->second.string();
        }

        const fs::path setup = root / "training_setup.json";
        const json saved = index.Get(setup);
        if (!index.Stamp(setup).empty()) model.setupPath = setup.string();
        if (saved.contains("agent_setup")) {
            RocketForge::TrainingSetupSnapshot snapshot;
            const std::vector<std::string> errors = GGL::JsonLoad(saved, snapshot);
            if (!errors.empty()) throw std::runtime_error(errors.front());
            if (snapshot.agentSetup) model.agentSetup = RocketForge::AgentSetupSnapshotToJson(*snapshot.agentSetup);
        }

        return model;
    }

    std::shared_ptr<ModelCatalogSnapshot> Scan(const QString& agent, ReadIndex& index)
    {
        index.Begin();
        auto result = std::make_shared<ModelCatalogSnapshot>();
        result->root = index.root.string();
        result->storage.agentName = agent.toStdString();

        const json registry = index.Get(index.root / "builds.json");
        std::map<std::string, bool> active;
        for (const json& build : registry.value("builds", json::array()))
            active[Text(build, "folder_name", Text(build, "build_name"))] = build.value("active", false);

        for (const fs::path& root : index.Dirs(index.root)) {
            const std::string folder = root.filename().string();
            if (folder == ".rocketforge" || folder == "Imported Models" || folder == "Static Models") continue;

            const json metadata = index.Get(root / "build.json");
            const std::vector<fs::path> segmentDirs = index.Dirs(root / "segments");
            if (metadata.empty() && segmentDirs.empty()) continue;

            const std::string name = Text(metadata, "build_name", folder);

            // Storage paths are derived from the physical folder. Metadata names are display identity.
            result->storage.builds.push_back({name, folder, Text(metadata, "created_at"), Text(metadata, "updated_at"), active[folder]});
            result->activeSegments[name] = Text(metadata, "active_segment_id");

            std::vector<TrainingSegmentMetadata>& segments = result->storage.segmentsByBuild[name];
            std::vector<GGL::SegmentVersionSummary>& summaries = result->storage.summariesByBuild[name];
            for (const fs::path& dir : segmentDirs) {
                const json saved = index.Get(dir / "segment.json");
                if (saved.empty()) continue;

                TrainingSegmentMetadata segment;
                const std::vector<std::string> errors = GGL::JsonLoad(saved, segment);
                if (!errors.empty()) throw std::runtime_error(errors.front());

                segment.folderName = dir.filename().string();
                segment.buildName = name;
                if (segment.segmentId.empty()) segment.segmentId = segment.folderName.substr(0, segment.folderName.find('_'));

                CatalogModel model = Summarize(index, dir, root, segment);
                summaries.push_back(model.version);
                segments.push_back(std::move(segment));
                result->models.push_back(std::move(model));
            }

            std::sort(segments.begin(), segments.end(), [](const auto& lhs, const auto& rhs) {
                const auto lhsRevision = BuildStorage::TryParseSegmentRevision(lhs.segmentId);
                const auto rhsRevision = BuildStorage::TryParseSegmentRevision(rhs.segmentId);
                if (lhsRevision && rhsRevision)
                    return std::tie(lhsRevision->major, lhsRevision->minor) < std::tie(rhsRevision->major, rhsRevision->minor);

                return lhsRevision.has_value() != rhsRevision.has_value() ? lhsRevision.has_value() : lhs.segmentId < rhs.segmentId;
            });
        }

        for (const fs::path& root : index.Dirs(index.root / "Imported Models")) {
            const std::string folder = root.filename().string();
            if (folder.empty() || folder.find("__") != std::string::npos) continue;

            const json metadata = index.Get(root / "imported_model.json");
            fs::path checkpoint;
            for (const char* file : {"POLICY.lt", "SHARED_HEAD.lt", "CRITIC.lt", "INFERENCE_POLICY.json"}) {
                if (!index.Stamp(root / file).empty()) {
                    checkpoint = root;
                    break;
                }
            }

            if (checkpoint.empty()) {
                const std::map<uint64_t, fs::path> candidates = index.Checkpoints(root);
                if (!candidates.empty()) checkpoint = candidates.rbegin()->second;
            }

            CatalogModel model;
            model.sourceLabel = "Imported Model";
            model.displayName = Text(metadata, "display_name", folder);
            model.checkpointPath = checkpoint.string();

            GGL::SegmentVersionSummary& value = model.version;
            value.id = folder;
            value.type = "imported_model";
            value.rootPath = root.string();
            value.buildName = Text(metadata, "source_build");
            value.stageName = Text(metadata, "source_stage", "Imported Models");
            value.sourceVersionId = Text(metadata, "source_version_id", "Imported");
            value.branchName = "N/A";
            value.notes = Text(metadata, "notes", "Imported model folder");
            value.createdAt = value.updatedAt = value.lastTrainedAt = Text(metadata, "imported_at");
            value.available = !checkpoint.empty();
            value.modelCount = value.available ? 1 : 0;
            value.latestTimesteps = metadata.value("latest_timesteps", uint64_t{0});
            if (!value.latestTimesteps && !checkpoint.empty())
                value.latestTimesteps = index.Get(checkpoint / "RUNNING_STATS.json").value("total_timesteps", uint64_t{0});

            result->models.push_back(std::move(model));
        }

        index.Commit();

        return result;
    }
}

std::vector<GGL::SegmentVersionSummary> ModelCatalogSnapshot::Versions(
    const std::string& build, const std::string& stage, bool includeInactive) const
{
    std::vector<GGL::SegmentVersionSummary> versions;
    for (const auto& [name, summaries] : storage.summariesByBuild) {
        if (!build.empty() && name != build) continue;

        for (const GGL::SegmentVersionSummary& version : summaries) {
            if (!includeInactive && version.type == "inactive") continue;
            if (!stage.empty() && version.stageName != stage) continue;
            versions.push_back(version);
        }
    }

    return versions;
}

std::vector<std::string> ModelCatalogSnapshot::TrainingStages(const std::string& build) const
{
    std::set<std::string> stages;
    for (const GGL::SegmentVersionSummary& version : Versions(build))
        if (!version.stageName.empty()) stages.insert(version.stageName);

    return { stages.begin(), stages.end() };
}

struct ModelCatalog::Impl {
    struct Entry {
        QString agent;
        fs::path root;
        std::shared_ptr<const ModelCatalogSnapshot> snapshot;
        std::shared_ptr<ReadIndex> index;
        std::string serialized;
        bool repair = false;
    };

    struct Result {
        QString agent;
        std::shared_ptr<ModelCatalogSnapshot> snapshot;
        std::string serialized;
        QStringList directories;
        QString error;
        int jsonReads = 0, directoryReads = 0;
        qint64 elapsedMs = 0;
    };

    std::map<QString, Entry> entries;
    std::set<QString> queued;
    std::future<Result> work;
    QString scanning;
    QTimer* poll = nullptr;
    QTimer* debounce = nullptr;
    QFileSystemWatcher* watcher = nullptr;
};

ModelCatalog& ModelCatalog::Instance()
{
    static ModelCatalog* service = new ModelCatalog(QCoreApplication::instance());
    return *service;
}

ModelCatalog::ModelCatalog(QObject* parent) : QObject(parent), m_impl(std::make_unique<Impl>())
{
    m_impl->poll = new QTimer(this);
    m_impl->poll->setInterval(25);
    connect(m_impl->poll, &QTimer::timeout, this, &ModelCatalog::Finish);

    m_impl->debounce = new QTimer(this);
    m_impl->debounce->setSingleShot(true);
    m_impl->debounce->setInterval(100);
    connect(m_impl->debounce, &QTimer::timeout, this, &ModelCatalog::StartNext);

    m_impl->watcher = new QFileSystemWatcher(this);
    connect(m_impl->watcher, &QFileSystemWatcher::directoryChanged, this, [this](const QString& path) {
        for (const auto& [agent, entry] : m_impl->entries)
            if (path.startsWith(QString::fromStdWString(entry.root.wstring()), Qt::CaseInsensitive)) Refresh(agent);
    });

    auto* fallback = new QTimer(this);
    fallback->setInterval(10000);
    connect(fallback, &QTimer::timeout, this, [this]() {
        for (const auto& [agent, entry] : m_impl->entries) Refresh(agent);
    });
    fallback->start();
}

ModelCatalog::~ModelCatalog() = default;

std::shared_ptr<const ModelCatalogSnapshot> ModelCatalog::Get(const QString& agent)
{
    const fs::path root = fs::absolute(ResolveAgentModelsFolder(agent.toStdString())).lexically_normal();
    Impl::Entry& entry = m_impl->entries[agent];
    if (entry.snapshot && entry.root == root) return entry.snapshot;

    entry = {};
    entry.agent = agent;
    entry.root = root;
    auto empty = std::make_shared<ModelCatalogSnapshot>();
    empty->root = root.string();
    empty->storage.agentName = agent.toStdString();
    entry.snapshot = empty;

    try {
        const json saved = Read(root / ".rocketforge" / "catalog.json");
        if (saved.value("version", 0) == CACHE_VERSION && saved.at("snapshot").at("root") == root.string()) {
            ModelCatalogSnapshot snapshot = saved.at("snapshot").get<ModelCatalogSnapshot>();
            snapshot.storage.agentName = agent.toStdString();
            entry.snapshot = std::make_shared<ModelCatalogSnapshot>(std::move(snapshot));
            entry.serialized = json(*entry.snapshot).dump();
        }
    } catch (...) {}

    m_impl->queued.insert(agent);
    m_impl->debounce->start();

    return entry.snapshot;
}

void ModelCatalog::Refresh(const QString& agent, bool repair)
{
    if (agent.isEmpty()) return;

    Get(agent);
    m_impl->entries.at(agent).repair |= repair;
    m_impl->queued.insert(agent);
    if (!m_impl->debounce->isActive()) m_impl->debounce->start();
}

bool ModelCatalog::IsRefreshing(const QString& agent) const
{
    return m_impl->scanning == agent || m_impl->queued.contains(agent);
}

void ModelCatalog::StartNext()
{
    if (m_impl->work.valid() || m_impl->queued.empty()) return;

    const QString agent = *m_impl->queued.begin();
    m_impl->queued.erase(agent);
    Impl::Entry& entry = m_impl->entries.at(agent);
    const fs::path root = entry.root;
    const bool repair = std::exchange(entry.repair, false);
    const std::shared_ptr<ReadIndex> existingIndex = entry.index;
    const std::string previous = entry.serialized;
    m_impl->scanning = agent;

    // Only the compact snapshot is read synchronously; the much larger file cache is loaded on the worker.
    std::shared_ptr<ReadIndex> index = existingIndex ? existingIndex : std::make_shared<ReadIndex>(fs::path());
    entry.index = index;
    m_impl->work = std::async(std::launch::async, [agent, root, index, existingIndex, repair, previous]() {
        Impl::Result result;
        result.agent = agent;
        QElapsedTimer elapsed;
        elapsed.start();

        try {
            if (!existingIndex) *index = ReadIndex(root);
            if (repair) {
                BuildStorage storage(root, GGL::DEFAULT_BUILD_NAME);
                storage.ListBuilds();
                index->entries = json::object();
            }

            // Create the derived-cache folder before recording the root directory's revision,
            // so the first cache publication does not invalidate that scan.
            std::error_code cacheError;
            if (fs::is_directory(root)) fs::create_directories(root / ".rocketforge", cacheError);

            result.snapshot = Scan(agent, *index);
            result.serialized = json(*result.snapshot).dump();
            result.directories = index->directories;
            result.directories.removeDuplicates();
            if (result.serialized != previous && fs::is_directory(root))
                Save(root / ".rocketforge" / "catalog.json", {{"version", CACHE_VERSION}, {"snapshot", json(*result.snapshot)}});
        } catch (const std::exception& error) {
            index->entries = json::object();
            result.error = QString::fromUtf8(error.what());
        }

        result.jsonReads = index->jsonReads;
        result.directoryReads = index->directoryReads;
        result.elapsedMs = elapsed.elapsed();

        return result;
    });
    m_impl->poll->start();
}

void ModelCatalog::Finish()
{
    if (!m_impl->work.valid() || m_impl->work.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;

    Impl::Result result = m_impl->work.get();
    m_impl->poll->stop();
    m_impl->scanning.clear();

    Impl::Entry& entry = m_impl->entries.at(result.agent);
    if (!result.error.isEmpty()) {
        emit Failed(result.agent, result.error);
    } else if (result.snapshot->root == entry.root.string()) {
        // Removed paths are dropped by QFileSystemWatcher;
        // periodic dependency checks also cover missing roots, file edits and operating-system watch limits.
        const QStringList watched = m_impl->watcher->directories();
        for (const QString& path : result.directories)
            if (!watched.contains(path)) m_impl->watcher->addPath(path);

        if (entry.serialized != result.serialized) {
            entry.snapshot = result.snapshot;
            entry.serialized = std::move(result.serialized);
            emit Changed(result.agent);
        }
    }

    qCDebug(catalogLog) << result.agent << result.elapsedMs << "ms; JSON reads" << result.jsonReads
        << "; directory reads" << result.directoryReads;
    emit RefreshCompleted(result.agent, result.jsonReads, result.directoryReads, result.elapsedMs);
    StartNext();
}
