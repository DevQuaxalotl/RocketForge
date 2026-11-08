#pragma once

#include <TrainingSetup/VersionManagement.h>
#include <QObject>
#include <memory>

namespace RocketForgeTraining {
/** Cached storage metadata for benchmark discovery. **/
struct AgentVersionCatalog {
    std::string agentName;
    std::vector<BuildSummary> builds;
    std::map<std::string, std::vector<TrainingSegmentMetadata>> segmentsByBuild;
    std::map<std::string, std::vector<GGL::SegmentVersionSummary>> summariesByBuild;
};
}

/** One browseable model, with all display-only filesystem work already completed. **/
struct CatalogModel {
    GGL::SegmentVersionSummary version;
    std::string sourceLabel = "RocketForge";
    std::string displayName;
    std::string checkpointPath;
    std::string setupPath;
    nlohmann::json agentSetup;
};

/** Immutable snapshot shared by all tabs; historical files remain authoritative. **/
struct ModelCatalogSnapshot {
    RocketForgeTraining::AgentVersionCatalog storage;
    std::map<std::string, std::string> activeSegments;
    std::vector<CatalogModel> models;
    std::string root;

    /** Filter cached training segments without reading source files. **/
    std::vector<GGL::SegmentVersionSummary> Versions(const std::string& build = {},
        const std::string& stage = {}, bool includeInactive = false) const;

    /** Stage names for cached active segments in the selected Build. **/
    std::vector<std::string> TrainingStages(const std::string& build = {}) const;
};

/**
 * Publishes cached rows immediately, then validates dependencies on one worker.
 * Only changed JSON/directory contents are loaded.
 * Notifications and a periodic metadata pass cover both live writers and edits made while the app was closed.
 **/
class ModelCatalog : public QObject {
    Q_OBJECT
public:
    static ModelCatalog& Instance();
    ~ModelCatalog() override;
    std::shared_ptr<const ModelCatalogSnapshot> Get(const QString& agent);

    /** Repair is reserved for an explicit maintenance request; ordinary reads never mutate source metadata. **/
    void Refresh(const QString& agent, bool repair = false);
    bool IsRefreshing(const QString& agent) const;

signals:
    void Changed(const QString& agent);
    void Failed(const QString& agent, const QString& message);
    void RefreshCompleted(const QString& agent, int jsonReads, int directoryReads, qint64 elapsedMs);

private:
    explicit ModelCatalog(QObject* parent);
    void StartNext();
    void Finish();
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
