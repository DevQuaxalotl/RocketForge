#pragma once

#include <ModelBenchmark/ModelBenchmark.h>
#include <ModelBenchmark/SpreadsheetXmlWriter.h>

#include <QString>
#include <QStringList>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <map>
#include <set>
#include <vector>

/** Metadata that identifies one exact model snapshot in a benchmark library. **/
struct ModelBenchmarkLibraryModel
{
    qint64 id = -1;
    QString logicalKey;
    QString displayName;
    QString agentName;
    QString buildName;
    QString stageName;
    QString branchName;
    QString versionId;
    QString sourceLabel;
    QString checkpointPath;
    QString trainingSetupPath;
    QString fingerprint;
    quint64 latestTimesteps = 0;
    QString lastTrainedAt;
    bool fingerprintVerified = false;
    bool availableLocally = false;
    /** Initial Saved Results selection state; not part of the stored identity. **/
    bool primary = true;
};

/** One changed-checkpoint collision detected before a library save. **/
struct ModelBenchmarkLibraryConflict
{
    int incomingModelIndex = -1;
    ModelBenchmarkLibraryModel incoming;
    std::vector<ModelBenchmarkLibraryModel> existing;
};

/** Saved-model selection used to rebuild an aggregated result. **/
struct ModelBenchmarkLibrarySelection
{
    qint64 modelId = -1;
    bool primary = true;
};

/** One comparison member, including temporarily excluded bots. **/
struct ModelBenchmarkComparisonMember
{
    qint64 modelId = -1;
    bool included = true;
    bool primary = true;

    bool operator==(const ModelBenchmarkComparisonMember&) const = default;
};

/** Named comparison stored alongside its exact model snapshots in the library. **/
struct ModelBenchmarkComparisonSet
{
    qint64 id = -1;
    QString name;
    double confidencePercent = 95.0;
    std::vector<ModelBenchmarkComparisonMember> members;
};

/** Shared conditions for coverage, opponent discovery, and Elo aggregation. **/
struct ModelBenchmarkLibraryFilter
{
    /** Zero selects all simulators; the legacy -1 selector aliases v2. **/
    int engineVersion = 0;

    /** Display-only grouping key, including legacy environments without an exact identity. **/
    QString environmentKey;

    /** Optional exact recorded environment; unidentified legacy runs never satisfy it. **/
    QString evaluationIdentity;
    int modeMask = 7;
    int stochasticKickoff = -1;
    float kickoffTemperature = 1.0f;
    QString runId;

    /** When enabled, only these simulators and their selected environments contribute. **/
    bool restrictEnvironments = false;
    std::set<int> engineVersions;

    /** A missing simulator entry accepts all environments; an empty list accepts none. **/
    std::map<int, std::vector<std::string>> environmentKeys;

    /** Shared setup selection for saved coverage, results, and live snapshots. **/
    bool MatchesEnvironmentSelection(int engine, const QString& key) const;
    bool Matches(const RocketForgeTraining::ModelBenchmarkResult& result) const;
    bool operator==(const ModelBenchmarkLibraryFilter&) const = default;
};

/** Saved configuration or recorded run available in the workspace toolbar. **/
struct ModelBenchmarkWorkspaceRecord
{
    QString id;
    QString label;
    QString settingsJson;
};

/** Completed games for one exact snapshot pair and mode, including draws. **/
struct ModelBenchmarkLibraryCoverage
{
    qint64 modelA = -1;
    qint64 modelB = -1;
    int mode = 1;
    qint64 games = 0;
    uint64_t lastSnapshotId = 0;

    /** Original provenance; empty for legacy results without an exact identity. **/
    QString evaluationIdentity;

    /** Identity projected onto modelA/modelB, independent of other models in the saved run. **/
    QString pairEvaluationIdentity;

    /** Recorded simulator (-1 for mixed v2/v3) and legacy compatibility marker retained as provenance. **/
    int engineVersion = 0;
    int compatibilityVersion = 0;

    /** Stable recorded environment key for explicit reuse selection. **/
    QString environmentKey;
};

/** Keeps shared environment settings and only the two models' prediction settings, in argument order. **/
QString ModelBenchmarkPairEnvironmentKey(const QString& identity, const QString& checkpointA, const QString& checkpointB);

/** Completed game counts for one recorded environment in the selected comparison. **/
struct ModelBenchmarkLibraryEnvironment
{
    RocketForgeTraining::ModelBenchmarkResult provenance;
    qint64 games = 0;
    std::map<int, qint64> modelGames;
};

/** Result and provenance returned by a saved-data query. **/
struct ModelBenchmarkLibraryAggregate
{
    RocketForgeTraining::ModelBenchmarkResult result;

    /** Selected-pair games before Primary filtering, used by Match Coverage. **/
    std::vector<RocketForgeTraining::ModelBenchmarkGameMetadata> coverageGames;
    int contributingRuns = 0;

    /** Completed games from the unsaved current run included in this view. **/
    int liveGames = 0;

    /** Candidate/bulk SQL statements issued by LoadAggregate, excluding the fixed model-list query. **/
    int loadQueryCount = 0;
    QString firstRunAt;
    QString latestRunAt;
    QStringList warnings;

    /** All eligible environments before display-only simulator/environment filters. **/
    std::map<std::string, ModelBenchmarkLibraryEnvironment> availableEnvironments;

    /** Environments actually contributing to the displayed result and export. **/
    std::map<std::string, ModelBenchmarkLibraryEnvironment> environments;
};

/** Human-readable provenance and export rows shared by the saved-results UI and tests. **/
QString ModelBenchmarkEnvironmentLabel(const ModelBenchmarkLibraryEnvironment& environment);
bool ModelBenchmarkHasMixedEnvironments(const ModelBenchmarkLibraryAggregate& aggregate);
void WriteModelBenchmarkEnvironmentWorksheet(RocketForgeTraining::SpreadsheetXmlWriter& writer, const ModelBenchmarkLibraryAggregate& aggregate);

/** Parsed legacy Excel export ready for the normal conflict/save workflow. **/
struct ModelBenchmarkLibraryImport
{
    QString runId;
    RocketForgeTraining::ModelBenchmarkResult result;
    std::vector<ModelBenchmarkLibraryModel> models;
    QStringList warnings;
};

/**
 * Owns one versioned SQLite model-benchmark library.
 * All mutations are transactional and model results are recomputed on read.
 **/
class ModelBenchmarkLibrary
{
public:
    ModelBenchmarkLibrary();
    ~ModelBenchmarkLibrary();

    bool Open(const std::filesystem::path& path, QString* error = nullptr);
    bool OpenReadOnly(const std::filesystem::path& path, QString* error = nullptr);
    void Close();
    bool IsOpen() const;
    std::filesystem::path Path() const;

    std::vector<ModelBenchmarkLibraryModel> ListModels(QString* error = nullptr) const;
    std::vector<ModelBenchmarkComparisonSet> ListComparisonSets(QString* error = nullptr) const;

    /** Creates a set when id is negative; otherwise updates it atomically. Returns its saved id and name. **/
    bool SaveComparisonSet(ModelBenchmarkComparisonSet& comparison, QString* error = nullptr);
    std::vector<ModelBenchmarkLibraryConflict> FindConflicts(
        const std::vector<ModelBenchmarkLibraryModel>& models,
        QString* error = nullptr) const;
    bool SaveRun(
        const QString& runId,
        const RocketForgeTraining::ModelBenchmarkResult& result,
        const std::vector<ModelBenchmarkLibraryModel>& models,
        const std::set<qint64>& replaceModelIds,
        QString* error = nullptr,
        const QString& settingsJson = {});
    ModelBenchmarkLibraryAggregate LoadAggregate(
        const std::vector<ModelBenchmarkLibrarySelection>& selection,
        float eloRangeConfidence,
        QString* error = nullptr,
        const std::function<bool()>& isCancelled = {},
        const ModelBenchmarkLibraryFilter& filter = {},
        const std::vector<ModelBenchmarkLibraryModel>& additionalModels = {}) const;
    std::vector<ModelBenchmarkLibraryCoverage> LoadCoverage(const ModelBenchmarkLibraryFilter& filter, QString* error = nullptr) const;

    /** Optional acceleration for writable libraries; call from a worker, never during window construction. **/
    static bool EnsureCoverageIndex(const std::filesystem::path& path);

    /** Lightweight catalog of recorded run environments, independent of roster and result filters. **/
    std::map<std::string, ModelBenchmarkLibraryEnvironment> ListEnvironments(QString* error = nullptr) const;
    std::vector<ModelBenchmarkWorkspaceRecord> ListPresets(QString* error = nullptr) const;
    std::vector<ModelBenchmarkWorkspaceRecord> ListRuns(QString* error = nullptr) const;

    /** Renames a saved preset without saving changes from the working configuration. **/
    bool RenamePreset(const QString& oldName, const QString& newName, QString* error = nullptr);

    /** Creates a preset without replacing an existing name. **/
    bool CreatePreset(const QString& name, const QString& settingsJson, QString* error = nullptr);
    bool SavePreset(const QString& name, const QString& settingsJson, QString* error = nullptr);

    static QString FingerprintCheckpoint(
        const std::filesystem::path& checkpointPath,
        const std::filesystem::path& trainingSetupPath = {},
        bool allowCached = false);
    static ModelBenchmarkLibraryImport ReadExcelExport(
        const std::filesystem::path& path,
        QString* error = nullptr);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

/** Recomputes saved plus live results from a fixed saved baseline and a cumulative snapshot. **/
ModelBenchmarkLibraryAggregate CombineModelBenchmarkLiveResults(
    const ModelBenchmarkLibraryAggregate& saved,
    const std::vector<ModelBenchmarkLibrarySelection>& selection,
    const RocketForgeTraining::ModelBenchmarkResult& live,
    const std::vector<qint64>& liveModelIds,
    const ModelBenchmarkLibraryFilter& filter);
