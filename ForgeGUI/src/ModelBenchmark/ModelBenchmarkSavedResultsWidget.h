#pragma once

#include "ModelBenchmarkLibrary.h"

#include <QWidget>

#include <atomic>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <vector>

class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QTableWidget;
class QTabWidget;
class QThread;

/** Saved benchmark-library browser embedded in the Model Benchmark tab. **/
class ModelBenchmarkSavedResultsWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ModelBenchmarkSavedResultsWidget(QWidget* parent = nullptr);
    ~ModelBenchmarkSavedResultsWidget() override;

    /** Embeds only the result views; the parent owns the single shared roster. **/
    void SetWorkspaceMode();

    /** Keeps run diagnostics available alongside the combined result tabs. **/
    void AddRunConsole(QWidget* console);
    void SetWorkspaceSelection(const std::vector<ModelBenchmarkLibrarySelection>& selection, const ModelBenchmarkLibraryFilter& filter);

    /** Uses a fixed saved baseline while cumulative live snapshots arrive. **/
    void BeginLiveComparison(
        const std::vector<ModelBenchmarkLibrarySelection>& selection,
        const std::vector<ModelBenchmarkLibraryModel>& additionalModels,
        const std::vector<qint64>& liveModelIds,
        const ModelBenchmarkLibraryFilter& filter);
    void UpdateLiveComparison(std::shared_ptr<const RocketForgeTraining::ModelBenchmarkResult> result);
    void EndLiveComparison();
    ModelBenchmarkLibrary& Library() { return m_library; }
    void ManageLibrary(bool createNew) { ChooseLibrary(createNew); }
    void ImportSavedResults() { ImportExcel(); }
    void ExportDisplayedResults();
    bool OpenLibrary(const std::filesystem::path& path, bool showErrors = true);
    std::filesystem::path LibraryPath() const;
    void SetKnownModels(const std::vector<ModelBenchmarkLibraryModel>& models);
    bool SaveRun(
        const QString& runId,
        const RocketForgeTraining::ModelBenchmarkResult& result,
        const std::vector<ModelBenchmarkLibraryModel>& models,
        bool notify = true,
        const QString& settingsJson = {});

signals:
    void LibraryPathChanged();
    void LibraryDataChanged();
    void ExportRequested(ModelBenchmarkLibraryAggregate aggregate);

private:
    friend class ModelBenchmarkSavedResultsTests;

    ModelBenchmarkComparisonSet CurrentComparison() const;
    bool IsComparisonModified() const;
    bool ConfirmComparisonChange();
    void RefreshComparisonSets();
    void UpdateComparisonControls();
    void LoadComparisonSet();
    void NewComparisonSet();
    bool SaveComparisonSet(bool saveAs);
    void ChooseLibrary(bool createNew);
    void ImportExcel();
    void RefreshModels();
    void ApplyKnownModels(
        const std::vector<ModelBenchmarkLibraryModel>& saved,
        const std::vector<ModelBenchmarkLibraryModel>& known,
        bool fingerprintsVerified);
    void StartFingerprintVerification(const std::vector<ModelBenchmarkLibraryModel>& saved);
    void RefreshModelFilters();
    void PopulateModelTable();
    void PopulatePoolTable();
    void RefreshSelectionTables();
    void QueueSelectionRefresh();
    void QueueResultsRefresh();
    void RemoveSelectedPoolModels();
    void RebuildResults();
    void PopulateResults(const ModelBenchmarkLibraryAggregate& aggregate);
    void PopulateEnvironmentCoverage(const ModelBenchmarkLibraryAggregate& aggregate);

    /** Refreshes only the selected result view from the latest snapshot. **/
    void PopulateCurrentResultTab();
    void PopulateCoverageTable();
    void PopulateStatsTable();
    void SetEmptyResults(const QString& message);

    ModelBenchmarkLibrary m_library;
    ModelBenchmarkLibraryFilter m_workspaceFilter;
    bool m_workspaceMode = false;
    bool m_liveComparison = false;
    bool m_liveRefreshPending = false;
    bool m_liveRefreshQueued = false;
    quint64 m_liveSession = 0;
    std::vector<ModelBenchmarkLibraryModel> m_liveAdditionalModels;
    std::vector<qint64> m_liveModelIds;
    std::shared_ptr<const RocketForgeTraining::ModelBenchmarkResult> m_liveResult;
    std::shared_ptr<const ModelBenchmarkLibraryAggregate> m_savedBaseline;
    ModelBenchmarkComparisonSet m_savedComparison;
    std::vector<ModelBenchmarkLibraryModel> m_knownModels;
    std::vector<ModelBenchmarkLibraryModel> m_rows;
    std::set<qint64> m_pooledModelIds;
    std::map<qint64, bool> m_includedByModelId;
    std::map<qint64, bool> m_primaryByModelId;
    std::vector<bool> m_displayedPrimaryFlags;
    RocketForgeTraining::ModelBenchmarkResult m_displayedResult;
    std::vector<RocketForgeTraining::ModelBenchmarkGameMetadata> m_displayedCoverageGames;
    std::set<QTableWidget*> m_staleResultTables;
    bool m_hasDisplayedResult = false;
    bool m_selectionRefreshQueued = false;
    bool m_resultsRefreshQueued = false;
    bool m_aggregateRefreshPending = false;
    quint64 m_aggregateRequestId = 0;
    QThread* m_aggregateThread = nullptr;
    std::shared_ptr<std::atomic_bool> m_aggregateCancellation;
    quint64 m_fingerprintRequestId = 0;
    bool m_fingerprintRefreshPending = false;
    QThread* m_fingerprintThread = nullptr;
    std::shared_ptr<std::atomic_bool> m_fingerprintCancellation;

    QLabel* m_libraryPathLabel = nullptr;
    QComboBox* m_comparisonCombo = nullptr;
    QLabel* m_comparisonStatusLabel = nullptr;
    QPushButton* m_loadComparisonButton = nullptr;
    QPushButton* m_newComparisonButton = nullptr;
    QPushButton* m_saveComparisonButton = nullptr;
    QPushButton* m_saveComparisonAsButton = nullptr;
    QPushButton* m_includeAllButton = nullptr;
    QPushButton* m_excludeAllButton = nullptr;
    QPushButton* m_primaryAllButton = nullptr;
    QPushButton* m_primaryNoneButton = nullptr;
    QPushButton* m_removeButton = nullptr;
    QLabel* m_poolStatusLabel = nullptr;
    QLineEdit* m_searchEdit = nullptr;
    QComboBox* m_agentFilterCombo = nullptr;
    QComboBox* m_buildFilterCombo = nullptr;
    QComboBox* m_stageFilterCombo = nullptr;
    QComboBox* m_branchFilterCombo = nullptr;
    QTableWidget* m_modelTable = nullptr;
    QTableWidget* m_poolTable = nullptr;
    QWidget* m_environmentControls = nullptr;
    QLabel* m_environmentSummaryLabel = nullptr;
    QTableWidget* m_environmentCoverageTable = nullptr;
    std::map<std::string, ModelBenchmarkLibraryEnvironment> m_displayedEnvironments;
    QTabWidget* m_resultTabs = nullptr;
    QTableWidget* m_rankingTable = nullptr;
    QTableWidget* m_breakdownTable = nullptr;
    QTableWidget* m_pairwiseTable = nullptr;
    QTableWidget* m_coverageTable = nullptr;
    QComboBox* m_statsModeCombo = nullptr;
    QTableWidget* m_statsTable = nullptr;
    QTableWidget* m_detailsTable = nullptr;
    QPushButton* m_exportButton = nullptr;
};
