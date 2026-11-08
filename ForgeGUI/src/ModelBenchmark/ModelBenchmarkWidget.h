#pragma once

#include "ModelBenchmarkLibrary.h"
#include "Common/ModelCatalog.h"
#include "Common/TableSortController.h"
#include "Common/WorkerThreadHost.h"
#include "Widgets/RunProgressPanel.h"

#include <GigaLearnCPP/SegmentVersioning.h>
#include <ModelBenchmark/ModelBenchmark.h>
#include <Settings/ForgeSettings.h>

#include <QWidget>

#include <cstdint>
#include <filesystem>
#include <future>
#include <map>
#include <optional>
#include <vector>

class AgentComboBox;
class QDialog;
class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLineEdit;
class QLabel;
class QTimer;
class QPlainTextEdit;
class QPushButton;
class QSplitter;
class QSpinBox;
class QTableWidget;
class QTabWidget;
class QWidget;
class ModelBenchmarkWorker;
class ModelBenchmarkSavedResultsWidget;
struct ModelBenchmarkLibraryModel;
class QTableWidgetItem;

/**
 * Model benchmarking workflow widget.
 * Lets users select saved model versions, run pairwise benchmarks, and inspect/export results.
 **/
class ModelBenchmarkWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ModelBenchmarkWidget(QWidget* parent = nullptr);
    ~ModelBenchmarkWidget() override;
    bool IsRunning() const;
    void SetTrainingActive(bool active);
    void ReloadPersistedSetup();
    void FollowTrainingAgent(const QString& agentName);
    void StartPersistedBenchmark();

signals:
    void RunningChanged(bool running);

private slots:
    void RefreshVersions();
    void OnContextChanged();
    void OnAddVersionsClicked();
    void OnAddExternalFolderClicked();
    void OnRemoveSelectedClicked();
    void OnClearSelectedClicked();
    void OnToggleSetupClicked();
    void OnRunClicked();
    void OnCancelClicked();
    void OnExportResultsClicked();
    void OnSaveResultsClicked();
    void OnProgressUpdated();
    void OnBenchmarkComplete(const RocketForgeTraining::ModelBenchmarkResult& result);
    void OnBenchmarkCancelled();
    void OnBenchmarkFailed(const QString& message, bool simulatorUnavailable);
    void UpdateInputState();

private:
    friend class ModelBenchmarkPickerTests;
    ModelBenchmarkWidget(QWidget* parent, bool persistSetup);

    struct BenchmarkVersionEntry {
        std::string agentName;
        std::string buildName;
        std::string stageName;
        std::string sourceLabel;
        GGL::SegmentVersionSummary version;
        std::filesystem::path checkpointPath;
        std::filesystem::path trainingSetupPath;
        RocketForgeTraining::ModelBenchmarkAgentSetupSource agentSetupSource =
            RocketForgeTraining::ModelBenchmarkAgentSetupSource::SavedSegment;
        bool primary = true;
        bool included = true;
        bool resultsOnly = false;
        qint64 libraryModelId = -1;
        QString fingerprint;
        std::string agentSetupJson;
        int colorIndex = -1;
        bool hasSavedAgentSetup = false;
        std::vector<RocketForge::AgentSetupDiff> agentSetupDiffs;
    };

    /** One discovery pass per agent; all picker filters operate on this catalog. */
    struct VersionCatalog {
        std::vector<std::string> builds;
        std::vector<BenchmarkVersionEntry> entries;
    };

    static VersionCatalog ScanVersionCatalog(const std::string& agentName, std::shared_ptr<const ModelCatalogSnapshot> snapshot);
    void EnsureVersionCatalog(bool refresh = false);
    void FinishVersionCatalogScan();
    void RefreshTrainingMetadata(BenchmarkVersionEntry& entry);
    void ShowVersionCatalog();
    void AddAvailableVersion(int row);
    void UpdateAvailableSelection();
    QString CurrentAgentName() const;
    QString CurrentBuildName() const;
    QString CurrentStageName() const;
    QString CurrentAvailableBranchName() const;
    void RefreshBuildSelector();
    void RefreshStageSelector();
    void RefreshAvailableBranchFilter();
    RocketForgeTraining::ModelBenchmarkConfig BuildConfig() const;
    static std::vector<ModelBenchmarkLibraryModel> BuildLibraryModels(
        const std::vector<BenchmarkVersionEntry>& entries, bool allowCached = false);
    void RefreshSavedKnownModels();
    void PopulateAvailableTable(bool preserveColumnWidths = false);
    void PopulateSelectedTable(bool preserveColumnWidths = false);
    void QueueSelectedTableRefresh();
    void ApplyAvailableRowFilters();
    void SortAvailableVersions();
    void SortSelectedVersions();
    int CompareBenchmarkColumn(int lhsRow, int rhsRow, int priorityColumn, bool selectedTable) const;
    static std::filesystem::path ResolveTrainingSetupPath(const BenchmarkVersionEntry& entry);
    static void RefreshAgentSetupStatus(BenchmarkVersionEntry& entry,
        const std::optional<RocketForge::AgentSetupSnapshot>* currentSetup = nullptr,
        const std::string& setupError = {});
    QTableWidgetItem* MakeSetupStatusItem(const BenchmarkVersionEntry& entry) const;

    /** Describes the effective setup and optionally labels its Trained/Current source. */
    QString BuildSetupStatusTooltip(const BenchmarkVersionEntry& entry, QString* sourceLabel = nullptr) const;
    std::string MakeEntryKey(const BenchmarkVersionEntry& entry) const;
    static std::string MakeDisplayName(const BenchmarkVersionEntry& entry);
    int NextAvailableModelColorIndex() const;
    void SetRunning(bool running);
    void AppendConsoleLine(const QString& text);
    QString BuildProgressStatus(double elapsedSeconds) const;
    QString BuildProgressDetailsStatus() const;
    void SetSetupSectionVisible(bool visible);
    void UpdateSectionToggleText();
    void LoadPersistedSetup();
    void SavePersistedSetup();
    void OnSetupChanged();
    RocketForgeTraining::ModelBenchmarkSettings BuildPersistedSetup() const;
    void ApplyPersistedSetup(const RocketForgeTraining::ModelBenchmarkSettings& settings);

    /**************** Unified Benchmark Workspace ****************/
    void ShowTestSetup();
    void ShowModelPicker();
    void RefreshWorkspace();
    void FinishWorkspaceRefresh();
    void ShowOpponentsPicker();
    void RefreshPresetList();
    void LoadWorkspacePreset();
    bool SaveWorkspacePreset(bool saveAs);
    void RenameWorkspacePreset();
    void NewWorkspacePreset();
    void LoadPreviousRun();
    void LoadLegacyComparison();
    void AddSavedModel(const ModelBenchmarkLibraryModel& model, bool resultsOnly = false);
    void AddOpponents();
    void ApplyWorkspaceSettings(const RocketForgeTraining::ModelBenchmarkSettings& settings);
    ModelBenchmarkLibraryFilter WorkspaceFilter() const;
    void UpdateCompatibilityControls();
    void ShowCompatibilityEnvironments(int engine);
    QString CompatibilitySummary() const;
    std::string CaptureModelSetup(const BenchmarkVersionEntry& entry) const;
    RocketForgeTraining::ModelBenchmarkConfig PrepareSchedule(bool showPreview, bool requireRunnable = false);
    void UpdateWorkspaceSummary();

    QDialog* m_testSetupDialog = nullptr;
    QDialog* m_modelPickerDialog = nullptr;
    QTableWidget* m_savedModelPicker = nullptr;
    QTableWidget* m_scheduleTable = nullptr;
    QComboBox* m_presetCombo = nullptr;
    QPushButton* m_renamePresetButton = nullptr;
    QPushButton* m_makeAllButton = nullptr;
    QComboBox* m_scheduleModeCombo = nullptr;
    QCheckBox* m_reuseSelectedEngineCheck = nullptr;
    std::map<int, QCheckBox*> m_reuseEngineChecks;
    std::map<int, QPushButton*> m_reuseEnvironmentButtons;
    std::map<int, std::vector<std::string>> m_reuseEnvironmentKeys;
    QLabel* m_setupSummaryLabel = nullptr;
    QLabel* m_activeLibraryLabel = nullptr;
    QLabel* m_presetStatusLabel = nullptr;
    QLabel* m_scheduleSummaryLabel = nullptr;
    QTimer* m_workspaceRefreshTimer = nullptr;
    QWidget* m_presetControls = nullptr;
    QWidget* m_modelControls = nullptr;
    QString m_presetName;
    QString m_savedPresetJson;
    QString m_runSettingsJson;
    QString m_coverageError;
    bool m_editingSetup = false;
    bool m_refreshingWorkspace = false;
    bool m_coverageDirty = true;

    /** A worker owns its database connection and copies; widgets only publish current generations. */
    struct WorkspaceData {
        std::vector<BenchmarkVersionEntry> selected;
        std::vector<ModelBenchmarkLibraryModel> models;
        std::vector<ModelBenchmarkLibraryCoverage> coverage;
        std::vector<ModelBenchmarkLibraryCoverage> discoveryCoverage;
        ModelBenchmarkLibraryFilter filter;
        ModelBenchmarkLibraryFilter coverageFilter;
        QString error;
    };

    std::future<WorkspaceData> m_workspaceScan;
    QTimer* m_workspacePollTimer = nullptr;
    uint64_t m_workspaceGeneration = 0;
    uint64_t m_scanningWorkspaceGeneration = 0;
    bool m_workspaceRefreshPending = false;
    bool m_runAfterWorkspaceRefresh = false;
    bool m_verifiedWorkspaceStart = false;
    bool m_previewAfterWorkspaceRefresh = false;
    bool m_opponentsAfterWorkspaceRefresh = false;
    ModelBenchmarkLibraryFilter m_loadedWorkspaceFilter;
    std::vector<ModelBenchmarkLibraryCoverage> m_discoveryCoverage;
    ModelBenchmarkLibraryFilter m_coverageFilter;
    ModelBenchmarkLibraryAggregate m_exportEnvironmentCoverage;
    std::vector<ModelBenchmarkLibraryCoverage> m_coverage;
    std::vector<ModelBenchmarkLibraryModel> m_libraryModels;
    std::vector<int> m_plannedModelIndices;
    std::map<QString, VersionCatalog> m_versionCatalogs;
    bool m_initializationPending = false;
    bool m_catalogRefreshPending = false;
    std::future<VersionCatalog> m_catalogScan;
    QString m_scanningAgent;
    QString m_displayedCatalogAgent;
    QString m_pendingStageFilter;
    QTimer* m_catalogPollTimer = nullptr;
    QTimer* m_setupSaveTimer = nullptr;
    QPushButton* m_refreshVersionsButton = nullptr;
    QLabel* m_catalogStatusLabel = nullptr;
    AgentComboBox* m_agentCombo = nullptr;
    QComboBox* m_buildCombo = nullptr;
    QComboBox* m_stageCombo = nullptr;
    QTableWidget* m_versionTable = nullptr;
    QTableWidget* m_selectedTable = nullptr;
    QWidget* m_setupContainer = nullptr;
    QSplitter* m_mainSplitter = nullptr;
    QPushButton* m_toggleSetupButton = nullptr;
    QCheckBox* m_hideImportedModelsCheck = nullptr;
    QComboBox* m_availableBranchCombo = nullptr;
    QLineEdit* m_availableSearchEdit = nullptr;
    QPushButton* m_addVersionsButton = nullptr;
    QPushButton* m_addExternalFolderButton = nullptr;
    QPushButton* m_removeSelectedButton = nullptr;
    QPushButton* m_clearSelectedButton = nullptr;
    QPushButton* m_toggleAllModelsButton = nullptr;

    QComboBox* m_engineCombo = nullptr;
    QSpinBox* m_arenasSpin = nullptr;
    QLabel* m_maxGamesLabel = nullptr;
    QSpinBox* m_maxGamesSpin = nullptr;
    QLineEdit* m_seedEdit = nullptr;
    QCheckBox* m_mode1v1Check = nullptr;
    QCheckBox* m_mode2v2Check = nullptr;
    QCheckBox* m_mode3v3Check = nullptr;
    QCheckBox* m_stochasticKickoffCheck = nullptr;
    QDoubleSpinBox* m_kickoffTemperatureSpin = nullptr;

    QPushButton* m_runButton = nullptr;
    QPushButton* m_cancelButton = nullptr;
    QPushButton* m_exportResultsButton = nullptr;
    QPushButton* m_saveResultsButton = nullptr;
    RunProgressPanel* m_progressPanel = nullptr;
    QTabWidget* m_workflowTabs = nullptr;
    ModelBenchmarkSavedResultsWidget* m_savedResultsWidget = nullptr;
    QPlainTextEdit* m_consoleText = nullptr;
    QString m_latestProgressTitle = "Ready";
    QString m_latestModeSummary = "--";
    int64_t m_latestProgressGames = 0;
    int m_latestDrawGames = 0;
    int64_t m_latestProgressMaxGames = 0;
    double m_latestProgressGamesPerSecond = 0.0;
    double m_latestProgressElapsedSeconds = 0.0;
    double m_latestEstimatedRemainingSeconds = -1.0;
    int m_latestActiveArenas = 0;
    int m_latestTotalArenas = 0;
    int m_latestPendingGames = 0;
    int m_latestScheduledGames = 0;
    int64_t m_latestArenaCreates = 0;
    int64_t m_latestArenaRebuilds = 0;
    int64_t m_latestArenaReuses = 0;
    int m_latestActive1v1Arenas = 0;
    int m_latestActive2v2Arenas = 0;
    int m_latestActive3v3Arenas = 0;
    int64_t m_latestCudaOomRecoveries = 0;
    bool m_latestCudaGraphsDisabledAfterOom = false;
    RocketForgeTraining::ModelBenchmarkGpuTelemetry m_latestGpuTelemetry = {};
    bool m_currentRunUntilMaxGames = false;

    WorkerThreadHost m_workerHost;
    ModelBenchmarkWorker* m_worker = nullptr;
    bool m_running = false;
    bool m_trainingActive = false;

    /** Test widgets must never read or overwrite the user's last-used setup. */
    const bool m_persistSetup;
    std::filesystem::path m_settingsPath;
    bool m_loadingPersistedSetup = false;
    bool m_hasLastResult = false;
    QString m_loadedAgentName;
    RocketForgeTraining::ModelBenchmarkResult m_lastResult = {};
    QString m_lastResultRunId;
    bool m_lastResultSavedToActiveLibrary = false;
    std::vector<BenchmarkVersionEntry> m_activeRunVersions;
    std::vector<ModelBenchmarkLibraryModel> m_activeRunLibraryModels;
    std::vector<ModelBenchmarkLibraryModel> m_lastRunLibraryModels;
    TableSortController m_availableSort;
    TableSortController m_selectedSort;
    bool m_availableTableColumnsInitialized = false;
    bool m_selectedTableColumnsInitialized = false;
    bool m_selectedTableRefreshQueued = false;
    std::vector<BenchmarkVersionEntry> m_availableVersions;
    std::vector<BenchmarkVersionEntry> m_selectedVersions;
    std::vector<BenchmarkVersionEntry> m_lastRunVersions;
};
