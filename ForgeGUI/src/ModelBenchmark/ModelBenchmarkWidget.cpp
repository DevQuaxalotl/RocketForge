#include "ModelBenchmarkWidget.h"

#include "Common/ModelCatalog.h"
#include "SimulatorUnavailableDialog.h"
#include "FormatUtils.h"
#include "ModelBenchmarkResultTables.h"
#include "ModelBenchmarkSavedResultsWidget.h"
#include "ModelBenchmarkWorker.h"

#include "Widgets/AgentComboBox.h"

#include <AgentModelPaths.h>
#include <Storage/BuildStorage.h>
#include <TrainingSetup/VersionManagement.h>
#include <ForgeTraining.h>
#include <ModelBenchmark/ModelBenchmark.h>
#include <ModelBenchmark/ModelBenchmarkExport.h>
#include <ModelBenchmark/CheckpointFingerprint.h>
#include <Storage/ImportedModelStorage.h>
#include <TrainingSetup/TrainingSetupSnapshot.h>
#include <GigaLearnCPP/TrainingStorage.h>
#include <GigaLearnInfer/AgentRegistry.h>
#include <GigaLearnInfer/AgentSetupSnapshot.h>
#include <RocketSim/Engine.h>

#include <QCheckBox>
#include <QComboBox>
#include <QDateTime>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QInputDialog>
#include <QMenu>
#include <QScrollArea>
#include <QScreen>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QMetaObject>
#include <QIcon>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QScrollBar>
#include <QSignalBlocker>
#include <QSplitter>
#include <QSpinBox>
#include <QStyle>
#include <QStandardItemModel>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QTabWidget>
#include <QTimer>
#include <QValidator>
#include <QVBoxLayout>
#include <QUuid>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace
{
    using namespace ForgeGUI::ModelBenchmarkTables;

    constexpr int VersionColumnUse = 0;
    constexpr int VersionColumnSetup = 1;
    constexpr int VersionColumnAgent = 2;
    constexpr int VersionColumnBuild = 3;
    constexpr int VersionColumnId = 4;
    constexpr int VersionColumnStage = 5;
    constexpr int VersionColumnBranch = 6;
    constexpr int VersionColumnLatest = 7;
    constexpr int VersionColumnRating = 8;
    constexpr int VersionColumnNotes = 9;

    constexpr int SelectedColumnUse = 0;
    constexpr int SelectedColumnPrimary = 1;
    constexpr int SelectedColumnSetup = 2;
    constexpr int SelectedColumnAgent = 3;
    constexpr int SelectedColumnBuild = 4;
    constexpr int SelectedColumnId = 5;
    constexpr int SelectedColumnLatest = 6;
    constexpr int SelectedColumnLastTrained = 7;
    constexpr int SelectedColumnStage = 8;
    constexpr int SelectedColumnBranch = 9;
    constexpr int SelectedColumnAgentSetup = 10;
    constexpr int SelectedColumnCheckpoint = 11;

    // Priority order for benchmark-entry sorting (lower = higher priority). Build then segment
    // first, matching the VersionManagement default philosophy.
    enum BenchmarkSortColumn {
        BenchSortBuild = 0,
        BenchSortId,          // segment/version (major, minor, id)
        BenchSortStage,
        BenchSortBranch,
        BenchSortAgent,
        BenchSortLatest,
        BenchSortLastTrained,
        BenchSortRating,
        BenchSortSetup,       // hasSavedAgentSetup + diff count
        BenchSortNotes,
        BenchSortPrimary,     // selected table only
        BenchSortAgentSetup,  // selected table only
        BenchSortCheckpoint,  // selected table only
        BenchSortCount
    };

    // Available table UI columns: Use, Setup, Agent, Build, Id, Stage, Branch, Latest, Rating, Notes.
    // The Use column (UI index 0) is a checkbox and non-sortable, so it maps to -1.
    const std::vector<int> AvailableUiOrder = {
        -1, BenchSortSetup, BenchSortAgent, BenchSortBuild, BenchSortId,
        BenchSortStage, BenchSortBranch, BenchSortLatest, BenchSortRating, BenchSortNotes,
    };

    // Selected table keeps training progress and dates visible beside the model identity.
    const std::vector<int> SelectedUiOrder = {
        -1, BenchSortPrimary, BenchSortSetup, BenchSortAgent, BenchSortBuild, BenchSortId,
        BenchSortLatest, BenchSortLastTrained, BenchSortStage, BenchSortBranch, BenchSortAgentSetup, BenchSortCheckpoint,
    };

    constexpr const char* AllBuildsLabel = "All Builds";
    constexpr const char* AllStagesLabel = "All Training Stages";
    constexpr const char* AllBranchesLabel = "All Branches";
    constexpr const char* IncludeTooltip =
        "<qt>Check to show this model in the results; uncheck to hide it. "
        "This does not change benchmark participation, which is controlled by Next run. Saved data is kept.</qt>";
    constexpr const char* NextRunTooltip =
        "<b>Target:</b> A model selected for testing in the next run.<br>"
        "<b>Opponent:</b> A model for targets to play against.<br>"
        "<b>Results only:</b> Use its saved games without running new games.<br><br>"
        "New games require at least one Target, including Target vs Target. "
        "Two Opponents never play new games against each other. "
        "Pairs already at the game target can be skipped.<br><br>"
        "The <b>Show</b> checkboxes control which models contribute saved games to combined results and Elo, regardless of role.";
    constexpr const char* RunBenchmarkTooltip =
        "<qt>Run the new games shown in Review Schedule using the current test setup. "
        "Combined results include completed games as they arrive; the run is saved automatically when it ends.</qt>";

    void SetHeaderTooltips(QTableWidget* table, const QStringList& tooltips)
    {
        for (int column = 0; column < table->columnCount() && column < tooltips.size(); ++column)
            table->horizontalHeaderItem(column)->setToolTip(tooltips[column].startsWith("<")
                ? tooltips[column] : "<qt>" + tooltips[column].toHtmlEscaped() + "</qt>");
    }

    class UInt64Validator : public QValidator
    {
    public:
        explicit UInt64Validator(QObject* parent = nullptr)
            : QValidator(parent)
        {
        }

        State validate(QString& input, int&) const override
        {
            if (input.isEmpty())
                return Intermediate;
            for (const QChar character : input) {
                if (!character.isDigit())
                    return Invalid;
            }

            bool ok = false;
            input.toULongLong(&ok, 10);
            return ok ? Acceptable : Invalid;
        }
    };

    uint64_t ParseBenchmarkSeed(const QLineEdit* edit)
    {
        if (!edit || edit->text().trimmed().isEmpty())
            return 0;

        bool ok = false;
        const qulonglong value = edit->text().trimmed().toULongLong(&ok, 10);
        if (!ok)
            throw std::runtime_error("Benchmark seed must be an unsigned 64-bit integer.");
        return (uint64_t)value;
    }

    std::string BranchNameOrDefault(const std::string& branchName)
    {
        return branchName.empty() ? "main" : branchName;
    }

    void FitPrimaryBenchmarkColumn(QTableWidget* table)
    {
        if (!table || !table->horizontalHeader())
            return;

        auto* header = table->horizontalHeader();
        table->resizeColumnToContents(SelectedColumnPrimary);
        const int headerWidth = header->fontMetrics().horizontalAdvance("Primary") + 24;
        header->resizeSection(SelectedColumnPrimary, std::max(headerWidth, table->columnWidth(SelectedColumnPrimary)));
    }

    std::vector<int> CaptureColumnWidths(QTableWidget* table)
    {
        std::vector<int> columnWidths;
        if (!table)
            return columnWidths;

        columnWidths.reserve(table->columnCount());
        for (int column = 0; column < table->columnCount(); column++)
            columnWidths.push_back(table->columnWidth(column));
        return columnWidths;
    }

    void RestoreColumnWidths(QTableWidget* table, const std::vector<int>& columnWidths)
    {
        if (!table || (int)columnWidths.size() != table->columnCount())
            return;

        for (int column = 0; column < table->columnCount(); column++)
            table->setColumnWidth(column, columnWidths[column]);
    }

    QString FormatElapsed(double seconds)
    {
        const int totalSeconds = std::max(0, (int)std::llround(seconds));
        const int hours = totalSeconds / 3600;
        const int minutes = (totalSeconds / 60) % 60;
        const int secs = totalSeconds % 60;
        if (hours > 0)
            return QString("%1:%2:%3")
                .arg(hours)
                .arg(minutes, 2, 10, QChar('0'))
                .arg(secs, 2, 10, QChar('0'));
        return QString("%1:%2")
            .arg(minutes)
            .arg(secs, 2, 10, QChar('0'));
    }

    QString FormatElapsedLong(double seconds)
    {
        const int totalSeconds = std::max(0, (int)std::llround(seconds));
        const int hours = totalSeconds / 3600;
        const int minutes = (totalSeconds / 60) % 60;
        const int secs = totalSeconds % 60;
        if (hours > 0)
            return QString("%1h:%2m:%3s").arg(hours).arg(minutes, 2, 10, QChar('0')).arg(secs, 2, 10, QChar('0'));
        if (minutes > 0)
            return QString("%1m:%2s").arg(minutes).arg(secs, 2, 10, QChar('0'));
        return QString("%1s").arg(secs);
    }

    int ProgressPercent(int64_t completedGames, int64_t maxGames)
    {
        if (completedGames <= 0 || maxGames <= 0)
            return 0;
        if (completedGames >= maxGames)
            return 100;
        return (int)((long double)completedGames * 100.0L / (long double)maxGames);
    }
}

ModelBenchmarkWidget::ModelBenchmarkWidget(QWidget* parent)
    : ModelBenchmarkWidget(parent, true)
{
}

ModelBenchmarkWidget::ModelBenchmarkWidget(QWidget* parent, bool persistSetup)
    : QWidget(parent), m_persistSetup(persistSetup),
      m_settingsPath(RocketForgeTraining::GetDefaultModelBenchmarkSettingsPath())
{
    qRegisterMetaType<RocketForgeTraining::ModelBenchmarkProgress>("RocketForgeTraining::ModelBenchmarkProgress");
    qRegisterMetaType<RocketForgeTraining::ModelBenchmarkResult>("RocketForgeTraining::ModelBenchmarkResult");

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(12, 10, 12, 10);
    root->setSpacing(8);
    setObjectName("modelBenchmarkWorkspace");
    setAttribute(Qt::WA_StyledBackground, true);

    m_presetControls = new QWidget(this);
    auto* presetLayout = new QVBoxLayout(m_presetControls);
    presetLayout->setContentsMargins(0, 0, 0, 0);
    presetLayout->setSpacing(8);

    auto* toolbar = new QHBoxLayout;
    presetLayout->addLayout(toolbar);
    toolbar->setContentsMargins(0, 0, 0, 0);
    toolbar->addWidget(new QLabel("Benchmark preset", this));

    m_presetCombo = new QComboBox(this);
    m_presetCombo->setObjectName("benchmarkPreset");
    m_presetCombo->setToolTip("<qt>A preset stores your model list, roles, test setup, and comparison filters. Select a preset to load it automatically.</qt>");
    m_presetCombo->setMinimumContentsLength(20);
    m_presetCombo->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
    toolbar->addWidget(m_presetCombo, 1);

    auto* newPreset = new QPushButton("New", this);
    newPreset->setObjectName("benchmarkNewPreset");
    newPreset->setToolTip("<qt>Create a named preset with an empty model list and default test settings.</qt>");
    toolbar->addWidget(newPreset);

    auto* savePreset = new QPushButton("Save", this);
    savePreset->setToolTip("<qt>Update the current named preset with this model list and setup. If it has no name yet, choose one. This does not run games.</qt>");
    auto* saveAs = new QPushButton("Save As…", this);
    saveAs->setToolTip("Save this model list and setup under a new name, keeping the original preset.");
    m_renamePresetButton = new QPushButton("Rename", this);
    m_renamePresetButton->setObjectName("benchmarkRenamePreset");
    m_renamePresetButton->setEnabled(false);
    m_renamePresetButton->setToolTip("Rename the selected saved preset. Unsaved setup changes remain unsaved.");
    auto* previousRun = new QPushButton("Use Previous Run…", this);
    previousRun->setToolTip("<qt>Restore the model list and recorded test setup from a saved run, then adjust them for your next test. Older runs may not have a complete setup recorded.</qt>");
    toolbar->addWidget(savePreset);
    toolbar->addWidget(saveAs);
    toolbar->addWidget(m_renamePresetButton);
    toolbar->addWidget(previousRun);

    m_toggleSetupButton = new QPushButton("Test Setup…", this);
    m_toggleSetupButton->setObjectName("benchmarkTestSetupButton");
    m_toggleSetupButton->setToolTip("<qt>Choose game targets, team sizes, physics engine, kickoff behavior, and stopping rules. Settings are remembered and can be saved in a preset.</qt>");
    auto* libraryButton = new QPushButton(this);
    libraryButton->setText("Library");
    libraryButton->setToolTip("<qt>Manage the database containing saved games and presets, import results, or load an older comparison set.</qt>");
    auto* libraryMenu = new QMenu(libraryButton);
    libraryMenu->setToolTipsVisible(true);
    libraryButton->setMenu(libraryMenu);
    root->addWidget(m_presetControls);

    m_presetStatusLabel = new QLabel(this);
    m_presetStatusLabel->setTextFormat(Qt::PlainText);
    m_presetStatusLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    m_presetStatusLabel->ensurePolished();

    int statusWidth = 0;
    for (const char* text : { "Working configuration", "Unsaved changes", "Preset saved" }) {
        m_presetStatusLabel->setText(text);
        statusWidth = std::max(statusWidth, m_presetStatusLabel->sizeHint().width());
    }

    m_presetStatusLabel->clear();
    m_presetStatusLabel->setFixedWidth(statusWidth);
    toolbar->addWidget(m_presetStatusLabel);
    m_presetStatusLabel->setToolTip("<qt>Shows whether the current configuration matches its named preset. Unsaved changes are remembered for this workspace, but use Save to update the named preset.</qt>");

    m_setupSummaryLabel = new QLabel(this);
    m_setupSummaryLabel->setWordWrap(true);
    auto* summary = new QHBoxLayout;
    summary->addWidget(m_setupSummaryLabel, 1);

    m_activeLibraryLabel = new QLabel("No library open", this);
    m_activeLibraryLabel->setObjectName("benchmarkActiveLibrary");
    m_activeLibraryLabel->setTextFormat(Qt::PlainText);
    m_activeLibraryLabel->setMaximumWidth(260);
    summary->addWidget(m_activeLibraryLabel);
    summary->addWidget(libraryButton);
    summary->addWidget(m_toggleSetupButton);
    presetLayout->addLayout(summary);

    m_mainSplitter = new QSplitter(Qt::Vertical, this);
    m_mainSplitter->setChildrenCollapsible(false);
    m_mainSplitter->setHandleWidth(6);

    m_setupContainer = new QWidget(this);
    auto* setupRoot = new QVBoxLayout(m_setupContainer);
    setupRoot->setContentsMargins(0, 0, 0, 0);
    setupRoot->setSpacing(6);

    m_modelControls = new QWidget(this);
    auto* modelTools = new QHBoxLayout(m_modelControls);
    modelTools->setContentsMargins(0, 0, 0, 0);
    auto* modelHeading = new QLabel("Comparison models", this);
    modelHeading->setToolTip("<qt>One shared list for scheduling new games and comparing saved results. Show controls visibility in results; Next run controls participation in new games.</qt>");
    QFont headingFont = modelHeading->font();
    headingFont.setBold(true);
    modelHeading->setFont(headingFont);
    modelTools->addWidget(modelHeading);

    auto* addModels = new QPushButton("Add Models…", this);
    addModels->setObjectName("benchmarkAddModelsButton");
    addModels->setToolTip("<qt>Add model versions from local checkpoints or the saved-results library. Models without a local checkpoint can still be compared using saved results.</qt>");
    m_removeSelectedButton = new QPushButton("Remove Selected", this);
    m_removeSelectedButton->setToolTip("<qt>Remove the highlighted rows from this comparison. Use Ctrl or Shift to select several rows. Their saved games stay in the library.</qt>");
    m_clearSelectedButton = new QPushButton("Clear All", this);
    m_clearSelectedButton->setToolTip("<qt>Empty the comparison model list. Saved games and named presets stay in the library.</qt>");
    modelTools->addStretch();
    modelTools->addWidget(addModels);

    m_toggleAllModelsButton = new QPushButton("Show All", this);
    m_toggleAllModelsButton->setEnabled(false);
    m_toggleAllModelsButton->setObjectName("benchmarkToggleAllModels");
    m_toggleAllModelsButton->setToolTip("<qt>Show every model in the results if any are hidden; otherwise hide every model. Benchmark participation stays unchanged.</qt>");
    modelTools->addWidget(m_toggleAllModelsButton);
    connect(m_toggleAllModelsButton, &QPushButton::clicked, this, [this]() {
        if (m_running || m_selectedVersions.empty())
            return;

        const bool include = std::any_of(m_selectedVersions.begin(), m_selectedVersions.end(),
            [](const BenchmarkVersionEntry& entry) { return !entry.included; });
        for (BenchmarkVersionEntry& entry : m_selectedVersions)
            entry.included = include;

        QueueSelectedTableRefresh();
    });

    m_makeAllButton = new QPushButton(this);
    m_makeAllButton->setText("Make All");
    m_makeAllButton->setObjectName("benchmarkMakeAll");
    m_makeAllButton->setToolTip("<qt>Set the next-run role for every model. Show checkboxes stay unchanged.</qt>");
    auto* makeAllMenu = new QMenu(m_makeAllButton);
    m_makeAllButton->setMenu(makeAllMenu);

    const QStringList roles = { "Target", "Opponent", "Results only" };
    for (int role = 0; role < roles.size(); ++role) {
        QAction* action = makeAllMenu->addAction(roles[role]);
        connect(action, &QAction::triggered, this, [this, role]() {
            if (m_running) return;

            for (BenchmarkVersionEntry& entry : m_selectedVersions) {
                entry.primary = role == 0;
                entry.resultsOnly = role == 2;
            }

            const bool preserveColumnWidths = true;
            PopulateSelectedTable(preserveColumnWidths);
            OnSetupChanged();
        });
    }

    modelTools->addWidget(m_makeAllButton);
    modelTools->addWidget(m_removeSelectedButton);
    modelTools->addWidget(m_clearSelectedButton);

    auto* details = new QCheckBox("Model details", this);
    details->setToolTip("Show training stage, branch, Agent Setup, and checkpoint folder columns.");
    details->setChecked(true);
    modelTools->addWidget(details);
    setupRoot->addWidget(m_modelControls);

    m_modelPickerDialog = new QDialog(this);
    m_modelPickerDialog->setObjectName("benchmarkModelPickerDialog");
    m_modelPickerDialog->setWindowTitle("Add benchmark models");
    m_modelPickerDialog->resize(1100, 650);

    auto* pickerRoot = new QVBoxLayout(m_modelPickerDialog);
    auto* pickerTabs = new QTabWidget(m_modelPickerDialog);
    auto* localModels = new QWidget(pickerTabs);
    auto* pickerLayout = new QVBoxLayout(localModels);
    auto* contextRow = new QHBoxLayout;
    contextRow->addWidget(new QLabel("Agent:", this));
    m_agentCombo = new AgentComboBox(this);
    m_agentCombo->setToolTip("<qt>Choose the bot family to browse. You can add models from several families to the same comparison.</qt>");
    contextRow->addWidget(m_agentCombo);
    contextRow->addWidget(new QLabel("Build:", this));
    m_buildCombo = new QComboBox(this);
    m_buildCombo->setToolTip("<qt>Filter available versions by the training build they belong to. This does not change models already in your comparison.</qt>");
    contextRow->addWidget(m_buildCombo);
    contextRow->addWidget(new QLabel("Training Stage:", this));
    m_stageCombo = new QComboBox(this);
    m_stageCombo->setToolTip("<qt>Filter versions by their training stage. This does not change the benchmark game mode or test setup.</qt>");
    contextRow->addWidget(m_stageCombo);
    contextRow->addWidget(new QLabel("Branch:", this));
    m_availableBranchCombo = new QComboBox(this);
    m_availableBranchCombo->setToolTip("Filter available versions by the training branch they belong to.");
    m_availableBranchCombo->addItem(AllBranchesLabel);
    contextRow->addWidget(m_availableBranchCombo);

    m_refreshVersionsButton = new QPushButton("Refresh Versions", this);
    m_refreshVersionsButton->setToolTip("Rescan this agent's saved versions and update the cached model catalog.");
    contextRow->addWidget(m_refreshVersionsButton);
    contextRow->addStretch();
    pickerLayout->addLayout(contextRow);


    auto* versionsGroup = new QGroupBox("Available Models", this);
    auto* versionsLayout = new QVBoxLayout(versionsGroup);
    auto* versionFilterRow = new QHBoxLayout;
    m_hideImportedModelsCheck = new QCheckBox("Hide Imported Models", versionsGroup);
    m_hideImportedModelsCheck->setToolTip("<qt>Hide imported checkpoints in this picker. Models already added to the comparison stay included.</qt>");
    m_availableSearchEdit = new QLineEdit(versionsGroup);
    m_availableSearchEdit->setPlaceholderText("Find a model by version, build, stage, branch, or notes...");
    m_availableSearchEdit->setClearButtonEnabled(true);
    m_availableSearchEdit->setToolTip("<qt>Filter this list by version, build, stage, branch, or notes. Add All Visible Models uses the filtered list.</qt>");
    versionFilterRow->addWidget(m_hideImportedModelsCheck);
    versionFilterRow->addSpacing(12);
    versionFilterRow->addWidget(new QLabel("Search:", versionsGroup));
    versionFilterRow->addWidget(m_availableSearchEdit, 1);
    versionFilterRow->addStretch();
    versionsLayout->addLayout(versionFilterRow);

    m_versionTable = new QTableWidget(0, 10, versionsGroup);
    m_versionTable->setHorizontalHeaderLabels({ "Use", "Setup", "Agent", "Build", "Segment", "Stage", "Branch", "Latest TS", "Rating", "Notes" });
    SetHeaderTooltips(m_versionTable, {
        "Checked means the model is already in the comparison. Click an unchecked row to add it.",
        "Hover over the icon to see whether saved Agent settings are available and how they compare with the current settings.",
        "The bot family this model uses.", "The training build this model belongs to.",
        "The saved training segment or version identifier.", "The training stage that produced this model.",
        "The training branch containing this version.", "Latest TS means the number of training timesteps reached by this checkpoint.",
        "A previously recorded skill rating for this model. This is separate from the benchmark Elo calculated in the results below.",
        "Notes saved with this model version. Search can match this text."
    });
    m_versionTable->setSelectionMode(QAbstractItemView::SingleSelection);
    m_versionTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_versionTable->setToolTip("Click a row to add that model. Checked models are already in your benchmark.");
    m_versionTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    auto* versionHeader = m_versionTable->horizontalHeader();
    versionHeader->setSectionsClickable(true);
    versionHeader->setSortIndicatorShown(false);
    versionHeader->setStretchLastSection(false);
    versionHeader->setSectionResizeMode(QHeaderView::Interactive);
    InstallCenteredCheckBoxDelegate(m_versionTable, VersionColumnUse);
    InstallCenteredIconDelegate(m_versionTable, VersionColumnSetup);
    versionsLayout->addWidget(m_versionTable);

    auto* versionButtonRow = new QHBoxLayout;
    m_addVersionsButton = new QPushButton("Add All Visible Models", versionsGroup);
    m_addVersionsButton->setToolTip("<qt>Add every available model currently shown by the picker filters. Models already in the comparison are not added twice.</qt>");
    m_addExternalFolderButton = new QPushButton("Add External Models...", versionsGroup);
    m_addExternalFolderButton->setToolTip("<qt>Choose a folder containing model checkpoints outside the normal training folders and add them using the selected Agent.</qt>");
    versionButtonRow->addWidget(m_addVersionsButton);
    versionButtonRow->addWidget(m_addExternalFolderButton);
    versionButtonRow->addStretch();
    versionsLayout->addLayout(versionButtonRow);

    m_catalogStatusLabel = new QLabel("Click a model row to add it to the benchmark.", versionsGroup);
    versionsLayout->addWidget(m_catalogStatusLabel);

    pickerLayout->addWidget(versionsGroup, 1);

    pickerTabs->addTab(localModels, "Local models");
    pickerTabs->setTabToolTip(0, "Browse checkpoint files available on this computer. Click a row to add a model for testing.");

    m_savedModelPicker = new QTableWidget(0, 3, pickerTabs);
    m_savedModelPicker->setHorizontalHeaderLabels({ "Saved model", "Version", "Checkpoint" });
    SetHeaderTooltips(m_savedModelPicker, {
        "A model with saved benchmark data in the active library. Click a row to add it.",
        "The version associated with these saved games.",
        "Whether the trained model files are available locally. Missing files allow saved-result comparisons but prevent new games."
    });
    m_savedModelPicker->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_savedModelPicker->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_savedModelPicker->horizontalHeader()->setStretchLastSection(true);
    pickerTabs->addTab(m_savedModelPicker, "Saved models · click to add");
    pickerTabs->setTabToolTip(1, "Browse models with results in the active library, including models whose local checkpoint files are unavailable.");
    m_savedModelPicker->setToolTip("<qt>Click a row to add this saved model. Without a local checkpoint it is added as Results only, so no new games run for it.</qt>");
    pickerRoot->addWidget(pickerTabs);

    auto* pickerButtons = new QDialogButtonBox(QDialogButtonBox::Close, m_modelPickerDialog);
    pickerRoot->addWidget(pickerButtons);
    connect(pickerButtons, &QDialogButtonBox::rejected, m_modelPickerDialog, &QDialog::reject);

    m_selectedTable = new QTableWidget(0, 12, versionsGroup);
    m_selectedTable->setHorizontalHeaderLabels({ "Show", "Next run", "Setup", "Agent", "Build", "Version", "Total Timesteps", "Last Trained", "Stage", "Branch", "Agent Setup", "Checkpoint Folder" });
    if (QTableWidgetItem* primaryHeader = m_selectedTable->horizontalHeaderItem(SelectedColumnPrimary))
        primaryHeader->setToolTip(NextRunTooltip);
    SetHeaderTooltips(m_selectedTable, {
        IncludeTooltip, NextRunTooltip,
        "Shows whether the model has a saved Agent setup and whether it differs from the current one. Hover over the row's icon for details.",
        "The bot family this model belongs to.", "The training build this model belongs to.",
        "The identifier of this saved model version.",
        "Total training timesteps reached by this checkpoint, including inherited training. These are not benchmark games. A dash means the count was not recorded.",
        "The recorded date and time this model was last trained, shown in your local time. A dash means no training date is available; import dates are not used.",
        "The training stage that produced this model.", "The training branch this version belongs to.",
        "Choose Trained to use the Agent settings saved with this model, or Current to use today's Agent settings. These control how the model runs, not the benchmark game target.",
        "The folder containing the model's trained weights. New games require these files; saved results can still be used without them."
    });
    m_selectedTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_selectedTable->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_selectedTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    auto* selectedHeader = m_selectedTable->horizontalHeader();
    selectedHeader->setSectionsClickable(true);
    selectedHeader->setSortIndicatorShown(false);
    selectedHeader->setStretchLastSection(false);
    selectedHeader->setSectionResizeMode(QHeaderView::Interactive);
    InstallCenteredCheckBoxDelegate(m_selectedTable, SelectedColumnUse);
    InstallCenteredIconDelegate(m_selectedTable, SelectedColumnSetup);
    m_selectedTable->setColumnWidth(SelectedColumnPrimary, 130);

    setupRoot->addWidget(m_selectedTable, 1);

    m_selectedTable->setObjectName("benchmarkComparisonModels");
    m_selectedTable->setMinimumHeight(130);
    m_selectedTable->setAlternatingRowColors(true);
    m_selectedTable->setWordWrap(false);
    m_selectedTable->setTextElideMode(Qt::ElideMiddle);
    m_selectedTable->verticalHeader()->hide();
    m_selectedTable->verticalHeader()->setDefaultSectionSize(32);

    connect(details, &QCheckBox::toggled, this, [this](bool show) {
        QHeaderView* header = m_selectedTable->horizontalHeader();
        header->setSectionResizeMode(SelectedColumnBuild, show ? QHeaderView::Interactive : QHeaderView::Stretch);
        for (int column : { SelectedColumnStage, SelectedColumnBranch, SelectedColumnAgentSetup, SelectedColumnCheckpoint })
            m_selectedTable->setColumnHidden(column, !show);
    });

    m_testSetupDialog = new QDialog(this);
    m_testSetupDialog->setObjectName("benchmarkTestSetupDialog");
    m_testSetupDialog->setWindowTitle("Benchmark test setup");
    m_testSetupDialog->resize(QSize(760, 560).boundedTo(screen()->availableGeometry().size() * 0.9));

    auto* dialogLayout = new QVBoxLayout(m_testSetupDialog);
    auto* setupHint = new QLabel("These settings are remembered and saved with your preset. Each run also keeps its own setup snapshot.", m_testSetupDialog);
    setupHint->setWordWrap(true);
    dialogLayout->addWidget(setupHint);

    auto* setupScroll = new QScrollArea(m_testSetupDialog);
    setupScroll->setWidgetResizable(true);
    setupScroll->setFrameShape(QFrame::NoFrame);

    auto* optionsGroup = new QGroupBox("Benchmark Options", this);
    auto* optionsLayout = new QFormLayout(optionsGroup);
    optionsLayout->setRowWrapPolicy(QFormLayout::WrapLongRows);

    m_engineCombo = new QComboBox(optionsGroup);
    m_engineCombo->setObjectName("modelBenchmarkEngine");
    for (const RocketSim::EngineInfo& engine : RocketSim::Engines()) {
        m_engineCombo->addItem(QString::fromStdString(engine.displayName), static_cast<int>(engine.id));
        if (engine.available)
            continue;

        const int index = m_engineCombo->count() - 1;
        if (auto* model = qobject_cast<QStandardItemModel*>(m_engineCombo->model()))
            model->item(index)->setEnabled(false);
        m_engineCombo->setItemData(index, QString::fromStdString(engine.unavailableReason), Qt::ToolTipRole);
    }
    m_engineCombo->setToolTip("<qt>Choose the physics engine used for new games. Compatibility below controls which simulators and environments count toward game targets and appear in results.</qt>");
    optionsLayout->addRow("Simulation Engine:", m_engineCombo);

    m_arenasSpin = new QSpinBox(optionsGroup);
    m_arenasSpin->setRange(1, std::numeric_limits<int>::max());
    m_arenasSpin->setValue(500);
    m_arenasSpin->setToolTip("<qt>Maximum number of games simulated at the same time. More arenas may improve speed but use more memory. This does not change the game target.</qt>");
    optionsLayout->addRow("Arenas:", m_arenasSpin);

    m_maxGamesSpin = new QSpinBox(optionsGroup);
    m_maxGamesSpin->setRange(1, 1000000);
    m_maxGamesSpin->setValue(500);
    m_maxGamesSpin->setToolTip("<qt>Games for each pair of models in each enabled mode, not a total across the whole test. Fill mode schedules only the games needed to reach the target after counting matching saved games. Additional mode schedules the specified number of new games per pair and mode. Completed draws count toward the target.</qt>");
    m_maxGamesLabel = new QLabel("Target Games per Matchup:", optionsGroup);
    optionsLayout->addRow(m_maxGamesLabel, m_maxGamesSpin);

    m_seedEdit = new QLineEdit(optionsGroup);
    m_seedEdit->setValidator(new UInt64Validator(m_seedEdit));
    m_seedEdit->setText("0");
    m_seedEdit->setToolTip("<qt>Controls the random starting sequence. Use 0 for a fresh seed each run, or enter a fixed number for reproducibility. Fixed-seed top-ups advance beyond saved starts so they do not replay them. The actual seed is recorded in the results.</qt>");
    optionsLayout->addRow("Seed (0 = auto):", m_seedEdit);

    auto* modeRow = new QWidget(optionsGroup);
    auto* modeLayout = new QHBoxLayout(modeRow);
    modeLayout->setContentsMargins(0, 0, 0, 0);
    modeRow->setToolTip("<qt>Choose the team sizes to test and include in combined results. Each enabled mode has its own game target. Models only run modes they support.</qt>");
    m_mode1v1Check = new QCheckBox("1v1", modeRow);
    m_mode1v1Check->setToolTip("<qt>One car per team. Count saved 1v1 games and schedule new 1v1 games toward a separate target.</qt>");
    m_mode2v2Check = new QCheckBox("2v2", modeRow);
    m_mode2v2Check->setToolTip("<qt>Two cars per team, all controlled by that team's model. Count and schedule 2v2 games separately from other modes.</qt>");
    m_mode3v3Check = new QCheckBox("3v3", modeRow);
    m_mode3v3Check->setToolTip("<qt>Three cars per team, all controlled by that team's model. Count and schedule 3v3 games separately from other modes.</qt>");
    m_mode1v1Check->setChecked(true);
    m_mode2v2Check->setChecked(true);
    m_mode3v3Check->setChecked(true);
    modeLayout->addWidget(m_mode1v1Check);
    modeLayout->addWidget(m_mode2v2Check);
    modeLayout->addWidget(m_mode3v3Check);
    optionsLayout->addRow("Modes:", modeRow);

    m_stochasticKickoffCheck = new QCheckBox("Stochastic kickoff inference", optionsGroup);
    m_stochasticKickoffCheck->setObjectName("modelBenchmarkStochasticKickoff");
    m_stochasticKickoffCheck->setToolTip(
        "<qt>Sample legal policy actions until the first car touches the ball, then use deterministic actions for the rest of the game.</qt>");
    optionsLayout->addRow(m_stochasticKickoffCheck);

    m_kickoffTemperatureSpin = new QDoubleSpinBox(optionsGroup);
    m_kickoffTemperatureSpin->setObjectName("modelBenchmarkKickoffTemperature");
    m_kickoffTemperatureSpin->setRange(0.01, 10.0);
    m_kickoffTemperatureSpin->setDecimals(2);
    m_kickoffTemperatureSpin->setSingleStep(0.05);
    m_kickoffTemperatureSpin->setValue(0.70);
    m_kickoffTemperatureSpin->setEnabled(false);
    m_kickoffTemperatureSpin->setToolTip(
        "<qt>Lower values favor the policy's preferred actions; higher values add variety. Applies only before the first ball touch.</qt>");
    optionsLayout->addRow("Kickoff temperature:", m_kickoffTemperatureSpin);

    m_scheduleModeCombo = new QComboBox(optionsGroup);
    m_scheduleModeCombo->addItems({ "Fill missing games to target", "Run new games" });
    m_scheduleModeCombo->setObjectName("benchmarkSchedulingMode");
    m_scheduleModeCombo->setToolTip("<qt>Fill missing games tops each matchup and mode up to a total that includes saved games. Run new games schedules a fresh batch even if enough saved games already exist.</qt>");
    m_scheduleModeCombo->setItemData(0, "<qt>Only schedule the difference between the target and matching saved games. Pairs already at or above the target run no new games but remain in the comparison. Early stopping is disabled.</qt>", Qt::ToolTipRole);
    m_scheduleModeCombo->setItemData(1, "<qt>Schedule the specified number of new games per matchup and mode, regardless of saved counts.</qt>", Qt::ToolTipRole);
    optionsLayout->insertRow(3, "Scheduling:", m_scheduleModeCombo);

    auto* compatibilityGroup = new QGroupBox("Compatibility", optionsGroup);
    auto* compatibilityLayout = new QVBoxLayout(compatibilityGroup);
    auto* compatibilityHelp = new QLabel("These simulators and environments count toward the game target and appear in results.", compatibilityGroup);
    compatibilityHelp->setWordWrap(true);
    compatibilityLayout->addWidget(compatibilityHelp);

    m_reuseSelectedEngineCheck = new QCheckBox("Use selected simulation engine and all its environments", compatibilityGroup);
    m_reuseSelectedEngineCheck->setObjectName("benchmarkReuseSelectedEngine");
    m_reuseSelectedEngineCheck->setChecked(true);
    m_reuseSelectedEngineCheck->setToolTip("<qt>Use the selected Simulation Engine and all of its environments, including future changes, for saved-game coverage and results. Turn this off to choose simulators and specific environments below.</qt>");
    compatibilityLayout->addWidget(m_reuseSelectedEngineCheck);

    for (int engine : { 2, 3 }) {
        auto* row = new QHBoxLayout;
        auto* check = new QCheckBox(m_engineCombo->itemText(m_engineCombo->findData(engine)), compatibilityGroup);
        check->setObjectName(QString("benchmarkReuseEngine%1").arg(engine));
        auto* environments = new QPushButton("All environments...", compatibilityGroup);
        environments->setObjectName(QString("benchmarkReuseEnvironments%1").arg(engine));
        environments->setToolTip("<qt>Choose which recorded environments from this simulator count toward the game target and appear in results. All environments includes future changes; a specific selection accepts only the environments you check.</qt>");

        m_reuseEngineChecks[engine] = check;
        m_reuseEnvironmentButtons[engine] = environments;
        row->addWidget(check);
        row->addStretch();
        row->addWidget(environments);
        compatibilityLayout->addLayout(row);
    }

    optionsLayout->addRow(compatibilityGroup);

    // Keep help available on labels as well as their controls.
    for (int row = 0; row < optionsLayout->rowCount(); ++row) {
        const QLayoutItem* field = optionsLayout->itemAt(row, QFormLayout::FieldRole);
        const QLayoutItem* label = optionsLayout->itemAt(row, QFormLayout::LabelRole);
        if (field && label && field->widget() && label->widget())
            label->widget()->setToolTip(field->widget()->toolTip());
    }

    setupScroll->setWidget(optionsGroup);
    dialogLayout->addWidget(setupScroll, 1);

    auto* setupButtons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, m_testSetupDialog);
    setupButtons->button(QDialogButtonBox::Ok)->setText("Apply setup");
    setupButtons->button(QDialogButtonBox::Ok)->setToolTip("<qt>Apply these settings and update the schedule and combined results. Use Save on the main screen to update a named preset.</qt>");
    setupButtons->button(QDialogButtonBox::Cancel)->setToolTip("Close Test Setup and restore the settings from before this dialog was opened.");
    dialogLayout->addWidget(setupButtons);
    connect(setupButtons, &QDialogButtonBox::accepted, m_testSetupDialog, &QDialog::accept);
    connect(setupButtons, &QDialogButtonBox::rejected, m_testSetupDialog, &QDialog::reject);

    auto* runRow = new QHBoxLayout;
    auto* preview = new QPushButton("Review Schedule…", this);
    preview->setToolTip("<qt>See saved and planned game counts for every eligible pair and mode before running. Covered pairs need no new games; blocked pairs need their checkpoint files or a different role.</qt>");
    runRow->addWidget(preview);

    m_scheduleSummaryLabel = new QLabel("Add models to build a schedule", this);
    m_scheduleSummaryLabel->setWordWrap(true);
    m_scheduleSummaryLabel->setMinimumWidth(0);
    runRow->addWidget(m_scheduleSummaryLabel, 1);

    m_runButton = new QPushButton("Run Benchmark", this);
    m_runButton->setObjectName("benchmarkRunButton");
    m_runButton->setDefault(false);
    m_runButton->setToolTip(RunBenchmarkTooltip);
    m_cancelButton = new QPushButton("Stop", this);
    m_cancelButton->setToolTip("<qt>Request a stop. Completed usable games returned by the runner are saved automatically; unfinished games do not count toward the target.</qt>");
    m_cancelButton->setEnabled(false);
    runRow->addWidget(m_runButton);
    runRow->addWidget(m_cancelButton);
    setupRoot->addLayout(runRow);

    m_scheduleTable = new QTableWidget(0, 7, this);
    m_scheduleTable->setHorizontalHeaderLabels({ "Model", "Opponent", "Modes", "Target Games", "Saved games", "Games needed", "Status" });
    SetHeaderTooltips(m_scheduleTable, {
        "The Target model in this matchup. When both models are Targets, the alphabetically earlier name appears here. Rows are sorted by Model, then Opponent.", "The other model in this matchup. This heading does not refer to its Next run role.",
        "Team sizes grouped by status for this model pair. Each enabled mode has a separate game target.",
        "Configured game target summed across the listed modes. In Additional mode this is the requested number of new games.",
        "Total completed saved games across the listed modes that match the current reuse settings. Draws count; unfinished games do not.",
        "Total games needed across the listed modes; only Scheduled rows contribute to the runnable total. In Fill mode this is the target minus saved games, with a minimum of zero.",
        "Scheduled means new games are planned. Covered means the saved count already meets the target. Checkpoint unavailable means new games need model files that are missing."
    });
    m_scheduleTable->horizontalHeader()->setDefaultAlignment(Qt::AlignCenter);
    m_scheduleTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_scheduleTable->setAlternatingRowColors(true);
    m_scheduleTable->verticalHeader()->hide();
    m_scheduleTable->hide();

    auto* resultArea = new QWidget(this);
    auto* resultRoot = new QVBoxLayout(resultArea);
    resultRoot->setContentsMargins(0, 0, 0, 0);

    auto* resultTools = new QHBoxLayout;
    auto* resultHeading = new QLabel("Combined results", this);
    resultHeading->setFont(headingFont);
    resultTools->addWidget(resultHeading);
    resultTools->addStretch();

    m_exportResultsButton = new QPushButton("Export to Excel", this);
    m_exportResultsButton->setToolTip("<qt>Export combined results to an Excel-readable file using the included models and current result filters.</qt>");
    m_saveResultsButton = new QPushButton("Retry Save", this);
    m_saveResultsButton->setToolTip("<qt>Try saving the latest completed result to the active library again after an automatic save failed. No games are rerun.</qt>");
    m_saveResultsButton->hide();
    resultTools->addWidget(m_saveResultsButton);
    resultTools->addWidget(m_exportResultsButton);
    resultRoot->addLayout(resultTools);

    m_savedResultsWidget = new ModelBenchmarkSavedResultsWidget(this);
    m_savedResultsWidget->SetWorkspaceMode();

    m_consoleText = new QPlainTextEdit(this);
    m_consoleText->setReadOnly(true);
    m_consoleText->setLineWrapMode(QPlainTextEdit::NoWrap);
    m_savedResultsWidget->AddRunConsole(m_consoleText);
    resultRoot->addWidget(m_savedResultsWidget, 1);

    m_mainSplitter->addWidget(m_setupContainer);
    m_mainSplitter->addWidget(resultArea);
    m_mainSplitter->setStretchFactor(0, 1);
    m_mainSplitter->setStretchFactor(1, 1);
    m_mainSplitter->setSizes({ 440, 440 });
    m_mainSplitter->handle(1)->setToolTip("Drag to give more space to the model list or the results.");
    root->addWidget(m_mainSplitter, 1);
    m_progressPanel = new RunProgressPanel(
        this,
        RunProgressPanel::ProgressPosition::AfterStatus);
    m_progressPanel->SetConsoleVisible(false);
    m_progressPanel->SetStatusFormatter([this](double elapsedSeconds) {
        return BuildProgressStatus(elapsedSeconds);
    });
    m_progressPanel->SetDetails({});
    m_progressPanel->SetDetailsToolTip(
        "Unscheduled games have not been created by the scheduler yet. "
        "Pending games are fully specified and waiting for an arena. "
        "Active games are currently assigned to arenas; completed games have finished simulation. "
        "Arenas C/R/R means newly created arenas, arenas rebuilt for a different configuration, "
        "and compatible arenas reused without rebuilding. "
        "GPU A/R is PyTorch allocated/reserved VRAM, and Graphs is the number of cached CUDA graphs. "
        "If one CUDA OOM is recovered, graph inference is disabled and the run continues eagerly. "
        "Runs finish at their scheduled game targets unless stopped manually.");

    for (QLabel* label : m_progressPanel->findChildren<QLabel*>()) {
        label->setWordWrap(true);
        label->setMinimumWidth(0);
    }
    root->addWidget(m_progressPanel);

    m_workspaceRefreshTimer = new QTimer(this);
    m_workspaceRefreshTimer->setSingleShot(true);
    m_workspaceRefreshTimer->setInterval(250);
    connect(m_workspaceRefreshTimer, &QTimer::timeout, this, &ModelBenchmarkWidget::RefreshWorkspace);

    connect(addModels, &QPushButton::clicked, this, &ModelBenchmarkWidget::ShowModelPicker);
    connect(m_presetCombo, &QComboBox::activated, this, &ModelBenchmarkWidget::LoadWorkspacePreset);
    connect(newPreset, &QPushButton::clicked, this, &ModelBenchmarkWidget::NewWorkspacePreset);
    connect(savePreset, &QPushButton::clicked, this, [this]() {
        const bool saveAsNewPreset = false;
        SaveWorkspacePreset(saveAsNewPreset);
    });
    connect(m_renamePresetButton, &QPushButton::clicked, this, &ModelBenchmarkWidget::RenameWorkspacePreset);
    connect(saveAs, &QPushButton::clicked, this, [this]() {
        const bool saveAsNewPreset = true;
        SaveWorkspacePreset(saveAsNewPreset);
    });
    connect(previousRun, &QPushButton::clicked, this, &ModelBenchmarkWidget::LoadPreviousRun);

    QAction* openLibrary = libraryMenu->addAction("Open library…");
    openLibrary->setToolTip("Switch to an existing database of benchmark results and presets.");
    connect(openLibrary, &QAction::triggered, this, [this]() {
        const bool createNew = false;
        m_savedResultsWidget->ManageLibrary(createNew);
    });

    QAction* newLibrary = libraryMenu->addAction("New library…");
    newLibrary->setToolTip("Create or choose a library for a separate collection of results and presets.");
    connect(newLibrary, &QAction::triggered, this, [this]() {
        const bool createNew = true;
        m_savedResultsWidget->ManageLibrary(createNew);
    });

    QAction* importResults = libraryMenu->addAction("Import Excel results…");
    importResults->setToolTip("Import a supported benchmark export into the active saved-results library.");
    connect(importResults, &QAction::triggered, this, [this]() { m_savedResultsWidget->ImportSavedResults(); });

    QAction* legacyComparison = libraryMenu->addAction("Load older comparison set…");
    legacyComparison->setToolTip("<qt>Load a comparison saved by the earlier interface. This restores its model selection while keeping the current test setup.</qt>");
    connect(legacyComparison, &QAction::triggered, this, &ModelBenchmarkWidget::LoadLegacyComparison);

    connect(m_savedResultsWidget, &ModelBenchmarkSavedResultsWidget::LibraryDataChanged, this, [this]() {
        ++m_workspaceGeneration;
        m_coverageDirty = true;
        m_workspaceRefreshTimer->start();
        if (!m_running) {
            m_scheduleSummaryLabel->setText("Updating schedule…");
            m_runButton->setEnabled(false);
        }
    });
    connect(m_savedModelPicker, &QTableWidget::cellClicked, this, [this](int row, int) {
        if (row >= 0 && row < (int)m_libraryModels.size()) {
            const bool preserveColumnWidths = true;
            AddSavedModel(m_libraryModels[row]);
            SortSelectedVersions();
            PopulateSelectedTable(preserveColumnWidths);
            OnSetupChanged();
        }
    });
    connect(preview, &QPushButton::clicked, this, [this]() {
        try {
            m_previewAfterWorkspaceRefresh = true;
            RefreshWorkspace();
        } catch (const std::exception& error) {
            QMessageBox::warning(this, "Unable to prepare schedule", QString::fromStdString(error.what()));
        }
    });
    connect(m_reuseSelectedEngineCheck, &QCheckBox::toggled, this, [this]() {
        UpdateCompatibilityControls();
        OnSetupChanged();
    });
    connect(m_engineCombo, &QComboBox::currentIndexChanged, this, [this]() { UpdateCompatibilityControls(); });
    for (const auto& [engine, check] : m_reuseEngineChecks) {
        connect(check, &QCheckBox::toggled, this, [this]() {
            UpdateCompatibilityControls();
            OnSetupChanged();
        });
        connect(m_reuseEnvironmentButtons.at(engine), &QPushButton::clicked, this, [this, engine]() { ShowCompatibilityEnvironments(engine); });
    }

    UpdateCompatibilityControls();
    connect(m_scheduleModeCombo, &QComboBox::currentIndexChanged, this, [this]() {
        UpdateInputState();
        OnSetupChanged();
    });

    {
        const std::string saved = RocketForgeTraining::LoadLastSelectedAgent();
        if (!saved.empty()) {
            const int agentIndex = m_agentCombo->findText(QString::fromStdString(saved));
            if (agentIndex >= 0)
                m_agentCombo->setCurrentIndex(agentIndex);
        }
    }

    RefreshBuildSelector();
    RefreshStageSelector();

    connect(m_refreshVersionsButton, &QPushButton::clicked, this, &ModelBenchmarkWidget::RefreshVersions);
    connect(m_agentCombo, &QComboBox::currentTextChanged, this, &ModelBenchmarkWidget::OnContextChanged);
    connect(m_buildCombo, &QComboBox::currentTextChanged, this, &ModelBenchmarkWidget::OnContextChanged);
    connect(m_stageCombo, &QComboBox::currentTextChanged, this, &ModelBenchmarkWidget::OnContextChanged);
    connect(m_hideImportedModelsCheck, &QCheckBox::toggled, this, [this](bool) {
        ApplyAvailableRowFilters();
        OnSetupChanged();
    });
    connect(m_availableBranchCombo, &QComboBox::currentTextChanged, this, [this](const QString&) { ApplyAvailableRowFilters(); });
    connect(m_availableSearchEdit, &QLineEdit::textChanged, this, [this](const QString&) { ApplyAvailableRowFilters(); });
    TableSortController::Config availableConfig;
    availableConfig.columnCount = BenchSortCount;
    availableConfig.columnsInUiOrder = AvailableUiOrder;
    availableConfig.defaultPriorityColumn = BenchSortBuild;
    availableConfig.rowKey = [this](int row) { return MakeEntryKey(m_availableVersions[row]); };
    availableConfig.compareColumn = [this](int lhs, int rhs, int col) {
        return CompareBenchmarkColumn(lhs, rhs, col, /*selectedTable*/ false);
    };
    m_availableSort.Configure(availableConfig);
    m_availableSort.AttachHeader(m_versionTable, [this]() {
        SortAvailableVersions();
        PopulateAvailableTable(true);
    });

    TableSortController::Config selectedConfig = availableConfig;
    selectedConfig.columnsInUiOrder = SelectedUiOrder;
    selectedConfig.rowKey = [this](int row) { return MakeEntryKey(m_selectedVersions[row]); };
    selectedConfig.compareColumn = [this](int lhs, int rhs, int col) {
        return CompareBenchmarkColumn(lhs, rhs, col, /*selectedTable*/ true);
    };
    m_selectedSort.Configure(selectedConfig);
    m_selectedSort.AttachHeader(m_selectedTable, [this]() {
        SortSelectedVersions();
        PopulateSelectedTable(true);
        OnSetupChanged();
    });

    // Agent ties fall through to the existing Build, then Version priorities.
    m_selectedSort.SetSort(BenchSortAgent, Qt::AscendingOrder);
    connect(m_selectedTable, &QTableWidget::itemChanged, this, [this](QTableWidgetItem* item) {
        if (!item || item->column() != SelectedColumnUse)
            return;
        const int row = item->row();
        if (row < 0 || row >= (int)m_selectedVersions.size())
            return;
        m_selectedVersions[row].included = item->checkState() == Qt::Checked;
        QueueSelectedTableRefresh();
    });
    connect(m_versionTable, &QTableWidget::cellClicked, this, [this](int row, int) {
        AddAvailableVersion(row);
    });
    connect(m_versionTable, &QTableWidget::cellActivated, this, [this](int row, int) {
        AddAvailableVersion(row);
    });
    connect(m_versionTable, &QTableWidget::itemChanged, this, [this](QTableWidgetItem* item) {
        if (item && item->column() == VersionColumnUse && item->checkState() == Qt::Checked)
            AddAvailableVersion(item->row());
    });
    connect(m_addVersionsButton, &QPushButton::clicked, this, &ModelBenchmarkWidget::OnAddVersionsClicked);
    connect(m_addExternalFolderButton, &QPushButton::clicked, this, &ModelBenchmarkWidget::OnAddExternalFolderClicked);
    connect(m_removeSelectedButton, &QPushButton::clicked, this, &ModelBenchmarkWidget::OnRemoveSelectedClicked);
    connect(m_clearSelectedButton, &QPushButton::clicked, this, &ModelBenchmarkWidget::OnClearSelectedClicked);
    connect(m_toggleSetupButton, &QPushButton::clicked, this, &ModelBenchmarkWidget::OnToggleSetupClicked);
    connect(m_runButton, &QPushButton::clicked, this, &ModelBenchmarkWidget::OnRunClicked);
    connect(m_cancelButton, &QPushButton::clicked, this, &ModelBenchmarkWidget::OnCancelClicked);
    connect(m_exportResultsButton, &QPushButton::clicked, this, [this]() {
        m_savedResultsWidget->ExportDisplayedResults();
    });
    connect(m_saveResultsButton, &QPushButton::clicked, this, &ModelBenchmarkWidget::OnSaveResultsClicked);
    connect(m_savedResultsWidget, &ModelBenchmarkSavedResultsWidget::LibraryPathChanged, this, [this]() {
        ++m_workspaceGeneration;
        m_runAfterWorkspaceRefresh = false;
        m_lastResultSavedToActiveLibrary = false;
        m_coverageDirty = true;
        m_savedPresetJson.clear();
        m_presetName.clear();
        for (BenchmarkVersionEntry& entry : m_selectedVersions) entry.libraryModelId = -1;

        RefreshPresetList();
        UpdateWorkspaceSummary();
        if (m_saveResultsButton)
            m_saveResultsButton->setEnabled(m_hasLastResult && !m_running);
        OnSetupChanged();
    });
    connect(m_savedResultsWidget, &ModelBenchmarkSavedResultsWidget::ExportRequested, this,
        [this](ModelBenchmarkLibraryAggregate aggregate) {
            const RocketForgeTraining::ModelBenchmarkResult previous = m_lastResult;
            const bool previousHasResult = m_hasLastResult;
            const QString previousRunId = m_lastResultRunId;
            const std::vector<BenchmarkVersionEntry> previousRunVersions = m_lastRunVersions;
            const std::vector<ModelBenchmarkLibraryModel> previousRunLibraryModels = m_lastRunLibraryModels;
            m_lastResult = aggregate.result;
            m_exportEnvironmentCoverage = std::move(aggregate);
            m_hasLastResult = true;
            m_lastResultRunId.clear();
            m_lastRunVersions.clear();
            m_lastRunLibraryModels.clear();
            OnExportResultsClicked();
            m_exportEnvironmentCoverage = {};
            m_lastResult = previous;
            m_hasLastResult = previousHasResult;
            m_lastResultRunId = previousRunId;
            m_lastRunVersions = previousRunVersions;
            m_lastRunLibraryModels = previousRunLibraryModels;
        });
    connect(m_arenasSpin, qOverload<int>(&QSpinBox::valueChanged), this, [this](int) { OnSetupChanged(); });
    connect(m_maxGamesSpin, qOverload<int>(&QSpinBox::valueChanged), this, [this](int) { OnSetupChanged(); });
    connect(m_seedEdit, &QLineEdit::editingFinished, this, [this]() { OnSetupChanged(); });
    connect(m_mode1v1Check, &QCheckBox::toggled, this, [this](bool) { OnSetupChanged(); });
    connect(m_mode2v2Check, &QCheckBox::toggled, this, [this](bool) { OnSetupChanged(); });
    connect(m_mode3v3Check, &QCheckBox::toggled, this, [this](bool) { OnSetupChanged(); });
    connect(m_stochasticKickoffCheck, &QCheckBox::toggled, this, [this](bool) {
        m_kickoffTemperatureSpin->setEnabled(!m_running && m_stochasticKickoffCheck->isChecked());
        OnSetupChanged();
    });
    connect(m_kickoffTemperatureSpin, qOverload<double>(&QDoubleSpinBox::valueChanged),
        this, [this](double) { OnSetupChanged(); });
    connect(m_engineCombo, &QComboBox::currentIndexChanged, this, [this](int) { OnSetupChanged(); });

    m_workspacePollTimer = new QTimer(this);
    m_workspacePollTimer->setInterval(25);
    connect(m_workspacePollTimer, &QTimer::timeout, this, &ModelBenchmarkWidget::FinishWorkspaceRefresh);

    m_catalogPollTimer = new QTimer(this);
    m_catalogPollTimer->setInterval(50);
    connect(m_catalogPollTimer, &QTimer::timeout, this, &ModelBenchmarkWidget::FinishVersionCatalogScan);

    m_setupSaveTimer = new QTimer(this);
    m_setupSaveTimer->setSingleShot(true);
    m_setupSaveTimer->setInterval(250);
    connect(m_setupSaveTimer, &QTimer::timeout, this, &ModelBenchmarkWidget::SavePersistedSetup);

    connect(&ModelCatalog::Instance(), &ModelCatalog::Changed, this, [this](const QString& agent) {
        m_versionCatalogs.erase(agent);

        const bool refresh = true;
        if (agent == CurrentAgentName()) EnsureVersionCatalog(refresh);
    });

    if (m_persistSetup) {
        m_initializationPending = true;
        setEnabled(false);
        QTimer::singleShot(250, this, [this]() {
            if (!m_initializationPending) return;
            ReloadPersistedSetup();
            RefreshPresetList();
            RefreshWorkspace();
        });
    }

    UpdateSectionToggleText();
    UpdateInputState();
}

ModelBenchmarkWidget::~ModelBenchmarkWidget()
{
    if (m_setupSaveTimer->isActive())
        SavePersistedSetup();

    if (m_worker)
        m_worker->Cancel();
    m_workerHost.Stop();
}

bool ModelBenchmarkWidget::IsRunning() const
{
    return m_running;
}

void ModelBenchmarkWidget::ReloadPersistedSetup()
{
    m_initializationPending = false;
    LoadPersistedSetup();
    setEnabled(true);
}

void ModelBenchmarkWidget::StartPersistedBenchmark()
{
    ReloadPersistedSetup();
    OnRunClicked();
}

void ModelBenchmarkWidget::SetTrainingActive(bool active)
{
    m_trainingActive = active;
    if (m_runButton) {
        m_runButton->setEnabled(!m_running && !m_trainingActive && !m_refreshingWorkspace && !m_coverageDirty);
        m_runButton->setToolTip(m_trainingActive
            ? "Training is running. Stop training before starting a model benchmark."
            : RunBenchmarkTooltip);
    }
}

QString ModelBenchmarkWidget::CurrentAgentName() const
{
    return m_agentCombo ? m_agentCombo->currentText() : QString();
}

QString ModelBenchmarkWidget::CurrentBuildName() const
{
    return m_buildCombo ? m_buildCombo->currentText() : QString(AllBuildsLabel);
}

QString ModelBenchmarkWidget::CurrentStageName() const
{
    return m_stageCombo ? m_stageCombo->currentText() : QString();
}

QString ModelBenchmarkWidget::CurrentAvailableBranchName() const
{
    return m_availableBranchCombo ? m_availableBranchCombo->currentText() : QString(AllBranchesLabel);
}

void ModelBenchmarkWidget::RefreshBuildSelector()
{
    if (!m_buildCombo)
        return;

    const QString previousBuild = m_buildCombo->currentText();
    QSignalBlocker blocker(m_buildCombo);
    m_buildCombo->clear();
    m_buildCombo->addItem(AllBuildsLabel);

    const auto catalog = m_versionCatalogs.find(CurrentAgentName());
    if (catalog != m_versionCatalogs.end()) {
        for (const std::string& build : catalog->second.builds)
            m_buildCombo->addItem(QString::fromStdString(build));
    }

    const int previousIndex = m_buildCombo->findText(previousBuild);
    m_buildCombo->setCurrentIndex(previousIndex >= 0 ? previousIndex : 0);
}

void ModelBenchmarkWidget::RefreshStageSelector()
{
    if (!m_stageCombo)
        return;

    const QString previousStage = m_stageCombo->currentText();
    const std::string buildFilter = CurrentBuildName() == AllBuildsLabel
        ? std::string()
        : CurrentBuildName().toStdString();

    QSignalBlocker blocker(m_stageCombo);
    m_stageCombo->clear();
    m_stageCombo->addItem(AllStagesLabel);

    std::set<std::string> stages;
    const auto catalog = m_versionCatalogs.find(CurrentAgentName());
    if (catalog != m_versionCatalogs.end()) {
        for (const BenchmarkVersionEntry& entry : catalog->second.entries) {
            if (entry.sourceLabel == "RocketForge" && !entry.stageName.empty()
                && (buildFilter.empty() || entry.buildName == buildFilter))
                stages.insert(entry.stageName);
        }
    }

    for (const std::string& stage : stages)
        m_stageCombo->addItem(QString::fromStdString(stage));

    const int previousIndex = m_stageCombo->findText(previousStage);
    m_stageCombo->setCurrentIndex(previousIndex >= 0 ? previousIndex : 0);
}

void ModelBenchmarkWidget::RefreshAvailableBranchFilter()
{
    if (!m_availableBranchCombo)
        return;

    const QString previousBranch = m_availableBranchCombo->currentText();
    std::set<std::string> branches;
    for (const BenchmarkVersionEntry& entry : m_availableVersions)
        branches.insert(BranchNameOrDefault(entry.version.branchName));

    QSignalBlocker blocker(m_availableBranchCombo);
    m_availableBranchCombo->clear();
    m_availableBranchCombo->addItem(AllBranchesLabel);
    for (const std::string& branch : branches)
        m_availableBranchCombo->addItem(QString::fromStdString(branch));

    const int previousIndex = m_availableBranchCombo->findText(previousBranch);
    m_availableBranchCombo->setCurrentIndex(previousIndex >= 0 ? previousIndex : 0);
}

void ModelBenchmarkWidget::RefreshVersions()
{
    const bool refresh = true;
    if (!m_running && !m_loadingPersistedSetup)
        EnsureVersionCatalog(refresh);
}

void ModelBenchmarkWidget::EnsureVersionCatalog(bool refresh)
{
    const QString agent = CurrentAgentName();
    if (agent.isEmpty())
        return;

    if (!refresh && m_versionCatalogs.contains(agent)) {
        ShowVersionCatalog();
        return;
    }

    if (m_displayedCatalogAgent != agent) {
        const bool preserveColumnWidths = true;
        m_availableVersions.clear();
        m_displayedCatalogAgent.clear();
        PopulateAvailableTable(preserveColumnWidths);
    }

    m_catalogStatusLabel->setText("Loading model catalog for " + agent + "...");
    if (m_catalogScan.valid()) {
        m_catalogRefreshPending |= refresh;
        return;
    }

    m_scanningAgent = agent;
    m_refreshVersionsButton->setEnabled(false);

    // The worker owns its inputs/results and never touches widgets or the active benchmark.
    try {
        const std::shared_ptr<const ModelCatalogSnapshot> snapshot = ModelCatalog::Instance().Get(agent);
        if (refresh) ModelCatalog::Instance().Refresh(agent);
        m_catalogScan = std::async(std::launch::async, &ModelBenchmarkWidget::ScanVersionCatalog, agent.toStdString(), snapshot);
        m_catalogPollTimer->start();
    } catch (const std::exception& e) {
        m_refreshVersionsButton->setEnabled(!m_running);
        m_catalogStatusLabel->setText("Unable to start discovery. Click Refresh Versions to retry.");
        AppendConsoleLine("Unable to start model discovery: " + QString::fromUtf8(e.what()));
    }
}

void ModelBenchmarkWidget::FinishVersionCatalogScan()
{
    if (!m_catalogScan.valid() || m_catalogScan.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
        return;

    m_catalogPollTimer->stop();
    const QString scannedAgent = m_scanningAgent;
    try {
        m_versionCatalogs[scannedAgent] = m_catalogScan.get();
        if (CurrentAgentName() == scannedAgent) {
            m_displayedCatalogAgent.clear();
            ShowVersionCatalog();
            if (!m_running) {
                // Keep selected checkpoints fixed; refresh only their cached setup status.
                std::map<std::string, const BenchmarkVersionEntry*> entries;
                for (const BenchmarkVersionEntry& entry : m_availableVersions)
                    entries.emplace(MakeEntryKey(entry), &entry);

                for (BenchmarkVersionEntry& selected : m_selectedVersions) {
                    const auto found = entries.find(MakeEntryKey(selected));
                    if (found == entries.end())
                        continue;

                    selected.trainingSetupPath = found->second->trainingSetupPath;
                    selected.hasSavedAgentSetup = found->second->hasSavedAgentSetup;
                    selected.agentSetupDiffs = found->second->agentSetupDiffs;
                }

                QueueSelectedTableRefresh();
            }
        }
    } catch (const std::exception& e) {
        m_catalogStatusLabel->setText("Catalog refresh failed. Click Refresh Versions to retry.");
        AppendConsoleLine("Unable to refresh model catalog: " + QString::fromUtf8(e.what()));
    }

    m_refreshVersionsButton->setEnabled(!m_running);

    // Switching agents during discovery cannot publish the old agent's rows into the new picker.
    if (m_catalogRefreshPending) {
        const bool refresh = true;
        m_catalogRefreshPending = false;
        EnsureVersionCatalog(refresh);
    } else if (CurrentAgentName() != scannedAgent) {
        EnsureVersionCatalog();
    }
}

void ModelBenchmarkWidget::ShowVersionCatalog()
{
    const auto catalog = m_versionCatalogs.find(CurrentAgentName());
    if (catalog == m_versionCatalogs.end())
        return;

    RefreshBuildSelector();
    RefreshStageSelector();
    if (!m_pendingStageFilter.isEmpty()) {
        QSignalBlocker blocker(m_stageCombo);
        const int index = m_stageCombo->findText(m_pendingStageFilter);
        m_stageCombo->setCurrentIndex(index >= 0 ? index : 0);
        m_pendingStageFilter.clear();
    }

    if (m_displayedCatalogAgent != CurrentAgentName()) {
        m_availableVersions = catalog->second.entries;
        m_displayedCatalogAgent = CurrentAgentName();
        SortAvailableVersions();
        RefreshAvailableBranchFilter();
        PopulateAvailableTable(m_availableTableColumnsInitialized);
        RefreshSavedKnownModels();
    } else {
        ApplyAvailableRowFilters();
    }
}

ModelBenchmarkWidget::VersionCatalog ModelBenchmarkWidget::ScanVersionCatalog(
    const std::string& agentName, std::shared_ptr<const ModelCatalogSnapshot> snapshot)
{
    VersionCatalog catalog;
    for (const RocketForgeTraining::BuildSummary& build : snapshot->storage.builds) catalog.builds.push_back(build.buildName);

    std::optional<RocketForge::AgentSetupSnapshot> currentSetup;
    std::string setupError;
    try {
        std::unique_ptr<Agent> agent = AgentRegistry::Create(agentName);
        if (!agent) throw std::runtime_error("AgentRegistry returned null");
        currentSetup = RocketForge::CaptureAgentSetupSnapshot(*agent);
    } catch (const std::exception& error) {
        setupError = error.what();
    }

    for (const CatalogModel& model : snapshot->models) {
        if (!model.version.available) continue;

        BenchmarkVersionEntry entry;
        entry.agentName = agentName;
        entry.sourceLabel = model.sourceLabel;
        entry.buildName = model.version.buildName;
        entry.stageName = model.sourceLabel == "Imported Model" ? std::string() : model.version.stageName;
        if (entry.buildName.empty() && model.sourceLabel == "Imported Model") entry.buildName = "Imported Model";
        entry.version = model.version;
        entry.checkpointPath = model.checkpointPath;
        entry.trainingSetupPath = model.setupPath;

        // modelCount gates local availability here, but the catalog counts only durable checkpoints.
        if (entry.version.latestIsAutosave && entry.version.modelCount == 0) entry.version.modelCount = 1;

        const std::optional<RocketForge::AgentSetupSnapshot> saved = RocketForge::AgentSetupSnapshotFromJson(model.agentSetup);
        entry.hasSavedAgentSetup = saved.has_value();
        if (saved && currentSetup) {
            entry.agentSetupDiffs = RocketForge::DiffAgentSetupSnapshots(*saved, *currentSetup);
        } else if (saved) {
            entry.agentSetupDiffs.push_back({"Current Agent", "saved setup", setupError});
        }

        if (!saved) entry.agentSetupSource = RocketForgeTraining::ModelBenchmarkAgentSetupSource::CurrentAgent;
        catalog.entries.push_back(std::move(entry));
    }

    return catalog;
}

void ModelBenchmarkWidget::OnContextChanged()
{
    if (m_loadingPersistedSetup)
        return;
    if (sender() == m_stageCombo || sender() == m_agentCombo)
        m_pendingStageFilter.clear();
    if (CurrentAgentName() != m_loadedAgentName) {
        m_loadedAgentName = CurrentAgentName();
        if (m_persistSetup)
            RocketForgeTraining::SaveLastSelectedAgent(m_loadedAgentName.toStdString());
        RefreshBuildSelector();
    }
    if (sender() == m_agentCombo || sender() == m_buildCombo)
        RefreshStageSelector();
    EnsureVersionCatalog();
    OnSetupChanged();
}

RocketForgeTraining::ModelBenchmarkSettings ModelBenchmarkWidget::BuildPersistedSetup() const
{
    RocketForgeTraining::ModelBenchmarkSettings settings;
    const QString stage = m_pendingStageFilter.isEmpty() ? CurrentStageName() : m_pendingStageFilter;
    settings.stageFilter = stage == AllStagesLabel ? std::string() : stage.toStdString();
    settings.hideImportedModels = m_hideImportedModelsCheck && m_hideImportedModelsCheck->isChecked();
    if (m_savedResultsWidget)
        settings.libraryPath = m_savedResultsWidget->LibraryPath();
    settings.engineVersion = m_engineCombo->currentData().toInt();
    settings.numArenas = m_arenasSpin->value();
    settings.maxGamesPerMode = m_maxGamesSpin->value();
    settings.fillMissingGames = m_scheduleModeCombo->currentIndex() == 0;

    // Keep older readers from restoring the retired role override.
    settings.allMatchups = false;
    settings.matchSavedConditions = true;
    settings.reuseSelectedEngine = m_reuseSelectedEngineCheck->isChecked();
    for (const auto& [engine, check] : m_reuseEngineChecks)
        if (!settings.reuseSelectedEngine && check->isChecked()) settings.reuseEngineVersions.push_back(engine);

    settings.reuseEnvironmentKeys = m_reuseEnvironmentKeys;
    settings.ratingTargetsOnly = false;
    settings.presetName = m_presetName.toStdString();
    settings.seed = ParseBenchmarkSeed(m_seedEdit);
    settings.run1v1 = m_mode1v1Check->isChecked();
    settings.run2v2 = m_mode2v2Check->isChecked();
    settings.run3v3 = m_mode3v3Check->isChecked();
    settings.stochasticKickoff = m_stochasticKickoffCheck->isChecked();
    settings.kickoffTemperature = (float)m_kickoffTemperatureSpin->value();
    settings.betterConfidence = 0.95f;
    settings.stopWhenRankingKnown = false;
    settings.eloRangeConfidence = 0.95f;
    settings.maxEloUncertaintyRange = 50.0f;
    settings.runUntilMaxGames = true;
    settings.collectOptionalStats = true;

    for (const BenchmarkVersionEntry& entry : m_selectedVersions) {
        RocketForgeTraining::ModelBenchmarkSelectedModelSettings model;
        model.agentName = entry.agentName;
        model.buildName = entry.buildName;
        model.sourceLabel = entry.sourceLabel;
        model.stageName = entry.stageName;
        model.versionId = entry.version.id;
        model.checkpointPath = entry.checkpointPath;
        model.trainingSetupPath = entry.trainingSetupPath;
        model.agentSetupSource = entry.agentSetupSource;
        model.primary = entry.primary;
        model.included = entry.included;
        model.resultsOnly = entry.resultsOnly;
        model.libraryModelId = entry.libraryModelId;
        model.fingerprint = entry.fingerprint.toStdString();
        model.agentSetupJson = entry.agentSetupJson;
        model.colorIndex = entry.colorIndex;
        model.latestTimesteps = entry.version.latestTimesteps;
        model.lastTrainedAt = entry.version.lastTrainedAt;
        model.notes = entry.version.notes;
        model.skillRating = entry.version.skillRating;
        model.hasSkillRating = entry.version.hasSkillRating;
        settings.selectedModels.push_back(std::move(model));
    }

    return settings;
}

void ModelBenchmarkWidget::ApplyPersistedSetup(const RocketForgeTraining::ModelBenchmarkSettings& settings)
{
    if (m_savedResultsWidget) {
        const std::filesystem::path defaultLibrary = RocketForgeTraining::DefaultModelBenchmarkLibraryPath();
        const std::filesystem::path libraryPath = settings.libraryPath.empty() ? defaultLibrary : settings.libraryPath;
        const bool showErrors = true;
        if (m_savedResultsWidget->LibraryPath() != libraryPath)
            m_savedResultsWidget->OpenLibrary(libraryPath, showErrors);
    }

    if (m_stageCombo) {
        const QString stageText = settings.stageFilter.empty()
            ? QString(AllStagesLabel)
            : QString::fromStdString(settings.stageFilter);
        m_pendingStageFilter = stageText;
        const int stageIdx = m_stageCombo->findText(stageText);
        m_stageCombo->setCurrentIndex(stageIdx >= 0 ? stageIdx : 0);
    }

    m_hideImportedModelsCheck->setChecked(settings.hideImportedModels);
    m_engineCombo->setCurrentIndex(m_engineCombo->findData(settings.engineVersion));
    m_arenasSpin->setValue(settings.numArenas);
    m_maxGamesSpin->setValue(settings.maxGamesPerMode);
    m_scheduleModeCombo->setCurrentIndex(settings.fillMissingGames ? 0 : 1);

    {
        const QSignalBlocker blocker(m_reuseSelectedEngineCheck);
        m_reuseSelectedEngineCheck->setChecked(settings.reuseSelectedEngine);
    }

    for (const auto& [engine, check] : m_reuseEngineChecks) {
        const QSignalBlocker blocker(check);
        check->setChecked(std::find(settings.reuseEngineVersions.begin(), settings.reuseEngineVersions.end(), engine) != settings.reuseEngineVersions.end());
    }

    m_reuseEnvironmentKeys = settings.reuseEnvironmentKeys;
    UpdateCompatibilityControls();
    m_presetName = QString::fromStdString(settings.presetName);
    m_seedEdit->setText(QString::number((qulonglong)settings.seed));
    m_mode1v1Check->setChecked(settings.run1v1);
    m_mode2v2Check->setChecked(settings.run2v2);
    m_mode3v3Check->setChecked(settings.run3v3);
    m_stochasticKickoffCheck->setChecked(settings.stochasticKickoff);
    m_kickoffTemperatureSpin->setValue(settings.kickoffTemperature);

    m_selectedVersions.clear();
    for (const auto& model : settings.selectedModels) {
        BenchmarkVersionEntry entry = {};
        entry.agentName = model.agentName.empty() ? CurrentAgentName().toStdString() : model.agentName;
        entry.buildName = model.buildName.empty() ? "N/A" : model.buildName;
        entry.stageName = model.stageName;
        entry.sourceLabel = model.sourceLabel.empty() ? "RocketForge" : model.sourceLabel;
        entry.version.id = model.versionId;
        GGL::SegmentVersionNumber parsedVersion;
        if (GGL::TryParseSegmentVersionId(model.versionId, parsedVersion)) {
            entry.version.major = parsedVersion.major;
            entry.version.minor = parsedVersion.minor;
        }
        entry.version.latestTimesteps = model.latestTimesteps;
        entry.version.lastTrainedAt = model.lastTrainedAt;
        entry.version.notes = model.notes;
        entry.version.skillRating = model.skillRating;
        entry.version.hasSkillRating = model.hasSkillRating;
        entry.version.modelCount = 1;
        entry.checkpointPath = model.checkpointPath;
        entry.trainingSetupPath = model.trainingSetupPath;
        entry.agentSetupSource = model.agentSetupSource;
        entry.primary = model.primary;
        entry.included = model.included;
        entry.resultsOnly = model.resultsOnly;
        entry.libraryModelId = model.libraryModelId;
        entry.fingerprint = QString::fromStdString(model.fingerprint);
        entry.agentSetupJson = model.agentSetupJson;
        entry.colorIndex = model.colorIndex;
        if (entry.colorIndex < 0)
            entry.colorIndex = NextAvailableModelColorIndex();
        // Restore immutable selections without re-enumerating storage.
        // Discovery updates their display status; execution still reads and verifies the selected files.
        entry.hasSavedAgentSetup = !entry.trainingSetupPath.empty();
        m_selectedVersions.push_back(std::move(entry));
    }
}

void ModelBenchmarkWidget::LoadPersistedSetup()
{
    if (!m_persistSetup || m_running)
        return;

    if (m_setupSaveTimer->isActive()) {
        m_setupSaveTimer->stop();
        SavePersistedSetup();
    }

    m_loadingPersistedSetup = true;
    m_loadedAgentName = CurrentAgentName();

    RocketForgeTraining::ModelBenchmarkSettings setup;
    const std::string agentName = m_loadedAgentName.toStdString();
    if (!agentName.empty())
        RocketForgeTraining::SaveLastSelectedAgent(agentName);

    RocketForgeTraining::ModelBenchmarkSettingsLoadResult result = RocketForgeTraining::LoadModelBenchmarkSettings(m_settingsPath);
    if (result.ok)
        setup = result.value;

    ApplyPersistedSetup(setup);
    m_loadingPersistedSetup = false;

    EnsureVersionCatalog();
    UpdateAvailableSelection();
    SortSelectedVersions();
    PopulateSelectedTable();
    UpdateInputState();
}

void ModelBenchmarkWidget::SavePersistedSetup()
{
    if (!m_persistSetup || m_loadingPersistedSetup || m_running || m_editingSetup)
        return;

    std::string err;
    if (!GGL::JsonSaveToFile(BuildPersistedSetup(), m_settingsPath, &err))
        AppendConsoleLine(QString("Unable to save benchmark setup: ") + QString::fromStdString(err));
}

void ModelBenchmarkWidget::OnSetupChanged()
{
    if (!m_loadingPersistedSetup && !m_running && !m_editingSetup) {
        ++m_workspaceGeneration;
        m_runAfterWorkspaceRefresh = false;
        m_previewAfterWorkspaceRefresh = false;
        m_opponentsAfterWorkspaceRefresh = false;
        m_setupSaveTimer->start();
        m_workspaceRefreshTimer->start();
        UpdateWorkspaceSummary();
        m_scheduleSummaryLabel->setText("Updating schedule…");
        m_scheduleSummaryLabel->setToolTip("Recalculating coverage and game budgets for the current selection.");
        m_runButton->setEnabled(false);
    }
}

std::filesystem::path ModelBenchmarkWidget::ResolveTrainingSetupPath(const BenchmarkVersionEntry& entry)
{
    if (!entry.trainingSetupPath.empty() && std::filesystem::exists(entry.trainingSetupPath))
        return entry.trainingSetupPath;
    if (entry.sourceLabel != "RocketForge" || entry.agentName.empty() || entry.buildName.empty() || entry.version.id.empty())
        return entry.trainingSetupPath;

    try {
        RocketForgeTraining::BuildStorage storage(
            RocketForgeTraining::ResolveAgentModelsFolder(entry.agentName),
            entry.buildName);
        std::optional<RocketForgeTraining::TrainingSegmentMetadata> segment =
            storage.LoadSegmentById(entry.version.id);
        if (!segment.has_value())
            return {};

        return storage.SegmentRoot(segment->folderName) / RocketForge::TrainingSetupSnapshot::FILE_NAME;
    } catch (const std::exception&) {
        return {};
    }
}

void ModelBenchmarkWidget::RefreshTrainingMetadata(BenchmarkVersionEntry& entry)
{
    if (!entry.version.lastTrainedAt.empty() || entry.sourceLabel != "RocketForge"
        || entry.checkpointPath.empty() || entry.version.latestTimesteps == 0)
        return;

    const std::shared_ptr<const ModelCatalogSnapshot> catalog = ModelCatalog::Instance().Get(QString::fromStdString(entry.agentName));
    for (const CatalogModel& model : catalog->models) {
        if (model.version.id == entry.version.id && model.version.buildName == entry.buildName
            && std::filesystem::path(model.checkpointPath) == entry.checkpointPath && entry.fingerprint.isEmpty()) {
            entry.version.lastTrainedAt = model.version.lastTrainedAt;
            return;
        }
    }
}

void ModelBenchmarkWidget::RefreshAgentSetupStatus(BenchmarkVersionEntry& entry,
    const std::optional<RocketForge::AgentSetupSnapshot>* currentSetup, const std::string& setupError)
{
    // Discovery already knows the exact snapshot path, including when no snapshot exists.
    if (!currentSetup)
        entry.trainingSetupPath = ResolveTrainingSetupPath(entry);
    entry.hasSavedAgentSetup = false;
    entry.agentSetupDiffs.clear();

    if (entry.trainingSetupPath.empty() || !std::filesystem::exists(entry.trainingSetupPath)) {
        if (entry.agentSetupJson.empty())
            entry.agentSetupSource = RocketForgeTraining::ModelBenchmarkAgentSetupSource::CurrentAgent;
        return;
    }

    std::optional<RocketForge::TrainingSetupSnapshot> savedSnapshot =
        RocketForge::LoadTrainingSetupSnapshot(entry.trainingSetupPath);
    if (!savedSnapshot.has_value() || !savedSnapshot->agentSetup.has_value()) {
        if (entry.agentSetupJson.empty())
            entry.agentSetupSource = RocketForgeTraining::ModelBenchmarkAgentSetupSource::CurrentAgent;
        return;
    }

    entry.hasSavedAgentSetup = true;
    try {
        if (currentSetup) {
            if (!currentSetup->has_value())
                throw std::runtime_error(setupError);

            entry.agentSetupDiffs = RocketForge::DiffAgentSetupSnapshots(*savedSnapshot->agentSetup, currentSetup->value());
            return;
        }

        std::unique_ptr<Agent> currentAgent = AgentRegistry::Create(entry.agentName);
        if (!currentAgent)
            throw std::runtime_error("AgentRegistry returned null");
        const RocketForge::AgentSetupSnapshot effective = RocketForge::CaptureAgentSetupSnapshot(*currentAgent);
        entry.agentSetupDiffs = RocketForge::DiffAgentSetupSnapshots(*savedSnapshot->agentSetup, effective);
    } catch (const std::exception& e) {
        entry.agentSetupDiffs.push_back({
            "Current Agent",
            "saved setup",
            std::string("failed to create current Agent: ") + e.what()
        });
    }
}

QString ModelBenchmarkWidget::BuildSetupStatusTooltip(const BenchmarkVersionEntry& entry, QString* sourceLabel) const
{
    const QString source = entry.agentSetupSource == RocketForgeTraining::ModelBenchmarkAgentSetupSource::SavedSegment
        ? "Trained" : "Current";
    if (sourceLabel) *sourceLabel = source;

    if (!entry.agentSetupJson.empty()) {
        QString tooltip = "Uses the " + source + " Agent settings recorded for this model in the saved benchmark.\n";
        bool matchesSource = false;
        try {
            const std::optional<RocketForge::AgentSetupSnapshot> recorded =
                RocketForge::AgentSetupSnapshotFromJson(nlohmann::json::parse(entry.agentSetupJson));
            if (!recorded) throw std::runtime_error("Recorded Agent setup is invalid");

            RocketForge::AgentSetupSnapshot prediction = *recorded;
            RocketForge::ResolveAgentSimulation(prediction);
            tooltip += prediction.useBallPrediction
                ? QString("Ball prediction: RocketSim v%1.\n").arg(prediction.ballPredictionEngineVersion)
                : "Ball prediction: disabled.\n";

            BenchmarkVersionEntry live = entry;
            live.agentSetupJson.clear();
            const std::optional<RocketForge::AgentSetupSnapshot> latest =
                RocketForge::AgentSetupSnapshotFromJson(nlohmann::json::parse(CaptureModelSetup(live)));
            if (!latest) throw std::runtime_error("Selected Agent setup is unavailable");

            const std::vector<RocketForge::AgentSetupDiff> diffs = RocketForge::DiffAgentSetupSnapshots(*recorded, *latest);
            matchesSource = diffs.empty();
            if (matchesSource) {
                tooltip += "These settings match the " + source + " setup available now.\n";
            } else {
                tooltip += "Differences from the " + source + " setup available now (recorded -> now):\n";
                for (const RocketForge::AgentSetupDiff& diff : diffs)
                    tooltip += QString("%1: %2 -> %3\n").arg(QString::fromStdString(diff.field),
                        QString::fromStdString(diff.savedValue), QString::fromStdString(diff.currentValue));
            }
        } catch (const std::exception& error) {
            tooltip += "Cannot compare with the setup available now: " + QString::fromUtf8(error.what()) + "\n";
        }

        // Frozen settings must not masquerade as today's agent after it changes.
        if (!matchesSource && sourceLabel) *sourceLabel = source + " (saved version)";
        return tooltip + "Choose Trained or Current to replace the recorded settings with that setup now. Test Setup independently selects the match simulator.";
    }

    const QString sourceText =
        entry.agentSetupSource == RocketForgeTraining::ModelBenchmarkAgentSetupSource::SavedSegment
            ? "This model will use the Agent setup saved when it was trained."
            : "This model will use the current Agent setup, including its ball predictor. Test Setup independently selects the match simulator.";

    if (!entry.hasSavedAgentSetup)
        return QString("No saved Agent setup metadata was found. ") + sourceText;

    if (entry.agentSetupDiffs.empty())
        return QString("Saved Agent setup matches the current Agent. ") + sourceText;

    QString tooltip = "Saved Agent setup differs from the current Agent:\n";
    for (const RocketForge::AgentSetupDiff& diff : entry.agentSetupDiffs) {
        tooltip += QString("%1: %2 -> %3\n")
            .arg(QString::fromStdString(diff.field),
                 QString::fromStdString(diff.savedValue),
                 QString::fromStdString(diff.currentValue));
    }
    tooltip += sourceText;
    return tooltip;
}

QTableWidgetItem* ModelBenchmarkWidget::MakeSetupStatusItem(const BenchmarkVersionEntry& entry) const
{
    auto* item = new QTableWidgetItem;
    item->setFlags(item->flags() & ~Qt::ItemIsEditable);
    item->setTextAlignment(Qt::AlignCenter);
    item->setToolTip("<qt>" + BuildSetupStatusTooltip(entry).toHtmlEscaped().replace("\n", "<br>") + "</qt>");

    if (!entry.hasSavedAgentSetup)
        item->setIcon(style()->standardIcon(QStyle::SP_MessageBoxInformation));
    else if (!entry.agentSetupDiffs.empty())
        item->setIcon(style()->standardIcon(QStyle::SP_MessageBoxWarning));
    else
        item->setIcon(style()->standardIcon(QStyle::SP_DialogApplyButton));

    return item;
}

std::string ModelBenchmarkWidget::MakeEntryKey(const BenchmarkVersionEntry& entry) const
{
    // Imported models belong to the agent, regardless of the stage selected when they were added.
    const std::string stage = entry.sourceLabel == "Imported Model" ? std::string() : entry.stageName;
    return entry.agentName + "|" + entry.buildName + "|" + stage + "|" + entry.sourceLabel + "|" + entry.version.id + "|" + entry.checkpointPath.string();
}

std::string ModelBenchmarkWidget::MakeDisplayName(const BenchmarkVersionEntry& entry)
{
    return entry.agentName + " / " + entry.buildName + " / " + entry.version.id;
}

int ModelBenchmarkWidget::NextAvailableModelColorIndex() const
{
    std::array<int, ModelColorCount> useCounts = {};
    for (const BenchmarkVersionEntry& entry : m_selectedVersions) {
        if (entry.colorIndex < 0)
            continue;
        useCounts[(size_t)entry.colorIndex % useCounts.size()]++;
    }

    return (int)std::distance(
        useCounts.begin(),
        std::min_element(useCounts.begin(), useCounts.end()));
}

int ModelBenchmarkWidget::CompareBenchmarkColumn(int lhsRow, int rhsRow, int priorityColumn, bool selectedTable) const
{
    const BenchmarkVersionEntry& a = selectedTable ? m_selectedVersions[lhsRow] : m_availableVersions[lhsRow];
    const BenchmarkVersionEntry& b = selectedTable ? m_selectedVersions[rhsRow] : m_availableVersions[rhsRow];

    switch (priorityColumn) {
    case BenchSortBuild:
        return TableSortController::CompareStrings(
            QString::fromStdString(a.buildName), QString::fromStdString(b.buildName));
    case BenchSortId: {
        int result = TableSortController::CompareValues(a.version.major, b.version.major);
        if (result == 0)
            result = TableSortController::CompareValues(a.version.minor, b.version.minor);
        if (result == 0)
            result = TableSortController::CompareStrings(
                QString::fromStdString(a.version.id), QString::fromStdString(b.version.id));
        return result;
    }
    case BenchSortStage:
        return TableSortController::CompareStrings(
            QString::fromStdString(a.stageName), QString::fromStdString(b.stageName));
    case BenchSortBranch:
        return TableSortController::CompareStrings(
            QString::fromStdString(BranchNameOrDefault(a.version.branchName)),
            QString::fromStdString(BranchNameOrDefault(b.version.branchName)));
    case BenchSortAgent:
        return TableSortController::CompareStrings(
            QString::fromStdString(a.agentName), QString::fromStdString(b.agentName));
    case BenchSortLatest:
        return TableSortController::CompareValues(a.version.latestTimesteps, b.version.latestTimesteps);
    case BenchSortLastTrained:
        return TableSortController::CompareValues(a.version.lastTrainedAt, b.version.lastTrainedAt);
    case BenchSortRating: {
        int result = TableSortController::CompareValues(a.version.hasSkillRating ? 1 : 0, b.version.hasSkillRating ? 1 : 0);
        if (result == 0 && a.version.hasSkillRating && b.version.hasSkillRating)
            result = TableSortController::CompareValues(a.version.skillRating, b.version.skillRating);
        return result;
    }
    case BenchSortSetup: {
        int result = TableSortController::CompareValues(a.hasSavedAgentSetup ? 1 : 0, b.hasSavedAgentSetup ? 1 : 0);
        if (result == 0)
            result = TableSortController::CompareValues((int)a.agentSetupDiffs.size(), (int)b.agentSetupDiffs.size());
        return result;
    }
    case BenchSortNotes:
        return TableSortController::CompareStrings(
            QString::fromStdString(a.version.notes), QString::fromStdString(b.version.notes));
    case BenchSortPrimary:
        return TableSortController::CompareValues(a.primary ? 1 : 0, b.primary ? 1 : 0);
    case BenchSortAgentSetup:
        return TableSortController::CompareValues((int)a.agentSetupSource, (int)b.agentSetupSource);
    case BenchSortCheckpoint:
        return TableSortController::CompareStrings(
            QString::fromStdString(a.checkpointPath.string()),
            QString::fromStdString(b.checkpointPath.string()));
    default:
        return 0;
    }
}

void ModelBenchmarkWidget::SortAvailableVersions()
{
    std::vector<int> order(m_availableVersions.size());
    for (int i = 0; i < (int)order.size(); i++)
        order[i] = i;

    std::sort(order.begin(), order.end(), [this](int lhs, int rhs) {
        return m_availableSort.CompareRows(lhs, rhs) < 0;
    });

    std::vector<BenchmarkVersionEntry> sorted;
    sorted.reserve(m_availableVersions.size());
    for (int index : order)
        sorted.push_back(std::move(m_availableVersions[index]));
    m_availableVersions = std::move(sorted);
}

void ModelBenchmarkWidget::SortSelectedVersions()
{
    std::vector<int> order(m_selectedVersions.size());
    for (int i = 0; i < (int)order.size(); i++)
        order[i] = i;

    std::sort(order.begin(), order.end(), [this](int lhs, int rhs) {
        return m_selectedSort.CompareRows(lhs, rhs) < 0;
    });

    std::vector<BenchmarkVersionEntry> sorted;
    sorted.reserve(m_selectedVersions.size());
    for (int index : order)
        sorted.push_back(std::move(m_selectedVersions[index]));
    m_selectedVersions = std::move(sorted);
}

void ModelBenchmarkWidget::ApplyAvailableRowFilters()
{
    if (!m_versionTable)
        return;

    const bool updatesEnabled = m_versionTable->updatesEnabled();
    m_versionTable->setUpdatesEnabled(false);
    const QString branchFilter = CurrentAvailableBranchName();
    const QString searchText = m_availableSearchEdit ? m_availableSearchEdit->text().trimmed() : QString();
    const bool hasBranchFilter = !branchFilter.isEmpty() && branchFilter != AllBranchesLabel;
    const QStringList searchTerms = searchText.split(' ', Qt::SkipEmptyParts);
    const std::string build = CurrentBuildName().toStdString();
    const std::string stage = CurrentStageName().toStdString();
    const bool hideImported = m_hideImportedModelsCheck->isChecked();
    int visibleRows = 0;

    for (int row = 0; row < m_versionTable->rowCount(); row++) {
        const BenchmarkVersionEntry& entry = m_availableVersions[row];
        const bool imported = entry.sourceLabel == "Imported Model";
        bool hideRow = imported ? hideImported
            : ((build != AllBuildsLabel && entry.buildName != build)
                || (stage != AllStagesLabel && entry.stageName != stage));
        if (!hideRow && hasBranchFilter)
            hideRow = QString::fromStdString(BranchNameOrDefault(entry.version.branchName)) != branchFilter;
        if (!hideRow && !searchTerms.isEmpty()) {
            const QString text = m_versionTable->item(row, VersionColumnNotes)->data(Qt::UserRole).toString();
            for (const QString& term : searchTerms) {
                if (!text.contains(term, Qt::CaseInsensitive)) {
                    hideRow = true;
                    break;
                }
            }
        }
        if (m_versionTable->isRowHidden(row) != hideRow)
            m_versionTable->setRowHidden(row, hideRow);
        visibleRows += !hideRow;
    }

    const QString countText = QString("%1 of %2 models | Click a row to add | %3 selected")
        .arg(visibleRows).arg(m_availableVersions.size()).arg(m_selectedVersions.size());
    m_catalogStatusLabel->setText(m_catalogScan.valid()
        && (m_scanningAgent == CurrentAgentName() || !m_versionCatalogs.contains(CurrentAgentName()))
        ? "Loading model catalog... | " + countText : countText);
    m_versionTable->setUpdatesEnabled(updatesEnabled);
}

void ModelBenchmarkWidget::PopulateAvailableTable(bool preserveColumnWidths)
{
    QSignalBlocker blocker(m_versionTable);
    const std::vector<int> columnWidths = preserveColumnWidths
        ? CaptureColumnWidths(m_versionTable)
        : std::vector<int>();
    const bool updatesEnabled = m_versionTable->updatesEnabled();
    m_versionTable->setUpdatesEnabled(false);

    std::set<std::string> selectedKeys;
    for (const BenchmarkVersionEntry& entry : m_selectedVersions)
        selectedKeys.insert(MakeEntryKey(entry));

    m_versionTable->setRowCount((int)m_availableVersions.size());
    for (int row = 0; row < (int)m_availableVersions.size(); row++) {
        const auto& entry = m_availableVersions[row];
        const auto& v = entry.version;
        auto* useItem = new QTableWidgetItem;
        useItem->setFlags((useItem->flags() | Qt::ItemIsUserCheckable) & ~Qt::ItemIsEditable);
        const bool selected = selectedKeys.contains(MakeEntryKey(entry));
        useItem->setCheckState(selected ? Qt::Checked : Qt::Unchecked);
        useItem->setToolTip(selected ? "Already added to the benchmark" : "Click this row to add the model");
        if (v.modelCount == 0 || entry.checkpointPath.empty() || selected)
            useItem->setFlags(useItem->flags() & ~Qt::ItemIsEnabled);

        m_versionTable->setItem(row, VersionColumnUse, useItem);
        m_versionTable->setItem(row, VersionColumnSetup, MakeSetupStatusItem(entry));
        m_versionTable->setItem(row, VersionColumnAgent, ReadOnlyItem(QString::fromStdString(entry.agentName)));
        m_versionTable->setItem(row, VersionColumnBuild, ReadOnlyItem(QString::fromStdString(entry.buildName)));
        m_versionTable->setItem(row, VersionColumnId, ReadOnlyItem(QString::fromStdString(v.id)));
        m_versionTable->setItem(row, VersionColumnStage, ReadOnlyItem(QString::fromStdString(entry.stageName)));
        m_versionTable->setItem(row, VersionColumnBranch, ReadOnlyItem(QString::fromStdString(BranchNameOrDefault(v.branchName))));
        m_versionTable->setItem(row, VersionColumnLatest, ReadOnlyItem(ForgeGUI::FormatTimesteps(v.latestTimesteps)));

        std::ostringstream rating;
        if (v.hasSkillRating)
            rating << std::fixed << std::setprecision(1) << v.skillRating;
        const QString ratingText = v.hasSkillRating ? QString::fromStdString(rating.str()) : QString();
        m_versionTable->setItem(row, VersionColumnRating, ReadOnlyItem(ratingText));

        auto* notesItem = ReadOnlyItem(QString::fromStdString(v.notes));
        notesItem->setTextAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        m_versionTable->setItem(row, VersionColumnNotes, notesItem);
        QStringList searchParts;
        for (int column = VersionColumnSetup; column <= VersionColumnNotes; ++column) {
            const QTableWidgetItem* item = m_versionTable->item(row, column);
            searchParts << item->text() << item->toolTip();
        }
        searchParts << QString::fromStdString(entry.sourceLabel);
        notesItem->setData(Qt::UserRole, searchParts.join(' '));
    }
    ApplyAvailableRowFilters();
    if (preserveColumnWidths)
        RestoreColumnWidths(m_versionTable, columnWidths);
    else if (!m_availableTableColumnsInitialized && m_versionTable->rowCount() > 0) {
        FitTableColumnsToContentsAndFillWidth(m_versionTable);
        m_availableTableColumnsInitialized = true;
    }
    m_versionTable->setUpdatesEnabled(updatesEnabled);
}

void ModelBenchmarkWidget::PopulateSelectedTable(bool preserveColumnWidths)
{
    const bool allIncluded = !m_selectedVersions.empty()
        && std::all_of(m_selectedVersions.begin(), m_selectedVersions.end(),
            [](const BenchmarkVersionEntry& entry) { return entry.included; });
    m_toggleAllModelsButton->setText(allIncluded ? "Hide All" : "Show All");
    m_toggleAllModelsButton->setEnabled(!m_selectedVersions.empty());

    const std::vector<int> columnWidths = preserveColumnWidths
        ? CaptureColumnWidths(m_selectedTable)
        : std::vector<int>();
    const bool updatesEnabled = m_selectedTable->updatesEnabled();
    m_selectedTable->setUpdatesEnabled(false);
    QSignalBlocker signalBlocker(m_selectedTable);

    m_selectedTable->setRowCount((int)m_selectedVersions.size());
    for (int row = 0; row < (int)m_selectedVersions.size(); row++) {
        const auto& entry = m_selectedVersions[row];
        auto* useItem = new QTableWidgetItem;
        useItem->setFlags((useItem->flags() | Qt::ItemIsUserCheckable) & ~Qt::ItemIsEditable);
        useItem->setCheckState(entry.included ? Qt::Checked : Qt::Unchecked);
        useItem->setToolTip(IncludeTooltip);
        m_selectedTable->setItem(row, SelectedColumnUse, useItem);

        auto* role = new QComboBox(m_selectedTable);
        role->addItems({ "Target", "Opponent", "Results only" });
        role->setCurrentIndex(entry.resultsOnly ? 2 : entry.primary ? 0 : 1);
        role->setEnabled(!m_running);
        role->setToolTip(NextRunTooltip);
        role->setItemData(0, "<qt>A model selected for testing in the next run. It can play any Target or Opponent. Saved games between shown models contribute to combined results regardless of role.</qt>", Qt::ToolTipRole);
        role->setItemData(1, "<qt>A comparison model for targets to play against. Two opponents do not play new games against each other. Saved games can still contribute to Elo.</qt>", Qt::ToolTipRole);
        role->setItemData(2, "<qt>Run no new games for this model. Its saved games against other shown models still contribute to combined results and Elo. Useful for older models or missing checkpoints.</qt>", Qt::ToolTipRole);
        connect(role, &QComboBox::currentIndexChanged, this, [this, row](int index) {
            if (row < 0 || row >= (int)m_selectedVersions.size()) return;

            m_selectedVersions[row].resultsOnly = index == 2;
            m_selectedVersions[row].primary = index == 0;
            OnSetupChanged();
        });
        m_selectedTable->setItem(row, SelectedColumnPrimary, new QTableWidgetItem);
        m_selectedTable->setCellWidget(row, SelectedColumnPrimary, role);

        m_selectedTable->setItem(row, SelectedColumnSetup, MakeSetupStatusItem(entry));
        m_selectedTable->setItem(row, SelectedColumnAgent, ReadOnlyItem(QString::fromStdString(entry.agentName)));
        m_selectedTable->setItem(row, SelectedColumnBuild, ReadOnlyItem(QString::fromStdString(entry.buildName)));
        m_selectedTable->setItem(row, SelectedColumnId, ReadOnlyItem(QString::fromStdString(entry.version.id)));
        m_selectedTable->setItem(row, SelectedColumnStage, ReadOnlyItem(QString::fromStdString(entry.stageName)));
        m_selectedTable->setItem(
            row,
            SelectedColumnBranch,
            ReadOnlyItem(QString::fromStdString(BranchNameOrDefault(entry.version.branchName))));
        m_selectedTable->setItem(
            row,
            SelectedColumnLatest,
            ReadOnlyItem(entry.version.latestTimesteps > 0
                ? ForgeGUI::FormatTimesteps(entry.version.latestTimesteps) : QString::fromUtf8("—")));
        m_selectedTable->item(row, SelectedColumnLatest)->setToolTip(entry.version.latestTimesteps > 0
            ? QString("%1 total training timesteps at this checkpoint.").arg(entry.version.latestTimesteps)
            : "Training timestep count was not recorded.");

        const QString recordedTime = QString::fromStdString(entry.version.lastTrainedAt);
        QDateTime trainedAt = QDateTime::fromString(recordedTime, Qt::ISODateWithMs);
        if (!trainedAt.isValid())
            trainedAt = QDateTime::fromString(recordedTime, "yyyyMMdd-HHmmss");
        trainedAt = trainedAt.toLocalTime();

        QTableWidgetItem* trainedItem = ReadOnlyItem(trainedAt.isValid()
            ? trainedAt.toString("yyyy-MM-dd HH:mm") : QString::fromUtf8("—"));
        trainedItem->setToolTip(trainedAt.isValid()
            ? "Last trained: " + trainedAt.toString("yyyy-MM-dd HH:mm:ss t") + " (local time)"
            : "No training date was recorded for this model.");
        m_selectedTable->setItem(row, SelectedColumnLastTrained, trainedItem);

        auto* setupCombo = new QComboBox(m_selectedTable);
        setupCombo->addItem("Trained", (int)RocketForgeTraining::ModelBenchmarkAgentSetupSource::SavedSegment);
        setupCombo->addItem("Current", (int)RocketForgeTraining::ModelBenchmarkAgentSetupSource::CurrentAgent);
        const bool capturedSetup = !entry.agentSetupJson.empty();
        QString sourceLabel;
        const QString setupTooltip = BuildSetupStatusTooltip(entry, &sourceLabel);
        const bool olderSetup = capturedSetup && sourceLabel.endsWith(" (saved version)");
        if (olderSetup) {
            setupCombo->addItem(sourceLabel, -1);
            setupCombo->setItemData(2, setupTooltip, Qt::ToolTipRole);
        }

        const int sourceIndex = setupCombo->findData(olderSetup ? -1 : (int)entry.agentSetupSource);
        setupCombo->setCurrentIndex(sourceIndex >= 0 ? sourceIndex : 0);

        // A captured run can switch to Current even when its training snapshot is unavailable.
        auto* setupItems = qobject_cast<QStandardItemModel*>(setupCombo->model());
        setupItems->item(0)->setEnabled(entry.hasSavedAgentSetup);
        setupCombo->setEnabled(!m_running && (entry.hasSavedAgentSetup || capturedSetup));
        setupCombo->setToolTip(setupTooltip);
        setupCombo->setItemData(0, "<qt>Use the Agent settings saved when this model was trained, such as its inputs, actions, timing, and ball predictor. Only available when those settings were recorded.</qt>", Qt::ToolTipRole);
        setupCombo->setItemData(1, "<qt>Use the Agent settings from the current application, including its ball predictor. These may differ from the settings used during this model's training.</qt>", Qt::ToolTipRole);
        const auto selectSetup = [this, row, setupCombo](int index) {
            if (index < 0 || setupCombo->itemData(index).toInt() < 0)
                return;
            if (row < 0 || row >= (int)m_selectedVersions.size())
                return;
            const RocketForgeTraining::ModelBenchmarkAgentSetupSource source =
                (RocketForgeTraining::ModelBenchmarkAgentSetupSource)setupCombo->itemData(index).toInt();
            if (m_selectedVersions[row].agentSetupJson.empty() && m_selectedVersions[row].agentSetupSource == source)
                return;

            m_selectedVersions[row].agentSetupJson.clear();
            m_selectedVersions[row].fingerprint.clear();
            m_selectedVersions[row].libraryModelId = -1;
            m_selectedVersions[row].agentSetupSource = source;
            QueueSelectedTableRefresh();
        };
        connect(setupCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, selectSetup);

        // Reselecting the displayed source explicitly releases a matching recorded snapshot too.
        connect(setupCombo, QOverload<int>::of(&QComboBox::activated), this, selectSetup);
        m_selectedTable->setCellWidget(row, SelectedColumnAgentSetup, setupCombo);

        QTableWidgetItem* checkpointItem = ReadOnlyItem(QString::fromStdString(entry.checkpointPath.string()));
        checkpointItem->setTextAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        m_selectedTable->setItem(row, SelectedColumnCheckpoint, checkpointItem);
    }
    if (preserveColumnWidths)
        RestoreColumnWidths(m_selectedTable, columnWidths);
    else if (!m_selectedTableColumnsInitialized) {
        // Size empty headers too; fit once more when the first model adds cell controls.
        // Keep identifiers compact and reserve spare width for the checkpoint path.
        for (int column = SelectedColumnPrimary; column < SelectedColumnCheckpoint; ++column) {
            m_selectedTable->resizeColumnToContents(column);
            const int minimumWidth = column == SelectedColumnAgent || column == SelectedColumnBuild ? 150 : 100;
            m_selectedTable->setColumnWidth(column, std::clamp(m_selectedTable->columnWidth(column), minimumWidth, 220));
        }

        m_selectedTableColumnsInitialized = m_selectedTable->rowCount() > 0;
    }

    QHeaderView* header = m_selectedTable->horizontalHeader();
    header->setSectionResizeMode(SelectedColumnUse, QHeaderView::ResizeToContents);
    header->setSectionResizeMode(SelectedColumnSetup, QHeaderView::ResizeToContents);
    header->setSectionResizeMode(SelectedColumnCheckpoint, QHeaderView::Stretch);
    m_selectedTable->setUpdatesEnabled(updatesEnabled);
}

void ModelBenchmarkWidget::QueueSelectedTableRefresh()
{
    if (m_selectedTableRefreshQueued)
        return;
    m_selectedTableRefreshQueued = true;
    QTimer::singleShot(0, this, [this]() {
        m_selectedTableRefreshQueued = false;
        SortSelectedVersions();
        PopulateSelectedTable();
        OnSetupChanged();
    });
}

void ModelBenchmarkWidget::UpdateAvailableSelection()
{
    QSignalBlocker blocker(m_versionTable);
    std::set<std::string> selectedKeys;
    for (const BenchmarkVersionEntry& entry : m_selectedVersions)
        selectedKeys.insert(MakeEntryKey(entry));

    for (int row = 0; row < (int)m_availableVersions.size(); ++row) {
        const BenchmarkVersionEntry& entry = m_availableVersions[row];
        QTableWidgetItem* item = m_versionTable->item(row, VersionColumnUse);
        const bool selected = selectedKeys.contains(MakeEntryKey(entry));
        const bool available = entry.version.modelCount > 0 && !entry.checkpointPath.empty();
        item->setCheckState(selected ? Qt::Checked : Qt::Unchecked);
        item->setFlags(available && !selected ? item->flags() | Qt::ItemIsEnabled : item->flags() & ~Qt::ItemIsEnabled);
        item->setToolTip(selected ? "Already added to the benchmark"
            : available ? "Click this row to add the model" : "No checkpoint is available for this model");
    }

    ApplyAvailableRowFilters();
}

void ModelBenchmarkWidget::AddAvailableVersion(int row)
{
    if (m_running || row < 0 || row >= (int)m_availableVersions.size() || m_versionTable->isRowHidden(row))
        return;

    BenchmarkVersionEntry entry = m_availableVersions[row];
    if (entry.version.modelCount == 0 || entry.checkpointPath.empty())
        return;

    const std::string key = MakeEntryKey(entry);
    if (std::any_of(m_selectedVersions.begin(), m_selectedVersions.end(),
        [&](const BenchmarkVersionEntry& selected) { return MakeEntryKey(selected) == key; }))
        return;

    entry.colorIndex = NextAvailableModelColorIndex();
    m_selectedVersions.push_back(std::move(entry));
    UpdateAvailableSelection();
    QueueSelectedTableRefresh();
    OnSetupChanged();
}

void ModelBenchmarkWidget::OnAddVersionsClicked()
{
    if (m_running)
        return;
    std::set<std::string> selectedKeys;
    for (const BenchmarkVersionEntry& entry : m_selectedVersions)
        selectedKeys.insert(MakeEntryKey(entry));

    for (int row = 0; row < (int)m_availableVersions.size(); ++row) {
        if (m_versionTable->isRowHidden(row))
            continue;
        BenchmarkVersionEntry entry = m_availableVersions[row];
        if (entry.version.modelCount == 0 || entry.checkpointPath.empty())
            continue;
        if (selectedKeys.insert(MakeEntryKey(entry)).second) {
            entry.colorIndex = NextAvailableModelColorIndex();
            m_selectedVersions.push_back(std::move(entry));
        }
    }
    UpdateAvailableSelection();
    QueueSelectedTableRefresh();
    OnSetupChanged();
}

void ModelBenchmarkWidget::OnAddExternalFolderClicked()
{
    if (m_running)
        return;

    const std::string agentName = CurrentAgentName().toStdString();
    std::string stageName = CurrentStageName() == AllStagesLabel ? std::string() : CurrentStageName().toStdString();

    const std::filesystem::path defaultExternalRoot = std::filesystem::path(TRAINING_ROOT).parent_path() / "Bots";
    const QString folder = QFileDialog::getExistingDirectory(
        this,
        "Add External Model Folder",
        QString::fromStdString(defaultExternalRoot.string()));
    if (folder.isEmpty())
        return;

    const std::filesystem::path root = folder.toStdWString();
    std::vector<std::filesystem::path> rootsToCheck;
    rootsToCheck.push_back(root);
    try {
        for (const auto& entry : std::filesystem::directory_iterator(root))
            if (entry.is_directory())
                rootsToCheck.push_back(entry.path());
    } catch (const std::exception&) {
        rootsToCheck.clear();
        rootsToCheck.push_back(root);
    }

    std::set<std::string> selectedKeys;
    for (const BenchmarkVersionEntry& entry : m_selectedVersions)
        selectedKeys.insert(MakeEntryKey(entry));

    int added = 0;
    for (const std::filesystem::path& candidateRoot : rootsToCheck) {
        std::filesystem::path modelFolder = GGL::TrainingStorage::ResolveModelFolder(candidateRoot);
        if (modelFolder.empty())
            continue;

        GGL::SegmentVersionSummary summary = {};
        summary.id = candidateRoot.filename().string();
        summary.modelCount = 1;
        summary.latestTimesteps = 0;
        summary.notes = "External model folder";

        BenchmarkVersionEntry entry = {};
        entry.agentName = agentName;
        entry.buildName = "External";
        entry.stageName = stageName;
        entry.sourceLabel = "External";
        entry.version = summary;
        entry.checkpointPath = modelFolder;
        RefreshAgentSetupStatus(entry);

        if (selectedKeys.insert(MakeEntryKey(entry)).second) {
            entry.colorIndex = NextAvailableModelColorIndex();
            m_selectedVersions.push_back(std::move(entry));
            added++;
        }
    }

    if (added == 0) {
        const QString message =
            "No loadable external model folders were found. Select either a model folder containing "
            "POLICY.lt/SHARED_HEAD.lt or a folder containing child model folders.";
        QMessageBox::information(
            this,
            "Model Benchmarking",
            message);
        return;
    }

    SortSelectedVersions();
    UpdateAvailableSelection();
    PopulateSelectedTable();
    OnSetupChanged();
}

void ModelBenchmarkWidget::OnRemoveSelectedClicked()
{
    if (m_running)
        return;

    std::set<int> rowsToRemove;
    for (QTableWidgetItem* item : m_selectedTable->selectedItems())
        rowsToRemove.insert(item->row());
    if (rowsToRemove.empty())
        return;

    std::vector<BenchmarkVersionEntry> kept;
    kept.reserve(m_selectedVersions.size());
    for (int row = 0; row < (int)m_selectedVersions.size(); row++)
        if (rowsToRemove.count(row) == 0)
            kept.push_back(m_selectedVersions[row]);
    m_selectedVersions = std::move(kept);

    UpdateAvailableSelection();
    PopulateSelectedTable();
    OnSetupChanged();
}

void ModelBenchmarkWidget::OnClearSelectedClicked()
{
    if (m_running)
        return;
    m_selectedVersions.clear();
    UpdateAvailableSelection();
    PopulateSelectedTable();
    OnSetupChanged();
}

void ModelBenchmarkWidget::OnToggleSetupClicked()
{
    ShowTestSetup();
}

RocketForgeTraining::ModelBenchmarkConfig ModelBenchmarkWidget::BuildConfig() const
{
    RocketForgeTraining::ModelBenchmarkConfig config;
    config.engine = RocketSim::EngineFromVersion(m_engineCombo->currentData().toInt());
    config.numArenas = m_arenasSpin->value();
    config.maxGamesPerMode = m_maxGamesSpin->value();
    config.seed = ParseBenchmarkSeed(m_seedEdit);
    config.run1v1 = m_mode1v1Check->isChecked();
    config.run2v2 = m_mode2v2Check->isChecked();
    config.run3v3 = m_mode3v3Check->isChecked();
    config.stochasticKickoff = m_stochasticKickoffCheck->isChecked();
    config.kickoffTemperature = (float)m_kickoffTemperatureSpin->value();
    config.betterConfidence = 0.95f;
    config.stopWhenRankingKnown = false;
    config.eloRangeConfidence = 0.95f;
    config.maxEloUncertaintyRange = 50.0f;
    config.runUntilMaxGames = true;
    config.collectOptionalStats = true;
    config.emitLiveResultSnapshots = true;

    for (const BenchmarkVersionEntry& entry : m_selectedVersions) {
        if (entry.resultsOnly) continue;
        std::filesystem::path checkpoint = entry.checkpointPath;
        config.models.push_back({
            MakeDisplayName(entry),
            entry.agentName,
            entry.stageName,
            entry.version.id,
            checkpoint,
            entry.trainingSetupPath,
            entry.agentSetupSource,
            entry.primary,
            entry.agentSetupJson
        });
    }
    return config;
}

std::vector<ModelBenchmarkLibraryModel> ModelBenchmarkWidget::BuildLibraryModels(
    const std::vector<BenchmarkVersionEntry>& entries, bool allowCached)
{
    std::vector<ModelBenchmarkLibraryModel> models;
    models.reserve(entries.size());
    for (const BenchmarkVersionEntry& entry : entries) {
        ModelBenchmarkLibraryModel model;
        model.displayName = QString::fromStdString(MakeDisplayName(entry));
        model.agentName = QString::fromStdString(entry.agentName);
        model.buildName = QString::fromStdString(entry.buildName);
        model.stageName = QString::fromStdString(entry.stageName);
        model.branchName = QString::fromStdString(BranchNameOrDefault(entry.version.branchName));
        model.versionId = QString::fromStdString(entry.version.id);
        model.sourceLabel = QString::fromStdString(entry.sourceLabel);
        model.checkpointPath = QString::fromStdString(entry.checkpointPath.string());
        model.trainingSetupPath = QString::fromStdString(entry.trainingSetupPath.string());
        model.latestTimesteps = entry.version.latestTimesteps;
        model.lastTrainedAt = QString::fromStdString(entry.version.lastTrainedAt);
        model.logicalKey = QStringList{
            model.agentName,
            model.buildName,
            model.stageName,
            model.versionId,
            model.sourceLabel
        }.join('|');
        model.fingerprint = ModelBenchmarkLibrary::FingerprintCheckpoint(
            entry.checkpointPath,
            entry.trainingSetupPath, allowCached);
        model.fingerprintVerified = !model.fingerprint.isEmpty();
        if (!model.fingerprintVerified) {
            model.fingerprint = "unverified:"
                + model.logicalKey
                + "|"
                + model.checkpointPath;
        }
        std::error_code checkpointError;
        model.availableLocally = !entry.checkpointPath.empty()
            && std::filesystem::exists(entry.checkpointPath, checkpointError)
            && !checkpointError;
        models.push_back(std::move(model));
    }
    return models;
}

void ModelBenchmarkWidget::RefreshSavedKnownModels()
{
    if (!m_savedResultsWidget)
        return;
    std::vector<ModelBenchmarkLibraryModel> models;
    models.reserve(m_availableVersions.size());
    for (const BenchmarkVersionEntry& entry : m_availableVersions) {
        ModelBenchmarkLibraryModel model;
        model.displayName = QString::fromStdString(MakeDisplayName(entry));
        model.agentName = QString::fromStdString(entry.agentName);
        model.buildName = QString::fromStdString(entry.buildName);
        model.stageName = QString::fromStdString(entry.stageName);
        model.branchName = QString::fromStdString(BranchNameOrDefault(entry.version.branchName));
        model.versionId = QString::fromStdString(entry.version.id);
        model.sourceLabel = QString::fromStdString(entry.sourceLabel);
        model.checkpointPath = QString::fromStdString(entry.checkpointPath.string());
        model.trainingSetupPath = QString::fromStdString(entry.trainingSetupPath.string());
        model.latestTimesteps = entry.version.latestTimesteps;
        model.lastTrainedAt = QString::fromStdString(entry.version.lastTrainedAt);
        model.logicalKey = QStringList{
            model.agentName,
            model.buildName,
            model.stageName,
            model.versionId,
            model.sourceLabel
        }.join('|');
        model.availableLocally = !entry.checkpointPath.empty() && entry.version.modelCount > 0;
        models.push_back(std::move(model));
    }
    m_savedResultsWidget->SetKnownModels(models);
}

void ModelBenchmarkWidget::OnSaveResultsClicked()
{
    if (!m_hasLastResult || !m_savedResultsWidget)
        return;
    if (m_lastRunLibraryModels.size() != m_lastResult.modelNames.size()) {
        QMessageBox::critical(
            this,
            "Save Benchmark Results",
            "The selected-model list changed after this benchmark completed. Run the benchmark again before saving it.");
        return;
    }
    if (m_lastResultRunId.isEmpty())
        m_lastResultRunId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    const bool notify = false;
    if (m_savedResultsWidget->SaveRun(m_lastResultRunId, m_lastResult, m_lastRunLibraryModels, notify, m_runSettingsJson)) {
        m_lastResultSavedToActiveLibrary = true;
        const std::ptrdiff_t reusable = std::count_if(m_lastResult.games.begin(), m_lastResult.games.end(),
            [](const RocketForgeTraining::ModelBenchmarkGameMetadata& game) { return game.mirrorCompleted; });
        AppendConsoleLine(QString("Saved %1 game records to the benchmark library; %2 count toward results and future runs.")
            .arg((qulonglong)m_lastResult.games.size()).arg((qulonglong)reusable));

        // Preserve the exact identities used for saving when the roster reconnects to library rows.
        for (size_t i = 0; i < m_lastRunVersions.size() && i < m_lastRunLibraryModels.size(); ++i) {
            for (BenchmarkVersionEntry& entry : m_selectedVersions)
                if (MakeEntryKey(entry) == MakeEntryKey(m_lastRunVersions[i]))
                    entry.fingerprint = m_lastRunLibraryModels[i].fingerprint;
        }

        m_saveResultsButton->setEnabled(false);
        m_saveResultsButton->hide();
        RefreshWorkspace();
    }
}

void ModelBenchmarkWidget::OnRunClicked()
{
    if (m_setupSaveTimer->isActive()) {
        m_setupSaveTimer->stop();
        SavePersistedSetup();
    }

    if (m_running)
        return;
    if (m_trainingActive) {
        QMessageBox::warning(
            this,
            "Cannot start benchmark",
            "Stop training before starting a model benchmark.");
        return;
    }

    if (!m_verifiedWorkspaceStart) {
        m_workspaceRefreshTimer->stop();
        m_runAfterWorkspaceRefresh = true;
        m_coverageDirty = true;
        RefreshWorkspace();
        return;
    }

    m_verifiedWorkspaceStart = false;
    try {
        const bool showPreview = false;
        const bool requireRunnable = true;
        RocketForgeTraining::ModelBenchmarkConfig config = PrepareSchedule(showPreview, requireRunnable);
        if (config.pairBudgets.empty()) {
            QMessageBox::information(this, "Benchmark coverage", "No new games are scheduled. Existing saved games remain included in the comparison.");
            return;
        }

        RocketForgeTraining::RequireModelBenchmarkEngines(config);

        std::vector<BenchmarkVersionEntry> participatingEntries;
        for (int index : m_plannedModelIndices) participatingEntries.push_back(m_selectedVersions[index]);

        std::vector<ModelBenchmarkLibraryModel> participatingLibraryModels = BuildLibraryModels(participatingEntries);
        for (int index = 0; index < (int)participatingEntries.size(); ++index) {
            const BenchmarkVersionEntry& entry = participatingEntries[index];
            if (!entry.fingerprint.isEmpty() && entry.fingerprint != participatingLibraryModels[index].fingerprint)
                throw std::runtime_error("Checkpoint or agent setup changed for " + MakeDisplayName(entry)
                    + ". Remove it and add the current model again, or keep the saved snapshot as Results only.");
            participatingLibraryModels[index].primary = config.models[index].primary;
        }

        RocketForgeTraining::ModelBenchmarkSettings runSettings = BuildPersistedSetup();
        for (int index = 0; index < (int)participatingEntries.size(); ++index) {
            runSettings.selectedModels[m_plannedModelIndices[index]].agentSetupJson = config.models[index].agentSetupJson;
        }

        m_runSettingsJson = QString::fromStdString(RocketForgeTraining::SerializeModelBenchmarkSettings(runSettings));
        m_activeRunVersions = std::move(participatingEntries);
        m_activeRunLibraryModels = std::move(participatingLibraryModels);
        const int eligibleMatchupCount = (int)config.pairBudgets.size();
        for (BenchmarkVersionEntry& entry : m_selectedVersions)
            entry.colorIndex = -1;
        PopulateSelectedTable(true);
        SavePersistedSetup();

        std::vector<ModelBenchmarkLibrarySelection> comparison;
        std::vector<ModelBenchmarkLibraryModel> temporaryModels;
        std::vector<qint64> liveModelIds;
        auto comparisonId = [&](int index) -> qint64 {
            const qint64 savedId = m_selectedVersions[index].libraryModelId;
            return savedId >= 0 ? savedId : -2 - (qint64)index;
        };
        for (int index : m_plannedModelIndices) liveModelIds.push_back(comparisonId(index));

        for (int index = 0; index < (int)m_selectedVersions.size(); ++index) {
            const BenchmarkVersionEntry& entry = m_selectedVersions[index];
            if (!entry.included) continue;

            const qint64 id = comparisonId(index);
            if (id < 0 && std::find(m_plannedModelIndices.begin(), m_plannedModelIndices.end(), index) == m_plannedModelIndices.end()) continue;

            comparison.push_back({id, true});
            if (id < 0) {
                ModelBenchmarkLibraryModel model;
                model.id = id;
                model.displayName = QString::fromStdString(MakeDisplayName(entry));
                model.agentName = QString::fromStdString(entry.agentName);
                model.versionId = QString::fromStdString(entry.version.id);
                model.checkpointPath = QString::fromStdString(entry.checkpointPath.string());
                temporaryModels.push_back(std::move(model));
            }
        }

        m_savedResultsWidget->BeginLiveComparison(comparison, temporaryModels, liveModelIds, WorkspaceFilter());
        SetRunning(true);
        SetSetupSectionVisible(true);
        m_hasLastResult = false;
        m_lastResult = {};
        m_latestProgressTitle = "Starting benchmark";
        m_latestProgressGames = 0;
        m_latestDrawGames = 0;
        m_latestProgressGamesPerSecond = 0.0;
        m_latestProgressElapsedSeconds = 0.0;
        m_latestEstimatedRemainingSeconds = -1.0;
        m_latestActiveArenas = 0;
        m_latestTotalArenas = config.numArenas;
        m_latestPendingGames = 0;
        m_latestScheduledGames = 0;
        m_latestArenaCreates = 0;
        m_latestArenaRebuilds = 0;
        m_latestArenaReuses = 0;
        m_latestActive1v1Arenas = 0;
        m_latestActive2v2Arenas = 0;
        m_latestActive3v3Arenas = 0;
        m_latestCudaOomRecoveries = 0;
        m_latestCudaGraphsDisabledAfterOom = false;
        m_latestGpuTelemetry = {};
        QStringList selectedModeNames;
        for (int mode : { 1, 2, 3 })
            if (std::any_of(config.pairBudgets.begin(), config.pairBudgets.end(),
                    [mode](const RocketForgeTraining::ModelBenchmarkPairBudget& pair) { return pair.mode == mode; }))
                selectedModeNames << QString("%1v%1").arg(mode);
        m_latestModeSummary = selectedModeNames.join(" / ");
        m_latestProgressMaxGames = 0;
        for (const RocketForgeTraining::ModelBenchmarkPairBudget& pair : config.pairBudgets) m_latestProgressMaxGames += pair.games;
        m_currentRunUntilMaxGames = config.runUntilMaxGames;
        m_progressPanel->BeginRun("Starting benchmark...");
        m_progressPanel->SetDetails(BuildProgressDetailsStatus());
        if (m_exportResultsButton)
            m_exportResultsButton->setEnabled(false);
        if (m_consoleText)
            m_consoleText->clear();
        AppendConsoleLine(QString("Starting Model Benchmarking run with %1.").arg(m_engineCombo->currentText()));
        AppendConsoleLine(QString("Models: %1  Pair-modes: %2  Arenas: %3  Game limit per pair/mode: %4")
            .arg((int)config.models.size())
            .arg(eligibleMatchupCount)
            .arg(config.numArenas)
            .arg(config.maxGamesPerMode));
        AppendConsoleLine(m_scheduleSummaryLabel->text());
        AppendConsoleLine(config.seed == 0
            ? "Seed: auto (the resolved run seed will be reported on completion)."
            : QString("Seed: %1").arg((qulonglong)config.seed));
        m_worker = new ModelBenchmarkWorker;

        connect(m_worker, &ModelBenchmarkWorker::LogLine, this, &ModelBenchmarkWidget::AppendConsoleLine);
        connect(m_worker, &ModelBenchmarkWorker::ProgressUpdated, this, &ModelBenchmarkWidget::OnProgressUpdated);
        connect(m_worker, &ModelBenchmarkWorker::Completed, this, &ModelBenchmarkWidget::OnBenchmarkComplete);
        connect(m_worker, &ModelBenchmarkWorker::Cancelled, this, &ModelBenchmarkWidget::OnBenchmarkCancelled);
        connect(m_worker, &ModelBenchmarkWorker::Failed, this, &ModelBenchmarkWidget::OnBenchmarkFailed);
        connect(m_worker, &QObject::destroyed, this, [this]() { m_worker = nullptr; });

        m_workerHost.Start<ModelBenchmarkWorker>(m_worker, [config](ModelBenchmarkWorker* worker) {
            worker->RunBenchmark(config);
        });
    } catch (const RocketSim::EngineUnavailableError& e) {
        m_activeRunVersions.clear();
        m_activeRunLibraryModels.clear();
        ShowSimulatorUnavailable(this, QString::fromUtf8(e.what()));
    } catch (const std::exception& e) {
        m_activeRunVersions.clear();
        m_activeRunLibraryModels.clear();
        QMessageBox::critical(this, "Model Benchmarking", QString::fromStdString(e.what()));
    }
}

void ModelBenchmarkWidget::OnCancelClicked()
{
    if (!m_running)
        return;
    if (m_worker)
        m_worker->Cancel();
    m_cancelButton->setEnabled(false);
    m_progressPanel->SetStatus("Stopping benchmark and saving collected results...");
    m_latestProgressTitle = "Stopping benchmark and saving collected results...";
    AppendConsoleLine("Stop requested. Collected game records will be saved; unfinished mirror groups do not count toward coverage.");
}

void ModelBenchmarkWidget::OnProgressUpdated()
{
    if (!m_worker) return;

    const std::optional<RocketForgeTraining::ModelBenchmarkProgress> latest = m_worker->TakeLatestProgress();
    if (!latest) return;

    const RocketForgeTraining::ModelBenchmarkProgress& progress = *latest;
    if (progress.maxGames > 0)
        m_latestProgressMaxGames = progress.maxGames;
    const int64_t maxGames = std::max<int64_t>(1, m_latestProgressMaxGames);
    const int64_t totalGames = (int64_t)progress.scoredGames + (int64_t)progress.noResultGames;
    m_latestProgressTitle = QString::fromStdString(progress.label);
    m_latestProgressGames = totalGames;
    m_latestDrawGames = progress.noResultGames;
    m_latestProgressGamesPerSecond = progress.snapshotGamesPerSecond;
    m_latestProgressElapsedSeconds = progress.elapsedSeconds;
    m_latestEstimatedRemainingSeconds = progress.estimatedRemainingSeconds;
    m_latestActiveArenas = progress.activeArenas;
    if (progress.totalArenas > 0)
        m_latestTotalArenas = progress.totalArenas;
    m_latestPendingGames = progress.pendingGames;
    m_latestScheduledGames = progress.scheduledGames;
    m_latestArenaCreates = progress.arenaCreates;
    m_latestArenaRebuilds = progress.arenaRebuilds;
    m_latestArenaReuses = progress.arenaReuses;
    m_latestActive1v1Arenas = progress.active1v1Arenas;
    m_latestActive2v2Arenas = progress.active2v2Arenas;
    m_latestActive3v3Arenas = progress.active3v3Arenas;
    m_latestCudaOomRecoveries = progress.cudaOomRecoveries;
    m_latestCudaGraphsDisabledAfterOom = progress.cudaGraphsDisabledAfterOom;
    m_latestGpuTelemetry = progress.gpu;
    m_progressPanel->SetProgress(totalGames, maxGames);
    m_progressPanel->SetStatus(BuildProgressStatus(progress.elapsedSeconds));
    m_progressPanel->SetDetails(BuildProgressDetailsStatus());

    if (progress.liveResult) {
        m_savedResultsWidget->UpdateLiveComparison(progress.liveResult);
    }

    QString line = QString("%1  Games: %2/%3  %4 games/sec  Elapsed: %5  Arenas: %6/%7"
                           " (1v1 %8, 2v2 %9, 3v3 %10)  Pending: %11  Scheduled: %12"
                           "  Creates/Rebuilds/Reuses: %13/%14/%15")
        .arg(QString::fromStdString(progress.label))
        .arg((qlonglong)totalGames)
        .arg((qlonglong)m_latestProgressMaxGames)
        .arg(progress.snapshotGamesPerSecond, 0, 'f', 2)
        .arg(FormatElapsed(progress.elapsedSeconds))
        .arg(progress.activeArenas)
        .arg(progress.totalArenas)
        .arg(progress.active1v1Arenas)
        .arg(progress.active2v2Arenas)
        .arg(progress.active3v3Arenas)
        .arg(progress.pendingGames)
        .arg(progress.scheduledGames)
        .arg(progress.arenaCreates)
        .arg(progress.arenaRebuilds)
        .arg(progress.arenaReuses);
    if (progress.gpu.available) {
        constexpr double bytesPerGiB = 1024.0 * 1024.0 * 1024.0;
        line += QString("  GPU A/R/P: %1/%2/%3 GiB  Graphs: %4")
            .arg(progress.gpu.allocatedBytes / bytesPerGiB, 0, 'f', 2)
            .arg(progress.gpu.reservedBytes / bytesPerGiB, 0, 'f', 2)
            .arg(progress.gpu.peakReservedBytes / bytesPerGiB, 0, 'f', 2)
            .arg((qulonglong)progress.gpu.graphEntries);
    }
    if (progress.cudaOomRecoveries > 0)
        line += QString("  CUDA OOM recoveries: %1 (eager fallback)").arg(progress.cudaOomRecoveries);
    if (!progress.details.empty())
        line += "\n" + QString::fromStdString(progress.details);
    if (!progress.estimateSummary.empty())
        line += "\n" + QString::fromStdString(progress.estimateSummary);
    AppendConsoleLine(line);
}

void ModelBenchmarkWidget::OnExportResultsClicked()
{
    if (!m_hasLastResult)
        return;

    const std::filesystem::path exportDir = RocketForgeTraining::DefaultModelBenchmarkExportFolder();
    std::error_code exportDirError;
    std::filesystem::create_directories(exportDir, exportDirError);
    if (exportDirError || !std::filesystem::is_directory(exportDir)) {
        const QString error = exportDirError
            ? QString::fromStdString(exportDirError.message())
            : QString("Path exists but is not a folder.");
        QMessageBox::critical(
            this,
            "Export Benchmark Results",
            "Unable to create the default export folder:\n"
            + QString::fromStdString(exportDir.string())
            + "\n\n"
            + error);
        return;
    }

    const std::filesystem::path defaultPath = exportDir / RocketForgeTraining::DefaultModelBenchmarkExportName();
    const QString path = QFileDialog::getSaveFileName(
        this,
        "Export Benchmark Results",
        QString::fromStdString(defaultPath.string()),
        "Excel 2003 XML Workbook (*.xml)");
    if (path.isEmpty())
        return;

    RocketForgeTraining::ModelBenchmarkExportInput input;
    input.result = &m_lastResult;
    input.runId = m_lastResultRunId.toStdString();

    // Saved-results exports clear the run's entries; those models are then identified by their checkpoint alone.
    for (size_t modelIndex = 0; modelIndex < m_lastResult.modelNames.size() && modelIndex < m_lastRunVersions.size(); ++modelIndex) {
        const BenchmarkVersionEntry& entry = m_lastRunVersions[modelIndex];
        const ModelBenchmarkLibraryModel* libraryModel = modelIndex < m_lastRunLibraryModels.size()
            ? &m_lastRunLibraryModels[modelIndex]
            : nullptr;
        const std::filesystem::path checkpointPath = modelIndex < m_lastResult.modelCheckpointPaths.size()
            ? m_lastResult.modelCheckpointPaths[modelIndex]
            : std::filesystem::path();

        RocketForgeTraining::ModelBenchmarkExportModel model;
        model.agentName = entry.agentName;
        model.buildName = entry.buildName;
        model.stageName = entry.stageName;
        model.branchName = BranchNameOrDefault(entry.version.branchName);
        model.sourceLabel = entry.sourceLabel;
        model.lastTrainedAt = entry.version.lastTrainedAt;
        model.latestTimesteps = entry.version.latestTimesteps;
        model.primary = !libraryModel || libraryModel->primary;
        model.fingerprint = libraryModel
            ? libraryModel->fingerprint.toStdString()
            : RocketForgeTraining::FingerprintCheckpoint(checkpointPath, entry.trainingSetupPath);
        input.models.push_back(std::move(model));
    }

    if (m_consoleText) {
        for (const QString& line : m_consoleText->toPlainText().split('\n'))
            input.consoleLines.push_back(line.toStdString());
    }

    if (!m_exportEnvironmentCoverage.environments.empty()) {
        input.environmentComparisonNote = ModelBenchmarkHasMixedEnvironments(m_exportEnvironmentCoverage)
            ? "Mixed environments: pooled Elo reflects the included historical games; unequal simulator coverage can affect rankings."
            : "Single recorded environment (unrecorded details remain unknown).";
        input.extraWorksheets = [this](RocketForgeTraining::SpreadsheetXmlWriter& writer) {
            WriteModelBenchmarkEnvironmentWorksheet(writer, m_exportEnvironmentCoverage);
        };
    }

    std::string error;
    if (!RocketForgeTraining::WriteModelBenchmarkExport(std::filesystem::path(path.toStdWString()), input, &error)) {
        QMessageBox::critical(this, "Export Benchmark Results", "Unable to write the export file.\n\n" + QString::fromStdString(error));
        return;
    }

    AppendConsoleLine("Exported benchmark results to " + path);
}

void ModelBenchmarkWidget::OnBenchmarkComplete(const RocketForgeTraining::ModelBenchmarkResult& result)
{
    m_savedResultsWidget->UpdateLiveComparison(std::make_shared<const RocketForgeTraining::ModelBenchmarkResult>(result));
    m_currentRunUntilMaxGames = false;
    m_latestProgressGames = (int64_t)result.scoredGames + (int64_t)result.noResultGames;
    m_latestDrawGames = result.noResultGames;
    if (result.maxGames > 0)
        m_latestProgressMaxGames = result.maxGames;
    m_latestProgressGamesPerSecond = result.gamesPerSecond;
    m_latestProgressElapsedSeconds = result.elapsedSeconds;
    m_latestEstimatedRemainingSeconds =
        m_latestProgressGames >= m_latestProgressMaxGames ? 0.0 : -1.0;
    m_latestActiveArenas = 0;
    m_latestPendingGames = 0;
    m_latestArenaCreates = result.arenaCreates;
    m_latestArenaRebuilds = result.arenaRebuilds;
    m_latestArenaReuses = result.arenaReuses;
    m_latestActive1v1Arenas = 0;
    m_latestActive2v2Arenas = 0;
    m_latestActive3v3Arenas = 0;
    m_latestCudaOomRecoveries = result.cudaOomRecoveries;
    m_latestCudaGraphsDisabledAfterOom = result.cudaGraphsDisabledAfterOom;
    m_latestGpuTelemetry = result.gpu;
    m_progressPanel->SetDetails(BuildProgressDetailsStatus());
    PopulateSelectedTable(true);
    m_lastRunVersions = std::move(m_activeRunVersions);
    m_lastRunLibraryModels = std::move(m_activeRunLibraryModels);
    m_lastResult = result;
    m_hasLastResult = true;
    m_lastResultRunId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    m_lastResultSavedToActiveLibrary = false;
    if (m_exportResultsButton)
        m_exportResultsButton->setEnabled(true);
    if (m_saveResultsButton)
        m_saveResultsButton->setEnabled(true);
    const QString stopReason = QString::fromStdString(result.stopReason);
    const bool terminatedEarly = stopReason.startsWith("Terminated early:");
    const bool cancelled = result.stopReason == "Cancelled";
    m_progressPanel->EndRun(QString("%1 %2  Time Elapsed: %3")
        .arg(cancelled ? "Stopped:" : terminatedEarly ? "Terminated early:" : "Complete:")
        .arg(terminatedEarly ? stopReason.mid(QString("Terminated early:").length()).trimmed() : stopReason)
        .arg(FormatElapsedLong(result.elapsedSeconds)), !cancelled && !terminatedEarly);
    AppendConsoleLine(QString("%1 %2  Time Elapsed: %3")
        .arg(cancelled ? "Benchmark stopped:" : terminatedEarly ? "Benchmark terminated early:" : "Benchmark complete:")
        .arg(terminatedEarly ? stopReason.mid(QString("Terminated early:").length()).trimmed() : stopReason)
        .arg(FormatElapsedLong(result.elapsedSeconds)));
    AppendConsoleLine(QString("Strategy: %1  Throughput: %2 games/sec")
        .arg(QString::fromStdString(result.benchmarkStrategy))
        .arg(result.gamesPerSecond, 0, 'f', 2));
    AppendConsoleLine("RocketSim: " + QString::fromStdString(
        RocketForgeTraining::ModelBenchmarkRocketSimVersionsText(result.rocketSimVersions)));
    if (!result.timingSummary.empty())
        AppendConsoleLine(QString::fromStdString(result.timingSummary));
    SetSetupSectionVisible(false);
    SetRunning(false);

    if (!result.games.empty()) OnSaveResultsClicked();
    if (!m_lastResultSavedToActiveLibrary && !result.games.empty()) m_saveResultsButton->show();

    m_coverageDirty = true;
    if (m_lastResultSavedToActiveLibrary || result.games.empty()) RefreshWorkspace();
    SavePersistedSetup();
}

void ModelBenchmarkWidget::OnBenchmarkCancelled()
{
    m_savedResultsWidget->EndLiveComparison();
    m_activeRunVersions.clear();
    m_activeRunLibraryModels.clear();
    m_currentRunUntilMaxGames = false;
    m_latestActiveArenas = 0;
    m_latestPendingGames = 0;
    m_latestActive1v1Arenas = 0;
    m_latestActive2v2Arenas = 0;
    m_latestActive3v3Arenas = 0;
    m_latestGpuTelemetry = {};
    m_progressPanel->SetDetails(BuildProgressDetailsStatus());
    SetRunning(false);
    RefreshWorkspace();
    const bool markComplete = false;
    m_progressPanel->EndRun("Benchmark stopped before any completed games were available.", markComplete);
    AppendConsoleLine("Benchmark stopped before any completed games were available to save.");
}

void ModelBenchmarkWidget::OnBenchmarkFailed(const QString& message, bool simulatorUnavailable)
{
    m_savedResultsWidget->EndLiveComparison();
    m_activeRunVersions.clear();
    m_activeRunLibraryModels.clear();
    m_currentRunUntilMaxGames = false;
    m_latestActiveArenas = 0;
    m_latestPendingGames = 0;
    m_latestActive1v1Arenas = 0;
    m_latestActive2v2Arenas = 0;
    m_latestActive3v3Arenas = 0;
    m_latestGpuTelemetry = {};
    m_progressPanel->SetDetails(BuildProgressDetailsStatus());
    SetRunning(false);
    RefreshWorkspace();
    m_progressPanel->EndRun("Benchmark failed.", /*markComplete*/ false);
    AppendConsoleLine("Benchmark failed: " + message);
    if (simulatorUnavailable)
        ShowSimulatorUnavailable(this, message);
    else
        QMessageBox::critical(this, "Model Benchmarking", message);
}

void ModelBenchmarkWidget::UpdateInputState()
{
    m_maxGamesLabel->setText(m_scheduleModeCombo->currentIndex() == 0 ? "Target Games per Matchup:" : "New Games per Matchup:");
}

void ModelBenchmarkWidget::SetRunning(bool running)
{
    const bool changed = m_running != running;
    m_running = running;
    m_presetControls->setEnabled(!running);
    m_modelControls->setEnabled(!running);
    m_makeAllButton->setEnabled(!running);
    m_refreshVersionsButton->setEnabled(!running && !m_catalogScan.valid());
    m_agentCombo->setEnabled(!running);
    m_buildCombo->setEnabled(!running);
    m_stageCombo->setEnabled(!running);
    m_versionTable->setEnabled(!running);
    m_selectedTable->setEnabled(!running);

    // Rows rebuilt during progress explicitly disable their role widgets.
    // Enabling the parent table alone does not clear that child-widget state.
    for (int row = 0; row < m_selectedTable->rowCount(); ++row) {
        if (QWidget* role = m_selectedTable->cellWidget(row, SelectedColumnPrimary))
            role->setEnabled(!running);
    }

    m_addVersionsButton->setEnabled(!running);
    m_addExternalFolderButton->setEnabled(!running);
    m_hideImportedModelsCheck->setEnabled(!running);
    if (m_availableBranchCombo)
        m_availableBranchCombo->setEnabled(!running);
    if (m_availableSearchEdit)
        m_availableSearchEdit->setEnabled(!running);
    m_removeSelectedButton->setEnabled(!running);
    m_clearSelectedButton->setEnabled(!running);
    m_arenasSpin->setEnabled(!running);
    m_maxGamesSpin->setEnabled(!running);
    m_seedEdit->setEnabled(!running);
    m_mode1v1Check->setEnabled(!running);
    m_mode2v2Check->setEnabled(!running);
    m_mode3v3Check->setEnabled(!running);
    m_stochasticKickoffCheck->setEnabled(!running);
    m_kickoffTemperatureSpin->setEnabled(!running && m_stochasticKickoffCheck->isChecked());
    m_engineCombo->setEnabled(!running);
    m_runButton->setEnabled(!running && !m_trainingActive);
    m_runButton->setToolTip(m_trainingActive
        ? "Training is running. Stop training before starting a model benchmark."
        : RunBenchmarkTooltip);
    m_cancelButton->setEnabled(running);
    if (m_exportResultsButton)
        m_exportResultsButton->setEnabled(!running);
    if (m_saveResultsButton)
        m_saveResultsButton->setEnabled(
            !running
            && m_hasLastResult
            && !m_lastResultRunId.isEmpty()
            && !m_lastResultSavedToActiveLibrary);
    UpdateInputState();
    if (changed)
        emit RunningChanged(running);
}

void ModelBenchmarkWidget::SetSetupSectionVisible(bool visible)
{
    if (m_setupContainer)
        m_setupContainer->setVisible(true);
    UpdateSectionToggleText();
}

void ModelBenchmarkWidget::UpdateSectionToggleText()
{
    if (m_toggleSetupButton)
        m_toggleSetupButton->setText("Test Setup…");
}

void ModelBenchmarkWidget::AppendConsoleLine(const QString& text)
{
    if (!m_consoleText)
        return;
    const QString timestamp = QDateTime::currentDateTime().toString("HH:mm:ss");
    m_consoleText->appendPlainText(QString("[%1] %2").arg(timestamp, text));
    m_consoleText->verticalScrollBar()->setValue(m_consoleText->verticalScrollBar()->maximum());
}

QString ModelBenchmarkWidget::BuildProgressStatus(double elapsedSeconds) const
{
    QString label = QString("%1  Games: %2/%3  %4 games/sec  Elapsed: %5")
        .arg(m_latestProgressTitle)
        .arg((qlonglong)m_latestProgressGames)
        .arg((qlonglong)m_latestProgressMaxGames)
        .arg(m_latestProgressGamesPerSecond, 0, 'f', 2)
        .arg(FormatElapsed(elapsedSeconds));
    label += QString("  Draws: %1").arg(m_latestDrawGames);

    if (m_currentRunUntilMaxGames) {
        const int64_t remainingGames = std::max<int64_t>(
            0,
            m_latestProgressMaxGames - m_latestProgressGames);
        double etaSeconds = -1.0;
        if (m_latestEstimatedRemainingSeconds >= 0.0) {
            const double sinceEstimate = std::max(0.0, elapsedSeconds - m_latestProgressElapsedSeconds);
            etaSeconds = std::max(0.0, m_latestEstimatedRemainingSeconds - sinceEstimate);
        }

        if (etaSeconds >= 0.0)
            label += QString("  ETA: %1").arg(FormatElapsed(etaSeconds));
        else if (remainingGames > 0)
            label += "  ETA: --";
        else
            label += "  ETA: 0:00";
    } else {
        label += "  ETA: --";
    }

    return label;
}

QString ModelBenchmarkWidget::BuildProgressDetailsStatus() const
{
    const int64_t scheduledGames = std::max<int64_t>(0, (int64_t)m_latestScheduledGames);
    const int64_t unscheduledGames = std::max<int64_t>(
        0,
        m_latestProgressMaxGames - scheduledGames);
    QString label = QString("Unscheduled: %1  Pending: %2  Active: %3/%4 "
                            "(1v1: %5, 2v2: %6, 3v3: %7)  Completed: %8  "
                            "Arenas C/R/R: %9/%10/%11")
        .arg((qlonglong)unscheduledGames)
        .arg(m_latestPendingGames)
        .arg(m_latestActiveArenas)
        .arg(m_latestTotalArenas)
        .arg(m_latestActive1v1Arenas)
        .arg(m_latestActive2v2Arenas)
        .arg(m_latestActive3v3Arenas)
        .arg((qlonglong)m_latestProgressGames)
        .arg(m_latestArenaCreates)
        .arg(m_latestArenaRebuilds)
        .arg(m_latestArenaReuses);

    if (m_latestGpuTelemetry.available) {
        constexpr double bytesPerGiB = 1024.0 * 1024.0 * 1024.0;
        label += QString("  GPU A/R: %1/%2 GiB  Graphs: %3")
            .arg(m_latestGpuTelemetry.allocatedBytes / bytesPerGiB, 0, 'f', 1)
            .arg(m_latestGpuTelemetry.reservedBytes / bytesPerGiB, 0, 'f', 1)
            .arg((qulonglong)m_latestGpuTelemetry.graphEntries);
    }
    if (m_latestCudaGraphsDisabledAfterOom) {
        label += QString("  OOM fallback: eager (%1)")
            .arg((qlonglong)m_latestCudaOomRecoveries);
    }
    return label;
}

std::string ModelBenchmarkWidget::CaptureModelSetup(const BenchmarkVersionEntry& entry) const
{
    if (!entry.agentSetupJson.empty()) return entry.agentSetupJson;

    if (entry.agentSetupSource == RocketForgeTraining::ModelBenchmarkAgentSetupSource::SavedSegment) {
        const std::optional<RocketForge::TrainingSetupSnapshot> saved = RocketForge::LoadTrainingSetupSnapshot(entry.trainingSetupPath);
        if (!saved || !saved->agentSetup)
            throw std::runtime_error("Saved Agent setup is unavailable for " + MakeDisplayName(entry));

        return RocketForge::AgentSetupSnapshotToJson(*saved->agentSetup).dump();
    }

    RocketForgeTraining::ModelBenchmarkConfig config;
    RocketForgeTraining::ModelBenchmarkModel model;
    model.agentName = entry.agentName;
    model.displayName = MakeDisplayName(entry);
    model.checkpointPath = entry.checkpointPath;
    model.trainingSetupPath = entry.trainingSetupPath;
    model.agentSetupSource = entry.agentSetupSource;
    config.models.push_back(std::move(model));

    return RocketForgeTraining::PrepareModelBenchmarkPrediction(config).models.front().agentSetupJson;
}

void ModelBenchmarkWidget::FollowTrainingAgent(const QString& agentName)
{
    m_agentCombo->FollowAgent(agentName);
}
