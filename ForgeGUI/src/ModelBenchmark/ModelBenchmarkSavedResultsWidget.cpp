#include "ModelBenchmarkSavedResultsWidget.h"
#include "ModelBenchmarkResultTables.h"

#include <QAbstractItemView>
#include <QColor>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QInputDialog>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSplitter>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QTabWidget>
#include <QThread>
#include <QTimer>
#include <QUuid>
#include <QVBoxLayout>

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <set>

using namespace RocketForgeTraining;
using namespace ForgeGUI::ModelBenchmarkTables;

namespace
{
    constexpr int ModelColumnModel = 0;
    constexpr int PoolColumnInclude = 0;
    constexpr int PoolColumnPrimary = 1;
    constexpr int PoolColumnModel = 2;

    constexpr const char* AllAgents = "All Agents";
    constexpr const char* AllBuilds = "All Builds";
    constexpr const char* AllStages = "All Stages";
    constexpr const char* AllBranches = "All Branches";

    constexpr const char* IncludeTooltip =
        "<qt>Check to show this model in the results; uncheck to hide it. Saved data is kept.</qt>";
    constexpr const char* PrimaryTooltip =
        "Primary models include their games against every included model. Games between two included non-primary models are excluded.";

    QString BranchDisplayName(const QString& branchName)
    {
        const QString trimmed = branchName.trimmed();
        return trimmed.isEmpty() ? "Unknown" : trimmed;
    }

    QTableWidgetItem* ReadOnly(const QString& text)
    {
        return ReadOnlyItem(text);
    }

    QTableWidgetItem* CheckItem(bool checked, bool enabled = true)
    {
        auto* item = new QTableWidgetItem;
        Qt::ItemFlags flags = Qt::ItemIsUserCheckable | Qt::ItemIsSelectable;
        if (enabled)
            flags |= Qt::ItemIsEnabled;
        item->setFlags(flags);
        item->setCheckState(checked ? Qt::Checked : Qt::Unchecked);
        item->setTextAlignment(Qt::AlignCenter);
        return item;
    }

    void PrepareTable(QTableWidget* table, const QStringList& headers, int rows)
    {
        table->clear();
        table->setColumnCount(headers.size());
        table->setHorizontalHeaderLabels(headers);
        table->setRowCount(rows);
        table->setEditTriggers(QAbstractItemView::NoEditTriggers);
        table->setSelectionBehavior(QAbstractItemView::SelectRows);
        table->horizontalHeader()->setStretchLastSection(true);
    }

    QColor BlendColor(const QColor& base, const QColor& accent, double accentAmount)
    {
        const double baseAmount = 1.0 - accentAmount;
        return QColor(
            (int)(base.red() * baseAmount + accent.red() * accentAmount),
            (int)(base.green() * baseAmount + accent.green() * accentAmount),
            (int)(base.blue() * baseAmount + accent.blue() * accentAmount));
    }

    std::vector<std::vector<int>> FindShortestCoveragePaths(
        const std::vector<std::vector<int>>& adjacency,
        int source,
        int target,
        int maxPaths = 3)
    {
        const int count = (int)adjacency.size();
        if (source < 0 || target < 0 || source >= count || target >= count || source == target)
            return {};

        std::vector<int> distance(count, -1);
        std::vector<std::vector<int>> parents(count);
        std::queue<int> pending;
        distance[source] = 0;
        pending.push(source);
        while (!pending.empty()) {
            const int current = pending.front();
            pending.pop();
            if (distance[target] >= 0 && distance[current] + 1 > distance[target])
                continue;
            for (int next : adjacency[current]) {
                if (distance[next] < 0) {
                    distance[next] = distance[current] + 1;
                    parents[next].push_back(current);
                    pending.push(next);
                } else if (distance[next] == distance[current] + 1) {
                    parents[next].push_back(current);
                }
            }
        }
        if (distance[target] < 0)
            return {};

        std::vector<std::vector<int>> paths;
        std::vector<int> reversePath{ target };
        std::function<void(int)> collect = [&](int current) {
            if ((int)paths.size() >= maxPaths)
                return;
            if (current == source) {
                paths.emplace_back(reversePath.rbegin(), reversePath.rend());
                return;
            }
            for (int parent : parents[current]) {
                reversePath.push_back(parent);
                collect(parent);
                reversePath.pop_back();
                if ((int)paths.size() >= maxPaths)
                    return;
            }
        };
        collect(target);
        return paths;
    }

    QString FormatCoveragePath(const QStringList& modelNames, const std::vector<int>& path)
    {
        QStringList names;
        for (int modelIndex : path) {
            if (modelIndex >= 0 && modelIndex < modelNames.size())
                names << modelNames[modelIndex];
        }
        return names.join(QString::fromUtf8(" \xE2\x86\x92 "));
    }

    struct AggregateLoadTask
    {
        ModelBenchmarkLibraryFilter filter;
        quint64 requestId = 0;
        std::filesystem::path libraryPath;
        std::vector<ModelBenchmarkLibrarySelection> selection;
        std::vector<bool> primaryFlags;
        float confidence = 0.95f;
        std::shared_ptr<std::atomic_bool> cancelled = std::make_shared<std::atomic_bool>(false);
        ModelBenchmarkLibraryAggregate aggregate;
        std::shared_ptr<const ModelBenchmarkLibraryAggregate> baseline;
        std::shared_ptr<const ModelBenchmarkResult> liveResult;
        std::vector<ModelBenchmarkLibraryModel> additionalModels;
        std::vector<qint64> liveModelIds;
        bool liveComparison = false;
        QString error;
    };

    struct FingerprintTask
    {
        quint64 requestId = 0;
        std::vector<ModelBenchmarkLibraryModel> known;
        std::set<QString> savedLogicalKeys;
        std::shared_ptr<std::atomic_bool> cancelled = std::make_shared<std::atomic_bool>(false);
    };
}

ModelBenchmarkSavedResultsWidget::ModelBenchmarkSavedResultsWidget(QWidget* parent)
    : QWidget(parent)
{
    setObjectName("modelBenchmarkSavedResultsPage");
    setAttribute(Qt::WA_StyledBackground, true);

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(8, 8, 8, 8);
    root->setSpacing(10);

    auto* libraryRow = new QHBoxLayout;
    libraryRow->addWidget(new QLabel("Library:", this));
    m_libraryPathLabel = new QLabel("No library open", this);
    m_libraryPathLabel->setMinimumWidth(0);
    m_libraryPathLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    m_libraryPathLabel->setTextFormat(Qt::PlainText);
    m_libraryPathLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    libraryRow->addWidget(m_libraryPathLabel, 1);
    auto* newButton = new QPushButton("New...", this);
    auto* openButton = new QPushButton("Open...", this);
    auto* importButton = new QPushButton("Import Excel...", this);
    m_exportButton = new QPushButton("Export Results...", this);
    m_exportButton->setEnabled(false);
    libraryRow->addWidget(newButton);
    libraryRow->addWidget(openButton);
    libraryRow->addWidget(importButton);
    libraryRow->addWidget(m_exportButton);
    root->addLayout(libraryRow);

    auto* comparisonRow = new QHBoxLayout;
    comparisonRow->setSpacing(8);
    comparisonRow->addWidget(new QLabel("Saved comparison:", this));
    m_comparisonCombo = new QComboBox(this);
    m_comparisonCombo->setMinimumContentsLength(22);
    m_comparisonCombo->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
    m_comparisonCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    m_comparisonCombo->setToolTip("Choose a named comparison from this library, then click Load.");
    comparisonRow->addWidget(m_comparisonCombo, 1);

    m_loadComparisonButton = new QPushButton("Load", this);
    m_newComparisonButton = new QPushButton("New Set", this);
    m_saveComparisonButton = new QPushButton("Save", this);
    m_saveComparisonAsButton = new QPushButton("Save As...", this);
    m_loadComparisonButton->setToolTip("Restore the saved bots, Show and Primary settings.");
    m_newComparisonButton->setToolTip("Start an empty comparison set.");
    m_saveComparisonButton->setToolTip("Save changes to this comparison, or name a new set.");
    m_saveComparisonAsButton->setToolTip("Save this comparison under a new name, keeping the original set.");
    for (QPushButton* button : { m_loadComparisonButton, m_newComparisonButton, m_saveComparisonButton, m_saveComparisonAsButton }) {
        button->setMinimumHeight(30);
        comparisonRow->addWidget(button);
    }
    root->addLayout(comparisonRow);

    auto* comparisonInfo = new QHBoxLayout;
    m_comparisonStatusLabel = new QLabel(this);
    m_comparisonStatusLabel->setTextFormat(Qt::PlainText);
    m_comparisonStatusLabel->setWordWrap(true);
    comparisonInfo->addWidget(m_comparisonStatusLabel, 1);
    root->addLayout(comparisonInfo);

    auto* filterRow = new QHBoxLayout;
    m_agentFilterCombo = new QComboBox(this);
    m_buildFilterCombo = new QComboBox(this);
    m_stageFilterCombo = new QComboBox(this);
    m_branchFilterCombo = new QComboBox(this);
    for (QComboBox* filter : { m_agentFilterCombo, m_buildFilterCombo, m_stageFilterCombo, m_branchFilterCombo }) {
        filter->setMinimumContentsLength(8);
        filter->setMinimumHeight(30);
        filter->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
        filter->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    }

    filterRow->addWidget(new QLabel("Agent:", this));
    filterRow->addWidget(m_agentFilterCombo);
    filterRow->addWidget(new QLabel("Build:", this));
    filterRow->addWidget(m_buildFilterCombo);
    filterRow->addWidget(new QLabel("Stage:", this));
    filterRow->addWidget(m_stageFilterCombo);
    filterRow->addWidget(new QLabel("Branch:", this));
    filterRow->addWidget(m_branchFilterCombo);

    auto* modelsPane = new QWidget(this);
    auto* modelsLayout = new QVBoxLayout(modelsPane);
    modelsLayout->setContentsMargins(0, 0, 0, 0);
    modelsLayout->setSpacing(8);

    auto* modelsHeader = new QHBoxLayout;
    auto* modelsTitle = new QLabel("Available Models", modelsPane);
    QFont headingFont = modelsTitle->font();
    headingFont.setBold(true);
    modelsTitle->setFont(headingFont);
    modelsHeader->addWidget(modelsTitle);

    auto* addHint = new QLabel("Click a saved model to add it to the comparison set", modelsPane);
    addHint->setEnabled(false);
    addHint->setWordWrap(true);
    modelsHeader->addWidget(addHint, 1);
    modelsHeader->addStretch();
    modelsLayout->addLayout(modelsHeader);

    m_searchEdit = new QLineEdit(modelsPane);
    m_searchEdit->setMinimumHeight(30);
    m_searchEdit->setPlaceholderText("Search agent, build, stage, branch, version, or snapshot");
    m_searchEdit->setClearButtonEnabled(true);
    modelsLayout->addWidget(m_searchEdit);
    modelsLayout->addLayout(filterRow);

    m_modelTable = new QTableWidget(0, 8, modelsPane);
    m_modelTable->setHorizontalHeaderLabels({
        "Model", "Agent", "Build", "Stage", "Branch", "Version", "Snapshot", "Local"
    });
    m_modelTable->setSelectionMode(QAbstractItemView::SingleSelection);
    m_modelTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_modelTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_modelTable->horizontalHeader()->setStretchLastSection(false);
    m_modelTable->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    m_modelTable->horizontalHeader()->setSectionResizeMode(ModelColumnModel, QHeaderView::Stretch);
    m_modelTable->setMinimumHeight(90);
    modelsLayout->addWidget(m_modelTable, 1);

    auto* poolPane = new QWidget(this);
    auto* poolLayout = new QVBoxLayout(poolPane);
    poolLayout->setContentsMargins(0, 0, 0, 0);
    poolLayout->setSpacing(8);

    auto* poolHeader = new QHBoxLayout;
    auto* poolTitle = new QLabel("Comparison Set", poolPane);
    poolTitle->setFont(headingFont);
    poolHeader->addWidget(poolTitle);

    m_poolStatusLabel = new QLabel("0 models", poolPane);
    m_poolStatusLabel->setEnabled(false);
    poolHeader->addWidget(m_poolStatusLabel);
    poolHeader->addStretch();
    poolLayout->addLayout(poolHeader);

    auto* inclusionRow = new QHBoxLayout;
    inclusionRow->setSpacing(8);
    m_includeAllButton = new QPushButton("Show All", this);
    m_includeAllButton->setToolTip("Reactivate every model currently in the comparison set.");
    m_excludeAllButton = new QPushButton("Hide All", this);
    m_excludeAllButton->setToolTip("Temporarily deactivate every model while keeping the comparison set intact.");
    m_removeButton = new QPushButton("Remove Selected", this);
    m_removeButton->setToolTip("Remove selected models from this comparison. Saved benchmark data is kept.");
    inclusionRow->addWidget(m_includeAllButton);
    inclusionRow->addWidget(m_excludeAllButton);
    inclusionRow->addStretch();
    inclusionRow->addWidget(m_removeButton);
    poolLayout->addLayout(inclusionRow);

    auto* primaryRow = new QHBoxLayout;
    primaryRow->setSpacing(8);
    m_primaryAllButton = new QPushButton("All Primary", this);
    m_primaryNoneButton = new QPushButton("None Primary", this);
    m_primaryAllButton->setToolTip("<qt>Make every model in this comparison Primary, including temporarily excluded models.</qt>");
    m_primaryNoneButton->setToolTip("<qt>Clear Primary for every model in this comparison, including temporarily excluded models.</qt>");
    primaryRow->addWidget(m_primaryAllButton);
    primaryRow->addWidget(m_primaryNoneButton);
    primaryRow->addStretch();
    poolLayout->addLayout(primaryRow);

    for (QPushButton* button : { m_includeAllButton, m_excludeAllButton, m_removeButton, m_primaryAllButton, m_primaryNoneButton })
        button->setMinimumHeight(30);

    m_poolTable = new QTableWidget(0, 3, poolPane);
    m_poolTable->setHorizontalHeaderLabels({ "Show", "Primary", "Model" });
    m_poolTable->horizontalHeaderItem(PoolColumnInclude)->setToolTip(IncludeTooltip);
    m_poolTable->horizontalHeaderItem(PoolColumnPrimary)->setToolTip(PrimaryTooltip);
    m_poolTable->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_poolTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_poolTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    InstallCenteredCheckBoxDelegate(m_poolTable, PoolColumnInclude);
    InstallCenteredCheckBoxDelegate(m_poolTable, PoolColumnPrimary);
    InstallModelColorDelegate(m_poolTable);
    m_poolTable->horizontalHeader()->setStretchLastSection(true);
    m_poolTable->horizontalHeader()->setSectionResizeMode(PoolColumnInclude, QHeaderView::ResizeToContents);
    m_poolTable->horizontalHeader()->setSectionResizeMode(PoolColumnPrimary, QHeaderView::ResizeToContents);
    m_poolTable->setMinimumWidth(410);
    m_poolTable->setMinimumHeight(100);
    poolLayout->addWidget(m_poolTable, 1);

    auto* selectionSplitter = new QSplitter(Qt::Horizontal, this);
    selectionSplitter->setChildrenCollapsible(false);
    selectionSplitter->setHandleWidth(8);
    selectionSplitter->addWidget(modelsPane);
    selectionSplitter->addWidget(poolPane);
    selectionSplitter->setStretchFactor(0, 2);
    selectionSplitter->setStretchFactor(1, 1);
    selectionSplitter->setSizes({ 840, 460 });
    selectionSplitter->handle(1)->setToolTip("Drag to resize the model browser and comparison set.");

    m_environmentControls = new QWidget(this);
    auto* environmentLayout = new QVBoxLayout(m_environmentControls);
    environmentLayout->setContentsMargins(0, 0, 0, 0);
    m_environmentSummaryLabel = new QLabel(this);
    m_environmentSummaryLabel->setTextFormat(Qt::PlainText);
    m_environmentSummaryLabel->setWordWrap(true);
    environmentLayout->addWidget(m_environmentSummaryLabel);
    root->addWidget(m_environmentControls);

    m_resultTabs = new QTabWidget(this);
    m_resultTabs->setObjectName("benchmarkResultTabs");
    m_rankingTable = new QTableWidget(m_resultTabs);
    m_breakdownTable = new QTableWidget(m_resultTabs);
    m_pairwiseTable = new QTableWidget(m_resultTabs);
    InstallModelColorDelegate(m_rankingTable);
    InstallModelColorDelegate(m_breakdownTable);
    InstallModelColorDelegate(m_pairwiseTable);
    auto* coveragePage = new QWidget(m_resultTabs);
    auto* coverageLayout = new QVBoxLayout(coveragePage);
    coverageLayout->setContentsMargins(0, 0, 0, 0);
    auto* coverageLegend = new QLabel(
        "Cells show the row model's combined record. Green = direct games, gray = indirect Elo path, "
        "red = disconnected, dimmed = excluded by the current Elo matchup filter.",
        coveragePage);
    coverageLegend->setWordWrap(true);
    coverageLayout->addWidget(coverageLegend);
    m_coverageTable = new QTableWidget(coveragePage);
    m_coverageTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_coverageTable->setSelectionMode(QAbstractItemView::SingleSelection);
    m_coverageTable->setSelectionBehavior(QAbstractItemView::SelectItems);
    m_coverageTable->setWordWrap(true);
    coverageLayout->addWidget(m_coverageTable, 1);
    auto* statsPage = new QWidget(m_resultTabs);
    auto* statsLayout = new QVBoxLayout(statsPage);
    statsLayout->setContentsMargins(0, 0, 0, 0);
    auto* statsFilterLayout = new QHBoxLayout;
    statsFilterLayout->addWidget(new QLabel("Mode:", statsPage));
    m_statsModeCombo = new QComboBox(statsPage);
    m_statsModeCombo->addItem("All Modes", 0);
    m_statsModeCombo->setToolTip("Show all saved benchmark modes together or one team size at a time.");
    statsFilterLayout->addWidget(m_statsModeCombo);
    statsFilterLayout->addSpacing(8);
    statsFilterLayout->addWidget(new QLabel(
        "Event stats are averages per 5-minute game.",
        statsPage));
    statsFilterLayout->addStretch();
    statsLayout->addLayout(statsFilterLayout);
    m_statsTable = new QTableWidget(statsPage);
    InstallModelColorDelegate(m_statsTable);
    statsLayout->addWidget(m_statsTable, 1);
    m_detailsTable = new QTableWidget(m_resultTabs);
    m_resultTabs->addTab(m_rankingTable, "Overall Ranking");
    const int breakdownTabIndex = m_resultTabs->addTab(m_breakdownTable, "Kickoff Breakdown");
    m_resultTabs->setTabToolTip(
        breakdownTabIndex,
        "Overall Elo uses every scored game. Kickoff and post-kickoff Elo split those same games by goal time. "
        "All three columns use the worst overall model's overall rating as zero, so timing Elo may be negative. "
        "Timing estimates show their confidence range and sample count; N/A means the subset does not connect every selected model.");
    m_resultTabs->addTab(m_pairwiseTable, "Pairwise Summary");
    m_resultTabs->addTab(statsPage, "Detailed Stats");
    m_resultTabs->addTab(coveragePage, "Match Coverage");
    m_environmentCoverageTable = new QTableWidget(m_resultTabs);
    m_resultTabs->addTab(m_environmentCoverageTable, "Environment Coverage");
    m_resultTabs->setTabToolTip(m_resultTabs->indexOf(m_environmentCoverageTable),
        "Completed games per model and environment, including draws. Each game counts once in the total row and once for each participating model. Zeroes reveal gaps in simulator coverage.");
    m_resultTabs->addTab(m_detailsTable, "Details");
    m_resultTabs->setTabToolTip(m_resultTabs->indexOf(m_rankingTable), "Compare model strength using the selected saved games. Higher Elo is stronger within this comparison; ranges show uncertainty. Hover over a heading for its definition.");
    m_resultTabs->setTabToolTip(m_resultTabs->indexOf(m_pairwiseTable), "See each model's wins, losses, and draws against individual opponents, split by team size.");
    m_resultTabs->setTabToolTip(m_resultTabs->indexOf(coveragePage), "<qt>See which models have played each other. Direct means saved games exist. Indirect means shared opponents link their Elo estimates. No path means a shared rating cannot be estimated. Hover over a cell to see the connection.</qt>");
    m_resultTabs->setTabToolTip(m_resultTabs->indexOf(statsPage), "Extra gameplay measures from saved runs that recorded them, such as touches and boost use. They do not determine Elo. Hover over a column heading for its definition.");
    m_resultTabs->setTabToolTip(m_resultTabs->indexOf(m_detailsTable), "See contributing run counts, dates, game totals, physics engines, and warnings about missing or incompatible saved data.");
    m_resultTabs->setMinimumHeight(160);

    auto* resultsSplitter = new QSplitter(Qt::Vertical, this);
    resultsSplitter->setChildrenCollapsible(false);
    resultsSplitter->setHandleWidth(8);
    resultsSplitter->addWidget(selectionSplitter);
    resultsSplitter->addWidget(m_resultTabs);
    resultsSplitter->setStretchFactor(0, 0);
    resultsSplitter->setStretchFactor(1, 1);
    resultsSplitter->setSizes({ 330, 450 });
    resultsSplitter->handle(1)->setToolTip("Drag to resize the model selection and saved results sections.");
    root->addWidget(resultsSplitter, 1);

    SetEmptyResults("Add at least two saved models to the comparison set and leave them included.");

    connect(newButton, &QPushButton::clicked, this, [this]() { ChooseLibrary(true); });
    connect(openButton, &QPushButton::clicked, this, [this]() { ChooseLibrary(false); });
    connect(importButton, &QPushButton::clicked, this, &ModelBenchmarkSavedResultsWidget::ImportExcel);
    connect(m_exportButton, &QPushButton::clicked, this, [this]() {
        if (m_hasDisplayedResult)
            ExportDisplayedResults();
    });
    connect(m_searchEdit, &QLineEdit::textChanged, this, [this]() { PopulateModelTable(); });
    for (QComboBox* filter : { m_agentFilterCombo, m_buildFilterCombo, m_stageFilterCombo, m_branchFilterCombo }) {
        connect(filter, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this]() {
            PopulateModelTable();
        });
    }
    connect(m_includeAllButton, &QPushButton::clicked, this, [this]() {
        for (qint64 modelId : m_pooledModelIds)
            m_includedByModelId[modelId] = true;
        RebuildResults();
        PopulatePoolTable();
    });
    connect(m_excludeAllButton, &QPushButton::clicked, this, [this]() {
        for (qint64 modelId : m_pooledModelIds)
            m_includedByModelId[modelId] = false;
        RebuildResults();
        PopulatePoolTable();
    });
    connect(m_removeButton, &QPushButton::clicked, this, &ModelBenchmarkSavedResultsWidget::RemoveSelectedPoolModels);
    for (QPushButton* button : { m_primaryAllButton, m_primaryNoneButton }) {
        connect(button, &QPushButton::clicked, this, [this, button]() {
            for (qint64 modelId : m_pooledModelIds)
                m_primaryByModelId[modelId] = button == m_primaryAllButton;
            RebuildResults();
            PopulatePoolTable();
        });
    }
    connect(m_poolTable, &QTableWidget::itemSelectionChanged, this, &ModelBenchmarkSavedResultsWidget::UpdateComparisonControls);
    connect(m_comparisonCombo, qOverload<int>(&QComboBox::currentIndexChanged), this,
        &ModelBenchmarkSavedResultsWidget::UpdateComparisonControls);
    connect(m_loadComparisonButton, &QPushButton::clicked, this, &ModelBenchmarkSavedResultsWidget::LoadComparisonSet);
    connect(m_newComparisonButton, &QPushButton::clicked, this, &ModelBenchmarkSavedResultsWidget::NewComparisonSet);
    connect(m_saveComparisonButton, &QPushButton::clicked, this, [this]() {
        const bool saveAs = false;
        SaveComparisonSet(saveAs);
    });
    connect(m_saveComparisonAsButton, &QPushButton::clicked, this, [this]() {
        const bool saveAs = true;
        SaveComparisonSet(saveAs);
    });
    connect(m_resultTabs, &QTabWidget::currentChanged, this, [this]() {
        PopulateCurrentResultTab();
    });
    connect(m_statsModeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this]() {
        PopulateStatsTable();
    });
    connect(m_modelTable, &QTableWidget::cellClicked, this, [this](int row, int) {
        QTableWidgetItem* modelItem = m_modelTable->item(row, ModelColumnModel);
        if (!modelItem)
            return;
        const qint64 modelId = modelItem->data(Qt::UserRole).toLongLong();
        if (modelId < 0)
            return;
        const bool newlyPooled = m_pooledModelIds.insert(modelId).second;
        if (!newlyPooled)
            return;
        m_includedByModelId[modelId] = true;
        m_primaryByModelId.try_emplace(modelId, true);
        UpdateComparisonControls();
        QueueSelectionRefresh();
    });
    connect(m_poolTable, &QTableWidget::itemChanged, this, [this](QTableWidgetItem* item) {
        if (!item || (item->column() != PoolColumnInclude && item->column() != PoolColumnPrimary))
            return;
        QTableWidgetItem* modelItem = m_poolTable->item(item->row(), PoolColumnModel);
        if (!modelItem)
            return;
        const qint64 modelId = modelItem->data(Qt::UserRole).toLongLong();
        if (!m_pooledModelIds.contains(modelId))
            return;
        if (item->column() == PoolColumnInclude)
            m_includedByModelId[modelId] = item->checkState() == Qt::Checked;
        else
            m_primaryByModelId[modelId] = item->checkState() == Qt::Checked;
        UpdateComparisonControls();
        QueueResultsRefresh();
    });

    RefreshComparisonSets();
}

ModelBenchmarkSavedResultsWidget::~ModelBenchmarkSavedResultsWidget()
{
    if (m_fingerprintCancellation)
        m_fingerprintCancellation->store(true);
    if (m_fingerprintThread) {
        m_fingerprintThread->quit();
        m_fingerprintThread->wait();
    }

    if (m_aggregateCancellation)
        m_aggregateCancellation->store(true);
    if (m_aggregateThread) {
        m_aggregateThread->quit();
        m_aggregateThread->wait();
    }
}

ModelBenchmarkComparisonSet ModelBenchmarkSavedResultsWidget::CurrentComparison() const
{
    ModelBenchmarkComparisonSet comparison;
    comparison.id = m_savedComparison.id;
    comparison.name = m_savedComparison.name;
    comparison.confidencePercent = 95.0;

    for (qint64 id : m_pooledModelIds) {
        const auto included = m_includedByModelId.find(id);
        const auto primary = m_primaryByModelId.find(id);
        comparison.members.push_back({ id,
            included == m_includedByModelId.end() || included->second,
            primary == m_primaryByModelId.end() || primary->second });
    }

    return comparison;
}

bool ModelBenchmarkSavedResultsWidget::IsComparisonModified() const
{
    const ModelBenchmarkComparisonSet current = CurrentComparison();
    return current.members != m_savedComparison.members;
}

bool ModelBenchmarkSavedResultsWidget::ConfirmComparisonChange()
{
    if (m_workspaceMode) return true;
    if (!IsComparisonModified())
        return true;

    const QMessageBox::StandardButton answer = QMessageBox::question(this, "Unsaved Comparison",
        "Save changes to the current comparison before continuing?",
        QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel, QMessageBox::Save);
    if (answer == QMessageBox::Save) {
        const bool saveAs = false;
        return SaveComparisonSet(saveAs);
    }

    return answer == QMessageBox::Discard;
}

void ModelBenchmarkSavedResultsWidget::RefreshComparisonSets()
{
    QSignalBlocker blocker(m_comparisonCombo);
    m_comparisonCombo->clear();
    m_comparisonCombo->addItem("Select a saved comparison...", qint64(-1));

    QString error;
    if (m_library.IsOpen()) {
        for (const ModelBenchmarkComparisonSet& comparison : m_library.ListComparisonSets(&error)) {
            m_comparisonCombo->addItem(comparison.name, comparison.id);
            m_comparisonCombo->setItemData(m_comparisonCombo->count() - 1,
                QString("%1\n%2 models | %3% confidence")
                    .arg(comparison.name).arg(comparison.members.size()).arg(95.0),
                Qt::ToolTipRole);
        }
    }

    const int savedIndex = m_comparisonCombo->findData(m_savedComparison.id);
    m_comparisonCombo->setCurrentIndex(savedIndex >= 0 ? savedIndex : 0);
    UpdateComparisonControls();

    if (!error.isEmpty())
        QMessageBox::critical(this, "Saved Comparisons", error);
}

void ModelBenchmarkSavedResultsWidget::UpdateComparisonControls()
{
    const bool open = m_library.IsOpen();
    const bool modified = IsComparisonModified();
    const bool hasMembers = !m_pooledModelIds.empty();

    m_comparisonCombo->setEnabled(open && m_comparisonCombo->count() > 1);
    m_loadComparisonButton->setEnabled(open && m_comparisonCombo->currentData().toLongLong() >= 0);
    m_newComparisonButton->setEnabled(open);
    m_saveComparisonButton->setEnabled(open && (m_savedComparison.id >= 0 ? modified : hasMembers));
    m_saveComparisonAsButton->setEnabled(open && hasMembers);
    m_includeAllButton->setEnabled(hasMembers);
    m_excludeAllButton->setEnabled(hasMembers);
    m_primaryAllButton->setEnabled(hasMembers);
    m_primaryNoneButton->setEnabled(hasMembers);
    m_removeButton->setEnabled(!m_poolTable->selectionModel()->selectedRows().isEmpty());

    const QString name = m_savedComparison.id >= 0 ? m_savedComparison.name : "New comparison";
    m_comparisonStatusLabel->setText(!open ? "Open a library to save and load named comparisons."
        : name + (modified ? "  |  Unsaved changes"
            : m_savedComparison.id >= 0 ? "  |  Saved" : "  |  Add models below, then Save to name this set."));
}

void ModelBenchmarkSavedResultsWidget::LoadComparisonSet()
{
    // Capture the target before a possible Save refreshes the preset selector.
    const qint64 id = m_comparisonCombo->currentData().toLongLong();
    if (id < 0 || !ConfirmComparisonChange())
        return;

    QString error;
    const std::vector<ModelBenchmarkComparisonSet> comparisons = m_library.ListComparisonSets(&error);
    if (!error.isEmpty()) {
        QMessageBox::critical(this, "Load Comparison", error);
        return;
    }

    const auto found = std::find_if(comparisons.begin(), comparisons.end(), [id](const auto& comparison) { return comparison.id == id; });
    if (found == comparisons.end()) {
        QMessageBox::warning(this, "Load Comparison", "This comparison no longer exists in the library.");
        RefreshComparisonSets();
        return;
    }

    m_savedComparison = *found;
    m_pooledModelIds.clear();
    m_includedByModelId.clear();
    m_primaryByModelId.clear();
    for (const ModelBenchmarkComparisonMember& member : found->members) {
        m_pooledModelIds.insert(member.modelId);
        m_includedByModelId[member.modelId] = member.included;
        m_primaryByModelId[member.modelId] = member.primary;
    }

    RefreshModels();
    RefreshComparisonSets();

    if (m_pooledModelIds.size() != found->members.size()) {
        QMessageBox::information(this, "Load Comparison",
            "Some models no longer have saved benchmark data and were left out. Save to update this comparison.");
    }
}

void ModelBenchmarkSavedResultsWidget::NewComparisonSet()
{
    if (!ConfirmComparisonChange())
        return;

    m_savedComparison = {};
    m_pooledModelIds.clear();
    m_includedByModelId.clear();
    m_primaryByModelId.clear();
    m_comparisonCombo->setCurrentIndex(0);
    RefreshSelectionTables();
}

bool ModelBenchmarkSavedResultsWidget::SaveComparisonSet(bool saveAs)
{
    ModelBenchmarkComparisonSet comparison = CurrentComparison();
    if (saveAs || comparison.id < 0) {
        QInputDialog dialog(this);
        dialog.setPalette(palette());
        dialog.setWindowTitle("Save Comparison As");
        dialog.setLabelText("Comparison name:");
        dialog.setTextValue(saveAs && !comparison.name.isEmpty() ? comparison.name + " (copy)" : "");
        dialog.setOkButtonText("Save");
        dialog.resize(440, dialog.sizeHint().height());
        dialog.findChild<QLineEdit*>()->setMaxLength(120);

        QPushButton* saveButton = dialog.findChild<QDialogButtonBox*>()->button(QDialogButtonBox::Ok);
        saveButton->setEnabled(!dialog.textValue().trimmed().isEmpty());
        connect(&dialog, &QInputDialog::textValueChanged, &dialog, [saveButton](const QString& name) {
            saveButton->setEnabled(!name.trimmed().isEmpty());
        });

        if (dialog.exec() != QDialog::Accepted)
            return false;

        comparison.id = -1;
        comparison.name = dialog.textValue().trimmed();
    }

    QString error;
    if (!m_library.SaveComparisonSet(comparison, &error)) {
        QMessageBox::warning(this, "Save Comparison", error);
        return false;
    }

    m_savedComparison = std::move(comparison);
    RefreshComparisonSets();

    return true;
}

bool ModelBenchmarkSavedResultsWidget::OpenLibrary(const std::filesystem::path& path, bool showErrors)
{
    EndLiveComparison();
    m_savedComparison = {};

    QString error;
    if (!m_library.Open(path, &error)) {
        m_pooledModelIds.clear();
        m_includedByModelId.clear();
        m_primaryByModelId.clear();
        m_libraryPathLabel->setText("No library open");
        m_libraryPathLabel->setToolTip({});
        RefreshModels();
        RefreshComparisonSets();
        emit LibraryPathChanged();
        if (showErrors)
            QMessageBox::critical(this, "Open Benchmark Library", error);
        return false;
    }

    m_pooledModelIds.clear();
    m_includedByModelId.clear();
    m_primaryByModelId.clear();
    m_libraryPathLabel->setText(QString::fromStdString(path.string()));
    m_libraryPathLabel->setToolTip(QString::fromStdString(path.string()));
    RefreshModels();
    RefreshComparisonSets();
    emit LibraryPathChanged();

    return true;
}

std::filesystem::path ModelBenchmarkSavedResultsWidget::LibraryPath() const
{
    return m_library.Path();
}

void ModelBenchmarkSavedResultsWidget::SetKnownModels(const std::vector<ModelBenchmarkLibraryModel>& models)
{
    m_knownModels = models;
    RefreshModels();
}

void ModelBenchmarkSavedResultsWidget::ChooseLibrary(bool createNew)
{
    const QString initial = m_library.Path().empty()
        ? QString()
        : QString::fromStdString(m_library.Path().string());
    const QString path = createNew
        ? QFileDialog::getSaveFileName(this, "Create Benchmark Library", initial, "Benchmark Library (*.sqlite *.db)")
        : QFileDialog::getOpenFileName(this, "Open Benchmark Library", initial, "Benchmark Library (*.sqlite *.db);;All Files (*)");
    if (!path.isEmpty() && ConfirmComparisonChange())
        OpenLibrary(std::filesystem::path(path.toStdString()));
}

bool ModelBenchmarkSavedResultsWidget::SaveRun(
    const QString& runId,
    const ModelBenchmarkResult& result,
    const std::vector<ModelBenchmarkLibraryModel>& models,
    bool notify,
    const QString& settingsJson)
{
    if (!m_library.IsOpen()) {
        QMessageBox::information(this, "Save Benchmark Results", "Open or create a benchmark library first.");
        return false;
    }

    QString error;
    const std::vector<ModelBenchmarkLibraryConflict> conflicts = m_library.FindConflicts(models, &error);
    if (!error.isEmpty()) {
        QMessageBox::critical(this, "Save Benchmark Results", error);
        return false;
    }

    std::set<qint64> replacements;
    for (const ModelBenchmarkLibraryConflict& conflict : conflicts) {
        if (!notify) continue;
        for (const ModelBenchmarkLibraryModel& existing : conflict.existing) {
            QMessageBox box(QMessageBox::Warning, "Changed Model Checkpoint",
                QString("%1 has the same logical version as a saved model but different weights.\n\n"
                        "Saved snapshot:\n%2 — %3 TS\n%4\n\n"
                        "New snapshot:\n%5 TS\n%6")
                    .arg(conflict.incoming.displayName)
                    .arg(existing.displayName)
                    .arg(existing.latestTimesteps)
                    .arg(existing.fingerprint.left(16))
                    .arg(conflict.incoming.latestTimesteps)
                    .arg(conflict.incoming.fingerprint.left(16)),
                QMessageBox::NoButton, this);
            QPushButton* keepBoth = box.addButton("Keep Both", QMessageBox::AcceptRole);
            QPushButton* overrideButton = box.addButton("Override This Snapshot", QMessageBox::DestructiveRole);
            QPushButton* cancel = box.addButton(QMessageBox::Cancel);
            box.setDefaultButton(keepBoth);
            box.exec();
            if (box.clickedButton() == cancel)
                return false;
            if (box.clickedButton() == overrideButton)
                replacements.insert(existing.id);
        }
    }

    if (!m_library.SaveRun(runId, result, models, replacements, &error, settingsJson)) {
        QMessageBox::critical(this, "Save Benchmark Results", error);
        return false;
    }

    EndLiveComparison();
    RefreshModels();
    for (int index = 0; index < (int)models.size(); ++index) {
        const ModelBenchmarkLibraryModel& incoming = models[index];
        const auto saved = std::find_if(m_rows.begin(), m_rows.end(), [&](const ModelBenchmarkLibraryModel& candidate) {
            return candidate.id >= 0
                && candidate.fingerprint == incoming.fingerprint
                && candidate.agentName == incoming.agentName
                && candidate.buildName == incoming.buildName
                && candidate.stageName == incoming.stageName
                && candidate.versionId == incoming.versionId;
        });
        if (saved == m_rows.end())
            continue;
        m_pooledModelIds.insert(saved->id);
        m_includedByModelId[saved->id] = true;
        m_primaryByModelId[saved->id] = incoming.primary;
    }

    RefreshSelectionTables();
    if (notify) QMessageBox::information(this, "Save Benchmark Results", "Benchmark results were saved to the active library.");
    emit LibraryDataChanged();

    return true;
}

void ModelBenchmarkSavedResultsWidget::ImportExcel()
{
    if (!m_library.IsOpen()) {
        QMessageBox::information(this, "Import Excel Benchmark", "Open or create a benchmark library first.");
        return;
    }
    const QString path = QFileDialog::getOpenFileName(
        this,
        "Import Excel Benchmark",
        QString(),
        "Excel 2003 XML Workbook (*.xml);;XML Files (*.xml)");
    if (path.isEmpty())
        return;
    QString error;
    ModelBenchmarkLibraryImport imported = ModelBenchmarkLibrary::ReadExcelExport(
        std::filesystem::path(path.toStdString()),
        &error);
    if (!error.isEmpty()) {
        QMessageBox::critical(this, "Import Excel Benchmark", error);
        return;
    }
    if (!imported.warnings.isEmpty()) {
        const auto response = QMessageBox::warning(
            this,
            "Import Excel Benchmark",
            imported.warnings.join("\n") + "\n\nContinue with the import?",
            QMessageBox::Yes | QMessageBox::Cancel,
            QMessageBox::Yes);
        if (response != QMessageBox::Yes)
            return;
    }
    SaveRun(imported.runId, imported.result, imported.models);
}

void ModelBenchmarkSavedResultsWidget::RefreshModels()
{
    ++m_fingerprintRequestId;

    QString error;
    const std::vector<ModelBenchmarkLibraryModel> saved = m_library.IsOpen()
        ? m_library.ListModels(&error)
        : std::vector<ModelBenchmarkLibraryModel>();
    if (!error.isEmpty()) {
        QMessageBox::critical(this, "Benchmark Library", error);
        return;
    }
    std::set<qint64> savedIds;
    for (const ModelBenchmarkLibraryModel& model : saved)
        savedIds.insert(model.id);

    // A background checkpoint refresh must not remove models whose first run is still live.
    if (m_liveComparison)
        for (const ModelBenchmarkLibraryModel& model : m_liveAdditionalModels) savedIds.insert(model.id);

    std::erase_if(m_pooledModelIds, [&](qint64 id) { return !savedIds.contains(id); });
    std::erase_if(m_includedByModelId, [&](const auto& entry) { return !savedIds.contains(entry.first); });
    std::erase_if(m_primaryByModelId, [&](const auto& entry) { return !savedIds.contains(entry.first); });

    const bool fingerprintsVerified = false;
    ApplyKnownModels(saved, m_knownModels, fingerprintsVerified);
    StartFingerprintVerification(saved);
}

void ModelBenchmarkSavedResultsWidget::ApplyKnownModels(
    const std::vector<ModelBenchmarkLibraryModel>& saved,
    const std::vector<ModelBenchmarkLibraryModel>& knownModels,
    bool fingerprintsVerified)
{
    m_rows = saved;

    // A path match is only provisional for a saved logical model.
    // Until content verification completes, do not expose a historical fingerprint as locally loadable
    // merely because its original path still exists.
    for (const ModelBenchmarkLibraryModel& known : knownModels) {
        if (!known.availableLocally)
            continue;
        for (ModelBenchmarkLibraryModel& row : m_rows) {
            if (row.logicalKey == known.logicalKey)
                row.availableLocally = false;
        }
    }

    for (const ModelBenchmarkLibraryModel& knownValue : knownModels) {
        ModelBenchmarkLibraryModel known = knownValue;
        for (ModelBenchmarkLibraryModel& row : m_rows) {
            if (row.logicalKey == known.logicalKey && row.branchName.isEmpty())
                row.branchName = known.branchName;
        }

        bool hasLogicalSnapshot = false;
        for (const ModelBenchmarkLibraryModel& savedModel : saved) {
            if (savedModel.logicalKey == known.logicalKey) {
                hasLogicalSnapshot = true;
                break;
            }
        }

        if (!hasLogicalSnapshot) {
            known.id = -1;
            m_rows.push_back(std::move(known));
            continue;
        }

        if (!known.availableLocally)
            continue;
        if (!fingerprintsVerified)
            continue;

        const bool exactSnapshotSaved = std::any_of(saved.begin(), saved.end(), [&](const auto& savedModel) {
            return savedModel.logicalKey == known.logicalKey
                && !known.fingerprint.isEmpty()
                && savedModel.fingerprint == known.fingerprint;
        });
        if (exactSnapshotSaved) {
            for (ModelBenchmarkLibraryModel& row : m_rows) {
                if (row.logicalKey == known.logicalKey && row.fingerprint == known.fingerprint) {
                    row.availableLocally = true;
                    if (row.branchName.isEmpty())
                        row.branchName = known.branchName;
                }
            }
        } else {
            known.id = -1;
            m_rows.push_back(std::move(known));
        }
    }

    RefreshModelFilters();
    RefreshSelectionTables();
}

void ModelBenchmarkSavedResultsWidget::StartFingerprintVerification(const std::vector<ModelBenchmarkLibraryModel>& saved)
{
    if (m_fingerprintThread) {
        m_fingerprintRefreshPending = true;
        if (m_fingerprintCancellation)
            m_fingerprintCancellation->store(true);
        return;
    }

    auto task = std::make_shared<FingerprintTask>();
    task->requestId = m_fingerprintRequestId;
    task->known = m_knownModels;
    for (const ModelBenchmarkLibraryModel& model : saved)
        task->savedLogicalKeys.insert(model.logicalKey);
    m_fingerprintCancellation = task->cancelled;

    QThread* thread = QThread::create([task]() {
        for (ModelBenchmarkLibraryModel& known : task->known) {
            if (task->cancelled->load())
                return;
            if (!known.availableLocally || !task->savedLogicalKeys.contains(known.logicalKey))
                continue;

            const bool allowCached = true;
            known.fingerprint = ModelBenchmarkLibrary::FingerprintCheckpoint(
                std::filesystem::path(known.checkpointPath.toStdString()),
                std::filesystem::path(known.trainingSetupPath.toStdString()),
                allowCached);
            known.fingerprintVerified = !known.fingerprint.isEmpty();
        }
    });
    thread->setParent(this);
    m_fingerprintThread = thread;
    connect(thread, &QThread::finished, this, [this, thread, task]() {
        if (m_fingerprintThread == thread) {
            m_fingerprintThread = nullptr;
            m_fingerprintCancellation.reset();
        }
        thread->deleteLater();

        if (m_fingerprintRefreshPending) {
            m_fingerprintRefreshPending = false;
            QTimer::singleShot(0, this, [this]() { RefreshModels(); });
            return;
        }
        if (task->requestId != m_fingerprintRequestId || task->cancelled->load())
            return;

        QString error;
        const std::vector<ModelBenchmarkLibraryModel> saved = m_library.IsOpen()
            ? m_library.ListModels(&error)
            : std::vector<ModelBenchmarkLibraryModel>();
        if (!error.isEmpty() || task->requestId != m_fingerprintRequestId)
            return;

        const bool fingerprintsVerified = true;
        ApplyKnownModels(saved, task->known, fingerprintsVerified);
    });
    thread->start();
}

void ModelBenchmarkSavedResultsWidget::RefreshModelFilters()
{
    std::set<QString> agents;
    std::set<QString> builds;
    std::set<QString> stages;
    std::set<QString> branches;
    for (const ModelBenchmarkLibraryModel& model : m_rows) {
        agents.insert(model.agentName);
        builds.insert(model.buildName);
        stages.insert(model.stageName);
        branches.insert(BranchDisplayName(model.branchName));
    }

    auto populate = [](QComboBox* combo, const QString& allLabel, const std::set<QString>& values) {
        if (!combo)
            return;
        const int previousIndex = combo->currentIndex();
        const QString previousValue = previousIndex > 0 ? combo->currentData().toString() : QString();
        QSignalBlocker blocker(combo);
        combo->clear();
        combo->addItem(allLabel);
        for (const QString& value : values)
            combo->addItem(value.isEmpty() ? "Unspecified" : value, value);
        const int restoredIndex = previousIndex > 0 ? combo->findData(previousValue) : 0;
        combo->setCurrentIndex(restoredIndex >= 0 ? restoredIndex : 0);
    };
    populate(m_agentFilterCombo, AllAgents, agents);
    populate(m_buildFilterCombo, AllBuilds, builds);
    populate(m_stageFilterCombo, AllStages, stages);
    populate(m_branchFilterCombo, AllBranches, branches);
}

void ModelBenchmarkSavedResultsWidget::PopulateModelTable()
{
    const QString search = m_searchEdit ? m_searchEdit->text().trimmed() : QString();
    auto matchesFilter = [](const QComboBox* combo, const QString& value) {
        return !combo || combo->currentIndex() <= 0 || combo->currentData().toString() == value;
    };

    std::vector<int> visible;
    for (int index = 0; index < (int)m_rows.size(); ++index) {
        const ModelBenchmarkLibraryModel& model = m_rows[index];
        if (model.id >= 0 && m_pooledModelIds.contains(model.id))
            continue;
        if (!matchesFilter(m_agentFilterCombo, model.agentName)
            || !matchesFilter(m_buildFilterCombo, model.buildName)
            || !matchesFilter(m_stageFilterCombo, model.stageName)
            || !matchesFilter(m_branchFilterCombo, BranchDisplayName(model.branchName)))
            continue;
        const QString haystack = QStringList{
            model.displayName, model.agentName, model.buildName, model.stageName,
            BranchDisplayName(model.branchName), model.versionId, model.checkpointPath,
            model.sourceLabel, model.fingerprint, QString::number(model.latestTimesteps)
        }.join(' ');
        if (search.isEmpty() || haystack.contains(search, Qt::CaseInsensitive))
            visible.push_back(index);
    }

    QSignalBlocker blocker(m_modelTable);
    m_modelTable->setRowCount((int)visible.size());
    const QColor disabledText = palette().color(QPalette::Disabled, QPalette::Text);
    for (int row = 0; row < (int)visible.size(); ++row) {
        const ModelBenchmarkLibraryModel& model = m_rows[visible[row]];
        const bool saved = model.id >= 0;
        const QString snapshot = model.latestTimesteps > 0
            ? QString::number(model.latestTimesteps) + " TS"
            : (model.fingerprintVerified ? model.fingerprint.left(10) : "Unverified");
        const QStringList values = {
            model.displayName,
            model.agentName,
            model.buildName,
            model.stageName,
            BranchDisplayName(model.branchName),
            model.versionId,
            snapshot,
            saved ? (model.availableLocally ? "Available" : "Library only") : "No saved data"
        };
        for (int column = 0; column < values.size(); ++column) {
            QTableWidgetItem* item = ReadOnly(values[column]);
            if (column == ModelColumnModel)
                item->setData(Qt::UserRole, model.id);
            if (saved) {
                item->setToolTip("Click to add " + model.displayName + " to the comparison set.");
            } else {
                item->setFlags(item->flags() & ~Qt::ItemIsEnabled);
                item->setForeground(disabledText);
                item->setToolTip("<qt>This local model has no saved benchmark data and cannot be added to the comparison set.</qt>");
            }
            m_modelTable->setItem(row, column, item);
        }
    }
    m_modelTable->clearSelection();
}

void ModelBenchmarkSavedResultsWidget::PopulatePoolTable()
{
    std::set<qint64> selectedIds;
    for (const QModelIndex& index : m_poolTable->selectionModel()->selectedRows(PoolColumnModel))
        selectedIds.insert(index.data(Qt::UserRole).toLongLong());

    std::vector<const ModelBenchmarkLibraryModel*> pooledModels;
    pooledModels.reserve(m_pooledModelIds.size());
    for (qint64 modelId : m_pooledModelIds) {
        const auto model = std::find_if(m_rows.begin(), m_rows.end(), [&](const auto& row) {
            return row.id == modelId;
        });
        if (model != m_rows.end())
            pooledModels.push_back(&*model);
    }

    std::map<QString, int> resultColors;
    if (m_hasDisplayedResult) {
        for (int row = 0; row < (int)m_displayedResult.rankings.size(); ++row) {
            resultColors[QString::fromStdString(m_displayedResult.rankings[row].modelName)] =
                ResultRowColorIndex(row);
        }
    }

    int includedCount = 0;
    QSignalBlocker blocker(m_poolTable);
    m_poolTable->setRowCount((int)pooledModels.size());
    const QColor inactiveBackground = BlendColor(
        palette().color(QPalette::Base),
        QColor(110, 115, 120),
        0.10);
    for (int row = 0; row < (int)pooledModels.size(); ++row) {
        const ModelBenchmarkLibraryModel& model = *pooledModels[row];
        const auto includedItr = m_includedByModelId.find(model.id);
        const bool included = includedItr == m_includedByModelId.end() || includedItr->second;
        const auto primaryItr = m_primaryByModelId.find(model.id);
        const bool primary = primaryItr == m_primaryByModelId.end() || primaryItr->second;
        if (included)
            includedCount++;

        QTableWidgetItem* includeItem = CheckItem(included);
        includeItem->setToolTip(IncludeTooltip);
        includeItem->setData(ModelFadedRole, !included);
        m_poolTable->setItem(row, PoolColumnInclude, includeItem);

        QTableWidgetItem* primaryItem = CheckItem(primary, included);
        primaryItem->setToolTip(included
            ? PrimaryTooltip
            : "Include this model before changing its Primary setting.");
        primaryItem->setData(ModelFadedRole, !included);
        m_poolTable->setItem(row, PoolColumnPrimary, primaryItem);

        const auto colorItr = resultColors.find(model.displayName);
        const int colorIndex = colorItr != resultColors.end()
            ? colorItr->second
            : ResultRowColorIndex(row);
        QTableWidgetItem* modelItem = ReadOnlyModelItem(model.displayName, colorIndex);
        modelItem->setData(Qt::UserRole, model.id);
        modelItem->setData(ModelFadedRole, !included);
        modelItem->setToolTip(included
            ? "This model is included in the displayed results. Select it and click Remove to return it to Models."
            : "This model remains in the comparison set but is excluded from all displayed results.");
        m_poolTable->setItem(row, PoolColumnModel, modelItem);

        if (!included) {
            for (int column = 0; column < m_poolTable->columnCount(); ++column)
                m_poolTable->item(row, column)->setBackground(inactiveBackground);
        }
    }

    m_poolTable->clearSelection();
    for (int row = 0; row < m_poolTable->rowCount(); ++row) {
        if (selectedIds.contains(m_poolTable->item(row, PoolColumnModel)->data(Qt::UserRole).toLongLong()))
            m_poolTable->selectionModel()->select(m_poolTable->model()->index(row, 0),
                QItemSelectionModel::Select | QItemSelectionModel::Rows);
    }

    const QString status = QString("%1 of %2 included")
        .arg(includedCount)
        .arg(pooledModels.size());
    m_poolStatusLabel->setText(status);
    UpdateComparisonControls();
}

void ModelBenchmarkSavedResultsWidget::RefreshSelectionTables()
{
    PopulateModelTable();
    RebuildResults();
    PopulatePoolTable();
}

void ModelBenchmarkSavedResultsWidget::QueueSelectionRefresh()
{
    if (m_selectionRefreshQueued)
        return;
    m_selectionRefreshQueued = true;
    QTimer::singleShot(0, this, [this]() {
        m_selectionRefreshQueued = false;
        RefreshSelectionTables();
    });
}

void ModelBenchmarkSavedResultsWidget::QueueResultsRefresh()
{
    if (m_resultsRefreshQueued)
        return;
    m_resultsRefreshQueued = true;
    // itemChanged is emitted from inside the checkbox delegate's setData(). Defer replacing
    // Comparison Set rows until the delegate returns, and leave the large Models table alone.
    QTimer::singleShot(0, this, [this]() {
        m_resultsRefreshQueued = false;
        RebuildResults();
        PopulatePoolTable();
    });
}

void ModelBenchmarkSavedResultsWidget::RemoveSelectedPoolModels()
{
    const QModelIndexList selectedRows = m_poolTable->selectionModel()->selectedRows(PoolColumnModel);
    if (selectedRows.empty())
        return;
    for (const QModelIndex& index : selectedRows) {
        const qint64 modelId = index.data(Qt::UserRole).toLongLong();
        m_pooledModelIds.erase(modelId);
        m_includedByModelId.erase(modelId);
    }
    RefreshSelectionTables();
}

void ModelBenchmarkSavedResultsWidget::RebuildResults()
{
    ++m_aggregateRequestId;
    if (!m_library.IsOpen()) {
        m_aggregateRefreshPending = false;
        if (m_aggregateCancellation)
            m_aggregateCancellation->store(true);
        SetEmptyResults("Open or create a benchmark library.");
        return;
    }
    std::vector<ModelBenchmarkLibrarySelection> selection;
    selection.reserve(m_pooledModelIds.size());
    // NOTE: Excluded comparison-set members must never reach LoadAggregate; omitting them here makes
    // every result tab, statistic, coverage path, Primary filter, and export behave as if absent.
    for (qint64 modelId : m_pooledModelIds) {
        const auto includedItr = m_includedByModelId.find(modelId);
        if (includedItr != m_includedByModelId.end() && !includedItr->second)
            continue;
        const auto primaryItr = m_primaryByModelId.find(modelId);
        selection.push_back({ modelId, primaryItr == m_primaryByModelId.end() || primaryItr->second });
    }
    if (selection.size() < 2) {
        m_aggregateRefreshPending = false;
        if (m_aggregateCancellation)
            m_aggregateCancellation->store(true);
        SetEmptyResults("Include at least two models from the comparison set with saved benchmark data.");
        return;
    }

    if (std::none_of(selection.begin(), selection.end(), [](const auto& member) { return member.primary; })) {
        m_aggregateRefreshPending = false;
        if (m_aggregateCancellation)
            m_aggregateCancellation->store(true);
        SetEmptyResults(m_workspaceMode
            ? "Include models in the comparison to view their saved games."
            : "Make at least one included model Primary to compare its saved games.");
        return;
    }

    m_exportButton->setEnabled(false);
    if (m_aggregateThread) {
        m_aggregateRefreshPending = true;
        if (m_aggregateCancellation)
            m_aggregateCancellation->store(true);
        return;
    }

    m_aggregateRefreshPending = false;
    auto task = std::make_shared<AggregateLoadTask>();
    task->requestId = m_aggregateRequestId;
    task->libraryPath = m_library.Path();
    task->selection = std::move(selection);
    task->filter = m_workspaceFilter;
    task->liveComparison = m_liveComparison;
    task->baseline = m_savedBaseline;
    task->liveResult = m_liveResult;
    task->additionalModels = m_liveAdditionalModels;
    task->liveModelIds = m_liveModelIds;
    task->confidence = 0.95f;
    task->primaryFlags.reserve(task->selection.size());
    for (const ModelBenchmarkLibrarySelection& selected : task->selection)
        task->primaryFlags.push_back(selected.primary);
    m_aggregateCancellation = task->cancelled;

    QThread* thread = QThread::create([task]() {
        try {
            if (!task->baseline) {
                ModelBenchmarkLibrary library;
                if (!library.OpenReadOnly(task->libraryPath, &task->error) || task->cancelled->load()) return;
                ModelBenchmarkLibraryAggregate baseline = library.LoadAggregate(task->selection, task->confidence, &task->error,
                    [cancelled = task->cancelled]() { return cancelled->load(); }, task->filter, task->additionalModels);
                if (!task->error.isEmpty() || task->cancelled->load()) return;
                task->baseline = std::make_shared<const ModelBenchmarkLibraryAggregate>(std::move(baseline));
            }

            if (task->cancelled->load()) return;

            task->aggregate = task->liveResult
                ? CombineModelBenchmarkLiveResults(*task->baseline, task->selection, *task->liveResult, task->liveModelIds, task->filter)
                : *task->baseline;
        } catch (const std::exception& error) {
            task->error = QString::fromUtf8(error.what());
        }
    });
    thread->setParent(this);
    m_aggregateThread = thread;
    connect(thread, &QThread::finished, this, [this, thread, task]() {
        if (m_aggregateThread == thread) {
            m_aggregateThread = nullptr;
            m_aggregateCancellation.reset();
        }
        thread->deleteLater();

        if (m_aggregateRefreshPending) {
            m_aggregateRefreshPending = false;
            QTimer::singleShot(0, this, [this]() {
                RebuildResults();
                PopulatePoolTable();
            });
            return;
        }
        if (task->requestId != m_aggregateRequestId || task->cancelled->load())
            return;

        if (!task->error.isEmpty()) {
            SetEmptyResults(task->error);
        } else {
            if (task->liveComparison) m_savedBaseline = task->baseline;
            m_displayedPrimaryFlags = task->primaryFlags;
            PopulateResults(task->aggregate);
        }

        PopulatePoolTable();
        if (m_liveComparison && m_liveRefreshPending) {
            m_liveRefreshPending = false;
            UpdateLiveComparison(m_liveResult);
        }
    });
    thread->start();
}

void ModelBenchmarkSavedResultsWidget::SetEmptyResults(const QString& message)
{
    m_hasDisplayedResult = false;
    m_staleResultTables.clear();
    m_displayedCoverageGames.clear();
    m_displayedEnvironments.clear();
    m_environmentSummaryLabel->clear();
    m_displayedPrimaryFlags.clear();
    m_exportButton->setEnabled(false);
    {
        QSignalBlocker blocker(m_statsModeCombo);
        m_statsModeCombo->clear();
        m_statsModeCombo->addItem("All Modes", 0);
    }
    for (QTableWidget* table : { m_rankingTable, m_breakdownTable, m_pairwiseTable,
             m_coverageTable, m_statsTable, m_environmentCoverageTable })
        PrepareTable(table, { "No Results" }, 1);

    m_rankingTable->setItem(0, 0, ReadOnly(message));
    PrepareTable(m_detailsTable, { "Details" }, 1);
    m_detailsTable->setItem(0, 0, ReadOnly(message));
}

void ModelBenchmarkSavedResultsWidget::PopulateResults(const ModelBenchmarkLibraryAggregate& aggregate)
{
    if (aggregate.contributingRuns == 0 && aggregate.liveGames == 0) {
        SetEmptyResults("No games match the shown models, modes, and Compatibility selection in Test Setup.");
        return;
    }

    const ModelBenchmarkResult& result = aggregate.result;
    m_displayedResult = result;
    m_displayedCoverageGames = aggregate.coverageGames;
    m_displayedEnvironments = aggregate.environments;
    PopulateEnvironmentCoverage(aggregate);
    m_hasDisplayedResult = aggregate.contributingRuns > 0 || aggregate.liveGames > 0;
    m_exportButton->setEnabled(m_hasDisplayedResult);

    // Coalesce hidden views to the newest snapshot.
    // Rebuilding and autosizing all tables on every live tick can flood the GUI with queued scrollbar layouts.
    m_staleResultTables = { m_rankingTable, m_breakdownTable, m_pairwiseTable, m_coverageTable, m_statsTable };

    const int selectedMode = m_statsModeCombo->currentData().toInt();
    {
        QSignalBlocker blocker(m_statsModeCombo);
        m_statsModeCombo->clear();
        m_statsModeCombo->addItem("All Modes", 0);
        if (result.has1v1)
            m_statsModeCombo->addItem("1v1", 1);
        if (result.has2v2)
            m_statsModeCombo->addItem("2v2", 2);
        if (result.has3v3)
            m_statsModeCombo->addItem("3v3", 3);
        const int modeIndex = m_statsModeCombo->findData(selectedMode);
        m_statsModeCombo->setCurrentIndex(modeIndex >= 0 ? modeIndex : 0);
    }

    PopulateCurrentResultTab();

    // Only legacy records carry an integration build, so newer ones omit that row.
    const bool showIntegration = !result.integrationRevision.empty();
    const int fixedRows = showIntegration ? 10 : 9;
    const int detailRows = fixedRows + aggregate.warnings.size();
    PrepareTable(m_detailsTable, { "Field", "Value" }, detailRows);
    m_detailsTable->setItem(0, 0, ReadOnly("Contributing Runs"));
    m_detailsTable->setItem(0, 1, ReadOnly(QString::number(aggregate.contributingRuns)));
    m_detailsTable->setItem(1, 0, ReadOnly("Saved Date Range"));
    m_detailsTable->setItem(1, 1, ReadOnly(aggregate.firstRunAt.isEmpty()
        ? "N/A"
        : aggregate.firstRunAt + " — " + aggregate.latestRunAt));
    m_detailsTable->setItem(2, 0, ReadOnly("Eligible Games"));
    m_detailsTable->setItem(2, 1, ReadOnly(QString::number(result.scoredGames + result.noResultGames)));
    m_detailsTable->setItem(3, 0, ReadOnly("Statistics"));
    m_detailsTable->setItem(3, 1, ReadOnly(result.statTotalsComplete ? "Lossless" : "Unavailable (incomplete source totals)"));
    m_detailsTable->setItem(4, 0, ReadOnly("Draws"));
    m_detailsTable->setItem(4, 1, ReadOnly(QString::number(result.noResultGames)));
    m_detailsTable->setItem(5, 0, ReadOnly("RocketSim Versions"));
    m_detailsTable->setItem(5, 1, ReadOnly(QString::fromStdString(ModelBenchmarkRocketSimVersionsText(result.rocketSimVersions))));
    m_detailsTable->setItem(6, 0, ReadOnly("Unsaved Current-run Games"));
    m_detailsTable->setItem(6, 1, ReadOnly(QString::number(aggregate.liveGames)));
    m_detailsTable->setItem(7, 0, ReadOnly("Physics Revision"));
    m_detailsTable->setItem(7, 1, ReadOnly(result.engineRevision.empty() ? "Unrecorded" : QString::fromStdString(result.engineRevision)));
    m_detailsTable->setItem(8, 0, ReadOnly("Prediction Configuration"));
    m_detailsTable->setItem(8, 1, ReadOnly(result.predictionConfigurationJson.empty()
        ? "Unrecorded"
        : QString::fromStdString(result.predictionConfigurationJson)));
    if (showIntegration) {
        m_detailsTable->setItem(9, 0, ReadOnly("Integration Source Revision"));
        m_detailsTable->setItem(9, 1, ReadOnly(QString::fromStdString(result.integrationRevision)));
    }
    for (int i = 0; i < aggregate.warnings.size(); ++i) {
        m_detailsTable->setItem(i + fixedRows, 0, ReadOnly("Warning"));
        m_detailsTable->setItem(i + fixedRows, 1, ReadOnly(aggregate.warnings[i]));
    }
    m_detailsTable->resizeColumnsToContents();
}

void ModelBenchmarkSavedResultsWidget::PopulateCurrentResultTab()
{
    if (!m_hasDisplayedResult) return;

    QWidget* page = m_resultTabs->currentWidget();
    for (QTableWidget* table : { m_rankingTable, m_breakdownTable, m_pairwiseTable, m_coverageTable, m_statsTable }) {
        if (page != table && !page->isAncestorOf(table)) continue;
        if (m_staleResultTables.erase(table) == 0) return;

        // Keep live values fresh without discarding column widths on every update.
        const bool refreshLayout = !m_liveComparison;
        if (table == m_rankingTable)
            ForgeGUI::ModelBenchmarkTables::PopulateRankingTable(table, m_displayedResult, refreshLayout);
        else if (table == m_breakdownTable)
            ForgeGUI::ModelBenchmarkTables::PopulateBreakdownTable(table, m_displayedResult, refreshLayout);
        else if (table == m_pairwiseTable)
            ForgeGUI::ModelBenchmarkTables::PopulatePairwiseTable(table, m_displayedResult, refreshLayout);
        else if (table == m_coverageTable)
            PopulateCoverageTable();
        else
            PopulateStatsTable();
        return;
    }
}

void ModelBenchmarkSavedResultsWidget::PopulateCoverageTable()
{
    const QStringList modelNames = [&]() {
        QStringList names;
        names.reserve((qsizetype)m_displayedResult.modelNames.size());
        for (const std::string& name : m_displayedResult.modelNames)
            names << QString::fromStdString(name);
        return names;
    }();
    const int modelCount = modelNames.size();
    auto isPrimary = [&](int modelIndex) {
        return modelIndex < 0 || modelIndex >= (int)m_displayedPrimaryFlags.size()
            || m_displayedPrimaryFlags[modelIndex];
    };

    std::vector<std::vector<ModelBenchmarkRecord>> records(
        modelCount,
        std::vector<ModelBenchmarkRecord>(modelCount));
    for (const ModelBenchmarkGameMetadata& game : m_displayedCoverageGames) {
        if (!game.mirrorCompleted
            || game.modelA < 0 || game.modelB < 0
            || game.modelA >= modelCount || game.modelB >= modelCount
            || game.modelA == game.modelB)
            continue;
        if (!game.goalScored) {
            records[game.modelA][game.modelB].noResults++;
            records[game.modelB][game.modelA].noResults++;
        } else if (game.includedInRating
            && (game.winner == game.modelA || game.winner == game.modelB)) {
            const int loser = game.winner == game.modelA ? game.modelB : game.modelA;
            records[game.winner][loser].wins++;
            records[loser][game.winner].losses++;
        }
    }

    // Elo can travel through rated outcomes only. Secondary-versus-secondary games remain visible,
    // but Primary filtering deliberately removes those edges from the displayed rating graph.
    std::vector<std::vector<int>> ratingAdjacency(modelCount);
    for (int a = 0; a < modelCount; ++a) {
        for (int b = a + 1; b < modelCount; ++b) {
            const ModelBenchmarkRecord& record = records[a][b];
            if ((!isPrimary(a) && !isPrimary(b)) || record.wins + record.losses == 0)
                continue;
            ratingAdjacency[a].push_back(b);
            ratingAdjacency[b].push_back(a);
        }
    }

    m_coverageTable->clear();
    m_coverageTable->setRowCount(modelCount);
    m_coverageTable->setColumnCount(modelCount);
    QStringList headers;
    headers.reserve(modelCount);
    for (int modelIndex = 0; modelIndex < modelCount; ++modelIndex)
        headers << modelNames[modelIndex] + (!m_workspaceMode && isPrimary(modelIndex) ? " (P)" : "");
    m_coverageTable->setHorizontalHeaderLabels(headers);
    m_coverageTable->setVerticalHeaderLabels(headers);

    const QColor base = palette().color(QPalette::Base);
    const QColor directColor = BlendColor(base, QColor(44, 160, 84), 0.32);
    const QColor indirectColor = BlendColor(base, QColor(120, 130, 140), 0.24);
    const QColor disconnectedColor = BlendColor(base, QColor(190, 70, 70), 0.28);
    const QColor excludedColor = BlendColor(base, QColor(105, 105, 105), 0.18);
    const QColor diagonalColor = BlendColor(base, QColor(120, 130, 140), 0.08);
    const QColor excludedText = palette().color(QPalette::Disabled, QPalette::Text);

    auto appendPaths = [&](QString& tooltip, const std::vector<std::vector<int>>& paths) {
        if (paths.empty())
            return;
        tooltip += paths.size() == 1 ? "\nIndirect Elo path:\n" : "\nShortest indirect Elo paths:\n";
        for (const std::vector<int>& path : paths)
            tooltip += "  " + FormatCoveragePath(modelNames, path) + "\n";
        tooltip.chop(1);
    };

    for (int row = 0; row < modelCount; ++row) {
        for (int column = 0; column < modelCount; ++column) {
            auto* item = ReadOnly(QString());
            item->setTextAlignment(Qt::AlignCenter);
            if (row == column) {
                item->setText(QString::fromUtf8("\xE2\x80\x94"));
                item->setBackground(diagonalColor);
                item->setToolTip(modelNames[row] + " compared with itself.");
                m_coverageTable->setItem(row, column, item);
                continue;
            }

            const ModelBenchmarkRecord& record = records[row][column];
            const int gameCount = record.wins + record.losses + record.noResults;
            const bool excluded = !isPrimary(row) && !isPrimary(column);
            const std::vector<std::vector<int>> paths = FindShortestCoveragePaths(
                ratingAdjacency,
                row,
                column);

            if (excluded) {
                item->setBackground(excludedColor);
                item->setForeground(excludedText);
                item->setText(gameCount > 0
                    ? QString("%1W / %2L / %3NR\n%4 game%5 (excluded)")
                        .arg(record.wins).arg(record.losses).arg(record.noResults)
                        .arg(gameCount).arg(gameCount == 1 ? "" : "s")
                    : "Excluded\nNo saved games");
                QString tooltip;
                if (gameCount > 0) {
                    tooltip = QString("Direct saved results for %1 against %2: %3 wins, %4 losses, "
                        "%5 no-results (%6 games).")
                        .arg(modelNames[row], modelNames[column])
                        .arg(record.wins).arg(record.losses).arg(record.noResults).arg(gameCount);
                } else {
                    tooltip = QString("No direct saved results for %1 against %2.")
                        .arg(modelNames[row], modelNames[column]);
                }
                tooltip += "\nNeither model is Primary, so their direct matchup is excluded from the displayed results and Elo calculation.";
                if (!paths.empty()) {
                    tooltip += " Their displayed Elo values are still linked through other games allowed by the current filter.";
                    appendPaths(tooltip, paths);
                }
                item->setToolTip("<qt>" + tooltip.toHtmlEscaped().replace("\n", "<br>") + "</qt>");
            } else if (gameCount > 0) {
                item->setBackground(directColor);
                item->setText(QString("%1W / %2L / %3NR\n%4 game%5")
                    .arg(record.wins).arg(record.losses).arg(record.noResults)
                    .arg(gameCount).arg(gameCount == 1 ? "" : "s"));
                QString tooltip = QString("Direct saved results for %1 against %2: %3 wins, %4 losses, "
                    "%5 no-results (%6 games).")
                    .arg(modelNames[row], modelNames[column])
                    .arg(record.wins).arg(record.losses).arg(record.noResults).arg(gameCount);
                if (record.wins + record.losses > 0) {
                    tooltip += "\nThis matchup contributes directly to the displayed Elo calculation.";
                } else {
                    tooltip += "\nThese are real direct games, but no-results do not provide an Elo comparison.";
                    if (!paths.empty()) {
                        tooltip += " The models are instead linked through rated games.";
                        appendPaths(tooltip, paths);
                    }
                }
                item->setToolTip("<qt>" + tooltip.toHtmlEscaped().replace("\n", "<br>") + "</qt>");
            } else if (!paths.empty()) {
                const int hops = (int)paths.front().size() - 1;
                item->setBackground(indirectColor);
                item->setText(QString("Indirect\n%1 hop%2")
                    .arg(hops).arg(hops == 1 ? "" : "s"));
                QString tooltip = QString("No direct saved results for %1 against %2. Their Elo comparison is inferred through other rated matchups.")
                    .arg(modelNames[row], modelNames[column]);
                appendPaths(tooltip, paths);
                item->setToolTip("<qt>" + tooltip.toHtmlEscaped().replace("\n", "<br>") + "</qt>");
            } else {
                item->setBackground(disconnectedColor);
                item->setText("No path");
                item->setToolTip(QString("<qt>No direct saved results or indirect rated path connects %1 and %2. "
                    "A shared Elo comparison is unavailable for this selection.</qt>")
                    .arg(modelNames[row].toHtmlEscaped(), modelNames[column].toHtmlEscaped()));
            }
            m_coverageTable->setItem(row, column, item);
        }
    }

    for (int modelIndex = 0; modelIndex < modelCount; ++modelIndex) {
        const QString primaryDescription = m_workspaceMode
            ? "Saved games against any included model contribute to combined results and Elo."
            : (isPrimary(modelIndex) ? "Primary model" : "Non-primary model");
        m_coverageTable->horizontalHeaderItem(modelIndex)->setToolTip(
            modelNames[modelIndex] + " — " + primaryDescription);
        m_coverageTable->verticalHeaderItem(modelIndex)->setToolTip(
            modelNames[modelIndex] + " — " + primaryDescription);
    }
    m_coverageTable->horizontalHeader()->setStretchLastSection(false);
    m_coverageTable->horizontalHeader()->setMinimumSectionSize(105);
    m_coverageTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    m_coverageTable->verticalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    m_coverageTable->verticalHeader()->setMinimumSectionSize(54);
    m_coverageTable->verticalHeader()->setMaximumSectionSize(120);
    m_coverageTable->resizeRowsToContents();
}

void ModelBenchmarkSavedResultsWidget::PopulateStatsTable()
{
    const std::vector<ModelBenchmarkStatRow>* statsRows = &m_displayedResult.optionalStats;
    const int selectedMode = m_statsModeCombo ? m_statsModeCombo->currentData().toInt() : 0;
    if (selectedMode > 0) {
        const auto modeItr = m_displayedResult.optionalStatsByMode.find(selectedMode);
        if (modeItr != m_displayedResult.optionalStatsByMode.end())
            statsRows = &modeItr->second;
        else {
            static const std::vector<ModelBenchmarkStatRow> emptyStats;
            statsRows = &emptyStats;
        }
    }

    const bool refreshLayout = !m_liveComparison;
    ForgeGUI::ModelBenchmarkTables::PopulateStatsTable(m_statsTable, m_displayedResult.rankings, *statsRows, refreshLayout);
}

void ModelBenchmarkSavedResultsWidget::SetWorkspaceMode()
{
    if (m_workspaceMode) return;

    m_workspaceMode = true;
    auto* legacy = new QWidget(this);
    legacy->setLayout(layout());
    legacy->hide();

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(0, 0, 0, 0);
    m_resultTabs->setParent(this);
    m_environmentControls->setParent(this);

    root->addWidget(m_environmentControls);
    m_environmentControls->show();
    root->addWidget(m_resultTabs);
    m_resultTabs->show();
}

void ModelBenchmarkSavedResultsWidget::AddRunConsole(QWidget* console)
{
    const int index = m_resultTabs->addTab(console, "Console");
    m_resultTabs->setTabToolTip(index, "Run messages, progress details, the resolved random seed, and the reason the run stopped.");
}

void ModelBenchmarkSavedResultsWidget::SetWorkspaceSelection(
    const std::vector<ModelBenchmarkLibrarySelection>& selection,
    const ModelBenchmarkLibraryFilter& filter)
{
    EndLiveComparison();
    m_pooledModelIds.clear();
    m_includedByModelId.clear();
    m_primaryByModelId.clear();
    for (const ModelBenchmarkLibrarySelection& model : selection) {
        m_pooledModelIds.insert(model.modelId);
        m_includedByModelId[model.modelId] = true;

        // Workspace membership alone controls Elo; roles only schedule new games.
        m_primaryByModelId[model.modelId] = m_workspaceMode || model.primary;
    }

    m_workspaceFilter = filter;
    QueueResultsRefresh();
}

void ModelBenchmarkSavedResultsWidget::ExportDisplayedResults()
{
    if (!m_hasDisplayedResult) return;

    ModelBenchmarkLibraryAggregate aggregate;
    aggregate.result = m_displayedResult;
    aggregate.environments = m_displayedEnvironments;
    emit ExportRequested(std::move(aggregate));
}

void ModelBenchmarkSavedResultsWidget::BeginLiveComparison(
    const std::vector<ModelBenchmarkLibrarySelection>& selection,
    const std::vector<ModelBenchmarkLibraryModel>& additionalModels,
    const std::vector<qint64>& liveModelIds,
    const ModelBenchmarkLibraryFilter& filter)
{
    SetWorkspaceSelection(selection, filter);
    m_liveComparison = true;
    m_liveAdditionalModels = additionalModels;
    m_liveModelIds = liveModelIds;
}

void ModelBenchmarkSavedResultsWidget::UpdateLiveComparison(std::shared_ptr<const ModelBenchmarkResult> result)
{
    if (!m_liveComparison || !result) return;

    m_liveResult = std::move(result);

    // Keep only the newest cumulative snapshot.
    // Let an in-flight result publish before starting another, so a fast runner cannot starve the display.
    if (m_aggregateThread) {
        m_liveRefreshPending = true;
        return;
    }

    if (m_liveRefreshQueued) return;

    m_liveRefreshQueued = true;
    const quint64 session = m_liveSession;
    QTimer::singleShot(500, this, [this, session]() {
        if (session != m_liveSession) return;
        m_liveRefreshQueued = false;
        QueueResultsRefresh();
    });
}

void ModelBenchmarkSavedResultsWidget::EndLiveComparison()
{
    if (!m_liveComparison) return;

    ++m_liveSession;
    ++m_aggregateRequestId;
    if (m_aggregateCancellation) m_aggregateCancellation->store(true);
    m_liveComparison = false;
    m_liveRefreshPending = m_liveRefreshQueued = false;
    m_liveResult.reset();
    m_savedBaseline.reset();
    m_liveAdditionalModels.clear();
    m_liveModelIds.clear();
}

void ModelBenchmarkSavedResultsWidget::PopulateEnvironmentCoverage(const ModelBenchmarkLibraryAggregate& aggregate)
{
    QStringList headers{"Model", "All environments"};
    QStringList counts;
    std::map<QString, qint64> simulatorGames;
    for (const auto& [key, environment] : aggregate.environments) {
        headers << ModelBenchmarkEnvironmentLabel(environment).replace(" / ", "\n");
        simulatorGames[QString::fromStdString(ModelBenchmarkRocketSimVersionsText(environment.provenance.rocketSimVersions))] += environment.games;
    }

    for (const auto& [simulator, games] : simulatorGames)
        counts << QString("%1: %2 games").arg(simulator).arg(games);

    const bool mixed = ModelBenchmarkHasMixedEnvironments(aggregate);
    m_environmentSummaryLabel->setText((mixed ? "Mixed environments — " : "") + counts.join("; ")
        + (mixed ? ". Pooled Elo reflects this mix; unequal coverage can affect rankings."
            : aggregate.result.evaluationIdentity.empty() ? ". Exact build and settings unrecorded." : ""));

    PrepareTable(m_environmentCoverageTable, headers, (int)aggregate.result.modelNames.size() + 1);
    m_environmentCoverageTable->setItem(0, 0, ReadOnly("Completed games"));

    std::vector<qint64> totals(aggregate.result.modelNames.size() + 1, 0);
    int column = 2;
    for (const auto& [key, environment] : aggregate.environments) {
        m_environmentCoverageTable->setItem(0, column, ReadOnly(QString::number(environment.games)));
        totals[0] += environment.games;

        const ModelBenchmarkResult& source = environment.provenance;
        m_environmentCoverageTable->horizontalHeaderItem(column)->setToolTip(
            source.evaluationIdentity.empty() ? "Exact build and settings unrecorded."
            : QString::fromStdString(source.evaluationIdentity));
        for (int model = 0; model < (int)aggregate.result.modelNames.size(); ++model) {
            const auto found = environment.modelGames.find(model);
            const qint64 count = found == environment.modelGames.end() ? 0 : found->second;
            totals[model + 1] += count;
            m_environmentCoverageTable->setItem(model + 1, column, ReadOnly(QString::number(count)));
        }
        ++column;
    }

    for (int model = 0; model < (int)aggregate.result.modelNames.size(); ++model)
        m_environmentCoverageTable->setItem(model + 1, 0, ReadOnly(QString::fromStdString(aggregate.result.modelNames[model])));
    for (int row = 0; row < (int)totals.size(); ++row)
        m_environmentCoverageTable->setItem(row, 1, ReadOnly(QString::number(totals[row])));

    m_environmentCoverageTable->resizeColumnsToContents();
}
