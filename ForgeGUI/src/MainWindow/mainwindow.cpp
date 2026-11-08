#include "MainWindow/mainwindow.h"
#include "ui_mainwindow.h"

#include "ConsoleCapture.h"
#include "DebugBridge.h"
#include "Debug/DebugRenderWorker.h"
#include "Debug/DebugTabWidget.h"
#include "MainWindow/SkillRatingsPanel.h"
#include "ModelBenchmark/ModelBenchmarkWidget.h"
#include "ParamTesting/ParamTestingWidget.h"
#include "RemoteSync/RemoteSyncPanel.h"
#include "RewardVisualizer/AnalysisWorker.h"
#include "RewardVisualizer/ImpactAnalysisWidget.h"
#include "RewardVisualizer/RewardVisualizerWindow.h"
#include "ThemeUtils.h"
#include "TrainingSetup/TrainingQueueWidget.h"
#include "TrainingSetup/TrainingSetupWindow.h"
#include "TrainingSetup/TrainingWorker.h"
#include "VersionManagementDialog.h"
#include "Widgets/AgentComboBox.h"
#include "Widgets/TrainingMonitorWindow.h"

#include <QApplication>
#include <QByteArray>
#include <QCheckBox>
#include <QCloseEvent>
#include <QColor>
#include <QComboBox>
#include <QCoreApplication>
#include <QEvent>
#include <QFile>
#include <QHBoxLayout>
#include <QLabel>
#include <QList>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QSettings>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QSplitter>
#include <QStackedWidget>
#include <QTabBar>
#include <QTextCharFormat>
#include <QTextCursor>
#include <QTextOption>
#include <QThread>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>
#include <iostream>

using namespace RocketForgeTraining;

namespace
{
    constexpr const char* WindowGeometrySettingsKey = "mainWindow/geometry";
    constexpr const char* WindowMaximizedSettingsKey = "mainWindow/maximized";
    constexpr const char* ColorsEnabledSettingsKey = "appearance/colorsEnabled";
    constexpr const char* MonitorAutoShowSettingsKey = "trainingMonitor/autoShowWhenMinimized";
    constexpr int MainSplitterHandleWidth = 8;
    constexpr int MainLeftPaneMinWidth = 960;
    constexpr int MainRightPaneMinWidth = 760;
    constexpr int MainWindowMinWidth = MainLeftPaneMinWidth + MainRightPaneMinWidth + MainSplitterHandleWidth + 16;
    constexpr int MainWindowMinHeight = 840;
    constexpr int SettingsSaveDebounceMilliseconds = 500;
    constexpr int TrainingDurationRefreshMilliseconds = 1000;
    constexpr int DebugThreadStopTimeoutMilliseconds = 3000;
    constexpr bool LoadSettingsOnStartup = true;

    QSettings AppSettings()
    {
        return QSettings("RocketForge", "ForgeGUI");
    }

    void ApplyMainSplitterSizes(QSplitter* splitter)
    {
        if (!splitter)
            return;

        splitter->setSizes(QList<int>{ MainLeftPaneMinWidth, MainRightPaneMinWidth });
    }

    void IgnoreHorizontalSizeHint(QWidget* widget)
    {
        QSizePolicy policy = widget->sizePolicy();
        policy.setHorizontalPolicy(QSizePolicy::Ignored);
        widget->setSizePolicy(policy);
    }
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

MainWindow::MainWindow(QWidget* parent) : MainWindow(parent, LoadSettingsOnStartup) {}

MainWindow::MainWindow(QWidget* parent, bool loadSettings)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    m_colorsEnabled = AppSettings().value(ColorsEnabledSettingsKey, true).toBool();
    if (QCoreApplication* application = QCoreApplication::instance())
        application->setProperty(ForgeGUI::ColorsEnabledApplicationProperty, m_colorsEnabled);

    // Themed before the remaining widgets are built so content-sized columns measure the theme's fonts and padding.
    ApplyTheme();

    setMinimumSize(MainWindowMinWidth, MainWindowMinHeight);
    resize(std::max(1800, MainWindowMinWidth), 920);

    ui->agentRowLayout->setStretch(0, 1);
    ui->agentRowLayout->setStretch(1, 1);
    ui->agentRowLayout->setStretch(2, 0);
    ui->agentRowLayout->setStretch(3, 0);

    SetupModeTabs();

    // Register types used in queued cross-thread signals.
    qRegisterMetaType<RocketForge::RewardAnalysisResult>("RocketForge::RewardAnalysisResult");
    qRegisterMetaType<ForgeSettings>("RocketForgeTraining::ForgeSettings");
    qRegisterMetaType<ForgeDebugSettings>("RocketForgeTraining::ForgeDebugSettings");

    SetupTrainingPane();
    ConfigureDashboardBlocks();
    SetupLogCapture();
    SetupWorkerThreads();
    SetupDebugTab();
    ConnectTrainingControls();
    SetupTrainingMonitor();
    ConfigurePaneSizing();

    m_currentAgent = ui->agentCombo->CurrentAgentName();
    if (loadSettings)
        LoadOrCreateForgeSettings();
    else
        m_autoSaveSuspended = true;

    m_statusStateLabel = new QLabel(this);
    ui->statusbar->addWidget(m_statusStateLabel, 1);

    if (loadSettings)
        RestoreWindowState();

    ApplyMainSplitterSizes(ui->mainSplitter);
    QTimer::singleShot(0, this, [this]() {
        if (ui && ui->mainSplitter)
            ApplyMainSplitterSizes(ui->mainSplitter);
    });

    UpdateStatusBarState();
}

void MainWindow::SetupModeTabs()
{
    auto* mainLayout = qobject_cast<QVBoxLayout*>(ui->centralwidget->layout());

    if (!mainLayout)
        return;

    // The designer's splitter becomes the Training page, so the other tools can present separate workflows.
    mainLayout->removeWidget(ui->mainSplitter);

    m_modeSelector = new QTabBar(this);
    m_modeSelector->addTab("Training");
    m_modeSelector->addTab("Debug");
    m_modeSelector->addTab("Model Benchmarking");
    m_modeSelector->addTab("Param Testing");
    m_modeSelector->addTab("Impact Analysis");
    m_modeSelector->addTab("File Management");
    m_modeSelector->setExpanding(false);

    m_colorToggleButton = new QPushButton(this);
    m_colorToggleButton->setObjectName("colorToggleButton");
    m_colorToggleButton->setCheckable(true);
    m_colorToggleButton->setText("Toggle Colors");
    m_colorToggleButton->setToolTip("Toggle accent colors for all tabs");

    auto* modeHeaderLayout = new QHBoxLayout();
    modeHeaderLayout->setContentsMargins(0, 0, 4, 0);
    modeHeaderLayout->setSpacing(4);
    modeHeaderLayout->addWidget(m_modeSelector);
    modeHeaderLayout->addStretch(1);
    modeHeaderLayout->addWidget(m_colorToggleButton);

    m_modeStack = new QStackedWidget(this);
    auto* trainingPage = new QWidget(m_modeStack);
    auto* trainingLayout = new QVBoxLayout(trainingPage);
    trainingLayout->setContentsMargins(0, 0, 0, 0);
    trainingLayout->addWidget(ui->mainSplitter);

    m_modelBenchmarkWidget = new ModelBenchmarkWidget(m_modeStack);
    m_paramTestingWidget = new ParamTestingWidget(m_modeStack);
    m_impactAnalysisWidget = new ImpactAnalysisWidget(m_modeStack);

    const bool trainingActive = false;
    m_fileManagementWidget = new VersionManagementDialog(
        QString(),
        QString(),
        QString(),
        trainingActive,
        QString(),
        QString(),
        QString(),
        QString(),
        m_modeStack);

    m_modeStack->addWidget(trainingPage);
    m_modeStack->addWidget(m_modelBenchmarkWidget);
    m_modeStack->addWidget(m_paramTestingWidget);
    m_modeStack->addWidget(m_impactAnalysisWidget);
    m_modeStack->addWidget(m_fileManagementWidget);

    mainLayout->addLayout(modeHeaderLayout);
    mainLayout->addWidget(m_modeStack, 1);

    connect(m_colorToggleButton, &QPushButton::toggled, this, [this](bool enabled) {
        const bool savePreference = true;
        SetColorsEnabled(enabled, savePreference);
    });
    connect(m_modeSelector, &QTabBar::currentChanged, this, [this](int index) {
        m_modeStack->setCurrentIndex(index);
        if (m_debugTab && index == m_modeStack->indexOf(m_debugTab))
            m_debugTab->RefreshLists();
        if (m_fileManagementWidget && index == m_modeStack->indexOf(m_fileManagementWidget))
            m_fileManagementWidget->Refresh();
        if (m_impactAnalysisWidget && index == m_modeStack->indexOf(m_impactAnalysisWidget))
            m_impactAnalysisWidget->Refresh();
    });

    connect(m_modelBenchmarkWidget, &ModelBenchmarkWidget::RunningChanged, this, &MainWindow::OnModelBenchmarkRunningChanged);
    connect(m_paramTestingWidget, &ParamTestingWidget::RunningChanged, this, &MainWindow::OnParamTestingRunningChanged);

    connect(m_paramTestingWidget,   &ParamTestingWidget::BenchmarkSetupReady,
            m_modelBenchmarkWidget, &ModelBenchmarkWidget::ReloadPersistedSetup);
    connect(m_paramTestingWidget, &ParamTestingWidget::BenchmarkRunRequested,
            this,                 &MainWindow::OnParamTestingBenchmarkRunRequested);

    connect(m_paramTestingWidget, &ParamTestingWidget::TrainingPreviewRequested,
            this,                 &MainWindow::OnParamTestingTrainingPreviewRequested);
    connect(m_paramTestingWidget, &ParamTestingWidget::TrainingPreviewStageStarted,
            this,                 &MainWindow::OnParamTestingPreviewStageStarted);
    connect(m_paramTestingWidget, &ParamTestingWidget::TrainingPreviewStageCompleted,
            this,                 &MainWindow::OnParamTestingPreviewStageCompleted);
    connect(m_paramTestingWidget, &ParamTestingWidget::TrainingPreviewFinished,
            this,                 &MainWindow::OnParamTestingPreviewFinished);
}

void MainWindow::SetupTrainingPane()
{
    m_queueWidget = new TrainingQueueWidget(this);
    ui->queueScrollArea->setWidget(m_queueWidget);
    ui->queueScrollArea->setWidgetResizable(true);

    // Remote sync sits at the bottom of the left pane, below the action buttons.
    m_remoteSyncPanel = new ForgeGUI::RemoteSyncPanel(QString::fromUtf8(PROJECT_ROOT), this);
    ui->leftPaneLayout->addWidget(m_remoteSyncPanel);

    connect(m_remoteSyncPanel, &ForgeGUI::RemoteSyncPanel::FlushSettingsRequested, this, &MainWindow::FlushPendingSettingsSave);
    connect(m_remoteSyncPanel, &ForgeGUI::RemoteSyncPanel::RunningChanged,         this, &MainWindow::OnRemoteSyncRunningChanged);
    connect(m_remoteSyncPanel, &ForgeGUI::RemoteSyncPanel::Finished,               this, &MainWindow::OnRemoteSyncFinished);
    connect(m_remoteSyncPanel, &ForgeGUI::RemoteSyncPanel::LineReceived,           this, [this](const QString& line) {
        AppendLog("[sync] " + line);
    });
}

void MainWindow::SetupLogCapture()
{
    m_capture = new ConsoleCapture(this);
    std::cout.rdbuf(m_capture);

    m_parser = new TrainingLogParser(this);

    connect(m_capture, &ConsoleCapture::LineReceived,   m_parser, &TrainingLogParser::ProcessLine);
    connect(m_parser,  &TrainingLogParser::EventParsed, this,     &MainWindow::OnLogEvent);
}

void MainWindow::SetupWorkerThreads()
{
    m_thread = new QThread(this);
    m_worker = new TrainingWorker;
    m_worker->moveToThread(m_thread);

    connect(m_worker, &TrainingWorker::Finished,         this, &MainWindow::OnTrainingFinished);
    connect(m_worker, &TrainingWorker::ErrorOccurred,    this, &MainWindow::OnTrainingError);
    connect(m_worker, &TrainingWorker::StageStarted,     this, &MainWindow::OnStageStarted);
    connect(m_worker, &TrainingWorker::StageCompleted,   this, &MainWindow::OnStageCompleted);
    connect(m_worker, &TrainingWorker::StageSkipped,     this, &MainWindow::OnStageSkipped);
    connect(m_worker, &TrainingWorker::StopAcknowledged, this, [this] {
        m_stopRequested = true;
        ui->saveAndStopBtn->setEnabled(false);
    });

    m_thread->start();

    m_bridge    = new DebugBridge(this);
    m_vizWindow = new RewardVisualizerWindow(m_bridge);
    connect(m_bridge, &DebugBridge::SnapshotReady, m_vizWindow, &RewardVisualizerWindow::OnSnapshotReady);

    // Debug renders are inference-only and run on their own thread, separate from queue training.
    // The worker has no parent: it lives on m_debugThread and the destructor deletes it.
    m_debugThread = new QThread(this);
    m_debugWorker = new DebugRenderWorker;
    m_debugWorker->SetDebugBridge(m_bridge);
    m_debugWorker->moveToThread(m_debugThread);

    connect(m_debugWorker, &DebugRenderWorker::Finished, this, &MainWindow::OnDebugRenderFinished);
    connect(m_debugWorker, &DebugRenderWorker::Failed,   this, &MainWindow::OnDebugRenderFailed);

    m_debugThread->start();
}

void MainWindow::SetupDebugTab()
{
    const int debugTabIndex = 1;
    m_debugTab = new DebugTabWidget(m_bridge, m_modeStack);
    m_modeStack->insertWidget(debugTabIndex, m_debugTab);

    connect(m_debugTab, &DebugTabWidget::StartDebugRequested, this, &MainWindow::OnStartDebugRequested);
    connect(m_debugTab, &DebugTabWidget::StopRequested,       this, [this]() {
        if (m_debugWorker)
            m_debugWorker->Stop();
    });

    connect(m_vizWindow, &RewardVisualizerWindow::SelectionChanged,           m_debugTab, &DebugTabWidget::SetRewardSelection);
    connect(m_vizWindow, &RewardVisualizerWindow::RewardVizConfigChanged,     m_debugTab, &DebugTabWidget::SetRewardVizConfig);
    connect(m_vizWindow, &RewardVisualizerWindow::CriticHeadSelectionChanged, m_debugTab, &DebugTabWidget::SetCriticHeadSelection);
    connect(m_vizWindow, &RewardVisualizerWindow::ShowCriticHeadsChanged,     m_debugTab, &DebugTabWidget::SetShowCriticHeads);
}

void MainWindow::ConnectTrainingControls()
{
    connect(m_queueWidget, &TrainingQueueWidget::changed, this, &MainWindow::OnQueueChanged);

    connect(ui->continueTrainingBtn, &QPushButton::clicked, this, &MainWindow::OnStartTrainingClicked);
    connect(ui->saveAndSkipButton,   &QPushButton::clicked, this, &MainWindow::OnSaveAndSkipClicked);
    connect(ui->saveAndStopBtn,      &QPushButton::clicked, this, &MainWindow::OnSaveAndStopClicked);
    connect(ui->staticOpponentsBtn,  &QPushButton::clicked, this, &MainWindow::OnStaticOpponentsClicked);
    connect(ui->skillReferencesBtn,  &QPushButton::clicked, this, &MainWindow::OnSkillReferencesClicked);
    connect(ui->rewardTableBtn,      &QPushButton::clicked, this, &MainWindow::OnRewardTableClicked);
    connect(ui->newBuildBtn,         &QPushButton::clicked, this, &MainWindow::OnNewBuildClicked);
    connect(ui->newBotBtn,           &QPushButton::clicked, this, &MainWindow::OnNewBotClicked);
    connect(ui->renameBuildBtn,      &QPushButton::clicked, this, &MainWindow::OnRenameBuildClicked);
    connect(ui->pruneSourceCheck,    &QCheckBox::clicked,   this, &MainWindow::OnPruneToggleClicked);

    connect(ui->agentCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int) { OnAgentChanged(); });
    connect(ui->buildCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int) { OnBuildChanged(); });

    for (QCheckBox* globalSetting : { ui->metricsCheck, ui->persistSnapshotsCheck, ui->performanceCoresCheck })
        connect(globalSetting, &QCheckBox::checkStateChanged, this, [this](int) { OnQueueChanged(); });

    m_saveDebounceTimer = new QTimer(this);
    m_saveDebounceTimer->setSingleShot(true);
    m_saveDebounceTimer->setInterval(SettingsSaveDebounceMilliseconds);
    connect(m_saveDebounceTimer, &QTimer::timeout, this, &MainWindow::SaveSettingsNow);

    m_trainingDurationRefreshTimer = new QTimer(this);
    m_trainingDurationRefreshTimer->setInterval(TrainingDurationRefreshMilliseconds);
    connect(m_trainingDurationRefreshTimer, &QTimer::timeout, this, &MainWindow::UpdateTrainingDurationDisplays);
}

void MainWindow::ConfigurePaneSizing()
{
    IgnoreHorizontalSizeHint(ui->logEdit);
    ui->logEdit->setSizeAdjustPolicy(QAbstractScrollArea::AdjustIgnored);
    ui->logEdit->setLineWrapMode(QPlainTextEdit::WidgetWidth);
    ui->logEdit->setWordWrapMode(QTextOption::WrapAnywhere);
    ui->logEdit->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    // Long checkpoint paths and run ids wrap instead of widening the window.
    IgnoreHorizontalSizeHint(ui->checkpointValueLabel);
    ui->checkpointValueLabel->setWordWrap(true);
    IgnoreHorizontalSizeHint(ui->runIdValueLabel);
    ui->runIdValueLabel->setWordWrap(true);

    const bool savePreference = false;
    SetColorsEnabled(m_colorsEnabled, savePreference);

    ui->mainSplitter->setChildrenCollapsible(false);
    ui->mainSplitter->setHandleWidth(MainSplitterHandleWidth);
    ui->leftPane->setMinimumWidth(MainLeftPaneMinWidth);
    ui->rightPane->setMinimumWidth(MainRightPaneMinWidth);

    // The initial sizes are set explicitly because stretch factors alone let the queue pane's large row controls
    // push the stats column too far right on startup.
    ui->mainSplitter->setStretchFactor(0, 56);
    ui->mainSplitter->setStretchFactor(1, 44);
    ApplyMainSplitterSizes(ui->mainSplitter);
}

MainWindow::~MainWindow()
{
    std::cout.rdbuf(nullptr);

    if (m_thread && m_thread->isRunning()) {
        m_worker->RequestStop();
        m_thread->quit();
        m_thread->wait();
    }

    // Flip the stop atomic first: an in-flight render blocks the worker thread inside
    // RunInferenceDebugRender, so quit() is only processed once the loop notices and returns.
    if (m_debugWorker)
        m_debugWorker->Stop();

    if (m_debugThread && m_debugThread->isRunning()) {
        m_debugThread->quit();
        m_debugThread->wait(DebugThreadStopTimeoutMilliseconds);
    }

    delete m_worker;
    delete m_debugWorker;
    delete m_vizWindow;
    delete m_trainingSetupWindow;
    delete m_monitorWindow;
    delete ui;
}

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

void MainWindow::ApplyTheme()
{
    QFile themeFile(":/theme.qss");

    if (themeFile.open(QIODevice::ReadOnly)) {
        const QString theme = QString::fromUtf8(themeFile.readAll());
        qApp->setStyleSheet(ForgeGUI::ApplyColorPreference(theme, m_colorsEnabled));
    }

    if (m_paramTestingWidget)
        m_paramTestingWidget->SetColorsEnabled(m_colorsEnabled);
    if (m_queueWidget)
        m_queueWidget->RefreshColorMode();

    if (m_modelBenchmarkWidget) {
        m_modelBenchmarkWidget->update();
        for (QWidget* child : m_modelBenchmarkWidget->findChildren<QWidget*>())
            child->update();
    }
}

void MainWindow::SetColorsEnabled(bool enabled, bool savePreference)
{
    m_colorsEnabled = enabled;
    if (QCoreApplication* application = QCoreApplication::instance())
        application->setProperty(ForgeGUI::ColorsEnabledApplicationProperty, enabled);

    if (m_colorToggleButton) {
        const QSignalBlocker blocker(m_colorToggleButton);
        m_colorToggleButton->setChecked(enabled);
    }

    ApplyTheme();

    if (savePreference)
        AppSettings().setValue(ColorsEnabledSettingsKey, enabled);
}

// ---------------------------------------------------------------------------
// Window events and state
// ---------------------------------------------------------------------------

void MainWindow::closeEvent(QCloseEvent* event)
{
    if (m_trainingActive) {
        event->ignore();

        if (m_closeAfterTraining)
            return;

        const QMessageBox::StandardButton answer = QMessageBox::question(
            this,
            "Training is running",
            "Save and stop training before closing?",
            QMessageBox::Yes | QMessageBox::Cancel,
            QMessageBox::Yes);

        if (answer == QMessageBox::Yes) {
            m_closeAfterTraining = true;
            if (!m_stopRequested)
                OnSaveAndStopClicked();
            AppendLog("Saving training before closing RocketForge...");
        }

        return;
    }

    if (m_remoteSyncPanel && m_remoteSyncPanel->IsRunning()) {
        const QMessageBox::StandardButton answer = QMessageBox::question(
            this,
            "Remote sync running",
            "A remote sync is running. Cancel it and exit?",
            QMessageBox::Yes | QMessageBox::No,
            QMessageBox::No);

        if (answer != QMessageBox::Yes) {
            m_closeAfterTraining = false;
            event->ignore();
            return;
        }

        m_remoteSyncPanel->CancelAndWait();
    }

    SaveWindowState();

    // The floating tool windows are parentless top-level windows, so they would otherwise
    // outlive this window and keep the application's event loop alive until closed by hand.
    if (m_vizWindow)
        m_vizWindow->close();
    if (m_trainingSetupWindow)
        m_trainingSetupWindow->close();
    if (m_monitorWindow)
        m_monitorWindow->hide();

    QMainWindow::closeEvent(event);
}

void MainWindow::changeEvent(QEvent* event)
{
    QMainWindow::changeEvent(event);

    if (event->type() != QEvent::WindowStateChange || !m_monitorWindow)
        return;

    if (isMinimized()) {
        const bool runActive = m_trainingActive || m_paramTestingPreviewActive;
        const bool autoShowEnabled = m_monitorAutoShowCheck && m_monitorAutoShowCheck->isChecked();

        if (runActive && autoShowEnabled && !m_monitorDismissedWhileMinimized && !m_monitorWindow->isVisible()) {
            SetTrainingMonitorVisible(true);
            m_monitorAutoShown = true;
        }

        return;
    }

    m_monitorDismissedWhileMinimized = false;
    if (m_monitorAutoShown)
        SetTrainingMonitorVisible(false);
    m_monitorAutoShown = false;
}

void MainWindow::RestoreWindowState()
{
    const QSettings settings = AppSettings();
    const QByteArray geometry = settings.value(WindowGeometrySettingsKey).toByteArray();
    if (!geometry.isEmpty())
        restoreGeometry(geometry);

    const bool restoreMaximized = settings.value(WindowMaximizedSettingsKey, false).toBool();
    if (restoreMaximized)
        setWindowState(windowState() | Qt::WindowMaximized);
    else if (width() < minimumWidth() || height() < minimumHeight())
        resize(std::max(width(), minimumWidth()), std::max(height(), minimumHeight()));
}

void MainWindow::SaveWindowState() const
{
    QSettings settings = AppSettings();
    settings.setValue(WindowGeometrySettingsKey, saveGeometry());
    settings.setValue(WindowMaximizedSettingsKey, isMaximized());
}

// ---------------------------------------------------------------------------
// Training monitor
// ---------------------------------------------------------------------------

void MainWindow::SetupTrainingMonitor()
{
    m_monitorWindow = new TrainingMonitorWindow([this]() { return BuildTrainingMonitorSnapshot(); });
    m_monitorWindow->setWindowIcon(windowIcon());

    m_monitorToggleButton = new QPushButton("Mini Monitor", this);
    m_monitorToggleButton->setCheckable(true);
    m_monitorToggleButton->setToolTip("Show a small always-on-top window with live training progress.");

    m_monitorAutoShowCheck = new QCheckBox("Show when minimized", this);
    m_monitorAutoShowCheck->setToolTip("Open the monitor automatically when this window is minimized during training.");
    m_monitorAutoShowCheck->setChecked(AppSettings().value(MonitorAutoShowSettingsKey, true).toBool());

    // Sit next to the run-control buttons, ahead of the spacer that pushes Stop to the right edge.
    const int spacerIndex = ui->actionButtonLayout->indexOf(ui->actionSpacer);
    ui->actionButtonLayout->insertWidget(spacerIndex, m_monitorToggleButton);
    ui->actionButtonLayout->insertWidget(spacerIndex + 1, m_monitorAutoShowCheck);

    connect(m_monitorToggleButton, &QPushButton::clicked, this, [this](bool checked) {
        m_monitorAutoShown = false;
        SetTrainingMonitorVisible(checked);
    });
    connect(m_monitorAutoShowCheck, &QCheckBox::toggled, this, [](bool checked) {
        AppSettings().setValue(MonitorAutoShowSettingsKey, checked);
    });
    connect(m_monitorWindow, &TrainingMonitorWindow::Dismissed, this, [this]() {
        if (isMinimized())
            m_monitorDismissedWhileMinimized = true;
        m_monitorAutoShown = false;

        const QSignalBlocker blocker(m_monitorToggleButton);
        m_monitorToggleButton->setChecked(false);
    });
}

void MainWindow::SetTrainingMonitorVisible(bool visible)
{
    if (!m_monitorWindow)
        return;

    m_monitorWindow->setVisible(visible);

    const QSignalBlocker blocker(m_monitorToggleButton);
    m_monitorToggleButton->setChecked(visible);
}

// ---------------------------------------------------------------------------
// Log panel
// ---------------------------------------------------------------------------

void MainWindow::AppendLog(const QString& text, const QString& colorHex)
{
    QTextCursor cursor(ui->logEdit->document());
    cursor.movePosition(QTextCursor::End);

    QTextCharFormat format;
    if (!colorHex.isEmpty())
        format.setForeground(QColor(colorHex));

    cursor.insertText(text + "\n", format);
    ui->logEdit->setTextCursor(cursor);
    ui->logEdit->ensureCursorVisible();
}
