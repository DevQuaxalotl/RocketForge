#include "TrainingMonitorWindow.h"

#include <QCloseEvent>
#include <QFontMetrics>
#include <QGridLayout>
#include <QGuiApplication>
#include <QHideEvent>
#include <QLabel>
#include <QProgressBar>
#include <QScreen>
#include <QSettings>
#include <QShowEvent>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>

namespace
{
    constexpr const char* PositionSettingsKey = "trainingMonitor/position";
    constexpr int WindowWidth = 300;
    constexpr int ScreenEdgeMargin = 24;
    constexpr int ProgressBarResolution = 1000;
    constexpr int RefreshIntervalMs = 1000;

    QLabel* MakeValueLabel(QWidget* parent)
    {
        auto* label = new QLabel("N/A", parent);
        label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        return label;
    }

    void AddRow(QGridLayout* layout, const QString& name, QLabel* value)
    {
        const int row = layout->rowCount();
        layout->addWidget(new QLabel(name, value->parentWidget()), row, 0);
        layout->addWidget(value, row, 1);
    }
}

TrainingMonitorWindow::TrainingMonitorWindow(std::function<TrainingMonitorSnapshot()> snapshotProvider)
    : QWidget(nullptr, Qt::Tool | Qt::WindowStaysOnTopHint)
    , m_snapshotProvider(std::move(snapshotProvider))
{
    setWindowTitle("Training Monitor");
    setFixedWidth(WindowWidth);

    m_statusLabel = new QLabel(this);
    QFont statusFont = m_statusLabel->font();
    statusFont.setBold(true);
    m_statusLabel->setFont(statusFont);

    m_errorLabel = new QLabel(this);
    m_errorLabel->setStyleSheet("color: #ff4444;");
    m_errorLabel->hide();

    m_stageLabel = new QLabel(this);

    m_timestepsValueLabel = MakeValueLabel(this);
    m_stepsPerSecondValueLabel = MakeValueLabel(this);
    m_etaValueLabel = MakeValueLabel(this);

    m_progressBar = new QProgressBar(this);
    m_progressBar->setRange(0, ProgressBarResolution);
    m_progressBar->setTextVisible(true);

    m_segmentTimeValueLabel = MakeValueLabel(this);
    m_sessionTimeValueLabel = MakeValueLabel(this);
    m_totalTimeValueLabel = MakeValueLabel(this);

    auto* progressGrid = new QGridLayout();
    progressGrid->setVerticalSpacing(2);
    AddRow(progressGrid, "Timesteps:", m_timestepsValueLabel);
    AddRow(progressGrid, "Steps/sec:", m_stepsPerSecondValueLabel);
    AddRow(progressGrid, "ETA:", m_etaValueLabel);

    auto* timingGrid = new QGridLayout();
    timingGrid->setVerticalSpacing(2);
    AddRow(timingGrid, "Segment time:", m_segmentTimeValueLabel);
    AddRow(timingGrid, "Session time:", m_sessionTimeValueLabel);
    AddRow(timingGrid, "Total time:", m_totalTimeValueLabel);

    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(10, 8, 10, 10);
    layout->setSpacing(4);
    layout->addWidget(m_statusLabel);
    layout->addWidget(m_errorLabel);
    layout->addWidget(m_stageLabel);
    layout->addLayout(progressGrid);
    layout->addWidget(m_progressBar);
    layout->addLayout(timingGrid);

    m_refreshTimer = new QTimer(this);
    m_refreshTimer->setInterval(RefreshIntervalMs);

    connect(m_refreshTimer, &QTimer::timeout, this, &TrainingMonitorWindow::Refresh);
}

void TrainingMonitorWindow::Refresh()
{
    if (m_snapshotProvider)
        ApplySnapshot(m_snapshotProvider());
}

void TrainingMonitorWindow::showEvent(QShowEvent* event)
{
    Refresh();

    if (!m_positionRestored) {
        adjustSize();
        RestorePosition();
        m_positionRestored = true;
    }

    m_refreshTimer->start();
    QWidget::showEvent(event);
}

void TrainingMonitorWindow::hideEvent(QHideEvent* event)
{
    m_refreshTimer->stop();
    SavePosition();
    QWidget::hideEvent(event);
}

void TrainingMonitorWindow::closeEvent(QCloseEvent* event)
{
    emit Dismissed();
    QWidget::closeEvent(event);
}

void TrainingMonitorWindow::ApplySnapshot(const TrainingMonitorSnapshot& snapshot)
{
    m_statusLabel->setText(snapshot.status);

    // Elide rather than wrap so a long error cannot grow the window; the tooltip keeps the full text.
    const int errorWidth = WindowWidth - layout()->contentsMargins().left() - layout()->contentsMargins().right();
    m_errorLabel->setText(QFontMetrics(m_errorLabel->font()).elidedText(snapshot.lastError, Qt::ElideRight, errorWidth));
    m_errorLabel->setToolTip(snapshot.lastError);
    m_errorLabel->setVisible(!snapshot.lastError.isEmpty());

    m_stageLabel->setText(snapshot.stage);
    m_timestepsValueLabel->setText(snapshot.totalTimesteps);
    m_stepsPerSecondValueLabel->setText(snapshot.stepsPerSecond);
    m_etaValueLabel->setText(snapshot.eta);

    if (snapshot.progressFraction.has_value()) {
        const double fraction = std::clamp(snapshot.progressFraction.value(), 0.0, 1.0);
        m_progressBar->setValue((int)std::lround(fraction * ProgressBarResolution));
        m_progressBar->setFormat("%p%");
    } else {
        m_progressBar->setValue(0);
        m_progressBar->setFormat(snapshot.progressFallbackText);
    }

    m_segmentTimeValueLabel->setText(snapshot.segmentTime);
    m_sessionTimeValueLabel->setText(snapshot.sessionTime);
    m_totalTimeValueLabel->setText(snapshot.totalTime);
}

void TrainingMonitorWindow::RestorePosition()
{
    const QSettings settings("RocketForge", "ForgeGUI");
    const QVariant saved = settings.value(PositionSettingsKey);

    // A saved position can point at a monitor that is no longer connected; fall back to the primary screen.
    if (saved.isValid() && QGuiApplication::screenAt(saved.toPoint())) {
        move(saved.toPoint());
        return;
    }

    if (const QScreen* screen = QGuiApplication::primaryScreen()) {
        const QRect available = screen->availableGeometry();
        move(available.right() - frameGeometry().width() - ScreenEdgeMargin, available.top() + ScreenEdgeMargin);
    }
}

void TrainingMonitorWindow::SavePosition() const
{
    QSettings settings("RocketForge", "ForgeGUI");
    settings.setValue(PositionSettingsKey, pos());
}
