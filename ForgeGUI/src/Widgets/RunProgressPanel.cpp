#include "RunProgressPanel.h"

#include <QElapsedTimer>
#include <QHBoxLayout>
#include <QLabel>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>

RunProgressPanel::RunProgressPanel(QWidget* parent, ProgressPosition progressPosition)
    : QWidget(parent)
{
    m_statusLabel = new QLabel("Ready", this);
    m_detailsLabel = new QLabel(this);
    m_detailsLabel->hide();

    m_progressBar = new QProgressBar(this);
    m_progressBar->setRange(0, 100);
    m_progressBar->setValue(0);

    m_console = new QPlainTextEdit(this);
    m_console->setReadOnly(true);
    m_console->setMaximumBlockCount(500);

    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);

    auto* row = new QHBoxLayout();
    auto* statusLayout = new QVBoxLayout();
    statusLayout->setContentsMargins(0, 0, 0, 0);
    statusLayout->setSpacing(2);
    statusLayout->addWidget(m_statusLabel);
    statusLayout->addWidget(m_detailsLabel);
    if (progressPosition == ProgressPosition::AfterStatus) {
        row->addLayout(statusLayout, 1);
        row->addWidget(m_progressBar);
    } else {
        row->addWidget(m_progressBar, 1);
        row->addLayout(statusLayout);
    }
    layout->addLayout(row);
    layout->addWidget(m_console, 1);

    m_refreshTimer = new QTimer(this);
    m_refreshTimer->setInterval(1000);
    connect(m_refreshTimer, &QTimer::timeout, this, &RunProgressPanel::RefreshElapsed);
}

void RunProgressPanel::SetConsoleVisible(bool visible)
{
    m_consoleEnabled = visible;
    m_console->setVisible(visible);
}

void RunProgressPanel::SetStatusFormatter(std::function<QString(double)> formatter)
{
    m_statusFormatter = std::move(formatter);
}

void RunProgressPanel::BeginRun(const QString& initialStatus)
{
    m_progressBar->setValue(0);
    m_statusLabel->setText(initialStatus);
    m_elapsed.restart();

    // Only the formatter path needs the 1s tick; widgets without one (e.g. ParamTesting)
    // drive their status directly, so don't run a do-nothing timer.
    if (m_statusFormatter)
        m_refreshTimer->start();
}

void RunProgressPanel::SetProgress(int value, int max)
{
    if (max <= 0) {
        m_progressBar->setValue(0);
        return;
    }

    m_progressBar->setValue(std::clamp((value * 100) / max, 0, 100));
}

void RunProgressPanel::SetStatus(const QString& text)
{
    m_statusLabel->setText(text);
}

void RunProgressPanel::SetDetails(const QString& text)
{
    m_detailsLabel->setText(text);
    m_detailsLabel->setVisible(!text.isEmpty());
}

void RunProgressPanel::SetDetailsToolTip(const QString& text)
{
    m_detailsLabel->setToolTip(text);
}

void RunProgressPanel::AppendLog(const QString& line)
{
    // Gate on the enabled flag, not live visibility: an enabled console must still buffer
    // lines while its tab is off-screen (QPlainTextEdit accumulates them for later display).
    if (m_consoleEnabled)
        m_console->appendPlainText(line);
}

void RunProgressPanel::EndRun(const QString& finalStatus, bool markComplete)
{
    m_refreshTimer->stop();
    if (markComplete)
        m_progressBar->setValue(100);
    m_statusLabel->setText(finalStatus);
}

void RunProgressPanel::Reset()
{
    m_refreshTimer->stop();
    m_progressBar->setValue(0);
    m_statusLabel->setText("Ready");
    m_console->clear();
}

double RunProgressPanel::ElapsedSeconds() const
{
    return m_elapsed.isValid() ? m_elapsed.elapsed() / 1000.0 : 0.0;
}

void RunProgressPanel::RefreshElapsed()
{
    if (m_statusFormatter)
        m_statusLabel->setText(m_statusFormatter(ElapsedSeconds()));
}
