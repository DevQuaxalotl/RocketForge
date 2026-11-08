#pragma once

#include <QElapsedTimer>
#include <QWidget>

#include <functional>

class QLabel;
class QProgressBar;
class QPlainTextEdit;
class QTimer;

/**
 * Shared run-progress presentation: a status label, a progress bar, an internal elapsed
 * timer, and an optional console log. Run/Cancel/Export buttons stay with the owning widget;
 * this panel owns only status + progress + elapsed + optional console.
 */
class RunProgressPanel : public QWidget
{
    Q_OBJECT

public:
    enum class ProgressPosition
    {
        BeforeStatus,
        AfterStatus
    };

    explicit RunProgressPanel(
        QWidget* parent = nullptr,
        ProgressPosition progressPosition = ProgressPosition::BeforeStatus);

    /** Include the built-in console log, or leave it out (default: included). */
    void SetConsoleVisible(bool visible);

    /**
     * Optional status formatter fed the elapsed seconds each tick. When set, the panel
     * refreshes the status line ~1x/sec during a run. When unset, status is only what
     * SetStatus / BeginRun / EndRun set explicitly.
     */
    void SetStatusFormatter(std::function<QString(double elapsedSeconds)> formatter);

    void BeginRun(const QString& initialStatus);
    void SetProgress(int value, int max);
    void SetStatus(const QString& text);
    /** Sets an optional second status line. An empty value hides the line. */
    void SetDetails(const QString& text);
    void SetDetailsToolTip(const QString& text);
    void AppendLog(const QString& line);
    void EndRun(const QString& finalStatus, bool markComplete = true);
    void Reset();

    double ElapsedSeconds() const;

private:
    void RefreshElapsed();

    QLabel* m_statusLabel = nullptr;
    QLabel* m_detailsLabel = nullptr;
    QProgressBar* m_progressBar = nullptr;
    QPlainTextEdit* m_console = nullptr;
    QTimer* m_refreshTimer = nullptr;
    QElapsedTimer m_elapsed;
    std::function<QString(double)> m_statusFormatter;
    bool m_consoleEnabled = true;
};
