#pragma once

#include <QString>
#include <QWidget>

#include <functional>
#include <optional>

class QCloseEvent;
class QHideEvent;
class QLabel;
class QProgressBar;
class QShowEvent;
class QTimer;

/** Display-ready training state shown by TrainingMonitorWindow. */
struct TrainingMonitorSnapshot
{
    QString status;

    /** Most recent training error; empty hides the error line. */
    QString lastError;

    QString stage;
    QString totalTimesteps;
    QString stepsPerSecond;
    QString eta;

    /** Fraction of the stage target reached, or nullopt when the target is unlimited or unknown. */
    std::optional<double> progressFraction;

    /** Progress-bar text used when progressFraction is nullopt (e.g. "Unlimited"). */
    QString progressFallbackText;

    QString segmentTime;
    QString sessionTime;
    QString totalTime;
};

/**
 * Small always-on-top window summarizing the running training session.
 * It is parentless so it stays visible while the main window is minimized.
 * State comes from a snapshot provider polled once per second while shown.
 */
class TrainingMonitorWindow : public QWidget
{
    Q_OBJECT

public:
    explicit TrainingMonitorWindow(std::function<TrainingMonitorSnapshot()> snapshotProvider);

    /** Pulls a fresh snapshot immediately instead of waiting for the next tick. */
    void Refresh();

signals:
    /** Emitted when the user closes the window from its title bar. */
    void Dismissed();

protected:
    void showEvent(QShowEvent* event) override;
    void hideEvent(QHideEvent* event) override;
    void closeEvent(QCloseEvent* event) override;

private:
    void ApplySnapshot(const TrainingMonitorSnapshot& snapshot);
    void RestorePosition();
    void SavePosition() const;

    std::function<TrainingMonitorSnapshot()> m_snapshotProvider;
    QTimer* m_refreshTimer = nullptr;
    bool m_positionRestored = false;

    QLabel* m_statusLabel = nullptr;
    QLabel* m_errorLabel = nullptr;
    QLabel* m_stageLabel = nullptr;
    QLabel* m_timestepsValueLabel = nullptr;
    QLabel* m_stepsPerSecondValueLabel = nullptr;
    QLabel* m_etaValueLabel = nullptr;
    QProgressBar* m_progressBar = nullptr;
    QLabel* m_segmentTimeValueLabel = nullptr;
    QLabel* m_sessionTimeValueLabel = nullptr;
    QLabel* m_totalTimeValueLabel = nullptr;
};
