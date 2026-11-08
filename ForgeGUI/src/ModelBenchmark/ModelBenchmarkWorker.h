#pragma once

#include <QObject>
#include <QString>

#include <ModelBenchmark/ModelBenchmark.h>
#include <ModelBenchmark/ModelBenchmarkLiveGames.h>

#include <atomic>
#include <mutex>
#include <optional>

/**
 * Runs one benchmark on the caller's worker thread with cooperative cancellation.
 * Each run gets a fresh worker; a cancellation requested before startup is retained.
 */
class ModelBenchmarkWorker : public QObject
{
    Q_OBJECT

public:
    explicit ModelBenchmarkWorker(QObject* parent = nullptr);
    std::optional<RocketForgeTraining::ModelBenchmarkProgress> TakeLatestProgress();

public slots:
    void RunBenchmark(RocketForgeTraining::ModelBenchmarkConfig config);
    void Cancel();

signals:
    void ProgressUpdated();
    void LogLine(QString line);
    void Completed(RocketForgeTraining::ModelBenchmarkResult result);
    void Cancelled();

    /** simulatorUnavailable marks a RocketSim::EngineUnavailableError reason. **/
    void Failed(QString message, bool simulatorUnavailable);

private:
    friend class ModelBenchmarkPickerTests;

    void PublishProgress(RocketForgeTraining::ModelBenchmarkProgress progress);

    /** Saves the last reconstructed snapshot through the normal completion path when possible. */
    void FinishWithoutResult(const QString& reason, bool failed);

    std::shared_ptr<const RocketForgeTraining::ModelBenchmarkResult> m_lastLiveResult;
    std::atomic_bool m_cancelled = false;
    std::mutex m_progressMutex;
    std::optional<RocketForgeTraining::ModelBenchmarkProgress> m_latestProgress;
    RocketForgeTraining::ModelBenchmarkLiveGames m_liveGames;
    bool m_progressNotificationPending = false;
};
