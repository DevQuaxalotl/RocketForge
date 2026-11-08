#include "ModelBenchmarkWorker.h"

#include <RocketSim/Engine.h>

ModelBenchmarkWorker::ModelBenchmarkWorker(QObject* parent) : QObject(parent) {}

void ModelBenchmarkWorker::RunBenchmark(RocketForgeTraining::ModelBenchmarkConfig config)
{
    {
        std::scoped_lock lock(m_progressMutex);
        m_latestProgress.reset();
        m_lastLiveResult.reset();
        m_liveGames.Reset();
        m_progressNotificationPending = false;
    }

    config.emitLiveGameDeltas = true;

    try {
        RocketForgeTraining::ModelBenchmarkResult result = RocketForgeTraining::RunModelBenchmark(
            config,
            [this](const RocketForgeTraining::ModelBenchmarkProgress& progress) { PublishProgress(progress); },
            [this] { return m_cancelled.load(); });
        emit Completed(std::move(result));
    } catch (const RocketSim::EngineUnavailableError& error) {
        emit Failed(QString::fromUtf8(error.what()), true);
    } catch (const std::exception& error) {
        FinishWithoutResult(m_cancelled.load() ? "Cancelled" : QString::fromUtf8(error.what()), !m_cancelled.load());
    } catch (...) {
        FinishWithoutResult(m_cancelled.load() ? "Cancelled" : "Unknown model benchmark error.", !m_cancelled.load());
    }
}

void ModelBenchmarkWorker::FinishWithoutResult(const QString& reason, bool failed)
{
    std::shared_ptr<const RocketForgeTraining::ModelBenchmarkResult> snapshot;
    {
        std::scoped_lock lock(m_progressMutex);
        snapshot = m_lastLiveResult;
    }

    if (snapshot && !snapshot->games.empty()) {
        RocketForgeTraining::ModelBenchmarkResult result = *snapshot;
        result.stopReason = failed ? "Terminated early: " + reason.toStdString() : "Cancelled";
        emit LogLine("Preserving the latest completed-game snapshot after the benchmark stopped.");
        emit Completed(std::move(result));
    } else if (failed) {
        const bool simulatorUnavailable = false;
        emit Failed(reason, simulatorUnavailable);
    } else {
        emit Cancelled();
    }
}

void ModelBenchmarkWorker::Cancel() { m_cancelled = true; }

std::optional<RocketForgeTraining::ModelBenchmarkProgress> ModelBenchmarkWorker::TakeLatestProgress()
{
    std::scoped_lock lock(m_progressMutex);
    std::optional<RocketForgeTraining::ModelBenchmarkProgress> progress = m_latestProgress;
    m_latestProgress.reset();
    m_progressNotificationPending = false;
    return progress;
}

void ModelBenchmarkWorker::PublishProgress(RocketForgeTraining::ModelBenchmarkProgress progress)
{
    bool notify = false;
    {
        std::scoped_lock lock(m_progressMutex);
        m_liveGames.Apply(progress);

        // UI consumption and status-only messages must not erase recoverable completed games.
        if (progress.liveResult) m_lastLiveResult = progress.liveResult;

        // Status-only updates must not discard a ranking snapshot waiting for the UI.
        if (!progress.liveResult && m_latestProgress)
            progress.liveResult = m_latestProgress->liveResult;

        m_latestProgress = std::move(progress);
        if (!m_progressNotificationPending) {
            m_progressNotificationPending = true;
            notify = true;
        }
    }

    if (notify) emit ProgressUpdated();
}
