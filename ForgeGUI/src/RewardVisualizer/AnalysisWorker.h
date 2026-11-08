#pragma once

#include <RewardAnalysis/RewardAnalysis.h>

#include <QObject>
#include <QString>

#include <atomic>

/**
 * AnalysisWorker
 *
 * Wraps the blocking RocketForge::RunRewardAnalysis() call so it can run on a
 * background QThread without freezing the UI. Follows the standard Qt worker-thread
 * pattern: the owning widget creates the worker without a parent, calls moveToThread,
 * starts the QThread, and dispatches Start() via QMetaObject::invokeMethod with a
 * Qt::QueuedConnection.
 *
 * The runId argument is echoed back in every emitted signal so the widget can ignore
 * stale signals from a previous run (the GUI event loop may still hold queued progress
 * updates from a Start() that has just been cancelled).
 *
 * Cancel() is callable from any thread (including while Start() is running). It just
 * sets an atomic flag that is polled inside RunRewardAnalysis between episodes.
 **/
class AnalysisWorker : public QObject
{
    Q_OBJECT

public:
    explicit AnalysisWorker(QObject* parent = nullptr);

public slots:
    /**
     * Runs the analysis synchronously on the worker thread. Call via
     * QMetaObject::invokeMethod(worker, [...]{ worker->Start(runId, ...); }, Qt::QueuedConnection)
     * from the GUI thread.
     *
     * runId is echoed back in every signal so the caller can filter stale results.
     * timesteps == 0 lets the stage default to its own batch size.
     * versionPath is the version folder used for saving the cached result.
     * checkpointPath is the numbered checkpoint subfolder containing POLICY.lt / SHARED_HEAD.lt.
     **/
    void Start(int runId,
               const QString& agentName,
               const QString& stageName,
               const QString& versionId,
               const QString& checkpointPath,
               const QString& versionPath,
               int timesteps);

    /**
     * Sets the cancel flag. Safe to call from any thread. RunRewardAnalysis polls
     * this between episodes and exits early when set. Does NOT block.
     **/
    void Cancel();

signals:
    /** Emitted on the worker thread after each completed episode. **/
    void progressUpdated(int runId, int current, int total, const QString& teamSizeLabel);

    /** Emitted on the worker thread when the analysis completes successfully. **/
    void analysisComplete(int runId, RocketForge::RewardAnalysisResult result);

    /** Emitted on the worker thread on failure. Not emitted when cancelled. **/
    void analysisFailed(int runId, const QString& error);

private:
    std::atomic<bool> m_cancelFlag { false };
};

Q_DECLARE_METATYPE(RocketForge::RewardAnalysisResult)
