#pragma once

#include <QObject>
#include <QString>

#include <atomic>

#include <ForgeTraining.h>
#include "Settings/ForgeSettings.h"

/**
 * TrainingWorker
 *
 * Runs the stage queue (RocketForgeTraining::RunTraining) on a background QThread so the UI stays responsive.
 * Training output reaches the GUI through the std::cout capture, not through this worker.
 * RequestStop and RequestSkipStage are thread-safe and are called directly from the UI thread,
 * because the worker thread is blocked inside RunTraining and would not process a queued call.
 *
 * Usage (standard Qt worker-thread pattern):
 *   m_thread = new QThread(this);
 *   m_worker = new TrainingWorker;        // no parent — owned manually
 *   m_worker->moveToThread(m_thread);
 *   connect(m_worker, &TrainingWorker::Finished,      this, &MainWindow::OnTrainingFinished);
 *   connect(m_worker, &TrainingWorker::ErrorOccurred, this, &MainWindow::OnTrainingError);
 *   m_thread->start();
 *
 * Dispatch a job from the UI thread:
 *   m_worker->PrepareRun();
 *   QMetaObject::invokeMethod(m_worker, "RunTraining", Qt::QueuedConnection,
 *       Q_ARG(RocketForgeTraining::ForgeSettings, settings),
 *       Q_ARG(QString, settingsPath));
 */
class TrainingWorker : public QObject
{
    Q_OBJECT

public:
    explicit TrainingWorker(QObject* parent = nullptr);

    /** Clears the previous run's stop request; call on the UI thread before dispatching RunTraining. */
    void PrepareRun();

public slots:
    void RunTraining(RocketForgeTraining::ForgeSettings settings, QString settingsPath);
    void RequestStop();
    void RequestSkipStage();

signals:
    /** The queue ended without error. A stopped run emits StopAcknowledged first. */
    void Finished();

    void StopAcknowledged();

    /** The run failed; Finished is not emitted. simulatorUnavailable marks a RocketSim::EngineUnavailableError reason. */
    void ErrorOccurred(QString message, bool simulatorUnavailable);

    void StageStarted(int index, QString stageName, QString versionId);
    void StageCompleted(int index);
    void StageSkipped(int index, QString reason);

private:
    std::atomic_bool m_stopRequested = false;
};
