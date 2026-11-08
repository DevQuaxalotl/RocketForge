#include "TrainingWorker.h"

TrainingWorker::TrainingWorker(QObject* parent) : QObject(parent) {}

void TrainingWorker::PrepareRun()
{
    m_stopRequested = false;
}

void TrainingWorker::RunTraining(RocketForgeTraining::ForgeSettings settings, QString settingsPath)
{
    try {
        RocketForgeTraining::StageEvents events;
        events.onStageStarted = [this](int index, const std::string& stageName, const std::string& versionId) {
            // RunTraining clears queue signals as it starts, so a Stop pressed while the queue was preparing is raised again.
            if (m_stopRequested)
                RocketForgeTraining::RequestStop();

            emit StageStarted(index, QString::fromStdString(stageName), QString::fromStdString(versionId));
        };
        events.onStageCompleted = [this](int index) {
            emit StageCompleted(index);
        };
        events.onStageSkipped = [this](int index, const std::string& reason) {
            emit StageSkipped(index, QString::fromStdString(reason));
        };

        const RocketForgeTraining::RunResult result = RocketForgeTraining::RunTraining(settings, settingsPath.toStdString(), events);

        if (result.status == RocketForgeTraining::RunStatus::Error) {
            emit ErrorOccurred(QString::fromStdString(result.errorMessage), result.unavailableEngine.has_value());
            return;
        }

        if (result.status == RocketForgeTraining::RunStatus::Stopped)
            emit StopAcknowledged();

        emit Finished();
    } catch (const RocketSim::EngineUnavailableError& error) {
        emit ErrorOccurred(QString::fromUtf8(error.what()), true);
    } catch (const std::exception& error) {
        emit ErrorOccurred(QString::fromUtf8(error.what()), false);
    } catch (...) {
        emit ErrorOccurred("Unknown exception in training thread.", false);
    }
}

void TrainingWorker::RequestStop()
{
    m_stopRequested = true;
    RocketForgeTraining::RequestStop();
}

void TrainingWorker::RequestSkipStage()
{
    RocketForgeTraining::RequestSkipStage();
}
