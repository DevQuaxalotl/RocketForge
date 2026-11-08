#include "AnalysisWorker.h"

#include <ForgeTraining.h>
#include <TrainingStage.h>
#include <RewardAnalysis/RewardAnalysis.h>

#include <exception>
#include <memory>
#include <string>

using namespace std;
using namespace RocketForge;

AnalysisWorker::AnalysisWorker(QObject* parent) : QObject(parent) {}

void AnalysisWorker::Start(int runId,
                           const QString& agentName,
                           const QString& stageName,
                           const QString& /*versionId*/,
                           const QString& checkpointPath,
                           const QString& versionPath,
                           int timesteps)
{
    // Reset the cancel flag for this run. Subsequent Cancel() calls during this
    // Start() will be observed inside RunRewardAnalysis via the isCancelled callback.
    m_cancelFlag.store(false);

    const string agentNameStd   = agentName.toStdString();
    const string stageNameStd   = stageName.toStdString();
    const string checkpointStd  = checkpointPath.toStdString();
    const string versionPathStd = versionPath.toStdString();

    try {
        unique_ptr<TrainingStage> stage(
            RocketForgeTraining::CreateTrainingStage(agentNameStd, stageNameStd, {}, {}));
        if (!stage) {
            const QString message = QString("Unknown agent/stage: %1 / %2").arg(agentName).arg(stageName);
            emit analysisFailed(runId, message);
            return;
        }

        RewardAnalysisResult result = RunRewardAnalysis(
            stage.get(),
            checkpointStd,
            timesteps,
            [this, runId](int done, int total, const string& label) {
                emit progressUpdated(runId, done, total, QString::fromStdString(label));
            },
            [this]() { return m_cancelFlag.load(); });

        // Discard the result silently if the user cancelled. The widget already
        // returned to the ready page when it called Cancel().
        if (m_cancelFlag.load())
            return;

        string saveError;
        SaveRewardAnalysisResult(result, versionPathStd, stageNameStd, &saveError);

        emit analysisComplete(runId, std::move(result));

    } catch (const exception& ex) {
        emit analysisFailed(runId, QString::fromStdString(ex.what()));
    } catch (...) {
        emit analysisFailed(runId, "Unknown error during reward analysis.");
    }
}

void AnalysisWorker::Cancel()
{
    m_cancelFlag.store(true);
}
