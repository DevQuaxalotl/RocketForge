#pragma once

#include <ParamTesting/ParamTesting.h>

#include <QObject>

#include <atomic>

class ParamTestingWorker : public QObject
{
	Q_OBJECT
public:
	explicit ParamTestingWorker(QObject* parent = nullptr);

public slots:
	void RunParamTesting(RocketForgeTraining::ParamTestingConfig config);
	void Cancel();

signals:
	void ProgressUpdated(RocketForgeTraining::ParamTestingProgress progress);
	void Completed(RocketForgeTraining::ParamTestingResult result);
	void Cancelled();

	/** simulatorUnavailable marks a RocketSim::EngineUnavailableError reason. **/
	void Failed(QString message, bool simulatorUnavailable);

private:
	std::atomic<bool> m_cancelled = false;
};
