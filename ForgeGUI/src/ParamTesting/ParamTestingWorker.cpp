#include "ParamTestingWorker.h"

#include <ForgeTraining.h>

#include <QString>

#include <exception>
#include <string>
#include <utility>

ParamTestingWorker::ParamTestingWorker(QObject* parent) : QObject(parent) {}

void ParamTestingWorker::RunParamTesting(RocketForgeTraining::ParamTestingConfig config)
{
	try {
		RocketForgeTraining::ParamTestingResult result = RocketForgeTraining::RunParamTesting(
			config,
			[this](const RocketForgeTraining::ParamTestingProgress& progress) {
				// Each test segment's queue resets the stop signals while it prepares, dropping a Cancel pressed meanwhile.
				if (progress.stageStarted && m_cancelled)
					RocketForgeTraining::RequestStop();

				emit ProgressUpdated(progress);
			},
			[this] { return m_cancelled.load(); });

		// A late Cancel still suppresses the automatic benchmark handoff.
		if (m_cancelled)
			emit Cancelled();
		else
			emit Completed(std::move(result));
	} catch (const RocketSim::EngineUnavailableError& e) {
		emit Failed(QString::fromUtf8(e.what()), true);
	} catch (const std::exception& e) {
		// Only the cooperative cancellation result counts as a cancel;
		// a disk or model failure stays visible even if the user also pressed Cancel.
		if (m_cancelled && std::string(e.what()) == "Param Testing cancelled.")
			emit Cancelled();
		else
			emit Failed(QString::fromUtf8(e.what()), false);
	} catch (...) {
		emit Failed("Unknown Param Testing error.", false);
	}
}

void ParamTestingWorker::Cancel()
{
	m_cancelled = true;
	RocketForgeTraining::RequestStop();
}
