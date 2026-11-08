#pragma once
#include "Report.h"
#include <pybind11/pybind11.h>

namespace GGL {
	struct GGL_TRAIN_IMEXPORT MetricSender {
		std::string curRunID;
		std::string projectName, groupName, runName;
		pybind11::module pyMod;

		/** Set once the run has been stopped from the wandb website; refreshed by each Send(). */
		bool remoteStopRequested = false;

		MetricSender(std::string projectName = {}, std::string groupName = {}, std::string runName = {}, std::string runID = {});
		
		RG_NO_COPY(MetricSender);

		/** True for report keys that stay in the console and GUI but are never sent to wandb. */
		static bool IsLocalOnlyMetric(const std::string& key) { return key == "SkillTest/NextRunTimestep"; }

		void Send(const Report& report);

		~MetricSender();
	};
}