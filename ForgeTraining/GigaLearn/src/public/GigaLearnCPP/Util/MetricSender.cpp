#include "MetricSender.h"

#include "Timer.h"

namespace py = pybind11;
using namespace GGL;

GGL::MetricSender::MetricSender(std::string _projectName, std::string _groupName, std::string _runName, std::string runID) :
	projectName(_projectName), groupName(_groupName), runName(_runName) {

	RG_LOG("Initializing MetricSender...");

	// The embedded interpreter runs GIL-released (see InitializeTraining), so hold the GIL
	// for every Python touch in this ctor (imports, init call, and the pyMod assignment).
	py::gil_scoped_acquire gil;

	try {
		pyMod = py::module::import("python_scripts.metric_receiver");
	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to import metrics receiver, exception: " << e.what());
	}

	try {
		auto returedRunID = pyMod.attr("init")(PY_EXEC_PATH, projectName, groupName, runName, runID);
		curRunID = returedRunID.cast<std::string>();
		RG_LOG(" > " << (runID.empty() ? "Starting" : "Continuing") << " run with ID : \"" << curRunID << "\"...");

	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to initialize in Python, exception: " << e.what());
	}

	RG_LOG(" > MetricSender initalized.");
}

void GGL::MetricSender::Send(const Report& report) {
	// Building and populating the py::dict is itself Python work, so acquire the GIL first;
	// it stays held until after reportDict is destroyed at the end of this function.
	py::gil_scoped_acquire gil;

	py::dict reportDict = {};

	for (auto& pair : report.data) {
		if (IsLocalOnlyMetric(pair.first))
			continue;

		reportDict[pair.first.c_str()] = pair.second;
	}

	try {
		remoteStopRequested = pyMod.attr("add_metrics")(reportDict).cast<bool>();
	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to add metrics, exception: " << e.what());
	}
}

GGL::MetricSender::~MetricSender() {
	py::gil_scoped_acquire gil;
	try {
		pyMod.attr("finish")();
	} catch (...) {}
	// Drop the Python reference while the GIL is held, so the pyMod member's own destructor
	// (running after this body, with the GIL released) has nothing to decref.
	pyMod = py::module();
}