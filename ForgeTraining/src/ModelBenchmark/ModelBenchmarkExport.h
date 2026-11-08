#pragma once

#include "ModelBenchmark.h"
#include "SpreadsheetXmlWriter.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace RocketForgeTraining
{
	/** Model identity columns of the export's Models sheet that a ModelBenchmarkResult does not carry. */
	struct ModelBenchmarkExportModel {
		std::string agentName;
		std::string buildName;
		std::string stageName;
		std::string branchName;
		std::string sourceLabel;
		std::string lastTrainedAt;
		std::string fingerprint;
		uint64_t latestTimesteps = 0;
		bool primary = true;
	};

	/** Everything one benchmark workbook holds besides what the result itself carries. */
	struct ModelBenchmarkExportInput {
		const ModelBenchmarkResult* result = nullptr;
		std::string runId;

		/**
		 * Index-aligned with result->modelNames.
		 * A model without an entry exports blank identity cells, Primary "Yes",
		 * and a fingerprint computed from its checkpoint path.
		 */
		std::vector<ModelBenchmarkExportModel> models;
		std::vector<std::string> consoleLines;

		/** Summary row describing pooled simulator environments; omitted when empty. */
		std::string environmentComparisonNote;

		/** Writes caller-owned worksheets between the Console and Models sheets. */
		std::function<void(SpreadsheetXmlWriter&)> extraWorksheets;
	};

	/**
	 * Writes the Excel 2003 XML workbook that the GUI exports and ModelBenchmarkLibrary::ReadExcelExport imports.
	 * Writes atomically (temp file + rename). Returns false and sets error on failure.
	 */
	bool WriteModelBenchmarkExport(const std::filesystem::path& path, const ModelBenchmarkExportInput& input, std::string* error);

	/** "model-benchmark-<yyyyMMdd-HHmmss>.xml", in local time. */
	std::string DefaultModelBenchmarkExportName();

	/** The repository's "Model Benchmarks" folder, beside ForgeTraining. */
	std::filesystem::path DefaultModelBenchmarkExportFolder();

	/** The results library the GUI opens when the benchmark settings name none. */
	std::filesystem::path DefaultModelBenchmarkLibraryPath();
}
