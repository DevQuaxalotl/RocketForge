#pragma once

#include <GigaLearnInfer/Framework.h>
#include <GigaLearnInfer/JsonArchive.h>

#include <cstdint>
#include <string>

namespace GGL {

	/** Build used when settings or saved metadata name none. */
	inline constexpr const char* DEFAULT_BUILD_NAME = "V1";

	/** Parsed semantic segment version identifier used by the segment versioning helpers. **/
	struct SegmentVersionNumber {
		int major = 1;
		int minor = 0;

		void Serialize(JsonArchive& ar) {
			// Explicit 0 defaults match the old hand-written loader (j.value("major", 0) / j.value("minor", 0))
			// for a present-but-incomplete versionOverride object — the struct's own in-class default is (1, 0).
			ar.Field("major", major, 0);
			ar.Field("minor", minor, 0);
		}
	};

	/**
	 * Read-only summary used by the GUI and CLI when presenting saved Training Segments.
	 * Fields are derived from segment metadata and then enriched with checkpoint info.
	 **/
	struct SegmentVersionSummary {
		std::string id;
		std::string type;
		int major = 1;
		int minor = 0;
		std::string notes;
		std::string createdAt;
		std::string updatedAt;
		std::string lastTrainedAt;
		std::string runName;
		std::string rootPath;
		std::string buildName;
		std::string sourceVersionId;
		std::string branchName = "main";
		std::string stageName;
		bool available = true;
		bool pinned = false;
		bool eligibleBenchmark = false;
		bool eligibleRLBot = false;
		uint64_t latestTimesteps = 0;

		/** True when latestTimesteps comes from an autosave under temp_checkpoints/ rather than a durable checkpoint. **/
		bool latestIsAutosave = false;
		size_t modelCount = 0;
		double totalTrainingSeconds = 0.0;
		double skillRating = 0.0;
		bool hasSkillRating = false;
	};

	/** Formats a semantic segment id using the RocketForge `v<major>.<minor>` convention. **/
	GGL_TRAIN_IMEXPORT std::string MakeSegmentVersionId(int major, int minor);
	/** Parses a semantic segment id from UI text, filenames, or stored metadata. **/
	GGL_TRAIN_IMEXPORT bool TryParseSegmentVersionId(const std::string& text, SegmentVersionNumber& out);
}
