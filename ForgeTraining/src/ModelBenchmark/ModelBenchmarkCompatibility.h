#pragma once

#include <string>

namespace RocketForgeTraining {
	/**
	 * Legacy benchmark compatibility markers retained in run provenance.
	 * Scheduling uses the user's simulator/environment selections
	 * and assumes all environments of the selected simulator are reusable by default.
	 * These values must not be incremented to invalidate saved coverage.
	 * Version 1 includes existing results whose exact build/settings were unrecorded.
	 **/
	constexpr int CurrentModelBenchmarkCompatibilityVersion(int engineVersion) {
		switch (engineVersion) {
		case 2:
		case 3:
			return 1;
		default:
			return 0;
		}
	}

	/** Missing historical markers mean version 1; malformed markers return zero. **/
	int RecordedModelBenchmarkCompatibilityVersion(const std::string& evaluationIdentity);
}
