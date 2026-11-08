#pragma once

#include <filesystem>
#include <cstdint>

namespace RocketForgeTraining {
	/** Process-owned writer lock shared by GUI, console, and scripted queue launches. **/
	class TrainingBuildLock {
	public:
		explicit TrainingBuildLock(const std::filesystem::path& buildDirectory);
		~TrainingBuildLock();

		TrainingBuildLock(const TrainingBuildLock&) = delete;
		TrainingBuildLock& operator=(const TrainingBuildLock&) = delete;

	private:
		intptr_t _handle = -1;
	};
}
