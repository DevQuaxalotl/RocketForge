#include "TrainingBuildLock.h"

#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace RocketForgeTraining {
	TrainingBuildLock::TrainingBuildLock(const std::filesystem::path& buildDirectory) {
		std::filesystem::create_directories(buildDirectory);
		const std::filesystem::path path = std::filesystem::canonical(buildDirectory) / ".training-writer.lock";

#ifdef _WIN32
		const DWORD noSharing = 0;
		const HANDLE file = CreateFileW(
			path.c_str(), GENERIC_READ | GENERIC_WRITE, noSharing, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
		if (file == INVALID_HANDLE_VALUE) {
			throw std::runtime_error(
				"This build is already in use by another trainer, or its writer lock is inaccessible: " + path.string());
		}

		_handle = reinterpret_cast<intptr_t>(file);
#else
		const int file = ::open(path.c_str(), O_CREAT | O_RDWR, 0600);
		if (file < 0)
			throw std::runtime_error("Cannot open training writer lock: " + path.string());

		if (::flock(file, LOCK_EX | LOCK_NB) != 0) {
			::close(file);
			throw std::runtime_error("This build is already in use by another trainer: " + path.string());
		}

		_handle = file;
#endif
	}

	TrainingBuildLock::~TrainingBuildLock() {
#ifdef _WIN32
		if (_handle != -1)
			CloseHandle(reinterpret_cast<HANDLE>(_handle));
#else
		if (_handle != -1)
			::close(int(_handle));
#endif
	}
}
