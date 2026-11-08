#include "Storage/BuildStorage.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <thread>

int main(int argc, char** argv)
{
	const int exitUsage = 2;
	const int exitGateTimeout = 3;
	const int exitRecordRejected = 4;

	if (argc != 5)
		return exitUsage;

	try {
		namespace fs = std::filesystem;
		const fs::path root(argv[2]);
		const fs::path gate(argv[3]);

		const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
		while (!fs::exists(gate)) {
			if (std::chrono::steady_clock::now() >= deadline)
				return exitGateTimeout;
			std::this_thread::sleep_for(std::chrono::milliseconds(5));
		}

		RocketForgeTraining::BuildStorage storage(root, GGL::DEFAULT_BUILD_NAME);
		for (int i = 0; i < 20; ++i) {
			if (std::string(argv[1]) == "repair") {
				(void)storage.ListBuilds();
			} else {
				const uint64_t time = std::stoull(argv[4]);
				const std::filesystem::path checkpoint = storage.SegmentCheckpointsRoot("v1.00_starter") / std::to_string(time);
				if (!storage.RecordSavedCheckpoint("v1.00", "v1.00_starter", time, checkpoint))
					return exitRecordRejected;
			}
		}

		return 0;
	} catch (const std::exception& error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
