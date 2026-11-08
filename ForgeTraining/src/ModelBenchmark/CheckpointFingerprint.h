#pragma once

#include <filesystem>
#include <string>

namespace RocketForgeTraining
{
	/**
	 * SHA-256 identity of a checkpoint's weight files (and setup); empty when unreadable.
	 * Byte-identical to the fingerprint the GUI library stores, so existing benchmark libraries and caches stay valid.
	 * Only background display should pass allowCached:
	 * the strict path rehashes and rejects a result when a writer changed the files mid-read.
	 */
	std::string FingerprintCheckpoint(
		const std::filesystem::path& checkpointPath, const std::filesystem::path& trainingSetupPath = {}, bool allowCached = false);
}
