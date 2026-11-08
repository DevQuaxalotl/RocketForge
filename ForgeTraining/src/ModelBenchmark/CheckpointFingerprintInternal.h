#pragma once

#include <Utils/Sha256.h>

#include <cstdint>
#include <istream>

namespace RocketForgeTraining
{
	/**
	 * Hashes the rest of the stream and returns whether exactly expectedBytes were read.
	 * istream::read reports an I/O error mid-file the same way as end of file,
	 * so the byte count is the only way to tell a truncated read from a complete one.
	 */
	bool HashStreamContents(std::istream& stream, uintmax_t expectedBytes, RocketForge::Sha256& hash);
}
