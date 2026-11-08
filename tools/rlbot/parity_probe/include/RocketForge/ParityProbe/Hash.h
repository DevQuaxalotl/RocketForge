#pragma once

#include <cstdint>
#include <span>
#include <string>

namespace RocketForge::ParityProbe
{
	/** Hashes bytes with SHA-256 and returns lowercase hexadecimal. **/
	std::string Sha256Hex(std::span<const uint8_t> bytes);

	/** Hashes contiguous IEEE-754 float32 values encoded little-endian. **/
	std::string Sha256Float32LE(std::span<const float> values);
}
