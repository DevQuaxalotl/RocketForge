#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace RocketForge
{
	/** Incremental SHA-256 (FIPS 180-4). Feed bytes with Update, then call Finish once. */
	class Sha256
	{
	public:
		Sha256();

		void Update(const void* data, size_t size);
		void Update(const std::string& text);

		/** Pads, returns the digest, and leaves the hash unusable until reconstructed. */
		std::array<uint8_t, 32> Finish();

		/** Lowercase hex, matching QCryptographicHash::toHex and sha256sum. */
		static std::string ToHex(const std::array<uint8_t, 32>& digest);

	private:
		void ProcessBlock(const uint8_t* block);

		std::array<uint32_t, 8> _state;
		std::array<uint8_t, 64> _buffer{};
		size_t _bufferedBytes = 0;
		uint64_t _totalBytes = 0;
	};
}
