#include "Sha256.h"

#include <algorithm>
#include <cstring>

using namespace RocketForge;

namespace
{
	// First 32 bits of the fractional parts of the cube roots of the first 64 primes (FIPS 180-4 §4.2.2).
	constexpr uint32_t ROUND_CONSTANTS[64] = {
		0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
		0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
		0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
		0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
		0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
		0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
		0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
		0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
	};

	constexpr uint32_t RotateRight(uint32_t value, int bits)
	{
		return (value >> bits) | (value << (32 - bits));
	}
}

Sha256::Sha256()
	// First 32 bits of the fractional parts of the square roots of the first 8 primes (FIPS 180-4 §5.3.3).
	: _state{ 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 }
{
}

void Sha256::Update(const void* data, size_t size)
{
	const uint8_t* bytes = static_cast<const uint8_t*>(data);
	_totalBytes += size;

	if (_bufferedBytes > 0) {
		const size_t taken = std::min(size, _buffer.size() - _bufferedBytes);
		std::memcpy(_buffer.data() + _bufferedBytes, bytes, taken);
		_bufferedBytes += taken;
		bytes += taken;
		size -= taken;

		if (_bufferedBytes < _buffer.size())
			return;

		ProcessBlock(_buffer.data());
		_bufferedBytes = 0;
	}

	for (; size >= _buffer.size(); bytes += _buffer.size(), size -= _buffer.size())
		ProcessBlock(bytes);

	std::memcpy(_buffer.data(), bytes, size);
	_bufferedBytes = size;
}

void Sha256::Update(const std::string& text)
{
	Update(text.data(), text.size());
}

std::array<uint8_t, 32> Sha256::Finish()
{
	// Padding is a 1 bit, zeros up to 56 bytes mod 64, then the message length in bits as a big-endian u64.
	const uint64_t totalBits = _totalBytes * 8;
	const uint8_t marker = 0x80;
	Update(&marker, 1);

	const uint8_t zero = 0;
	while (_bufferedBytes != 56)
		Update(&zero, 1);

	uint8_t length[8];
	for (int i = 0; i < 8; i++)
		length[i] = static_cast<uint8_t>(totalBits >> (56 - 8 * i));
	Update(length, sizeof(length));

	std::array<uint8_t, 32> digest{};
	for (size_t word = 0; word < _state.size(); word++)
		for (int i = 0; i < 4; i++)
			digest[word * 4 + i] = static_cast<uint8_t>(_state[word] >> (24 - 8 * i));

	return digest;
}

std::string Sha256::ToHex(const std::array<uint8_t, 32>& digest)
{
	static constexpr char HEX_DIGITS[] = "0123456789abcdef";

	std::string hex;
	hex.reserve(digest.size() * 2);
	for (uint8_t byte : digest) {
		hex.push_back(HEX_DIGITS[byte >> 4]);
		hex.push_back(HEX_DIGITS[byte & 0x0f]);
	}

	return hex;
}

void Sha256::ProcessBlock(const uint8_t* block)
{
	// w = message schedule; a..h = working variables (FIPS 180-4 §6.2.2).
	uint32_t w[64];
	for (int i = 0; i < 16; i++) {
		const uint8_t* word = block + i * 4;
		w[i] = (uint32_t(word[0]) << 24) | (uint32_t(word[1]) << 16) | (uint32_t(word[2]) << 8) | uint32_t(word[3]);
	}

	for (int i = 16; i < 64; i++) {
		const uint32_t s0 = RotateRight(w[i - 15], 7) ^ RotateRight(w[i - 15], 18) ^ (w[i - 15] >> 3);
		const uint32_t s1 = RotateRight(w[i - 2], 17) ^ RotateRight(w[i - 2], 19) ^ (w[i - 2] >> 10);
		w[i] = w[i - 16] + s0 + w[i - 7] + s1;
	}

	uint32_t a = _state[0], b = _state[1], c = _state[2], d = _state[3];
	uint32_t e = _state[4], f = _state[5], g = _state[6], h = _state[7];

	for (int i = 0; i < 64; i++) {
		const uint32_t sum1 = RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
		const uint32_t choose = (e & f) ^ (~e & g);
		const uint32_t temp1 = h + sum1 + choose + ROUND_CONSTANTS[i] + w[i];
		const uint32_t sum0 = RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
		const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
		const uint32_t temp2 = sum0 + majority;

		h = g;
		g = f;
		f = e;
		e = d + temp1;
		d = c;
		c = b;
		b = a;
		a = temp1 + temp2;
	}

	_state[0] += a;
	_state[1] += b;
	_state[2] += c;
	_state[3] += d;
	_state[4] += e;
	_state[5] += f;
	_state[6] += g;
	_state[7] += h;
}
