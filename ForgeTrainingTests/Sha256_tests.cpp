#include "TestSupport.h"

#include <Utils/Sha256.h>

#include <string>

using namespace RocketForge;

TEST(Sha256, MatchesNistVectors)
{
	struct Case {
		const char* label;
		std::string message;
		const char* expectedHex;
	};
	const Case cases[] = {
		{ "empty", "", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" },
		{ "abc", "abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" },
		{ "two-block padding",
			"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
			"248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1" },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);
		Sha256 hash;
		hash.Update(testCase.message);
		EXPECT_EQ(Sha256::ToHex(hash.Finish()), testCase.expectedHex);
	}
}

TEST(Sha256, MatchesNistMillionCharacterVectorFedInChunks)
{
	const std::string chunk(1000, 'a');
	Sha256 hash;
	for (int i = 0; i < 1000; i++)
		hash.Update(chunk.data(), chunk.size());

	EXPECT_EQ(Sha256::ToHex(hash.Finish()), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(Sha256, SplitUpdatesMatchOneUpdate)
{
	const std::string message = "The quick brown fox jumps over the lazy dog, then keeps running past one block boundary.";

	Sha256 whole;
	whole.Update(message);
	const std::string expected = Sha256::ToHex(whole.Finish());

	for (size_t split = 0; split <= message.size(); split++) {
		SCOPED_TRACE("split at " + std::to_string(split));
		Sha256 parts;
		parts.Update(message.substr(0, split));
		parts.Update(message.substr(split));
		EXPECT_EQ(Sha256::ToHex(parts.Finish()), expected);
	}
}
