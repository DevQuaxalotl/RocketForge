#include "TestSupport.h"

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

TEST(TestSupport, TestTempDirIsFreshAndEmpty) {
	const fs::path dir = RocketForgeTests::TestTempDir();
	std::ofstream(dir / "leftover.txt") << "x";

	const fs::path again = RocketForgeTests::TestTempDir();

	ASSERT_EQ(again, dir);
	ASSERT_TRUE(fs::is_directory(again));
	ASSERT_TRUE(fs::is_empty(again));
}

TEST(TestSupport, TestTempDirIsNamedForTheRunningTestAndLabel) {
	const fs::path plain = RocketForgeTests::TestTempDir();
	const fs::path labeled = RocketForgeTests::TestTempDir("second");

	ASSERT_EQ(plain.filename().string(), "TestSupport.TestTempDirIsNamedForTheRunningTestAndLabel");
	ASSERT_EQ(labeled.filename().string(), "TestSupport.TestTempDirIsNamedForTheRunningTestAndLabel.second");
	ASSERT_EQ(plain.parent_path(), labeled.parent_path());
}

TEST(TestSupport, CollisionMeshesAvailableRequiresASoccarCmfFile) {
	const fs::path root = RocketForgeTests::TestTempDir();
	ASSERT_FALSE(RocketForgeTests::CollisionMeshesAvailable(root));

	fs::create_directories(root / "soccar");
	std::ofstream(root / "soccar" / "readme.txt") << "not a mesh";
	ASSERT_FALSE(RocketForgeTests::CollisionMeshesAvailable(root));

	std::ofstream(root / "soccar" / "mesh_0.cmf") << "mesh";
	ASSERT_TRUE(RocketForgeTests::CollisionMeshesAvailable(root));
}
