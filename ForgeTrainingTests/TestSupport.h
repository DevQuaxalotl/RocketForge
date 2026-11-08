#pragma once

#include <gtest/gtest.h>

#include <RocketSim/Engine.h>

#include <filesystem>
#include <string>

namespace RocketForgeTests {

	/**
	 * Process-wide setup every RocketForge test program runs after InitGoogleTest.
	 * Fatal assertion failures throw, so an ASSERT_* inside a helper aborts the calling test rather than only returning from the helper.
	 * Also removes every TestTempDir when the program finishes.
	 */
	void InstallTestEnvironment();

	/**
	 * Returns an empty directory unique to the running test (and label), recreated on every call.
	 * Everything under the per-process root is removed when the test program finishes.
	 */
	std::filesystem::path TestTempDir(const std::string& label = "");

	std::filesystem::path RepoCollisionMeshRoot();

	/** True when <meshRoot>/<modeFolder> holds at least one .cmf collision mesh. */
	bool CollisionMeshesAvailable(const std::filesystem::path& meshRoot, const std::string& modeFolder = "soccar");

	/** Initializes RocketSim from RepoCollisionMeshRoot() once per process. */
	void EnsureRocketSimInitialized();
}

// Skips are only for missing environment (GPU, assets absent from the checkout) or an opt-in run, never to hide a failure.
// These are macros because GTEST_SKIP() returns from the enclosing function,
// so they must be written directly in the TEST body, not in a helper.

#define ROCKETFORGE_REQUIRE_COLLISION_MESHES()                                                                    \
	do {                                                                                                           \
		const std::filesystem::path _meshRoot = RocketForgeTests::RepoCollisionMeshRoot();                         \
		if (!RocketForgeTests::CollisionMeshesAvailable(_meshRoot))                                                \
			GTEST_SKIP() << "Collision meshes are not in this checkout (" << _meshRoot << ")";                     \
	} while (0)

#define ROCKETFORGE_REQUIRE_ROCKETSIM()                 \
	do {                                                \
		ROCKETFORGE_REQUIRE_COLLISION_MESHES();         \
		RocketForgeTests::EnsureRocketSimInitialized(); \
	} while (0)

#define ROCKETFORGE_REQUIRE_DROPSHOT_MESHES()                                                                     \
	do {                                                                                                           \
		const std::filesystem::path _meshRoot = RocketForgeTests::RepoCollisionMeshRoot();                         \
		if (!RocketForgeTests::CollisionMeshesAvailable(_meshRoot, "dropshot"))                                    \
			GTEST_SKIP() << "Dropshot collision meshes are not in this checkout (" << _meshRoot / "dropshot" << ")"; \
	} while (0)

#define ROCKETFORGE_REQUIRE_ENGINE(engine)                                            \
	do {                                                                              \
		const RocketSim::EngineInfo& _engineInfo = RocketSim::GetEngineInfo(engine); \
		if (!_engineInfo.available)                                                   \
			GTEST_SKIP() << _engineInfo.unavailableReason;                            \
	} while (0)

// Needs <torch/cuda.h> in the including file; TestSupport itself stays torch-free for torch-less bot test targets.
#define ROCKETFORGE_REQUIRE_CUDA()                   \
	do {                                             \
		if (!torch::cuda::is_available())            \
			GTEST_SKIP() << "CUDA is not available"; \
	} while (0)
