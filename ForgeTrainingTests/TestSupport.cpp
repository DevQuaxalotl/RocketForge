#include "TestSupport.h"

#include <RLGymCPP/Framework.h>
#include <RocketSim/Arena.h>

#include <mutex>
#include <system_error>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

	// GoogleTest's documented pattern for propagating fatal failures out of subroutines.
	// The runner recognizes AssertionException and does not report the failure a second time.
	class ThrowOnFatalFailure : public ::testing::EmptyTestEventListener {
	public:
		void OnTestPartResult(const ::testing::TestPartResult& result) override {
			// A failure with no source file is GoogleTest reporting an exception it already caught.
			// Throwing from inside its catch block would escape the test and silently end the whole run.
			if (result.type() == ::testing::TestPartResult::kFatalFailure && result.file_name() != nullptr)
				throw ::testing::AssertionException(result);
		}
	};

	int CurrentProcessId() {
#ifdef _WIN32
		return _getpid();
#else
		return static_cast<int>(getpid());
#endif
	}

	// Keyed by process so concurrent test processes (ctest -j) never share a directory.
	fs::path ProcessTempRoot() {
		return fs::temp_directory_path() / "rocketforge_tests" / std::to_string(CurrentProcessId());
	}

	class RemoveProcessTempRoot : public ::testing::Environment {
	public:
		void TearDown() override {
			std::error_code ignored;
			fs::remove_all(ProcessTempRoot(), ignored);
		}
	};
}

void RocketForgeTests::InstallTestEnvironment() {
	::testing::UnitTest::GetInstance()->listeners().Append(new ThrowOnFatalFailure);
	::testing::AddGlobalTestEnvironment(new RemoveProcessTempRoot);
}

fs::path RocketForgeTests::TestTempDir(const std::string& label) {
	const ::testing::TestInfo* testInfo = ::testing::UnitTest::GetInstance()->current_test_info();

	std::string folderName = testInfo ? std::string(testInfo->test_suite_name()) + "." + testInfo->name() : "OutsideTest";
	if (!label.empty())
		folderName += "." + label;

	const fs::path dir = ProcessTempRoot() / folderName;
	fs::remove_all(dir);
	fs::create_directories(dir);

	return dir;
}

fs::path RocketForgeTests::RepoCollisionMeshRoot() {
	return RocketSim::CollisionMeshesFolder();
}

bool RocketForgeTests::CollisionMeshesAvailable(const fs::path& meshRoot, const std::string& modeFolder) {
	const fs::path modeDir = meshRoot / modeFolder;

	std::error_code error;
	if (!fs::is_directory(modeDir, error))
		return false;

	for (const fs::directory_entry& entry : fs::directory_iterator(modeDir, error))
		if (entry.is_regular_file() && entry.path().extension() == ".cmf")
			return true;

	return false;
}

void RocketForgeTests::EnsureRocketSimInitialized() {
	static std::once_flag once;
	std::call_once(once, []() {
		const bool silent = true;
		RocketSim::Init(RepoCollisionMeshRoot(), silent);
	});
}
