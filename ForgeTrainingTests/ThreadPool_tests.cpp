#include <gtest/gtest.h>

#include <RLGymCPP/ThreadPool.h>

#include <cstdlib>
#include <thread>

// The collection thread pool sizes itself from the cgroup CPU quota and NUMA layout, not just hardware_concurrency():
// in a container that reports the host's CPUs (e.g. 256) while the quota may grant only 27 cores,
// and oversubscribing the quota collapses steps per second.

using namespace RLGC;

// ── ParseCgroupV2CpuMaxCores ────────────────────────────────────────────────
// cgroup v2 /sys/fs/cgroup/cpu.max format is "<quota_us> <period_us>", or
// "max <period_us>" when unlimited.

TEST(ThreadPool, ParseCgroupV2CpuMaxCores) {
	struct Case {
		const char* label;
		const char* contents;
		int expected;
	};

	const Case cases[] = {
		{ "fractional quota floors", "2720000 100000", 27 },
		{ "unlimited", "max 100000", 0 },
		{ "unlimited without period", "max", 0 },
		{ "two cores", "200000 100000", 2 },
		{ "one core", "100000 100000", 1 },
		{ "positive quota below one core still gets a thread", "50000 100000", 1 },
		{ "empty", "", 0 },
		{ "whitespace", "   ", 0 },
		{ "garbage", "nonsense", 0 },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);
		ASSERT_EQ(ParseCgroupV2CpuMaxCores(testCase.contents), testCase.expected);
	}
}

// ── ParseCgroupV1QuotaCores ─────────────────────────────────────────────────
// cgroup v1 uses two files: cpu.cfs_quota_us (-1 == unlimited) and cpu.cfs_period_us.

TEST(ThreadPool, ParseCgroupV1QuotaCores) {
	struct Case {
		const char* label;
		long long quotaUs;
		long long periodUs;
		int expected;
	};

	const Case cases[] = {
		{ "fractional quota floors", 2720000, 100000, 27 },
		{ "unlimited", -1, 100000, 0 },
		{ "positive quota below one core still gets a thread", 50000, 100000, 1 },
		{ "zero period", 100000, 0, 0 },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);
		ASSERT_EQ(ParseCgroupV1QuotaCores(testCase.quotaUs, testCase.periodUs), testCase.expected);
	}
}

// ── ResolveThreadCount ──────────────────────────────────────────────────────
// Picks the worker count. Explicit override (>0) always wins; otherwise start at
// hardware concurrency and cap to the cgroup quota when one exists. Never < 1.

TEST(ThreadPool, ResolveThreadCount) {
	struct Case {
		const char* label;
		unsigned int hardwareConcurrency;
		int envOverride;
		int cgroupQuotaCores;
		unsigned int expected;
	};

	const Case cases[] = {
		{ "quota caps visible cores", 256u, 0, 27, 27u },
		{ "override below quota", 256u, 8, 27, 8u },
		{ "override wins even above the quota", 256u, 64, 27, 64u },
		{ "negative override ignored", 256u, -1, 27, 27u },
		{ "no quota uses hardware", 24u, 0, 0, 24u },
		{ "quota above hardware", 8u, 0, 27, 8u },
		{ "undetected hardware (0) still gets a thread", 0u, 0, 0, 1u },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);
		ASSERT_EQ(ResolveThreadCount(testCase.hardwareConcurrency, testCase.envOverride, testCase.cgroupQuotaCores), testCase.expected);
	}
}

// ── ParseCpuList ────────────────────────────────────────────────────────────
// Parses Linux cpu-list syntax ("0-2,4") as found in /sys .../cpulist.

TEST(ThreadPool, ParseCpuList) {
	struct Case {
		const char* label;
		const char* cpuList;
		std::vector<int> expected;
	};

	const Case cases[] = {
		{ "range and single", "0-2,4", { 0, 1, 2, 4 } },
		{ "single", "5", { 5 } },
		{ "unsorted with duplicates", "4,0-1,2,2", { 0, 1, 2, 4 } },
		{ "empty", "", {} },
		{ "whitespace", "   ", {} },
		{ "trailing newline", "0-3\n", { 0, 1, 2, 3 } },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);
		ASSERT_EQ(ParseCpuList(testCase.cpuList), testCase.expected);
	}
}

TEST(ParseCpuList, TwoNumaRanges_Count) {
	ASSERT_EQ(ParseCpuList("0-63,128-191").size(), (size_t)128);
}

// ── SelectBestNodeCpus ──────────────────────────────────────────────────────
// Picks the NUMA node with the most CPUs that survive the allowed (affinity) mask.

TEST(ThreadPool, SelectBestNodeCpus) {
	struct Case {
		const char* label;
		std::vector<std::vector<int>> nodes;
		std::vector<int> allowed;
		std::vector<int> expected;
	};

	const Case cases[] = {
		{ "tie returns the first node", { { 0, 1, 2, 3 }, { 4, 5, 6, 7 } }, { 0, 1, 2, 3, 4, 5, 6, 7 }, { 0, 1, 2, 3 } },
		{ "only the second node allowed", { { 0, 1, 2, 3 }, { 4, 5, 6, 7 } }, { 4, 5, 6, 7 }, { 4, 5, 6, 7 } },
		{ "most allowed CPUs wins (2 vs 4)", { { 0, 1, 2, 3 }, { 4, 5, 6, 7 } }, { 0, 1, 4, 5, 6, 7 }, { 4, 5, 6, 7 } },
		{ "no NUMA nodes returns the allowed set", {}, { 1, 2, 3 }, { 1, 2, 3 } },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);
		ASSERT_EQ(SelectBestNodeCpus(testCase.nodes, testCase.allowed), testCase.expected);
	}
}

// ── ResolvePinnedThreadCount ────────────────────────────────────────────────
// Like ResolveThreadCount, then capped to a single node's CPU count (unless an
// explicit override is given, which always wins).

TEST(ThreadPool, ResolvePinnedThreadCount) {
	struct Case {
		const char* label;
		unsigned int hardwareConcurrency;
		int envOverride;
		int cgroupQuotaCores;
		size_t nodeCpuCount;
		unsigned int expected;
	};

	const Case cases[] = {
		{ "node smaller than quota", 256u, 0, 27, 16, 16u },
		{ "quota smaller than node", 256u, 0, 27, 128, 27u },
		{ "no quota uses the node", 256u, 0, 0, 128, 128u },
		{ "override below both", 256u, 8, 27, 128, 8u },
		{ "override wins even above the node", 256u, 64, 27, 16, 64u },
		{ "unknown node size ignored", 256u, 0, 27, 0, 27u },
		{ "nothing detected still gets a thread", 0u, 0, 0, 0, 1u },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);
		const unsigned int threadCount = ResolvePinnedThreadCount(
			testCase.hardwareConcurrency, testCase.envOverride, testCase.cgroupQuotaCores, testCase.nodeCpuCount);
		ASSERT_EQ(threadCount, testCase.expected);
	}
}

// ── StartBatchedJobs ────────────────────────────────────────────────────────
// Batched dispatch enqueues index-claiming jobs rather than one job per index;
// these verify the claiming loop still runs every index exactly once.

TEST(BatchedJobs, EachIndexRunsOnce) {
	// More indices than threads, fewer, and exactly as many.
	// In the last case the synchronous caller joins the claim loop, so numThreads claimants share numThreads indices.
	const int indexCounts[] = { 1000, 3, g_ThreadPool.GetNumThreads() };

	for (int num : indexCounts) {
		std::vector<std::atomic<int>> callCounts(num);

		g_ThreadPool.StartBatchedJobs([&](int idx) { callCounts[idx]++; }, num, false);

		for (int i = 0; i < num; i++)
			ASSERT_EQ(callCounts[i].load(), 1);
	}
}

TEST(BatchedJobs, Sync_CallerJoinsClaimLoop) {
	// A synchronous single-index batch enqueues no worker jobs at all: the
	// calling thread claims the index itself.
	std::thread::id runThread;

	g_ThreadPool.StartBatchedJobs([&](int) { runThread = std::this_thread::get_id(); }, 1, false);

	ASSERT_EQ(runThread, std::this_thread::get_id());
}

TEST(BatchedJobs, ZeroIndices_ReturnsWithoutRunning) {
	std::atomic<int> calls = 0;

	g_ThreadPool.StartBatchedJobs([&](int) { calls++; }, 0, false);

	ASSERT_EQ(calls.load(), 0);
}

// ── ResolveIdleSpinMicros ───────────────────────────────────────────────────
// Idle workers poll for work before futex-sleeping; a non-negative RLGC_SPIN_US
// override wins (0 disables spinning), otherwise the default budget applies.

TEST(ThreadPool, ResolveIdleSpinMicros) {
	struct Case {
		const char* label;
		int envOverride;
		int expected;
	};

	const Case cases[] = {
		{ "unset uses the default", -1, DEFAULT_IDLE_SPIN_MICROS },
		{ "zero disables spinning", 0, 0 },
		{ "explicit budget", 1500, 1500 },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);
		ASSERT_EQ(ResolveIdleSpinMicros(testCase.envOverride), testCase.expected);
	}
}

TEST(SpinningPool, RunsJobsAndShutsDownCleanly) {
	// Idle workers are mid-spin when the destructor requests stop; this hangs
	// (or leaks a spinning thread) if the spin loop misses the stop request.
#ifdef _WIN32
	_putenv_s("RLGC_SPIN_US", "500");
#else
	setenv("RLGC_SPIN_US", "500", 1);
#endif

	std::atomic<int> calls = 0;
	{
		ThreadPool pool;
		pool.StartBatchedJobs([&](int) { calls++; }, 100, false);
	}

#ifdef _WIN32
	_putenv_s("RLGC_SPIN_US", "");
#else
	unsetenv("RLGC_SPIN_US");
#endif

	ASSERT_EQ(calls.load(), 100);
}

TEST(BatchedJobs, Async_CompletesAfterWait) {
	const int num = 500;
	std::vector<std::atomic<int>> callCounts(num);

	g_ThreadPool.StartBatchedJobs([&](int idx) { callCounts[idx]++; }, num, true);
	g_ThreadPool.WaitUntilDone();

	for (int i = 0; i < num; i++)
		ASSERT_EQ(callCounts[i].load(), 1);
}

TEST(SelectPerformanceCoreCpus, PicksHighestEfficiencyClass) {
	std::vector<ProcessorCoreInfo> hybridCores;
	std::vector<int> expectedPerformanceCpus;
	for (int coreIndex = 0; coreIndex < 8; coreIndex++) {
		hybridCores.push_back({ { coreIndex * 2, coreIndex * 2 + 1 }, 1 });
		expectedPerformanceCpus.push_back(coreIndex * 2);
		expectedPerformanceCpus.push_back(coreIndex * 2 + 1);
	}
	for (int coreIndex = 0; coreIndex < 8; coreIndex++)
		hybridCores.push_back({ { 16 + coreIndex }, 0 });

	// Efficiency cores listed first must not change the sorted result.
	std::vector<ProcessorCoreInfo> efficiencyFirst(hybridCores.rbegin(), hybridCores.rend());

	struct Case {
		const char* label;
		std::vector<ProcessorCoreInfo> cores;
		std::vector<int> expected;
	};

	const Case cases[] = {
		{ "hybrid selects the class-1 CPUs", hybridCores, expectedPerformanceCpus },
		{ "hybrid in reverse order is still sorted", efficiencyFirst, expectedPerformanceCpus },
		{ "uniform class is not hybrid", { { { 0, 1 }, 0 }, { { 2, 3 }, 0 } }, {} },
		{ "no cores", {}, {} },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);
		ASSERT_EQ(SelectPerformanceCoreCpus(testCase.cores), testCase.expected);
	}
}
