#pragma once
#include "Framework.h"

#include <thread_pool.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace RLGC {
	// ── Thread sizing + NUMA pinning (see ThreadPool.cpp) ───────────────────────
	// In a container, std::thread::hardware_concurrency() reports the host's CPU
	// count, not the cgroup CPU quota actually granted. Sizing the pool to the host
	// count oversubscribes the quota and collapses collection throughput. On a
	// multi-socket host, spreading workers across NUMA nodes adds cross-node memory
	// latency. These helpers size to the real CPU budget AND pin to one NUMA node.

	// Parse a cgroup v2 cpu.max file ("<quota_us> <period_us>" or "max <period_us>").
	// Returns the number of whole cores the quota allows (floor, min 1 for any
	// positive quota), or 0 when unlimited or unparseable.
	int ParseCgroupV2CpuMaxCores(const std::string& contents);

	// Parse cgroup v1 cpu.cfs_quota_us / cpu.cfs_period_us. quotaUs < 0 means
	// unlimited. Returns whole cores (floor, min 1) or 0 for unlimited/invalid.
	int ParseCgroupV1QuotaCores(long long quotaUs, long long periodUs);

	// Choose the worker-thread count. A positive envOverride always wins; otherwise
	// start from hardwareConcurrency and cap to cgroupQuotaCores when it is > 0.
	// Always returns at least 1.
	unsigned int ResolveThreadCount(unsigned int hardwareConcurrency, int envOverride, int cgroupQuotaCores);

	// Parse Linux cpu-list syntax ("0-2,4") as found in /sys/.../cpulist into a
	// sorted, de-duplicated list of CPU indices. Tolerant of surrounding whitespace.
	std::vector<int> ParseCpuList(const std::string& cpuList);

	// From per-NUMA-node CPU lists, pick the node whose CPUs (after intersecting with
	// the allowed/affinity mask) form the largest set; ties resolve to the first node.
	// Returns allowedCpus unchanged when nodeCpuLists is empty.
	std::vector<int> SelectBestNodeCpus(const std::vector<std::vector<int>>& nodeCpuLists, const std::vector<int>& allowedCpus);

	// Like ResolveThreadCount, then capped to cpuLimit (one NUMA node's CPU count, or the process affinity's on Windows).
	// An explicit (positive) envOverride still wins and is NOT capped. cpuLimit == 0
	// means "no CPU info" → no cap. Always returns at least 1.
	unsigned int ResolvePinnedThreadCount(unsigned int hardwareConcurrency, int envOverride, int cgroupQuotaCores, size_t cpuLimit);

	// The CPUs of the chosen NUMA node (allowed ∩ best node), computed once. Empty on
	// non-Linux or when topology can't be read.
	const std::vector<int>& PreferredNodeCpus();

	// Pin the calling thread to PreferredNodeCpus(). No-op when that set is empty or
	// off Linux. Threads created afterward inherit the affinity.
	void PinCurrentThreadToPreferredNode();

	/** One physical core: its logical CPUs and the OS efficiency class (higher is faster, e.g. P-cores over E-cores). */
	struct ProcessorCoreInfo {
		std::vector<int> logicalCpus;
		int efficiencyClass = 0;
	};

	/** Sorted logical CPUs of the cores with the highest efficiency class. Empty when every core shares one class (not a hybrid CPU). */
	std::vector<int> SelectPerformanceCoreCpus(const std::vector<ProcessorCoreInfo>& cores);

	/**
	 * Windows only: restricts the process to SelectPerformanceCoreCpus() and disables EcoQoS power throttling, logging what it did.
	 * Returns false (a logged no-op) off Windows, on a non-hybrid CPU, or when the machine has more than one processor group.
	 * Call before the first use of g_ThreadPool, which sizes itself from the process affinity.
	 */
	bool RestrictProcessToPerformanceCores();

	// Resolve the effective pool size from the live environment: the RLGC_NUM_THREADS
	// override, std::thread::hardware_concurrency(), the cgroup CPU quota, and the
	// chosen NUMA node's CPU count (on Windows, the process affinity mask's CPU count).
	unsigned int EffectiveThreadCount();

	// How long an idle worker (and a batch-barrier wait) polls for new work before
	// futex-sleeping. A still-spinning worker lets the producer skip the kernel
	// wake — worthwhile on virtualized kernels where each wake costs microseconds —
	// but spinners occupy CPUs and starve the producer/inference threads when the
	// pool already fills every logical CPU. Off by default; opt in per machine via
	// RLGC_SPIN_US after measuring.
	constexpr int DEFAULT_IDLE_SPIN_MICROS = 0;

	// Choose the idle-spin budget. A non-negative envOverride wins (0 disables spinning); otherwise DEFAULT_IDLE_SPIN_MICROS.
	int ResolveIdleSpinMicros(int envOverride);

	// Resolve the idle-spin budget from the live environment (RLGC_SPIN_US override).
	int EffectiveIdleSpinMicros();

	// Modified version of https://stackoverflow.com/questions/26516683/reusing-thread-in-loop-c
	struct ThreadPool {

		dp::thread_pool<>* _tp = nullptr;
		std::once_flag _initOnce;

		ThreadPool() = default;

		RG_NO_COPY(ThreadPool);

		~ThreadPool() {
			delete _tp;
		}

		// The pool is created on first use, not at construction:
		// g_ThreadPool is a global, so eager construction would spawn every worker thread
		// (and print the startup log) for utility invocations that never step an environment.
		dp::thread_pool<>& Pool() {
			std::call_once(_initOnce, [this] {
				unsigned int numThreads = EffectiveThreadCount();
				const std::chrono::microseconds idleSpin{ EffectiveIdleSpinMicros() };

				const std::vector<int>& nodeCpus = PreferredNodeCpus();
				if (!nodeCpus.empty()) {
					std::cerr << "[RLGC] Collection thread pool: " << numThreads
						<< " worker threads pinned to " << nodeCpus.size() << " NUMA-node CPUs"
						<< " (hardware_concurrency=" << std::thread::hardware_concurrency()
						<< ", idle_spin=" << idleSpin.count() << "us)." << std::endl;
				} else {
					std::cerr << "[RLGC] Collection thread pool: " << numThreads << " worker threads"
						<< " (hardware_concurrency=" << std::thread::hardware_concurrency()
						<< ", idle_spin=" << idleSpin.count() << "us)." << std::endl;
				}

				// Pin each worker to the chosen NUMA node so collection stays on one socket.
				_tp = new dp::thread_pool<>(numThreads, [](std::size_t) { PinCurrentThreadToPreferredNode(); }, idleSpin);
			});
			return *_tp;
		}

		template <typename Function, typename... Args> requires std::invocable<Function, Args...>
		void StartJobAsync(Function&& func, Args &&...args) {
			Pool().enqueue_detach(func, args...);
		}

		/**
		 * Runs func(0..num-1) across the pool. Enqueues one index-claiming job per
		 * worker instead of one job per index: each enqueue costs a lock plus a
		 * semaphore wake (a syscall on Linux), which dominates tiny per-index jobs —
		 * severely so under virtualization (WSL2, cloud VMs), where a wake can mean
		 * un-parking a vCPU. Claiming indices from a shared counter caps the wake
		 * count at the pool size while keeping per-index load balancing.
		 *
		 * On a synchronous call the calling thread takes one claimant's place and
		 * joins the claim loop itself: it would otherwise just sleep on the barrier,
		 * so this saves a wake and usually the barrier's sleep/wake round trip too.
		 */
		void StartBatchedJobs(std::function<void(int)> func, int num, bool async) {

			if (num > 0) {
				struct BatchState {
					std::function<void(int)> func;
					std::atomic<int> nextIdx = 0;
					int num = 0;
				};
				auto batch = std::make_shared<BatchState>();
				batch->func = std::move(func);
				batch->num = num;

				int numJobs = std::min(num, GetNumThreads());
				if (!async)
					numJobs--;

				for (int i = 0; i < numJobs; i++) {
					StartJobAsync([batch] {
						int idx;
						while ((idx = batch->nextIdx.fetch_add(1)) < batch->num)
							batch->func(idx);
					});
				}

				if (!async) {
					int idx;
					while ((idx = batch->nextIdx.fetch_add(1)) < batch->num)
						batch->func(idx);
				}
			}

			if (!async)
				WaitUntilDone();
		}

		void WaitUntilDone() {
			// No pool yet means no jobs were ever started; don't create one just to wait.
			if (_tp)
				_tp->wait_for_tasks();
		}

		/** True once the pool has been created and sized. */
		bool IsStarted() const {
			return _tp != nullptr;
		}

		int GetNumThreads() {
			return (int)Pool().size();
		}
	};

	extern ThreadPool g_ThreadPool;
}