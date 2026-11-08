#include "ThreadPool.h"

#include <thread>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <iterator>

#ifdef __linux__
#include <sched.h>
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <bit>
#endif

#include <iostream>

RLGC::ThreadPool RLGC::g_ThreadPool = {};

int RLGC::ParseCgroupV2CpuMaxCores(const std::string& contents) {
	std::istringstream iss(contents);
	std::string quotaStr, periodStr;
	if (!(iss >> quotaStr))
		return 0;                       // empty / whitespace
	if (quotaStr == "max")
		return 0;                       // unlimited
	if (!(iss >> periodStr))
		return 0;                       // malformed (quota with no period)

	long long quota = 0, period = 0;
	try {
		quota = std::stoll(quotaStr);
		period = std::stoll(periodStr);
	} catch (...) {
		return 0;                       // non-numeric tokens
	}
	return ParseCgroupV1QuotaCores(quota, period);
}

int RLGC::ParseCgroupV1QuotaCores(long long quotaUs, long long periodUs) {
	if (quotaUs < 0 || periodUs <= 0)
		return 0;                       // unlimited (quota < 0) or invalid period
	long long cores = quotaUs / periodUs;
	if (quotaUs > 0 && cores < 1)
		cores = 1;                      // a sub-one-core quota still needs one thread
	return (int)cores;
}

unsigned int RLGC::ResolveThreadCount(unsigned int hardwareConcurrency, int envOverride, int cgroupQuotaCores) {
	if (envOverride > 0)
		return (unsigned int)envOverride;   // explicit user override always wins

	unsigned int n = hardwareConcurrency;
	if (cgroupQuotaCores > 0 && (unsigned int)cgroupQuotaCores < n)
		n = (unsigned int)cgroupQuotaCores; // cap to the cgroup CPU budget
	if (n < 1)
		n = 1;                              // hardware_concurrency() can report 0
	return n;
}

std::vector<int> RLGC::ParseCpuList(const std::string& cpuList) {
	std::vector<int> result;
	std::stringstream ss(cpuList);
	std::string token;
	while (std::getline(ss, token, ',')) {
		// Trim surrounding whitespace (handles trailing newline from /sys files).
		size_t a = token.find_first_not_of(" \t\r\n");
		if (a == std::string::npos)
			continue;
		size_t b = token.find_last_not_of(" \t\r\n");
		token = token.substr(a, b - a + 1);

		size_t dash = token.find('-');
		try {
			if (dash == std::string::npos) {
				result.push_back(std::stoi(token));
			} else {
				int lo = std::stoi(token.substr(0, dash));
				int hi = std::stoi(token.substr(dash + 1));
				for (int i = lo; i <= hi; ++i)
					result.push_back(i);
			}
		} catch (...) {
			// Skip malformed tokens.
		}
	}
	std::sort(result.begin(), result.end());
	result.erase(std::unique(result.begin(), result.end()), result.end());
	return result;
}

std::vector<int> RLGC::SelectBestNodeCpus(const std::vector<std::vector<int>>& nodeCpuLists, const std::vector<int>& allowedCpus) {
	if (nodeCpuLists.empty())
		return allowedCpus;

	std::vector<int> allowedSorted = allowedCpus;
	std::sort(allowedSorted.begin(), allowedSorted.end());

	std::vector<int> best;
	for (const auto& node : nodeCpuLists) {
		std::vector<int> nodeSorted = node;
		std::sort(nodeSorted.begin(), nodeSorted.end());

		std::vector<int> inter;
		std::set_intersection(
			nodeSorted.begin(), nodeSorted.end(),
			allowedSorted.begin(), allowedSorted.end(),
			std::back_inserter(inter));

		if (inter.size() > best.size())   // strict '>' → ties keep the earlier node
			best = std::move(inter);
	}
	return best;
}

unsigned int RLGC::ResolvePinnedThreadCount(unsigned int hardwareConcurrency, int envOverride, int cgroupQuotaCores, size_t cpuLimit) {
	unsigned int base = ResolveThreadCount(hardwareConcurrency, envOverride, cgroupQuotaCores);
	if (envOverride > 0)
		return base;                       // explicit override wins; not node-capped
	if (cpuLimit > 0 && cpuLimit < (size_t)base)
		base = (unsigned int)cpuLimit; // cap to one NUMA node's CPUs, or the affinity mask's CPUs on Windows
	if (base < 1)
		base = 1;
	return base;
}

// Current thread's allowed CPUs (respects any external taskset/cpuset). Linux only.
static std::vector<int> ReadAllowedCpus() {
#ifdef __linux__
	cpu_set_t set;
	CPU_ZERO(&set);
	std::vector<int> cpus;
	if (sched_getaffinity(0, sizeof(set), &set) == 0) {
		for (int i = 0; i < CPU_SETSIZE; ++i)
			if (CPU_ISSET(i, &set))
				cpus.push_back(i);
	}
	return cpus;
#else
	return {};
#endif
}

// Per-NUMA-node CPU lists from /sys (node0, node1, ...). Linux only.
static std::vector<std::vector<int>> ReadNodeCpuLists() {
#ifdef __linux__
	std::vector<std::vector<int>> nodes;
	for (int n = 0; ; ++n) {
		std::ifstream f("/sys/devices/system/node/node" + std::to_string(n) + "/cpulist");
		if (!f)
			break;
		std::string line;
		std::getline(f, line);
		nodes.push_back(RLGC::ParseCpuList(line));
	}
	return nodes;
#else
	return {};
#endif
}

const std::vector<int>& RLGC::PreferredNodeCpus() {
	// Computed once: the allowed CPUs of the single best NUMA node.
	static const std::vector<int> cpus = []() -> std::vector<int> {
		std::vector<int> allowed = ReadAllowedCpus();
		std::vector<int> best = SelectBestNodeCpus(ReadNodeCpuLists(), allowed);
		if (best.empty())
			best = allowed;                // fallback: no node topology available
		return best;
	}();
	return cpus;
}

void RLGC::PinCurrentThreadToPreferredNode() {
#ifdef __linux__
	const std::vector<int>& cpus = PreferredNodeCpus();
	if (cpus.empty())
		return;
	cpu_set_t set;
	CPU_ZERO(&set);
	for (int c : cpus)
		if (c >= 0 && c < CPU_SETSIZE)
			CPU_SET(c, &set);
	sched_setaffinity(0, sizeof(set), &set);   // 0 = calling thread
#endif
}

// Read the container's CPU quota (cgroup v2 then v1). Returns whole cores, or 0
// when there is no quota / not on Linux. Not unit-tested directly — the parsing it
// delegates to is covered by ParseCgroupV2CpuMaxCores / ParseCgroupV1QuotaCores.
static int DetectCgroupCpuQuotaCores() {
#ifdef __linux__
	// cgroup v2: a single file "cpu.max" with "<quota_us> <period_us>" or "max ...".
	{
		std::ifstream f("/sys/fs/cgroup/cpu.max");
		if (f) {
			std::string line;
			std::getline(f, line);
			int cores = RLGC::ParseCgroupV2CpuMaxCores(line);
			if (cores > 0)
				return cores;
		}
	}
	// cgroup v1: two files under cpu/.
	{
		std::ifstream qf("/sys/fs/cgroup/cpu/cpu.cfs_quota_us");
		std::ifstream pf("/sys/fs/cgroup/cpu/cpu.cfs_period_us");
		if (qf && pf) {
			long long quota = -1, period = 0;
			qf >> quota;
			pf >> period;
			int cores = RLGC::ParseCgroupV1QuotaCores(quota, period);
			if (cores > 0)
				return cores;
		}
	}
#endif
	return 0;
}

// Read an integer environment variable, or invalidValue when unset/unparseable.
static int ReadIntEnv(const char* name, int invalidValue) {
	const char* raw = nullptr;
#ifdef _WIN32
	char* buf = nullptr;
	size_t len = 0;
	if (_dupenv_s(&buf, &len, name) == 0)
		raw = buf;
#else
	raw = std::getenv(name);
#endif
	int value = invalidValue;
	if (raw) {
		try {
			value = std::stoi(raw);
		} catch (...) {
			value = invalidValue;
		}
	}
#ifdef _WIN32
	free(buf);
#endif
	return value;
}

std::vector<int> RLGC::SelectPerformanceCoreCpus(const std::vector<ProcessorCoreInfo>& cores) {
	if (cores.empty())
		return {};

	int lowestClass = cores.front().efficiencyClass;
	int highestClass = lowestClass;
	for (const ProcessorCoreInfo& core : cores) {
		lowestClass = std::min(lowestClass, core.efficiencyClass);
		highestClass = std::max(highestClass, core.efficiencyClass);
	}

	if (lowestClass == highestClass)
		return {};

	std::vector<int> performanceCpus;
	for (const ProcessorCoreInfo& core : cores) {
		if (core.efficiencyClass == highestClass)
			performanceCpus.insert(performanceCpus.end(), core.logicalCpus.begin(), core.logicalCpus.end());
	}

	std::sort(performanceCpus.begin(), performanceCpus.end());
	return performanceCpus;
}

#ifdef _WIN32
// Cores with their efficiency class. Callers must have checked there is only one processor group, so bit positions are CPU indices.
static std::vector<RLGC::ProcessorCoreInfo> ReadProcessorCores() {
	DWORD bufferBytes = 0;
	GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bufferBytes);
	if (bufferBytes == 0)
		return {};

	std::vector<char> buffer(bufferBytes);
	if (!GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buffer.data(), &bufferBytes))
		return {};

	std::vector<RLGC::ProcessorCoreInfo> cores;
	for (DWORD offset = 0; offset < bufferBytes;) {
		const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* record = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)(buffer.data() + offset);

		RLGC::ProcessorCoreInfo core;
		core.efficiencyClass = record->Processor.EfficiencyClass;
		for (WORD groupIndex = 0; groupIndex < record->Processor.GroupCount; groupIndex++) {
			const GROUP_AFFINITY& group = record->Processor.GroupMask[groupIndex];
			for (int bit = 0; bit < (int)(sizeof(KAFFINITY) * 8); bit++) {
				if ((group.Mask >> bit) & 1)
					core.logicalCpus.push_back(bit);
			}
		}
		cores.push_back(std::move(core));

		offset += record->Size;
	}
	return cores;
}

// CPUs in the process affinity mask, or 0 when it cannot be read (including when the process spans several processor groups).
static size_t ProcessAffinityCpuCount() {
	DWORD_PTR processMask = 0;
	DWORD_PTR systemMask = 0;
	if (!GetProcessAffinityMask(GetCurrentProcess(), &processMask, &systemMask))
		return 0;
	return (size_t)std::popcount((unsigned long long)processMask);
}
#endif

bool RLGC::RestrictProcessToPerformanceCores() {
#ifdef _WIN32
	if (GetActiveProcessorGroupCount() > 1) {
		std::cerr << "[RLGC] Performance cores only: skipped, this machine has more than one processor group." << std::endl;
		return false;
	}

	std::vector<int> performanceCpus = SelectPerformanceCoreCpus(ReadProcessorCores());
	if (performanceCpus.empty()) {
		std::cerr << "[RLGC] Performance cores only: skipped, this CPU is not hybrid." << std::endl;
		return false;
	}

	DWORD_PTR affinityMask = 0;
	for (int cpu : performanceCpus)
		affinityMask |= (DWORD_PTR)1 << cpu;

	if (!SetProcessAffinityMask(GetCurrentProcess(), affinityMask)) {
		std::cerr << "[RLGC] Performance cores only: SetProcessAffinityMask failed (error " << GetLastError() << ")." << std::endl;
		return false;
	}
	std::cerr << "[RLGC] Performance cores only: restricted process to " << performanceCpus.size() << " CPUs (mask 0x" << std::hex
		<< (unsigned long long)affinityMask << std::dec << ")." << std::endl;

	if (g_ThreadPool.IsStarted())
		std::cerr << "[RLGC] Performance cores only: the collection pool was already sized at " << g_ThreadPool.GetNumThreads()
			<< " threads and keeps that size until restart." << std::endl;

	// Without this, Windows may still schedule the process at efficiency-core speed (EcoQoS).
	PROCESS_POWER_THROTTLING_STATE throttling = {};
	throttling.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
	throttling.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
	throttling.StateMask = 0;
	if (SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &throttling, sizeof(throttling)))
		std::cerr << "[RLGC] Performance cores only: power throttling disabled." << std::endl;
	else
		std::cerr << "[RLGC] Performance cores only: could not disable power throttling (error " << GetLastError() << ")." << std::endl;

	return true;
#else
	std::cerr << "[RLGC] Performance cores only: skipped, only supported on Windows." << std::endl;
	return false;
#endif
}

unsigned int RLGC::EffectiveThreadCount() {
#ifdef _WIN32
	// Cap at the CPUs the process affinity allows.
	const size_t cpuLimit = ProcessAffinityCpuCount();
#else
	const size_t cpuLimit = PreferredNodeCpus().size();
#endif

	return ResolvePinnedThreadCount(std::thread::hardware_concurrency(), ReadIntEnv("RLGC_NUM_THREADS", 0), DetectCgroupCpuQuotaCores(), cpuLimit);
}

int RLGC::ResolveIdleSpinMicros(int envOverride) {
	if (envOverride >= 0)
		return envOverride;
	return DEFAULT_IDLE_SPIN_MICROS;
}

int RLGC::EffectiveIdleSpinMicros() {
	return ResolveIdleSpinMicros(ReadIntEnv("RLGC_SPIN_US", -1));
}