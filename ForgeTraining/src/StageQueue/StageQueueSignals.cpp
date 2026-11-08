#include "StageQueueSignals.h"

#include <atomic>

namespace {
	std::atomic<bool> g_stopRequested{ false };
	std::atomic<bool> g_skipRequested{ false };
}

void RocketForgeTraining::StageQueueSignalState::Reset() {
	g_stopRequested.store(false, std::memory_order_release);
	g_skipRequested.store(false, std::memory_order_release);
}

bool RocketForgeTraining::StageQueueSignalState::ConsumeStop() {
	return g_stopRequested.exchange(false, std::memory_order_acq_rel);
}

bool RocketForgeTraining::StageQueueSignalState::ConsumeSkip() {
	return g_skipRequested.exchange(false, std::memory_order_acq_rel);
}

void RocketForgeTraining::StageQueueSignalState::RaiseStop() {
	g_stopRequested.store(true, std::memory_order_release);
}

void RocketForgeTraining::StageQueueSignalState::RaiseSkip() {
	g_skipRequested.store(true, std::memory_order_release);
}

bool RocketForgeTraining::StageQueueSignalState::IsAnyRaised() {
	return g_stopRequested.load(std::memory_order_acquire) || g_skipRequested.load(std::memory_order_acquire);
}
