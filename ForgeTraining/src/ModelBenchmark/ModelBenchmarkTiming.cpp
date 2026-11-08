#include "ModelBenchmarkInternal.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace RocketForgeTraining::ModelBenchmarkInternal
{
	double SecondsSince(std::chrono::steady_clock::time_point start) {
		return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
	}

	std::string FormatBenchmarkTiming(const BenchmarkTiming& timing, double simulationSeconds, int finishedGames) {
		const double trackedSeconds =
			timing.physicsSeconds +
			timing.obsMaskSeconds +
			timing.gpuTransferSeconds +
			timing.inferenceSeconds +
			timing.actionParseSeconds +
			timing.actionStepSeconds +
			timing.finishSeconds +
			timing.schedulerSeconds +
			timing.slotAssignSeconds +
			timing.batchPrepSeconds;
		const double unattributedSeconds = std::max(0.0, simulationSeconds - trackedSeconds);

		std::ostringstream out;
		out << "Performance:"
			<< "\n  Games/sec: " << std::fixed << std::setprecision(2)
			<< (simulationSeconds > 0.0 ? finishedGames / simulationSeconds : 0.0)
			<< "\n  Physics first half: " << timing.physicsSeconds << "s"
			<< "\n  Obs/masks: " << timing.obsMaskSeconds << "s"
			<< "\n  GPU transfer: " << timing.gpuTransferSeconds << "s"
			<< "\n  Inference: " << timing.inferenceSeconds << "s"
			<< "\n  Action parse: " << timing.actionParseSeconds << "s"
			<< "\n  Action step: " << timing.actionStepSeconds << "s"
			<< "\n  Finish bookkeeping: " << timing.finishSeconds << "s"
			<< "\n  Scheduler/refill: " << timing.schedulerSeconds << "s"
			<< "\n  Slot assign: " << timing.slotAssignSeconds << "s"
			<< " (arena setup " << timing.slotArenaSetupSeconds
			<< "s, creates " << timing.slotArenaCreates
			<< ", rebuilds " << timing.slotArenaRebuilds
			<< ", reuses " << timing.slotArenaReuses
			<< ", side runtime " << timing.slotSideRuntimeSeconds
			<< "s, snapshot " << timing.slotSnapshotSeconds
			<< "s, state init " << timing.slotStateInitSeconds
			<< "s, builder reset " << timing.slotBuilderResetSeconds << "s)"
			<< "\n  Batch prep: " << timing.batchPrepSeconds << "s"
			<< "\n  CUDA OOM recoveries: " << timing.cudaOomRecoveries
			<< " (graphs disabled: "
			<< (timing.cudaGraphsDisabledAfterOom ? "yes" : "no") << ")"
			<< "\n  Post-processing: " << timing.postProcessingSeconds << "s"
			<< "\n  Unattributed: " << unattributedSeconds << "s";
		return out.str();
	}

	std::string ModeName(int mode) {
		return std::to_string(mode) + "v" + std::to_string(mode);
	}

	double DefaultModeDurationWeight(int mode) {
		// Used only as a cold-start tie-breaker before enough completed games
		// exist to estimate mode length from the current run.
		switch (mode) {
		case 1: return 1.00;
		case 2: return 1.75;
		case 3: return 2.20;
		default: return std::max(1.0, 1.0 + 0.5 * std::max(0, mode - 1));
		}
	}

	double FinishedGameSeconds(int finishSteps, bool useVariableTiming, int tickSkip) {
		const int64_t physicsTicks = useVariableTiming
			? std::max(0, finishSteps)
			: (int64_t)std::max(0, finishSteps) * std::max(1, tickSkip);
		return physicsTicks / 120.0;
	}

	ModelBenchmarkGoalTiming ClassifyGoalTiming(
		bool goalScored,
		double goalTimeSeconds,
		double kickoffGoalWindowSeconds) {
		if (!goalScored)
			return ModelBenchmarkGoalTiming::NoGoal;
		return goalTimeSeconds <= kickoffGoalWindowSeconds
			? ModelBenchmarkGoalTiming::Kickoff
			: ModelBenchmarkGoalTiming::PostKickoff;
	}
}
