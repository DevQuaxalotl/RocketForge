#pragma once

namespace RocketForgeTraining {

	/** Stop and skip flags. The runner queries these between stages and resets them at each stage's start. */
	struct StageQueueSignalState {
		static void Reset();              // sets both flags to false
		static bool ConsumeStop();        // reads-and-clears the stop flag
		static bool ConsumeSkip();        // reads-and-clears the skip flag

		static void RaiseStop();          // sets stop = true and forwards to learner
		static void RaiseSkip();          // sets skip = true and forwards to learner
		static bool IsAnyRaised();        // reads both flags without clearing them
	};
}
