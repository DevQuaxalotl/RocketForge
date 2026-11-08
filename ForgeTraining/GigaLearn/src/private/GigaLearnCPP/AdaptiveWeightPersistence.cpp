#include "AdaptiveWeightPersistence.h"

const GGL::EnumNames<RLGC::AdaptivePhase> RLGC::ADAPTIVE_PHASE_NAMES = { {
	{ RLGC::AdaptivePhase::Discovering, "Discovering" },
	{ RLGC::AdaptivePhase::Settled,     "Settled" },
	{ RLGC::AdaptivePhase::Mastered,    "Mastered" },
	{ RLGC::AdaptivePhase::Relearning,  "Relearning" },
} };

void RLGC::SerializeFields(GGL::JsonArchive& ar, AdaptivePersistEntry& entry) {
	ar.Field("name", entry.name);
	ar.Field("player_count", entry.playerCount);
	ar.Field("weight", entry.weight);
	ar.Field("raw_rate_ema", entry.rawRateEma);
	ar.Field("pending_exit_steps", entry.pendingExitSteps);

	// A missing "phase" marks a checkpoint written before phases existed, so it must stay absent on load.
	if (ar.IsSaving()) {
		if (entry.phase.has_value()) {
			AdaptivePhase phase = *entry.phase;
			ar.EnumField("phase", phase, ADAPTIVE_PHASE_NAMES);
		}
	} else if (ar.RawLoad()->contains("phase")) {
		const size_t errorsBefore = ar.GetErrors().size();
		AdaptivePhase phase = AdaptivePhase::Discovering;
		ar.EnumField("phase", phase, ADAPTIVE_PHASE_NAMES);
		if (ar.GetErrors().size() == errorsBefore)
			entry.phase = phase;
	}
}
