#include "EventDecayTracker.h"
#include <algorithm>
#include <cmath>

namespace ForgeDebug {

	void EventDecayTracker::Update(float deltaTime) {
		for (auto& firing : activeFirings)
			firing.timeElapsed += deltaTime;

		// Prune expired firings
		activeFirings.erase(
			std::remove_if(activeFirings.begin(), activeFirings.end(),
				[this](const EventFiring& f) { return f.timeElapsed >= decayDuration; }),
			activeFirings.end()
		);
	}

	void EventDecayTracker::RecordEvent(float rawValue) {
		activeFirings.push_back({ 0.f, rawValue });
	}

	float EventDecayTracker::GetDecayValue() const {
		float maxDecay = 0.f;
		for (const auto& firing : activeFirings) {
			const float decay = std::max(0.f, 1.f - firing.timeElapsed / decayDuration);
			maxDecay = std::max(maxDecay, decay);
		}
		return maxDecay;
	}

	float EventDecayTracker::GetDisplayValue() const {
		if (activeFirings.empty())
			return 0.f;

		// Use the most recent firing
		const auto& latest = activeFirings.back();
		const float decay = std::max(0.f, 1.f - latest.timeElapsed / decayDuration);
		return latest.initialValue * decay;
	}

	void EventDecayTracker::Reset() {
		activeFirings.clear();
	}

}
