#pragma once
#include <vector>

namespace ForgeDebug {

	struct EventFiring {
		/** Seconds since the event fired */
		float timeElapsed;

		/** The raw value when it fired */
		float initialValue;
	};

	class EventDecayTracker {
	public:
		/** Seconds for the event to fully fade */
		float decayDuration = 2.0f;

		/** Update all tracked events by deltaTime, prune expired */
		void Update(float deltaTime);

		/** Record a new event firing */
		void RecordEvent(float rawValue);

		/**
		 * Get the current decay value in [0, 1].
		 * Returns 1.0 at the moment of firing, fading to 0.0 over decayDuration.
		 * If multiple events overlap, returns the max.
		 */
		float GetDecayValue() const;

		/** Get the display value (initialValue * decayFraction) for the most recent active firing */
		float GetDisplayValue() const;

		void Reset();

	private:
		std::vector<EventFiring> activeFirings;
	};

}
