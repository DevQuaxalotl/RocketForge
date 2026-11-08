#pragma once

#include <cmath>
#include <cstdint>

namespace GGL {
	/**
	 * Float value that can either stay constant or follow a sine-wave schedule.
	 * Update() caches the active value so training code can read it cheaply.
	 **/
	struct ScheduledFloat {
		float target = 0.f;
		float amplitude = 0.f;
		uint64_t period = 0;
		uint64_t startTimestep = 0;
		float phaseOffsetRadians = 0.f;
		uint64_t lastTimestep = 0;
		bool hasLastTimestep = false;
		float currentValue = 0.f;

		ScheduledFloat() = default;

		ScheduledFloat(float value)
			: target(value), currentValue(value) {}

		ScheduledFloat(float target, float amplitude, uint64_t period)
			: target(target), amplitude(amplitude), period(period), currentValue(target) {}

		ScheduledFloat& operator=(float value) {
			target = value;
			amplitude = 0.f;
			period = 0;
			startTimestep = 0;
			phaseOffsetRadians = 0.f;
			lastTimestep = 0;
			hasLastTimestep = false;
			currentValue = value;
			return *this;
		}

		operator float() const {
			return Get();
		}

		float Get() const {
			return hasLastTimestep ? currentValue : target;
		}

		float MinValue() const {
			return target - std::abs(amplitude);
		}

		float MaxValue() const {
			return target + std::abs(amplitude);
		}

		bool IsScheduled() const {
			return amplitude != 0.f && period > 0;
		}

		bool HasFiniteInputs() const {
			return std::isfinite(target) && std::isfinite(amplitude) && std::isfinite(phaseOffsetRadians);
		}

		float ValueAt(uint64_t totalTimesteps) const {
			if (!IsScheduled())
				return target;

			const uint64_t elapsed = totalTimesteps > startTimestep
				? totalTimesteps - startTimestep
				: 0;
			const double phase = std::fmod(
				static_cast<double>(elapsed),
				static_cast<double>(period)) / static_cast<double>(period);
			constexpr double TWO_PI = 6.283185307179586476925286766559;
			return target + amplitude * static_cast<float>(std::sin(phase * TWO_PI + phaseOffsetRadians));
		}

		float Update(uint64_t totalTimesteps) {
			lastTimestep = totalTimesteps;
			hasLastTimestep = true;
			currentValue = ValueAt(totalTimesteps);
			return currentValue;
		}
	};

	inline float Resolve(float value) {
		return value;
	}

	inline float Resolve(const ScheduledFloat& value) {
		return value.Get();
	}
}
