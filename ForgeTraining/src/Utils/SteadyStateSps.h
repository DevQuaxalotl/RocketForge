#pragma once

#include <cstdint>
#include <string>

namespace RocketForge
{
	/** Accumulates per-iteration timing into steady-state steps-per-second, skipping the first `warmupIterations`. */
	struct SteadyStateSps
	{
		int warmupIterations = 0;

		void AddIteration(uint64_t timesteps, double collectionSeconds, double consumptionSeconds);

		int MeasuredIterations() const;

		/** Σ timesteps / Σ (collection + consumption) over the measured iterations. **/
		double OverallSps() const;

		/** Σ timesteps / Σ collection over the measured iterations. **/
		double CollectionSps() const;

		/** Σ timesteps / Σ consumption over the measured iterations. **/
		double ConsumptionSps() const;

		std::string Summary() const;

	private:
		int _iterationsSeen = 0;
		int _measuredIterations = 0;
		uint64_t _timesteps = 0;
		double _collectionSeconds = 0;
		double _consumptionSeconds = 0;
	};
}
