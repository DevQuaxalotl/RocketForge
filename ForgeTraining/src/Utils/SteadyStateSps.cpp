#include "SteadyStateSps.h"

#include <cmath>
#include <sstream>

namespace
{
	double SafeRate(uint64_t timesteps, double seconds)
	{
		return seconds > 0 ? timesteps / seconds : 0;
	}
}

void RocketForge::SteadyStateSps::AddIteration(uint64_t timesteps, double collectionSeconds, double consumptionSeconds)
{
	_iterationsSeen++;
	if (_iterationsSeen <= warmupIterations)
		return;

	_measuredIterations++;
	_timesteps += timesteps;
	_collectionSeconds += collectionSeconds;
	_consumptionSeconds += consumptionSeconds;
}

int RocketForge::SteadyStateSps::MeasuredIterations() const
{
	return _measuredIterations;
}

double RocketForge::SteadyStateSps::OverallSps() const
{
	return SafeRate(_timesteps, _collectionSeconds + _consumptionSeconds);
}

double RocketForge::SteadyStateSps::CollectionSps() const
{
	return SafeRate(_timesteps, _collectionSeconds);
}

double RocketForge::SteadyStateSps::ConsumptionSps() const
{
	return SafeRate(_timesteps, _consumptionSeconds);
}

std::string RocketForge::SteadyStateSps::Summary() const
{
	std::ostringstream summary;
	if (_measuredIterations == 0) {
		summary << "Steady state: no measured iterations (" << warmupIterations << " warmup)";
		return summary.str();
	}

	const double totalSeconds = _collectionSeconds + _consumptionSeconds;
	const double collectionPercent = totalSeconds > 0 ? 100.0 * _collectionSeconds / totalSeconds : 0;

	summary << "Steady state (" << _measuredIterations << " iterations after " << warmupIterations << " warmup): overall "
			<< std::llround(OverallSps()) << " SPS, collection " << std::llround(CollectionSps()) << " SPS, consumption "
			<< std::llround(ConsumptionSps()) << " SPS, collection " << std::llround(collectionPercent) << "% of time";
	return summary.str();
}
