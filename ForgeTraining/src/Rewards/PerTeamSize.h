#pragma once

#include <initializer_list>
#include <stdexcept>

namespace RocketForge {

	/**
	 * A value that is either shared across all team sizes or specified independently per team size
	 * (1v1, 2v2, 3v3), resolved for a given matchup by its max team size. Shared base for RewardWeight
	 * and AdaptiveWeightConfig: holds the storage, resolution, and exactly-three validation both need.
	 */
	template <typename T>
	struct PerTeamSize {
		/** Number of supported team sizes: 1v1, 2v2, 3v3. */
		static constexpr int TEAM_SIZE_COUNT = 3;

		bool usePerTeamSize = false;
		T shared = {};
		T perTeamSize[TEAM_SIZE_COUNT] = {};

		/** Resolves the value for a specific max team size (1, 2, or 3). */
		T Resolve(int maxTeamSize) const {
			if (!usePerTeamSize)
				return shared;

			if (maxTeamSize < 1 || maxTeamSize > TEAM_SIZE_COUNT)
				throw std::runtime_error("PerTeamSize only supports max team sizes of 1, 2, and 3");

			return perTeamSize[maxTeamSize - 1];
		}

		/** True if pred holds for the shared value (shared mode) or any per-size value (per-size mode). */
		template <typename Pred>
		bool Any(Pred pred) const {
			if (!usePerTeamSize)
				return pred(shared);

			for (int i = 0; i < TEAM_SIZE_COUNT; i++)
				if (pred(perTeamSize[i]))
					return true;
			return false;
		}

	protected:
		void SetShared(const T& value) {
			usePerTeamSize = false;
			shared = value;
			for (int i = 0; i < TEAM_SIZE_COUNT; i++)
				perTeamSize[i] = T{};
		}

		void SetPerTeamSize(std::initializer_list<T> values) {
			if (values.size() != TEAM_SIZE_COUNT)
				throw std::runtime_error("PerTeamSize requires exactly three values: { 1v1, 2v2, 3v3 }");

			usePerTeamSize = true;
			shared = T{};

			int i = 0;
			for (const T& value : values)
				perTeamSize[i++] = value;
		}
	};
}
