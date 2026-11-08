#pragma once

#include "PolicyVersionManager.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace GGL {
	/** Tracks one opponent category's share with a bounded, exponentially-decayed sample history. **/
	class OpponentSampleTracker {
	public:
		OpponentSampleTracker(float targetFraction, uint64_t sampleWindow) :
			targetFraction(std::clamp(targetFraction, 0.f, 1.f)),
			sampleWindow(std::max<uint64_t>(sampleWindow, 1)) {
		}

		void Reset() {
			categorySamples = 0;
			totalSamples = 0;
		}

		void AddSamples(uint64_t category, uint64_t total) {
			categorySamples += category;
			totalSamples += total;
			if (totalSamples <= sampleWindow)
				return;

			const double scale = sampleWindow / totalSamples;
			categorySamples *= scale;
			totalSamples *= scale;
		}

		float GetSampleFraction() const {
			return totalSamples > 0 ? (float)(categorySamples / totalSamples) : 0.f;
		}

		float GetCorrectedTargetFraction() const {
			if (totalSamples <= 0)
				return targetFraction;

			// Feedback corrects episode-length and uneven-team effects without making
			// assignment deterministic when the rolling fraction crosses the target.
			const float error = targetFraction - GetSampleFraction();
			return std::clamp(targetFraction + error * 0.5f, 0.f, 1.f);
		}

		/** Converts a learner-sample fraction into an episode-assignment probability. **/
		static float SampleFractionToEpisodeProbability(
			float sampleFraction,
			int normalLearnerPlayers,
			int opponentLearnerPlayers) {
			if (sampleFraction <= 0 || normalLearnerPlayers <= 0 || opponentLearnerPlayers <= 0)
				return 0.f;
			if (sampleFraction >= 1)
				return 1.f;

			const float normalSamples = (float)normalLearnerPlayers;
			const float opponentSamples = (float)opponentLearnerPlayers;
			const float denominator = opponentSamples * (1.f - sampleFraction)
				+ sampleFraction * normalSamples;
			return denominator > 0
				? std::clamp(sampleFraction * normalSamples / denominator, 0.f, 1.f)
				: 0.f;
		}

		/**
		 * Converts this category's corrected learner-sample target into an episode
		 * probability when it is the only opponent category being scheduled.
		 **/
		float GetEpisodeProbability(int normalLearnerPlayers, int opponentLearnerPlayers) const {
			return SampleFractionToEpisodeProbability(
				GetCorrectedTargetFraction(),
				normalLearnerPlayers,
				opponentLearnerPlayers);
		}

		float GetTargetFraction() const {
			return targetFraction;
		}

	private:
		float targetFraction;
		double sampleWindow;
		double categorySamples = 0;
		double totalSamples = 0;
	};

	/** Returns true only at the native observation/inference boundary. **/
	inline bool ShouldInferStaticOpponentAction(int cadenceStep) {
		return cadenceStep == 0;
	}

	/** Advances and wraps one static policy's cadence relative to learner steps. **/
	inline int AdvanceStaticOpponentCadence(int cadenceStep, int cadenceRatio) {
		const int safeCadenceRatio = std::max(cadenceRatio, 1);
		return cadenceStep + 1 >= safeCadenceRatio ? 0 : cadenceStep + 1;
	}

	/** Tracks timestep rotation boundaries and retains rotations delayed by draining episodes. **/
	class OpponentRotationClock {
	public:
		explicit OpponentRotationClock(uint64_t intervalTimesteps) :
			intervalTimesteps(intervalTimesteps) {
			if (intervalTimesteps == 0)
				throw std::invalid_argument("Opponent rotation interval must be positive");
		}

		void Observe(uint64_t totalTimesteps) {
			const uint64_t currentBucket = totalTimesteps / intervalTimesteps;
			if (!initialized) {
				lastObservedBucket = currentBucket;
				initialized = true;
				return;
			}
			if (currentBucket <= lastObservedBucket)
				return;

			pendingRotations += currentBucket - lastObservedBucket;
			lastObservedBucket = currentBucket;
		}

		bool IsPending() const {
			return pendingRotations > 0;
		}

		void ConsumeOne() {
			if (pendingRotations > 0)
				pendingRotations--;
		}

		void ClearPending() {
			pendingRotations = 0;
		}

		uint64_t GetPendingCount() const {
			return pendingRotations;
		}

		uint64_t GetIntervalTimesteps() const {
			return intervalTimesteps;
		}

		uint64_t GetNextRotationTimestep() const {
			return initialized ? (lastObservedBucket + 1) * intervalTimesteps : intervalTimesteps;
		}

	private:
		uint64_t intervalTimesteps;
		uint64_t lastObservedBucket = 0;
		uint64_t pendingRotations = 0;
		bool initialized = false;
	};

	/**
	 * Maintains selectable recent opponents plus one optional drain-only version.
	 * New episodes never select the draining version, while arenas already using it
	 * may finish normally before its inference slot is recycled. Cohort membership
	 * and episode assignments are selected uniformly at random.
	 **/
	class RecentOpponentStack {
	public:
		explicit RecentOpponentStack(
			int selectableCapacity,
			uint64_t rotationIntervalTimesteps = 10'000'000,
			uint64_t randomSeed = std::random_device{}()) :
			selectableCapacity(selectableCapacity),
			rotationClock(rotationIntervalTimesteps),
			randomEngine(randomSeed) {
			if (selectableCapacity <= 0)
				throw std::invalid_argument("RecentOpponentStack selectable capacity must be positive");
		}

		void Update(
			std::vector<PolicyVersionKey> availableVersions,
			const std::map<PolicyVersionKey, int>& activeArenaCounts,
			uint64_t totalTimesteps) {
			rotationClock.Observe(totalTimesteps);
			std::sort(availableVersions.begin(), availableVersions.end(), [](const auto& a, const auto& b) {
				if (a.timesteps != b.timesteps)
					return a.timesteps < b.timesteps;
				return a.sourceVersionId < b.sourceVersionId;
			});
			availableVersions.erase(std::unique(availableVersions.begin(), availableVersions.end()), availableVersions.end());

			const std::set<PolicyVersionKey> availableSet(availableVersions.begin(), availableVersions.end());
			std::erase_if(selectable, [&](const PolicyVersionKey& key) {
				return !availableSet.contains(key) && !HasActiveArena(key, activeArenaCounts);
			});
			if (draining.has_value() && !HasActiveArena(draining.value(), activeArenaCounts))
				draining.reset();

			while ((int)selectable.size() < selectableCapacity) {
				std::optional<PolicyVersionKey> candidate = SelectRandomUnused(availableVersions);
				if (!candidate.has_value())
					break;
				selectable.push_back(candidate.value());
			}

			if (!rotationClock.IsPending() || draining.has_value()
				|| (int)selectable.size() < selectableCapacity)
				return;

			std::optional<PolicyVersionKey> candidate = SelectRandomUnused(availableVersions);
			if (!candidate.has_value()) {
				rotationClock.ClearPending();
				return;
			}

			const size_t replacementIdx = RandomIndex(selectable.size());
			const PolicyVersionKey outgoing = selectable[replacementIdx];
			selectable[replacementIdx] = candidate.value();
			if (HasActiveArena(outgoing, activeArenaCounts))
				draining = outgoing;
			rotationClock.ConsumeOne();
			rotations++;
		}

		std::optional<PolicyVersionKey> SelectForEpisode() {
			if (selectable.empty())
				return std::nullopt;
			return selectable[RandomIndex(selectable.size())];
		}

		const std::vector<PolicyVersionKey>& GetSelectable() const {
			return selectable;
		}

		const std::optional<PolicyVersionKey>& GetDraining() const {
			return draining;
		}

		std::vector<PolicyVersionKey> GetProtectedVersions() const {
			std::vector<PolicyVersionKey> result = selectable;
			if (draining.has_value())
				result.push_back(draining.value());
			return result;
		}

		uint64_t GetRotationCount() const {
			return rotations;
		}

		uint64_t GetPendingRotationCount() const {
			return rotationClock.GetPendingCount();
		}

		uint64_t GetNextRotationTimestep() const {
			return rotationClock.GetNextRotationTimestep();
		}

	private:
		int selectableCapacity;
		OpponentRotationClock rotationClock;
		mutable std::mt19937_64 randomEngine;
		std::vector<PolicyVersionKey> selectable;
		std::optional<PolicyVersionKey> draining;
		uint64_t rotations = 0;

		static bool HasActiveArena(
			const PolicyVersionKey& key,
			const std::map<PolicyVersionKey, int>& activeArenaCounts) {
			auto itr = activeArenaCounts.find(key);
			return itr != activeArenaCounts.end() && itr->second > 0;
		}

		bool IsUsed(const PolicyVersionKey& key) const {
			return std::find(selectable.begin(), selectable.end(), key) != selectable.end()
				|| (draining.has_value() && draining.value() == key);
		}

		std::optional<PolicyVersionKey> SelectRandomUnused(
			const std::vector<PolicyVersionKey>& availableVersions) const {
			std::vector<PolicyVersionKey> unused;
			for (const PolicyVersionKey& version : availableVersions)
				if (!IsUsed(version))
					unused.push_back(version);
			return unused.empty()
				? std::nullopt
				: std::optional<PolicyVersionKey>(unused[RandomIndex(unused.size())]);
		}

		size_t RandomIndex(size_t count) const {
			return std::uniform_int_distribution<size_t>(0, count - 1)(randomEngine);
		}
	};

	/** Static policy metadata needed by the bounded opponent cohort. **/
	struct StaticOpponentCandidate {
		int versionIndex = -1;
		int maxTeamSize = 1;
	};

	/**
	 * Keeps a random bounded set of static policies selectable while allowing one
	 * removed policy to drain from active episodes.
	 **/
	class StaticOpponentCohort {
	public:
		explicit StaticOpponentCohort(
			int selectableCapacity,
			uint64_t rotationIntervalTimesteps = 10'000'000,
			uint64_t randomSeed = std::random_device{}()) :
			selectableCapacity(selectableCapacity),
			rotationClock(rotationIntervalTimesteps),
			randomEngine(randomSeed) {
			if (selectableCapacity <= 0)
				throw std::invalid_argument("StaticOpponentCohort selectable capacity must be positive");
		}

		void Update(
			std::vector<StaticOpponentCandidate> availableCandidates,
			const std::map<int, int>& activeArenaCounts,
			uint64_t totalTimesteps) {
			rotationClock.Observe(totalTimesteps);
			NormalizeCandidates(availableCandidates);
			candidates = std::move(availableCandidates);
			maxTeamSizeByVersion.clear();
			maxAvailableTeamSize = 1;
			for (const StaticOpponentCandidate& candidate : candidates) {
				maxTeamSizeByVersion[candidate.versionIndex] = candidate.maxTeamSize;
				maxAvailableTeamSize = std::max(maxAvailableTeamSize, candidate.maxTeamSize);
			}

			const std::set<int> availableVersions = MakeAvailableVersionSet();
			std::erase_if(selectable, [&](int versionIndex) {
				return !availableVersions.contains(versionIndex)
					&& !HasActiveArena(versionIndex, activeArenaCounts);
			});
			if (draining.has_value() && !HasActiveArena(draining.value(), activeArenaCounts))
				draining.reset();

			if (!HasCompatibleVersion(maxAvailableTeamSize)) {
				const std::optional<int> compatibilityAnchor = SelectRandomUnused(true);
				if (compatibilityAnchor.has_value()) {
					selectable.push_back(compatibilityAnchor.value());
				}
			}
			while ((int)selectable.size() < selectableCapacity) {
				const std::optional<int> candidate = SelectRandomUnused(false);
				if (!candidate.has_value())
					break;
				selectable.push_back(candidate.value());
			}

			if (!rotationClock.IsPending() || draining.has_value()
				|| (int)selectable.size() < selectableCapacity)
				return;

			const std::optional<std::pair<int, int>> replacement = SelectRandomReplacement();
			if (!replacement.has_value()) {
				rotationClock.ClearPending();
				return;
			}

			const int incoming = replacement->first;
			const int replacementIdx = replacement->second;
			const int outgoing = selectable[replacementIdx];
			selectable[replacementIdx] = incoming;
			if (HasActiveArena(outgoing, activeArenaCounts))
				draining = outgoing;
			rotationClock.ConsumeOne();
			rotations++;
		}

		bool HasCompatibleVersion(int teamSize) const {
			return std::any_of(selectable.begin(), selectable.end(), [&](int versionIndex) {
				return MaxTeamSize(versionIndex) >= teamSize;
			});
		}

		/** Returns whether an existing arena assignment can continue into another episode. **/
		bool CanContinueEpisode(int versionIndex, int teamSize) const {
			return MaxTeamSize(versionIndex) >= teamSize
				&& std::find(selectable.begin(), selectable.end(), versionIndex) != selectable.end();
		}

		std::optional<int> SelectForEpisode(
			int teamSize,
			std::optional<int> previousVersion = std::nullopt) {
			std::vector<int> compatible;
			for (int versionIndex : selectable) {
				if (MaxTeamSize(versionIndex) >= teamSize)
					compatible.push_back(versionIndex);
			}
			// A random-per-episode visualizer should visibly exercise different models
			// when another compatible choice exists, while still supporting a one-model list.
			if (previousVersion.has_value() && compatible.size() > 1)
				std::erase(compatible, previousVersion.value());
			return compatible.empty()
				? std::nullopt
				: std::optional<int>(compatible[RandomIndex(compatible.size())]);
		}

		const std::vector<int>& GetSelectable() const {
			return selectable;
		}

		const std::optional<int>& GetDraining() const {
			return draining;
		}

		uint64_t GetRotationCount() const {
			return rotations;
		}

		uint64_t GetPendingRotationCount() const {
			return rotationClock.GetPendingCount();
		}

		uint64_t GetNextRotationTimestep() const {
			return rotationClock.GetNextRotationTimestep();
		}

	private:
		int selectableCapacity;
		OpponentRotationClock rotationClock;
		mutable std::mt19937_64 randomEngine;
		int maxAvailableTeamSize = 1;
		std::vector<StaticOpponentCandidate> candidates;
		std::vector<int> selectable;
		std::optional<int> draining;
		std::map<int, int> maxTeamSizeByVersion;
		uint64_t rotations = 0;

		static void NormalizeCandidates(std::vector<StaticOpponentCandidate>& values) {
			std::sort(values.begin(), values.end(), [](const auto& a, const auto& b) {
				return a.versionIndex < b.versionIndex;
			});
			std::vector<StaticOpponentCandidate> normalized;
			for (const StaticOpponentCandidate& value : values) {
				if (value.versionIndex < 0)
					continue;
				if (!normalized.empty() && normalized.back().versionIndex == value.versionIndex) {
					normalized.back().maxTeamSize = std::max(normalized.back().maxTeamSize, value.maxTeamSize);
					continue;
				}
				normalized.push_back({ value.versionIndex, std::max(value.maxTeamSize, 1) });
			}
			values = std::move(normalized);
		}

		std::set<int> MakeAvailableVersionSet() const {
			std::set<int> result;
			for (const StaticOpponentCandidate& candidate : candidates)
				result.insert(candidate.versionIndex);
			return result;
		}

		static bool HasActiveArena(int versionIndex, const std::map<int, int>& activeArenaCounts) {
			const auto itr = activeArenaCounts.find(versionIndex);
			return itr != activeArenaCounts.end() && itr->second > 0;
		}

		bool IsUsed(int versionIndex) const {
			return std::find(selectable.begin(), selectable.end(), versionIndex) != selectable.end()
				|| (draining.has_value() && draining.value() == versionIndex);
		}

		int MaxTeamSize(int versionIndex) const {
			const auto itr = maxTeamSizeByVersion.find(versionIndex);
			return itr != maxTeamSizeByVersion.end() ? itr->second : 0;
		}

		std::optional<int> SelectRandomUnused(bool requireMaximumCompatibility) const {
			std::vector<int> unused;
			for (const StaticOpponentCandidate& candidate : candidates) {
				if (IsUsed(candidate.versionIndex)
					|| (requireMaximumCompatibility && candidate.maxTeamSize < maxAvailableTeamSize))
					continue;
				unused.push_back(candidate.versionIndex);
			}
			return unused.empty()
				? std::nullopt
				: std::optional<int>(unused[RandomIndex(unused.size())]);
		}

		std::optional<std::pair<int, int>> SelectRandomReplacement() const {
			std::vector<std::pair<int, int>> replacements;
			for (const StaticOpponentCandidate& incoming : candidates) {
				if (IsUsed(incoming.versionIndex))
					continue;
				for (int replacementIdx = 0; replacementIdx < (int)selectable.size(); replacementIdx++) {
					bool preservesMaximumCompatibility = incoming.maxTeamSize >= maxAvailableTeamSize;
					for (int idx = 0; idx < (int)selectable.size() && !preservesMaximumCompatibility; idx++) {
						preservesMaximumCompatibility = idx != replacementIdx
							&& MaxTeamSize(selectable[idx]) >= maxAvailableTeamSize;
					}
					if (preservesMaximumCompatibility)
						replacements.emplace_back(incoming.versionIndex, replacementIdx);
				}
			}
			return replacements.empty()
				? std::nullopt
				: std::optional<std::pair<int, int>>(replacements[RandomIndex(replacements.size())]);
		}

		size_t RandomIndex(size_t count) const {
			return std::uniform_int_distribution<size_t>(0, count - 1)(randomEngine);
		}
	};
}
