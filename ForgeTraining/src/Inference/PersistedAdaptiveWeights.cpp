#include "PersistedAdaptiveWeights.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <string>

using namespace RLGC;

namespace RocketForge
{
	std::vector<AdaptivePersistEntry> LoadPersistedAdaptiveWeights(const std::filesystem::path& checkpointFolder)
	{
		std::vector<AdaptivePersistEntry> entries;

		std::ifstream file(checkpointFolder / "RUNNING_STATS.json");
		if (!file.is_open())
			return entries;

		// Non-throwing parse — codebase convention is to avoid exceptions.
		nlohmann::json j = nlohmann::json::parse(file, nullptr, /*allow_exceptions=*/false);
		if (j.is_discarded() || !j.is_object())
			return entries;

		const nlohmann::json::const_iterator section = j.find("adaptive_reward_weights");
		if (section == j.end() || !section->is_array())
			return entries;

		for (const nlohmann::json& item : *section) {
			if (!item.is_object())
				continue;

			AdaptivePersistEntry entry;
			entry.name        = item.value("name", std::string{});
			entry.playerCount = item.value("player_count", 0);
			entry.weight      = item.value("weight", 0.f);
			entry.rawRateEma  = item.value("raw_rate_ema", 0.f);

			if (!entry.name.empty())
				entries.push_back(entry);
		}

		return entries;
	}

	void ApplyPersistedAdaptiveWeights(
		const std::vector<AdaptivePersistEntry>& entries, int blue, int orange, std::vector<WeightedReward>& rewards)
	{
		if (entries.empty())
			return;

		for (WeightedReward& weighted : rewards) {
			if (!weighted.adaptiveSettings.enabled)
				continue;

			const AdaptivePersistEntry* entry =
				AdaptiveRewardWeightController::FindEntryForArena(entries, weighted.reward->GetName(), blue, orange);
			if (entry)
				weighted.weight = entry->weight;
		}
	}
}
