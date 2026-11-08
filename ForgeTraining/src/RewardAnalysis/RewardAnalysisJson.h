// ForgeTraining/src/RewardAnalysis/RewardAnalysisJson.h
#pragma once

#include "RewardAnalysis.h"

#include <nlohmann/json.hpp>

namespace RocketForge
{

/** JSON serialization helpers for RewardAnalysis data structs. Pulled into its own
    header so consumers that only need the data structs (e.g. the GUI widget) don't
    have to drag in nlohmann/json.hpp. **/
void to_json(nlohmann::json& j, const RewardImpactEntry& entry);
void from_json(const nlohmann::json& j, RewardImpactEntry& entry);
void to_json(nlohmann::json& j, const TeamSizeImpactResult& result);
void from_json(const nlohmann::json& j, TeamSizeImpactResult& result);
void to_json(nlohmann::json& j, const RewardAnalysisResult& result);
void from_json(const nlohmann::json& j, RewardAnalysisResult& result);

} // namespace RocketForge
