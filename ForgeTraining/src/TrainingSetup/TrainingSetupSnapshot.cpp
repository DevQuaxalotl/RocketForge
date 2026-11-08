#include <RocketSim/Engine.h>
#include "TrainingSetupSnapshot.h"

#include "AgentModelPaths.h"
#include "Storage/BuildStorage.h"
#include "StaticOpponents/StaticOpponentSelection.h"
#include "TrainingStage.h"

#include <GigaLearnCPP/PPO/PPOLearnerConfig.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>

using namespace RocketForge;

namespace
{
	std::string FormatValue(const std::string& value)
	{
		return value;
	}

	std::string FormatValue(const char* value)
	{
		return value ? std::string(value) : std::string();
	}

	std::string FormatValue(bool value)
	{
		return value ? "true" : "false";
	}

	std::string FormatAlgorithm(GGL::LearnerAlgorithm algorithm)
	{
		return algorithm == GGL::LearnerAlgorithm::PPG ? "PPG" : "PPO";
	}

	template<typename T>
	std::string FormatValue(T value)
	{
		std::ostringstream stream;
		stream << std::setprecision(6) << std::defaultfloat << value;
		return stream.str();
	}

	std::string FormatValue(const GGL::ScheduledFloat& value)
	{
		return FormatValue(value.Get());
	}

	std::string FormatTeamSizes(const std::vector<WeightedTeamSize>& teamSizes)
	{
		if (teamSizes.empty())
			return "(none)";

		std::ostringstream stream;
		for (size_t i = 0; i < teamSizes.size(); ++i)
		{
			if (i > 0)
				stream << ", ";
			stream << teamSizes[i].teamSizeA << "v" << teamSizes[i].teamSizeB << " x" << FormatValue(teamSizes[i].weight);
		}

		return stream.str();
	}

	void AddParameter(TrainingSetupSnapshot& snapshot, const std::string& group, const std::string& name, const std::string& value)
	{
		snapshot.hyperparameters.push_back({ group, name, value });
	}

	template<typename T>
	void AddParameter(TrainingSetupSnapshot& snapshot, const std::string& group, const std::string& name, T value)
	{
		AddParameter(snapshot, group, name, FormatValue(value));
	}

	std::string RewardCategoryToString(RLGC::RewardCategory category)
	{
		return category == RLGC::RewardCategory::EVENT ? "event" : "continuous";
	}

	void AddScheduledFloatParameters(TrainingSetupSnapshot& snapshot, const std::string& name, const GGL::ScheduledFloat& scheduled)
	{
		if (!scheduled.IsScheduled())
			return;

		AddParameter(snapshot, "Schedules", name + " Target", scheduled.target);
		AddParameter(snapshot, "Schedules", name + " Amplitude", scheduled.amplitude);
		AddParameter(snapshot, "Schedules", name + " Min", scheduled.MinValue());
		AddParameter(snapshot, "Schedules", name + " Max", scheduled.MaxValue());
		AddParameter(snapshot, "Schedules", name + " Period Timesteps", scheduled.period);
		if (scheduled.startTimestep != 0)
			AddParameter(snapshot, "Schedules", name + " Start Timestep", scheduled.startTimestep);
	}

	std::string DisplayValue(const std::string& value)
	{
		return value.empty() ? "(empty)" : value;
	}

	void AddSetupDiff(
		std::vector<TrainingSetupDiff>& diffs,
		const std::string& field,
		const std::string& savedValue,
		const std::string& currentValue)
	{
		if (savedValue == currentValue)
			return;

		diffs.push_back({ field, DisplayValue(savedValue), DisplayValue(currentValue) });
	}

	std::string FormatRange(const RLGC::RewardValueRange& range)
	{
		return "[" + FormatValue(range.minValue) + ", " + FormatValue(range.maxValue) + "]";
	}

	std::string FormatRewardSummary(const RewardTableEntry& reward)
	{
		return reward.name +
			" (weight " + FormatValue(reward.weight) +
			", range " + FormatRange(reward.range) +
			", effective max " + FormatValue(reward.effectiveMax) +
			", " + RewardCategoryToString(reward.category) + ")";
	}

	void AddRewardDiffs(
		std::vector<TrainingSetupDiff>& diffs,
		const std::vector<RewardTableEntry>& savedRewards,
		const std::vector<RewardTableEntry>& currentRewards)
	{
		AddSetupDiff(diffs, "Reward Count", FormatValue(savedRewards.size()), FormatValue(currentRewards.size()));

		const size_t sharedCount = std::min(savedRewards.size(), currentRewards.size());
		for (size_t i = 0; i < sharedCount; ++i) {
			const RewardTableEntry& saved = savedRewards[i];
			const RewardTableEntry& current = currentRewards[i];
			const std::string label = !saved.name.empty() && saved.name == current.name
				? "Reward / " + saved.name
				: "Reward " + FormatValue(i + 1);

			AddSetupDiff(diffs, label + " / Name", saved.name, current.name);
			AddSetupDiff(diffs, label + " / Weight", FormatValue(saved.weight), FormatValue(current.weight));
			AddSetupDiff(diffs, label + " / Range", FormatRange(saved.range), FormatRange(current.range));
			AddSetupDiff(diffs, label + " / Effective Max", FormatValue(saved.effectiveMax), FormatValue(current.effectiveMax));
			AddSetupDiff(diffs, label + " / Category", RewardCategoryToString(saved.category), RewardCategoryToString(current.category));
		}

		for (size_t i = sharedCount; i < savedRewards.size(); ++i)
			AddSetupDiff(diffs, "Reward " + FormatValue(i + 1), FormatRewardSummary(savedRewards[i]), "(missing)");

		for (size_t i = sharedCount; i < currentRewards.size(); ++i)
			AddSetupDiff(diffs, "Reward " + FormatValue(i + 1), "(missing)", FormatRewardSummary(currentRewards[i]));
	}

	using ParameterKey = std::pair<std::string, std::string>;

	std::map<ParameterKey, std::string> BuildIdentityParameterMap(const std::vector<TrainingSetupParameter>& parameters)
	{
		std::map<ParameterKey, std::string> result;
		for (const TrainingSetupParameter& parameter : parameters) {
			if (!TrainingSetupParameterAffectsIdentity(parameter))
				continue;

			// Top-level snapshot fields report Agent, Stage, and Version ID.
			if (parameter.group == "Versioning")
				continue;

			result[{ parameter.group, parameter.name }] = parameter.value;
		}

		return result;
	}

	void AddHyperparameterDiffs(
		std::vector<TrainingSetupDiff>& diffs,
		const std::vector<TrainingSetupParameter>& savedParameters,
		const std::vector<TrainingSetupParameter>& currentParameters)
	{
		const std::map<ParameterKey, std::string> saved = BuildIdentityParameterMap(savedParameters);
		const std::map<ParameterKey, std::string> current = BuildIdentityParameterMap(currentParameters);

		std::set<ParameterKey> keys;
		for (const auto& entry : saved)
			keys.insert(entry.first);
		for (const auto& entry : current)
			keys.insert(entry.first);

		for (const ParameterKey& key : keys) {
			const auto savedIt = saved.find(key);
			const auto currentIt = current.find(key);
			const std::string savedValue = savedIt != saved.end() ? savedIt->second : "(missing)";
			const std::string currentValue = currentIt != current.end() ? currentIt->second : "(missing)";
			AddSetupDiff(diffs, key.first + " / " + key.second, savedValue, currentValue);
		}
	}
}

RewardComparisonTable TrainingSetupSnapshot::BuildRewardTable() const
{
	return RewardComparisonTable::Build(rewards);
}

void RLGC::SerializeFields(GGL::JsonArchive& ar, RLGC::RewardValueRange& range)
{
	ar.Field("min", range.minValue, 0.f);
	ar.Field("max", range.maxValue, 0.f);
}

const GGL::EnumNames<RLGC::RewardCategory> RocketForge::REWARD_CATEGORY_NAMES = { {
	{ RLGC::RewardCategory::EVENT,      "event" },
	{ RLGC::RewardCategory::CONTINUOUS, "continuous" },
} };

void RocketForge::TrainingSetupParameter::Serialize(GGL::JsonArchive& ar)
{
	ar.Field("group", group);
	ar.Field("name", name);
	ar.Field("value", value);
}

void RocketForge::TrainingSetupSnapshot::Serialize(GGL::JsonArchive& ar)
{
	// schema_version is a save-only forward-compatibility marker; loading ignores it.
	if (ar.IsSaving())
		(*ar.Raw())["schema_version"] = 1;

	ar.Field("agent", agentName);
	ar.Field("stage", stageName);
	ar.Field("version_id", versionId);
	ar.Field("saved_at", savedAt);
	ar.Field("engine_version", engineVersion, 0);
	ar.Field("engine_revision", engineRevision);
	ar.Field("agent_setup", agentSetup);

	if (ar.IsLoading() && agentSetup)
		ResolveAgentSimulation(*agentSetup, engineVersion);

	ar.Field("rewards", rewards);
	ar.Field("hyperparameters", hyperparameters);
}

bool RocketForge::TrainingSetupParameterAffectsIdentity(const TrainingSetupParameter& parameter)
{
	if (parameter.group == "Versioning") {
		if (parameter.name == "Version ID" || parameter.name == "Saved At" || parameter.name == "Run Name")
			return false;
	}

	if (parameter.group == "Metrics" &&
		(parameter.name == "Send Metrics" ||
			parameter.name == "Project Name" ||
			parameter.name == "Group Name" ||
			parameter.name == "Run Name" ||
			parameter.name == "Add Rewards To Metrics")) {
		return false;
	}

	if (parameter.group == "Skill Test")
		return false;

	return true;
}

std::vector<TrainingSetupDiff> RocketForge::DiffTrainingSetupSnapshots(
	const TrainingSetupSnapshot& saved,
	const TrainingSetupSnapshot& current)
{
	std::vector<TrainingSetupDiff> diffs;
	AddSetupDiff(diffs, "Agent", saved.agentName, current.agentName);
	AddSetupDiff(diffs, "Training Stage", saved.stageName, current.stageName);

	// A zero (unrecorded) saved engine is legacy provenance, not a setup change.
	if (saved.engineVersion != 0 && saved.engineVersion != current.engineVersion)
		AddSetupDiff(diffs, "Simulator", std::to_string(saved.engineVersion), std::to_string(current.engineVersion));

	AddSetupDiff(diffs, "Version ID", saved.versionId, current.versionId);

	if (saved.agentSetup.has_value() != current.agentSetup.has_value()) {
		AddSetupDiff(
			diffs,
			"Agent Setup",
			saved.agentSetup.has_value() ? "present" : "(missing)",
			current.agentSetup.has_value() ? "present" : "(missing)");
	} else if (saved.agentSetup.has_value()) {
		for (const AgentSetupDiff& diff : DiffAgentSetupSnapshots(saved.agentSetup.value(), current.agentSetup.value()))
			AddSetupDiff(diffs, "Agent Setup / " + diff.field, diff.savedValue, diff.currentValue);
	}

	AddRewardDiffs(diffs, saved.rewards, current.rewards);
	AddHyperparameterDiffs(diffs, saved.hyperparameters, current.hyperparameters);

	return diffs;
}

TrainingSetupSnapshot RocketForge::MakeTrainingSetupSnapshot(const TrainingStage* stage, const std::string& versionId)
{
	TrainingSetupSnapshot snapshot = {};
	if (!stage)
		return snapshot;

	snapshot.agentName = stage->agent ? stage->agent->agentName : std::string();
	snapshot.stageName = stage->GetStageName();
	snapshot.versionId = !versionId.empty() ? versionId : stage->learnerConfig.runtime.versionId;
	snapshot.savedAt = RocketForgeTraining::BuildStorage::MakeTimestamp();
	snapshot.engineVersion = static_cast<int>(stage->learnerConfig.engine);
	snapshot.engineRevision = RocketSim::GetEngineInfo(stage->learnerConfig.engine).revision;
	if (stage->agent)
		snapshot.agentSetup = CaptureAgentSetupSnapshot(*stage->agent);

	std::vector<RLGC::WeightedReward> rewards = stage->GetConfiguredRewards();
	snapshot.rewards.reserve(rewards.size());
	for (const RLGC::WeightedReward& weightedReward : rewards)
	{
		RewardTableEntry entry = {};
		entry.name = weightedReward.reward->GetName();
		entry.weight = weightedReward.weight;
		entry.range = weightedReward.reward->GetValueRange();
		entry.category = weightedReward.reward->GetCategory();

		const float absMax = std::abs(entry.range.maxValue);
		const float absMin = std::abs(entry.range.minValue);
		const float peak = (absMax >= absMin) ? entry.range.maxValue : entry.range.minValue;
		entry.effectiveMax = peak * weightedReward.weight;

		snapshot.rewards.push_back(std::move(entry));
		delete weightedReward.reward;
	}

	const GGL::LearnerConfig& config = stage->learnerConfig;
	const GGL::PPOLearnerConfig& ppo = config.ppo;
	const GGL::PPGLearnerConfig& ppg = config.ppg;

	AddParameter(snapshot, "Versioning", "Agent", snapshot.agentName);
	AddParameter(snapshot, "Versioning", "Stage", snapshot.stageName);
	AddParameter(snapshot, "Versioning", "Version ID", snapshot.versionId);
	AddParameter(snapshot, "Versioning", "Saved At", snapshot.savedAt);
	AddParameter(snapshot, "Versioning", "Run Name", stage->GetRunName());

	AddParameter(snapshot, "Environment", "Games", config.numGames);
	AddParameter(snapshot, "Environment", "Tick Skip", config.tickSkip);
	AddParameter(snapshot, "Environment", "Action Delay", config.actionDelay);
	AddParameter(snapshot, "Environment", "Team Sizes", FormatTeamSizes(stage->GetWeightedTeamSizes()));

	AddParameter(snapshot, "Training", "Algorithm", FormatAlgorithm(config.algorithm));
	AddParameter(snapshot, "Training", "Timesteps / Iteration", stage->GetTimestepsPerIteration());
	AddParameter(snapshot, "Training", "Batch Size", ppo.batchSize);
	AddParameter(snapshot, "Training", "Micro Batch Size", ppo.microBatchSize == 0 ? ppo.batchSize : ppo.microBatchSize);
	AddParameter(snapshot, "Training", "Epochs", ppo.epochs);
	AddParameter(snapshot, "Training", "Random Seed", config.randomSeed);
	AddParameter(snapshot, "Training", "Learning Rate", stage->GetLearningRate());
	AddParameter(snapshot, "Training", "Policy LR", ppo.policyLR);
	AddParameter(snapshot, "Training", "Critic LR", ppo.criticLR);
	AddParameter(snapshot, "Training", "Entropy Scale", ppo.entropyScale);
	AddParameter(snapshot, "Training", "GAE Gamma", ppo.gaeGamma);
	AddParameter(snapshot, "Training", "GAE Lambda", ppo.gaeLambda);
	AddParameter(snapshot, "Training", "Clip Range", ppo.clipRange);
	AddParameter(snapshot, "Training", "Reward Clip Range", ppo.rewardClipRange);
	AddParameter(snapshot, "Training", "Policy Temperature", ppo.policyTemperature);
	AddParameter(snapshot, "Training", "Min Action Probability", ppo.minActionProbability);
	AddParameter(snapshot, "Training", "Mask Entropy", ppo.maskEntropy);
	AddParameter(snapshot, "Training", "Deterministic", ppo.deterministic);
	AddParameter(snapshot, "Training", "Use Half Precision", ppo.useHalfPrecision);
	AddParameter(snapshot, "Training", "Overbatching", ppo.overbatching);
	AddParameter(snapshot, "Training", "Max Episode Duration", ppo.maxEpisodeDuration);
	AddScheduledFloatParameters(snapshot, "PPO/Entropy Scale", ppo.entropyScale);

	// Recorded only when non-default so agents on defaults keep their existing setup hash.
	if (ppo.normalizeAdvantages)
	{
		AddParameter(snapshot, "Training", "Normalize Advantages", ppo.normalizeAdvantages);
		AddParameter(snapshot, "Training", "Advantage Norm Epsilon", ppo.advantageNormEpsilon);
	}

	if (ppo.policyGradClipNorm != GGL::PPOLearnerConfig{}.policyGradClipNorm)
		AddParameter(snapshot, "Training", "Policy Grad Clip Norm", ppo.policyGradClipNorm);

	// Only recorded when the decomposed critic is on, so agents without it keep their setup hash.
	if (ppo.decomposedCritic.enabled)
	{
		AddParameter(snapshot, "Training", "Critic Head Loss Weighting",
			std::string(GGL::CRITIC_HEAD_LOSS_WEIGHTING_NAMES.NameOf(ppo.decomposedCritic.lossWeighting)));
		AddParameter(snapshot, "Training", "PopArt Beta", ppo.decomposedCritic.popArtBeta);
		AddParameter(snapshot, "Training", "PopArt Sigma Floor", ppo.decomposedCritic.popArtSigmaFloor);
	}

	if (config.algorithm == GGL::LearnerAlgorithm::PPG)
	{
		AddParameter(snapshot, "PPG", "Policy Epochs", ppg.policyEpochs);
		AddParameter(snapshot, "PPG", "Critic Epochs", ppg.criticEpochs);
		AddParameter(snapshot, "PPG", "Aux Phase Interval", ppg.auxPhaseInterval);
		AddParameter(snapshot, "PPG", "Aux Epochs", ppg.auxEpochs);
		AddParameter(snapshot, "PPG", "Aux Batch Size", ppg.auxBatchSize == 0 ? ppo.batchSize : ppg.auxBatchSize);
		AddParameter(snapshot, "PPG", "Aux Policy LR", ppg.auxPolicyLR);
		AddParameter(snapshot, "PPG", "Aux Critic LR", ppg.auxCriticLR);
		AddParameter(snapshot, "PPG", "Aux Value Loss Scale", ppg.auxValueLossScale);
		AddParameter(snapshot, "PPG", "Aux Policy Clone Loss Scale", ppg.auxPolicyCloneLossScale);
		AddParameter(snapshot, "PPG", "Aux True Critic Loss Scale", ppg.auxTrueCriticLossScale);
		AddParameter(snapshot, "PPG", "Aux Grad Clip", ppg.auxGradClip);
	}

	AddParameter(snapshot, "Saving", "Timesteps / Save", config.tsPerSave);
	AddParameter(snapshot, "Saving", "Timesteps / Version", config.tsPerVersion);
	AddParameter(snapshot, "Saving", "Checkpoints To Keep", config.checkpointsToKeep);
	AddParameter(snapshot, "Saving", "Save Policy Versions", config.savePolicyVersions);
	AddParameter(snapshot, "Saving", "Max Recent Versions", config.maxRecentVersions);
	AddParameter(snapshot, "Saving", "Train Against Recent Versions", config.trainAgainstRecentVersions);
	AddParameter(snapshot, "Saving", "Train Against Recent Percent", config.trainAgainstRecentPercent);
	AddParameter(snapshot, "Saving", "Recent Opponent Cohort Size", config.recentOpponentCohortSize);
	AddParameter(snapshot, "Saving", "Recent Opponent Rotation Timesteps", config.recentOpponentRotationTimesteps);
	AddParameter(snapshot, "Saving", "Train Against Static Opponents", config.trainAgainstStaticOpponents);
	AddParameter(snapshot, "Saving", "Train Against Static Percent", config.trainAgainstStaticPercent);
	AddParameter(snapshot, "Saving", "Static Opponent Cohort Size", config.staticOpponentCohortSize);
	AddParameter(snapshot, "Saving", "Static Opponent Rotation Timesteps", config.staticOpponentRotationTimesteps);

	// Recorded only when non-default so agents on defaults keep their existing setup hash.
	if (config.staticOpponentsUseOwnBallPrediction)
		AddParameter(snapshot, "Saving", "Static Opponents Use Own Ball Prediction", config.staticOpponentsUseOwnBallPrediction);

	AddParameter(snapshot, "Saving", "Reset Run ID On Load", config.runtime.resetRunIdOnLoad);
	AddParameter(snapshot, "Saving", "Warm Start Policy", config.warmStartPolicy);
	AddParameter(snapshot, "Saving", "Warm Start Iterations", config.warmStartIterations);
	AddParameter(snapshot, "Saving", "Warm Start Policy LR Multiplier", config.warmStartPolicyLRMult);

	const std::string staticBuildName = config.runtime.buildName.empty() ? GGL::DEFAULT_BUILD_NAME : config.runtime.buildName;
	const std::vector<RocketForgeTraining::StaticOpponentSelectionEntry> staticSelections =
		RocketForgeTraining::LoadStaticOpponentSelection(snapshot.agentName, staticBuildName);
	AddParameter(snapshot, "Saving", "Static Opponent Count", static_cast<int>(staticSelections.size()));

	if (!staticSelections.empty()) {
		std::ostringstream selectionSummary;
		for (size_t i = 0; i < staticSelections.size(); i++) {
			if (i > 0)
				selectionSummary << ", ";
			selectionSummary << staticSelections[i].displayName;
		}
		AddParameter(snapshot, "Saving", "Static Opponent Sources", selectionSummary.str());
	}

	AddParameter(snapshot, "Metrics", "Send Metrics", config.sendMetrics);
	AddParameter(snapshot, "Metrics", "Project Name", config.metricsProjectName);
	AddParameter(snapshot, "Metrics", "Group Name", config.metricsGroupName);
	AddParameter(snapshot, "Metrics", "Run Name", config.metricsRunName);
	AddParameter(snapshot, "Metrics", "Skill Tracker Enabled", config.skillTracker.enabled);
	AddParameter(snapshot, "Metrics", "Add Rewards To Metrics", config.addRewardsToMetrics);

	AddParameter(snapshot, "Skill Test", "Target Confidence", config.skillTracker.targetConfidence);
	AddParameter(snapshot, "Skill Test", "Reference Games", config.skillTracker.referenceGamesPerOpponent);
	AddParameter(snapshot, "Skill Test", "Reference Calibration Games", config.skillTracker.referenceCalibrationGames);
	AddParameter(snapshot, "Skill Test", "Rating Drift Per 100M", config.skillTracker.ratingDriftSigmaPer100MTimesteps);
	AddParameter(snapshot, "Skill Test", "Anchor Retire Win Prob", config.skillTracker.anchorRetireExpectedWinProb);

	AddParameter(snapshot, "Normalization", "Standardize Obs", config.standardizeObs);
	AddParameter(snapshot, "Normalization", "Standardize Returns", config.standardizeReturns);
	AddParameter(snapshot, "Normalization", "Min Obs STD", config.minObsSTD);
	AddParameter(snapshot, "Normalization", "Max Obs Mean Range", config.maxObsMeanRange);
	AddParameter(snapshot, "Normalization", "Max Obs Samples", config.maxObsSamples);
	AddParameter(snapshot, "Normalization", "Max Return Samples", config.maxReturnSamples);

	return snapshot;
}

std::optional<TrainingSetupSnapshot> RocketForge::LoadTrainingSetupSnapshot(const std::filesystem::path& path, std::string* error)
{
	using nlohmann::json;

	try
	{
		if (!std::filesystem::exists(path))
		{
			if (error)
				*error = "Training setup snapshot not found.";
			return std::nullopt;
		}

		std::ifstream in(path);
		if (!in.good())
		{
			if (error)
				*error = "Failed to open " + path.string();
			return std::nullopt;
		}

		json j = json::parse(in, nullptr, false);
		if (!j.is_object())
		{
			if (error)
				*error = "Invalid training setup snapshot.";
			return std::nullopt;
		}

		TrainingSetupSnapshot snapshot = {};
		const std::vector<std::string> loadErrors = GGL::JsonLoad(j, snapshot);
		if (HasEngineFieldError(loadErrors))
		{
			if (error)
			{
				*error = "Invalid simulator field in " + path.string() + ":";
				for (const std::string& loadError : loadErrors)
					*error += "\n" + loadError;
			}
			return std::nullopt;
		}

		// An agent setup without an obs builder or action parser is unusable, but JsonLoad only records a field-level error for it.
		// Drop just that optional rather than failing the whole load.
		if (snapshot.agentSetup.has_value() &&
			(snapshot.agentSetup->obsBuilderName.empty() || snapshot.agentSetup->actionParserName.empty()))
		{
			snapshot.agentSetup.reset();
		}

		return snapshot;
	}
	catch (const std::exception& e)
	{
		if (error)
			*error = e.what();
		return std::nullopt;
	}
}

std::optional<TrainingSetupSnapshot> RocketForge::LoadExistingTrainingSetupSnapshot(const std::filesystem::path& path)
{
	if (!std::filesystem::exists(path))
		return std::nullopt;

	std::string error;
	std::optional<TrainingSetupSnapshot> snapshot = LoadTrainingSetupSnapshot(path, &error);
	if (!snapshot)
		throw std::runtime_error("Cannot read saved training setup " + path.string() + ": " + error);

	return snapshot;
}

AgentSetupSnapshot RocketForge::ResolveSavedAgentSetup(const std::optional<TrainingSetupSnapshot>& saved, AgentSetupSnapshot current)
{
	if (saved && saved->agentSetup)
		return *saved->agentSetup;

	current.trainingEngineVersion = 0;
	current.ballPredictionEngineVersion = 0;
	ResolveAgentSimulation(current, saved ? saved->engineVersion : 0);
	return current;
}

TrainingSetupSnapshot RocketForgeTraining::CaptureTrainingSetupSnapshot(const TrainingStage* stage)
{
	return RocketForge::MakeTrainingSetupSnapshot(stage);
}

std::optional<TrainingSetupSnapshot> RocketForgeTraining::LoadTrainingSetupSnapshot(
	const TrainingStage* stage,
	const std::string& versionId)
{
	if (!stage || versionId.empty())
		return std::nullopt;

	const std::filesystem::path agentModelsFolder = ResolveAgentModelsFolder(stage->agent.get());
	BuildStorage probe(agentModelsFolder, GGL::DEFAULT_BUILD_NAME);
	std::vector<BuildSummary> builds = probe.ListBuilds();
	if (builds.empty())
		builds.push_back({ GGL::DEFAULT_BUILD_NAME, GGL::DEFAULT_BUILD_NAME, {}, {}, true });

	for (const BuildSummary& build : builds) {
		if (!stage->learnerConfig.runtime.buildName.empty() && build.buildName != stage->learnerConfig.runtime.buildName)
			continue;

		BuildStorage buildStorage(agentModelsFolder, build.buildName);
		std::optional<TrainingSegmentMetadata> segment = buildStorage.LoadSegmentById(versionId);
		if (!segment.has_value() || segment->stageName != stage->GetStageName())
			continue;

		return RocketForge::LoadTrainingSetupSnapshot(
			buildStorage.SegmentRoot(segment->folderName) / RocketForge::TrainingSetupSnapshot::FILE_NAME);
	}

	return std::nullopt;
}

bool RocketForgeTraining::HasTrainingSetupSnapshot(const TrainingStage* stage, const std::string& versionId)
{
	if (!stage || versionId.empty())
		return false;

	const std::filesystem::path agentModelsFolder = ResolveAgentModelsFolder(stage->agent.get());
	BuildStorage probe(agentModelsFolder, GGL::DEFAULT_BUILD_NAME);
	std::vector<BuildSummary> builds = probe.ListBuilds();
	if (builds.empty())
		builds.push_back({ GGL::DEFAULT_BUILD_NAME, GGL::DEFAULT_BUILD_NAME, {}, {}, true });

	for (const BuildSummary& build : builds) {
		if (!stage->learnerConfig.runtime.buildName.empty() && build.buildName != stage->learnerConfig.runtime.buildName)
			continue;

		BuildStorage buildStorage(agentModelsFolder, build.buildName);
		std::optional<TrainingSegmentMetadata> segment = buildStorage.LoadSegmentById(versionId);
		if (!segment.has_value() || segment->stageName != stage->GetStageName())
			continue;

		return std::filesystem::exists(buildStorage.SegmentRoot(segment->folderName) / RocketForge::TrainingSetupSnapshot::FILE_NAME);
	}

	return false;
}
