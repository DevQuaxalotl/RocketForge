#include "StageQueueRunner.h"
#include <GigaLearnInfer/AgentRegistry.h>
#include <RocketSim/Engine.h>
#include "StageQueueSignals.h"
#include "ForgeTrainingInternal.h"
#include "AgentModelPaths.h"
#include "Storage/BuildStorage.h"
#include "Storage/TrainingBuildLock.h"
#include "StaticOpponents/StaticOpponentSelection.h"
#include "TrainingSetup/TrainingSetupSnapshot.h"
#include "TrainingSetup/VersionManagement.h"
#include "TrainingStage.h"

#include <GigaLearnCPP/TrainingStorage.h>
#include <GigaLearnCPP/SegmentVersioning.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <vector>

namespace fs = std::filesystem;

namespace RocketForgeTraining {

	namespace {
		std::string BuildNameOrDefault(const ForgeSettings& settings)
		{
			return settings.buildName.empty() ? GGL::DEFAULT_BUILD_NAME : settings.buildName;
		}

		std::string NormalizeBranchName(const std::string& branchName)
		{
			const size_t first = branchName.find_first_not_of(" \t\r\n");
			if (first == std::string::npos)
				return "main";

			const size_t last = branchName.find_last_not_of(" \t\r\n");
			return branchName.substr(first, last - first + 1);
		}

		std::string SelectedBootstrapSegmentId(const StageSettings& entry)
		{
			if (entry.bootstrap.mode == BootstrapConfig::Mode::ContinueExisting ||
				entry.bootstrap.mode == BootstrapConfig::Mode::ContinueLatest) {
				return entry.bootstrap.selectedExistingVersionId;
			}

			if (entry.bootstrap.mode == BootstrapConfig::Mode::LoadFromOtherStage)
				return entry.bootstrap.sourceVersionId;

			return {};
		}

		bool BranchModeStartsNewPath(BootstrapConfig::BranchMode mode)
		{
			return mode == BootstrapConfig::BranchMode::CreateNew ||
				mode == BootstrapConfig::BranchMode::ContinueSelected;
		}

		bool SegmentNeverCheckpointed(const CheckpointInfo& info)
		{
			return info.segmentExists && !info.hasCheckpoints;
		}

		// A run that failed before its first checkpoint leaves a segment with no model to load
		// or branch from. Retrain it in place from the segment it was originally created from;
		// the bootstrap fields can't be trusted for that, since they may name the empty segment itself.
		StageAction RetrainSegmentInPlace(const std::string& segmentId, const CheckpointInfo& info)
		{
			StageAction action;
			action.kind = ActionKind::StartNew;
			action.startNewReuseSegmentId = segmentId;
			action.startNewSourceSegmentId = info.parentSegmentId;
			return action;
		}

		std::string MakeWandbGroupName(const std::string& buildName)
		{
			return buildName;
		}

		std::string MakeWandbRunName(
			const std::string& wandbGroup,
			const std::string& segmentId,
			const std::string& stageName)
		{
			return wandbGroup + " - " + segmentId + " - " + stageName;
		}

		bool UpdateWandbNames(const ForgeSettings& settings, TrainingSegmentMetadata& segment)
		{
			const std::string oldProject = segment.wandbProject;
			const std::string oldGroup = segment.wandbGroup;
			const std::string oldRunName = segment.wandbRunName;
			const std::string buildName = segment.buildName.empty()
				? BuildNameOrDefault(settings)
				: segment.buildName;

			segment.wandbProject = settings.agentName;
			segment.wandbGroup = MakeWandbGroupName(buildName);
			segment.wandbRunName = MakeWandbRunName(segment.wandbGroup, segment.segmentId, segment.stageName);

			return oldProject != segment.wandbProject ||
				oldGroup != segment.wandbGroup ||
				oldRunName != segment.wandbRunName;
		}

		fs::path LatestCheckpointPath(const BuildStorage& storage, const TrainingSegmentMetadata& segment)
		{
			return GGL::TrainingStorage::GetMostRecentCheckpointIn(storage.SegmentCheckpointsRoot(segment.folderName));
		}

		std::optional<TrainingSegmentMetadata> LatestBuildSegment(const BuildStorage& storage)
		{
			BuildMetadata build = storage.LoadBuild();
			if (build.activeSegmentId.empty())
				return std::nullopt;

			return storage.LoadSegmentById(build.activeSegmentId);
		}

		// Shared by the real run and its dry-walk resolvers so they all decide from the same first-stage target.
		void PrepareFirstStageContinueLatest(ForgeSettings& settings, const BuildStorage& storage)
		{
			if (settings.stages.empty() || settings.stages.front().bootstrap.mode != BootstrapConfig::Mode::ContinueLatest)
				return;

			std::optional<TrainingSegmentMetadata> latest = LatestBuildSegment(storage);
			settings.stages.front().bootstrap.selectedExistingVersionId = latest.has_value() ? latest->segmentId : std::string();
			if (latest.has_value() && settings.stages.front().bootstrap.branchMode == BootstrapConfig::BranchMode::ContinueExisting)
				settings.stages.front().branchName = NormalizeBranchName(latest->branchName);
		}

		uint64_t ParseCheckpointTimesteps(const fs::path& checkpoint)
		{
			const std::string name = checkpoint.filename().string();
			if (!GGL::TrainingStorage::IsAllDigits(name))
				return 0;
			return static_cast<uint64_t>(std::stoull(name));
		}

		fs::path PreferredSegmentResumeRoot(const BuildStorage& storage, const TrainingSegmentMetadata& segment)
		{
			const std::optional<ResolvedCheckpoint> latest = storage.LatestSegmentCheckpoint(segment.folderName);
			if (!latest.has_value())
				return storage.SegmentCheckpointsRoot(segment.folderName);

			if (latest->isAutosave)
				RG_LOG("Resuming " << segment.segmentId << " from autosave " << latest->folder);

			return latest->root;
		}

		std::vector<std::string> RecentPolicyVersionSourceIds(
			const BuildStorage& storage,
			const TrainingSegmentMetadata& segment)
		{
			std::vector<std::string> sourceIds = {};
			std::set<std::string> seen = {};
			TrainingSegmentMetadata current = segment;

			while (!current.segmentId.empty() && !seen.contains(current.segmentId)) {
				sourceIds.push_back(current.segmentId);
				seen.insert(current.segmentId);

				if (current.parentSegmentId.empty() || seen.contains(current.parentSegmentId))
					break;

				std::optional<TrainingSegmentMetadata> parent = storage.LoadSegmentById(current.parentSegmentId);
				if (!parent.has_value()) {
					sourceIds.push_back(current.parentSegmentId);
					break;
				}
				current = parent.value();
			}

			return sourceIds;
		}

		void CleanupSegmentTempCheckpoints(const BuildStorage& storage, const TrainingSegmentMetadata& segment)
		{
			const fs::path tempRoot = storage.SegmentTempCheckpointsRoot(segment.folderName);
			if (tempRoot.empty() || !fs::exists(tempRoot))
				return;

			if (GGL::TrainingStorage::FindNumberedDirs(storage.SegmentCheckpointsRoot(segment.folderName)).empty())
				return;

			try {
				fs::remove_all(tempRoot);
			} catch (const std::exception& e) {
				RG_LOG("Warning: failed to remove segment temp checkpoints " << tempRoot << ": " << e.what());
			}
		}

		void ConfigureStageIdentityForSegmentSnapshot(
			const BuildStorage& storage,
			const ForgeSettings& settings,
			RocketForge::TrainingStage* stage,
			const std::string& segmentId,
			const std::string& stageName)
		{
			GGL::LearnerRuntimeConfig& runtimeConfig = stage->learnerConfig.runtime;
			runtimeConfig.buildName = BuildNameOrDefault(settings);
			runtimeConfig.segmentId = segmentId;
			runtimeConfig.trainingStageName = stageName;
			runtimeConfig.staticOpponentsRegistryPath = storage.StaticOpponentsPath();
			runtimeConfig.skillReferencesRegistryPath = storage.SkillReferencesPath();
			stage->ApplyParameterOverrides();
		}

		void SaveSegmentTrainingSetupSnapshot(
			const BuildStorage& storage,
			const ForgeSettings& settings,
			RocketForge::TrainingStage* stage,
			const TrainingSegmentMetadata& segment)
		{
			ConfigureStageIdentityForSegmentSnapshot(storage, settings, stage, segment.segmentId, segment.stageName);
			RocketForge::TrainingSetupSnapshot snapshot = RocketForge::MakeTrainingSetupSnapshot(stage, segment.segmentId);
			std::string error = {};
			if (!GGL::JsonSaveToFile(
				snapshot,
				storage.SegmentRoot(segment.folderName) / RocketForge::TrainingSetupSnapshot::FILE_NAME,
				&error)) {
				RG_ERR_CLOSE("Failed to save training setup snapshot for " << segment.segmentId << ": " << error);
			}
		}
	}

	CheckpointInfo InspectSegmentFromDisk(const std::string& agentName, const std::string& buildName, const std::string& segmentId)
	{
		CheckpointInfo info;
		if (segmentId.empty())
			return info;

		BuildStorage storage(ResolveAgentModelsFolder(agentName), buildName);
		std::optional<TrainingSegmentMetadata> segment = storage.LoadSegmentById(segmentId);
		info.segmentExists = segment.has_value();
		if (!segment.has_value())
			return info;
		info.stageName = segment->stageName;
		info.parentSegmentId = segment->parentSegmentId;

		std::set<int64_t> checkpoints = GGL::TrainingStorage::FindNumberedDirs(storage.SegmentCheckpointsRoot(segment->folderName));
		info.hasCheckpoints = !checkpoints.empty();
		if (info.hasCheckpoints)
			info.latestTimesteps = static_cast<uint64_t>(*checkpoints.rbegin());

		return info;
	}

	StageAction DecideAction(const ForgeSettings& settings, size_t i, const SegmentInspector& inspect)
	{
		const StageSettings& entry = settings.stages[i];
		StageAction action;

		if (!entry.createdSegmentId.empty()) {
			CheckpointInfo info = inspect(settings.agentName, entry.stageName, entry.createdSegmentId);
			const bool userExplicitlyChoseNew = (i == 0) && entry.bootstrap.mode == BootstrapConfig::Mode::New;
			const std::string selectedBootstrapSegmentId = SelectedBootstrapSegmentId(entry);
			const bool userPickedDifferentExistingSegment = (i == 0) &&
				!selectedBootstrapSegmentId.empty() &&
				selectedBootstrapSegmentId != entry.createdSegmentId;
			std::string overrideVersionId;
			if (entry.versionOverride.has_value())
				overrideVersionId = GGL::MakeSegmentVersionId(entry.versionOverride->major, entry.versionOverride->minor);
			const bool userRequestedNewRevisionFromSegment = (i == 0) &&
				!selectedBootstrapSegmentId.empty() &&
				((!overrideVersionId.empty() && overrideVersionId != entry.createdSegmentId) ||
				 BranchModeStartsNewPath(entry.bootstrap.branchMode) ||
				 entry.versionIncrement != VersionIncrement::None);
			const bool userRequestedNewRevisionFromLaterSegment =
				(i > 0) &&
				(BranchModeStartsNewPath(entry.bootstrap.branchMode) ||
				 (!overrideVersionId.empty() && overrideVersionId != entry.createdSegmentId) ||
				 entry.versionIncrement != VersionIncrement::None);
			const bool bypassCreatedSegment =
				userExplicitlyChoseNew ||
				userPickedDifferentExistingSegment ||
				userRequestedNewRevisionFromSegment ||
				userRequestedNewRevisionFromLaterSegment;

			if (!bypassCreatedSegment && info.segmentExists && info.hasCheckpoints && info.latestTimesteps >= entry.targetTotalTimesteps) {
				action.kind = ActionKind::Skip;
				action.skipReason = "Segment already at or past target.";
				action.skipSegmentId = entry.createdSegmentId;
				return action;
			}

			if (SegmentNeverCheckpointed(info) && !bypassCreatedSegment) {
				action = RetrainSegmentInPlace(entry.createdSegmentId, info);
				if (i > 0)
					action.startNewSourceSegmentId = settings.stages[i-1].createdSegmentId;
				return action;
			}

			if (info.segmentExists && !bypassCreatedSegment) {
				action.kind = ActionKind::Continue;
				action.continueSegmentId = entry.createdSegmentId;
				return action;
			}
		}

		// 1) First-stage bootstrap.
		if (i == 0) {
			switch (entry.bootstrap.mode) {
				case BootstrapConfig::Mode::New:
					action.kind = ActionKind::StartNew;
					return action;

				case BootstrapConfig::Mode::ContinueLatest:
				case BootstrapConfig::Mode::ContinueExisting: {
					// If no version was selected (e.g. CLI fresh-agent path), treat as StartNew.
					if (entry.bootstrap.selectedExistingVersionId.empty()) {
						action.kind = ActionKind::StartNew;
						return action;
					}

					CheckpointInfo info = inspect(settings.agentName, entry.stageName, entry.bootstrap.selectedExistingVersionId);
					if (BranchModeStartsNewPath(entry.bootstrap.branchMode) ||
						entry.versionOverride.has_value() ||
						entry.versionIncrement != VersionIncrement::None) {
						if (SegmentNeverCheckpointed(info))
							return RetrainSegmentInPlace(entry.bootstrap.selectedExistingVersionId, info);

						action.kind = ActionKind::StartNew;
						action.startNewSourceStage     = entry.stageName;
						action.startNewSourceSegmentId = entry.bootstrap.selectedExistingVersionId;
						return action;
					}

					if (info.hasCheckpoints && info.latestTimesteps >= entry.targetTotalTimesteps) {
						action.kind = ActionKind::Skip;
						action.skipReason = "Selected existing version already at target.";
						action.skipSegmentId = entry.bootstrap.selectedExistingVersionId;
						return action;
					}

					if (SegmentNeverCheckpointed(info))
						return RetrainSegmentInPlace(entry.bootstrap.selectedExistingVersionId, info);

					action.kind = ActionKind::Continue;
					action.continueSegmentId = entry.bootstrap.selectedExistingVersionId;
					return action;
				}

				case BootstrapConfig::Mode::LoadFromOtherStage:
					if (entry.bootstrap.sourceVersionId.empty()) {
						action.kind = ActionKind::StartNew;
						return action;
					}

					if (BranchModeStartsNewPath(entry.bootstrap.branchMode) ||
						entry.versionOverride.has_value() ||
						entry.versionIncrement != VersionIncrement::None) {
						const CheckpointInfo info = inspect(settings.agentName, entry.bootstrap.sourceStage, entry.bootstrap.sourceVersionId);
						if (SegmentNeverCheckpointed(info))
							return RetrainSegmentInPlace(entry.bootstrap.sourceVersionId, info);

						action.kind = ActionKind::StartNew;
						action.startNewSourceStage     = entry.bootstrap.sourceStage;
						action.startNewSourceSegmentId = entry.bootstrap.sourceVersionId;
						return action;
					}

					{
						CheckpointInfo info = inspect(settings.agentName, entry.bootstrap.sourceStage, entry.bootstrap.sourceVersionId);
						if (info.hasCheckpoints && info.latestTimesteps >= entry.targetTotalTimesteps) {
							action.kind = ActionKind::Skip;
							action.skipReason = "Selected existing segment already at target.";
							action.skipSegmentId = entry.bootstrap.sourceVersionId;
							return action;
						}

						if (SegmentNeverCheckpointed(info))
							return RetrainSegmentInPlace(entry.bootstrap.sourceVersionId, info);
					}

					action.kind = ActionKind::Continue;
					action.continueSegmentId = entry.bootstrap.sourceVersionId;
					return action;
			}
		}

		// 2) Subsequent stage with no created segment yet -> StartNew from previous stage.
		const StageSettings& prev = settings.stages[i - 1];
		if (!prev.createdSegmentId.empty()) {
			CheckpointInfo prevInfo = inspect(settings.agentName, prev.stageName, prev.createdSegmentId);
			if (prevInfo.hasCheckpoints && prevInfo.latestTimesteps >= entry.targetTotalTimesteps) {
				action.kind = ActionKind::Skip;
				action.skipReason = "Previous segment already at or past target.";
				action.skipSegmentId = prev.createdSegmentId;
				return action;
			}
		}

		action.kind = ActionKind::StartNew;
		action.startNewSourceStage     = prev.stageName;
		action.startNewSourceSegmentId = prev.createdSegmentId;
		return action;
	}

	bool PrepareQueueLineage(ForgeSettings& settings, const SegmentInspector& inspect)
	{
		// Reconcile the persisted queue with the segment chain that will actually run.
		// Once an entry trains or switches outputs, every descendant must be recreated.

		bool mutated = false;
		bool invalidateFollowing = false;
		std::string previousOutputSegmentId;

		for (size_t i = 0; i < settings.stages.size(); ++i) {
			StageSettings& entry = settings.stages[i];
			if (invalidateFollowing) {
				if (!entry.createdSegmentId.empty()) {
					entry.createdSegmentId.clear();
					mutated = true;
				}
				continue;
			}

			// A completed queue entry either aliases its predecessor (because its target
			// was already reached) or owns a child segment of that predecessor. Anything
			// else is stale after a stage edit, reorder, or external settings change.
			if (i > 0 && !entry.createdSegmentId.empty() &&
				entry.createdSegmentId != previousOutputSegmentId) {
				const CheckpointInfo info = inspect(settings.agentName, entry.stageName, entry.createdSegmentId);
				const bool stageChanged = info.segmentExists &&
					!info.stageName.empty() && info.stageName != entry.stageName;
				const bool parentChanged = info.segmentExists &&
					info.parentSegmentId != previousOutputSegmentId;
				if (!info.segmentExists || stageChanged || parentChanged) {
					entry.createdSegmentId.clear();
					mutated = true;
					invalidateFollowing = true;
					continue;
				}
			}

			const std::string originalSegmentId = entry.createdSegmentId;
			const StageAction action = DecideAction(settings, i, inspect);
			if (action.kind == ActionKind::Skip)
				previousOutputSegmentId = action.skipSegmentId;
			else if (action.kind == ActionKind::Continue)
				previousOutputSegmentId = action.continueSegmentId;
			else
				previousOutputSegmentId = action.startNewReuseSegmentId;

			// Continuing or creating changes the weights that every later entry must
			// bootstrap from. A skip is stable only when it resolves to the same output ID.
			if (action.kind != ActionKind::Skip || previousOutputSegmentId != originalSegmentId)
				invalidateFollowing = true;
		}

		return mutated;
	}

	void FlipBootstrapToContinueExisting(StageSettings& entry, const std::string& segmentId)
	{
		// Keep createdSegmentId and the selected resume target in lockstep. Downstream
		// entries use createdSegmentId as their bootstrap source, so updating only the
		// visible selection silently leaves them on the old lineage.
		entry.createdSegmentId = segmentId;
		entry.bootstrap.mode = BootstrapConfig::Mode::ContinueExisting;
		entry.bootstrap.branchMode = BootstrapConfig::BranchMode::ContinueExisting;
		entry.bootstrap.selectedExistingVersionId = segmentId;
		entry.versionIncrement = VersionIncrement::None;
		entry.versionOverride.reset();

		// Source-stage fields no longer apply once we've baked their effect into the
		// new segment's checkpoint; clear them to keep the JSON tidy.
		entry.bootstrap.sourceStage.clear();
		entry.bootstrap.sourceVersionId.clear();
	}

	namespace {
		bool IsRevisionAfterOrEqual(const GGL::SegmentVersionNumber& a, const GGL::SegmentVersionNumber& b)
		{
			return a.major > b.major || (a.major == b.major && a.minor >= b.minor);
		}

		std::optional<GGL::SegmentVersionNumber> FindLatestSegmentRevision(const BuildStorage& storage)
		{
			std::optional<GGL::SegmentVersionNumber> latest = {};
			for (const TrainingSegmentMetadata& segment : storage.ListSegments()) {
				std::optional<GGL::SegmentVersionNumber> parsed = BuildStorage::TryParseSegmentRevision(segment.segmentId);
				if (!parsed.has_value())
					continue;

				if (!latest.has_value() || IsRevisionAfterOrEqual(parsed.value(), latest.value()))
					latest = parsed.value();
			}
			return latest;
		}

		std::optional<GGL::SegmentVersionNumber> FindLatestSegmentRevisionInMajor(const BuildStorage& storage, int major)
		{
			std::optional<GGL::SegmentVersionNumber> latest = {};
			for (const TrainingSegmentMetadata& segment : storage.ListSegments()) {
				std::optional<GGL::SegmentVersionNumber> parsed = BuildStorage::TryParseSegmentRevision(segment.segmentId);
				if (!parsed.has_value() || parsed->major != major)
					continue;

				if (!latest.has_value() || IsRevisionAfterOrEqual(parsed.value(), latest.value()))
					latest = parsed.value();
			}
			return latest;
		}

		std::string MakeSegmentRevisionId(const GGL::SegmentVersionNumber& version)
		{
			return BuildStorage::MakeSegmentId(version.major, version.minor);
		}

		std::optional<GGL::SegmentVersionNumber> FindMinorIncrementBaseRevision(
			const BuildStorage& storage,
			const std::string& parentSegmentId)
		{
			std::optional<GGL::SegmentVersionNumber> parent = BuildStorage::TryParseSegmentRevision(parentSegmentId);
			if (!parent.has_value())
				return FindLatestSegmentRevision(storage);

			std::optional<GGL::SegmentVersionNumber> latestInParentMajor =
				FindLatestSegmentRevisionInMajor(storage, parent->major);
			if (latestInParentMajor.has_value() &&
				IsRevisionAfterOrEqual(latestInParentMajor.value(), parent.value())) {
				return latestInParentMajor;
			}

			return parent;
		}
	}

	std::string ResolveNewSegmentId(
		const BuildStorage& storage,
		const StageSettings& entry,
		const std::string& reuseSegmentId,
		const std::string& parentSegmentId)
	{
		if (!reuseSegmentId.empty())
			return reuseSegmentId;

		if (entry.versionOverride.has_value())
			return MakeSegmentRevisionId(entry.versionOverride.value());

		if (entry.versionIncrement == VersionIncrement::Major) {
			std::optional<GGL::SegmentVersionNumber> latest = FindLatestSegmentRevision(storage);
			if (!latest.has_value())
				return BuildStorage::MakeSegmentId(1, 0);

			GGL::SegmentVersionNumber next = latest.value();
			next.major += 1;
			next.minor = 0;
			return MakeSegmentRevisionId(next);
		}

		std::optional<GGL::SegmentVersionNumber> base = FindMinorIncrementBaseRevision(storage, parentSegmentId);
		if (!base.has_value())
			return BuildStorage::MakeSegmentId(1, 0);

		GGL::SegmentVersionNumber next = base.value();
		next.minor += 1;
		return MakeSegmentRevisionId(next);
	}

	std::vector<QueuePlanEntry> PlanQueue(const ForgeSettings& settingsIn, const BuildStorage& storage, const SegmentInspector& inspect)
	{
		std::vector<QueuePlanEntry> plan;

		// Operate on a copy: planning must never mutate the caller's settings.
		ForgeSettings settings = settingsIn;
		PrepareFirstStageContinueLatest(settings, storage);
		PrepareQueueLineage(settings, inspect);

		for (size_t i = 0; i < settings.stages.size(); ++i) {
			StageSettings& entry = settings.stages[i];
			const StageAction action = DecideAction(settings, i, inspect);

			QueuePlanEntry planEntry;
			planEntry.stageIndex = i;
			planEntry.stageName = entry.stageName;
			planEntry.kind = action.kind;
			planEntry.targetTotalTimesteps = entry.targetTotalTimesteps;

			switch (action.kind) {
				case ActionKind::Skip:
					planEntry.segmentId = action.skipSegmentId;
					planEntry.skipReason = action.skipReason;
					break;
				case ActionKind::Continue:
					planEntry.segmentId = action.continueSegmentId;
					break;
				case ActionKind::StartNew:
					planEntry.sourceStage = action.startNewSourceStage;
					planEntry.sourceSegmentId = action.startNewSourceSegmentId;
					planEntry.segmentId = ResolveNewSegmentId(storage, entry, action.startNewReuseSegmentId, action.startNewSourceSegmentId);
					break;
			}

			if (action.kind != ActionKind::StartNew && !planEntry.segmentId.empty())
				planEntry.currentTimesteps = inspect(settings.agentName, entry.stageName, planEntry.segmentId).latestTimesteps;

			// Later entries bootstrap from this entry's output, as a real run records once the stage finishes.
			if (!planEntry.segmentId.empty())
				FlipBootstrapToContinueExisting(entry, planEntry.segmentId);

			plan.push_back(std::move(planEntry));
		}

		return plan;
	}

	std::vector<QueuePlanEntry> PlanQueue(const ForgeSettings& settings)
	{
		const std::string buildName = BuildNameOrDefault(settings);
		BuildStorage storage(ResolveAgentModelsFolder(settings.agentName), buildName);
		SegmentInspector inspect = [&](const std::string& agentName, const std::string&, const std::string& id) {
			return InspectSegmentFromDisk(agentName, buildName, id);
		};
		return PlanQueue(settings, storage, inspect);
	}

	QueueStart ResolveQueueStart(const ForgeSettings& settings, const BuildStorage& storage, const SegmentInspector& inspect)
	{
		QueueStart result;

		for (const QueuePlanEntry& planEntry : PlanQueue(settings, storage, inspect)) {
			if (planEntry.kind == ActionKind::Skip)
				continue;

			result.resolved = true;
			result.stageName = planEntry.stageName;
			result.segmentId = planEntry.segmentId;
			result.isNew = planEntry.kind == ActionKind::StartNew;
			return result;
		}

		return result;
	}

	QueueStart ResolveQueueStart(const ForgeSettings& settings)
	{
		const std::string buildName = BuildNameOrDefault(settings);
		BuildStorage storage(ResolveAgentModelsFolder(settings.agentName), buildName);
		SegmentInspector inspect = [&](const std::string& agentName, const std::string&, const std::string& id) {
			return InspectSegmentFromDisk(agentName, buildName, id);
		};
		return ResolveQueueStart(settings, storage, inspect);
	}

	std::vector<std::string> ResolveQueueSegmentIds(const ForgeSettings& settings, const BuildStorage& storage, const SegmentInspector& inspect)
	{
		std::vector<std::string> ids = {};
		std::set<std::string> seen = {};
		auto addId = [&](const std::string& id) {
			if (!id.empty() && seen.insert(id).second)
				ids.push_back(id);
		};

		for (const StageSettings& entry : settings.stages) {
			addId(entry.createdSegmentId);
			addId(entry.bootstrap.selectedExistingVersionId);
			addId(entry.bootstrap.sourceVersionId);
		}

		for (const QueuePlanEntry& planEntry : PlanQueue(settings, storage, inspect)) {
			addId(planEntry.sourceSegmentId);
			addId(planEntry.segmentId);
		}

		return ids;
	}

	namespace {
		void UpdateSegmentSetupMetadata(
			BuildStorage& storage,
			const ForgeSettings& settings,
			RocketForge::TrainingStage* stage,
			const StageSettings& entry,
			TrainingSegmentMetadata& segment,
			const std::string& setupHash)
		{
			segment.buildName = BuildNameOrDefault(settings);
			segment.branchName = NormalizeBranchName(entry.branchName);
			segment.stageName = entry.stageName;
			segment.setupHash = setupHash;
			segment.updatedAt = BuildStorage::MakeTimestamp();
			segment.notes = entry.versionNotes;
			UpdateWandbNames(settings, segment);
			storage.SaveSegment(segment);
			SaveSegmentTrainingSetupSnapshot(storage, settings, stage, segment);
		}

		TrainingSegmentMetadata CreateSegmentMetadata(
			BuildStorage& storage,
			const ForgeSettings& settings,
			RocketForge::TrainingStage* stage,
			const StageSettings& entry,
			const std::string& reuseSegmentId,
			const std::string& parentSegmentId,
			const fs::path& sourceCheckpoint)
		{
			const std::string timestamp = BuildStorage::MakeTimestamp();
			const std::string buildName = BuildNameOrDefault(settings);
			const std::string segmentId = ResolveNewSegmentId(storage, entry, reuseSegmentId, parentSegmentId);

			stage->ApplySegmentId(segmentId, entry.versionNotes);
			ConfigureStageIdentityForSegmentSnapshot(storage, settings, stage, segmentId, entry.stageName);
			RocketForge::TrainingSetupSnapshot snapshot = RocketForge::MakeTrainingSetupSnapshot(stage, segmentId);

			TrainingSegmentMetadata segment = {};
			segment.segmentId = segmentId;
			segment.folderName = storage.MakeSegmentFolderName(segmentId, entry.stageName);
			segment.buildName = buildName;
			segment.parentSegmentId = parentSegmentId;
			segment.branchName = NormalizeBranchName(entry.branchName);
			segment.status = "active";
			segment.stageName = entry.stageName;
			segment.setupHash = BuildStorage::MakeSetupHash(snapshot);
			segment.createdAt = timestamp;
			segment.updatedAt = timestamp;
			segment.startTimestep = ParseCheckpointTimesteps(sourceCheckpoint);
			segment.sourceCheckpointPath = sourceCheckpoint.empty() ? std::string() : storage.MakeRelativeToBuild(sourceCheckpoint);
			UpdateWandbNames(settings, segment);
			segment.notes = entry.versionNotes;
			return segment;
		}

		void SaveBuildAfterSegment(BuildStorage& storage, const TrainingSegmentMetadata& segment)
		{
			BuildMetadata build = storage.LoadBuild();
			build.activeSegmentId = segment.segmentId;
			build.activeTimesteps = segment.endTimestep;
			build.totalTrainedTimesteps = 0;

			for (const TrainingSegmentMetadata& existing : storage.ListSegments())
				build.totalTrainedTimesteps += existing.endTimestep >= existing.startTimestep ? existing.endTimestep - existing.startTimestep : 0;

			build.updatedAt = BuildStorage::MakeTimestamp();
			storage.SaveBuild(build);
		}

		void RefreshSegmentCheckpointMetadata(
			BuildStorage& storage,
			const std::string& segmentId,
			const std::string& folderName,
			uint64_t timesteps,
			const fs::path& checkpointFolder)
		{
			if (storage.RecordSavedCheckpoint(segmentId, folderName, timesteps, checkpointFolder))
				return;

			RG_LOG("Reconciling checkpoint metadata for segment " << segmentId);
			std::optional<TrainingSegmentMetadata> segment = storage.LoadSegmentById(segmentId);
			if (!segment.has_value()) {
				RG_LOG("Warning: checkpoint save completed for missing segment " << segmentId);
				return;
			}

			storage.RefreshSegmentCheckpoints(segment.value(), true);
			storage.SaveSegment(segment.value());
			SaveBuildAfterSegment(storage, segment.value());
		}

		void ConfigureStageForSegment(
			BuildStorage& storage,
			const ForgeSettings& settings,
			RocketForge::TrainingStage* stage,
			TrainingSegmentMetadata& segment,
			const fs::path& checkpointLoadFolder)
		{
			const fs::path checkpointFolder = storage.SegmentCheckpointsRoot(segment.folderName);
			const fs::path tempCheckpointFolder = storage.SegmentTempCheckpointsRoot(segment.folderName);

			if (checkpointLoadFolder.empty()) {
				if (const std::optional<fs::path> promoted = storage.PromoteNewestAutosave(segment))
					RG_LOG("Promoted autosave " << promoted->filename() << " of " << segment.segmentId << " before resuming");
			}

			stage->ApplySegmentId(segment.segmentId, segment.notes);
			GGL::LearnerRuntimeConfig& runtimeConfig = stage->learnerConfig.runtime;
			runtimeConfig.buildName = BuildNameOrDefault(settings);
			runtimeConfig.segmentId = segment.segmentId;
			runtimeConfig.trainingStageName = segment.stageName;
			runtimeConfig.checkpointFolder = tempCheckpointFolder;
			runtimeConfig.checkpointSnapshotFolder = checkpointFolder;
			runtimeConfig.persistCheckpointSnapshotsDuringTraining = settings.globals.persistSegmentSnapshotsDuringTraining;
			runtimeConfig.checkpointLoadFolder = checkpointLoadFolder.empty()
				? PreferredSegmentResumeRoot(storage, segment)
				: checkpointLoadFolder;
			runtimeConfig.recentPolicyVersionsFolder = storage.RecentPolicyVersionsRoot();
			runtimeConfig.recentPolicyVersionSourceIds = RecentPolicyVersionSourceIds(storage, segment);
			runtimeConfig.skillBenchmarkFolder = storage.SkillBenchmarksRoot();
			runtimeConfig.staticOpponentsRegistryPath = storage.StaticOpponentsPath();
			runtimeConfig.skillReferencesRegistryPath = storage.SkillReferencesPath();
			stage->learnerConfig.metricsProjectName = settings.agentName;
			stage->learnerConfig.metricsGroupName = segment.wandbGroup;
			stage->learnerConfig.metricsRunName = MakeWandbRunName(segment.wandbGroup, segment.segmentId, segment.stageName);
			runtimeConfig.onCheckpointSaved = [&storage, stage, segmentId = segment.segmentId, folderName = segment.folderName](uint64_t timesteps, const fs::path& checkpointFolder) {
				try {
					RefreshSegmentCheckpointMetadata(storage, segmentId, folderName, timesteps, checkpointFolder);
					RefreshRLBotExportAfterCheckpoint(stage, segmentId, timesteps, checkpointFolder, folderName);
				} catch (const std::exception& e) {
					RG_LOG("Warning: failed to refresh segment metadata after checkpoint save: " << e.what());
				}
			};
			stage->ApplyParameterOverrides();

			// Capture before training can prune the source or replace autosaves.
			if (segment.skillStartRatings.empty()) {
				segment.skillStartRatings = storage.ResolveSegmentSkillStartRatings(segment, stage->learnerConfig.skillTracker.initialMu);
				if (!segment.skillStartRatings.empty())
					storage.SaveSegment(segment);
			}
		}
	}

	std::vector<RocketSim::EngineId> ResolveTrainingSimulation(ForgeSettings& settings)
	{
		std::vector<RocketSim::EngineId> engines;
		if (settings.agentName.empty())
			return engines;

		BuildStorage storage(ResolveAgentModelsFolder(settings.agentName), BuildNameOrDefault(settings));
		ForgeSettings prepared = settings;
		PrepareFirstStageContinueLatest(prepared, storage);

		const SegmentInspector inspect = [&](const std::string&, const std::string&, const std::string& id) {
			return InspectSegmentFromDisk(settings.agentName, BuildNameOrDefault(settings), id);
		};
		PrepareQueueLineage(prepared, inspect);

		const std::unique_ptr<Agent> agent = AgentRegistry::Create(settings.agentName);
		const RocketForge::AgentSetupSnapshot defaults = RocketForge::CaptureAgentSetupSnapshot(*agent);
		RocketForge::AgentSetupSnapshot inherited = defaults;
		bool trainsAgainstStaticOpponents = false;
		const auto addEngine = [&engines](RocketSim::EngineId engine) {
			if (std::find(engines.begin(), engines.end(), engine) == engines.end())
				engines.push_back(engine);
		};

		for (size_t i = 0; i < settings.stages.size(); ++i) {
			const StageSettings& entry = prepared.stages[i];
			const StageAction action = DecideAction(prepared, i, inspect);
			const std::string source = action.kind == ActionKind::Continue ? action.continueSegmentId
				: action.kind == ActionKind::Skip ? action.skipSegmentId
				: action.startNewSourceSegmentId;
			RocketForge::AgentSetupSnapshot setup = entry.agentSetupOverride.value_or(defaults);

			// A new descendant inherits its parent's physics even before the parent exists on disk.
			if (i > 0)
				RocketForge::CopyAgentSimulation(setup, inherited);

			if (!source.empty()) {
				const std::optional<TrainingSegmentMetadata> segment = storage.LoadSegmentById(source);
				if (!segment)
					throw std::runtime_error("Cannot restore saved simulator: segment " + source + " was not found in build " + BuildNameOrDefault(settings));

				const fs::path path = storage.SegmentRoot(segment->folderName) / RocketForge::TrainingSetupSnapshot::FILE_NAME;
				const auto saved = RocketForge::LoadExistingTrainingSetupSnapshot(path);
				RocketForge::CopyAgentSimulation(setup, RocketForge::ResolveSavedAgentSetup(saved, defaults));
			}

			RocketForge::ResolveAgentSimulation(setup);
			inherited = setup;

			// Later stages bootstrap from the segment this stage leaves behind.
			// A stage that starts a new segment has no id yet, so the next stage inherits `setup` rather than a stale createdSegmentId.
			prepared.stages[i].createdSegmentId = action.kind == ActionKind::Continue ? action.continueSegmentId :
				action.kind == ActionKind::Skip ? action.skipSegmentId : std::string();

			if (action.kind == ActionKind::Skip)
				continue;

			settings.stages[i].agentSetupOverride = setup;
			trainsAgainstStaticOpponents = trainsAgainstStaticOpponents || entry.staticOpponentsEnabled;
			for (RocketSim::EngineId engine : RocketForge::RequiredEngines(setup))
				addEngine(engine);
		}

		if (trainsAgainstStaticOpponents)
			for (RocketSim::EngineId engine : ResolveStaticOpponentEngines(settings.agentName, BuildNameOrDefault(settings)))
				addEngine(engine);

		return engines;
	}

	namespace {
		bool PersistQueueSettings(const ForgeSettings& settings, const RunPersistence& persistence, std::string* error)
		{
			if (persistence.settingsPath.empty())
				return true;

			if (!persistence.persistedGlobals.has_value())
				return GGL::JsonSaveToFile(settings, persistence.settingsPath, error);

			ForgeSettings persisted = settings;
			persisted.globals = *persistence.persistedGlobals;

			return GGL::JsonSaveToFile(persisted, persistence.settingsPath, error);
		}
	}

	RunResult RunStageQueue(ForgeSettings& settings, const RunPersistence& persistence, const StageEvents& events)
	{
		// Cleared before the slow preparation below, so a stop requested during it ends the run before the first stage.
		StageQueueSignalState::Reset();

		RunResult result;
		try {
			for (RocketSim::EngineId engine : ResolveTrainingSimulation(settings))
				RocketSim::RequireEngine(engine);
		} catch (const RocketSim::EngineUnavailableError& error) {
			result.status = RunStatus::Error;
			result.errorMessage = error.what();
			result.unavailableEngine = error.engine;
			return result;
		} catch (const std::exception& error) {
			result.status = RunStatus::Error;
			result.errorMessage = error.what();
			return result;
		}

		const std::string buildName = BuildNameOrDefault(settings);
		BuildStorage buildStorage(ResolveAgentModelsFolder(settings.agentName), buildName);
		TrainingBuildLock writerLock(buildStorage.BuildRoot());
		buildStorage.EnsureBuildExists();

		PrepareFirstStageContinueLatest(settings, buildStorage);

		SegmentInspector inspector = [&](const std::string& agentName, const std::string& stageName, const std::string& id) {
			(void)stageName;
			return InspectSegmentFromDisk(agentName, buildName, id);
		};

		// Reconcile stale descendants before any stage can resolve its bootstrap source.
		if (PrepareQueueLineage(settings, inspector)) {
			std::string saveErr;
			if (!PersistQueueSettings(settings, persistence, &saveErr))
				std::cerr << "[Queue] Warning: failed to persist queue-lineage prep: " << saveErr << "\n";
		}

		for (size_t i = 0; i < settings.stages.size(); ++i) {
			StageSettings& entry = settings.stages[i];

			// Check for in-flight signals raised between stages.
			if (StageQueueSignalState::ConsumeStop()) {
				result.status = RunStatus::Stopped;
				return result;
			}
			// Skip raised between stages: still advance, but log it.
			if (StageQueueSignalState::ConsumeSkip()) {
				// An uncreated skipped entry aliases the previous queue output. Recording that
				// alias keeps the following entry from accidentally bootstrapping from scratch.
				if (entry.createdSegmentId.empty() && i > 0 &&
					!settings.stages[i - 1].createdSegmentId.empty()) {
					const std::string inheritedSegmentId = settings.stages[i - 1].createdSegmentId;
					if (std::optional<TrainingSegmentMetadata> segment = buildStorage.LoadSegmentById(inheritedSegmentId))
						entry.branchName = NormalizeBranchName(segment->branchName);
					FlipBootstrapToContinueExisting(entry, inheritedSegmentId);

					std::string saveErr;
					if (!PersistQueueSettings(settings, persistence, &saveErr))
						std::cerr << "[Queue] Warning: failed to persist between-stage skip target: " << saveErr << "\n";
				}
				if (events.onStageSkipped)
					events.onStageSkipped((int)i, "Skip raised between stages.");
				continue;
			}

			StageAction action = DecideAction(settings, i, inspector);

			if (action.kind == ActionKind::Skip) {
				std::cout << "[Queue] Stage " << (i+1) << " (" << entry.stageName << ") skipped: " << action.skipReason << "\n";

				if (!action.skipSegmentId.empty() && entry.createdSegmentId != action.skipSegmentId) {
					if (std::optional<TrainingSegmentMetadata> segment = buildStorage.LoadSegmentById(action.skipSegmentId))
						entry.branchName = NormalizeBranchName(segment->branchName);
					FlipBootstrapToContinueExisting(entry, action.skipSegmentId);

					std::string saveErr;
					if (!PersistQueueSettings(settings, persistence, &saveErr))
						std::cerr << "[Queue] Warning: failed to persist skipped segment target: " << saveErr << "\n";
				}

				if (events.onStageSkipped)
					events.onStageSkipped((int)i, action.skipReason);

				continue;
			}

			auto stage = CreateTrainingStage(settings.agentName, entry.stageName, settings.globals, entry);

			if (!stage) {
				result.status = RunStatus::Error;
				result.failedStageIndex = (int)i;
				result.errorMessage = "Failed to create training stage: " + entry.stageName;
				return result;
			}

			if (events.onStageCreated)
				events.onStageCreated((int)i, stage.get());

			double stageTrainingElapsedSeconds = 0.0;
			auto runTraining = [&](const std::function<void()>& launch) {
				const auto startTime = std::chrono::steady_clock::now();
				try {
					launch();
				} catch (...) {
					stageTrainingElapsedSeconds +=
						std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
					throw;
				}
				stageTrainingElapsedSeconds +=
					std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
			};

			auto refreshTrainedSegment = [&](bool stageFinished) {
				if (entry.createdSegmentId.empty())
					return;

				std::optional<TrainingSegmentMetadata> segment = buildStorage.LoadSegmentById(entry.createdSegmentId);
				if (!segment.has_value())
					return;

				// A failed stage never wrote its final durable save, so its progress since the last one lives only in autosaves
				if (!stageFinished) {
					try {
						if (const std::optional<fs::path> promoted = buildStorage.PromoteNewestAutosave(segment.value()))
							RG_LOG("Promoted autosave " << promoted->filename() << " of failed " << segment->segmentId);
					} catch (const std::exception& e) {
						RG_LOG("Warning: failed to promote the newest autosave of " << segment->segmentId << ": " << e.what());
					}
				}

				buildStorage.RefreshSegmentCheckpoints(segment.value(), true);
				if (stageTrainingElapsedSeconds > 0.0)
					segment->totalTrainingSeconds += stageTrainingElapsedSeconds;
				buildStorage.SaveSegment(segment.value());
				SaveBuildAfterSegment(buildStorage, segment.value());
				if (stageFinished)
					CleanupSegmentTempCheckpoints(buildStorage, segment.value());
				stageTrainingElapsedSeconds = 0.0;
			};

			try {
				if (action.kind == ActionKind::Continue) {
					std::optional<TrainingSegmentMetadata> segment = buildStorage.LoadSegmentById(action.continueSegmentId);
					if (!segment.has_value())
						RG_ERR_CLOSE("Segment \"" << action.continueSegmentId << "\" was not found in Build " << buildName);

					if (entry.bootstrap.branchMode == BootstrapConfig::BranchMode::ContinueExisting)
						entry.branchName = NormalizeBranchName(segment->branchName);

					if (entry.createdSegmentId != segment->segmentId) {
						FlipBootstrapToContinueExisting(entry, segment->segmentId);

						std::string saveErr;
						if (!PersistQueueSettings(settings, persistence, &saveErr))
							std::cerr << "[Queue] Warning: failed to persist continued segment target: " << saveErr << "\n";
					}

					if (UpdateWandbNames(settings, segment.value())) {
						segment->updatedAt = BuildStorage::MakeTimestamp();
						buildStorage.SaveSegment(segment.value());
					}

					stage->ApplySegmentId(segment->segmentId, entry.versionNotes);

					ConfigureStageIdentityForSegmentSnapshot(buildStorage, settings, stage.get(), segment->segmentId, segment->stageName);

					RocketForge::TrainingSetupSnapshot snapshot = RocketForge::MakeTrainingSetupSnapshot(stage.get(), segment->segmentId);
					const std::string setupHash = BuildStorage::MakeSetupHash(snapshot);
					const bool setupHashMatches = buildStorage.SegmentSetupHashMatches(segment.value(), snapshot, setupHash);
					const bool rendering = settings.globals.render;

					if (!setupHashMatches && !entry.allowSegmentSetupMismatch && !rendering) {
						const fs::path sourceCheckpoint = LatestCheckpointPath(buildStorage, segment.value());

						if (sourceCheckpoint.empty())
							RG_ERR_CLOSE("Segment \"" << segment->segmentId << "\" setup changed, but it has no checkpoint to branch from.");

						TrainingSegmentMetadata nextSegment = CreateSegmentMetadata(
							buildStorage,
							settings,
							stage.get(),
							entry,
							{},
							segment->segmentId,
							sourceCheckpoint);

						ConfigureStageForSegment(buildStorage, settings, stage.get(), nextSegment, sourceCheckpoint.parent_path());
						buildStorage.SaveSegment(nextSegment);
						SaveSegmentTrainingSetupSnapshot(buildStorage, settings, stage.get(), nextSegment);

						entry.createdSegmentId = nextSegment.segmentId;
						FlipBootstrapToContinueExisting(entry, nextSegment.segmentId);

						std::string saveErr;
						if (!PersistQueueSettings(settings, persistence, &saveErr))
							std::cerr << "[Queue] Warning: failed to persist createdSegmentId: " << saveErr << "\n";

						if (events.onStageStarted)
							events.onStageStarted((int)i, entry.stageName, nextSegment.segmentId);

						runTraining([&]() {
							LaunchNewStageTraining(stage.get(), nullptr, !entry.keepRunId, settings.globals.pruneSourceCheckpoints, true);
						});
					} else {
						if (!setupHashMatches && entry.allowSegmentSetupMismatch && !rendering) {
							UpdateSegmentSetupMetadata(buildStorage, settings, stage.get(), entry, segment.value(), setupHash);
							entry.allowSegmentSetupMismatch = false;

							std::string saveErr;
							if (!PersistQueueSettings(settings, persistence, &saveErr))
								std::cerr << "[Queue] Warning: failed to clear segment setup mismatch override: " << saveErr << "\n";
						}

						ConfigureStageForSegment(buildStorage, settings, stage.get(), segment.value(), fs::path());

						if (events.onStageStarted)
							events.onStageStarted((int)i, entry.stageName, segment->segmentId);

						runTraining([&]() {
							LaunchContinueTraining(stage.get(), nullptr, false, settings.globals.pruneSourceCheckpoints, true);
						});
					}

				}
				// New stage
				else {
					fs::path sourceCheckpoint = {};
					std::string parentSegmentId = {};
					std::optional<TrainingSegmentMetadata> parentSegment = std::nullopt;
					if (!action.startNewSourceSegmentId.empty()) {
						parentSegment = buildStorage.LoadSegmentById(action.startNewSourceSegmentId);
						if (parentSegment.has_value()) {
							parentSegmentId = parentSegment->segmentId;
							sourceCheckpoint = LatestCheckpointPath(buildStorage, parentSegment.value());
							if (sourceCheckpoint.empty())
								RG_ERR_CLOSE("Source segment \"" << parentSegment->segmentId << "\" has no saved checkpoint.");
						}
					}

					if (parentSegment.has_value() && entry.bootstrap.branchMode == BootstrapConfig::BranchMode::ContinueExisting)
						entry.branchName = NormalizeBranchName(parentSegment->branchName);
					else
						entry.branchName = NormalizeBranchName(entry.branchName);

					TrainingSegmentMetadata segment = CreateSegmentMetadata(
						buildStorage,
						settings,
						stage.get(),
						entry,
						action.startNewReuseSegmentId,
						parentSegmentId,
						sourceCheckpoint);
					ConfigureStageForSegment(buildStorage, settings, stage.get(), segment, sourceCheckpoint.parent_path());
					buildStorage.SaveSegment(segment);
					SaveSegmentTrainingSetupSnapshot(buildStorage, settings, stage.get(), segment);

					entry.createdSegmentId = segment.segmentId;
					FlipBootstrapToContinueExisting(entry, segment.segmentId);

					std::string saveErr;
					if (!PersistQueueSettings(settings, persistence, &saveErr))
						std::cerr << "[Queue] Warning: failed to persist createdSegmentId: " << saveErr << "\n";
					if (events.onStageStarted)
						events.onStageStarted((int)i, entry.stageName, segment.segmentId);

					runTraining([&]() {
						LaunchNewStageTraining(stage.get(), nullptr, !entry.keepRunId, settings.globals.pruneSourceCheckpoints, true);
					});
				}
			} catch (const std::exception& e) {
				refreshTrainedSegment(false);
				result.status = RunStatus::Error;
				result.failedStageIndex = (int)i;
				if (const auto* unavailable = dynamic_cast<const RocketSim::EngineUnavailableError*>(&e)) {
					result.unavailableEngine = unavailable->engine;
					result.errorMessage = e.what();
				} else {
					result.errorMessage = std::string("Stage ") + entry.stageName + " failed: " + e.what();
				}
				return result;
			}

			if (!entry.createdSegmentId.empty()) {
				refreshTrainedSegment(true);
			}

			// After learner exits, inspect signals.
			const bool stopFired = StageQueueSignalState::ConsumeStop();
			if (stopFired) {
				result.status = RunStatus::Stopped;
				return result;
			}

			const bool skipFired = StageQueueSignalState::ConsumeSkip();
			if (skipFired) {
				CheckpointInfo info = inspector(settings.agentName, entry.stageName, entry.createdSegmentId);
				if (info.hasCheckpoints && info.latestTimesteps > 0)
					entry.targetTotalTimesteps = info.latestTimesteps;

				std::string saveErr;
				if (!PersistQueueSettings(settings, persistence, &saveErr))
					std::cerr << "[Queue] Warning: failed to persist skip-rewritten target: " << saveErr << "\n";

				if (events.onStageSkipped)
					events.onStageSkipped((int)i, "Skipped by user; target rewritten to current progress.");

				continue;
			}

			if (events.onStageCompleted)
				events.onStageCompleted((int)i);
		}

		// All stages finished. Reset state so the next launch starts completely fresh:
		//   - Stage 0 bootstrap → New (so the GUI shows "start from scratch" by default).
		//   - createdSegmentId cleared on all stages so the GUI shows accurate "Will create"
		//     labels after reloading from disk, and DecideAction has no stale IDs to resume.
		if (!settings.stages.empty()) {
			auto& stage0 = settings.stages[0];
			stage0.bootstrap.mode = BootstrapConfig::Mode::New;
			stage0.bootstrap.branchMode = BootstrapConfig::BranchMode::ContinueExisting;
			stage0.bootstrap.selectedExistingVersionId.clear();

			for (auto& stage : settings.stages) {
				stage.branchName = "main";
				stage.createdSegmentId.clear();
			}

			std::string saveErr;
			if (!PersistQueueSettings(settings, persistence, &saveErr))
				std::cerr << "[Queue] Warning: failed to reset state after completion: " << saveErr << "\n";
		}

		result.status = RunStatus::Completed;
		return result;
	}
}
