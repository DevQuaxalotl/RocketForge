#pragma once

#include "Settings/ForgeSettings.h"
#include "ForgeTraining.h"

#include <RocketSim/Engine.h>

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace RocketForge { class TrainingStage; }

namespace RocketForgeTraining {
	class BuildStorage;

	enum class ActionKind { Skip, Continue, StartNew };

	struct StageAction {
		ActionKind  kind = ActionKind::Skip;
		std::string skipReason;
		std::string skipSegmentId;
		std::string continueSegmentId;
		std::string startNewSourceStage;
		std::string startNewSourceSegmentId;
		std::string startNewReuseSegmentId;
	};

	struct CheckpointInfo {
		bool     segmentExists   = false;
		bool     hasCheckpoints  = false;
		uint64_t latestTimesteps = 0;
		std::string stageName;
		std::string parentSegmentId;
	};

	using SegmentInspector = std::function<CheckpointInfo (const std::string& agentName, const std::string& stageName, const std::string& segmentId)>;

	/** Pure-logic action selector for a queue entry. Used by tests with an injected inspector. */
	StageAction DecideAction(const ForgeSettings& settings, size_t stageIndex, const SegmentInspector& inspect);

	/**
	 * Clears queued segment IDs after the first entry whose output will change, and
	 * rejects completed-prefix entries whose saved stage or parent no longer matches.
	 * Returns true if any stage was mutated. Pure logic; callers persist.
	 **/
	bool PrepareQueueLineage(ForgeSettings& settings, const SegmentInspector& inspect);

	/**
	 * Records segmentId as the queue entry's output and flips its bootstrap to
	 * ContinueExisting. Source-stage fields are cleared.
	 **/
	void FlipBootstrapToContinueExisting(StageSettings& entry, const std::string& segmentId);

	/**
	 * Resolves the segment ID assigned when a queue entry creates a new segment.
	 * Minor increments stay inside parentSegmentId's major revision when possible.
	 **/
	std::string ResolveNewSegmentId(
		const BuildStorage& storage,
		const StageSettings& entry,
		const std::string& reuseSegmentId,
		const std::string& parentSegmentId);

	/** What one queue entry will do when the queue runs, computed without side effects. */
	struct QueuePlanEntry {
		size_t stageIndex = 0;
		std::string stageName;
		ActionKind kind = ActionKind::Skip;

		/** Continue target, or the id a new segment would receive. Empty for Skip without a segment. */
		std::string segmentId;
		std::string sourceStage;
		std::string sourceSegmentId;
		std::string skipReason;
		uint64_t currentTimesteps = 0;
		uint64_t targetTotalTimesteps = 0;
	};

	/**
	 * Plans every queue entry by mirroring RunStageQueue's pre-loop prep + DecideAction walk,
	 * but never mutating settings or touching the trainer. The injectable overload is for tests.
	 * Ids for segments that do not exist yet are predictions from the current disk state,
	 * so consecutive major-version bumps in one queue may share a predicted id.
	 **/
	std::vector<QueuePlanEntry> PlanQueue(const ForgeSettings& settings, const BuildStorage& storage, const SegmentInspector& inspect);
	std::vector<QueuePlanEntry> PlanQueue(const ForgeSettings& settings);

	/** What a queued run will begin with, computed without side effects. */
	struct QueueStart {
		std::string stageName;   // first stage that will actually run
		std::string segmentId;   // segment continued, or the id a new one would receive
		bool        isNew = false;   // true when the first real action is StartNew
		bool        resolved = false;// false when the queue is empty or fully complete
	};

	/** Resolves the first stage+segment a queue run will start at: the first non-Skip entry of PlanQueue. */
	QueueStart ResolveQueueStart(
		const ForgeSettings& settings,
		const BuildStorage& storage,
		const SegmentInspector& inspect);
	QueueStart ResolveQueueStart(const ForgeSettings& settings);

	/**
	 * Every segment id a queue run may read or write, deduplicated:
	 * ids the settings name, continue and skip targets, StartNew sources,
	 * and the id each not-yet-created segment would receive, as planned by PlanQueue.
	 **/
	std::vector<std::string> ResolveQueueSegmentIds(
		const ForgeSettings& settings,
		const BuildStorage& storage,
		const SegmentInspector& inspect);

	/** Reads a segment's existence, stage, parent, and latest checkpoint from the build on disk. */
	CheckpointInfo InspectSegmentFromDisk(const std::string& agentName, const std::string& buildName, const std::string& segmentId);

	/**
	 * Restores each stage's saved simulator/prediction settings into its agentSetupOverride. Read-only on disk.
	 * Returns every engine the stages that will train need (each one's training and prediction engine,
	 * plus the Build's static opponents' when a stage trains against them), once each, in queue order.
	 */
	std::vector<RocketSim::EngineId> ResolveTrainingSimulation(ForgeSettings& settings);

	/**
	* Runs a queue of stages defined by ForgeSettings. Settings are mutated in place
	* (created segment IDs written, skip-rewritten targets) and saved through persistence
	* after each mutation.
	*/
	RunResult RunStageQueue(ForgeSettings& settings, const RunPersistence& persistence, const StageEvents& events);
}
