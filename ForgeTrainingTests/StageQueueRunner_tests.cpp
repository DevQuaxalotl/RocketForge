#include <gtest/gtest.h>

#include "TestSupport.h"

#include "ForgeTraining.h"
#include "StageQueue/StageQueueRunner.h"
#include "StageRegistry.h"
#include "StaticOpponents/StaticOpponentSelection.h"
#include "StaticOpponents/StaticOpponentSelectionInternal.h"
#include "Storage/BuildStorage.h"
#include "TrainingSetup/TrainingSetupSnapshot.h"
#include "TrainingStage.h"

#include <ATen/Parallel.h>
#include <GigaLearnInfer/AgentRegistry.h>
#include <RocketSim/Engine.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using namespace RocketForgeTraining;

namespace {
	fs::path TempStageQueueStorageRoot(const std::string& name)
	{
		return RocketForgeTests::TestTempDir(name);
	}

	void SaveSegmentForVersionTest(BuildStorage& storage, const std::string& segmentId)
	{
		TrainingSegmentMetadata segment = {};
		segment.segmentId = segmentId;
		segment.folderName = storage.MakeSegmentFolderName(segmentId, "stage");
		storage.SaveSegment(segment);
	}

	SegmentInspector NullInspector()
	{
		return [](const std::string&, const std::string&, const std::string&) {
			return CheckpointInfo{};
		};
	}

	SegmentInspector InspectorFor(const std::string& segmentId, uint64_t timesteps)
	{
		return [=](const std::string&, const std::string&, const std::string& id) {
			CheckpointInfo info;
			if (id == segmentId) {
				info.segmentExists = true;
				info.hasCheckpoints = true;
				info.latestTimesteps = timesteps;
			}
			return info;
		};
	}
}

TEST(Runner, FirstStageNew_StartsNew)
{
	ForgeSettings s;
	StageSettings st;
	st.stageName = "x";
	st.targetTotalTimesteps = 1000;
	st.bootstrap.mode = BootstrapConfig::Mode::New;
	s.stages = { st };

	StageAction a = DecideAction(s, 0, NullInspector());
	ASSERT_TRUE(a.kind == ActionKind::StartNew);
	ASSERT_TRUE(a.startNewSourceStage.empty());
	ASSERT_TRUE(a.startNewSourceSegmentId.empty());
}

TEST(Runner, CreatedSegmentAtTarget_IsSkipped)
{
	ForgeSettings s;
	StageSettings st;
	st.stageName = "x";
	st.targetTotalTimesteps = 1000;
	st.createdSegmentId = "v1.00";
	st.bootstrap.mode = BootstrapConfig::Mode::ContinueExisting;
	st.bootstrap.selectedExistingVersionId = "v1.00";
	s.stages = { st };

	StageAction a = DecideAction(s, 0, InspectorFor("v1.00", 1500));
	ASSERT_TRUE(a.kind == ActionKind::Skip);
	ASSERT_EQ(a.skipSegmentId, std::string("v1.00"));
}

TEST(Runner, CreatedSegmentBelowTarget_IsContinued)
{
	ForgeSettings s;
	StageSettings st;
	st.stageName = "x";
	st.targetTotalTimesteps = 1000;
	st.createdSegmentId = "v1.00";
	st.bootstrap.mode = BootstrapConfig::Mode::ContinueExisting;
	st.bootstrap.selectedExistingVersionId = "v1.00";
	s.stages = { st };

	StageAction a = DecideAction(s, 0, InspectorFor("v1.00", 500));
	ASSERT_TRUE(a.kind == ActionKind::Continue);
	ASSERT_EQ(a.continueSegmentId, std::string("v1.00"));
}

TEST(Runner, CreatedSegmentExistsButNoCheckpoints_ReusesId)
{
	ForgeSettings s;
	StageSettings st;
	st.stageName = "x";
	st.targetTotalTimesteps = 1000;
	st.createdSegmentId = "v1.00";
	st.bootstrap.mode = BootstrapConfig::Mode::ContinueExisting;
	st.bootstrap.selectedExistingVersionId = "v1.00";
	s.stages = { st };
	SegmentInspector phantom = [](const std::string&, const std::string&, const std::string& id) {
		CheckpointInfo info;
		if (id == "v1.00")
			info.segmentExists = true;
		return info;
	};

	StageAction a = DecideAction(s, 0, phantom);
	ASSERT_TRUE(a.kind == ActionKind::StartNew);
	ASSERT_EQ(a.startNewReuseSegmentId, std::string("v1.00"));
}

TEST(Runner, ContinueExistingSelectedSegmentWithoutCheckpoints_ReusesId)
{
	ForgeSettings s;
	StageSettings st;
	st.stageName = "x";
	st.targetTotalTimesteps = 1000;
	st.bootstrap.mode = BootstrapConfig::Mode::ContinueExisting;
	st.bootstrap.selectedExistingVersionId = "v1.00";
	s.stages = { st };
	SegmentInspector phantom = [](const std::string&, const std::string&, const std::string& id) {
		CheckpointInfo info;
		if (id == "v1.00")
			info.segmentExists = true;
		return info;
	};

	StageAction a = DecideAction(s, 0, phantom);
	ASSERT_TRUE(a.kind == ActionKind::StartNew);
	ASSERT_EQ(a.startNewReuseSegmentId, std::string("v1.00"));
	ASSERT_TRUE(a.startNewSourceSegmentId.empty());
}

TEST(Runner, VersionIncrementFromSegmentWithoutCheckpoints_ReusesIdInsteadOfBranching)
{
	ForgeSettings s;
	StageSettings st;
	st.stageName = "x";
	st.targetTotalTimesteps = 1000;
	st.createdSegmentId = "v1.00";
	st.versionIncrement = VersionIncrement::Minor;
	st.bootstrap.mode = BootstrapConfig::Mode::ContinueExisting;
	st.bootstrap.selectedExistingVersionId = "v1.00";
	s.stages = { st };
	SegmentInspector phantom = [](const std::string&, const std::string&, const std::string& id) {
		CheckpointInfo info;
		if (id == "v1.00")
			info.segmentExists = true;
		return info;
	};

	StageAction a = DecideAction(s, 0, phantom);
	ASSERT_TRUE(a.kind == ActionKind::StartNew);
	ASSERT_EQ(a.startNewReuseSegmentId, std::string("v1.00"));
	ASSERT_TRUE(a.startNewSourceSegmentId.empty());
}

TEST(Runner, LoadFromOwnSegmentWithoutCheckpoints_DoesNotUseItselfAsSource)
{
	ForgeSettings s;
	StageSettings st;
	st.stageName = "x";
	st.targetTotalTimesteps = 1000;
	st.createdSegmentId = "v1.07";
	st.bootstrap.mode = BootstrapConfig::Mode::LoadFromOtherStage;
	st.bootstrap.sourceStage = "x";
	st.bootstrap.sourceVersionId = "v1.07";
	s.stages = { st };
	SegmentInspector phantom = [](const std::string&, const std::string&, const std::string& id) {
		CheckpointInfo info;
		if (id == "v1.07")
			info.segmentExists = true;
		return info;
	};

	StageAction a = DecideAction(s, 0, phantom);
	ASSERT_TRUE(a.kind == ActionKind::StartNew);
	ASSERT_EQ(a.startNewReuseSegmentId, std::string("v1.07"));
	ASSERT_TRUE(a.startNewSourceSegmentId.empty());
}

TEST(Runner, RetrainedSegmentWithoutCheckpoints_StartsFromItsRecordedParent)
{
	ForgeSettings s;
	StageSettings st;
	st.stageName = "x";
	st.targetTotalTimesteps = 1000;
	st.createdSegmentId = "v1.07";
	st.bootstrap.mode = BootstrapConfig::Mode::LoadFromOtherStage;
	st.bootstrap.sourceStage = "x";
	st.bootstrap.sourceVersionId = "v1.07";
	s.stages = { st };
	SegmentInspector inspect = [](const std::string&, const std::string&, const std::string& id) {
		CheckpointInfo info;
		if (id == "v1.07") {
			info.segmentExists = true;
			info.parentSegmentId = "v1.05";
		} else if (id == "v1.05") {
			info.segmentExists = true;
			info.hasCheckpoints = true;
			info.latestTimesteps = 500;
		}
		return info;
	};

	StageAction a = DecideAction(s, 0, inspect);
	ASSERT_TRUE(a.kind == ActionKind::StartNew);
	ASSERT_EQ(a.startNewReuseSegmentId, std::string("v1.07"));
	ASSERT_EQ(a.startNewSourceSegmentId, std::string("v1.05"));
}

TEST(Runner, ContinueExisting_HonorsSelectedSegment)
{
	ForgeSettings s;
	StageSettings st;
	st.stageName = "x";
	st.targetTotalTimesteps = 1000;
	st.createdSegmentId = "v1.00";
	st.bootstrap.mode = BootstrapConfig::Mode::ContinueExisting;
	st.bootstrap.selectedExistingVersionId = "v1.01";
	s.stages = { st };
	SegmentInspector inspect = [](const std::string&, const std::string&, const std::string& id) {
		CheckpointInfo info;
		if (id == "v1.00") {
			info.segmentExists = true;
			info.hasCheckpoints = true;
			info.latestTimesteps = 500;
		} else if (id == "v1.01") {
			info.segmentExists = true;
			info.hasCheckpoints = true;
			info.latestTimesteps = 100;
		}
		return info;
	};

	StageAction a = DecideAction(s, 0, inspect);
	ASSERT_TRUE(a.kind == ActionKind::Continue);
	ASSERT_EQ(a.continueSegmentId, std::string("v1.01"));
}

TEST(Runner, NewMode_CreatedSegmentWithCheckpoints_StartsFreshSegment)
{
	ForgeSettings s;
	StageSettings st;
	st.stageName = "x";
	st.targetTotalTimesteps = 1000;
	st.createdSegmentId = "v1.00";
	st.bootstrap.mode = BootstrapConfig::Mode::New;
	st.versionIncrement = VersionIncrement::Minor;
	s.stages = { st };

	StageAction a = DecideAction(s, 0, InspectorFor("v1.00", 500));
	ASSERT_TRUE(a.kind == ActionKind::StartNew);
	ASSERT_TRUE(a.startNewReuseSegmentId.empty());
	ASSERT_TRUE(a.continueSegmentId.empty());
}

TEST(Runner, SecondStageWithoutCreatedSegment_BootstrapsFromPreviousSegment)
{
	ForgeSettings s;
	StageSettings a;
	a.stageName = "ball-chase";
	a.targetTotalTimesteps = 1000;
	a.createdSegmentId = "v1.00";
	StageSettings b;
	b.stageName = "tactics";
	b.targetTotalTimesteps = 2000;
	s.stages = { a, b };

	StageAction act = DecideAction(s, 1, NullInspector());
	ASSERT_TRUE(act.kind == ActionKind::StartNew);
	ASSERT_EQ(act.startNewSourceStage, std::string("ball-chase"));
	ASSERT_EQ(act.startNewSourceSegmentId, std::string("v1.00"));
}

TEST(Runner, SecondStageBelowPreviousCheckpoint_IsSkippedWithPreviousSegment)
{
	ForgeSettings s;
	StageSettings a;
	a.stageName = "ball-chase";
	a.targetTotalTimesteps = 1000;
	a.createdSegmentId = "v1.00";
	StageSettings b;
	b.stageName = "tactics";
	b.targetTotalTimesteps = 2000;
	s.stages = { a, b };

	StageAction act = DecideAction(s, 1, InspectorFor("v1.00", 3000));
	ASSERT_TRUE(act.kind == ActionKind::Skip);
	ASSERT_EQ(act.skipSegmentId, std::string("v1.00"));
}

TEST(Runner, LoadFromOtherStage_NoIncrement_ContinuesSelectedSource)
{
	ForgeSettings s;
	StageSettings st;
	st.stageName = "x";
	st.targetTotalTimesteps = 1000;
	st.bootstrap.mode = BootstrapConfig::Mode::LoadFromOtherStage;
	st.bootstrap.sourceStage = "prev-stage";
	st.bootstrap.sourceVersionId = "v2.03";
	s.stages = { st };

	StageAction a = DecideAction(s, 0, InspectorFor("v2.03", 500));
	ASSERT_TRUE(a.kind == ActionKind::Continue);
	ASSERT_EQ(a.continueSegmentId, std::string("v2.03"));
}

TEST(Runner, ContinueLatest_NoIncrement_ContinuesResolvedLatest)
{
	ForgeSettings s;
	StageSettings st;
	st.stageName = "x";
	st.targetTotalTimesteps = 1000;
	st.bootstrap.mode = BootstrapConfig::Mode::ContinueLatest;
	st.bootstrap.selectedExistingVersionId = "v2.03";
	s.stages = { st };

	StageAction a = DecideAction(s, 0, InspectorFor("v2.03", 500));
	ASSERT_TRUE(a.kind == ActionKind::Continue);
	ASSERT_EQ(a.continueSegmentId, std::string("v2.03"));
}

TEST(Runner, ContinueLatest_AtTarget_SkipsWithResolvedSegment)
{
	ForgeSettings s;
	StageSettings st;
	st.stageName = "x";
	st.targetTotalTimesteps = 1000;
	st.bootstrap.mode = BootstrapConfig::Mode::ContinueLatest;
	st.bootstrap.selectedExistingVersionId = "v2.03";
	s.stages = { st };

	StageAction a = DecideAction(s, 0, InspectorFor("v2.03", 1500));
	ASSERT_TRUE(a.kind == ActionKind::Skip);
	ASSERT_EQ(a.skipSegmentId, std::string("v2.03"));
}

TEST(Runner, LoadFromOtherStage_WithMinorIncrement_StartsNewFromSource)
{
	ForgeSettings s;
	StageSettings st;
	st.stageName = "x";
	st.targetTotalTimesteps = 1000;
	st.versionIncrement = VersionIncrement::Minor;
	st.bootstrap.mode = BootstrapConfig::Mode::LoadFromOtherStage;
	st.bootstrap.sourceStage = "prev-stage";
	st.bootstrap.sourceVersionId = "v2.03";
	s.stages = { st };

	StageAction a = DecideAction(s, 0, InspectorFor("v2.03", 500));
	ASSERT_TRUE(a.kind == ActionKind::StartNew);
	ASSERT_EQ(a.startNewSourceStage, std::string("prev-stage"));
	ASSERT_EQ(a.startNewSourceSegmentId, std::string("v2.03"));
}

TEST(Runner, CreateNewBranchFromSource_StartsNewWithoutRevisionIncrement)
{
	ForgeSettings s;
	StageSettings st;
	st.stageName = "x";
	st.targetTotalTimesteps = 1000;
	st.bootstrap.mode = BootstrapConfig::Mode::LoadFromOtherStage;
	st.bootstrap.branchMode = BootstrapConfig::BranchMode::CreateNew;
	st.bootstrap.sourceStage = "prev-stage";
	st.bootstrap.sourceVersionId = "v2.03";
	st.branchName = "experiment";
	s.stages = { st };

	StageAction a = DecideAction(s, 0, InspectorFor("v2.03", 500));
	ASSERT_TRUE(a.kind == ActionKind::StartNew);
	ASSERT_EQ(a.startNewSourceStage, std::string("prev-stage"));
	ASSERT_EQ(a.startNewSourceSegmentId, std::string("v2.03"));
}

TEST(Runner, SelectExistingBranchFromSource_StartsNewWithoutRevisionIncrement)
{
	ForgeSettings s;
	StageSettings st;
	st.stageName = "x";
	st.targetTotalTimesteps = 1000;
	st.bootstrap.mode = BootstrapConfig::Mode::LoadFromOtherStage;
	st.bootstrap.branchMode = BootstrapConfig::BranchMode::ContinueSelected;
	st.bootstrap.sourceStage = "prev-stage";
	st.bootstrap.sourceVersionId = "v2.03";
	st.branchName = "Triple Entropy";
	s.stages = { st };

	StageAction a = DecideAction(s, 0, InspectorFor("v2.03", 500));
	ASSERT_TRUE(a.kind == ActionKind::StartNew);
	ASSERT_EQ(a.startNewSourceStage, std::string("prev-stage"));
	ASSERT_EQ(a.startNewSourceSegmentId, std::string("v2.03"));
}

TEST(Runner, SelectExistingBranchOnLaterCreatedSegment_StartsNewFromPreviousSegment)
{
	ForgeSettings s;
	StageSettings first;
	first.stageName = "warmup";
	first.targetTotalTimesteps = 1000;
	first.createdSegmentId = "v1.00";
	StageSettings second;
	second.stageName = "main";
	second.targetTotalTimesteps = 2000;
	second.createdSegmentId = "v1.01";
	second.bootstrap.branchMode = BootstrapConfig::BranchMode::ContinueSelected;
	second.branchName = "Triple Entropy";
	s.stages = { first, second };

	StageAction a = DecideAction(s, 1, InspectorFor("v1.01", 500));
	ASSERT_TRUE(a.kind == ActionKind::StartNew);
	ASSERT_EQ(a.startNewSourceStage, std::string("warmup"));
	ASSERT_EQ(a.startNewSourceSegmentId, std::string("v1.00"));
	ASSERT_TRUE(a.continueSegmentId.empty());
}

TEST(Runner, ContinueExisting_WithVersionOverride_StartsNewFromSelectedSameStage)
{
	ForgeSettings s;
	StageSettings st;
	st.stageName = "x";
	st.targetTotalTimesteps = 1000;
	st.bootstrap.mode = BootstrapConfig::Mode::ContinueExisting;
	st.bootstrap.selectedExistingVersionId = "v1.00";
	st.versionOverride = GGL::SegmentVersionNumber{ 1, 1 };
	s.stages = { st };

	StageAction a = DecideAction(s, 0, InspectorFor("v1.00", 500));
	ASSERT_TRUE(a.kind == ActionKind::StartNew);
	ASSERT_EQ(a.startNewSourceStage, std::string("x"));
	ASSERT_EQ(a.startNewSourceSegmentId, std::string("v1.00"));
	ASSERT_TRUE(a.continueSegmentId.empty());
}

TEST(Runner, ContinueExisting_WithMinorIncrement_StartsNewFromSelectedSameStage)
{
	ForgeSettings s;
	StageSettings st;
	st.stageName = "x";
	st.targetTotalTimesteps = 1000;
	st.bootstrap.mode = BootstrapConfig::Mode::ContinueExisting;
	st.bootstrap.selectedExistingVersionId = "v1.00";
	st.versionIncrement = VersionIncrement::Minor;
	s.stages = { st };

	StageAction a = DecideAction(s, 0, InspectorFor("v1.00", 500));
	ASSERT_TRUE(a.kind == ActionKind::StartNew);
	ASSERT_EQ(a.startNewSourceStage, std::string("x"));
	ASSERT_EQ(a.startNewSourceSegmentId, std::string("v1.00"));
	ASSERT_TRUE(a.continueSegmentId.empty());
}

TEST(Runner, NextMinorFromSelectedSegment_StaysInSelectedMajor)
{
	fs::path root = TempStageQueueStorageRoot("minor_selected_major");
	BuildStorage storage(root, "V1");
	SaveSegmentForVersionTest(storage, "v1.00");
	SaveSegmentForVersionTest(storage, "v1.03");
	SaveSegmentForVersionTest(storage, "v2.04");

	StageSettings st;
	st.versionIncrement = VersionIncrement::Minor;

	ASSERT_EQ(ResolveNewSegmentId(storage, st, {}, "v1.00"), std::string("v1.04"));

	fs::remove_all(root);
}

TEST(Runner, NextMajorFromSelectedSegment_UsesHighestExistingMajor)
{
	fs::path root = TempStageQueueStorageRoot("major_global");
	BuildStorage storage(root, "V1");
	SaveSegmentForVersionTest(storage, "v1.03");
	SaveSegmentForVersionTest(storage, "v2.04");

	StageSettings st;
	st.versionIncrement = VersionIncrement::Major;

	ASSERT_EQ(ResolveNewSegmentId(storage, st, {}, "v1.03"), std::string("v3.00"));

	fs::remove_all(root);
}

TEST(Runner, ContinueExisting_WithAllocatedOverrideTarget_ResumesCreatedSegment)
{
	ForgeSettings s;
	StageSettings st;
	st.stageName = "x";
	st.targetTotalTimesteps = 1000;
	st.createdSegmentId = "v1.01";
	st.bootstrap.mode = BootstrapConfig::Mode::ContinueExisting;
	st.bootstrap.selectedExistingVersionId = "v1.01";
	st.versionOverride = GGL::SegmentVersionNumber{ 1, 1 };
	s.stages = { st };

	StageAction a = DecideAction(s, 0, InspectorFor("v1.01", 500));
	ASSERT_TRUE(a.kind == ActionKind::Continue);
	ASSERT_EQ(a.continueSegmentId, std::string("v1.01"));
}

TEST(Runner, QueueLineage_ClearsCreatedSegmentIdsAfterFreshFirstStage)
{
	ForgeSettings s;
	StageSettings a;
	a.stageName = "warmup";
	a.targetTotalTimesteps = 1000;
	a.bootstrap.mode = BootstrapConfig::Mode::New;
	a.createdSegmentId = "v1.00";
	StageSettings b;
	b.stageName = "main";
	b.targetTotalTimesteps = 2000;
	b.createdSegmentId = "v2.00";
	s.stages = { a, b };

	bool mutated = PrepareQueueLineage(s, NullInspector());
	ASSERT_TRUE(mutated);
	ASSERT_EQ(s.stages[0].createdSegmentId, std::string("v1.00"));
	ASSERT_TRUE(s.stages[1].createdSegmentId.empty());
}

TEST(Runner, QueueLineage_PreservesCompletedMatchingSegments)
{
	ForgeSettings s;
	StageSettings a;
	a.stageName = "warmup";
	a.targetTotalTimesteps = 1000;
	a.bootstrap.mode = BootstrapConfig::Mode::ContinueExisting;
	a.bootstrap.selectedExistingVersionId = "v1.00";
	a.createdSegmentId = "v1.00";
	StageSettings b;
	b.stageName = "main";
	b.targetTotalTimesteps = 2000;
	b.createdSegmentId = "v2.00";
	s.stages = { a, b };

	SegmentInspector inspect = [](const std::string&, const std::string&, const std::string& id) {
		CheckpointInfo info;
		info.segmentExists = true;
		info.hasCheckpoints = true;
		if (id == "v1.00") {
			info.latestTimesteps = 1000;
			info.stageName = "warmup";
		} else if (id == "v2.00") {
			info.latestTimesteps = 2000;
			info.stageName = "main";
			info.parentSegmentId = "v1.00";
		}
		return info;
	};

	bool mutated = PrepareQueueLineage(s, inspect);
	ASSERT_TRUE(!mutated);
	ASSERT_EQ(s.stages[0].createdSegmentId, std::string("v1.00"));
	ASSERT_EQ(s.stages[1].createdSegmentId, std::string("v2.00"));
}

TEST(Runner, QueueLineage_ClearsLaterIdsWhenFirstStageCreatesBranch)
{
	ForgeSettings s;
	StageSettings a;
	a.stageName = "warmup";
	a.targetTotalTimesteps = 1000;
	a.bootstrap.mode = BootstrapConfig::Mode::LoadFromOtherStage;
	a.bootstrap.branchMode = BootstrapConfig::BranchMode::CreateNew;
	a.bootstrap.sourceStage = "source";
	a.bootstrap.sourceVersionId = "v1.00";
	a.createdSegmentId = "v1.01";
	StageSettings b;
	b.stageName = "main";
	b.targetTotalTimesteps = 2000;
	b.createdSegmentId = "v2.00";
	s.stages = { a, b };

	bool mutated = PrepareQueueLineage(s, NullInspector());
	ASSERT_TRUE(mutated);
	ASSERT_EQ(s.stages[0].createdSegmentId, std::string("v1.01"));
	ASSERT_TRUE(s.stages[1].createdSegmentId.empty());
}

TEST(Runner, QueueLineage_ClearsLaterIdsWhenFirstStageSelectionChanges)
{
	ForgeSettings s;
	StageSettings first;
	first.stageName = "warmup";
	first.targetTotalTimesteps = 1000;
	first.bootstrap.mode = BootstrapConfig::Mode::ContinueExisting;
	first.bootstrap.selectedExistingVersionId = "v1.01";
	first.createdSegmentId = "v1.00";
	StageSettings second;
	second.stageName = "main";
	second.targetTotalTimesteps = 2000;
	second.createdSegmentId = "v1.02";
	StageSettings third;
	third.stageName = "finetune";
	third.targetTotalTimesteps = 3000;
	third.createdSegmentId = "v1.03";
	s.stages = { first, second, third };

	bool mutated = PrepareQueueLineage(s, NullInspector());
	ASSERT_TRUE(mutated);
	ASSERT_EQ(s.stages[0].createdSegmentId, std::string("v1.00"));
	ASSERT_TRUE(s.stages[1].createdSegmentId.empty());
	ASSERT_TRUE(s.stages[2].createdSegmentId.empty());
}

TEST(Runner, QueueLineage_ClearsLaterIdsWhenEarlierSegmentWillContinue)
{
	ForgeSettings s;
	StageSettings first;
	first.stageName = "warmup";
	first.targetTotalTimesteps = 1000;
	first.bootstrap.mode = BootstrapConfig::Mode::ContinueExisting;
	first.bootstrap.selectedExistingVersionId = "v1.00";
	first.createdSegmentId = "v1.00";
	StageSettings second;
	second.stageName = "main";
	second.targetTotalTimesteps = 2000;
	second.createdSegmentId = "v1.01";
	s.stages = { first, second };

	bool mutated = PrepareQueueLineage(s, InspectorFor("v1.00", 500));
	ASSERT_TRUE(mutated);
	ASSERT_EQ(s.stages[0].createdSegmentId, std::string("v1.00"));
	ASSERT_TRUE(s.stages[1].createdSegmentId.empty());
}

TEST(Runner, QueueLineage_LaterBranchChangeClearsOnlyDescendants)
{
	ForgeSettings s;
	StageSettings first;
	first.stageName = "warmup";
	first.targetTotalTimesteps = 1000;
	first.bootstrap.mode = BootstrapConfig::Mode::ContinueExisting;
	first.bootstrap.selectedExistingVersionId = "v1.00";
	first.createdSegmentId = "v1.00";
	StageSettings second;
	second.stageName = "main";
	second.targetTotalTimesteps = 2000;
	second.createdSegmentId = "v1.01";
	second.bootstrap.branchMode = BootstrapConfig::BranchMode::CreateNew;
	StageSettings third;
	third.stageName = "finetune";
	third.targetTotalTimesteps = 3000;
	third.createdSegmentId = "v1.02";
	s.stages = { first, second, third };

	SegmentInspector inspect = [](const std::string&, const std::string&, const std::string& id) {
		CheckpointInfo info;
		info.segmentExists = true;
		info.hasCheckpoints = true;
		if (id == "v1.00") {
			info.latestTimesteps = 1000;
			info.stageName = "warmup";
		} else if (id == "v1.01") {
			info.latestTimesteps = 2000;
			info.stageName = "main";
			info.parentSegmentId = "v1.00";
		}
		return info;
	};

	bool mutated = PrepareQueueLineage(s, inspect);
	ASSERT_TRUE(mutated);
	ASSERT_EQ(s.stages[1].createdSegmentId, std::string("v1.01"));
	ASSERT_TRUE(s.stages[2].createdSegmentId.empty());
}

TEST(Runner, QueueLineage_StageEditClearsEditedSegmentAndDescendants)
{
	ForgeSettings s;
	StageSettings first;
	first.stageName = "warmup";
	first.targetTotalTimesteps = 1000;
	first.bootstrap.mode = BootstrapConfig::Mode::ContinueExisting;
	first.bootstrap.selectedExistingVersionId = "v1.00";
	first.createdSegmentId = "v1.00";
	StageSettings second;
	second.stageName = "new-main";
	second.targetTotalTimesteps = 2000;
	second.createdSegmentId = "v1.01";
	StageSettings third;
	third.stageName = "finetune";
	third.targetTotalTimesteps = 3000;
	third.createdSegmentId = "v1.02";
	s.stages = { first, second, third };

	SegmentInspector inspect = [](const std::string&, const std::string&, const std::string& id) {
		CheckpointInfo info;
		info.segmentExists = true;
		info.hasCheckpoints = true;
		if (id == "v1.00") {
			info.latestTimesteps = 1000;
			info.stageName = "warmup";
		} else if (id == "v1.01") {
			info.latestTimesteps = 2000;
			info.stageName = "old-main";
			info.parentSegmentId = "v1.00";
		}
		return info;
	};

	bool mutated = PrepareQueueLineage(s, inspect);
	ASSERT_TRUE(mutated);
	ASSERT_TRUE(s.stages[1].createdSegmentId.empty());
	ASSERT_TRUE(s.stages[2].createdSegmentId.empty());
}

TEST(Runner, LaterCreatedSegment_WithMinorIncrementStartsNewFromPrevious)
{
	ForgeSettings s;
	StageSettings first;
	first.stageName = "warmup";
	first.createdSegmentId = "v1.00";
	StageSettings second;
	second.stageName = "main";
	second.createdSegmentId = "v1.01";
	second.versionIncrement = VersionIncrement::Minor;
	s.stages = { first, second };

	StageAction action = DecideAction(s, 1, InspectorFor("v1.01", 500));
	ASSERT_TRUE(action.kind == ActionKind::StartNew);
	ASSERT_EQ(action.startNewSourceStage, std::string("warmup"));
	ASSERT_EQ(action.startNewSourceSegmentId, std::string("v1.00"));
}

TEST(Runner, NewSegmentCreated_FlipsBootstrapToContinueExisting)
{
	StageSettings entry;
	entry.bootstrap.mode = BootstrapConfig::Mode::New;
	entry.versionIncrement = VersionIncrement::Major;
	entry.versionOverride = GGL::SegmentVersionNumber{ 2, 0 };

	FlipBootstrapToContinueExisting(entry, "v1.05");
	ASSERT_EQ(entry.createdSegmentId, std::string("v1.05"));
	ASSERT_TRUE(entry.bootstrap.mode == BootstrapConfig::Mode::ContinueExisting);
	ASSERT_TRUE(entry.bootstrap.branchMode == BootstrapConfig::BranchMode::ContinueExisting);
	ASSERT_EQ(entry.bootstrap.selectedExistingVersionId, std::string("v1.05"));
	ASSERT_TRUE(entry.versionIncrement == VersionIncrement::None);
	ASSERT_FALSE(entry.versionOverride.has_value());
	ASSERT_TRUE(entry.bootstrap.sourceStage.empty());
	ASSERT_TRUE(entry.bootstrap.sourceVersionId.empty());
}

TEST(Runner, LoadFromOtherStageCreatedSegment_FlipsBootstrapAndClearsSource)
{
	StageSettings entry;
	entry.bootstrap.mode = BootstrapConfig::Mode::LoadFromOtherStage;
	entry.bootstrap.sourceStage = "prev";
	entry.bootstrap.sourceVersionId = "v2.03";

	FlipBootstrapToContinueExisting(entry, "v1.07");
	ASSERT_TRUE(entry.bootstrap.mode == BootstrapConfig::Mode::ContinueExisting);
	ASSERT_EQ(entry.bootstrap.selectedExistingVersionId, std::string("v1.07"));
	ASSERT_TRUE(entry.bootstrap.sourceStage.empty());
	ASSERT_TRUE(entry.bootstrap.sourceVersionId.empty());
}

TEST(Runner, ContinueExisting_EmptySelectedVersionId_FallsBackToStartNew)
{
	ForgeSettings s;
	StageSettings st;
	st.stageName = "x";
	st.targetTotalTimesteps = 1000;
	st.bootstrap.mode = BootstrapConfig::Mode::ContinueExisting;
	s.stages = { st };

	StageAction a = DecideAction(s, 0, NullInspector());
	ASSERT_TRUE(a.kind == ActionKind::StartNew);
	ASSERT_TRUE(a.continueSegmentId.empty());
	ASSERT_TRUE(a.startNewSourceStage.empty());
}

TEST(ResolveQueueStart, EmptyQueue_NotResolved)
{
	ForgeSettings s;
	fs::path root = TempStageQueueStorageRoot("rqs_empty");
	BuildStorage storage(root, "V1");

	QueueStart start = ResolveQueueStart(s, storage, NullInspector());
	ASSERT_TRUE(!start.resolved);
}

TEST(ResolveQueueStart, FreshFirstStage_PredictsNewSegment)
{
	ForgeSettings s;
	StageSettings st;
	st.stageName = "ball-chase";
	st.targetTotalTimesteps = 1000;
	st.bootstrap.mode = BootstrapConfig::Mode::New;
	s.stages = { st };

	fs::path root = TempStageQueueStorageRoot("rqs_fresh");
	BuildStorage storage(root, "V1");

	QueueStart start = ResolveQueueStart(s, storage, NullInspector());
	ASSERT_TRUE(start.resolved);
	ASSERT_TRUE(start.isNew);
	ASSERT_EQ(start.stageName, std::string("ball-chase"));
	ASSERT_EQ(start.segmentId, std::string("v1.00"));
}

TEST(ResolveQueueStart, SkipsCompleteThenContinues)
{
	ForgeSettings s;

	StageSettings s0;
	s0.stageName = "ball-chase";
	s0.targetTotalTimesteps = 1000;
	s0.createdSegmentId = "v1.00";
	s0.bootstrap.mode = BootstrapConfig::Mode::ContinueExisting;
	s0.bootstrap.selectedExistingVersionId = "v1.00";

	StageSettings s1;
	s1.stageName = "basic-score";
	s1.targetTotalTimesteps = 1000;
	s1.createdSegmentId = "v1.01";
	s1.bootstrap.mode = BootstrapConfig::Mode::ContinueExisting;
	s1.bootstrap.selectedExistingVersionId = "v1.01";

	s.stages = { s0, s1 };

	// v1.00 is at/over target (skip); v1.01 is below target (continue).
	SegmentInspector inspect = [](const std::string&, const std::string&, const std::string& id) {
		CheckpointInfo info;
		info.segmentExists = true;
		info.hasCheckpoints = true;
		if (id == "v1.00") {
			info.latestTimesteps = 1500;
			info.stageName = "ball-chase";
		} else {
			info.latestTimesteps = 500;
			info.stageName = "basic-score";
			info.parentSegmentId = "v1.00";
		}
		return info;
	};

	fs::path root = TempStageQueueStorageRoot("rqs_continue");
	BuildStorage storage(root, "V1");

	QueueStart start = ResolveQueueStart(s, storage, inspect);
	ASSERT_TRUE(start.resolved);
	ASSERT_TRUE(!start.isNew);
	ASSERT_EQ(start.stageName, std::string("basic-score"));
	ASSERT_EQ(start.segmentId, std::string("v1.01"));
}

TEST(ResolveQueueStart, PropagatesSkippedSegmentThroughQueue)
{
	ForgeSettings s;
	StageSettings first;
	first.stageName = "warmup";
	first.targetTotalTimesteps = 1000;
	first.bootstrap.mode = BootstrapConfig::Mode::ContinueExisting;
	first.bootstrap.selectedExistingVersionId = "v1.00";
	StageSettings second;
	second.stageName = "main";
	second.targetTotalTimesteps = 2000;
	StageSettings third;
	third.stageName = "finetune";
	third.targetTotalTimesteps = 6000;
	s.stages = { first, second, third };

	fs::path root = TempStageQueueStorageRoot("rqs_skip_chain");
	BuildStorage storage(root, "V1");

	QueueStart start = ResolveQueueStart(s, storage, InspectorFor("v1.00", 5000));
	ASSERT_TRUE(start.resolved);
	ASSERT_TRUE(start.isNew);
	ASSERT_EQ(start.stageName, std::string("finetune"));
	ASSERT_EQ(start.segmentId, std::string("v1.01"));
}

TEST(ResolveQueueStart, AllComplete_NotResolved)
{
	ForgeSettings s;
	StageSettings st;
	st.stageName = "ball-chase";
	st.targetTotalTimesteps = 1000;
	st.createdSegmentId = "v1.00";
	st.bootstrap.mode = BootstrapConfig::Mode::ContinueExisting;
	st.bootstrap.selectedExistingVersionId = "v1.00";
	s.stages = { st };

	fs::path root = TempStageQueueStorageRoot("rqs_done");
	BuildStorage storage(root, "V1");

	QueueStart start = ResolveQueueStart(s, storage, InspectorFor("v1.00", 5000));
	ASSERT_TRUE(!start.resolved);
}

TEST(PlanQueue, DescribesEveryEntry)
{
	struct Case {
		std::string label;
		std::function<ForgeSettings()> makeSettings;
		SegmentInspector inspect;
		std::vector<QueuePlanEntry> expected;
	};

	const auto continueEntry = [](const std::string& stage, const std::string& segmentId, uint64_t target) {
		StageSettings entry;
		entry.stageName = stage;
		entry.targetTotalTimesteps = target;
		entry.createdSegmentId = segmentId;
		entry.bootstrap.mode = BootstrapConfig::Mode::ContinueExisting;
		entry.bootstrap.selectedExistingVersionId = segmentId;
		return entry;
	};

	const auto row = [](ActionKind kind, const std::string& segmentId, const std::string& sourceStage, const std::string& sourceSegmentId,
						uint64_t current, uint64_t target) {
		QueuePlanEntry planEntry;
		planEntry.kind = kind;
		planEntry.segmentId = segmentId;
		planEntry.sourceStage = sourceStage;
		planEntry.sourceSegmentId = sourceSegmentId;
		planEntry.currentTimesteps = current;
		planEntry.targetTotalTimesteps = target;
		return planEntry;
	};

	const Case cases[] = {
		{
			"fresh queue starts a new segment",
			[] {
				ForgeSettings s;
				StageSettings entry;
				entry.stageName = "ball-chase";
				entry.targetTotalTimesteps = 1000;
				entry.bootstrap.mode = BootstrapConfig::Mode::New;
				s.stages = { entry };
				return s;
			},
			NullInspector(),
			{ row(ActionKind::StartNew, "v1.00", "", "", 0, 1000) },
		},
		{
			"existing segment below target continues",
			[&] {
				ForgeSettings s;
				s.stages = { continueEntry("ball-chase", "v1.00", 1000) };
				return s;
			},
			InspectorFor("v1.00", 500),
			{ row(ActionKind::Continue, "v1.00", "", "", 500, 1000) },
		},
		{
			"complete entry is skipped and the next starts from it",
			[&] {
				ForgeSettings s;
				StageSettings next;
				next.stageName = "basic-score";
				next.targetTotalTimesteps = 2000;
				next.bootstrap.mode = BootstrapConfig::Mode::LoadFromOtherStage;
				next.bootstrap.sourceStage = "ball-chase";
				s.stages = { continueEntry("ball-chase", "v1.00", 1000), next };
				return s;
			},
			InspectorFor("v1.00", 1500),
			{
				row(ActionKind::Skip, "v1.00", "", "", 1500, 1000),
				row(ActionKind::StartNew, "v1.01", "ball-chase", "v1.00", 0, 2000),
			},
		},
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);

		const ForgeSettings settings = testCase.makeSettings();
		const fs::path root = TempStageQueueStorageRoot("plan_queue");
		BuildStorage storage(root, "V1");

		const std::vector<QueuePlanEntry> plan = PlanQueue(settings, storage, testCase.inspect);
		ASSERT_EQ(plan.size(), testCase.expected.size());

		for (size_t i = 0; i < plan.size(); ++i) {
			SCOPED_TRACE(i);
			const QueuePlanEntry& expected = testCase.expected[i];
			EXPECT_EQ(plan[i].stageIndex, i);
			EXPECT_EQ(plan[i].stageName, settings.stages[i].stageName);
			EXPECT_EQ(plan[i].kind, expected.kind);
			EXPECT_EQ(plan[i].segmentId, expected.segmentId);
			EXPECT_EQ(plan[i].sourceStage, expected.sourceStage);
			EXPECT_EQ(plan[i].sourceSegmentId, expected.sourceSegmentId);
			EXPECT_EQ(plan[i].currentTimesteps, expected.currentTimesteps);
			EXPECT_EQ(plan[i].targetTotalTimesteps, expected.targetTotalTimesteps);
		}

		fs::remove_all(root);
	}
}

namespace {
	bool ContainsId(const std::vector<std::string>& ids, const std::string& id)
	{
		return std::find(ids.begin(), ids.end(), id) != ids.end();
	}
}

TEST(ResolveQueueSegmentIds, FreshFirstStage_ListsPredictedNewId)
{
	ForgeSettings s;
	StageSettings st;
	st.stageName = "ball-chase";
	st.targetTotalTimesteps = 1000;
	st.bootstrap.mode = BootstrapConfig::Mode::New;
	s.stages = { st };

	fs::path root = TempStageQueueStorageRoot("rqsi_fresh");
	BuildStorage storage(root, "V1");

	std::vector<std::string> ids = ResolveQueueSegmentIds(s, storage, NullInspector());
	ASSERT_EQ(ids.size(), size_t(1));
	ASSERT_EQ(ids[0], std::string("v1.00"));

	fs::remove_all(root);
}

TEST(ResolveQueueSegmentIds, CollectsSettingsIdsAndNextNewId)
{
	ForgeSettings s;

	StageSettings s0;
	s0.stageName = "ball-chase";
	s0.targetTotalTimesteps = 1000;
	s0.createdSegmentId = "v1.01";
	s0.bootstrap.mode = BootstrapConfig::Mode::LoadFromOtherStage;
	s0.bootstrap.sourceStage = "warmup";
	s0.bootstrap.sourceVersionId = "v1.00";

	StageSettings s1;
	s1.stageName = "basic-score";
	s1.targetTotalTimesteps = 1000;

	s.stages = { s0, s1 };

	fs::path root = TempStageQueueStorageRoot("rqsi_next");
	BuildStorage storage(root, "V1");
	SaveSegmentForVersionTest(storage, "v1.00");
	SaveSegmentForVersionTest(storage, "v1.01");

	// Stage 0 resumes its selected source v1.00 (below target); stage 1 then starts new from it
	// and receives the next minor id after the highest existing v1.x segment (v1.01) -> v1.02.
	std::vector<std::string> ids = ResolveQueueSegmentIds(s, storage, InspectorFor("v1.00", 500));
	ASSERT_TRUE(ContainsId(ids, "v1.00"));
	ASSERT_TRUE(ContainsId(ids, "v1.01"));
	ASSERT_TRUE(ContainsId(ids, "v1.02"));
	ASSERT_EQ(ids.size(), size_t(3));

	fs::remove_all(root);
}

TEST(ResolveQueueSegmentIds, ContinueLatest_ResolvesActiveSegment)
{
	ForgeSettings s;
	StageSettings st;
	st.stageName = "ball-chase";
	st.targetTotalTimesteps = 1000;
	st.bootstrap.mode = BootstrapConfig::Mode::ContinueLatest;
	s.stages = { st };

	fs::path root = TempStageQueueStorageRoot("rqsi_latest");
	BuildStorage storage(root, "V1");
	SaveSegmentForVersionTest(storage, "v1.03");
	BuildMetadata build = storage.LoadBuild();
	build.activeSegmentId = "v1.03";
	storage.SaveBuild(build);

	std::vector<std::string> ids = ResolveQueueSegmentIds(s, storage, InspectorFor("v1.03", 500));
	ASSERT_TRUE(ContainsId(ids, "v1.03"));

	fs::remove_all(root);
}

TEST(ResolveQueueSegmentIds, SkippedSegment_IsCollected)
{
	ForgeSettings s;

	StageSettings s0;
	s0.stageName = "ball-chase";
	s0.targetTotalTimesteps = 1000;
	s0.createdSegmentId = "v1.00";
	s0.bootstrap.mode = BootstrapConfig::Mode::ContinueExisting;
	s0.bootstrap.selectedExistingVersionId = "v1.00";

	StageSettings s1;
	s1.stageName = "basic-score";
	s1.targetTotalTimesteps = 1000;

	s.stages = { s0, s1 };

	fs::path root = TempStageQueueStorageRoot("rqsi_skip");
	BuildStorage storage(root, "V1");

	// v1.00 is past both targets, so stage 0 skips on it and stage 1 skips on its predecessor's output.
	std::vector<std::string> ids = ResolveQueueSegmentIds(s, storage, InspectorFor("v1.00", 1500));
	ASSERT_TRUE(ContainsId(ids, "v1.00"));
	ASSERT_EQ(ids.size(), size_t(1));

	fs::remove_all(root);
}

namespace {
	const std::string ENGINE_QUEUE_AGENT = "EngineQueueTest";
	const std::string ENGINE_QUEUE_STAGE = "EngineQueueTest";
	const std::string ENGINE_QUEUE_BUILD = "EngineQueueTest";
	constexpr uint64_t ENGINE_QUEUE_ITERATION = 256;

	class EngineQueueStage : public RocketForge::TrainingStage {
	public:
		explicit EngineQueueStage(std::unique_ptr<Agent> agent) : TrainingStage(std::move(agent)) {
			stageName = ENGINE_QUEUE_STAGE;
			timestepsPerIteration = ENGINE_QUEUE_ITERATION;
			learnerConfig.numGames = 2;
			learnerConfig.deviceType = GGL::LearnerDeviceType::CPU;
			learnerConfig.ppo.tsPerItr = ENGINE_QUEUE_ITERATION;
			learnerConfig.ppo.batchSize = ENGINE_QUEUE_ITERATION;
			learnerConfig.ppo.microBatchSize = ENGINE_QUEUE_ITERATION;
			learnerConfig.ppo.epochs = 1;
			learnerConfig.ppo.maxEpisodeDuration = 1.0;
			learnerConfig.tsPerSave = ENGINE_QUEUE_ITERATION;
			learnerConfig.skillTracker.enabled = false;
			learnerConfig.sendMetrics = false;
			learnerConfig.randomSeed = 123;
		}
	};

	/**
	 * Restores torch's intra-op thread count, which every Learner sets for the whole process.
	 * Later tests compare float outputs whose reduction order depends on it.
	 */
	class TorchThreadCountGuard {
	public:
		TorchThreadCountGuard() : _threads(at::get_num_threads()) {}
		~TorchThreadCountGuard() { at::set_num_threads(_threads); }

	private:
		int _threads;
	};

	/** (Re)registers the fixture agent with its own default engines; a zero prediction engine follows the training engine. */
	void RegisterEngineQueueFixtures(const fs::path& modelsRoot, int engineVersion, int predictionEngineVersion = 0) {
		AgentRegistry::Register(ENGINE_QUEUE_AGENT, [modelsRoot, engineVersion, predictionEngineVersion] {
			auto agent = std::make_unique<Agent>();
			agent->agentName = ENGINE_QUEUE_AGENT;
			agent->trainingEngineVersion = engineVersion;
			agent->ballPredictionEngineVersion = predictionEngineVersion;
			agent->modelPath = modelsRoot;
			agent->sharedHead.layerSizes = { 16 };
			agent->policy.layerSizes = { 16 };
			agent->critic.layerSizes = { 16 };
			return agent;
		});
		RocketForge::StageRegistry::Register(ENGINE_QUEUE_STAGE, [](std::unique_ptr<Agent> agent) {
			return RocketForge::TrainingStage::Create<EngineQueueStage>(std::move(agent));
		}, {});
	}

	StageSettings MakeEngineQueueStage(uint64_t targetTimesteps) {
		StageSettings stage;
		stage.stageName = ENGINE_QUEUE_STAGE;
		stage.targetTotalTimesteps = targetTimesteps;
		return stage;
	}

	ForgeSettings MakeEngineQueue(std::vector<StageSettings> stages) {
		ForgeSettings settings;
		settings.agentName = ENGINE_QUEUE_AGENT;
		settings.buildName = ENGINE_QUEUE_BUILD;
		settings.globals.metricsEnabled = false;
		settings.globals.skillTestingEnabled = false;
		settings.stages = std::move(stages);
		return settings;
	}

	std::set<std::string> SegmentIds(const BuildStorage& storage) {
		std::set<std::string> ids;
		for (const TrainingSegmentMetadata& segment : storage.ListSegments())
			ids.insert(segment.segmentId);
		return ids;
	}

	/** Trains one iteration of a new segment and returns its id, or empty when no segment was created. */
	std::string TrainNewSegment(const BuildStorage& storage, const fs::path& settingsPath) {
		const std::set<std::string> before = SegmentIds(storage);

		ForgeSettings settings = MakeEngineQueue({ MakeEngineQueueStage(ENGINE_QUEUE_ITERATION) });
		if (RunTraining(settings, settingsPath).status != RunStatus::Completed)
			return {};

		for (const std::string& id : SegmentIds(storage))
			if (!before.count(id))
				return id;
		return {};
	}

	fs::path TrainingSetupPath(const BuildStorage& storage, const TrainingSegmentMetadata& segment) {
		return storage.SegmentRoot(segment.folderName) / RocketForge::TrainingSetupSnapshot::FILE_NAME;
	}

	/** Rewrites a segment's training_setup.json as it was saved before snapshots recorded an engine. */
	bool StripRecordedEngine(const BuildStorage& storage, const std::string& segmentId) {
		const std::optional<TrainingSegmentMetadata> segment = storage.LoadSegmentById(segmentId);
		if (!segment)
			return false;

		const fs::path path = TrainingSetupPath(storage, *segment);
		nlohmann::json setup;
		std::ifstream(path) >> setup;
		setup.erase("engine_version");
		setup.erase("engine_revision");
		setup.at("agent_setup").erase("training_engine_version");
		setup.at("agent_setup").erase("ball_prediction_engine_version");
		std::ofstream(path) << setup.dump(2);
		return true;
	}

	int RecordedEngineVersion(const BuildStorage& storage, const TrainingSegmentMetadata& segment) {
		const std::optional<RocketForge::TrainingSetupSnapshot> snapshot = RocketForge::LoadTrainingSetupSnapshot(TrainingSetupPath(storage, segment));
		return snapshot ? snapshot->engineVersion : -1;
	}

	std::optional<TrainingSegmentMetadata> ChildSegment(const BuildStorage& storage, const std::string& parentSegmentId) {
		for (const TrainingSegmentMetadata& segment : storage.ListSegments())
			if (segment.parentSegmentId == parentSegmentId)
				return segment;
		return std::nullopt;
	}
}

TEST(StageQueueRunner, StaleCreatedSegmentIdDoesNotChangeTheBootstrapEngine)
{
	ROCKETFORGE_REQUIRE_COLLISION_MESHES();
	ROCKETFORGE_REQUIRE_ENGINE(RocketSim::EngineId::V3);

	const TorchThreadCountGuard threadCountGuard;
	const fs::path root = RocketForgeTests::TestTempDir();
	const fs::path settingsPath = root / "forge_settings.json";
	BuildStorage storage(root / "Models", ENGINE_QUEUE_BUILD);

	RegisterEngineQueueFixtures(root / "Models", 2);
	const std::string legacySegmentId = TrainNewSegment(storage, settingsPath);
	ASSERT_FALSE(legacySegmentId.empty());
	ASSERT_TRUE(StripRecordedEngine(storage, legacySegmentId));

	RegisterEngineQueueFixtures(root / "Models", 3);
	const std::string v3SegmentId = TrainNewSegment(storage, settingsPath);
	ASSERT_FALSE(v3SegmentId.empty());

	// Stage 0 branches off the legacy segment while its createdSegmentId still names the v3 segment.
	// Stage 1 bootstraps from stage 0's new v2 segment, so both stages must train on v2.
	StageSettings first = MakeEngineQueueStage(ENGINE_QUEUE_ITERATION * 2);
	first.bootstrap.mode = BootstrapConfig::Mode::ContinueExisting;
	first.bootstrap.selectedExistingVersionId = legacySegmentId;
	first.versionIncrement = VersionIncrement::Minor;
	first.createdSegmentId = v3SegmentId;
	ForgeSettings settings = MakeEngineQueue({ first, MakeEngineQueueStage(ENGINE_QUEUE_ITERATION * 3) });

	ForgeSettings resolved = settings;
	EXPECT_EQ(ResolveTrainingSimulation(resolved), std::vector<RocketSim::EngineId>{ RocketSim::EngineId::V2 });
	ASSERT_TRUE(resolved.stages[1].agentSetupOverride);
	EXPECT_EQ(resolved.stages[1].agentSetupOverride->trainingEngineVersion, 2);

	testing::internal::CaptureStdout();
	const RunResult result = RunTraining(settings, settingsPath);
	const std::string log = testing::internal::GetCapturedStdout();

	ASSERT_EQ(result.status, RunStatus::Completed) << result.errorMessage;
	EXPECT_NE(log.find("Training simulator: RocketSim v2"), std::string::npos);
	EXPECT_EQ(log.find("Training simulator: RocketSim v3"), std::string::npos);

	const std::optional<TrainingSegmentMetadata> firstSegment = ChildSegment(storage, legacySegmentId);
	ASSERT_TRUE(firstSegment);
	EXPECT_EQ(RecordedEngineVersion(storage, *firstSegment), 2);

	const std::optional<TrainingSegmentMetadata> secondSegment = ChildSegment(storage, firstSegment->segmentId);
	ASSERT_TRUE(secondSegment);
	EXPECT_EQ(RecordedEngineVersion(storage, *secondSegment), 2);
}

TEST(StageQueueRunner, UnavailableEngineFailsBeforeAnyWork)
{
	const RocketSim::Testing::ScopedEngineUnavailable unavailable(RocketSim::EngineId::V3, "test reason");

	const fs::path root = RocketForgeTests::TestTempDir();
	RegisterEngineQueueFixtures(root / "Models", 3);

	ForgeSettings settings = MakeEngineQueue({ MakeEngineQueueStage(ENGINE_QUEUE_ITERATION) });
	const RunResult result = RunTraining(settings, root / "forge_settings.json");

	EXPECT_EQ(result.status, RunStatus::Error);
	EXPECT_EQ(result.errorMessage, "test reason");
	EXPECT_EQ(result.unavailableEngine, std::optional<RocketSim::EngineId>(RocketSim::EngineId::V3));
	EXPECT_FALSE(fs::exists(root / "Models" / ENGINE_QUEUE_BUILD / "segments"));
}

TEST(StageQueueRunner, StopDuringPreparationEndsTheRunBeforeAnyStage)
{
	ROCKETFORGE_REQUIRE_COLLISION_MESHES();

	const TorchThreadCountGuard threadCountGuard;
	const fs::path root = RocketForgeTests::TestTempDir();
	const fs::path modelsRoot = root / "Models";
	BuildStorage storage(modelsRoot, ENGINE_QUEUE_BUILD);
	RegisterEngineQueueFixtures(modelsRoot, 2);

	// The engine precheck creates the agent before any stage, so a stop raised there arrives mid-preparation.
	const std::shared_ptr<bool> stopRaised = std::make_shared<bool>(false);
	AgentRegistry::Register(ENGINE_QUEUE_AGENT, [modelsRoot, stopRaised] {
		if (!*stopRaised) {
			*stopRaised = true;
			RequestStop();
		}

		auto agent = std::make_unique<Agent>();
		agent->agentName = ENGINE_QUEUE_AGENT;
		agent->trainingEngineVersion = 2;
		agent->modelPath = modelsRoot;
		agent->sharedHead.layerSizes = { 16 };
		agent->policy.layerSizes = { 16 };
		agent->critic.layerSizes = { 16 };
		return agent;
	});

	ForgeSettings settings = MakeEngineQueue({ MakeEngineQueueStage(ENGINE_QUEUE_ITERATION) });
	const RunResult result = RunTraining(settings, root / "forge_settings.json");

	EXPECT_TRUE(*stopRaised);
	EXPECT_EQ(result.status, RunStatus::Stopped);
	EXPECT_TRUE(SegmentIds(storage).empty());
}

TEST(StageQueueRunner, UnavailablePredictionEngineFailsBeforeAnyWork)
{
	const RocketSim::Testing::ScopedEngineUnavailable unavailable(RocketSim::EngineId::V3, "test reason");

	const TorchThreadCountGuard threadCountGuard;
	const fs::path root = RocketForgeTests::TestTempDir();
	RegisterEngineQueueFixtures(root / "Models", 2, 3);

	ForgeSettings settings = MakeEngineQueue({ MakeEngineQueueStage(ENGINE_QUEUE_ITERATION) });
	const RunResult result = RunTraining(settings, root / "forge_settings.json");

	EXPECT_EQ(result.status, RunStatus::Error);
	EXPECT_EQ(result.errorMessage, "test reason");
	EXPECT_EQ(result.unavailableEngine, std::optional<RocketSim::EngineId>(RocketSim::EngineId::V3));
	EXPECT_FALSE(fs::exists(root / "Models" / ENGINE_QUEUE_BUILD / "segments"));
}

TEST(StageQueueRunner, MissingSourceSegmentFailsTheEnginePrecheck)
{
	const fs::path root = RocketForgeTests::TestTempDir();
	RegisterEngineQueueFixtures(root / "Models", 2);

	StageSettings stage = MakeEngineQueueStage(ENGINE_QUEUE_ITERATION);
	stage.bootstrap.mode = BootstrapConfig::Mode::ContinueExisting;
	stage.bootstrap.selectedExistingVersionId = "v9.00";
	ForgeSettings settings = MakeEngineQueue({ stage });

	EXPECT_THROW(ResolveTrainingSimulation(settings), std::runtime_error);
}

namespace {
	const std::string OPPONENT_BUILD = "EngineQueueOpponent";

	/** Saves one checkpointed segment of the fixture agent in OPPONENT_BUILD, with `setupText` as its training_setup.json (none when empty). */
	StaticOpponentSelectionEntry MakeOpponentSegment(const fs::path& modelsRoot, const std::string& setupText) {
		BuildStorage storage(modelsRoot, OPPONENT_BUILD);
		storage.EnsureBuildExists();
		TrainingSegmentMetadata segment;
		segment.segmentId = "v1.00";
		segment.stageName = ENGINE_QUEUE_STAGE;
		segment.buildName = OPPONENT_BUILD;
		segment.folderName = storage.MakeSegmentFolderName(segment.segmentId, segment.stageName);
		storage.SaveSegment(segment);
		const fs::path checkpoint = storage.SegmentCheckpointsRoot(segment.folderName) / "100";
		fs::create_directories(checkpoint);
		std::ofstream(checkpoint / "POLICY.lt") << "policy";
		if (!setupText.empty())
			std::ofstream(storage.SegmentRoot(segment.folderName) / RocketForge::TrainingSetupSnapshot::FILE_NAME) << setupText;

		StaticOpponentSelectionEntry entry;
		entry.sourceAgentName = ENGINE_QUEUE_AGENT;
		entry.sourceBuildName = OPPONENT_BUILD;
		entry.sourceStageName = ENGINE_QUEUE_STAGE;
		entry.sourceVersionId = segment.segmentId;
		return entry;
	}

	std::string SetupTextWithTrainingEngine(int engineVersion) {
		RocketForge::TrainingSetupSnapshot snapshot;
		snapshot.engineVersion = engineVersion;
		snapshot.agentSetup = RocketForge::CaptureAgentSetupSnapshot(*AgentRegistry::Create(ENGINE_QUEUE_AGENT));
		snapshot.agentSetup->trainingEngineVersion = engineVersion;
		snapshot.agentSetup->ballPredictionEngineVersion = 0;
		nlohmann::json saved;
		GGL::JsonSave(snapshot, saved);
		return saved.dump();
	}
}

TEST(StaticOpponentSource, LegacySegmentsRunOnV2AndBrokenSetupsThrow)
{
	using namespace RocketForgeTraining::StaticOpponentSelectionInternal;
	const fs::path root = RocketForgeTests::TestTempDir();
	RegisterEngineQueueFixtures(root / "Models", 3);

	const StaticOpponentSelectionEntry legacy = MakeOpponentSegment(root / "Models", "");
	const RocketForge::AgentSetupSnapshot setup = ResolveSourceVersion(legacy).entry.agentSetup.value();
	EXPECT_EQ(setup.trainingEngineVersion, 2);
	EXPECT_EQ(setup.ballPredictionEngineVersion, 2);

	// A stale setup captured at selection time must not stand in for the segment's own record.
	StaticOpponentSelectionEntry cached = legacy;
	cached.agentSetup = RocketForge::CaptureAgentSetupSnapshot(*AgentRegistry::Create(ENGINE_QUEUE_AGENT));
	EXPECT_EQ(ResolveSourceVersion(cached).entry.agentSetup->trainingEngineVersion, 2);

	const StaticOpponentSelectionEntry broken = MakeOpponentSegment(root / "Models", "broken");
	EXPECT_THROW(ResolveSourceVersion(broken), std::runtime_error);
}

TEST(StageQueueRunner, UnavailableStaticOpponentEngineFailsBeforeAnyWork)
{
	const TorchThreadCountGuard threadCountGuard;
	const fs::path root = RocketForgeTests::TestTempDir();
	RegisterEngineQueueFixtures(root / "Models", 2);
	const StaticOpponentSelectionEntry opponent = MakeOpponentSegment(root / "Models", SetupTextWithTrainingEngine(3));
	SaveStaticOpponentSelection(ENGINE_QUEUE_AGENT, ENGINE_QUEUE_BUILD, { opponent });

	StageSettings stage = MakeEngineQueueStage(ENGINE_QUEUE_ITERATION);
	stage.staticOpponentsEnabled = true;
	ForgeSettings settings = MakeEngineQueue({ stage });
	EXPECT_EQ(ResolveTrainingSimulation(settings), (std::vector<RocketSim::EngineId>{ RocketSim::EngineId::V2, RocketSim::EngineId::V3 }));

	const RocketSim::Testing::ScopedEngineUnavailable unavailable(RocketSim::EngineId::V3, "test reason");
	settings = MakeEngineQueue({ stage });
	const RunResult result = RunTraining(settings, root / "forge_settings.json");

	EXPECT_EQ(result.status, RunStatus::Error);
	EXPECT_EQ(result.unavailableEngine, std::optional<RocketSim::EngineId>(RocketSim::EngineId::V3));
	EXPECT_FALSE(fs::exists(root / "Models" / ENGINE_QUEUE_BUILD / "segments"));
}

TEST(StageQueueRunner, PersistedGlobalsKeepRunOverridesOutOfTheSettingsFile)
{
	ROCKETFORGE_REQUIRE_COLLISION_MESHES();

	const TorchThreadCountGuard threadCountGuard;
	const fs::path root = RocketForgeTests::TestTempDir();
	const fs::path settingsPath = root / "forge_settings.json";
	RegisterEngineQueueFixtures(root / "Models", 2);

	ForgeSettings saved = MakeEngineQueue({ MakeEngineQueueStage(ENGINE_QUEUE_ITERATION) });
	saved.globals.persistSegmentSnapshotsDuringTraining = false;
	saved.stages[0].bootstrap.selectedExistingVersionId = "v9.99";
	ASSERT_TRUE(GGL::JsonSaveToFile(saved, settingsPath));

	ForgeSettings run = saved;
	run.globals.persistSegmentSnapshotsDuringTraining = true;
	RunPersistence persistence{ settingsPath, saved.globals };
	ASSERT_EQ(RunTraining(run, persistence).status, RunStatus::Completed);

	ForgeSettings reloaded;
	std::vector<std::string> errors;
	ASSERT_TRUE(GGL::JsonLoadFromFile(settingsPath, reloaded, errors));
	EXPECT_FALSE(reloaded.globals.persistSegmentSnapshotsDuringTraining);
	ASSERT_EQ(reloaded.stages.size(), 1u);

	// The completion reset is runner bookkeeping and must still reach the file.
	EXPECT_TRUE(reloaded.stages[0].bootstrap.selectedExistingVersionId.empty());
}

TEST(StageQueueRunner, EmptySettingsPathNeverWritesTheSettingsFile)
{
	ROCKETFORGE_REQUIRE_COLLISION_MESHES();

	const TorchThreadCountGuard threadCountGuard;
	const fs::path root = RocketForgeTests::TestTempDir();
	const fs::path settingsPath = root / "forge_settings.json";
	BuildStorage storage(root / "Models", ENGINE_QUEUE_BUILD);
	RegisterEngineQueueFixtures(root / "Models", 2);

	const ForgeSettings queue = MakeEngineQueue({ MakeEngineQueueStage(ENGINE_QUEUE_ITERATION), MakeEngineQueueStage(ENGINE_QUEUE_ITERATION * 2) });
	ASSERT_TRUE(GGL::JsonSaveToFile(queue, settingsPath));

	const auto readBytes = [&] {
		std::ifstream in(settingsPath, std::ios::binary);
		return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
	};
	const std::string before = readBytes();
	const size_t segmentsBefore = SegmentIds(storage).size();

	ForgeSettings adHoc = MakeEngineQueue({ MakeEngineQueueStage(ENGINE_QUEUE_ITERATION) });
	ASSERT_EQ(RunTraining(adHoc, RunPersistence{}).status, RunStatus::Completed);

	EXPECT_EQ(readBytes(), before);
	EXPECT_EQ(SegmentIds(storage).size(), segmentsBefore + 1);
}
