#include "Commands.h"

#include "Cli/CliErrors.h"
#include "Cli/CliResolve.h"

#include "ForgeTraining.h"
#include "RewardAnalysis/RewardAnalysis.h"
#include "SimulationAnalysis/SimulationAnalysis.h"
#include "TrainingSetup/RewardComparison.h"
#include "TrainingStage.h"

#include <GigaLearnCPP/TrainingStorage.h>

#include <climits>
#include <filesystem>
#include <iostream>

using namespace std;
using namespace RocketForge;
using namespace RocketForge::Cli;

namespace fs = std::filesystem;

namespace {

	const char* const REWARD_TABLE_USAGE =
		"Usage: ForgeConsoleTrainer reward-table --stage <name> [options]\n"
		"\n"
		"Prints a reward comparison matrix for a stage without starting training.\n"
		"\n"
		"Options:\n"
		"  --stage <name>       Stage to print (required; see `list stages`)\n"
		"  --agent <name>       Agent the stage belongs to (default: the GUI-selected agent, else example-agent)\n";

	const char* const REWARD_ANALYSIS_USAGE =
		"Usage: ForgeConsoleTrainer reward-analysis [options]\n"
		"\n"
		"Runs a segment's newest durable checkpoint for N timesteps per team size and reports each reward's discounted return contribution.\n"
		"The result is saved in the segment folder, where the GUI loads it.\n"
		"\n"
		"Options:\n"
		"  --agent <name>       Agent to analyse (default: the GUI-selected agent, else example-agent)\n"
		"  --build <name>       Build to analyse (default: the build named by forge_settings.json)\n"
		"  --segment <id>       Segment to analyse (default: the build's active segment; see `list segments`)\n"
		"  --stage <name>       Stage to analyse on (default: the segment's trained stage)\n"
		"  --timesteps <n>      Timesteps per team size (default: the stage's batch size)\n"
		"  --file <path>        forge_settings.json that names the default build\n";

	const char* const ACTION_TRACE_USAGE =
		"Usage: ForgeConsoleTrainer action-trace [options]\n"
		"\n"
		"Runs a saved policy offline and writes selected policy actions plus per-tick applied controls to a CSV and a summary JSON.\n"
		"The setup and checkpoint come from the resolved segment unless --setup and --checkpoint-path give them.\n"
		"\n"
		"Options:\n"
		"  --agent <name>           Agent to trace (default: the GUI-selected agent, else example-agent)\n"
		"  --build <name>           Build to trace (default: the build named by forge_settings.json)\n"
		"  --segment <id>           Segment to trace (default: the build's active segment; see `list segments`)\n"
		"  --checkpoint <ts>        Checkpoint timesteps within the segment (default: newest; see `list checkpoints`)\n"
		"  --file <path>            forge_settings.json that names the default build\n"
		"  --setup <path>           training_setup.json or agent_setup JSON source, instead of the segment's\n"
		"  --checkpoint-path <path> Folder containing POLICY.lt, instead of the segment's checkpoint\n"
		"  --out <path>             CSV output path (default: action_trace.csv)\n"
		"  --episodes <n>           Number of rollouts (default: 8)\n"
		"  --max-ticks <n>          Physics ticks per rollout (default: 720)\n"
		"  --team-size <n>          Cars per team for generated resets, 1-3 (default: 1)\n"
		"  --seed <n>               Deterministic reset variation seed (default: 1234)\n"
		"  --kickoff                Trace normal kickoff starts (default reset mode)\n"
		"  --wall-probe             Use the original side-wall probe reset\n"
		"  --stage-reset <name>     Use any training stage's state setter for each rollout\n"
		"  --snapshot-csv <p>       Replay from an RLBot trace row's obs_state_snapshot\n"
		"  --snapshot-tick <n>      physics_tick row to load from --snapshot-csv\n"
		"  --snapshot-bot-index <n> Choose a bot_index when multiple rows share the snapshot tick\n"
		"  --snapshot-car-id <n>    Choose a car_id when multiple rows share the snapshot tick\n"
		"  --action-sequence-csv <p> Use selected_action_id rows from an RLBot trace CSV\n"
		"  --control-replay-csv <p> Apply per-tick applied_* controls from an RLBot trace CSV\n"
		"  --control-replay-input-delay-ticks <n> Delay replayed controls by n physics ticks\n"
		"  --control-replay-use-last-input Use RLBot flatbuffer last_input controls for replay\n"
		"  --control-replay-focus-policy   Drive the snapshot focus car with the offline policy\n"
		"  --force-action <n>       Force one policy action id instead of model selection\n"
		"  --stochastic             Sample actions instead of taking argmax\n"
		"  --temperature <f>        Policy temperature when stochastic (default: 1)\n"
		"\n"
		"Without a reset mode, action-trace uses a normal kickoff reset.\n"
		"The CSV is generic; the summary JSON also includes wall-dash and speed-flip mining helpers when applicable.\n";

	int RunRewardTableCommand(const CliArgs& args)
	{
		const string agentName = ResolveAgent(args);
		const string stageName = ResolveStage(args, true);

		PrintResolved(cout, { { "agent", agentName }, { "stage", stageName } });

		const unique_ptr<TrainingStage> stage = RocketForgeTraining::CreateTrainingStage(agentName, stageName, {}, {});
		if (!stage)
			throw CliRuntimeError("could not create stage '" + stageName + "' for agent '" + agentName + "'");

		vector<RLGC::WeightedReward> weightedRewards = stage->GetRewards();

		RewardComparisonTable::Build(weightedRewards).PrintToConsole();

		for (RLGC::WeightedReward& weightedReward : weightedRewards)
			delete weightedReward.reward;

		return 0;
	}

	int RunRewardAnalysisCommand(const CliArgs& args)
	{
		const string agentName = ResolveAgent(args);
		const string buildName = ResolveBuild(args, agentName, ResolveSettingsPath(args, agentName));
		const GGL::SegmentVersionSummary segment = ResolveSegment(args, agentName, buildName);

		string stageName = ResolveStage(args, false);
		if (stageName.empty())
			stageName = segment.stageName;
		if (stageName.empty())
			throw CliUsageError("--stage is required; the segment records no trained stage. Run `ForgeConsoleTrainer list stages`");

		const optional<uint64_t> requestedTimesteps = args.GetUint64("--timesteps");
		if (requestedTimesteps.has_value() && *requestedTimesteps > static_cast<uint64_t>(INT_MAX))
			throw CliUsageError("--timesteps is too large (got " + to_string(*requestedTimesteps) + ")");

		const fs::path segmentRoot = segment.rootPath;
		const fs::path checkpointPath = GGL::TrainingStorage::GetMostRecentCheckpointIn(segmentRoot / "checkpoints");
		if (!fs::exists(checkpointPath))
			throw CliRuntimeError("segment " + segment.id + " has no durable checkpoint to analyse: " + checkpointPath.string());

		PrintResolved(cout, { { "agent", agentName }, { "build", buildName }, { "segment", segment.id }, { "stage", stageName } });

		const unique_ptr<TrainingStage> stage = RocketForgeTraining::CreateTrainingStage(agentName, stageName, {}, {});
		if (!stage)
			throw CliRuntimeError("could not create stage '" + stageName + "' for agent '" + agentName + "'");

		// Zero defers to the stage's batch size.
		const int requested = static_cast<int>(requestedTimesteps.value_or(0));
		const int timesteps = requested > 0 ? requested : stage->GetTimestepsPerIteration();

		cout << "Running reward analysis: " << agentName << "/" << stageName << " (" << timesteps << " timesteps per team size)\n";
		cout.flush();

		int lastPercent = -1;
		const auto reportProgress = [&lastPercent](int done, int total, const string& label) {
			const int percent = total > 0 ? done * 100 / total : 100;
			if (percent == lastPercent)
				return;

			lastPercent = percent;
			cout << "ANALYSIS_PROGRESS " << done << "/" << total << " " << label << "\n";
			cout.flush();
		};

		const RewardAnalysisResult result = RunRewardAnalysis(stage.get(), checkpointPath, timesteps, reportProgress);

		// Saved in the segment folder so the GUI loads it when the segment is reopened.
		const fs::path savePath = segmentRoot / RewardAnalysisCacheFileName(stageName);
		string saveError;
		if (!SaveRewardAnalysisResult(result, segmentRoot, stageName, &saveError))
			throw CliRuntimeError("could not save the reward analysis to " + savePath.string() + ": " + saveError);

		cout << "Saved reward analysis to " << savePath << "\n";
		return 0;
	}

}

vector<CliCommand> RocketForge::Cli::AnalysisCommands()
{
	CliCommand rewardTable;
	rewardTable.name = "reward-table";
	rewardTable.group = "Debug & analysis";
	rewardTable.summary = "Prints a reward comparison matrix for a stage (does not start training)";
	rewardTable.usage = REWARD_TABLE_USAGE;
	rewardTable.options.valueOptions = { "--agent", "--stage" };
	rewardTable.run = RunRewardTableCommand;

	CliCommand rewardAnalysis;
	rewardAnalysis.name = "reward-analysis";
	rewardAnalysis.group = "Debug & analysis";
	rewardAnalysis.summary = "Collects N timesteps of experience and reports the discounted return contribution of each reward";
	rewardAnalysis.usage = REWARD_ANALYSIS_USAGE;
	rewardAnalysis.options.valueOptions = { "--agent", "--build", "--segment", "--stage", "--timesteps", "--file" };
	rewardAnalysis.run = RunRewardAnalysisCommand;

	CliCommand actionTrace;
	actionTrace.name = "action-trace";
	actionTrace.group = "Debug & analysis";
	actionTrace.summary = "Traces a saved policy's selected and per-tick applied actions to CSV";
	actionTrace.usage = ACTION_TRACE_USAGE;
	actionTrace.options.valueOptions = {
		"--agent", "--build", "--segment", "--checkpoint", "--file", "--setup", "--checkpoint-path", "--out", "--episodes", "--max-ticks",
		"--team-size", "--seed", "--stage-reset", "--snapshot-csv", "--snapshot-tick", "--snapshot-bot-index", "--snapshot-car-id",
		"--action-sequence-csv", "--control-replay-csv", "--control-replay-input-delay-ticks", "--force-action", "--temperature",
	};
	actionTrace.options.switches = {
		"--kickoff", "--wall-probe", "--control-replay-use-last-input", "--control-replay-focus-policy", "--stochastic",
	};
	actionTrace.run = RunActionTraceCommand;

	return { rewardTable, rewardAnalysis, actionTrace };
}
