#include "SummaryAnalysis.h"

#include "Options.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>

namespace RocketForge::SimulationAnalysis
{
	using namespace RLGC;

	nlohmann::json ActionToJson(const Action& action)
	{
		return nlohmann::json{
			{ "throttle", action.throttle },
			{ "steer", action.steer },
			{ "pitch", action.pitch },
			{ "yaw", action.yaw },
			{ "roll", action.roll },
			{ "jump", action.jump },
			{ "boost", action.boost },
			{ "handbrake", action.handbrake },
		};
	}

	int StableActionId(const std::vector<TraceSample>& samples, size_t start, size_t count, bool& mixed)
	{
		std::map<int, int> counts;
		for (size_t i = start; i < start + count && i < samples.size(); ++i)
			counts[samples[i].appliedActionId]++;

		int bestId = -1;
		int bestCount = -1;
		for (const auto& [id, idCount] : counts)
		{
			if (idCount > bestCount)
			{
				bestId = id;
				bestCount = idCount;
			}
		}

		mixed = counts.size() > 1;
		return bestId;
	}

	std::vector<WallDashPairStats> BuildWallDashPairSummary(const std::vector<TraceSample>& samples)
	{
		// Optional analysis helper: find stable 4+4 action pairs around jump edges
		// near a wall. The raw CSV remains useful even when this summary is empty.
		std::map<std::pair<int, int>, WallDashPairStats> byPair;

		for (size_t i = 0; i + 15 < samples.size(); ++i)
		{
			const TraceSample& start = samples[i];
			if (!start.jumpEdge || !start.nearWall)
				continue;
			if (samples[i + 7].episode != start.episode || samples[i + 15].episode != start.episode)
				continue;

			int jumpEdges = 0;
			int flipTicks = 0;
			int wallTicks = 0;
			for (size_t j = i; j < i + 16; ++j)
			{
				jumpEdges += samples[j].jumpEdge ? 1 : 0;
				flipTicks += samples[j].isFlipping ? 1 : 0;
				wallTicks += samples[j].wallContact ? 1 : 0;
			}

			if (jumpEdges < 2 && flipTicks < 2)
				continue;

			bool firstMixed = false;
			bool secondMixed = false;
			const int firstActionId = StableActionId(samples, i, 4, firstMixed);
			const int secondActionId = StableActionId(samples, i + 4, 4, secondMixed);
			if (firstActionId < 0 || secondActionId < 0 || firstMixed || secondMixed)
				continue;

			auto key = std::make_pair(firstActionId, secondActionId);
			WallDashPairStats& stats = byPair[key];
			if (stats.count == 0)
			{
				stats.firstActionId = firstActionId;
				stats.secondActionId = secondActionId;
				stats.exampleEpisode = start.episode;
				stats.exampleTick = start.physicsTick;
			}
			stats.count++;
			stats.startSpeedSum += start.speed;
			stats.endSpeedSum += samples[i + 7].speed;
			stats.speedDeltaSum += samples[i + 7].speed - start.speed;
			stats.jumpEdgeSum += jumpEdges;
			stats.flipTickSum += flipTicks;
			stats.wallTickSum += wallTicks;
		}

		std::vector<WallDashPairStats> result;
		result.reserve(byPair.size());
		for (auto& [_, stats] : byPair)
			result.push_back(stats);

		std::sort(result.begin(), result.end(), [](const WallDashPairStats& a, const WallDashPairStats& b) {
			if (a.count != b.count)
				return a.count > b.count;
			return a.speedDeltaSum > b.speedDeltaSum;
		});
		return result;
	}

	bool HasBallTouchBefore(const std::vector<TraceSample>& samples, size_t index)
	{
		const int episode = samples[index].episode;
		for (size_t i = index; i > 0; --i)
		{
			const TraceSample& sample = samples[i - 1];
			if (sample.episode != episode)
				break;
			if (sample.ballTouchedTick)
				return true;
		}
		return false;
	}

	std::vector<SequenceStats> BuildSpeedFlipSummary(
		const std::vector<TraceSample>& samples,
		int ticksPerAction,
		int parts)
	{
		// Optional analysis helper: compress early jump/flip windows into repeated
		// action sequences, useful for turning traced behavior into macro actions.
		std::map<std::vector<int>, SequenceStats> bySequence;
		const int totalTicks = ticksPerAction * parts;

		for (size_t i = 0; i + (size_t)totalTicks < samples.size(); ++i)
		{
			const TraceSample& start = samples[i];
			if (!start.jumpEdge || start.physicsTick > 140 || HasBallTouchBefore(samples, i))
				continue;
			if (samples[i + totalTicks - 1].episode != start.episode)
				continue;

			int jumpEdges = 0;
			int flipTicks = 0;
			int ballTouchTicks = 0;
			for (size_t j = i; j < i + totalTicks; ++j)
			{
				jumpEdges += samples[j].jumpEdge ? 1 : 0;
				flipTicks += samples[j].isFlipping ? 1 : 0;
				ballTouchTicks += samples[j].ballTouchedTick ? 1 : 0;
			}
			if (ballTouchTicks > 0 || flipTicks < 2)
				continue;

			std::vector<int> actionIds;
			actionIds.reserve(parts);
			bool stable = true;
			for (int part = 0; part < parts; ++part)
			{
				bool mixed = false;
				const int actionId = StableActionId(samples, i + (size_t)(part * ticksPerAction), ticksPerAction, mixed);
				if (actionId < 0 || mixed)
				{
					stable = false;
					break;
				}
				actionIds.push_back(actionId);
			}
			if (!stable)
				continue;

			SequenceStats& stats = bySequence[actionIds];
			if (stats.count == 0)
			{
				stats.actionIds = actionIds;
				stats.ticksPerAction = ticksPerAction;
				stats.exampleEpisode = start.episode;
				stats.exampleTick = start.physicsTick;
			}
			stats.count++;
			stats.startSpeedSum += start.speed;
			stats.endSpeedSum += samples[i + totalTicks - 1].speed;
			stats.speedDeltaSum += samples[i + totalTicks - 1].speed - start.speed;
			stats.jumpEdgeSum += jumpEdges;
			stats.flipTickSum += flipTicks;
			stats.ballTouchTickSum += ballTouchTicks;
		}

		std::vector<SequenceStats> result;
		result.reserve(bySequence.size());
		for (auto& [_, stats] : bySequence)
			result.push_back(stats);

		std::sort(result.begin(), result.end(), [](const SequenceStats& a, const SequenceStats& b) {
			if (a.count != b.count)
				return a.count > b.count;
			return a.speedDeltaSum > b.speedDeltaSum;
		});
		return result;
	}

	void WriteSummary(
		const std::filesystem::path& summaryPath,
		const SimulationAnalysisOptions& options,
		const Agent& sourceAgent,
		const std::vector<TraceSample>& samples,
		const std::vector<WallDashPairStats>& wallDashPairSummary,
		const std::vector<SequenceStats>& speedFlip4TickSummary,
		const std::vector<SequenceStats>& speedFlip2TickSummary,
		const std::vector<Action>& actions)
	{
		// The summary is deliberately additive: the CSV is the source of truth,
		// and these helper sections just pre-aggregate common macro-mining views.
		nlohmann::json wallDashCandidates = nlohmann::json::array();
		for (const WallDashPairStats& stats : wallDashPairSummary)
		{
			const float invCount = stats.count > 0 ? 1.f / (float)stats.count : 0.f;
			wallDashCandidates.push_back({
				{ "first_action_id", stats.firstActionId },
				{ "second_action_id", stats.secondActionId },
				{ "count", stats.count },
				{ "mean_start_speed", stats.startSpeedSum * invCount },
				{ "mean_end_speed", stats.endSpeedSum * invCount },
				{ "mean_speed_delta", stats.speedDeltaSum * invCount },
				{ "mean_jump_edges_16_ticks", stats.jumpEdgeSum * invCount },
				{ "mean_flip_ticks_16_ticks", stats.flipTickSum * invCount },
				{ "mean_wall_ticks_16_ticks", stats.wallTickSum * invCount },
				{ "example_episode", stats.exampleEpisode },
				{ "example_tick", stats.exampleTick },
				{ "first_action", ActionToJson(actions[stats.firstActionId]) },
				{ "second_action", ActionToJson(actions[stats.secondActionId]) },
			});
		}

		auto sequenceCandidatesToJson = [&](const std::vector<SequenceStats>& sequenceSummary) {
			nlohmann::json result = nlohmann::json::array();
			for (const SequenceStats& stats : sequenceSummary)
			{
				const float invCount = stats.count > 0 ? 1.f / (float)stats.count : 0.f;
				nlohmann::json actionJson = nlohmann::json::array();
				for (int actionId : stats.actionIds)
					actionJson.push_back(ActionToJson(actions[actionId]));

				result.push_back({
					{ "action_ids", stats.actionIds },
					{ "ticks_per_action", stats.ticksPerAction },
					{ "count", stats.count },
					{ "mean_start_speed", stats.startSpeedSum * invCount },
					{ "mean_end_speed", stats.endSpeedSum * invCount },
					{ "mean_speed_delta", stats.speedDeltaSum * invCount },
					{ "mean_jump_edges", stats.jumpEdgeSum * invCount },
					{ "mean_flip_ticks", stats.flipTickSum * invCount },
					{ "mean_ball_touch_ticks", stats.ballTouchTickSum * invCount },
					{ "example_episode", stats.exampleEpisode },
					{ "example_tick", stats.exampleTick },
					{ "actions", actionJson },
				});
			}
			return result;
		};

		nlohmann::json root = {
			{ "trace_mode", TraceModeName(options) },
			{ "source_agent", sourceAgent.agentName },
			{ "source_obs_builder", sourceAgent.obsBuilderName },
			{ "source_action_parser", sourceAgent.actionParserName },
			{ "source_tick_skip", sourceAgent.tickSkip },
			{ "source_action_delay", sourceAgent.actionDelay },
			{ "setup_path", options.setupPath.string() },
			{ "checkpoint_path", options.checkpointPath.string() },
			{ "csv_path", options.outPath.string() },
			{ "episodes", options.episodes },
			{ "max_ticks", options.maxTicks },
			{ "control_replay_input_delay_ticks", options.controlReplayInputDelayTicks },
			{ "control_replay_use_last_input", options.controlReplayUseLastInput },
			{ "samples", samples.size() },
			{ "wall_dash_candidate_pairs", wallDashCandidates },
			{ "speed_flip_sequences_4tick", sequenceCandidatesToJson(speedFlip4TickSummary) },
			{ "speed_flip_sequences_2tick", sequenceCandidatesToJson(speedFlip2TickSummary) },
		};

		std::ofstream out(summaryPath);
		if (!out.good())
			throw std::runtime_error("failed to open summary output: " + summaryPath.string());
		out << root.dump(4) << '\n';
	}
}