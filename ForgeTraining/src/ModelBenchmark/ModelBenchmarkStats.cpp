#include "ModelBenchmarkInternal.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace RocketForgeTraining::ModelBenchmarkInternal
{
	namespace
	{
		constexpr double BenchmarkGameSeconds = 5.0 * 60.0;
		constexpr float MinFlipResetHeight = 200.0f;
		constexpr float MaxFlipResetBallDistance = 130.0f;
		constexpr uint64_t MaxPassTouchTicks = 2 * 120;
		constexpr float MinPassDistance = 500.0f;
		constexpr float MinPassDistanceSq = MinPassDistance * MinPassDistance;
		constexpr float HighAirMinHeight = RLGC::CommonValues::GOAL_HEIGHT;

		const RLGC::Player* FindPlayer(const RLGC::GameState& state, uint32_t carId) {
			auto itr = std::find_if(
				state.players.begin(),
				state.players.end(),
				[carId](const RLGC::Player& player) { return player.carId == carId; });
			return itr != state.players.end() ? &*itr : nullptr;
		}

		bool AcquiredFlipResetFromBall(
			const RLGC::Player& player,
			const RLGC::GameState& state,
			const RLGC::GameState& prev) {
			const RLGC::Player* previous = FindPlayer(prev, player.carId);
			if (!previous
				|| !player.ballTouchedStep
				|| !player.GotFlipReset()
				|| previous->GotFlipReset())
				return false;

			return player.pos.z >= MinFlipResetHeight
				&& (state.ball.pos - player.pos).Length() <= MaxFlipResetBallDistance;
		}

		bool StartedBallTouch(
			const RLGC::Player& player,
			const RLGC::GameState& prev) {
			if (!player.ballTouchedStep)
				return false;
			const RLGC::Player* previous = FindPlayer(prev, player.carId);
			return !previous || !previous->ballTouchedTick;
		}

		bool IsOpponentSide(Team team, float y) {
			return team == Team::BLUE ? y > 0.0f : y < 0.0f;
		}

		void AccumulateBoostPickupStats(RunningStats& stats, const RLGC::GameState& state, const RLGC::GameState& prev, Team team) {
			if (!state.lastArena || state.boostPads.empty() || prev.boostPads.empty())
				return;

			const int padCount = std::min((int)state.boostPads.size(), (int)prev.boostPads.size());
			for (int padIdx = 0; padIdx < padCount; padIdx++) {
				if (!prev.boostPads[padIdx] || state.boostPads[padIdx])
					continue;
				if (padIdx >= (int)state.boostPadIndexMap.size())
					continue;

				const int arenaPadIdx = state.boostPadIndexMap[padIdx];
				if (arenaPadIdx < 0 || arenaPadIdx >= (int)state.lastArena->_boostPads.size())
					continue;

				const BoostPad* pad = state.lastArena->_boostPads[arenaPadIdx];
				const RLGC::Player* closest = nullptr;
				float closestDist = std::numeric_limits<float>::max();
				for (const RLGC::Player& player : state.players) {
					float dist = (player.pos - pad->config.pos).Length();
					if (dist < closestDist) {
						closestDist = dist;
						closest = &player;
					}
				}
				if (!closest || closest->team != team)
					continue;

				if (pad->config.isBig) {
					stats.bigBoostPads += 1.0;
					if (IsOpponentSide(team, pad->config.pos.y))
						stats.boostSteals += 1.0;
				} else {
					stats.smallBoostPads += 1.0;
				}
			}
		}
	}

	BenchmarkPassCounts AccumulateCompletedPasses(
		BenchmarkPassTracker& tracker,
		std::vector<BenchmarkPassTouch> touches) {
		BenchmarkPassCounts counts = {};
		std::sort(touches.begin(), touches.end(), [](const auto& a, const auto& b) {
			if (a.tick != b.tick)
				return a.tick < b.tick;
			return a.carId < b.carId;
		});

		for (size_t first = 0; first < touches.size();) {
			const uint64_t tick = touches[first].tick;
			size_t end = first + 1;
			while (end < touches.size() && touches[end].tick == tick)
				end++;

			if (tracker.hasProcessedTouchTick && tick <= tracker.processedThroughTick) {
				first = end;
				continue;
			}

			tracker.hasProcessedTouchTick = true;
			tracker.processedThroughTick = tick;
			if (end - first != 1) {
				// Same-tick contacts are contested, so they break rather than extend possession.
				tracker.hasLastUncontestedTouch = false;
				first = end;
				continue;
			}

			const BenchmarkPassTouch& touch = touches[first];
			if (tracker.hasLastUncontestedTouch) {
				const BenchmarkPassTouch& previous = tracker.lastUncontestedTouch;
				const uint64_t elapsedTicks = touch.tick - previous.tick;
				const bool completedPass = touch.carId != previous.carId
					&& touch.team == previous.team
					&& elapsedTicks <= MaxPassTouchTicks
					&& touch.ballPosition.DistSq(previous.ballPosition) >= MinPassDistanceSq;
				if (completedPass) {
					if (previous.team == Team::BLUE)
						counts.blue += 1.0;
					else
						counts.orange += 1.0;
				}
			}

			tracker.hasLastUncontestedTouch = true;
			tracker.lastUncontestedTouch = touch;
			first = end;
		}
		return counts;
	}

	BenchmarkPassCounts AccumulateCompletedPasses(
		BenchmarkPassTracker& tracker,
		const RLGC::GameState& state) {
		std::vector<BenchmarkPassTouch> touches;
		if (!state.lastArena)
			return {};
		touches.reserve(state.lastArena->_cars.size());
		for (Car* car : state.lastArena->_cars) {
			const BallHitInfo hit = car->GetState().ballHitInfo;
			if (!hit.isValid)
				continue;
			touches.push_back({ car->id, car->team, hit.tickCountWhenHit, hit.ballPos });
		}
		return AccumulateCompletedPasses(tracker, std::move(touches));
	}

	void AccumulateStepStats(RunningStats& stats, const RLGC::GameState& state, const RLGC::GameState& prev, Team team, float deltaSeconds) {
		int players = 0;
		double ballInOpponentHalf = 0.5;
		const float teamRelativeBallY = team == Team::BLUE ? state.ball.pos.y : -state.ball.pos.y;
		if (std::abs(teamRelativeBallY) > 1e-3f)
			ballInOpponentHalf = teamRelativeBallY > 0.0f ? 1.0 : 0.0;
		const float thirdBoundary = state.GetYWall() / 3.0f;

		AccumulateBoostPickupStats(stats, state, prev, team);
		for (const RLGC::Player& player : state.players) {
			const bool startedBallTouch = StartedBallTouch(player, prev);
			if (player.team != team) {
				stats.goalsConceded += player.eventState.goal ? 1.0 : 0.0;
				stats.opponentBallTouches += startedBallTouch ? 1.0 : 0.0;
				continue;
			}
			players++;
			stats.goals += player.eventState.goal ? 1.0 : 0.0;
			stats.shots += player.eventState.shot ? 1.0 : 0.0;
			stats.assists += player.eventState.assist ? 1.0 : 0.0;
			stats.saves += player.eventState.save ? 1.0 : 0.0;
			stats.ballTouches += startedBallTouch ? 1.0 : 0.0;
			stats.flipResets += AcquiredFlipResetFromBall(player, state, prev) ? 1.0 : 0.0;
			stats.bumps += player.eventState.bump ? 1.0 : 0.0;
			stats.demos += player.eventState.demo ? 1.0 : 0.0;
			const RLGC::Player* previous = FindPlayer(prev, player.carId);
			if (previous
				&& !previous->isDemoed
				&& !player.isDemoed
				&& player.tickCountSinceUpdate > previous->tickCountSinceUpdate) {
				// State resets and respawns can lower boost without consuming it, so only
				// accumulate depletion across a continuous car-state timeline.
				stats.boostConsumed += std::max(0.0f, previous->boost - player.boost);
			}
			stats.boostSum += std::clamp((double)player.boost, 0.0, 100.0);
			stats.speedSum += player.vel.Length();
			if (!player.isOnGround) {
				stats.aerialTimeSum += deltaSeconds;
				if (player.pos.z >= HighAirMinHeight)
					stats.highAerialTimeSum += deltaSeconds;
				else
					stats.lowAerialTimeSum += deltaSeconds;
			}
			stats.lowBoostTimeSum += player.boost <= 20.0f ? deltaSeconds : 0.0;
			stats.supersonicTimeSum += player.isSupersonic ? deltaSeconds : 0.0;
			stats.playerTimeSum += deltaSeconds;
			stats.speedTowardsBallSum += std::max(0.0f, player.vel.Dot((state.ball.pos - player.pos).Normalized()));
			stats.distanceToBallSum += (state.ball.pos - player.pos).Length();
			stats.playerSamples++;
			if (player.ballTouchedStep) {
				stats.touchHeightSum += state.ball.pos.z;
				stats.touchCount += 1.0;
			}
		}
		if (players > 0) {
			stats.gameTimeSum += std::max(0.0f, deltaSeconds);
			if (players > 1)
				stats.passGameTimeSum += std::max(0.0f, deltaSeconds);
			stats.opponentHalfBallSteps += ballInOpponentHalf;
			if (teamRelativeBallY > thirdBoundary)
				stats.offensiveThirdBallSteps += 1.0;
			else if (teamRelativeBallY < -thirdBoundary)
				stats.defensiveThirdBallSteps += 1.0;
			else
				stats.neutralThirdBallSteps += 1.0;
			stats.stepSamples++;
		}
	}

	void MergeRunningStats(RunningStats& destination, const RunningStats& source) {
		destination.definitionVersion = std::min(destination.definitionVersion, source.definitionVersion);
		destination.goals += source.goals;
		destination.shots += source.shots;
		destination.assists += source.assists;
		destination.saves += source.saves;
		destination.goalsConceded += source.goalsConceded;
		destination.passes += source.passes;
		destination.passGameTimeSum += source.passGameTimeSum;
		destination.ballTouches += source.ballTouches;
		destination.opponentBallTouches += source.opponentBallTouches;
		destination.flipResets += source.flipResets;
		destination.bumps += source.bumps;
		destination.demos += source.demos;
		destination.boostSteals += source.boostSteals;
		destination.bigBoostPads += source.bigBoostPads;
		destination.smallBoostPads += source.smallBoostPads;
		destination.boostConsumed += source.boostConsumed;
		destination.gameTimeSum += source.gameTimeSum;
		destination.boostSum += source.boostSum;
		destination.speedSum += source.speedSum;
		destination.aerialTimeSum += source.aerialTimeSum;
		destination.lowAerialTimeSum += source.lowAerialTimeSum;
		destination.highAerialTimeSum += source.highAerialTimeSum;
		destination.lowBoostTimeSum += source.lowBoostTimeSum;
		destination.supersonicTimeSum += source.supersonicTimeSum;
		destination.playerTimeSum += source.playerTimeSum;
		destination.opponentHalfBallSteps += source.opponentHalfBallSteps;
		destination.offensiveThirdBallSteps += source.offensiveThirdBallSteps;
		destination.neutralThirdBallSteps += source.neutralThirdBallSteps;
		destination.defensiveThirdBallSteps += source.defensiveThirdBallSteps;
		destination.touchHeightSum += source.touchHeightSum;
		destination.touchCount += source.touchCount;
		destination.speedTowardsBallSum += source.speedTowardsBallSum;
		destination.distanceToBallSum += source.distanceToBallSum;
		destination.stepSamples += source.stepSamples;
		destination.playerSamples += source.playerSamples;
	}

	ModelBenchmarkOptionalStats FinalizeStats(const RunningStats& stats) {
		ModelBenchmarkOptionalStats out = {};
		out.expandedStatsAvailable = stats.definitionVersion >=
			MODEL_BENCHMARK_EXPANDED_STAT_DEFINITION_VERSION;
		out.efficiencyStatsAvailable = stats.definitionVersion >=
			MODEL_BENCHMARK_STAT_DEFINITION_VERSION;
		out.equivalentGames = stats.gameTimeSum / BenchmarkGameSeconds;
		const auto perGame = [equivalentGames = out.equivalentGames](double total) {
			return equivalentGames > 0.0 ? total / equivalentGames : 0.0;
		};
		out.goals = perGame(stats.goals);
		out.shots = perGame(stats.shots);
		if (stats.shots > 0.0)
			out.shotConversionPercent = 100.0 * stats.goals / stats.shots;
		out.assists = perGame(stats.assists);
		if (stats.goals > 0.0)
			out.assistedGoalPercent = 100.0 * stats.assists / stats.goals;
		out.saves = perGame(stats.saves);
		const double shotsOnTargetAgainst = stats.saves + stats.goalsConceded;
		if (out.efficiencyStatsAvailable && shotsOnTargetAgainst > 0.0)
			out.savePercent = 100.0 * stats.saves / shotsOnTargetAgainst;
		if (out.expandedStatsAvailable && stats.passGameTimeSum > 0.0) {
			out.passes = stats.passes / (stats.passGameTimeSum / BenchmarkGameSeconds);
			out.passesAvailable = true;
		}
		out.ballTouches = perGame(stats.ballTouches);
		const double allBallTouches = stats.ballTouches + stats.opponentBallTouches;
		if (out.efficiencyStatsAvailable && allBallTouches > 0.0)
			out.touchSharePercent = 100.0 * stats.ballTouches / allBallTouches;
		out.flipResets = perGame(stats.flipResets);
		out.bumps = perGame(stats.bumps);
		out.demos = perGame(stats.demos);
		out.boostSteals = perGame(stats.boostSteals);
		out.bigBoostPads = perGame(stats.bigBoostPads);
		out.smallBoostPads = perGame(stats.smallBoostPads);
		out.samples = stats.stepSamples;
		if (stats.stepSamples > 0) {
			out.fieldDomination = 100.0 * stats.opponentHalfBallSteps / stats.stepSamples;
			out.offensiveThirdPercent = 100.0 * stats.offensiveThirdBallSteps / stats.stepSamples;
			out.neutralThirdPercent = 100.0 * stats.neutralThirdBallSteps / stats.stepSamples;
			out.defensiveThirdPercent = 100.0 * stats.defensiveThirdBallSteps / stats.stepSamples;
		}
		if (stats.playerSamples > 0) {
			out.averageBoost = stats.boostSum / stats.playerSamples;
			out.averageSpeed = stats.speedSum / stats.playerSamples;
			out.averageSpeedTowardsBall = stats.speedTowardsBallSum / stats.playerSamples;
			out.averageDistanceToBall = stats.distanceToBallSum / stats.playerSamples;
		}
		if (stats.playerTimeSum > 0) {
			if (out.efficiencyStatsAvailable)
				out.boostConsumptionPerMinute = 60.0 * stats.boostConsumed / stats.playerTimeSum;
			out.aerialTimePercent = 100.0 * stats.aerialTimeSum / stats.playerTimeSum;
			out.lowAirTimePercent = 100.0 * stats.lowAerialTimeSum / stats.playerTimeSum;
			out.highAirTimePercent = 100.0 * stats.highAerialTimeSum / stats.playerTimeSum;
			out.lowBoostTimePercent = 100.0 * stats.lowBoostTimeSum / stats.playerTimeSum;
			out.averageSupersonicTime = 100.0 * stats.supersonicTimeSum / stats.playerTimeSum;
		}
		if (stats.touchCount > 0)
			out.averageTouchHeight = stats.touchHeightSum / stats.touchCount;
		return out;
	}
}
