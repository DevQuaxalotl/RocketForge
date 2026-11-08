#include <RocketSim/BallPredTracker.h>

#include "Backend/ArenaBackend.h"

#include <cstring>

namespace RocketSim {

BallPredTracker::BallPredTracker(Arena* arena, size_t numPredTicks) : numPredTicks(numPredTicks) {
	ballPredArena = Arena::Create(arena->gameMode, arena->GetArenaConfig(), arena->GetTickRate());

	lastUpdateTickCount = 0;
	predScoreStepIndex = -1;

	predData.Reset(numPredTicks);
	UpdatePredFromArena(arena);
}

BallPredTracker::~BallPredTracker() {
	delete ballPredArena;
}

void BallPredTracker::SetNumPredTicks(size_t newNumPredTicks) {
	if (newNumPredTicks == numPredTicks)
		return;

	numPredTicks = newNumPredTicks;
	predScoreStepIndex = -1;

	if (!predData.empty()) {
		BallState seed = predData[0];
		ForceUpdateAllPred(seed);
	} else {
		// Storage only; the next update fills it.
		predData.Reset(numPredTicks);
	}
}

void BallPredTracker::UpdatePredFromArena(Arena* arena) {
	BallState ballState = arena->ball->GetState();

	if (arena->gameMode == GameMode::DROPSHOT) {
		const DropshotTilesState tiles = arena->GetDropshotTilesState();
		if (std::memcmp(&tiles, &_sentTiles, sizeof(DropshotTilesState)) != 0) {
			ballPredArena->SetDropshotTilesState(tiles);
			_sentTiles = tiles;
			ForceUpdateAllPred(ballState);
			lastUpdateTickCount = arena->tickCount;
			return;
		}
	}

	int ticksSinceLastUpdate = arena->tickCount - lastUpdateTickCount;
	UpdatePredManual(ballState, ticksSinceLastUpdate);
}

void BallPredTracker::UpdatePredManual(const BallState& curBallState, int ticksSinceLastUpdate) {
	bool needsFullRepred;
	if (ticksSinceLastUpdate == 0) {
		// On v2 a same-tick ball set leaves the prediction stale until the next tick,
		// which is what v2-trained models have always observed.
		// Otherwise nothing has been simulated since the last update, so a different ball means it was set.
		if (ballPredArena->GetEngine() == EngineId::V2)
			needsFullRepred = predData.empty();
		else
			needsFullRepred = predData.empty() || !predData[0].Matches(curBallState);
	} else if (static_cast<size_t>(ticksSinceLastUpdate) < predData.size()) {
		if (predData[ticksSinceLastUpdate].Matches(curBallState)) {
			needsFullRepred = false;
			predData.Advance(static_cast<size_t>(ticksSinceLastUpdate));

			if (predScoreStepIndex >= 0) {
				predScoreStepIndex -= ticksSinceLastUpdate;
				if (predScoreStepIndex < 0)
					predScoreStepIndex = -1;
			}

			PredictRemainingTicks();
		} else {
			needsFullRepred = true;
		}
	} else {
		needsFullRepred = true;
	}

	if (needsFullRepred)
		ForceUpdateAllPred(curBallState);

	lastUpdateTickCount += ticksSinceLastUpdate;
}

void BallPredTracker::ForceUpdateAllPred(const BallState& initialBallState) {
	ballPredArena->ball->SetState(initialBallState);

	predData.Reset(numPredTicks);
	predScoreStepIndex = -1;

	predData.Push(initialBallState);

	if (ballPredArena->IsBallScored())
		predScoreStepIndex = 0;

	PredictRemainingTicks();
}

void BallPredTracker::PredictRemainingTicks() {
	if (predData.size() >= numPredTicks)
		return;

	const size_t firstNewIndex = predData.size();
	const int numNewTicks = static_cast<int>(numPredTicks - firstNewIndex);
	const BallState start = predData.back();

	int firstScoringTick;
	if (BallState* inPlace = predData.AppendContiguous(numNewTicks)) {
		firstScoringTick = ballPredArena->_backend->PredictBall(start, numNewTicks, inPlace);
	} else {
		_predictScratch.resize(numNewTicks);
		firstScoringTick = ballPredArena->_backend->PredictBall(start, numNewTicks, _predictScratch.data());
		for (const BallState& state : _predictScratch)
			predData.Push(state);
	}

	// IsBallScored() is positional and the prediction arena never resets on a goal,
	// so a scored ball can bounce back out of the net; only the earliest crossing counts.
	if (firstScoringTick >= 0 && predScoreStepIndex < 0)
		predScoreStepIndex = static_cast<int>(firstNewIndex) + firstScoringTick;
}

BallState BallPredTracker::GetBallStateForTime(float predTime) const {
	if (predData.empty())
		RS_ERR_CLOSE("BallPredTracker::GetBallStateForTime(): Predicted ball data is empty, update prediction before calling");

	int index = RS_CLAMP(predTime / ballPredArena->tickTime, 0, predData.size() - 1);
	return predData[index];
}

}
