#include <RocketSim/GameEventTracker.h>

namespace RocketSim {

static bool GetShooterPasser(
	Arena* arena,
	Team team,
	Car*& shooterOut,
	bool findPasser,
	Car*& passerOut,
	uint64_t maxShooterTicks,
	uint64_t maxPasserTicks) {

	shooterOut = passerOut = nullptr;
	uint64_t shooterHitTick = 0;

	for (Car* car : arena->_cars) {
		if (car->team != team)
			continue;

		const BallHitInfo hit = car->GetState().ballHitInfo;
		if (!hit.isValid)
			continue;

		if (hit.tickCountWhenHit + maxShooterTicks >= arena->tickCount) {
			if (!shooterOut || hit.tickCountWhenHit > shooterHitTick) {
				shooterOut = car;
				shooterHitTick = hit.tickCountWhenHit;
			}
		}
	}

	if (shooterOut && findPasser) {
		uint64_t passerHitTick = 0;
		for (Car* car : arena->_cars) {
			if (car->team != team || car == shooterOut)
				continue;

			const BallHitInfo hit = car->GetState().ballHitInfo;
			if (!hit.isValid)
				continue;

			if (hit.tickCountWhenHit + maxPasserTicks >= shooterHitTick) {
				if (!passerOut || hit.tickCountWhenHit > passerHitTick) {
					passerOut = car;
					passerHitTick = hit.tickCountWhenHit;
				}
			}
		}
	}

	return shooterOut != nullptr;
}

void GameEventTracker::Update(Arena* arena) {
	bool scored = arena->IsBallScored();

	float tickrate = arena->GetTickRate();
	const BallState ballState = arena->ball->GetState();
	uint64_t ballUpdateCount = ballState.tickCountSinceUpdate;

	if (ballUpdateCount > _lastBallUpdateCount || !autoStateSetDetection) {
		// Game is continuing
		uint64_t deltaTicks = ballUpdateCount - _lastBallUpdateCount;
		float deltaTime = deltaTicks * arena->tickTime;

		if (scored && !_ballScoredLast) {
			Car* shooter;
			Car* passer;
			if (GetShooterPasser(
				arena,
				RS_TEAM_FROM_Y(-ballState.pos.y),
				shooter, true, passer,
				config.goalMaxTouchTime * tickrate,
				config.passMaxTouchTime * tickrate
			)) {

				if (_goalCallback.func)
					_goalCallback.func(arena, shooter, passer, _goalCallback.userInfo);
			}
		} else {
			if (!_ballShot) {

				if (_shotCooldown > 0) {
					_shotCooldown = RS_MAX(_shotCooldown - deltaTime, 0);
				} else {

					float speedSq = ballState.vel.LengthSq();
					if (speedSq >= config.shotMinSpeed * config.shotMinSpeed) {
						Team goalTeam;
						if (arena->IsBallProbablyGoingIn(config.shotMinScoreTime, config.predScoreExtraMargin, &goalTeam)) {
							Team shooterTeam = RS_OPPOSITE_TEAM(goalTeam);

							uint64_t shotMinTouchDelayTicks = config.shotTouchMinDelay * tickrate;

							Car* shooter;
							Car* passer;
							if (GetShooterPasser(
								arena,
								shooterTeam,
								shooter, true, passer,
								deltaTicks + shotMinTouchDelayTicks,
								config.passMaxTouchTime * tickrate
							)) {

								uint64_t ticksSinceHit = arena->tickCount - shooter->GetState().ballHitInfo.tickCountWhenHit;
								if (ticksSinceHit >= shotMinTouchDelayTicks) {
									_ballShot = true;
									_ballShotGoalTeam = goalTeam;
									_shotCooldown = config.shotEventCooldown;
									if (_shotCallback.func)
										_shotCallback.func(arena, shooter, passer, _shotCallback.userInfo);
								}
							}
						}
					}
				}
			} else {
				bool willScore = arena->IsBallProbablyGoingIn(config.shotMinScoreTime, config.predScoreExtraMargin);
				if (!willScore) {
					// The shot stopped going in: a save if the defending team touched it just now, otherwise a miss
					Car* saver;
					Car* unusedPasser;
					if (GetShooterPasser(
						arena,
						_ballShotGoalTeam,
						saver, false, unusedPasser,
						deltaTicks,
						0
					)) {

						if (_saveCallback.func)
							_saveCallback.func(arena, saver, _saveCallback.userInfo);
					}

					_ballShot = false;
				}
			}
		}
	} else if (ballUpdateCount == _lastBallUpdateCount) {
		return;
	} else {
		// The ball was state-set since the last update
		ResetPersistentInfo();
	}

	_ballScoredLast = scored;
	_lastBallUpdateCount = ballUpdateCount;
}

void GameEventTracker::ResetPersistentInfo() {
	_ballScoredLast = false;
	_ballShot = false;
	_shotCooldown = 0;
}

}
