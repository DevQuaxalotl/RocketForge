#pragma once

#include "Arena.h"

namespace RocketSim {

typedef std::function<void(Arena* arena, Car* shooter, Car* passer, void* userInfo)> ShotEventFn;
typedef std::function<void(Arena* arena, Car* scorer, Car* passer, void* userInfo)> GoalEventFn;
typedef std::function<void(Arena* arena, Car* saver, void* userInfo)> SaveEventFn;

/** Thresholds for GameEventTracker. They are not the game's own values. **/
struct GameEventTrackerConfig {
	/** Minimum ball speed toward the net (uu/s) for a shot; 2300 is car max speed. **/
	float shotMinSpeed = 1750;

	/** Minimum time since the shooter's touch for a shot to count. **/
	float shotTouchMinDelay = 0.3f;

	/** Extra goal margin for the scoring prediction; negative makes it stricter. **/
	float predScoreExtraMargin = 0;

	/** Minimum time between shot events, so repeated blocked shots can't be farmed. **/
	float shotEventCooldown = 1.0f;

	/** Maximum predicted time to the goal for a shot to count. **/
	float shotMinScoreTime = 2.0f;

	/** Maximum time between the scorer's touch and the goal. **/
	float goalMaxTouchTime = 4.0f;

	/** Maximum time between the passer's touch and the shooter's. **/
	float passMaxTouchTime = 2.0f;
};

/**
 * Tracks shots, saves, goals and assists, firing the registered callback when Update() sees one.
 * Bumps and demos aren't tracked here; Arena::SetCarBumpCallback reports them.
 */
struct GameEventTracker {
	GameEventTrackerConfig config = {};

	struct {
		ShotEventFn func = nullptr;
		void* userInfo = nullptr;
	} _shotCallback;
	void SetShotCallback(ShotEventFn callbackFn, void* userInfo = nullptr) {
		_shotCallback.func = callbackFn;
		_shotCallback.userInfo = userInfo;
	}

	struct {
		GoalEventFn func = nullptr;
		void* userInfo = nullptr;
	} _goalCallback;
	void SetGoalCallback(GoalEventFn callbackFn, void* userInfo = nullptr) {
		_goalCallback.func = callbackFn;
		_goalCallback.userInfo = userInfo;
	}

	struct {
		SaveEventFn func = nullptr;
		void* userInfo = nullptr;
	} _saveCallback;
	void SetSaveCallback(SaveEventFn callbackFn, void* userInfo = nullptr) {
		_saveCallback.func = callbackFn;
		_saveCallback.userInfo = userInfo;
	}

	float _shotCooldown = 0;
	bool _ballShot = false;
	Team _ballShotGoalTeam = {};
	bool _ballScoredLast = false;

	/** Ball tickCountSinceUpdate at the last Update(): a decrease means the ball was state-set, no change skips the update. **/
	uint64_t _lastBallUpdateCount = 0;

	/** Call often enough not to miss events; with tick skip, once per step. **/
	void Update(Arena* arena);

	/** Clears persistent info when the ball is state-set; turn off to call ResetPersistentInfo() yourself. **/
	bool autoStateSetDetection = true;

	/** Clears the info kept between updates. **/
	void ResetPersistentInfo();
};

}
