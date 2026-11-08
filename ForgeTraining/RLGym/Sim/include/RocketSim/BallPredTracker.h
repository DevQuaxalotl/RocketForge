#pragma once

#include "Arena.h"

namespace RocketSim {

/**
 * Fixed-capacity ring buffer for predicted ball states.
 * External readers see a sequential view: indices [0, size()) return states
 * in chronological order starting from the most recent "now" tick.
 **/
struct BallPredBuffer {
	/** Backing storage; sized once by Reset() and never resized afterwards. **/
	std::vector<BallState> data;

	/** Index of the logical element [0] in data. **/
	size_t head = 0;

	/** Number of valid elements currently in the buffer. **/
	size_t count = 0;

	/** Slots past count are never read, so storage is only rebuilt when the capacity changes. **/
	void Reset(size_t capacity) {
		if (data.size() != capacity)
			data.assign(capacity, BallState());
		head = 0;
		count = 0;
	}

	size_t size() const     { return count; }
	size_t capacity() const { return data.size(); }
	bool   empty() const    { return count == 0; }

	const BallState& operator[](size_t i) const { return data[(head + i) % data.size()]; }
	BallState&       operator[](size_t i)       { return data[(head + i) % data.size()]; }
	const BallState& front() const { return (*this)[0]; }
	const BallState& back()  const { return (*this)[count - 1]; }

	/** Precondition: count < capacity. **/
	void Push(const BallState& s) {
		data[(head + count) % data.size()] = s;
		++count;
	}

	/** Appends n slots to write in place, or returns null when they would wrap past the end of storage. Precondition: count + n <= capacity. **/
	BallState* AppendContiguous(size_t n) {
		const size_t start = (head + count) % data.size();
		if (start + n > data.size())
			return nullptr;
		count += n;
		return data.data() + start;
	}

	/** Precondition: n <= count. **/
	void Advance(size_t n) {
		head = (head + n) % data.size();
		count -= n;
	}
};

/** Predicts the ball of an arena on its own ball-only arena of the same engine and game mode. **/
struct BallPredTracker {
	Arena* ballPredArena;
	BallPredBuffer predData;
	size_t numPredTicks;

	int lastUpdateTickCount;

	/** Earliest prediction index at which the ball is scored, or -1. **/
	int predScoreStepIndex;

	/** Creates its own ball-only copy of `arena`; the arena itself is only read. **/
	BallPredTracker(Arena* arena, size_t numPredTicks);
	~BallPredTracker();

	BallPredTracker(const BallPredTracker& other) = delete;
	BallPredTracker& operator=(const BallPredTracker& other) = delete;

	void SetNumPredTicks(size_t newNumPredTicks);

	/** Updates from the arena's current ball and tick count; doesn't need to run every tick. **/
	void UpdatePredFromArena(Arena* arena);

	/** UpdatePredFromArena without an arena: the caller supplies the ball and the ticks since the last update. **/
	void UpdatePredManual(const BallState& curBallState, int ticksSinceLastUpdate);

	/** Re-predicts every tick from `initialBallState`. **/
	void ForceUpdateAllPred(const BallState& initialBallState);

	/** Predicted ball state `predTime` seconds ahead. **/
	BallState GetBallStateForTime(float predTime) const;

private:
	/** Extends predData to numPredTicks, starting from its last state. **/
	void PredictRemainingTicks();

	/** Holds predictions that would wrap past the end of predData's storage. **/
	std::vector<BallState> _predictScratch;

	/** The Dropshot tiles last copied into ballPredArena; predicting can damage that arena's own tiles, so they aren't compared. **/
	DropshotTilesState _sentTiles;
};

}
