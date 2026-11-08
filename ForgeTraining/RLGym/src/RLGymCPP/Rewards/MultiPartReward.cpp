#include "MultiPartReward.h"

#include <algorithm>

using namespace std;
using namespace RLGC;

namespace {
	// Every caller runs a hook once per step on each live view, in any order.
	// A view calling again once it has already seen the current round means a new step has begun.
	bool StartsNewRound(uint64_t& ownerRound, uint64_t& viewRound) {
		const bool startsRound = viewRound == ownerRound;
		if (startsRound)
			ownerRound++;

		viewRound = ownerRound;
		return startsRound;
	}
}

string MultiPartReward::GetName() {
	if (_cachedName.empty())
		_cachedName = GetRewardTypeName(typeid(*this));

	return _cachedName;
}

Reward* MultiPartReward::Part(int partIdx) {
	shared_ptr<MultiPartReward> owner = weak_from_this().lock();
	if (!owner)
		RG_ERR_CLOSE("MultiPartReward " << GetName() << " must be created with std::make_shared before calling Part()");

	_parts = GetParts();
	if (partIdx < 0 || partIdx >= PartCount())
		RG_ERR_CLOSE("MultiPartReward " << GetName() << " has no part " << partIdx << " (it has " << PartCount() << ")");

	vector<string> names = {};
	for (const RewardPart& part : _parts) {
		string name = part.nameOverride.empty() ? GetName() + "_" + part.suffix : part.nameOverride;
		if (find(names.begin(), names.end(), name) != names.end())
			RG_ERR_CLOSE("MultiPartReward " << GetName() << " has two parts named " << name);

		names.push_back(std::move(name));
	}

	return new MultiPartRewardView(std::move(owner), partIdx, names[partIdx]);
}

void MultiPartReward::RunReset(uint64_t& viewRound, const GameState& initialState) {
	if (!StartsNewRound(_resetRound, viewRound))
		return;

	_valuesValid = false;
	Reset(initialState);
}

void MultiPartReward::RunPreStep(uint64_t& viewRound, const GameState& state) {
	if (StartsNewRound(_preStepRound, viewRound))
		PreStep(state);
}

const vector<float>& MultiPartReward::GetValues(const GameState& state, bool isFinal) {
	const int partCount = PartCount();
	const bool upToDate = _valuesValid && _valuesRound == _preStepRound && _valuesState == &state &&
		_valuesTick == state.lastTickCount && _valuesIsFinal == isFinal && _values.size() == state.players.size() * (size_t)partCount;

	if (upToDate)
		return _values;

	_values.assign(state.players.size() * partCount, 0.f);
	for (size_t playerIdx = 0; playerIdx < state.players.size(); playerIdx++)
		GetPartRewards(state.players[playerIdx], state, isFinal, span<float>(_values.data() + playerIdx * partCount, partCount));

	_valuesValid = true;
	_valuesRound = _preStepRound;
	_valuesState = &state;
	_valuesTick = state.lastTickCount;
	_valuesIsFinal = isFinal;

	return _values;
}

MultiPartRewardView::MultiPartRewardView(shared_ptr<MultiPartReward> owner, int partIdx, string name)
	: _owner(std::move(owner)), _partIdx(partIdx), _name(std::move(name)) {

	_seenResetRound = _owner->_resetRound;
	_seenPreStepRound = _owner->_preStepRound;
}

float MultiPartRewardView::GetReward(const Player& player, const GameState& state, bool isFinal) {
	const vector<float>& values = _owner->GetValues(state, isFinal);

	for (size_t playerIdx = 0; playerIdx < state.players.size(); playerIdx++)
		if (state.players[playerIdx].carId == player.carId)
			return values[playerIdx * _owner->PartCount() + _partIdx];

	RG_ERR_CLOSE("MultiPartReward part " << _name << ": no player with carId " << player.carId << " in the state");
}

vector<float> MultiPartRewardView::GetAllRewards(const GameState& state, bool isFinal) {
	vector<float> rewards;
	FillRewards(state, isFinal, rewards);

	return rewards;
}

void MultiPartRewardView::GetAllRewardsInto(const GameState& state, bool isFinal, vector<float>& out) {
	if (typeid(*this) != typeid(MultiPartRewardView)) {
		Reward::GetAllRewardsInto(state, isFinal, out);
		return;
	}

	FillRewards(state, isFinal, out);
}

void MultiPartRewardView::FillRewards(const GameState& state, bool isFinal, vector<float>& rewards) {
	const vector<float>& values = _owner->GetValues(state, isFinal);
	const int partCount = _owner->PartCount();

	rewards.resize(state.players.size());
	for (size_t playerIdx = 0; playerIdx < state.players.size(); playerIdx++)
		rewards[playerIdx] = values[playerIdx * partCount + _partIdx];

}
