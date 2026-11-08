#pragma once
#include "Reward.h"

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace RLGC {

	/** One output of a MultiPartReward. */
	struct RewardPart {
		/** Reported as "<Owner>_<suffix>" unless nameOverride is set */
		std::string suffix;

		RewardCategory category = RewardCategory::CONTINUOUS;

		/** Full reported name, used verbatim; keeps an existing reward's name stable once it becomes a part */
		std::string nameOverride = {};
	};

	/**
	 * One reward with several independently weighted outputs that share its state.
	 * Each Part() view is an ordinary Reward, so every part gets its own weight, wrapper, critic head, metrics and debug row.
	 * Reset and PreStep run once per step however many parts are registered,
	 * and GetPartRewards runs once per player per step for all parts together.
	 * Must be created with std::make_shared, since every view shares ownership of it.
	 */
	class MultiPartReward : public std::enable_shared_from_this<MultiPartReward> {
	public:
		/** Indices of the default parts; rewards that override GetParts() declare their own PartIndex. */
		enum PartIndex { CONT, EVENT };

		/** The outputs, in the order GetPartRewards writes them. Must return the same parts on every call. Defaults to a continuous part and an event part. */
		virtual std::vector<RewardPart> GetParts() const {
			return { { "Cont", RewardCategory::CONTINUOUS }, { "Event", RewardCategory::EVENT } };
		}

		virtual void Reset(const GameState& initialState) {}

		virtual void PreStep(const GameState& state) {}

		/** Writes one value per part for this player, in GetParts() order. Parts left unwritten stay 0. */
		virtual void GetPartRewards(const Player& player, const GameState& state, bool isFinal, std::span<float> outPerPart) = 0;

		virtual RewardValueRange GetValueRange(int partIdx) const { return { 0.f, 1.f }; }

		virtual std::vector<DebugLine> GetDebugLines(int partIdx, const Player& player, const GameState& state) const { return {}; }

		/** The name parts are reported under; defaults to the class name, like Reward::GetName(). */
		virtual std::string GetName();

		/** A new Reward reporting only partIdx. The caller owns it, and it keeps this owner alive. */
		Reward* Part(int partIdx);

		virtual ~MultiPartReward() = default;

	private:
		friend class MultiPartRewardView;

		std::string _cachedName = {};
		std::vector<RewardPart> _parts = {};

		uint64_t _resetRound = 0;
		uint64_t _preStepRound = 0;

		/** Values indexed [playerIndex * PartCount() + partIdx], valid for the step the fields below describe */
		std::vector<float> _values = {};
		bool _valuesValid = false;
		uint64_t _valuesRound = 0;
		const GameState* _valuesState = nullptr;
		uint64_t _valuesTick = 0;
		bool _valuesIsFinal = false;

		void RunReset(uint64_t& viewRound, const GameState& initialState);
		void RunPreStep(uint64_t& viewRound, const GameState& state);
		const std::vector<float>& GetValues(const GameState& state, bool isFinal);

		int PartCount() const { return (int)_parts.size(); }
	};

	/** The Reward handed out by MultiPartReward::Part(); reports one part of its owner. */
	class MultiPartRewardView : public Reward {
	public:
		MultiPartRewardView(std::shared_ptr<MultiPartReward> owner, int partIdx, std::string name);

		void Reset(const GameState& initialState) override { _owner->RunReset(_seenResetRound, initialState); }

		void PreStep(const GameState& state) override { _owner->RunPreStep(_seenPreStepRound, state); }

		float GetReward(const Player& player, const GameState& state, bool isFinal) override;

		std::vector<float> GetAllRewards(const GameState& state, bool isFinal) override;

		void GetAllRewardsInto(const GameState& state, bool isFinal, std::vector<float>& out) override;

		std::string GetName() override { return _name; }

		RewardCategory GetCategory() const override { return _owner->_parts[_partIdx].category; }

		RewardValueRange GetValueRange() const override { return _owner->GetValueRange(_partIdx); }

		std::vector<DebugLine> GetDebugLines(const Player& player, const GameState& state) const override {
			return _owner->GetDebugLines(_partIdx, player, state);
		}

	private:
		void FillRewards(const GameState& state, bool isFinal, std::vector<float>& out);

		std::shared_ptr<MultiPartReward> _owner;
		int _partIdx;
		std::string _name;

		/** The owner's Reset / PreStep rounds this view last saw; see StartsNewRound in MultiPartReward.cpp */
		uint64_t _seenResetRound;
		uint64_t _seenPreStepRound;
	};
}
