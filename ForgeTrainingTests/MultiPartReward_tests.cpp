#include <gtest/gtest.h>

#include <RLGymCPP/Rewards/MultiPartReward.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>

#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace RLGC;

namespace {
	/** Two parts whose values are simple functions of the player, counting every hook it runs. */
	class CountingReward : public MultiPartReward {
	public:
		std::vector<RewardPart> parts = { { "Cont", RewardCategory::CONTINUOUS }, { "Event", RewardCategory::EVENT } };

		int resetCount = 0;
		int preStepCount = 0;
		int partRewardCalls = 0;

		std::vector<RewardPart> GetParts() const override { return parts; }

		void Reset(const GameState& initialState) override { resetCount++; }
		void PreStep(const GameState& state) override { preStepCount++; }

		void GetPartRewards(const Player& player, const GameState& state, bool isFinal, std::span<float> outPerPart) override {
			partRewardCalls++;
			outPerPart[CONT] = (float)player.carId;
			outPerPart[EVENT] = isFinal ? 100.f + (float)player.carId : -(float)player.carId;
		}

		RewardValueRange GetValueRange(int partIdx) const override {
			return partIdx == CONT ? RewardValueRange{ 0.f, 5.f } : RewardValueRange{ -1.f, 1.f };
		}

		std::vector<DebugLine> GetDebugLines(int partIdx, const Player& player, const GameState& state) const override {
			return std::vector<DebugLine>(partIdx + 1);
		}
	};

	/** Relies on the default continuous and event parts. */
	class DefaultPartsReward : public MultiPartReward {
	public:
		void GetPartRewards(const Player& player, const GameState& state, bool isFinal, std::span<float> outPerPart) override {
			outPerPart[CONT] = 1.f;
			outPerPart[EVENT] = 2.f;
		}
	};

	/** Plain-Reward twin of CountingReward's continuous part. */
	class CarIdReward : public Reward {
	public:
		float GetReward(const Player& player, const GameState& state, bool isFinal) override { return (float)player.carId; }
	};

	/** An owner plus one view per part, as a stage registers them. */
	struct Parts {
		std::shared_ptr<CountingReward> owner = std::make_shared<CountingReward>();
		std::unique_ptr<Reward> cont;
		std::unique_ptr<Reward> event;

		Parts() : cont(owner->Part(CountingReward::CONT)), event(owner->Part(CountingReward::EVENT)) {}
	};

	GameState MakeState(int playerCount, uint64_t tickCount = 0) {
		GameState state;
		state.lastTickCount = tickCount;

		for (int i = 0; i < playerCount; i++) {
			Player player = {};
			player.index = i;
			player.carId = (uint32_t)(i + 1);
			player.team = (i % 2 == 0) ? Team::BLUE : Team::ORANGE;
			state.players.push_back(player);
		}

		return state;
	}
}

TEST(MultiPartReward, PartNamesDefaultToOwnerNameAndSuffix) {
	Parts parts;

	ASSERT_EQ(parts.cont->GetName(), "CountingReward_Cont");
	ASSERT_EQ(parts.event->GetName(), "CountingReward_Event");
}

TEST(MultiPartReward, DefaultPartsAreContinuousThenEvent) {
	std::shared_ptr<DefaultPartsReward> owner = std::make_shared<DefaultPartsReward>();
	std::unique_ptr<Reward> cont(owner->Part(DefaultPartsReward::CONT));
	std::unique_ptr<Reward> event(owner->Part(DefaultPartsReward::EVENT));
	GameState state = MakeState(1);

	ASSERT_EQ(cont->GetName(), "DefaultPartsReward_Cont");
	ASSERT_EQ(event->GetName(), "DefaultPartsReward_Event");
	ASSERT_EQ(cont->GetCategory(), RewardCategory::CONTINUOUS);
	ASSERT_EQ(event->GetCategory(), RewardCategory::EVENT);
	ASSERT_EQ(cont->GetReward(state.players[0], state, false), 1.f);
	ASSERT_EQ(event->GetReward(state.players[0], state, false), 2.f);
	ASSERT_THROW(owner->Part(2), std::runtime_error);
}

TEST(MultiPartReward, NameOverrideIsUsedVerbatim) {
	std::shared_ptr<CountingReward> owner = std::make_shared<CountingReward>();
	owner->parts[CountingReward::EVENT].nameOverride = "LegacyEventReward";

	std::unique_ptr<Reward> event(owner->Part(CountingReward::EVENT));

	ASSERT_EQ(event->GetName(), "LegacyEventReward");
}

TEST(MultiPartReward, CategoryRangeAndDebugLinesArePerPart) {
	Parts parts;
	GameState state = MakeState(1);

	ASSERT_EQ(parts.cont->GetCategory(), RewardCategory::CONTINUOUS);
	ASSERT_EQ(parts.event->GetCategory(), RewardCategory::EVENT);

	ASSERT_EQ(parts.cont->GetValueRange().maxValue, 5.f);
	ASSERT_EQ(parts.event->GetValueRange().minValue, -1.f);

	ASSERT_EQ(parts.cont->GetDebugLines(state.players[0], state).size(), 1u);
	ASSERT_EQ(parts.event->GetDebugLines(state.players[0], state).size(), 2u);
}

TEST(MultiPartReward, DebugToggleIsPerPart) {
	Parts parts;

	parts.event->SetDebugEnabled(true);

	ASSERT_FALSE(parts.cont->IsDebugEnabled());
	ASSERT_TRUE(parts.event->IsDebugEnabled());
}

TEST(MultiPartReward, OwnerLivesUntilLastPartIsDeleted) {
	std::shared_ptr<CountingReward> owner = std::make_shared<CountingReward>();
	std::weak_ptr<CountingReward> watcher = owner;

	Reward* cont = owner->Part(CountingReward::CONT);
	Reward* event = owner->Part(CountingReward::EVENT);
	owner.reset();

	delete event;
	ASSERT_FALSE(watcher.expired());

	delete cont;
	ASSERT_TRUE(watcher.expired());
}

TEST(MultiPartReward, PartOutsideTheOwnersPartsFails) {
	std::shared_ptr<CountingReward> owner = std::make_shared<CountingReward>();

	ASSERT_THROW(owner->Part(2), std::runtime_error);
	ASSERT_THROW(owner->Part(-1), std::runtime_error);
}

TEST(MultiPartReward, OwnerNotHeldBySharedPtrFails) {
	CountingReward owner;

	ASSERT_THROW(owner.Part(CountingReward::CONT), std::runtime_error);
}

TEST(MultiPartReward, DuplicatePartNamesFail) {
	std::shared_ptr<CountingReward> owner = std::make_shared<CountingReward>();
	owner->parts[CountingReward::EVENT].nameOverride = "CountingReward_Cont";

	ASSERT_THROW(owner->Part(CountingReward::CONT), std::runtime_error);
}

TEST(MultiPartReward, PreStepRunsOncePerStepAcrossParts) {
	Parts parts;

	for (uint64_t tick = 1; tick <= 3; tick++) {
		GameState state = MakeState(2, tick);
		parts.cont->PreStep(state);
		parts.event->PreStep(state);
	}

	ASSERT_EQ(parts.owner->preStepCount, 3);
}

TEST(MultiPartReward, ResetRunsOncePerResetAcrossParts) {
	Parts parts;
	GameState state = MakeState(2);

	for (int episode = 0; episode < 2; episode++) {
		parts.cont->Reset(state);
		parts.event->Reset(state);
	}

	ASSERT_EQ(parts.owner->resetCount, 2);
}

TEST(MultiPartReward, HooksRunEveryStepWithOnePartLeft) {
	// Weight filtering deletes the parts a stage doesn't train.
	Parts parts;
	parts.cont.reset();
	GameState state = MakeState(2);

	parts.event->Reset(state);
	for (int step = 0; step < 3; step++)
		parts.event->PreStep(state);

	ASSERT_EQ(parts.owner->resetCount, 1);
	ASSERT_EQ(parts.owner->preStepCount, 3);
}

TEST(MultiPartReward, HookCalledAgainOnTheSameViewStartsANewStep) {
	Parts parts;
	GameState state = MakeState(2);

	parts.cont->PreStep(state);
	parts.cont->PreStep(state);
	parts.event->PreStep(state);

	ASSERT_EQ(parts.owner->preStepCount, 2);
}

TEST(MultiPartReward, PartRewardsComputedOncePerPlayerPerStep) {
	Parts parts;

	for (uint64_t tick = 1; tick <= 2; tick++) {
		GameState state = MakeState(3, tick);
		parts.cont->PreStep(state);
		parts.event->PreStep(state);

		parts.cont->GetAllRewards(state, false);
		parts.event->GetAllRewards(state, false);
		parts.event->GetReward(state.players[0], state, false);
	}

	ASSERT_EQ(parts.owner->partRewardCalls, 6);
}

TEST(MultiPartReward, EachPartReportsItsOwnValues) {
	Parts parts;
	GameState state = MakeState(3);
	parts.cont->PreStep(state);
	parts.event->PreStep(state);

	ASSERT_EQ(parts.cont->GetAllRewards(state, false), (std::vector<float>{ 1.f, 2.f, 3.f }));
	ASSERT_EQ(parts.event->GetAllRewards(state, false), (std::vector<float>{ -1.f, -2.f, -3.f }));

	// A copy of a player is matched by carId, not by address.
	const Player copy = state.players[1];
	ASSERT_EQ(parts.event->GetReward(copy, state, false), -2.f);
}

TEST(MultiPartReward, IsFinalSelectsMatchingValuesWithinAStep) {
	Parts parts;
	GameState state = MakeState(2);
	parts.event->PreStep(state);

	ASSERT_EQ(parts.event->GetAllRewards(state, false), (std::vector<float>{ -1.f, -2.f }));
	ASSERT_EQ(parts.event->GetAllRewards(state, true), (std::vector<float>{ 101.f, 102.f }));
}

TEST(MultiPartReward, ValuesRecomputeForTheSameStateObjectOnTheNextStep) {
	Parts parts;
	GameState state = MakeState(2);
	parts.cont->PreStep(state);
	parts.cont->GetAllRewards(state, false);

	state.players[0].carId = 10;
	parts.cont->PreStep(state);

	ASSERT_EQ(parts.cont->GetAllRewards(state, false), (std::vector<float>{ 10.f, 2.f }));
}

TEST(MultiPartReward, NewStateRecomputesWithoutPreStep) {
	Parts parts;
	GameState first = MakeState(2, 1);
	parts.cont->PreStep(first);
	parts.cont->GetAllRewards(first, false);

	GameState second = MakeState(2, 2);
	second.players[1].carId = 7;

	ASSERT_EQ(parts.cont->GetAllRewards(second, false), (std::vector<float>{ 1.f, 7.f }));
}

TEST(MultiPartReward, ResetInvalidatesValues) {
	Parts parts;
	GameState state = MakeState(2);
	parts.cont->Reset(state);
	parts.cont->PreStep(state);
	parts.cont->GetAllRewards(state, false);
	const int callsBeforeReset = parts.owner->partRewardCalls;

	parts.cont->Reset(state);
	parts.cont->GetAllRewards(state, false);

	ASSERT_EQ(parts.owner->partRewardCalls, callsBeforeReset + 2);
}

TEST(MultiPartReward, PlayerCountChangeOnTheSameStateRecomputes) {
	Parts parts;
	GameState state = MakeState(2);
	parts.cont->PreStep(state);
	parts.cont->GetAllRewards(state, false);

	state.players.push_back(MakeState(3).players[2]);

	ASSERT_EQ(parts.cont->GetAllRewards(state, false), (std::vector<float>{ 1.f, 2.f, 3.f }));
}

TEST(MultiPartReward, EmptyStateProducesNoValues) {
	Parts parts;
	GameState state = MakeState(0);
	parts.cont->PreStep(state);

	ASSERT_TRUE(parts.cont->GetAllRewards(state, false).empty());
}

TEST(MultiPartReward, UnknownPlayerFails) {
	Parts parts;
	GameState state = MakeState(2);
	parts.cont->PreStep(state);

	const Player stranger = MakeState(5).players[4];

	ASSERT_THROW(parts.cont->GetReward(stranger, state, false), std::runtime_error);
}

TEST(MultiPartReward, ZeroSumWrappedPartMatchesEquivalentReward) {
	std::shared_ptr<CountingReward> owner = std::make_shared<CountingReward>();
	const float teamSpirit = 0.5f;
	ZeroSumReward wrappedPart(owner->Part(CountingReward::CONT), teamSpirit);
	ZeroSumReward wrappedPlain(new CarIdReward(), teamSpirit);
	Reward& part = wrappedPart;
	Reward& plain = wrappedPlain;

	GameState state = MakeState(4);
	part.PreStep(state);
	plain.PreStep(state);

	ASSERT_EQ(part.GetAllRewards(state, false), plain.GetAllRewards(state, false));
	ASSERT_EQ(part.GetName(), "CountingReward_Cont");
}
