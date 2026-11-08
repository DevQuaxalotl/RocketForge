#pragma once
#include "PerPlayerReward.h"
#include "MultiPartReward.h"
#include "../Math.h"

#include <cmath>

namespace RLGC {

	template<bool PlayerEventState::* VAR, bool NEGATIVE>
	class PlayerDataEventReward : public PerPlayerReward<PlayerDataEventReward<VAR, NEGATIVE>> {
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			bool val =  player.eventState.*VAR;

			if (NEGATIVE) {
				return -(float)val;
			} else {
				return (float)val;
			}
		}

		RewardCategory GetCategory() const override { return RewardCategory::EVENT; }
		RewardValueRange GetValueRange() const override { return NEGATIVE ? RewardValueRange{ -1.f, 0.f } : RewardValueRange{ 0.f, 1.f }; }

	public:
		/**
		 * Explicit, compiler-independent name. The default Reward::GetName() derives from
		 * typeid(*this).name(), which renders this template differently on MSVC (e.g.
		 * "PlayerDataEventReward<0,0>") than on GCC/Clang.
		 **/
		std::string GetName() override {
			const char* field = "Unknown";
			if      (VAR == &PlayerEventState::goal)     field = "Goal";
			else if (VAR == &PlayerEventState::save)     field = "Save";
			else if (VAR == &PlayerEventState::assist)   field = "Assist";
			else if (VAR == &PlayerEventState::shot)     field = "Shot";
			else if (VAR == &PlayerEventState::shotPass) field = "ShotPass";
			else if (VAR == &PlayerEventState::bump)     field = "Bump";
			else if (VAR == &PlayerEventState::bumped)   field = "Bumped";
			else if (VAR == &PlayerEventState::demo)     field = "Demo";
			else if (VAR == &PlayerEventState::demoed)   field = "Demoed";

			return std::string("PlayerEvent") + field + (NEGATIVE ? "Penalty" : "Reward");
		}
	};

	typedef PlayerDataEventReward<&PlayerEventState::goal, false> PlayerGoalReward; // NOTE: Given only to the player who last touched the ball on the opposing team
	typedef PlayerDataEventReward<&PlayerEventState::assist, false> AssistReward;
	typedef PlayerDataEventReward<&PlayerEventState::shot, false> ShotReward;
	typedef PlayerDataEventReward<&PlayerEventState::shotPass, false> ShotPassReward;
	typedef PlayerDataEventReward<&PlayerEventState::save, false> SaveReward;
	typedef PlayerDataEventReward<&PlayerEventState::bump, false> BumpReward;
	typedef PlayerDataEventReward<&PlayerEventState::bumped, true> BumpedPenalty;
	typedef PlayerDataEventReward<&PlayerEventState::demo, false> DemoReward;
	typedef PlayerDataEventReward<&PlayerEventState::demoed, true> DemoedPenalty;

	// Rewards a goal by anyone on the team
	// NOTE: Already zero-sum
	class GoalReward : public PerPlayerReward<GoalReward> {
	public:
		float concedeScale;
		GoalReward(float concedeScale = -1) : concedeScale(concedeScale) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			if (!state.goalScored)
				return 0;

			bool scored = (player.team != RS_TEAM_FROM_Y(state.ball.pos.y));
			return scored ? 1 : concedeScale;
		}

		RewardCategory GetCategory() const override { return RewardCategory::EVENT; }
		RewardValueRange GetValueRange() const override { return { concedeScale, 1.f }; }
	};

	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/common_rewards/misc_rewards.py
	class VelocityReward : public PerPlayerReward<VelocityReward> {
	public:
		bool isNegative;
		VelocityReward(bool isNegative = false) : isNegative(isNegative) {}
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			return player.vel.Length() / CommonValues::CAR_MAX_SPEED * (1 - 2 * isNegative);
		}

		RewardValueRange GetValueRange() const override { return isNegative ? RewardValueRange{ -1.f, 0.f } : RewardValueRange{ 0.f, 1.f }; }
	};

	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/common_rewards/ball_goal_rewards.py
	class VelocityBallToGoalReward : public PerPlayerReward<VelocityBallToGoalReward> {
	public:
		bool ownGoal = false;
		VelocityBallToGoalReward(bool ownGoal = false) : ownGoal(ownGoal) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			bool targetOrangeGoal = player.team == Team::BLUE;
			if (ownGoal)
				targetOrangeGoal = !targetOrangeGoal;

			Vec targetPos = targetOrangeGoal ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;

			Vec ballDirToGoal = (targetPos - state.ball.pos).Normalized();
			return ballDirToGoal.Dot(state.ball.vel / CommonValues::BALL_MAX_SPEED);
		}

		RewardValueRange GetValueRange() const override { return { -1.f, 1.f }; }

		std::vector<DebugLine> GetDebugLines(const Player& player, const GameState& state) const override {
			bool targetOrangeGoal = player.team == Team::BLUE;
			if (ownGoal)
				targetOrangeGoal = !targetOrangeGoal;
			Vec targetPos = targetOrangeGoal ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;

			// Draw line from ball toward target goal
			Vec ballDirToGoal = (targetPos - state.ball.pos).Normalized();
			float rewardVal = ballDirToGoal.Dot(state.ball.vel / CommonValues::BALL_MAX_SPEED);
			float g = RS_MAX(0, rewardVal);
			float r = RS_MAX(0, -rewardVal);
			return { { state.ball.pos, targetPos, Vec(r, g, 0) } };
		}
	};

	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/common_rewards/player_ball_rewards.py
	class VelocityPlayerToBallReward : public PerPlayerReward<VelocityPlayerToBallReward> {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			Vec dirToBall = (state.ball.pos - player.pos).Normalized();
			Vec normVel = player.vel / CommonValues::CAR_MAX_SPEED;
			return dirToBall.Dot(normVel);
		}

		RewardValueRange GetValueRange() const override { return { -1.f, 1.f }; }

		std::vector<DebugLine> GetDebugLines(const Player& player, const GameState& state) const override {
			Vec dirToBall = (state.ball.pos - player.pos).Normalized();
			float rewardVal = dirToBall.Dot(player.vel / CommonValues::CAR_MAX_SPEED);
			// Green when positive, red when negative
			float g = RS_MAX(0, rewardVal);
			float r = RS_MAX(0, -rewardVal);
			return { { player.pos, state.ball.pos, Vec(r, g, 0) } };
		}
	};

	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/common_rewards/player_ball_rewards.py
	class FaceBallReward : public PerPlayerReward<FaceBallReward> {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			Vec dirToBall = (state.ball.pos - player.pos).Normalized();
			return player.rotMat.forward.Dot(dirToBall);
		}

		RewardValueRange GetValueRange() const override { return { -1.f, 1.f }; }

		std::vector<DebugLine> GetDebugLines(const Player& player, const GameState& state) const override {
			// Draw the car's forward direction as a line
			Vec lineEnd = player.pos + player.rotMat.forward * 500.f;
			float rewardVal = player.rotMat.forward.Dot((state.ball.pos - player.pos).Normalized());
			float g = RS_MAX(0, rewardVal);
			float r = RS_MAX(0, -rewardVal);
			return { { player.pos, lineEnd, Vec(r, g, 0) } };
		}
	};

	class TouchBallReward : public PerPlayerReward<TouchBallReward> {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			return player.ballTouchedStep;
		}

		RewardCategory GetCategory() const override { return RewardCategory::EVENT; }
	};

	class SpeedReward : public PerPlayerReward<SpeedReward> {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			return player.vel.Length() / CommonValues::CAR_MAX_SPEED;
		}
	};

	class WavedashReward : public PerPlayerReward<WavedashReward> {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			if (!player.prev)
				return 0;

			if (player.isOnGround && (player.prev->isFlipping && !player.prev->isOnGround)) {
				return 1;
			} else {
				return 0;
			}
		}

		RewardCategory GetCategory() const override { return RewardCategory::EVENT; }
	};

	class PickupBoostReward : public PerPlayerReward<PickupBoostReward> {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!player.prev)
				return 0;

			if (player.boost > player.prev->boost) {
				return sqrtf(player.boost / 100.f) - sqrtf(player.prev->boost / 100.f);
			} else {
				return 0;
			}
		}

		RewardCategory GetCategory() const override { return RewardCategory::EVENT; }
	};

	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/common_rewards/misc_rewards.py
	class SaveBoostReward : public PerPlayerReward<SaveBoostReward> {
	public:
		float exponent;
		SaveBoostReward(float exponent = 0.5f) : exponent(exponent) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			return RS_CLAMP(powf(player.boost / 100, exponent), 0, 1);
		}
	};


	class AirReward : public PerPlayerReward<AirReward> {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			return !player.isOnGround;
		}
	};

	// Mostly based on the classic Necto rewards
	// Total reward output for speeding the ball up to MAX_REWARDED_BALL_SPEED is 1.0
	// The bot can do this slowly (putting) or quickly (shooting)
	class TouchAccelReward : public PerPlayerReward<TouchAccelReward> {
	public:
		constexpr static float MAX_REWARDED_BALL_SPEED = RLGC::Math::KPHToVel(110);

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!state.prev)
				return 0;

			if (player.ballTouchedStep) {
				float prevSpeedFrac = RS_MIN(1, state.prev->ball.vel.Length() / MAX_REWARDED_BALL_SPEED);
				float curSpeedFrac = RS_MIN(1, state.ball.vel.Length() / MAX_REWARDED_BALL_SPEED);

				if (curSpeedFrac > prevSpeedFrac) {
					return (curSpeedFrac - prevSpeedFrac);
				} else {
					// Not speeding up the ball so we don't care
					return 0;
				}
			} else {
				return 0;
			}
		}

		RewardCategory GetCategory() const override { return RewardCategory::EVENT; }
	};

	class StrongTouchReward : public PerPlayerReward<StrongTouchReward> {
	public:
		float minRewardedVel, maxRewardedVel;
		StrongTouchReward(float minSpeedKPH = 20, float maxSpeedKPH = 130) {
			minRewardedVel = RLGC::Math::KPHToVel(minSpeedKPH);
			maxRewardedVel = RLGC::Math::KPHToVel(maxSpeedKPH);
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!state.prev)
				return 0;

			if (player.ballTouchedStep) {
				float hitForce = (state.ball.vel - state.prev->ball.vel).Length();
				if (hitForce < minRewardedVel)
					return 0;

				return RS_MIN(1, hitForce / maxRewardedVel);
			} else {
				return 0;
			}
		}

		RewardCategory GetCategory() const override { return RewardCategory::EVENT; }
	};

	/**
	 * Rewards boost on one concave curve, so low-boost amounts are worth relatively more.
	 * Cont pays for boost held each step; Event pays for the boost collected this step.
	 * See Documentation/Features/Rewards.md for how multi-part rewards are registered and weighted.
	 */
	class BoostReward : public MultiPartReward {
	public:
		float exponent;

		BoostReward(float exponent = 0.5f) : exponent(exponent) {}

		void GetPartRewards(const Player& player, const GameState& state, bool isFinal, std::span<float> outPerPart) override {
			const float heldValue = BoostValue(player.boost);
			outPerPart[CONT] = heldValue;

			if (player.prev && player.boost > player.prev->boost)
				outPerPart[EVENT] = heldValue - BoostValue(player.prev->boost);
		}

	private:
		float BoostValue(float boost) const { return powf(boost / 100.f, exponent); }
	};
}