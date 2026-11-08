#pragma once

#include <TrainingStage.h>

#include <RLGymCPP/Rewards/ZeroSumReward.h>

/**
 * Base class for the example stages.
 * Declares every reward, state setter, and their settings used by any stage. All weights
 * default to 0, and each stage's constructor raises only the weights it trains with. The
 * parent TrainingStage filters out 0-weight rewards/setters, so a 0 weight means "unused".
 * See Documentation/Features/TrainingStages.md.
 */
class ExampleTrainingStage : public RocketForge::TrainingStage
{
public:
	ExampleTrainingStage(std::unique_ptr<Agent> agent);

	virtual std::vector<std::unique_ptr<RLGC::TerminalCondition>> GetTerminalConditions() const override;

protected:
	virtual std::vector<RocketForge::RewardSpec> GetRewardSpecs() const override;

	virtual std::vector<RocketForge::WeightedStateSetter> GetStateSetters() const override;

	/**************** Settings ****************/

	/** Seconds without any ball touch before an episode is truncated. */
	float noTouchTimeoutSeconds = 0.f;

	// TouchBallReward
	RLGC::ZeroSumSettings touchBallZeroSum;

	// StrongTouchReward
	float minStrongTouchKPH = 0.f;
	float maxStrongTouchKPH = 0.f;

	/**************** Reward Weights ****************/

	/**
	 * A RewardWeight can be set per team size as { 1v1, 2v2, 3v3 } instead of a single value.
	 * Any weight that needs it can use this type (see Documentation/Features/TrainingStages.md).
	 */
	RocketForge::RewardWeight rw_touchBall = 0.f;


	float rw_strongTouch = 0.f;
	float rw_velocityToBall = 0.f;
	float rw_faceBall = 0.f;
	float rw_playerSpeed = 0.f;
	float rw_velocityBallToGoal = 0.f;
	float rw_goal = 0.f;

	float rw_air = 0.f;
	RocketForge::AdaptiveWeightConfig aw_air = {};

	float rw_boost_cont = 0.f;
	float rw_boost_event = 0.f;
	RocketForge::AdaptiveWeightConfig aw_boost_event = {};

	/** BumpedPenalty already emits negative values, so its weight stays positive. */
	float rw_bumped = 0.f;
	RocketForge::AdaptiveWeightConfig aw_bumped = {};

	/**************** State Setter Weights ****************/
	float sw_kickoff = 0.f;
	float sw_fuzzedKickoff = 0.f;
	float sw_randomCarsOnGround = 0.f;
	float sw_randomCarsInAir = 0.f;
};

/** Early stage that teaches driving to and touching the ball */
class ExampleTrainingStage_BallChase : public ExampleTrainingStage
{
public:
	ExampleTrainingStage_BallChase(std::unique_ptr<Agent> agent);
};

/** Teaches pushing the ball into the goal. Parents from the base so it starts clean of ball-chase weights. */
class ExampleTrainingStage_BasicScore : public ExampleTrainingStage
{
public:
	ExampleTrainingStage_BasicScore(std::unique_ptr<Agent> agent);
};

/** Builds on basic score with boost management, adaptive weights, and self-play against older versions */
class ExampleTrainingStage_Advanced : public ExampleTrainingStage_BasicScore
{
public:
	ExampleTrainingStage_Advanced(std::unique_ptr<Agent> agent);
};
