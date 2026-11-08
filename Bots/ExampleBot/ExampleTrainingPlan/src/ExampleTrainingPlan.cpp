#include "ExampleTrainingPlan.h"

#include <StageRegistry.h>

// Rewards
#include <RLGymCPP/Rewards/CommonRewards.h>

// Setters
#include <RLGymCPP/StateSetters/KickoffState.h>
#include <RLGymCPP/StateSetters/FuzzedKickoffState.h>
#include <RLGymCPP/StateSetters/RandomState.h>

// Terminal Conditions
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>

using namespace std;
using namespace RLGC;
using namespace RocketForge;

// Stages are hidden from agents whose maxTeamSize is below minAgentTeamSize.
// Keep it in step with weightedTeamSizes below and ExampleTrainingAgent::maxTeamSize.
static constexpr StageCompatInfo EXAMPLE_COMPAT_INFO{ .minAgentTeamSize = 1 };

ExampleTrainingStage::ExampleTrainingStage(std::unique_ptr<Agent> agent) : TrainingStage(std::move(agent))
{
	// ExampleAgent is 1v1 only. With a larger ExampleTrainingAgent::maxTeamSize and EXAMPLE_COMPAT_INFO.minAgentTeamSize,
	// a multi-mode agent could use { { 1, 1.f }, { 2, 1.f }, { 3, 1.f }, { 1, 2, 0.1f }, { 2, 3, 0.1f } }
	// (see Documentation/Features/TrainingStages.md).
	weightedTeamSizes = { { 1, 1.f } };
	
	// Stages can also change learnerConfig: see LearnerConfig for the run settings and PPOLearnerConfig for learnerConfig.ppo.

	/**************** Settings ****************/
	noTouchTimeoutSeconds = 15.f;
	
	minStrongTouchKPH = 20.f;
	maxStrongTouchKPH = 130.f;
	
	// An opponent's touch costs half as much as your own earns (see Documentation/Features/Rewards.md).
	// teamSpirit shares a teammate's touch; it has no effect in 1v1, where the team is the player alone.
	touchBallZeroSum.opponentScale = 0.5f;
	touchBallZeroSum.teamSpirit = 0.75f;
}

vector<RewardSpec> ExampleTrainingStage::GetRewardSpecs() const
{
	// Every Part() shares this one instance, so multi-part owners must be created with make_shared (see Documentation/Features/Rewards.md)
	shared_ptr<BoostReward> boostReward = make_shared<BoostReward>();

	return {
	// Ball rewards
	{ new ZeroSumReward(new TouchBallReward(), touchBallZeroSum), rw_touchBall },
	{ new StrongTouchReward(minStrongTouchKPH, maxStrongTouchKPH), rw_strongTouch },
	{ new VelocityPlayerToBallReward(), rw_velocityToBall },
	{ new FaceBallReward(), rw_faceBall },

	// Goal rewards
	{ new VelocityBallToGoalReward(), rw_velocityBallToGoal },
	{ new GoalReward(), rw_goal },

	// Player movement rewards
	{ new SpeedReward(), rw_playerSpeed },
	{ new AirReward(), rw_air, aw_air },

	// Boost rewards
	{ boostReward->Part(BoostReward::CONT), rw_boost_cont },
	{ boostReward->Part(BoostReward::EVENT), rw_boost_event, aw_boost_event },

	// Punishments
	{ new BumpedPenalty(), rw_bumped, aw_bumped },
	};
}

vector<WeightedStateSetter> ExampleTrainingStage::GetStateSetters() const
{
	const bool randomBallSpeed = true;
	const bool randomCarSpeed = true;
	const bool carsOnGround = true;

	return {
	{ new KickoffState(), sw_kickoff },
	{ new FuzzedKickoffState(), sw_fuzzedKickoff },
	{ new RandomState(randomBallSpeed, randomCarSpeed, carsOnGround), sw_randomCarsOnGround },
	{ new RandomState(randomBallSpeed, randomCarSpeed, !carsOnGround), sw_randomCarsInAir },
	};
}

vector<unique_ptr<TerminalCondition>> ExampleTrainingStage::GetTerminalConditions() const
{
	vector<unique_ptr<TerminalCondition>> conditions;
	conditions.push_back(make_unique<NoTouchCondition>(noTouchTimeoutSeconds));
	conditions.push_back(make_unique<GoalScoreCondition>());
	return conditions;
}

/**********************************************
* Ball Chase
**********************************************/

static constexpr string_view NAME_BALLCHASE = "example-ball-chase";
REGISTER_STAGE(NAME_BALLCHASE.data(), ExampleTrainingStage_BallChase, EXAMPLE_COMPAT_INFO);

ExampleTrainingStage_BallChase::ExampleTrainingStage_BallChase(std::unique_ptr<Agent> agent) : ExampleTrainingStage(std::move(agent))
{
	stageName = NAME_BALLCHASE;

	learnerConfig.tsPerVersion = 25'000'000;

	// Don't need skill tracking yet
	learnerConfig.skillTracker.enabled = false;

	learnerConfig.ppo.entropyScale = 0.02f;

	/**************** Reward Weights ****************/
	rw_touchBall = 50.f;
	rw_velocityToBall = 5.f;
	rw_faceBall = 1.f; // Keeps it from learning to drive backward at the ball
	rw_air = 0.15f; // Keeps it from forgetting how to jump

	/**************** State Setter Weights ****************/
	sw_kickoff = 1.f;
	sw_randomCarsInAir = 1.f;
}

/**********************************************
* Basic Score
**********************************************/

static constexpr string_view NAME_BASIC_SCORE = "example-basic-score";
REGISTER_STAGE(NAME_BASIC_SCORE.data(), ExampleTrainingStage_BasicScore, EXAMPLE_COMPAT_INFO);

ExampleTrainingStage_BasicScore::ExampleTrainingStage_BasicScore(std::unique_ptr<Agent> agent) : ExampleTrainingStage(std::move(agent))
{
	stageName = NAME_BASIC_SCORE;

	timestepsPerIteration = 100'000;

	learnerConfig.tsPerSave = 50'000'000;
	learnerConfig.tsPerVersion = 100'000'000;

	learnerConfig.skillTracker.enabled = true;

	/**************** Reward Weights ****************/
	rw_goal = 300.f;
	rw_velocityBallToGoal = 40.f;
	rw_touchBall = 5.f;
	rw_velocityToBall = 2.f;
	rw_playerSpeed = 3.f;
	rw_air = 0.15f;

	/**************** State Setter Weights ****************/
	sw_fuzzedKickoff = 1.f;
	sw_randomCarsInAir = 0.5f;
}

/**********************************************
* Advanced
**********************************************/

static constexpr string_view NAME_ADVANCED = "example-advanced";
REGISTER_STAGE(NAME_ADVANCED.data(), ExampleTrainingStage_Advanced, EXAMPLE_COMPAT_INFO);

ExampleTrainingStage_Advanced::ExampleTrainingStage_Advanced(std::unique_ptr<Agent> agent) : ExampleTrainingStage_BasicScore(std::move(agent))
{
	stageName = NAME_ADVANCED;

	timestepsPerIteration = 200'000;
	learnerConfig.tsPerSave = 100'000'000;
	learningRate = 1e-4;

	// Longer credit horizon (~13s at 15Hz) so goals credit the plays that set them up
	learnerConfig.ppo.gaeGamma = 0.995f;
	learnerConfig.ppo.maxEpisodeDuration = 300.f;

	// Self-play against saved older versions (see Documentation/Features/TrainingStages.md)
	learnerConfig.savePolicyVersions = true;
	learnerConfig.tsPerVersion = 250'000'000;
	learnerConfig.trainAgainstRecentVersions = true;
	learnerConfig.trainAgainstRecentPercent = 0.1f;

	/**************** Settings ****************/
	noTouchTimeoutSeconds = 30.f;

	minStrongTouchKPH = 40.f;

	/**************** Reward Weights ****************/
	// Weights inherited from basic score carry over; only the changes are set here
	rw_goal = 1000.f;
	rw_touchBall = 2.f;
	rw_velocityToBall = 0.f;
	rw_strongTouch = 10.f;

	rw_boost_cont = 2.f;
	rw_boost_event = 100.f;

	// Adaptive weights tune each weight from how often the agent produces the reward (see Documentation/Features/AdaptiveRewardWeights.md).
	// Aim for ~1 second of full AirReward per 30 seconds of play, letting the weight range over [0.02, 0.5].
	aw_air = AdaptiveWeightSettings::TargetReward(AdaptiveTargetContinuous(1.f, 30.f), 0.02f, 0.5f);

	// Event targets count fires: ~6 full pickups per 100 seconds
	aw_boost_event = AdaptiveWeightSettings::TargetReward(AdaptiveTargetEvent(6.f, 100.f, agentTickSkip), 25.f, 400.f);

	// A punishment's target is a ceiling: it pushes until being bumped drops to ~1 per 60 seconds, then fades
	rw_bumped = 20.f;
	aw_bumped = AdaptiveWeightSettings::TargetPunishment(AdaptiveTargetEvent(1.f, 60.f, agentTickSkip), 0.f, 50.f);

	/**************** State Setter Weights ****************/
	sw_fuzzedKickoff = 3.f;
	sw_randomCarsOnGround = 0.5f;
	sw_randomCarsInAir = 0.5f;
}
