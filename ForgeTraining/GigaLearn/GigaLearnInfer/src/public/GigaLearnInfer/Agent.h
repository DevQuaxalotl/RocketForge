#pragma once

#include "ModelConfig.h"

#include <filesystem>
#include <string>

namespace c10
{
	struct Device;
}

namespace GGL
{
	class ModelSet;
}

namespace RLGC
{
	class ObsBuilder;
	class ActionParser;
}

/**
 * A bot's fixed identity: what it observes, which actions it can take, its network shape, its tick timing and its simulator.
 * Every training stage trains the same Agent, and inference runs it unchanged,
 * so changing these settings on an existing build generally means starting a new build.
 * A bot subclasses Agent, sets these in its constructor and registers it with REGISTER_AGENT.
 * See Documentation/AddingABot.md.
 */
class RG_IMEXPORT Agent
{
public:
	Agent();

	virtual ~Agent() = default;

	/**************** Settings ****************/

	/** Name of agent. Used for training selection, checkpoint path, and wandb project name */
	std::string agentName;

	/** Named observation builder selected by saved Agent setup metadata. */
	std::string obsBuilderName;

	/** Named action parser selected by saved Agent setup metadata. */
	std::string actionParserName;

	/** Simulator used for new training builds. Saved checkpoints retain their original simulator. */
	int trainingEngineVersion = 3;

	/** Ball predictor: 0 follows trainingEngineVersion; 2 and 3 explicitly select RocketSim. */
	int ballPredictionEngineVersion = 0;

	/**
	 * Ticks (at 120 Hz) between actions.
	 * Higher trains faster; lower reacts and inputs faster.
	 */
	int tickSkip = 8;

	/**
	 * Ticks between observing the state and applying the action inferred from it.
	 * With a tickSkip of 8 and actionDelay of 7, the environment steps 7 ticks with the old actions,
	 * applies the new ones for the remaining tick, then gives rewards and builds the next observation.
	 * Its default is derived from tickSkip only at construction, so set both together.
	 */
	int actionDelay = tickSkip - 1;

	/** Maximum team size used to construct Obs */
	int maxTeamSize = 1;

	/** Whether runtime inference observations need GameState::GetFutureBallState(). */
	bool useBallPrediction = false;

	/** Partial configs needed by infer for policy, critic, and shared head */
	GGL::PartialModelConfig sharedHead, policy, critic;

	/** Used to set addLayerNorm of all networks */
	bool addLayerNorm = true;

	/** Used to set optimType of all networks */
	GGL::ModelOptimType otimType = GGL::ModelOptimType::ADAM;

	/** Used to set activationType of all networks */
	GGL::ModelActivationType modelActivationType = GGL::ModelActivationType::RELU;

	/** Game mode this agent plays */
	RocketSim::GameMode gameMode;

	/** RocketSim hitbox configuration this agent uses in simulation */
	RocketSim::CarConfig carConfig = RocketSim::CAR_CONFIG_OCTANE;

	/**
	 * Train the critic with one value head per reward instead of one combined value.
	 * Experimental and not a proven technique.
	 * It changes the critic's checkpoint format, so it can't be toggled on an existing build.
	 */
	bool decomposedCritic = false;

	/**************** Overridable ****************/

	/** Creates the Observation Builder used to define the inputs for the network */
	virtual RLGC::ObsBuilder* MakeObsBuilder() const;

	/** Creates the Action Parser used to define the outputs for the network */
	virtual RLGC::ActionParser* MakeActionParser() const;

	/** Returns the number of inputs the Obs uses */
	virtual int GetObsSize() const;

	/** Returns the largest symmetric team size eligible for model benchmarking. */
	virtual int GetMaxBenchmarkTeamSize() const;

	/** Clones this Agent while preserving derived type information */
	virtual std::unique_ptr<Agent> Clone() const;

	/** Constructs the policy models used by inference-only runtimes. */
	virtual void MakeInferenceModels(
		int obsSize,
		int numActions,
		const c10::Device& device,
		GGL::ModelSet& outModels) const;

	/**************** Runtime ****************/

	/** Path to the folder containing POLICY.lt and SHARED_HEAD.lt */
	std::filesystem::path modelPath;

	/** True when the packaged policy can be evaluated but cannot be resumed for training. */
	bool inferenceOnly = false;

	/** Whether frozen-opponent inference should always choose the highest-logit action. */
	bool inferenceDeterministic = false;

	/** Wraps even custom observation builders with this agent's saved prediction settings. */
	RLGC::ObsBuilder* MakeConfiguredObsBuilder() const;

	/** Resolves and validates the prediction engine independently of the match simulator. */
	int GetBallPredictionEngineVersion() const;
};
