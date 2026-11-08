#pragma once

#include <GigaLearnInfer/Agent.h>
#include <GigaLearnInfer/JsonArchive.h>
#include <nlohmann/json.hpp>
#include <RocketSim/Engine.h>

#include <optional>
#include <string>
#include <vector>

namespace RocketSim
{
	void SerializeFields(GGL::JsonArchive& ar, Vec& vec);
	void SerializeFields(GGL::JsonArchive& ar, WheelPairConfig& config);
}

namespace RocketForge
{
	/** Serializable subset of GGL::PartialModelConfig used by Agent setup metadata. **/
	struct PartialModelConfigSnapshot
	{
		std::vector<int> layerSizes;
		GGL::ModelActivationType activationType = GGL::ModelActivationType::RELU;
		GGL::ModelOptimType optimType = GGL::ModelOptimType::ADAM;
		bool addLayerNorm = true;
		bool addOutputLayer = true;

		void Serialize(GGL::JsonArchive& ar);
	};

	/** Public Agent configuration that affects training and inference compatibility. **/
	struct AgentSetupSnapshot
	{
		std::string agentName;
		std::string obsBuilderName;
		std::string actionParserName;
		RocketSim::GameMode gameMode = RocketSim::GameMode::SOCCAR;
		RocketSim::CarConfig carConfig = RocketSim::CAR_CONFIG_OCTANE;
		int tickSkip = 8;
		int actionDelay = 7;
		int maxTeamSize = 1;

		/** Zero means legacy metadata; resolve against the enclosing training snapshot. **/
		int trainingEngineVersion = 0;
		int ballPredictionEngineVersion = 0;
		bool useBallPrediction = false;
		bool addLayerNorm = true;
		GGL::ModelOptimType optimType = GGL::ModelOptimType::ADAM;
		GGL::ModelActivationType activationType = GGL::ModelActivationType::RELU;
		PartialModelConfigSnapshot sharedHead;
		PartialModelConfigSnapshot policy;
		PartialModelConfigSnapshot critic;
		bool decomposedCritic = false;

		void Serialize(GGL::JsonArchive& ar);
	};

	/**
	 * Resolves legacy settings using recorded training provenance (unknown historical runs use v2).
	 * Throws when the setup and `recordedEngine` both name a training engine and they differ.
	 */
	void ResolveAgentSimulation(AgentSetupSnapshot& setup, int recordedEngine = 0);

	/**
	 * Training engine, then the prediction engine when it differs.
	 * Every observation builder steps the predictor, even with useBallPrediction off.
	 */
	std::vector<RocketSim::EngineId> RequiredEngines(const AgentSetupSnapshot& setup);

	/** True when any JSON load error is on a simulator field (a key ending in engine_version). **/
	bool HasEngineFieldError(const std::vector<std::string>& loadErrors);

	/** Copies only simulator and prediction settings, preserving the rest of the destination setup. **/
	void CopyAgentSimulation(AgentSetupSnapshot& destination, const AgentSetupSnapshot& source);

	/** One saved-vs-current Agent setup change. **/
	struct AgentSetupDiff
	{
		std::string field;
		std::string savedValue;
		std::string currentValue;
	};

	/** Captures public Agent configuration, excluding runtime modelPath. **/
	AgentSetupSnapshot CaptureAgentSetupSnapshot(const Agent& agent);

	/** Applies a previously captured setup to an Agent instance before stage construction. **/
	void ApplyAgentSetupSnapshot(Agent& agent, const AgentSetupSnapshot& snapshot);

	/** Returns display-ready saved -> current Agent setup differences. **/
	std::vector<AgentSetupDiff> DiffAgentSetupSnapshots(
		const AgentSetupSnapshot& saved,
		const AgentSetupSnapshot& current);

	/** Returns true when two Agent setup snapshots have no tracked differences. **/
	bool SameAgentSetupSnapshot(const AgentSetupSnapshot& a, const AgentSetupSnapshot& b);

	nlohmann::json AgentSetupSnapshotToJson(const AgentSetupSnapshot& snapshot);

	/** The JSON a training setup hash covers: the saved JSON plus the retired prediction-rate field, pinned at 0. **/
	nlohmann::json AgentSetupSnapshotToHashJson(const AgentSetupSnapshot& snapshot);
	std::optional<AgentSetupSnapshot> AgentSetupSnapshotFromJson(const nlohmann::json& j);
}
