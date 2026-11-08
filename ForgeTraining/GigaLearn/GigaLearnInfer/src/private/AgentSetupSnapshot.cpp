#include <GigaLearnInfer/AgentSetupSnapshot.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <sstream>

using nlohmann::json;

namespace RocketSim
{
	void SerializeFields(GGL::JsonArchive& ar, Vec& vec)
	{
		ar.Field("x", vec.x);
		ar.Field("y", vec.y);
		ar.Field("z", vec.z);
		ar.Field("w", vec._w);
	}

	void SerializeFields(GGL::JsonArchive& ar, WheelPairConfig& config)
	{
		ar.Field("wheel_radius", config.wheelRadius);
		ar.Field("suspension_rest_length", config.suspensionRestLength);
		ar.Field("connection_point_offset", config.connectionPointOffset);
	}
}

namespace RocketForge
{
	namespace
	{
		std::string FormatBool(bool value)
		{
			return value ? "true" : "false";
		}

		bool NearlyEqual(float a, float b)
		{
			return std::fabs(a - b) <= 0.0001f;
		}

		bool SameVec(const RocketSim::Vec& a, const RocketSim::Vec& b)
		{
			return NearlyEqual(a.x, b.x)
				&& NearlyEqual(a.y, b.y)
				&& NearlyEqual(a.z, b.z)
				&& NearlyEqual(a._w, b._w);
		}

		bool SameWheelPairConfig(const RocketSim::WheelPairConfig& a, const RocketSim::WheelPairConfig& b)
		{
			return NearlyEqual(a.wheelRadius, b.wheelRadius)
				&& NearlyEqual(a.suspensionRestLength, b.suspensionRestLength)
				&& SameVec(a.connectionPointOffset, b.connectionPointOffset);
		}
		
		bool SameCarConfig(const RocketSim::CarConfig& a, const RocketSim::CarConfig& b)
		{
			return SameVec(a.hitboxSize, b.hitboxSize)
				&& SameVec(a.hitboxPosOffset, b.hitboxPosOffset)
				&& SameWheelPairConfig(a.frontWheels, b.frontWheels)
				&& SameWheelPairConfig(a.backWheels, b.backWheels)
				&& a.threeWheels == b.threeWheels
				&& NearlyEqual(a.dodgeDeadzone, b.dodgeDeadzone);
		}

		std::string FormatLayers(const std::vector<int>& layerSizes)
		{
			if (layerSizes.empty())
				return "[]";

			std::ostringstream out;
			out << "[";
			for (size_t i = 0; i < layerSizes.size(); ++i) {
				if (i > 0)
					out << ", ";
				out << layerSizes[i];
			}
			out << "]";
			return out.str();
		}

		std::string FormatGameMode(RocketSim::GameMode mode)
		{
			switch (mode) {
			case RocketSim::GameMode::SOCCAR: return "soccar";
			case RocketSim::GameMode::HOOPS: return "hoops";
			case RocketSim::GameMode::HEATSEEKER: return "heatseeker";
			case RocketSim::GameMode::SNOWDAY: return "snowday";
			case RocketSim::GameMode::DROPSHOT: return "dropshot";
			case RocketSim::GameMode::THE_VOID: return "void";
			default: return std::to_string(static_cast<int>(mode));
			}
		}
		
		std::string FormatCarConfig(const RocketSim::CarConfig& config)
		{
			if (SameCarConfig(config, RocketSim::CAR_CONFIG_OCTANE)) return "Octane";
			if (SameCarConfig(config, RocketSim::CAR_CONFIG_DOMINUS)) return "Dominus";
			if (SameCarConfig(config, RocketSim::CAR_CONFIG_PLANK)) return "Plank";
			if (SameCarConfig(config, RocketSim::CAR_CONFIG_BREAKOUT)) return "Breakout";
			if (SameCarConfig(config, RocketSim::CAR_CONFIG_HYBRID)) return "Hybrid";
			if (SameCarConfig(config, RocketSim::CAR_CONFIG_MERC)) return "Merc";
			if (SameCarConfig(config, RocketSim::CAR_CONFIG_PSYCLOPS)) return "Psyclops";
			return "Custom";
		}

		std::string FormatOptimType(GGL::ModelOptimType type)
		{
			switch (type) {
			case GGL::ModelOptimType::ADAM: return "ADAM";
			case GGL::ModelOptimType::ADAMW: return "ADAMW";
			case GGL::ModelOptimType::ADAGRAD: return "ADAGRAD";
			case GGL::ModelOptimType::RMSPROP: return "RMSPROP";
			case GGL::ModelOptimType::MAGSGD: return "MAGSGD";
			default: return std::to_string(static_cast<int>(type));
			}
		}

		std::string FormatActivationType(GGL::ModelActivationType type)
		{
			switch (type) {
			case GGL::ModelActivationType::RELU: return "RELU";
			case GGL::ModelActivationType::LEAKY_RELU: return "LEAKY_RELU";
			case GGL::ModelActivationType::SIGMOID: return "SIGMOID";
			case GGL::ModelActivationType::TANH: return "TANH";
			default: return std::to_string(static_cast<int>(type));
			}
		}

		std::string NormalizePresetName(std::string text)
		{
			text.erase(
				std::remove_if(text.begin(), text.end(), [](unsigned char c) {
					return c == '_' || c == '-' || std::isspace(c);
				}),
				text.end());
			std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
				return static_cast<char>(std::tolower(c));
			});
			return text;
		}

		RocketSim::CarConfig CarConfigFromPresetName(const std::string& name)
		{
			const std::string normalized = NormalizePresetName(name);
			if (normalized == "octane") return RocketSim::CAR_CONFIG_OCTANE;
			if (normalized == "dominus") return RocketSim::CAR_CONFIG_DOMINUS;
			if (normalized == "plank" || normalized == "batmobile") return RocketSim::CAR_CONFIG_PLANK;
			if (normalized == "breakout") return RocketSim::CAR_CONFIG_BREAKOUT;
			if (normalized == "hybrid") return RocketSim::CAR_CONFIG_HYBRID;
			if (normalized == "merc") return RocketSim::CAR_CONFIG_MERC;
			if (normalized == "psyclops") return RocketSim::CAR_CONFIG_PSYCLOPS;
			return RocketSim::CAR_CONFIG_OCTANE;
		}
		
		void SerializeCarConfigFields(GGL::JsonArchive& ar, RocketSim::CarConfig& config)
		{
			ar.Field("hitbox_size", config.hitboxSize, RocketSim::Vec{});
			ar.Field("hitbox_pos_offset", config.hitboxPosOffset, RocketSim::Vec{});
			ar.Field("front_wheels", config.frontWheels, RocketSim::WheelPairConfig{});
			ar.Field("back_wheels", config.backWheels, RocketSim::WheelPairConfig{});
			ar.Field("three_wheels", config.threeWheels, false);
			ar.Field("dodge_deadzone", config.dodgeDeadzone, 0.5f);
		}

		PartialModelConfigSnapshot CapturePartialModelConfig(const GGL::PartialModelConfig& config)
		{
			PartialModelConfigSnapshot snapshot = {};
			snapshot.layerSizes = config.layerSizes;
			snapshot.activationType = config.activationType;
			snapshot.optimType = config.optimType;
			snapshot.addLayerNorm = config.addLayerNorm;
			snapshot.addOutputLayer = config.addOutputLayer;
			return snapshot;
		}

		GGL::PartialModelConfig ToPartialModelConfig(const PartialModelConfigSnapshot& snapshot)
		{
			GGL::PartialModelConfig config = {};
			config.layerSizes = snapshot.layerSizes;
			config.activationType = snapshot.activationType;
			config.optimType = snapshot.optimType;
			config.addLayerNorm = snapshot.addLayerNorm;
			config.addOutputLayer = snapshot.addOutputLayer;
			return config;
		}

		void AddDiff(
			std::vector<AgentSetupDiff>& diffs,
			const std::string& field,
			const std::string& savedValue,
			const std::string& currentValue)
		{
			if (savedValue == currentValue)
				return;

			diffs.push_back({ field, savedValue, currentValue });
		}

		void AddPartialModelConfigDiffs(
			std::vector<AgentSetupDiff>& diffs,
			const std::string& prefix,
			const PartialModelConfigSnapshot& saved,
			const PartialModelConfigSnapshot& current)
		{
			AddDiff(diffs, prefix + " Layers", FormatLayers(saved.layerSizes), FormatLayers(current.layerSizes));
			AddDiff(diffs, prefix + " Activation", FormatActivationType(saved.activationType), FormatActivationType(current.activationType));
			AddDiff(diffs, prefix + " Optimizer", FormatOptimType(saved.optimType), FormatOptimType(current.optimType));
			AddDiff(diffs, prefix + " Layer Norm", FormatBool(saved.addLayerNorm), FormatBool(current.addLayerNorm));
			AddDiff(diffs, prefix + " Output Layer", FormatBool(saved.addOutputLayer), FormatBool(current.addOutputLayer));
		}
	}

	void PartialModelConfigSnapshot::Serialize(GGL::JsonArchive& ar)
	{
		ar.Field("layer_sizes", layerSizes);
		ar.EnumIntField("activation_type", activationType);
		ar.EnumIntField("optim_type", optimType);
		ar.Field("add_layer_norm", addLayerNorm);
		ar.Field("add_output_layer", addOutputLayer);
	}

	void AgentSetupSnapshot::Serialize(GGL::JsonArchive& ar)
	{
		ar.Field("agent_name", agentName);
		ar.RequiredField("obs_builder", obsBuilderName);
		ar.RequiredField("action_parser", actionParserName);
		ar.EnumIntField("game_mode", gameMode);
		
		if (ar.IsSaving()) {
			const std::string preset = FormatCarConfig(carConfig);
			if (preset != "Custom") {
				(*ar.Raw())["car_config"] = preset;
			} else {
				GGL::JsonArchive carScope = ar.Child("car_config");
				SerializeCarConfigFields(carScope, carConfig);
			}
		} else if (ar.Contains("car_config")) {
			const nlohmann::json& node = ar.RawLoad()->at("car_config");
			if (node.is_string()) {
				carConfig = CarConfigFromPresetName(node.get<std::string>());
			} else if (node.is_object()) {
				GGL::JsonArchive carScope = ar.Child("car_config");
				SerializeCarConfigFields(carScope, carConfig);
			}
			// else: present-but-scalar (e.g. a number or bool) - leave carConfig untouched.
		} else {
			GGL::JsonArchive carScope = ar.Child("car_config");
			SerializeCarConfigFields(carScope, carConfig);
		}

		ar.Field("tick_skip", tickSkip);
		ar.Field("action_delay", actionDelay, tickSkip - 1);   // default depends on tick_skip, loaded just above
		ar.Field("max_team_size", maxTeamSize);
		ar.Field("training_engine_version", trainingEngineVersion);
		ar.Field("ball_prediction_engine_version", ballPredictionEngineVersion);
		ar.Field("use_ball_prediction", useBallPrediction);
		ar.Field("add_layer_norm", addLayerNorm);
		ar.EnumIntField("optim_type", optimType);
		ar.EnumIntField("activation_type", activationType);
		ar.Field("shared_head", sharedHead);
		ar.Field("policy", policy);
		ar.Field("critic", critic);

		// Written only when on, so agents without it keep their existing setup JSON and hash.
		if (ar.IsLoading() || decomposedCritic)
			ar.Field("decomposed_critic", decomposedCritic);
	}

	AgentSetupSnapshot CaptureAgentSetupSnapshot(const Agent& agent)
	{
		AgentSetupSnapshot snapshot = {};
		snapshot.agentName = agent.agentName;
		snapshot.obsBuilderName = agent.obsBuilderName;
		snapshot.actionParserName = agent.actionParserName;
		snapshot.gameMode = agent.gameMode;
		snapshot.carConfig = agent.carConfig;
		snapshot.tickSkip = agent.tickSkip;
		snapshot.actionDelay = agent.actionDelay;
		snapshot.maxTeamSize = agent.maxTeamSize;
		snapshot.trainingEngineVersion = agent.trainingEngineVersion;
		snapshot.ballPredictionEngineVersion = agent.GetBallPredictionEngineVersion();
		snapshot.useBallPrediction = agent.useBallPrediction;
		snapshot.addLayerNorm = agent.addLayerNorm;
		snapshot.optimType = agent.otimType;
		snapshot.activationType = agent.modelActivationType;
		snapshot.sharedHead = CapturePartialModelConfig(agent.sharedHead);
		snapshot.policy = CapturePartialModelConfig(agent.policy);
		snapshot.critic = CapturePartialModelConfig(agent.critic);
		snapshot.decomposedCritic = agent.decomposedCritic;
		return snapshot;
	}

	void ResolveAgentSimulation(AgentSetupSnapshot& setup, int recordedEngine)
	{
		if (setup.trainingEngineVersion != 0 && recordedEngine != 0 && setup.trainingEngineVersion != recordedEngine)
			throw std::runtime_error("Saved setup is contradictory: its agent trained on RocketSim v" + std::to_string(setup.trainingEngineVersion)
				+ " but the snapshot records v" + std::to_string(recordedEngine));

		if (setup.trainingEngineVersion == 0)
			setup.trainingEngineVersion = recordedEngine == 0 ? 2 : recordedEngine;
		if (setup.ballPredictionEngineVersion == 0)
			setup.ballPredictionEngineVersion = setup.trainingEngineVersion;

		if ((setup.trainingEngineVersion != 2 && setup.trainingEngineVersion != 3) ||
			(setup.ballPredictionEngineVersion != 2 && setup.ballPredictionEngineVersion != 3))
			throw std::runtime_error("Agent setup requires an unsupported simulator (expected RocketSim v2 or v3)");
	}

	std::vector<RocketSim::EngineId> RequiredEngines(const AgentSetupSnapshot& setup)
	{
		AgentSetupSnapshot resolved = setup;
		ResolveAgentSimulation(resolved);

		const RocketSim::EngineId training = RocketSim::EngineFromVersion(resolved.trainingEngineVersion);
		const RocketSim::EngineId prediction = RocketSim::EngineFromVersion(resolved.ballPredictionEngineVersion);

		if (prediction == training)
			return { training };
		return { training, prediction };
	}

	bool HasEngineFieldError(const std::vector<std::string>& loadErrors)
	{
		for (const std::string& error : loadErrors) {
			const std::string path = error.substr(0, error.find(": "));
			if (path.ends_with("engine_version"))
				return true;
		}
		return false;
	}

	void CopyAgentSimulation(AgentSetupSnapshot& destination, const AgentSetupSnapshot& source)
	{
		destination.trainingEngineVersion = source.trainingEngineVersion;
		destination.ballPredictionEngineVersion = source.ballPredictionEngineVersion;
		destination.useBallPrediction = source.useBallPrediction;
	}

	void ApplyAgentSetupSnapshot(Agent& agent, const AgentSetupSnapshot& snapshot)
	{
		agent.agentName = snapshot.agentName;
		agent.obsBuilderName = snapshot.obsBuilderName;
		agent.actionParserName = snapshot.actionParserName;
		agent.gameMode = snapshot.gameMode;
		agent.carConfig = snapshot.carConfig;
		agent.tickSkip = snapshot.tickSkip;
		agent.actionDelay = snapshot.actionDelay;
		agent.maxTeamSize = snapshot.maxTeamSize;

		AgentSetupSnapshot simulation = snapshot;
		ResolveAgentSimulation(simulation);
		agent.trainingEngineVersion = simulation.trainingEngineVersion;
		agent.ballPredictionEngineVersion = simulation.ballPredictionEngineVersion;

		agent.useBallPrediction = snapshot.useBallPrediction;
		agent.addLayerNorm = snapshot.addLayerNorm;
		agent.otimType = snapshot.optimType;
		agent.modelActivationType = snapshot.activationType;
		agent.sharedHead = ToPartialModelConfig(snapshot.sharedHead);
		agent.policy = ToPartialModelConfig(snapshot.policy);
		agent.critic = ToPartialModelConfig(snapshot.critic);
		agent.decomposedCritic = snapshot.decomposedCritic;
	}

	std::vector<AgentSetupDiff> DiffAgentSetupSnapshots(
		const AgentSetupSnapshot& saved,
		const AgentSetupSnapshot& current)
	{
		std::vector<AgentSetupDiff> diffs;
		AddDiff(diffs, "Agent Name", saved.agentName, current.agentName);
		AddDiff(diffs, "Observation Builder", saved.obsBuilderName, current.obsBuilderName);
		AddDiff(diffs, "Action Parser", saved.actionParserName, current.actionParserName);
		AddDiff(diffs, "Game Mode", FormatGameMode(saved.gameMode), FormatGameMode(current.gameMode));
		AddDiff(diffs, "Car Config", FormatCarConfig(saved.carConfig), FormatCarConfig(current.carConfig));
		AddDiff(diffs, "Tick Skip", std::to_string(saved.tickSkip), std::to_string(current.tickSkip));
		AddDiff(diffs, "Action Delay", std::to_string(saved.actionDelay), std::to_string(current.actionDelay));
		AddDiff(diffs, "Max Team Size", std::to_string(saved.maxTeamSize), std::to_string(current.maxTeamSize));

		AgentSetupSnapshot savedSimulation = saved;
		AgentSetupSnapshot currentSimulation = current;
		ResolveAgentSimulation(savedSimulation);
		ResolveAgentSimulation(currentSimulation);
		AddDiff(diffs, "Training Simulator", std::to_string(savedSimulation.trainingEngineVersion), std::to_string(currentSimulation.trainingEngineVersion));
		AddDiff(diffs, "Ball Prediction Simulator", std::to_string(savedSimulation.ballPredictionEngineVersion), std::to_string(currentSimulation.ballPredictionEngineVersion));
		AddDiff(diffs, "Use Ball Prediction", FormatBool(saved.useBallPrediction), FormatBool(current.useBallPrediction));
		AddDiff(diffs, "Default Layer Norm", FormatBool(saved.addLayerNorm), FormatBool(current.addLayerNorm));
		AddDiff(diffs, "Default Optimizer", FormatOptimType(saved.optimType), FormatOptimType(current.optimType));
		AddDiff(diffs, "Default Activation", FormatActivationType(saved.activationType), FormatActivationType(current.activationType));
		AddPartialModelConfigDiffs(diffs, "Shared Head", saved.sharedHead, current.sharedHead);
		AddPartialModelConfigDiffs(diffs, "Policy", saved.policy, current.policy);
		AddPartialModelConfigDiffs(diffs, "Critic", saved.critic, current.critic);
		AddDiff(diffs, "Decomposed Critic", FormatBool(saved.decomposedCritic), FormatBool(current.decomposedCritic));
		return diffs;
	}

	bool SameAgentSetupSnapshot(const AgentSetupSnapshot& a, const AgentSetupSnapshot& b)
	{
		return DiffAgentSetupSnapshots(a, b).empty();
	}

	json AgentSetupSnapshotToJson(const AgentSetupSnapshot& snapshot)
	{
		json j;
		GGL::JsonSave(snapshot, j);
		return j;
	}

	json AgentSetupSnapshotToHashJson(const AgentSetupSnapshot& snapshot)
	{
		json j = AgentSetupSnapshotToJson(snapshot);

		// Keeps the hashes of setups saved while the field existed.
		j["ball_pred_tick_rate"] = 0.f;
		return j;
	}

	std::optional<AgentSetupSnapshot> AgentSetupSnapshotFromJson(const json& j)
	{
		if (!j.is_object())
			return std::nullopt;

		AgentSetupSnapshot snapshot = {};
		std::vector<std::string> missingRequiredFields;
		const std::vector<std::string> errors = GGL::JsonLoad(j, snapshot, &missingRequiredFields);

		if (!missingRequiredFields.empty() || HasEngineFieldError(errors))
			return std::nullopt;

		return snapshot;
	}
}
