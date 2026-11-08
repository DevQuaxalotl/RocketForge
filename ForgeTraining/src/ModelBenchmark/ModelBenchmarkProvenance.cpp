#include "ModelBenchmark.h"
#include "ModelBenchmarkCompatibility.h"

#include <nlohmann/json.hpp>
#include <limits>

namespace RocketForgeTraining {
	int RecordedModelBenchmarkCompatibilityVersion(const std::string& evaluationIdentity) {
		// Legacy results remain reusable without inventing exact provenance.
		// A malformed explicit marker must not silently become a member of the baseline version.
		if (evaluationIdentity.empty())
			return 1;

		const nlohmann::json identity = nlohmann::json::parse(evaluationIdentity, nullptr, false);
		if (!identity.is_object())
			return 0;

		const auto marker = identity.find("benchmarkCompatibilityVersion");
		if (marker == identity.end())
			return 1;
		if (!marker->is_number_integer() || *marker < 1 || *marker > std::numeric_limits<int>::max())
			return 0;

		return marker->get<int>();
	}

	void RecordModelBenchmarkEnvironment(
		ModelBenchmarkResult& result,
		const ModelBenchmarkConfig& config,
		const std::string& family,
		const std::string& revision,
		const std::string& carOrder,
		const std::map<int, std::string>& predictorRevisions,
		bool noBallRot) {
		using Json = nlohmann::json;

		result.engineVersion = result.rocketSimVersions.size() == 1 ? *result.rocketSimVersions.begin() : -1;
		result.engineFamily = family;
		result.engineRevision = revision;
		result.engineCarOrder = carOrder;

		// The same weights can appear more than once with different captured setups.
		// Keep every profile rather than overwriting it under a shared checkpoint path.
		std::map<std::string, int> checkpointCounts;
		for (const ModelBenchmarkModel& model : config.models)
			++checkpointCounts[model.checkpointPath.lexically_normal().generic_string()];

		auto modelKey = [&](size_t index) {
			std::string key = config.models[index].checkpointPath.lexically_normal().generic_string();
			if (checkpointCounts[key] > 1)
				key += "#benchmark-model-" + std::to_string(index);
			return key;
		};

		Json profiles = Json::object();
		for (size_t i = 0; i < config.modelPredictionEngines.size() && i < config.models.size(); ++i) {
			const int engine = config.modelPredictionEngines[i];
			if (engine == 0 || (result.rocketSimVersions.size() == 1 && engine == *result.rocketSimVersions.begin()))
				continue;

			const auto found = predictorRevisions.find(engine);
			const std::string predictorRevision = found == predictorRevisions.end() ? "unrecorded" : found->second;
			profiles[modelKey(i)] = { { "engine", engine }, { "revision", predictorRevision } };
		}

		// Prediction always runs at the native rate.
		// It is recorded as 0 to match identities saved before the rate setting was retired.
		const float nativePredictionRate = 0.f;
		Json savedProfiles = Json::object();
		for (size_t i = 0; i < config.models.size(); ++i) {
			const Json setup = Json::parse(config.models[i].agentSetupJson, nullptr, false);
			if (setup.is_object()) {
				savedProfiles[modelKey(i)] = {
					{ "engine", setup.value("ball_prediction_engine_version", 0) },
					{ "tickRate", nativePredictionRate },
					{ "enabled", setup.value("use_ball_prediction", false) }
				};
			}
		}

		result.predictionConfigurationJson = profiles.dump();

		const int engineVersion = result.rocketSimVersions.size() == 1 ? *result.rocketSimVersions.begin() : 0;
		const bool frozenStarts = !config.startingStateDirectory.empty();
		Json identity = Json{
			{ "benchmarkCompatibilityVersion", CurrentModelBenchmarkCompatibilityVersion(engineVersion) },
			{ "physicsFamily", family },
			{ "physicsRevision", revision },
			{ "carOrder", carOrder },
			{ "externalPrediction", profiles },
			{ "savedPredictionProfiles", savedProfiles },
			{ "tickRate", 120 },
			{ "noBallRot", noBallRot },
			{ "startingProtocol", frozenStarts ? "explicit-common-fuzzed-kickoff-v1" : "native-fuzzed-kickoff" },
			{ "evaluationRng", "native" },
			{ "physicsSampling", frozenStarts ? "local-every-tick-first-goal-v1" : "native-chunks" },
			{ "arenaReset", "public-state-restore" }
		};

		// Preserve existing deterministic identities and explicit history selections.
		// Sampling settings distinguish new environments only when sampling is enabled.
		if (config.stochasticKickoff) {
			identity["stochasticKickoff"] = true;
			identity["kickoffTemperature"] = config.kickoffTemperature;
		}

		result.evaluationIdentity = identity.dump();
	}
}
