#include "BenchmarkStartingStates.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>
#include <algorithm>

namespace RocketForgeTraining::ModelBenchmarkInternal {
	using Json = nlohmann::json;

	static Json Vector(const RocketSim::Vec& vec) { return { vec.x, vec.y, vec.z }; }
	static RocketSim::Vec Vector(const Json& json) { return { json.at(0), json.at(1), json.at(2) }; }

	static Json Physical(const RocketSim::PhysState& state) {
		return {
			{ "pos", Vector(state.pos) },
			{ "vel", Vector(state.vel) },
			{ "angVel", Vector(state.angVel) },
			{ "rotation", { Vector(state.rotMat.forward), Vector(state.rotMat.right), Vector(state.rotMat.up) } }
		};
	}

	static void Physical(RocketSim::PhysState& state, const Json& json) {
		state.pos = Vector(json.at("pos"));
		state.vel = Vector(json.at("vel"));
		state.angVel = Vector(json.at("angVel"));

		const Json& rotation = json.at("rotation");
		state.rotMat = { Vector(rotation.at(0)), Vector(rotation.at(1)), Vector(rotation.at(2)) };
	}

	static std::filesystem::path Filename(const std::filesystem::path& directory, int mode, uint64_t seed) {
		return directory / (std::to_string(mode) + "-" + std::to_string(seed) + ".json");
	}

	static Json Encode(const GGL::PolicyMatchSnapshot& snapshot, int mode, uint64_t seed) {
		Json result = {
			{ "schema", 1 },
			{ "mode", mode },
			{ "snapshotSeed", seed },
			{ "ball", Physical(snapshot.ball) },
			{ "cars", Json::array() }
		};
		for (const GGL::PolicyMatchCarSnapshot& car : snapshot.cars) {
			Json out = { { "team", int(car.team) }, { "phys", Physical(car.state) }, { "flipRelTorque", Vector(car.state.flipRelTorque) } };
			out["isOnGround"] = car.state.isOnGround;
			out["hasJumped"] = car.state.hasJumped;
			out["hasDoubleJumped"] = car.state.hasDoubleJumped;
			out["hasFlipped"] = car.state.hasFlipped;
			out["jumpTime"] = car.state.jumpTime;
			out["flipTime"] = car.state.flipTime;
			out["isFlipping"] = car.state.isFlipping;
			out["isJumping"] = car.state.isJumping;
			out["airTime"] = car.state.airTime;
			out["airTimeSinceJump"] = car.state.airTimeSinceJump;
			out["boost"] = car.state.boost;
			out["timeSinceBoosted"] = car.state.timeSinceBoosted;
			out["isBoosting"] = car.state.isBoosting;
			out["boostingTime"] = car.state.boostingTime;
			out["isSupersonic"] = car.state.isSupersonic;
			out["supersonicTime"] = car.state.supersonicTime;
			out["handbrakeVal"] = car.state.handbrakeVal;
			out["isAutoFlipping"] = car.state.isAutoFlipping;
			out["autoFlipTimer"] = car.state.autoFlipTimer;
			out["autoFlipTorqueScale"] = car.state.autoFlipTorqueScale;
			out["isDemoed"] = car.state.isDemoed;
			out["demoRespawnTimer"] = car.state.demoRespawnTimer;
			out["wheelsWithContact"] = {
				car.state.wheelsWithContact[0],
				car.state.wheelsWithContact[1],
				car.state.wheelsWithContact[2],
				car.state.wheelsWithContact[3]
			};
			result["cars"].push_back(std::move(out));
		}

		return result;
	}

	GGL::PolicyMatchSnapshot ReadBenchmarkStartingState(const std::filesystem::path& directory, int mode, uint64_t seed) {
		std::ifstream input(Filename(directory, mode, seed));
		if (!input)
			throw std::runtime_error("Missing frozen benchmark starting state: " + Filename(directory, mode, seed).string());

		const Json json = Json::parse(input);
		if (json.at("schema") != 1 || json.at("mode") != mode || json.at("snapshotSeed") != seed)
			throw std::runtime_error("Frozen benchmark start identity mismatch");

		GGL::PolicyMatchSnapshot snapshot;
		Physical(snapshot.ball, json.at("ball"));

		for (const Json& row : json.at("cars")) {
			GGL::PolicyMatchCarSnapshot car;
			car.team = RocketSim::Team(row.at("team").get<int>());
			Physical(car.state, row.at("phys"));
			car.state.flipRelTorque = Vector(row.at("flipRelTorque"));
			row.at("isOnGround").get_to(car.state.isOnGround);
			row.at("hasJumped").get_to(car.state.hasJumped);
			row.at("hasDoubleJumped").get_to(car.state.hasDoubleJumped);
			row.at("hasFlipped").get_to(car.state.hasFlipped);
			row.at("jumpTime").get_to(car.state.jumpTime);
			row.at("flipTime").get_to(car.state.flipTime);
			row.at("isFlipping").get_to(car.state.isFlipping);
			row.at("isJumping").get_to(car.state.isJumping);
			row.at("airTime").get_to(car.state.airTime);
			row.at("airTimeSinceJump").get_to(car.state.airTimeSinceJump);
			row.at("boost").get_to(car.state.boost);
			row.at("timeSinceBoosted").get_to(car.state.timeSinceBoosted);
			row.at("isBoosting").get_to(car.state.isBoosting);
			row.at("boostingTime").get_to(car.state.boostingTime);
			row.at("isSupersonic").get_to(car.state.isSupersonic);
			row.at("supersonicTime").get_to(car.state.supersonicTime);
			row.at("handbrakeVal").get_to(car.state.handbrakeVal);
			row.at("isAutoFlipping").get_to(car.state.isAutoFlipping);
			row.at("autoFlipTimer").get_to(car.state.autoFlipTimer);
			row.at("autoFlipTorqueScale").get_to(car.state.autoFlipTorqueScale);
			row.at("isDemoed").get_to(car.state.isDemoed);
			row.at("demoRespawnTimer").get_to(car.state.demoRespawnTimer);
			for (int i = 0; i < 4; ++i)
				car.state.wheelsWithContact[i] = row.at("wheelsWithContact").at(i);

			snapshot.cars.push_back(std::move(car));
		}

		if (snapshot.cars.size() != size_t(mode * 2))
			throw std::runtime_error("Frozen benchmark start has wrong car count");

		return snapshot;
	}

	// SplitMix64 finalizer: spreads consecutive ids into well-distributed seeds.
	static uint64_t Mix(uint64_t value) {
		value += 0x9e3779b97f4a7c15ULL;
		value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
		value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
		return value ^ (value >> 31);
	}

	void PrepareBenchmarkStartingStates(const ModelBenchmarkConfig& config) {
		if (config.seed == 0 || config.startingStateDirectory.empty() || !config.usePairBudgets)
			throw std::runtime_error("Preparing frozen starts requires a seed, directory, and explicit pair budgets");

		std::filesystem::create_directories(config.startingStateDirectory);

		for (int mode = 1; mode <= 3; ++mode) {
			int games = 0;
			for (const ModelBenchmarkPairBudget& pair : config.pairBudgets)
				if (pair.mode == mode)
					games = std::max(games, pair.games);

			for (int i = 0; i < (games + 1) / 2; ++i) {
				const uint64_t id = config.snapshotOffset + uint64_t(mode) * BENCHMARK_START_MODE_ID_STRIDE + i;
				const uint64_t seed = Mix(config.seed + id);
				const std::filesystem::path file = Filename(config.startingStateDirectory, mode, seed);
				if (std::filesystem::exists(file)) {
					ReadBenchmarkStartingState(config.startingStateDirectory, mode, seed);
					continue;
				}

				const GGL::PolicyMatchSnapshot state = GGL::SamplePolicyMatchStartState(
					GGL::PolicyMatchStartType::Kickoff, mode, RocketSim::CAR_CONFIG_OCTANE, seed, config.engine);

				std::ofstream out(file);
				out << Encode(state, mode, seed).dump(2);
				out.close();
				if (!out)
					throw std::runtime_error("Cannot write frozen benchmark start");

				const GGL::PolicyMatchSnapshot reloaded = ReadBenchmarkStartingState(config.startingStateDirectory, mode, seed);
				if (Encode(state, mode, seed) != Encode(reloaded, mode, seed))
					throw std::runtime_error("Frozen start roundtrip mismatch");
			}
		}
	}
}
