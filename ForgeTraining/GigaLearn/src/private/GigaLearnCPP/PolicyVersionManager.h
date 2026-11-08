#pragma once

#include "GigaLearnInfer/Models.h"
#include <GigaLearnInfer/Agent.h>
#include <GigaLearnCPP/TrainingStorage.h>
#include <GigaLearnCPP/SkillTrackerConfig.h>
#include <GigaLearnCPP/Eval/PolicyMatchRunner.h>
#include <GigaLearnCPP/Util/Report.h>
#include <GigaLearnCPP/Util/RenderSender.h>

#include <RLGymCPP/EnvSet/EnvSet.h>

#include "SkillFixedSeries.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <functional>
#include <optional>
#include <set>

namespace GGL {

	/** Skill estimate for a specific playlist or evaluation mode. **/
	struct SkillEstimate {
		float mu = 0;
		float sigma = 0;
	};

	/**
	 * Aggregated skill estimates for a policy version or benchmark.
	 * The map key is the playlist label (for example "1v1" or "2v2").
	 **/
	struct SkillRating {
		std::map<std::string, SkillEstimate> data;

		static const std::vector<std::string>& ModeNames() {
			static const std::vector<std::string> modes = { "1v1", "2v2", "3v3" };
			return modes;
		}

		static std::string GetModeName(const RLGC::GameState& state) {
			int playersOnTeams[2] = { 0, 0 };
			for (auto& player : state.players)
				playersOnTeams[(int)player.team]++;

			int minPlayersOnTeam = RS_MIN(playersOnTeams[0], playersOnTeams[1]);
			int maxPlayersOnTeam = RS_MAX(playersOnTeams[0], playersOnTeams[1]);

			std::string name = RS_STR(minPlayersOnTeam << "v" << maxPlayersOnTeam);
			return name;
		}

		SkillEstimate& GetEstimate(const std::string& name, const SkillTrackerConfig& config) {
			if (data.contains(name)) {
				return data[name];
			} else {
				data[name] = SkillEstimate{ config.initialMu, config.initialSigma };
				return data[name];
			}
		}

		SkillEstimate& GetEstimate(const RLGC::GameState& state, const SkillTrackerConfig& config) {
			return GetEstimate(GetModeName(state), config);
		}

		void EnsureDefaultModes(const SkillTrackerConfig& config) {
			for (const std::string& mode : ModeNames())
				GetEstimate(mode, config);
		}

		/** Seeds default modes from the previous policy rating, falling back to the unknown prior. **/
		void SeedFromPrevious(const SkillRating& previousRatings, const SkillTrackerConfig& config) {
			data = previousRatings.data;
			for (const std::string& mode : ModeNames()) {
				if (!previousRatings.data.contains(mode))
					data[mode] = SkillEstimate{ config.initialMu, config.initialSigma };
			}
		}

		/** Seeds a fresh skill test from prior means while resetting uncertainty. **/
		void SeedForSkillTest(const SkillRating& previousRatings, const SkillTrackerConfig& config) {
			data = {};
			for (const std::string& mode : ModeNames()) {
				auto itr = previousRatings.data.find(mode);
				if (itr == previousRatings.data.end()) {
					data[mode] = SkillEstimate{ config.initialMu, config.initialSigma };
				} else {
					data[mode] = SkillEstimate{ itr->second.mu, config.initialSigma };
				}
			}
		}

		/**
		 * Seeds a skill test that follows an earlier one, carrying mu
		 * and growing sigma by the drift accumulated over elapsedTimesteps (in quadrature, capped at the unknown-prior sigma).
		 **/
		void SeedForSkillTestWithDrift(const SkillRating& previousRatings, uint64_t elapsedTimesteps, const SkillTrackerConfig& config) {
			const double driftSigma = (double)config.ratingDriftSigmaPer100MTimesteps;
			const double driftVariance = driftSigma * driftSigma * ((double)elapsedTimesteps / 1e8);

			data = {};
			for (const std::string& mode : ModeNames()) {
				auto itr = previousRatings.data.find(mode);
				if (itr == previousRatings.data.end()) {
					data[mode] = SkillEstimate{ config.initialMu, config.initialSigma };
				} else {
					const double previousSigma = itr->second.sigma;
					const float sigma = (float)std::sqrt(previousSigma * previousSigma + driftVariance);
					data[mode] = SkillEstimate{ itr->second.mu, RS_MIN(config.initialSigma, sigma) };
				}
			}
		}

		// NOTE: intentionally not on JsonArchive. Load needs a SkillTrackerConfig to fill in
		// per-mode defaults (mu/sigma) for keys that are present but incomplete, and also
		// accepts a legacy bare-number-mu shape for old files; Serialize(JsonArchive&) has no
		// way to thread that extra config through the generic std::map<std::string, T> path.
		nlohmann::json ToJSON() const {
			nlohmann::json j = {};
			for (auto pair : data) {
				j[pair.first] = {
					{ "mu", pair.second.mu },
					{ "sigma", pair.second.sigma }
				};
			}
			return j;
		}

		void ReadFromJSON(const nlohmann::json& j, const SkillTrackerConfig& config) {
			data = {};
			for (auto& pair : j.items()) {
				if (pair.value().is_object()) {
					data[pair.key()] = SkillEstimate{
						pair.value().value("mu", config.initialMu),
						pair.value().value("sigma", config.initialSigma)
					};
				} else if (pair.value().is_number()) {
					data[pair.key()] = SkillEstimate{ pair.value().get<float>(), config.initialSigma };
				}
			}
		}
	};

	struct PolicyVersion {
		uint64_t timesteps;
		ModelSet models;
		SkillRating ratings;
		std::string label;
		std::string sourceAgentName;
		std::string sourceBuildName;
		std::string sourceStageName;
		std::string sourceVersionId;
		std::string staticSelectionKey;
		std::filesystem::path rootFolder;
		/** True only for policies produced by this manager and saved under versionId. **/
		bool saveWithCurrentVersion = false;
		/** Source runtime for a frozen policy whose obs/actions differ from the learner. **/
		std::unique_ptr<Agent> inferenceAgent;
		int inferenceObsSize = 0;
		int inferenceNumActions = 0;
		bool inferenceDeterministic = false;
		int inferenceCadenceRatio = 1;
		int inferenceActionApplyStep = 0;
		int inferenceActionApplyTick = 0;
		bool inferenceRequiresEarlyAction = false;
	};

	/** Stable identity used by episode-sticky opponents while the version vector changes. **/
	struct PolicyVersionKey {
		std::string sourceVersionId;
		uint64_t timesteps = 0;

		bool operator==(const PolicyVersionKey&) const = default;
		bool operator<(const PolicyVersionKey& other) const {
			return sourceVersionId != other.sourceVersionId
				? sourceVersionId < other.sourceVersionId
				: timesteps < other.timesteps;
		}
	};

	/**
	 * Saved benchmark model used by the skill tracker.
	 * folderName is kept separate from the logical id so benchmark storage can
	 * move between roots without losing identity.
	 **/
	struct SkillBenchmark {
		std::string id;
		std::string folderName;
		std::string createdAt;
		std::string sourceStageName;
		std::string sourceVersionId;
		uint64_t timesteps;
		std::filesystem::path folder;
		ModelSet models;
		SkillRating ratings;
		bool modelsSaved = false;
	};

	/** Opponent runtime for a fixed series; label is used only for logs. */
	struct FixedSeriesOpponent {
		std::string label;
		PolicyMatchRuntime runtime;
	};

	struct PolicyVersionManager {
		std::vector<PolicyVersion> versions;
		/** Versions currently assigned to live arenas and therefore not eligible for pruning. **/
		std::set<PolicyVersionKey> protectedVersionKeys;
		std::filesystem::path saveFolder;
		/** Shared filesystem helper for stage-scoped versions and benchmarks. **/
		TrainingStorage storage;
		/** Optional build-level registry used to load static opponent references without copying models. **/
		std::filesystem::path staticOpponentsRegistryPath;

		/** Entry array LoadStaticOpponents reads from that registry; SKILL_REFERENCES_LIST_KEY loads skill references. **/
		std::string registryListKey = "static_opponents";

		/** Loaded skill references the skill test plays; null or empty when the Build has none. Not owned. **/
		std::vector<PolicyVersion>* skillReferences = nullptr;

		/** Registry the skill references were loaded from, where measured reference ratings are written back. **/
		std::filesystem::path skillReferencesRegistryPath;

		/** When set, loaded static opponents predict the ball with this engine instead of their saved one. **/
		std::optional<RocketSim::EngineId> staticOpponentBallPredictionEngine;

		std::string versionId;
		/** Segment ids whose shared policy artifacts are eligible for lineage-aware loading. **/
		std::vector<std::string> recentPolicyVersionSourceIds;
		std::string sourceStageName;
		int maxVersions;
		uint64_t tsPerVersion;
		bool managePolicyVersions;
		RLGC::EnvSetConfig baseEnvSetConfig;

		//////////////////

		struct {
			SkillTrackerConfig config;

			std::vector<SkillBenchmark> benchmarks;

			uint64_t nextRunTimestep = 0;

			/** Timestep of the last completed skill test; 0 when none has run (or legacy stats). **/
			uint64_t lastRunTimestep = 0;

			SkillRating curRatings = {};

			/** Benchmark the anchor MMR is measured against; PruneBenchmarks keeps it until it is retired. Empty when none. **/
			std::string anchorBenchmarkId;
		} skill;

		RenderSender* renderSender;

		PolicyVersionManager(
			std::filesystem::path saveFolder, int maxVersions, uint64_t tsPerVersion,
			const SkillTrackerConfig& skillTrackerConfig, const RLGC::EnvSetConfig& envSetConfig,
			bool managePolicyVersions, RenderSender* renderSender = NULL);

		// NOTE: Passed models should not be already cloned
		PolicyVersion& AddVersion(ModelSet modelsToClone, uint64_t timesteps);
		void ClearVersions();
		PolicyVersion* FindVersion(const PolicyVersionKey& key);
		void SetProtectedVersions(const std::vector<PolicyVersionKey>& keys);
		void PruneInMemoryVersions(const std::optional<PolicyVersionKey>& preserveKey = std::nullopt);

		void SaveVersions(uint64_t curTimesteps);
		void LoadVersions(ModelSet modelsTemplate, uint64_t curTimesteps);
		/** Applies the shared recent-policy cache limit, pruning newer unusable entries first. **/
		void PruneSavedVersions(uint64_t curTimesteps);
		void LoadStaticOpponents(
			ModelSet modelsTemplate,
			const std::function<std::unique_ptr<Agent>(const std::string&)>& agentFactory = {});

		SkillBenchmark& AddBenchmark(ModelSet modelsToClone, uint64_t timesteps);
		void SaveBenchmarks();
		void LoadBenchmarks(ModelSet modelsTemplate, uint64_t curTimesteps);
		/** Ensures an initial fixed skill anchor exists before training advances. **/
		bool EnsureInitialSkillBenchmark(ModelSet modelsToClone, uint64_t curTimesteps);
		void PruneBenchmarks();

		void SortVersions();

		/** Selects fixed benchmark anchors for a skill mode. **/
		static std::vector<int> BuildSkillBenchmarkPool(
			const SkillRating& currentRatings,
			const std::string& mode,
			const std::vector<SkillBenchmark>& benchmarks,
			const SkillTrackerConfig& config,
			int targetCount);

		/** Timesteps since the last skill test; 0 when none has run or when resuming from an earlier checkpoint. **/
		static uint64_t GetTimestepsSinceLastSkillTest(uint64_t totalTimesteps, uint64_t lastRunTimestep);

		/** Updates current skill against a fixed benchmark anchor. **/
		static void UpdateSkillEstimateAgainstFixedOpponent(
			SkillEstimate& current,
			const SkillEstimate& opponent,
			bool currentWon,
			const SkillTrackerConfig& config);

		/**
		 * Index of the benchmark to measure anchor MMR against, or -1 when none qualifies.
		 * Keeps the current anchor while the current policy is not lopsided against it by config.anchorRetireExpectedWinProb;
		 * otherwise picks the highest-rated benchmark that is not.
		 **/
		static int SelectAnchorBenchmark(
			const std::vector<SkillBenchmark>& benchmarks,
			const std::string& currentAnchorId,
			const SkillRating& currentRatings,
			const SkillTrackerConfig& config);

		/** Report-key form of a skill reference's display name: each run of '/' and whitespace becomes one '-'. **/
		static std::string MakeReferenceMetricLabel(const std::string& displayName);

		/**
		 * Fixed-series runtime for a skill reference:
		 * its source agent's obs, actions and timing when it has one, otherwise the learner's.
		 * The runtime borrows the reference's models and agent, so it must not outlive the reference.
		 **/
		FixedSeriesOpponent MakeSkillReferenceOpponent(class PPOLearner* ppo, PolicyVersion& reference, int teamSize) const;

		/** Index of the benchmark with this id; -1 when it is no longer present. **/
		static int FindBenchmarkById(const std::vector<SkillBenchmark>& benchmarks, const std::string& id);

		/**
		 * Stores one mode's MMR on the skill reference whose PolicyVersion::staticSelectionKey is selectionKey,
		 * leaving every other field of the registry as it was.
		 * Logs and changes nothing when no reference matches.
		 **/
		static void WriteSkillReferenceRating(
			const std::filesystem::path& registryPath,
			const std::string& selectionKey,
			const std::string& mode,
			float rating);

		/**
		 * Reports MMR = baseRating + the current policy's rating gap from its record against a fixed opponent,
		 * under ratingKey and sigmaKey, and logs the record beside it.
		 * Returns false, reporting nothing, when no game was scored.
		 **/
		static bool ReportFixedOpponentMMR(
			Report& report,
			const std::string& ratingKey,
			const std::string& sigmaKey,
			const std::string& opponentName,
			const std::string& modeName,
			float baseRating,
			const FixedSeriesRecord& record,
			const SkillTrackerConfig& config);

		void RunSkillMatches(class PPOLearner* ppo, Report& report, uint64_t totalTimesteps);

		/**
		 * Plays every opponent from the subject's side; subject defaults to the current policy.
		 * Records are from the subject's perspective.
		 * Games follow BuildFixedSeriesGames seeded by skill.config.fixedMatchSeed, and every scheduled game runs.
		 */
		std::vector<FixedSeriesRecord> RunFixedSkillSeries(
			class PPOLearner* ppo,
			const std::vector<FixedSeriesOpponent>& opponents,
			int teamSize,
			int gamesPerOpponent,
			const FixedSeriesOpponent* subject = nullptr);

		void OnIteration(class PPOLearner* ppo, Report& report, int64_t totalTimesteps, int64_t prevTotalTimesteps);

		void AddRunningStatsToJSON(nlohmann::json& json);
		void LoadRunningStatsFromJSON(const nlohmann::json& json);

		~PolicyVersionManager();
	};
}
