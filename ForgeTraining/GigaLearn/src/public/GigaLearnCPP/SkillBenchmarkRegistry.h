#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <GigaLearnInfer/Framework.h>
#include <GigaLearnInfer/JsonArchive.h>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

namespace GGL {

	/** Lightweight index entry for a saved skill benchmark folder. **/
	struct SkillBenchmarkSummary {
		std::string id;
		std::string sourceStageName;
		std::string sourceVersionId;
		std::string createdAt;
		std::string path;
		uint64_t timesteps = 0;

		void Serialize(JsonArchive& ar) {
			ar.Field("id", id);
			ar.Field("source_stage_name", sourceStageName);
			ar.Field("source_version_id", sourceVersionId);
			ar.Field("created_at", createdAt);
			ar.Field("timesteps", timesteps);
			ar.Field("path", path);
		}
	};

	/**
	 * Persistent index of shared skill benchmark artifacts.
	 * Benchmarks are stored outside individual versions, so the registry keeps
	 * the source version linkage needed for cleanup and UI display.
	 **/
	class SkillBenchmarkRegistry {
	public:
		static constexpr const char* REGISTRY_FILE_NAME = "benchmarks.json";
		static constexpr const char* ANCHOR_BENCHMARK_ID_KEY = "anchor_benchmark_id";

		SkillBenchmarkRegistry(
			std::filesystem::path registryPath,
			std::filesystem::path rootFolder,
			std::string agentName = {}) :
			registryPath(std::move(registryPath)),
			rootFolder(std::move(rootFolder)),
			agentName(std::move(agentName)) {
			Load();
		}

		void Load() {
			if (std::filesystem::exists(registryPath)) {
				std::ifstream fIn(registryPath);
				if (fIn.good()) {
					data = nlohmann::json::parse(fIn, nullptr, false);
					if (!data.is_discarded() && data.is_object()) {
						EnsureRoot();
						return;
					}
				}
			}

			data = {};
			EnsureRoot();
		}

		void Save() const {
			std::filesystem::create_directories(registryPath.parent_path());
			std::ofstream fOut(registryPath);
			if (fOut.good())
				fOut << data.dump(4);
		}

		void UpsertBenchmark(const SkillBenchmarkSummary& benchmark) {
			EnsureRoot();

			// The registry stores paths relative to rootFolder; swap that in before saving.
			SkillBenchmarkSummary relativeBenchmark = benchmark;
			relativeBenchmark.path = MakeRelativePath(benchmark.path);

			nlohmann::json value;
			JsonSave(relativeBenchmark, value);

			if (nlohmann::json* existing = FindBenchmark(benchmark.id)) {
				*existing = value;
			} else {
				data["benchmarks"].push_back(value);
			}

			Save();
		}

		void RemoveByPath(const std::filesystem::path& folder) {
			EnsureRoot();
			const std::string relPath = MakeRelativePath(folder);
			auto& benchmarks = data["benchmarks"];
			for (auto it = benchmarks.begin(); it != benchmarks.end();) {
				if (it->value("path", "") == relPath)
					it = benchmarks.erase(it);
				else
					++it;
			}
			Save();
		}

		std::vector<SkillBenchmarkSummary> RemoveBySource(
			const std::string& sourceStageName,
			const std::string& sourceVersionId) {
			EnsureRoot();

			std::vector<SkillBenchmarkSummary> removed = {};
			auto& benchmarks = data["benchmarks"];
			for (auto it = benchmarks.begin(); it != benchmarks.end();) {
				bool matchesStage = it->value("source_stage_name", "") == sourceStageName;
				bool matchesVersion = it->value("source_version_id", "") == sourceVersionId;
				if (!matchesStage || !matchesVersion) {
					++it;
					continue;
				}

				SkillBenchmarkSummary summary = {};
				summary.id = it->value("id", "");
				summary.sourceStageName = it->value("source_stage_name", "");
				summary.sourceVersionId = it->value("source_version_id", "");
				summary.createdAt = it->value("created_at", "");
				summary.timesteps = it->value("timesteps", (uint64_t)0);
				summary.path = it->value("path", "");
				removed.push_back(summary);
				it = benchmarks.erase(it);
			}

			Save();
			return removed;
		}

		/** Benchmark the skill test measures anchor MMR against; empty when none has been chosen. **/
		std::string GetAnchorBenchmarkId() const {
			auto itr = data.find(ANCHOR_BENCHMARK_ID_KEY);
			return itr != data.end() && itr->is_string() ? itr->get<std::string>() : std::string();
		}

		void SetAnchorBenchmarkId(const std::string& id) {
			EnsureRoot();
			data[ANCHOR_BENCHMARK_ID_KEY] = id;
			Save();
		}

		std::vector<SkillBenchmarkSummary> ListBenchmarks() const {
			std::vector<SkillBenchmarkSummary> result = {};
			for (const auto& benchmark : data.value("benchmarks", nlohmann::json::array())) {
				SkillBenchmarkSummary summary = {};
				JsonLoad(benchmark, summary);
				result.push_back(summary);
			}
			return result;
		}

	private:
		nlohmann::json& EnsureRoot() {
			if (!data.is_object())
				data = nlohmann::json::object();
			data["schema_version"] = data.value("schema_version", 1);
			data["agent"] = data.value("agent", agentName);
			if (!data.contains("benchmarks") || !data["benchmarks"].is_array())
				data["benchmarks"] = nlohmann::json::array();
			return data;
		}

		std::string MakeRelativePath(const std::filesystem::path& path) const {
			std::error_code ec;
			std::filesystem::path rel = std::filesystem::relative(path, rootFolder, ec);
			if (ec)
				rel = path;
			return rel.generic_string();
		}

		nlohmann::json* FindBenchmark(const std::string& benchmarkId) {
			for (auto& benchmark : data["benchmarks"]) {
				if (benchmark.value("id", "") == benchmarkId)
					return &benchmark;
			}
			return nullptr;
		}

		std::filesystem::path registryPath;
		std::filesystem::path rootFolder;
		std::string agentName;
		nlohmann::json data;
	};
}
