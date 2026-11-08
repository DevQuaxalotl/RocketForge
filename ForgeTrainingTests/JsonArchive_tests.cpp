#include <gtest/gtest.h>

#include "TestSupport.h"

#include <GigaLearnCPP/Util/WelfordStat.h>
#include <GigaLearnInfer/JsonArchive.h>
#include "RewardAnalysis/RewardAnalysisJson.h"

#include <nlohmann/json.hpp>

#include <fstream>

using nlohmann::json;

namespace {

	struct SimpleConfig {
		std::string name = "default";
		int count = 7;
		float ratio = 0.5f;
		bool enabled = true;
		uint64_t timesteps = 0;
		std::filesystem::path outputPath;

		void Serialize(GGL::JsonArchive& ar) {
			ar.Field("name", name);
			ar.Field("count", count);
			ar.Field("ratio", ratio);
			ar.Field("enabled", enabled);
			ar.Field("timesteps", timesteps);
			ar.Field("outputPath", outputPath);
		}
	};
}

TEST(JsonArchive, SavesPrimitiveFields) {
	SimpleConfig config;
	config.name = "stage-one";
	config.count = 42;
	config.ratio = 1.25f;
	config.enabled = false;
	config.timesteps = 9000000000ull;
	config.outputPath = "Models/V1";

	json j;
	GGL::JsonSave(config, j);

	ASSERT_EQ(j["name"].get<std::string>(), "stage-one");
	ASSERT_EQ(j["count"].get<int>(), 42);
	ASSERT_EQ(j["ratio"].get<float>(), 1.25f);
	ASSERT_EQ(j["enabled"].get<bool>(), false);
	ASSERT_EQ(j["timesteps"].get<uint64_t>(), 9000000000ull);
	ASSERT_EQ(j["outputPath"].get<std::string>(), "Models/V1");
}

TEST(JsonArchive, LoadsPrimitiveFields) {
	const json j = {
		{ "name", "loaded" }, { "count", 3 }, { "ratio", 2.5f },
		{ "enabled", false }, { "timesteps", 123ull }, { "outputPath", "a/b" },
	};

	SimpleConfig config;
	std::vector<std::string> errors = GGL::JsonLoad(j, config);

	ASSERT_TRUE(errors.empty());
	ASSERT_EQ(config.name, "loaded");
	ASSERT_EQ(config.count, 3);
	ASSERT_EQ(config.ratio, 2.5f);
	ASSERT_EQ(config.enabled, false);
	ASSERT_EQ(config.timesteps, 123ull);
	ASSERT_EQ(config.outputPath.string(), "a/b");
}

TEST(JsonArchive, MissingKeyKeepsInClassDefault) {
	const json j = { { "name", "only-name" } };

	SimpleConfig config;
	std::vector<std::string> errors = GGL::JsonLoad(j, config);

	ASSERT_TRUE(errors.empty());
	ASSERT_EQ(config.name, "only-name");
	ASSERT_EQ(config.count, 7);           // untouched in-class default
	ASSERT_EQ(config.enabled, true);
}

TEST(JsonArchive, ExplicitDefaultAppliedWhenMissing) {
	struct WithDefault {
		std::string branch;
		void Serialize(GGL::JsonArchive& ar) { ar.Field("branch", branch, std::string("main")); }
	};

	WithDefault value;
	std::vector<std::string> errors = GGL::JsonLoad(json::object(), value);

	ASSERT_TRUE(errors.empty());
	ASSERT_EQ(value.branch, "main");
}

TEST(JsonArchive, WrongTypeRecordsErrorAndKeepsDefault) {
	const json j = { { "count", "not-a-number" } };

	SimpleConfig config;
	std::vector<std::string> errors = GGL::JsonLoad(j, config);

	ASSERT_EQ(errors.size(), size_t(1));
	ASSERT_TRUE(errors[0].find("count") != std::string::npos);
	ASSERT_EQ(config.count, 7);
}

TEST(JsonArchive, SaveModeNeverMutatesMembers) {
	SimpleConfig config;
	config.name = "untouched";

	json j;
	GGL::JsonSave(config, j);

	ASSERT_EQ(config.name, "untouched");
	ASSERT_EQ(config.count, 7);
}

namespace {

	struct InnerConfig {
		int depth = 1;
		std::string label = "inner";

		void Serialize(GGL::JsonArchive& ar) {
			ar.Field("depth", depth);
			ar.Field("label", label);
		}
	};

	struct OuterConfig {
		InnerConfig inner;
		std::vector<InnerConfig> children;
		std::optional<InnerConfig> maybeInner;
		std::vector<int> numbers;
		std::map<std::string, float> weights;

		void Serialize(GGL::JsonArchive& ar) {
			ar.Field("inner", inner);
			ar.Field("children", children);
			ar.Field("maybeInner", maybeInner);
			ar.Field("numbers", numbers);
			ar.Field("weights", weights);
		}
	};

	// Stand-in for a third-party type we cannot modify (RocketSim::Vec pattern).
	struct ForeignPoint {
		float x = 0.f, y = 0.f;
	};

	void SerializeFields(GGL::JsonArchive& ar, ForeignPoint& point) {
		ar.Field("x", point.x);
		ar.Field("y", point.y);
	}

	struct HolderOfForeign {
		ForeignPoint point;
		void Serialize(GGL::JsonArchive& ar) { ar.Field("point", point); }
	};
}

TEST(JsonArchive, RoundTripsNestedAndContainers) {
	OuterConfig outer;
	outer.inner.depth = 5;
	outer.children = { InnerConfig{ 2, "a" }, InnerConfig{ 3, "b" } };
	outer.maybeInner = InnerConfig{ 9, "opt" };
	outer.numbers = { 1, 2, 3 };
	outer.weights = { { "speed", 0.5f }, { "goal", 2.f } };

	json j;
	GGL::JsonSave(outer, j);

	OuterConfig loaded;
	std::vector<std::string> errors = GGL::JsonLoad(j, loaded);

	ASSERT_TRUE(errors.empty());
	ASSERT_EQ(loaded.inner.depth, 5);
	ASSERT_EQ(loaded.children.size(), size_t(2));
	ASSERT_EQ(loaded.children[1].label, "b");
	ASSERT_TRUE(loaded.maybeInner.has_value());
	ASSERT_EQ(loaded.maybeInner->depth, 9);
	ASSERT_EQ(loaded.numbers.size(), size_t(3));
	ASSERT_EQ(loaded.numbers, outer.numbers);
	ASSERT_EQ(loaded.weights.size(), size_t(2));
	ASSERT_EQ(loaded.weights.at("speed"), 0.5f);
	ASSERT_EQ(loaded.weights.at("goal"), 2.f);
}

TEST(JsonArchive, NulloptSavesAsNullAndLoadsBack) {
	OuterConfig outer;
	outer.maybeInner.reset();

	json j;
	GGL::JsonSave(outer, j);
	ASSERT_TRUE(j["maybeInner"].is_null());

	OuterConfig loaded;
	loaded.maybeInner = InnerConfig{};   // pre-set to prove null clears it
	GGL::JsonLoad(j, loaded);
	ASSERT_FALSE(loaded.maybeInner.has_value());
}

TEST(JsonArchive, AdlSerializeFieldsRoundTrips) {
	HolderOfForeign holder;
	holder.point = { 3.f, 4.f };

	json j;
	GGL::JsonSave(holder, j);
	ASSERT_EQ(j["point"]["x"].get<float>(), 3.f);

	HolderOfForeign loaded;
	std::vector<std::string> errors = GGL::JsonLoad(j, loaded);
	ASSERT_TRUE(errors.empty());
	ASSERT_EQ(loaded.point.y, 4.f);
}

TEST(JsonArchive, VectorElementErrorHasIndexedPath) {
	const json j = { { "children", json::array({ json{ { "depth", "bad" } } }) } };

	OuterConfig loaded;
	std::vector<std::string> errors = GGL::JsonLoad(j, loaded);

	ASSERT_EQ(errors.size(), size_t(1));
	ASSERT_TRUE(errors[0].find("children[0].depth") != std::string::npos);
}

TEST(JsonArchive, ChildScopeGroupsFlatMembers) {
	struct WithScope {
		std::string project, runId;

		void Serialize(GGL::JsonArchive& ar) {
			GGL::JsonArchive wandb = ar.Child("wandb");
			wandb.Field("project", project);
			wandb.Field("run_id", runId);
		}
	};

	WithScope value;
	value.project = "rocket";
	value.runId = "abc123";

	json j;
	GGL::JsonSave(value, j);
	ASSERT_EQ(j["wandb"]["project"].get<std::string>(), "rocket");

	WithScope loaded;
	GGL::JsonLoad(j, loaded);
	ASSERT_EQ(loaded.runId, "abc123");
}

namespace {

	enum class TestMode { Alpha, Beta, Gamma };

	const GGL::EnumNames<TestMode> TEST_MODE_NAMES = { {
		{ TestMode::Alpha, "alpha" },
		{ TestMode::Beta,  "beta" },
		{ TestMode::Gamma, "gamma" },
	} };

	struct EnumHolder {
		TestMode mode = TestMode::Alpha;
		TestMode intMode = TestMode::Beta;
		std::string renamed;
		std::string nullable;

		void Serialize(GGL::JsonArchive& ar) {
			ar.EnumField("mode", mode, TEST_MODE_NAMES);
			ar.EnumIntField("intMode", intMode);
			ar.FieldWithLegacy("newKey", { "oldKey" }, renamed);
			ar.FieldNullIfEmpty("nullable", nullable);
		}
	};
}

TEST(JsonArchive, EnumFieldSavesString) {
	EnumHolder holder;
	holder.mode = TestMode::Gamma;

	json j;
	GGL::JsonSave(holder, j);
	ASSERT_EQ(j["mode"].get<std::string>(), "gamma");
	ASSERT_EQ(j["intMode"].get<int>(), 1);
}

TEST(JsonArchive, EnumFieldLoadsStringAndInt) {
	const json j = { { "mode", "beta" }, { "intMode", 2 } };

	EnumHolder holder;
	std::vector<std::string> errors = GGL::JsonLoad(j, holder);

	ASSERT_TRUE(errors.empty());
	ASSERT_TRUE(holder.mode == TestMode::Beta);
	ASSERT_TRUE(holder.intMode == TestMode::Gamma);
}

TEST(JsonArchive, UnknownEnumStringRecordsErrorKeepsDefault) {
	const json j = { { "mode", "delta" } };

	EnumHolder holder;
	std::vector<std::string> errors = GGL::JsonLoad(j, holder);

	ASSERT_EQ(errors.size(), size_t(1));
	ASSERT_TRUE(errors[0].find("mode") != std::string::npos);
	ASSERT_TRUE(holder.mode == TestMode::Alpha);
}

TEST(JsonArchive, LegacyKeyFallbackOnLoadOnly) {
	const json legacy = { { "oldKey", "from-legacy" } };
	EnumHolder holder;
	GGL::JsonLoad(legacy, holder);
	ASSERT_EQ(holder.renamed, "from-legacy");

	// Primary key wins over legacy when both present.
	const json both = { { "newKey", "primary" }, { "oldKey", "stale" } };
	EnumHolder holder2;
	GGL::JsonLoad(both, holder2);
	ASSERT_EQ(holder2.renamed, "primary");

	// Save writes the primary key only.
	holder.renamed = "value";
	json j;
	GGL::JsonSave(holder, j);
	ASSERT_TRUE(j.contains("newKey"));
	ASSERT_FALSE(j.contains("oldKey"));
}

TEST(JsonArchive, NullIfEmptyRoundTrips) {
	EnumHolder holder;
	holder.nullable = "";

	json j;
	GGL::JsonSave(holder, j);
	ASSERT_TRUE(j["nullable"].is_null());

	holder.nullable = "set";
	GGL::JsonSave(holder, j);
	ASSERT_EQ(j["nullable"].get<std::string>(), "set");

	EnumHolder loaded;
	loaded.nullable = "junk";
	GGL::JsonLoad(json{ { "nullable", nullptr } }, loaded);
	ASSERT_EQ(loaded.nullable, "");
}

TEST(JsonArchive, FileHelpersRoundTrip) {
	namespace fs = std::filesystem;
	const fs::path tempPath = RocketForgeTests::TestTempDir() / "jsonarchive.json";

	EnumHolder holder;
	holder.renamed = "persisted";

	std::string saveError;
	ASSERT_TRUE(GGL::JsonSaveToFile(holder, tempPath, 2, &saveError));

	EnumHolder loaded;
	std::vector<std::string> errors;
	ASSERT_TRUE(GGL::JsonLoadFromFile(tempPath, loaded, errors));
	ASSERT_TRUE(errors.empty());
	ASSERT_EQ(loaded.renamed, "persisted");

	fs::remove(tempPath);

	// Missing file reports failure with an error, and does not throw.
	ASSERT_FALSE(GGL::JsonLoadFromFile(tempPath, loaded, errors));
	ASSERT_FALSE(errors.empty());
}

TEST(JsonArchive, SaveToFilePreservesExistingIndent) {
	namespace fs = std::filesystem;
	const fs::path tempPath = RocketForgeTests::TestTempDir() / "indent.json";

	// No file yet: detection falls back to the default indent.
	ASSERT_EQ(GGL::DetectJsonIndent(tempPath), GGL::kDefaultJsonIndent);

	EnumHolder holder;
	holder.renamed = "indented";

	// Seed the file at a non-default 4-space indent, then confirm detection reads it back.
	ASSERT_TRUE(GGL::JsonSaveToFile(holder, tempPath, 4));
	ASSERT_EQ(GGL::DetectJsonIndent(tempPath), 4);

	// The indent-less overload re-saves while keeping the file's existing 4-space indent.
	ASSERT_TRUE(GGL::JsonSaveToFile(holder, tempPath));
	ASSERT_EQ(GGL::DetectJsonIndent(tempPath), 4);

	fs::remove(tempPath);

	// A brand-new file written by the indent-less overload uses the default indent.
	ASSERT_TRUE(GGL::JsonSaveToFile(holder, tempPath));
	ASSERT_EQ(GGL::DetectJsonIndent(tempPath), GGL::kDefaultJsonIndent);

	fs::remove(tempPath);
}

TEST(JsonArchive, SchemaGatedLoadRejectsWrongVersion) {
	namespace fs = std::filesystem;
	const fs::path tempPath = RocketForgeTests::TestTempDir() / "schema.json";

	{
		std::ofstream fOut(tempPath);
		fOut << R"({ "schemaVersion": 2, "newKey": "x" })";
	}

	GGL::JsonFileLoadResult<EnumHolder> result = GGL::JsonLoadFromFileWithSchema<EnumHolder>(tempPath, "schemaVersion", 1);
	ASSERT_FALSE(result.ok);
	ASSERT_EQ(result.errors.size(), size_t(1));
	ASSERT_TRUE(result.errors[0].find("schemaVersion") != std::string::npos);

	{
		std::ofstream fOut(tempPath);
		fOut << R"({ "schemaVersion": 1, "newKey": "loaded" })";
	}

	result = GGL::JsonLoadFromFileWithSchema<EnumHolder>(tempPath, "schemaVersion", 1);
	ASSERT_TRUE(result.ok);
	ASSERT_EQ(result.value.renamed, "loaded");

	fs::remove(tempPath);

	// Missing file fails with ok = false and a descriptive error.
	result = GGL::JsonLoadFromFileWithSchema<EnumHolder>(tempPath, "schemaVersion", 1);
	ASSERT_FALSE(result.ok);
	ASSERT_FALSE(result.errors.empty());
}

TEST(JsonArchive, SchemaGatedLoadRejectsNonIntegerSchemaKeyWithoutThrowing) {
	namespace fs = std::filesystem;
	const fs::path tempPath = RocketForgeTests::TestTempDir() / "schema_badtype.json";

	{
		std::ofstream fOut(tempPath);
		fOut << R"({ "schemaVersion": "2", "newKey": "x" })";
	}

	// Must return a failure result rather than throwing when the schema key is present
	// but not an integer (e.g. a hand-edited file).
	GGL::JsonFileLoadResult<EnumHolder> result = GGL::JsonLoadFromFileWithSchema<EnumHolder>(tempPath, "schemaVersion", 1);
	ASSERT_FALSE(result.ok);
	ASSERT_FALSE(result.errors.empty());
	ASSERT_TRUE(result.errors[0].find("schemaVersion") != std::string::npos);

	fs::remove(tempPath);
}

TEST(JsonArchive, EnumFieldSaveRecordsErrorForOutOfTableValue) {
	EnumHolder holder;
	holder.mode = static_cast<TestMode>(99);

	json j;
	GGL::JsonArchive ar = GGL::JsonArchive::MakeSaver(j);
	holder.Serialize(ar);

	ASSERT_EQ(j["mode"].get<std::string>(), "");
	ASSERT_EQ(ar.GetErrors().size(), size_t(1));
	ASSERT_TRUE(ar.GetErrors()[0].find("mode") != std::string::npos);
}

namespace {

	struct RequiredFieldHolder {
		std::string requiredName;
		std::string optionalName;

		void Serialize(GGL::JsonArchive& ar) {
			ar.RequiredField("requiredName", requiredName);
			ar.Field("optionalName", optionalName);
		}
	};
}

TEST(JsonArchive, RequiredFieldRecordsPathInMissingRequiredFields) {
	const json j = { { "optionalName", "set" } };

	RequiredFieldHolder holder;
	std::vector<std::string> missingRequired;
	std::vector<std::string> errors = GGL::JsonLoad(j, holder, &missingRequired);

	ASSERT_EQ(missingRequired.size(), size_t(1));
	ASSERT_EQ(missingRequired[0], "requiredName");

	// GetMissingRequiredFields is a subset of GetErrors, by path rather than message text.
	ASSERT_EQ(errors.size(), size_t(1));
	ASSERT_TRUE(errors[0].find("requiredName") != std::string::npos);
}

TEST(JsonArchive, RequiredFieldPresentLeavesMissingRequiredFieldsEmpty) {
	const json j = { { "requiredName", "value" }, { "optionalName", "set" } };

	RequiredFieldHolder holder;
	std::vector<std::string> missingRequired;
	GGL::JsonLoad(j, holder, &missingRequired);

	ASSERT_TRUE(missingRequired.empty());
	ASSERT_EQ(holder.requiredName, "value");
}

TEST(JsonArchive, SaveToFileSurfacesFieldErrorsButStillSucceeds) {
	namespace fs = std::filesystem;
	const fs::path tempPath = RocketForgeTests::TestTempDir() / "jsonarchive_saveerr.json";

	EnumHolder holder;
	holder.mode = static_cast<TestMode>(99);

	std::string saveError;
	// The write itself still succeeds (the file is valid JSON) -- existing callers branch on
	// this return value for IO failure, not for field-level data-modeling issues.
	ASSERT_TRUE(GGL::JsonSaveToFile(holder, tempPath, 2, &saveError));
	ASSERT_TRUE(saveError.find("mode") != std::string::npos);
	ASSERT_TRUE(saveError.find("no entry in the name table") != std::string::npos);

	json written;
	{
		std::ifstream fIn(tempPath);
		written = json::parse(fIn);
	}
	ASSERT_EQ(written["mode"].get<std::string>(), "");

	fs::remove(tempPath);
}

TEST(WelfordStat, SerializeRoundTrip) {
	GGL::WelfordStat stat;
	stat.runningMean = 1.5;
	stat.runningVariance = 0.25;
	stat.count = 10;

	nlohmann::json j = stat.ToJSON();
	ASSERT_EQ(j["mean"].get<double>(), 1.5);

	GGL::WelfordStat loaded;
	loaded.ReadFromJSON(j);
	ASSERT_EQ(loaded.count, stat.count);
}

TEST(BatchedWelfordStat, ReadsItsOwnOutput) {
	GGL::BatchedWelfordStat stat(2);
	stat.runningMeans = { 1.0, 2.0 };
	stat.runningVariances = { 0.1, 0.2 };
	stat.count = 5;

	nlohmann::json j = stat.ToJSON();

	GGL::BatchedWelfordStat loaded(2);
	loaded.ReadFromJSON(j);

	// This is the drift-bug regression test: before the migration, ToJSON wrote
	// "means"/"vars" while ReadFromJSON read "mean"/"var" and lost the data.
	ASSERT_EQ(loaded.runningMeans.size(), size_t(2));
	ASSERT_EQ(loaded.runningMeans[1], 2.0);
	ASSERT_EQ(loaded.count, 5);
}

TEST(RewardAnalysis, JsonRoundTripViaSerialize) {
	RocketForge::RewardAnalysisResult result = {};
	result.agentName = "a";
	result.gaeGamma = 0.95f;

	RocketForge::TeamSizeImpactResult teamResult = {};
	teamResult.label = "1v1";
	teamResult.teamSizeA = 1;

	RocketForge::RewardImpactEntry entry = {};
	entry.name = "TouchBall";
	entry.category = RLGC::RewardCategory::EVENT;
	teamResult.rewards.push_back(entry);
	result.teamSizes.push_back(teamResult);

	nlohmann::json j = result;   // ADL to_json

	ASSERT_EQ(j["team_sizes"][0]["rewards"][0]["category"].get<std::string>(), "EVENT");

	RocketForge::RewardAnalysisResult loaded = j.get<RocketForge::RewardAnalysisResult>();
	ASSERT_EQ(loaded.teamSizes[0].rewards[0].name, "TouchBall");
	ASSERT_TRUE(loaded.teamSizes[0].rewards[0].category == RLGC::RewardCategory::EVENT);
}
