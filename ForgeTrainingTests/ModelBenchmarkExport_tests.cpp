#include "TestSupport.h"

#include <ModelBenchmark/ModelBenchmarkExport.h>
#include <ModelBenchmark/SpreadsheetXmlWriter.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef ROCKETFORGE_TESTS_HAVE_QT
#include "ModelBenchmark/ModelBenchmarkLibrary.h"
#endif

using namespace RocketForgeTraining;
namespace fs = std::filesystem;

namespace
{
	struct WorksheetLayout {
		std::string name;
		std::vector<std::string> header;
	};

	std::string ReadFile(const fs::path& path)
	{
		std::ifstream file(path, std::ios::binary);
		return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
	}

	std::string Between(const std::string& text, size_t& position, const std::string& open, const std::string& close)
	{
		const size_t begin = text.find(open, position);
		if (begin == std::string::npos) {
			position = std::string::npos;
			return {};
		}

		const size_t valueBegin = begin + open.size();
		const size_t end = text.find(close, valueBegin);
		position = end == std::string::npos ? end : end + close.size();
		return text.substr(valueBegin, end - valueBegin);
	}

	// Reads each worksheet's name and first row; the workbooks under test come from known writers, so no XML parser is needed.
	std::vector<WorksheetLayout> ReadLayout(const std::string& xml)
	{
		std::vector<WorksheetLayout> layout;
		size_t position = 0;
		while (true) {
			const std::string name = Between(xml, position, "<Worksheet ss:Name=\"", "\"");
			if (position == std::string::npos)
				break;

			WorksheetLayout sheet{ name, {} };
			const size_t tableEnd = xml.find("</Table>", position);
			size_t rowPosition = position;
			const std::string headerRow = Between(xml, rowPosition, "<Row>", "</Row>");
			if (rowPosition != std::string::npos && rowPosition <= tableEnd) {
				for (size_t cellPosition = 0;;) {
					const std::string cell = Between(headerRow, cellPosition, "<Data ss:Type=\"String\">", "</Data>");
					if (cellPosition == std::string::npos)
						break;

					sheet.header.push_back(cell);
				}
			}

			layout.push_back(std::move(sheet));
		}

		return layout;
	}

	ModelBenchmarkConfig TwoModelConfig()
	{
		ModelBenchmarkConfig config;
		config.run3v3 = false;

		ModelBenchmarkModel alpha;
		alpha.displayName = "Alpha <v1.0>";
		alpha.versionId = "v1.0";

		ModelBenchmarkModel beta;
		beta.displayName = "Beta & \"v1.2\"";
		beta.versionId = "v1.2";

		config.models = { alpha, beta };
		return config;
	}

	ModelBenchmarkResult TwoModelResult(const ModelBenchmarkConfig& config)
	{
		std::vector<ModelBenchmarkGameMetadata> games;
		for (int i = 0; i < 12; ++i) {
			ModelBenchmarkGameMetadata game;
			game.gameId = 1000 + i;
			game.snapshotId = 50 + i / 2;
			game.modelA = 0;
			game.modelB = 1;
			game.blueModel = i % 2 ? 1 : 0;
			game.orangeModel = i % 2 ? 0 : 1;
			game.mode = i < 8 ? 1 : 2;
			game.goalScored = i != 5;
			game.winner = !game.goalScored ? -1 : (i % 3 == 0 ? 1 : 0);
			game.goalTimeSeconds = game.goalScored ? (i % 4 == 0 ? 6.25 : 47.125) : 300.0;
			game.goalTiming = !game.goalScored ? ModelBenchmarkGoalTiming::NoGoal
				: (i % 4 == 0 ? ModelBenchmarkGoalTiming::Kickoff : ModelBenchmarkGoalTiming::PostKickoff);
			game.includedInRating = game.goalScored;
			game.mirrorCompleted = true;
			games.push_back(game);
		}

		ModelBenchmarkStatTotals totals;
		totals.goals = 7;
		totals.shots = 13.5;
		totals.gameTimeSum = 1234.5678;
		totals.playerTimeSum = 1234.5678;
		totals.speedSum = 1.0e7 / 3.0;
		totals.stepSamples = 120000;
		totals.playerSamples = 240000;
		const std::vector<ModelBenchmarkStatTotalsRow> statTotals = {
			{ 0, 1, 1, totals }, { 1, 0, 1, totals }, { 0, 1, 2, totals }, { 1, 0, 2, totals }
		};

		ModelBenchmarkResult result = SummarizeModelBenchmarkData(config, games, statTotals);
		result.rocketSimVersions = { 2 };
		result.seed = 12345678901234567ull;
		result.predictionConfigurationJson = "{\"a\":\"<b>&\"}";
		result.modelVersionIds = { "v1.0", "v1.2" };
		result.modelCheckpointPaths = { "C:/Fixture/Checkpoints/ckptA", "C:/Fixture/Checkpoints/ckptB" };
		return result;
	}

	std::vector<ModelBenchmarkExportModel> TwoExportModels()
	{
		ModelBenchmarkExportModel alpha;
		alpha.agentName = "example-agent";
		alpha.buildName = "Main Build";
		alpha.stageName = "example-ball-chase";
		alpha.branchName = "main";
		alpha.sourceLabel = "RocketForge";
		alpha.lastTrainedAt = "2026-10-20T12:00:00Z";
		alpha.fingerprint = "sha256-alpha";
		alpha.latestTimesteps = 123456789;

		ModelBenchmarkExportModel beta = alpha;
		beta.buildName = "Side & Build";
		beta.branchName = "experiment";
		beta.fingerprint = "sha256-beta";
		beta.latestTimesteps = 987654321;
		beta.primary = false;

		return { alpha, beta };
	}
}

TEST(ModelBenchmarkExport, EscapesAndSplitsLikeExcel)
{
	std::ostringstream out;
	SpreadsheetXmlWriter writer(out);
	writer.OpenWorksheet("Odd/Name");
	writer.Row({ "a<b&\"c\"" });
	writer.CloseWorksheet();

	const std::vector<std::string> lines(65536, "line");
	writer.Lines("Console", lines);
	writer.Finish();

	const std::string xml = out.str();
	EXPECT_NE(xml.find("<Cell><Data ss:Type=\"String\">a&lt;b&amp;&quot;c&quot;</Data></Cell>"), std::string::npos);

	std::vector<std::string> names;
	for (const WorksheetLayout& sheet : ReadLayout(xml))
		names.push_back(sheet.name);

	// Once a list splits, every sheet is numbered, the first one included.
	const std::vector<std::string> expectedNames = { "Odd_Name", "Console 1", "Console 2" };
	EXPECT_EQ(names, expectedNames);
	EXPECT_EQ(xml.substr(xml.size() - std::string("</Workbook>").size()), "</Workbook>");
}

TEST(ModelBenchmarkExport, MatchesTheGuiExportLayout)
{
	const fs::path fixturePath = fs::path(FORGE_SOURCE_DIR) / "ForgeTrainingTests" / "fixtures" / "model_benchmark_gui_export.xml";
	const std::vector<WorksheetLayout> expected = ReadLayout(ReadFile(fixturePath));
	ASSERT_FALSE(expected.empty());

	const ModelBenchmarkConfig config = TwoModelConfig();
	const ModelBenchmarkResult result = TwoModelResult(config);

	ModelBenchmarkExportInput input;
	input.result = &result;
	input.runId = "run-layout";
	input.models = TwoExportModels();
	input.consoleLines = { "[10:00:00] Benchmark started" };

	const fs::path exportPath = RocketForgeTests::TestTempDir("layout") / "export.xml";
	std::string error;
	ASSERT_TRUE(WriteModelBenchmarkExport(exportPath, input, &error)) << error;

	const std::vector<WorksheetLayout> actual = ReadLayout(ReadFile(exportPath));
	ASSERT_EQ(actual.size(), expected.size());
	for (size_t i = 0; i < expected.size(); ++i) {
		SCOPED_TRACE(expected[i].name);
		EXPECT_EQ(actual[i].name, expected[i].name);
		EXPECT_EQ(actual[i].header, expected[i].header);
	}
}

TEST(ModelBenchmarkExport, ReplacesTheTargetAndReportsUnwritablePaths)
{
	const ModelBenchmarkConfig config = TwoModelConfig();
	const ModelBenchmarkResult result = TwoModelResult(config);

	ModelBenchmarkExportInput input;
	input.result = &result;

	const fs::path folder = RocketForgeTests::TestTempDir("atomic");
	const fs::path exportPath = folder / "export.xml";
	std::ofstream(exportPath) << "stale";

	std::string error;
	ASSERT_TRUE(WriteModelBenchmarkExport(exportPath, input, &error)) << error;
	EXPECT_EQ(ReadFile(exportPath).rfind("<?xml", 0), size_t(0));

	size_t fileCount = 0;
	for ([[maybe_unused]] const fs::directory_entry& entry : fs::directory_iterator(folder))
		++fileCount;

	EXPECT_EQ(fileCount, size_t(1));

	EXPECT_FALSE(WriteModelBenchmarkExport(folder / "missing" / "export.xml", input, &error));
	EXPECT_FALSE(error.empty());
}

TEST(ModelBenchmarkExport, DefaultsMatchTheGuiFolderAndName)
{
	EXPECT_EQ(DefaultModelBenchmarkExportFolder(), fs::path(TRAINING_ROOT).parent_path() / "Model Benchmarks");
	EXPECT_TRUE(std::regex_match(DefaultModelBenchmarkExportName(), std::regex(R"(model-benchmark-\d{8}-\d{6}\.xml)")));
}

#ifdef ROCKETFORGE_TESTS_HAVE_QT
TEST(ModelBenchmarkExport, RoundTripsThroughLibraryImport)
{
	const ModelBenchmarkConfig config = TwoModelConfig();
	const ModelBenchmarkResult result = TwoModelResult(config);

	ModelBenchmarkExportInput input;
	input.result = &result;
	input.runId = "run-roundtrip";
	input.models = TwoExportModels();

	const fs::path exportPath = RocketForgeTests::TestTempDir("roundtrip") / "export.xml";
	std::string writeError;
	ASSERT_TRUE(WriteModelBenchmarkExport(exportPath, input, &writeError)) << writeError;

	QString error;
	const ModelBenchmarkLibraryImport imported = ModelBenchmarkLibrary::ReadExcelExport(exportPath, &error);
	ASSERT_TRUE(error.isEmpty()) << error.toStdString();
	EXPECT_TRUE(imported.warnings.isEmpty()) << imported.warnings.join("\n").toStdString();
	EXPECT_EQ(imported.runId, QString("run-roundtrip"));

	ASSERT_EQ(imported.models.size(), size_t(2));
	for (size_t i = 0; i < imported.models.size(); ++i) {
		SCOPED_TRACE(i);
		EXPECT_EQ(imported.models[i].displayName.toStdString(), result.modelNames[i]);
		EXPECT_EQ(imported.models[i].fingerprint.toStdString(), input.models[i].fingerprint);
		EXPECT_EQ(imported.models[i].primary, input.models[i].primary);
		EXPECT_EQ(imported.models[i].latestTimesteps, input.models[i].latestTimesteps);
	}

	ASSERT_EQ(imported.result.games.size(), result.games.size());

	const ModelBenchmarkResult rebuilt = SummarizeModelBenchmarkData(config, imported.result.games, imported.result.statTotals);
	EXPECT_EQ(rebuilt.scoredGames, result.scoredGames);
	ASSERT_EQ(rebuilt.rankings.size(), result.rankings.size());
	for (size_t i = 0; i < result.rankings.size(); ++i) {
		SCOPED_TRACE(result.rankings[i].modelName);
		EXPECT_EQ(rebuilt.rankings[i].modelIndex, result.rankings[i].modelIndex);
		EXPECT_EQ(rebuilt.rankings[i].rank, result.rankings[i].rank);
		EXPECT_NEAR(rebuilt.rankings[i].relativeElo, result.rankings[i].relativeElo, 1e-9);
	}
}
#else
TEST(ModelBenchmarkExport, RequiresQtForLibraryImport) { GTEST_SKIP() << "Qt is unavailable"; }
#endif
