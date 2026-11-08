#include "TestSupport.h"
#include <GigaLearnInfer/AgentSetupSnapshot.h>

#ifdef ROCKETFORGE_TESTS_HAVE_QT
#include "ModelBenchmark/ModelBenchmarkLibrary.h"
#include <ModelBenchmark/ModelBenchmarkCompatibility.h>

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSqlDatabase>
#include <QSqlQuery>
#include <QTemporaryDir>
#include <QXmlStreamReader>
#include <QUuid>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>

using namespace RocketForgeTraining;
namespace fs = std::filesystem;

namespace
{
    ModelBenchmarkGameMetadata ScoredGame(int winner, uint64_t id)
    {
        ModelBenchmarkGameMetadata game;
        game.gameId = id;
        game.snapshotId = id;
        game.modelA = 0;
        game.modelB = 1;
        game.blueModel = 0;
        game.orangeModel = 1;
        game.mode = 1;
        game.winner = winner;
        game.goalTimeSeconds = 20.0;
        game.goalTiming = ModelBenchmarkGoalTiming::PostKickoff;
        game.goalScored = true;
        game.includedInRating = true;
        game.mirrorCompleted = true;
        return game;
    }

    ModelBenchmarkLibraryModel SavedModel(
        const QString& name,
        const QString& version,
        const QString& fingerprint)
    {
        ModelBenchmarkLibraryModel model;
        model.displayName = name;
        model.agentName = "TestAgent";
        model.buildName = "TestBuild";
        model.stageName = "TestStage";
        model.branchName = "test-branch";
        model.versionId = version;
        model.sourceLabel = "RocketForge";
        model.logicalKey = QStringList{
            model.agentName,
            model.buildName,
            model.stageName,
            model.versionId,
            model.sourceLabel
        }.join('|');
        model.fingerprint = fingerprint;
        model.fingerprintVerified = true;
        return model;
    }
}

TEST(ModelBenchmarkLibrary, SimulatorProvenanceSurvivesReopenAndFiltering)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const fs::path path(directory.filePath("engines.sqlite").toStdString());
    QString error;
    ModelBenchmarkLibrary library;
    ASSERT_TRUE(library.Open(path, &error));

    const ModelBenchmarkLibraryModel a = SavedModel("A", "a", "a");
    const ModelBenchmarkLibraryModel b = SavedModel("B", "b", "b");
    const ModelBenchmarkLibraryModel c = SavedModel("C", "c", "c");
    ModelBenchmarkResult run;
    run.modelNames = { "A", "B" };
    run.games = { ScoredGame(0, 1) };
    run.rocketSimVersions = { 2 };
    ASSERT_TRUE(library.SaveRun("v2", run, { a, b }, {}, &error));
    run.rocketSimVersions = { 3 };
    ASSERT_TRUE(library.SaveRun("v3", run, { b, c }, {}, &error));
    run.rocketSimVersions = { 0 };
    ASSERT_TRUE(library.SaveRun("legacy", run, { b, c }, {}, &error));

    const std::vector<ModelBenchmarkLibraryModel> models = library.ListModels(&error);
    ASSERT_EQ(models.size(), size_t(3));
    library.Close();

    // Exercise actual old JSON, where the provenance field did not exist at all.
    const QString connection = QUuid::createUuid().toString();
    {
        QSqlDatabase db = QSqlDatabase::addDatabase("QSQLITE", connection);
        db.setDatabaseName(QString::fromStdString(path.string()));
        ASSERT_TRUE(db.open());
        QSqlQuery query(db);
        ASSERT_TRUE(query.exec("SELECT result_json FROM runs WHERE run_id='legacy'"));
        ASSERT_TRUE(query.next());
        QJsonObject json = QJsonDocument::fromJson(query.value(0).toByteArray()).object();
        json.remove("rocketSimVersions");
        query.prepare("UPDATE runs SET result_json=? WHERE run_id='legacy'");
        query.addBindValue(QJsonDocument(json).toJson(QJsonDocument::Compact));
        ASSERT_TRUE(query.exec());
    }
    QSqlDatabase::removeDatabase(connection);

    const float confidence = 0.95f;
    ASSERT_TRUE(library.OpenReadOnly(path, &error));
    ModelBenchmarkLibraryAggregate result = library.LoadAggregate({ { models[0].id, true }, { models[1].id, true } }, confidence, &error);
    ASSERT_TRUE(error.isEmpty());
    ASSERT_EQ(result.result.rocketSimVersions, std::set<int>({ 2 }));
    ASSERT_EQ(result.contributingRuns, 1);

    result = library.LoadAggregate({ { models[0].id, true }, { models[1].id, false }, { models[2].id, false } }, confidence, &error);
    ASSERT_EQ(result.result.rocketSimVersions, std::set<int>({ 2 }));
    ASSERT_EQ(result.contributingRuns, 1);

    result = library.LoadAggregate({ { models[0].id, true }, { models[1].id, true }, { models[2].id, true } }, confidence, &error);
    ASSERT_TRUE(error.isEmpty());
    ASSERT_EQ(result.contributingRuns, 3);
    ASSERT_EQ(result.result.scoredGames, 3);
    ASSERT_EQ(result.environments.size(), size_t(2));
    ASSERT_TRUE(ModelBenchmarkHasMixedEnvironments(result));
    ASSERT_TRUE(result.result.evaluationIdentity.empty());
    ASSERT_FALSE(result.warnings.isEmpty());

    // Unknown legacy revisions cannot erase the known V2/V3 distinction.
    ModelBenchmarkLibraryFilter filter;
    filter.engineVersion = 3;
    result = library.LoadAggregate({ { models[1].id, true }, { models[2].id, true } }, confidence, &error, {}, filter);
    ASSERT_TRUE(error.isEmpty());
    ASSERT_EQ(result.contributingRuns, 1);
    ASSERT_EQ(result.result.rocketSimVersions, std::set<int>({ 3 }));

    filter.engineVersion = 0;
    const std::vector<ModelBenchmarkLibraryCoverage> coverage = library.LoadCoverage(filter, &error);
    ASSERT_EQ(coverage.size(), size_t(3));

    filter.engineVersion = -1;
    result = library.LoadAggregate({ { models[1].id, true }, { models[2].id, true } }, confidence, &error, {}, filter);
    ASSERT_TRUE(error.isEmpty());
    ASSERT_EQ(result.result.scoredGames, 1);
    ASSERT_EQ(result.result.rocketSimVersions, std::set<int>({ 2 }));
    ASSERT_FALSE(ModelBenchmarkHasMixedEnvironments(result));
}

TEST(ModelBenchmarkLibrary, LegacySimulatorMigrationPreservesRecordedVersionsAndHistory) {
    QTemporaryDir directory;
    const fs::path path(directory.filePath("legacy-v2.sqlite").toStdString());
    ModelBenchmarkLibrary library;
    QString error;
    ASSERT_TRUE(library.Open(path, &error));

    const std::vector<ModelBenchmarkLibraryModel> models{SavedModel("A", "1", "a"), SavedModel("B", "1", "b")};
    ModelBenchmarkResult run;
    run.modelNames = {"A", "B"};
    run.games = {ScoredGame(0, 1)};
    for (const char* id : {"missing", "zero", "scalar-v3", "explicit-v3", "mixed"})
        ASSERT_TRUE(library.SaveRun(id, run, models, {}, &error, "original setup"));

    library.Close();

    const QString connection = QUuid::createUuid().toString();
    {
        QSqlDatabase db = QSqlDatabase::addDatabase("QSQLITE", connection);
        db.setDatabaseName(QString::fromStdString(path.string()));
        ASSERT_TRUE(db.open());
        QSqlQuery query(db);
        ASSERT_TRUE(query.exec("UPDATE library_meta SET schema_version=5"));
        const std::vector<std::pair<QString, QByteArray>> records{
            {"missing", R"({"futureField":{"keep":true}})"},
            {"zero", R"({"rocketSimVersions":[0],"engineVersion":0})"},
            {"scalar-v3", R"({"engineVersion":3})"},
            {"explicit-v3", R"({"rocketSimVersions":[3],"engineRevision":"recorded-v3"})"},
            {"mixed", R"({"rocketSimVersions":[2,3]})"}};
        for (const auto& [id, json] : records) {
            query.prepare("UPDATE runs SET result_json=? WHERE run_id=?");
            query.addBindValue(json);
            query.addBindValue(id);
            ASSERT_TRUE(query.exec());
        }
    }
    QSqlDatabase::removeDatabase(connection);

    // Read-only loading must apply the same interpretation without a schema write.
    ASSERT_TRUE(library.OpenReadOnly(path, &error));
    ModelBenchmarkLibraryFilter filter;
    filter.engineVersion = 2;
    std::vector<ModelBenchmarkLibraryCoverage> coverage = library.LoadCoverage(filter, &error);
    ASSERT_TRUE(error.isEmpty());
    ASSERT_EQ(coverage.size(), size_t(1));
    ASSERT_EQ(coverage[0].games, 2);

    filter.engineVersion = 3;
    coverage = library.LoadCoverage(filter, &error);
    ASSERT_EQ(coverage.size(), size_t(1));
    ASSERT_EQ(coverage[0].games, 2);
    library.Close();

    ASSERT_TRUE(library.Open(path, &error));
    library.Close();
    {
        QSqlDatabase db = QSqlDatabase::addDatabase("QSQLITE", connection);
        db.setDatabaseName(QString::fromStdString(path.string()));
        ASSERT_TRUE(db.open());
        QSqlQuery query(db);
        ASSERT_TRUE(query.exec("SELECT schema_version FROM library_meta"));
        ASSERT_TRUE(query.next());
        EXPECT_EQ(query.value(0).toInt(), 6);
        ASSERT_TRUE(query.exec("SELECT count(*) FROM games"));
        ASSERT_TRUE(query.next());
        EXPECT_EQ(query.value(0).toInt(), 5);
        ASSERT_TRUE(query.exec("SELECT count(*) FROM run_setups WHERE settings_json='original setup'"));
        ASSERT_TRUE(query.next());
        EXPECT_EQ(query.value(0).toInt(), 5);

        ASSERT_TRUE(query.exec("SELECT run_id,result_json FROM runs"));
        while (query.next()) {
            const QString id = query.value(0).toString();
            const QJsonObject object = QJsonDocument::fromJson(query.value(1).toByteArray()).object();
            const QJsonArray versions = object["rocketSimVersions"].toArray();
            EXPECT_EQ(versions, id == "mixed" ? QJsonArray({2, 3}) : id.contains("v3") ? QJsonArray({3}) : QJsonArray({2}));
            if (id == "missing")
                EXPECT_TRUE(object["futureField"].toObject()["keep"].toBool());

            if (id == "explicit-v3")
                EXPECT_EQ(object["engineRevision"].toString(), QString("recorded-v3"));

            EXPECT_FALSE(object.contains("integrationRevision"));
        }
    }
    QSqlDatabase::removeDatabase(connection);

    // Migration is idempotent.
    ASSERT_TRUE(library.Open(path, &error));
    filter.engineVersion = 0;
    filter.restrictEnvironments = true;
    filter.engineVersions = {2, 3};
    filter.runId = "mixed";
    coverage = library.LoadCoverage(filter, &error);
    ASSERT_EQ(coverage.size(), size_t(1));
    EXPECT_EQ(coverage[0].engineVersion, -1);
    EXPECT_TRUE(filter.MatchesEnvironmentSelection(coverage[0].engineVersion, coverage[0].environmentKey));

    filter.engineVersions = {3};
    EXPECT_FALSE(filter.MatchesEnvironmentSelection(coverage[0].engineVersion, coverage[0].environmentKey));
    EXPECT_TRUE(library.LoadCoverage(filter, &error).empty());
}

TEST(ModelBenchmarkLibrary, LegacySimulatorMigrationRollsBackUnreadableHistory) {
    QTemporaryDir directory;
    const fs::path path(directory.filePath("invalid-history.sqlite").toStdString());
    ModelBenchmarkLibrary library;
    QString error;
    ASSERT_TRUE(library.Open(path, &error));
    library.Close();

    const QString connection = QUuid::createUuid().toString();
    {
        QSqlDatabase db = QSqlDatabase::addDatabase("QSQLITE", connection);
        db.setDatabaseName(QString::fromStdString(path.string()));
        ASSERT_TRUE(db.open());
        QSqlQuery query(db);
        ASSERT_TRUE(query.exec("UPDATE library_meta SET schema_version=5"));
        ASSERT_TRUE(query.exec("INSERT INTO runs(run_id,created_at,result_json) VALUES('legacy','old','{}'),('broken','old','broken JSON')"));
    }
    QSqlDatabase::removeDatabase(connection);

    ASSERT_FALSE(library.Open(path, &error));
    ASSERT_FALSE(error.isEmpty());
    {
        QSqlDatabase db = QSqlDatabase::addDatabase("QSQLITE", connection);
        db.setDatabaseName(QString::fromStdString(path.string()));
        ASSERT_TRUE(db.open());
        QSqlQuery query(db);
        ASSERT_TRUE(query.exec("SELECT schema_version FROM library_meta"));
        ASSERT_TRUE(query.next());
        EXPECT_EQ(query.value(0).toInt(), 5);
        ASSERT_TRUE(query.exec("SELECT result_json FROM runs WHERE run_id='legacy'"));
        ASSERT_TRUE(query.next());
        EXPECT_EQ(query.value(0).toByteArray(), QByteArray("{}"));
    }
    QSqlDatabase::removeDatabase(connection);
}

TEST(ModelBenchmarkLibrary, SimulatorVersionsDefaultToHistoricalV2AndKeepExplicitVersions) {
    EXPECT_EQ(ResolveModelBenchmarkSimulatorVersions({}), std::set<int>{2});
    EXPECT_EQ(ResolveModelBenchmarkSimulatorVersions({0}, 3), std::set<int>{3});
    EXPECT_EQ(ResolveModelBenchmarkSimulatorVersions({0, 3}), (std::set<int>{2, 3}));
    EXPECT_EQ(ModelBenchmarkRocketSimVersionsText({0}), std::string("v2"));
}

TEST(ModelBenchmarkLibrary, EnvironmentLabelNamesABuildOnlyWhenOneWasRecorded) {
    ModelBenchmarkLibraryEnvironment environment;
    environment.provenance.rocketSimVersions = {3};
    environment.provenance.evaluationIdentity = "{}";
    EXPECT_FALSE(ModelBenchmarkEnvironmentLabel(environment).contains("build"));

    environment.provenance.integrationRevision = "0123456789abcdef";
    EXPECT_TRUE(ModelBenchmarkEnvironmentLabel(environment).contains("build 0123456789ab"));
}

TEST(ModelBenchmarkLibrary, WorkspaceCoverageAndSnapshots)
{
    QTemporaryDir directory;
    ModelBenchmarkLibrary library;
    QString error;
    const fs::path path(directory.filePath("workspace.sqlite").toStdString());
    ASSERT_TRUE(library.Open(path, &error));

    const ModelBenchmarkLibraryModel a = SavedModel("FixtureA 7.34", "7.34", "new");
    const ModelBenchmarkLibraryModel b = SavedModel("Opponent", "1", "opponent");
    ModelBenchmarkResult run;
    run.modelNames = { "FixtureA 7.34", "Opponent" };
    run.rocketSimVersions = { 3 };
    for (int i = 0; i < 650; ++i)
        run.games.push_back(ScoredGame(i % 2, i + 1));

    // Draws count toward the game target; incomplete mirrors do not.
    run.games.back().goalScored = false;
    run.games.back().winner = -1;
    run.games.back().includedInRating = false;
    ModelBenchmarkGameMetadata unfinished = ScoredGame(0, 651);
    unfinished.mirrorCompleted = false;
    unfinished.includedInRating = false;
    run.games.push_back(unfinished);
    ASSERT_TRUE(library.SaveRun("first", run, { a, b }, {}, &error, "original setup"));
    ASSERT_FALSE(library.SaveRun("first", run, { a, b }, {}, &error, "changed setup"));
    ASSERT_TRUE(library.SavePreset("FixtureA comparison", "preset v1", &error));
    ASSERT_TRUE(library.SavePreset("FixtureA comparison", "preset v2", &error));

    run.rocketSimVersions = { 2 };
    ASSERT_TRUE(library.SaveRun("other engine", run, { a, b }, {}, &error));
    run.rocketSimVersions = { 3 };
    run.stochasticKickoff = true;
    ASSERT_TRUE(library.SaveRun("other kickoff", run, { a, b }, {}, &error));
    run.stochasticKickoff = false;
    run.games = { ScoredGame(0, 1) };
    run.games[0].mode = 2;
    ASSERT_TRUE(library.SaveRun("2v2", run, { a, b }, {}, &error));
    library.Close();

    ASSERT_TRUE(library.OpenReadOnly(path, &error));
    ModelBenchmarkLibraryFilter filter;
    filter.engineVersion = 3;
    filter.stochasticKickoff = 0;
    filter.modeMask = 1;
    std::vector<ModelBenchmarkLibraryCoverage> coverage = library.LoadCoverage(filter, &error);
    ASSERT_TRUE(error.isEmpty());
    ASSERT_EQ(coverage.size(), size_t(1));
    ASSERT_EQ(coverage.front().games, qint64(650));
    ASSERT_EQ(coverage.front().lastSnapshotId, uint64_t(651));

    const std::vector<ModelBenchmarkLibraryModel> models = library.ListModels();
    const float confidence = 0.95f;
    const std::vector<ModelBenchmarkLibrarySelection> selection = { { models[0].id, true }, { models[1].id, true } };
    ModelBenchmarkLibraryAggregate aggregate = library.LoadAggregate(selection, confidence, &error, {}, filter);
    ASSERT_EQ(aggregate.contributingRuns, 1);
    ASSERT_EQ(aggregate.result.scoredGames, 650);
    ASSERT_EQ(aggregate.result.rankings.front().totalRecord.wins + aggregate.result.rankings.front().totalRecord.losses, 649);

    filter.modeMask = 3;
    coverage = library.LoadCoverage(filter, &error);
    ASSERT_EQ(coverage.size(), size_t(2));
    ASSERT_EQ(coverage[1].mode, 2);
    ASSERT_EQ(coverage[1].games, qint64(1));

    const std::vector<ModelBenchmarkWorkspaceRecord> presets = library.ListPresets(&error);
    ASSERT_EQ(presets.size(), size_t(1));
    ASSERT_EQ(presets.front().settingsJson, QString("preset v2"));

    for (const ModelBenchmarkWorkspaceRecord& recorded : library.ListRuns(&error)) {
        if (recorded.id == "first")
            ASSERT_EQ(recorded.settingsJson, QString("original setup"));
    }
}

TEST(ModelBenchmarkLibrary, ImportsSimulatorProvenance)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const fs::path path(directory.filePath("engines.xml").toStdString());
    const std::vector<std::tuple<std::string, std::string, std::set<int>>> cases{
        {"RocketSim Versions", "v2", {2}}, {"RocketSim Versions", "v3", {3}},
        {"RocketSim Versions", "Unknown", {2}}, {"RocketSim Versions", "Unknown, v3", {2, 3}},
        {"Engine Version", "3", {3}}, {"Engine Version", "-1", {2, 3}}};

    for (const auto& [field, value, versions] : cases) {
        {
            std::ofstream out(path);
            out << R"(<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet" xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">
<Worksheet ss:Name="Summary"><Table><Row><Cell><Data>Field</Data></Cell><Cell><Data>Value</Data></Cell></Row>
<Row><Cell><Data>)" << field << R"(</Data></Cell><Cell><Data>)" << value << R"(</Data></Cell></Row></Table></Worksheet>
<Worksheet ss:Name="Models"><Table><Row><Cell><Data>Model Index</Data></Cell><Cell><Data>Display Name</Data></Cell></Row>
<Row><Cell><Data>0</Data></Cell><Cell><Data>A</Data></Cell></Row>
<Row><Cell><Data>1</Data></Cell><Cell><Data>B</Data></Cell></Row></Table></Worksheet></Workbook>)";
        }

        QString error;
        const ModelBenchmarkLibraryImport imported = ModelBenchmarkLibrary::ReadExcelExport(path, &error);
        ASSERT_TRUE(error.isEmpty());
        ASSERT_EQ(imported.result.rocketSimVersions, ResolveModelBenchmarkSimulatorVersions(versions));
    }
}

TEST(ModelBenchmarkLibrary, ComparisonSetsPersistAndSaveAtomically)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const fs::path path(directory.filePath("comparisons.sqlite").toStdString());
    ModelBenchmarkLibrary library;
    QString error;
    ASSERT_TRUE(library.Open(path, &error));

    ModelBenchmarkResult run;
    run.modelNames = { "A", "B", "C" };
    run.games = { ScoredGame(0, 1) };
    ASSERT_TRUE(library.SaveRun(
        "run", run, { SavedModel("A", "a", "a"), SavedModel("B", "b", "b"), SavedModel("C", "c", "c") }, {}, &error));
    const std::vector<ModelBenchmarkLibraryModel> models = library.ListModels(&error);
    ASSERT_EQ(models.size(), size_t(3));

    ModelBenchmarkComparisonSet comparison;
    comparison.name = "  Weekly contenders  ";
    comparison.confidencePercent = 92.5;
    comparison.members = { { models[0].id, true, false }, { models[1].id, false, true } };
    ASSERT_TRUE(library.SaveComparisonSet(comparison, &error));
    ASSERT_TRUE(error.isEmpty());
    ASSERT_TRUE(comparison.id >= 0);
    ASSERT_EQ(comparison.name, QString("Weekly contenders"));
    const ModelBenchmarkComparisonSet original = comparison;
    library.Close();

    ASSERT_TRUE(library.OpenReadOnly(path, &error));
    std::vector<ModelBenchmarkComparisonSet> loaded = library.ListComparisonSets(&error);
    ASSERT_TRUE(error.isEmpty());
    ASSERT_EQ(loaded.size(), size_t(1));
    ASSERT_EQ(loaded[0].members, comparison.members);
    ASSERT_EQ(loaded[0].confidencePercent, 92.5);
    ASSERT_EQ(loaded[0].name, comparison.name);
    ASSERT_FALSE(library.SaveComparisonSet(comparison, &error));
    library.Close();
    ASSERT_TRUE(library.Open(path, &error));

    // A failed update must roll back the renamed set, confidence, and deleted members.
    comparison.name = "Invalid update";
    comparison.confidencePercent = 80.0;
    comparison.members.push_back({ 999999, true, true });
    ASSERT_FALSE(library.SaveComparisonSet(comparison, &error));
    ASSERT_FALSE(error.isEmpty());
    loaded = library.ListComparisonSets(&error);
    ASSERT_EQ(loaded[0].name, original.name);
    ASSERT_EQ(loaded[0].confidencePercent, original.confidencePercent);
    ASSERT_EQ(loaded[0].members, original.members);

    comparison = original;
    comparison.members = { { models[0].id, false, false }, { models[2].id, true, true } };
    ASSERT_TRUE(library.SaveComparisonSet(comparison, &error));
    loaded = library.ListComparisonSets(&error);
    ASSERT_EQ(loaded.size(), size_t(1));
    ASSERT_EQ(loaded[0].members, comparison.members);

    ModelBenchmarkComparisonSet copy = comparison;
    copy.id = -1;
    copy.name = "WEEKLY CONTENDERS";
    ASSERT_FALSE(library.SaveComparisonSet(copy, &error));
    ASSERT_EQ(copy.id, qint64(-1));
    copy.name = "Baseline review";
    ASSERT_TRUE(library.SaveComparisonSet(copy, &error));
    ASSERT_TRUE(copy.id != comparison.id);

    comparison.name = "Renamed comparison";
    comparison.members.clear();
    ASSERT_TRUE(library.SaveComparisonSet(comparison, &error));
    loaded = library.ListComparisonSets(&error);
    ASSERT_EQ(loaded.size(), size_t(2));
    ASSERT_EQ(loaded[0].name, copy.name);
    ASSERT_EQ(loaded[0].members, copy.members);
    ASSERT_TRUE(loaded[1].members.empty());
    ASSERT_EQ(loaded[1].name, comparison.name);

    ModelBenchmarkComparisonSet invalid = copy;
    invalid.name = "  ";
    ASSERT_FALSE(library.SaveComparisonSet(invalid, &error));
    invalid.name = QString(121, 'x');
    ASSERT_FALSE(library.SaveComparisonSet(invalid, &error));

    invalid = copy;
    invalid.confidencePercent = std::numeric_limits<double>::quiet_NaN();
    ASSERT_FALSE(library.SaveComparisonSet(invalid, &error));

    invalid = copy;
    invalid.members.push_back(copy.members[0]);
    ASSERT_FALSE(library.SaveComparisonSet(invalid, &error));
    ASSERT_EQ(library.ListComparisonSets(&error)[0].members, copy.members);
    library.Close();
}

TEST(ModelBenchmarkLibrary, ComparisonSetsMigrateVersionTwo)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const fs::path path(directory.filePath("legacy.sqlite").toStdString());
    ModelBenchmarkLibrary library;
    QString error;
    ASSERT_TRUE(library.Open(path, &error));

    ModelBenchmarkResult run;
    run.modelNames = { "A", "B" };
    run.games = { ScoredGame(0, 1) };
    ASSERT_TRUE(library.SaveRun("existing-run", run, { SavedModel("A", "a", "a"), SavedModel("B", "b", "b") }, {}, &error));
    const std::vector<ModelBenchmarkLibraryModel> models = library.ListModels(&error);
    library.Close();

    const QString connection = QUuid::createUuid().toString();
    {
        QSqlDatabase db = QSqlDatabase::addDatabase("QSQLITE", connection);
        db.setDatabaseName(QString::fromStdString(path.string()));
        ASSERT_TRUE(db.open());
        QSqlQuery query(db);
        ASSERT_TRUE(query.exec("DROP TABLE comparison_members"));
        ASSERT_TRUE(query.exec("DROP TABLE comparison_sets"));
        ASSERT_TRUE(query.exec("UPDATE library_meta SET schema_version=2"));
    }
    QSqlDatabase::removeDatabase(connection);

    const float confidence = 0.95f;
    ASSERT_TRUE(library.Open(path, &error));
    ASSERT_TRUE(library.ListComparisonSets(&error).empty());
    ASSERT_TRUE(error.isEmpty());
    ASSERT_EQ(library.ListModels(&error).size(), size_t(2));
    ASSERT_EQ(library.LoadAggregate({ { models[0].id, true }, { models[1].id, true } }, confidence, &error).result.scoredGames, 1);

    ModelBenchmarkComparisonSet comparison;
    comparison.name = "First saved comparison";
    comparison.members = { { models[0].id, true, true }, { models[1].id, false, false } };
    ASSERT_TRUE(library.SaveComparisonSet(comparison, &error));
    library.Close();
    ASSERT_TRUE(library.Open(path, &error));
    ASSERT_EQ(library.ListComparisonSets(&error)[0].members, comparison.members);
}

TEST(ModelBenchmarkLibrary, AggregatesRunsAndDetectsChangedCheckpoint)
{
    const fs::path path = RocketForgeTests::TestTempDir("library") / "library.sqlite";
    std::error_code cleanupError;
    fs::remove(path, cleanupError);

    ModelBenchmarkLibrary library;
    QString error;
    ASSERT_TRUE(library.Open(path, &error));
    ASSERT_TRUE(error.isEmpty());

    ModelBenchmarkResult first;
    first.stochasticKickoff = true;
    first.kickoffTemperature = 0.65f;
    first.modelNames = { "A", "B" };
    first.modelVersionIds = { "v1.10", "v1.20" };
    first.games = { ScoredGame(0, 1), ScoredGame(1, 2) };
    const std::vector<ModelBenchmarkLibraryModel> models = {
        SavedModel("A", "v1.10", "fingerprint-a"),
        SavedModel("B", "v1.20", "fingerprint-b")
    };
    ASSERT_TRUE(library.SaveRun("run-1", first, models, {}, &error));
    ASSERT_FALSE(library.SaveRun("run-1", first, models, {}, &error));

    std::vector<ModelBenchmarkLibraryModel> saved = library.ListModels(&error);
    ASSERT_EQ(saved.size(), size_t(2));
    ASSERT_TRUE(saved[0].branchName == "test-branch");
    ModelBenchmarkLibraryAggregate oneRun = library.LoadAggregate(
        { { saved[0].id, true }, { saved[1].id, true } },
        0.95f,
        &error);
    ASSERT_EQ(oneRun.contributingRuns, 1);
    ASSERT_EQ(oneRun.result.scoredGames, 2);
    ASSERT_EQ(oneRun.loadQueryCount, 4);
    ASSERT_TRUE(oneRun.result.stochasticKickoff);
    ASSERT_EQ(oneRun.result.kickoffTemperature, 0.65f);
    ASSERT_FALSE(oneRun.result.kickoffInferenceSettingsMixed);

    ModelBenchmarkResult second = first;
    second.stochasticKickoff = false;
    ASSERT_TRUE(library.SaveRun("run-2", second, models, {}, &error));
    ModelBenchmarkLibraryAggregate twoRuns = library.LoadAggregate(
        { { saved[0].id, true }, { saved[1].id, true } },
        0.95f,
        &error);
    ASSERT_EQ(twoRuns.contributingRuns, 2);
    ASSERT_EQ(twoRuns.result.scoredGames, 4);
    ASSERT_EQ(twoRuns.loadQueryCount, 4);
    ASSERT_TRUE(twoRuns.result.kickoffInferenceSettingsMixed);
    ASSERT_FALSE(twoRuns.firstRunAt.isEmpty());
    ASSERT_FALSE(twoRuns.latestRunAt.isEmpty());

    std::vector<ModelBenchmarkLibraryModel> changed = models;
    changed[0].fingerprint = "continued-training-fingerprint";
    const auto conflicts = library.FindConflicts(changed, &error);
    ASSERT_EQ(conflicts.size(), size_t(1));
    ASSERT_EQ(conflicts[0].incomingModelIndex, 0);
    ASSERT_FALSE(conflicts[0].existing.empty());

    ModelBenchmarkResult unrelatedRun = first;
    unrelatedRun.modelNames = { "B", "C" };
    const ModelBenchmarkLibraryModel modelC = SavedModel("C", "v1.30", "fingerprint-c");
    ASSERT_TRUE(library.SaveRun("unrelated-run", unrelatedRun, { models[1], modelC }, {}, &error));

    qint64 oldModelAId = -1;
    for (const ModelBenchmarkLibraryModel& model : saved) {
        if (model.versionId == "v1.10")
            oldModelAId = model.id;
    }
    ASSERT_TRUE(oldModelAId >= 0);
    ASSERT_TRUE(library.SaveRun("run-3", first, changed, { oldModelAId }, &error));
    saved = library.ListModels(&error);
    ASSERT_TRUE(error.isEmpty());
    ASSERT_EQ(saved.size(), size_t(3));
    qint64 changedModelAId = -1;
    qint64 modelBId = -1;
    qint64 modelCId = -1;
    for (const ModelBenchmarkLibraryModel& model : saved) {
        if (model.versionId == "v1.10") {
            changedModelAId = model.id;
            ASSERT_TRUE(model.fingerprint == "continued-training-fingerprint");
        } else if (model.versionId == "v1.20") {
            modelBId = model.id;
        } else if (model.versionId == "v1.30") {
            modelCId = model.id;
        }
    }
    ASSERT_TRUE(changedModelAId >= 0);
    ASSERT_TRUE(modelBId >= 0);
    ASSERT_TRUE(modelCId >= 0);
    const ModelBenchmarkLibraryAggregate replacement = library.LoadAggregate(
        { { changedModelAId, true }, { modelBId, true } },
        0.95f,
        &error);
    ASSERT_TRUE(error.isEmpty());
    ASSERT_EQ(replacement.contributingRuns, 1);
    ASSERT_EQ(replacement.result.scoredGames, 2);
    const ModelBenchmarkLibraryAggregate unrelated = library.LoadAggregate(
        { { modelBId, true }, { modelCId, true } },
        0.95f,
        &error);
    ASSERT_TRUE(error.isEmpty());
    ASSERT_EQ(unrelated.contributingRuns, 1);
    ASSERT_EQ(unrelated.result.scoredGames, 2);

    ModelBenchmarkLibrary readOnly;
    ASSERT_TRUE(readOnly.OpenReadOnly(path, &error));
    const ModelBenchmarkLibraryAggregate readOnlyAggregate = readOnly.LoadAggregate(
        { { changedModelAId, true }, { modelBId, true } },
        0.95f,
        &error);
    ASSERT_TRUE(error.isEmpty());
    ASSERT_EQ(readOnlyAggregate.contributingRuns, 1);
    ASSERT_EQ(readOnlyAggregate.result.scoredGames, 2);
    readOnly.Close();

    library.Close();
    fs::remove(path, cleanupError);
}

TEST(ModelBenchmarkLibrary, MigratesBranchMetadataFromVersionOne)
{
    const fs::path path = RocketForgeTests::TestTempDir("v1_migration") / "library.sqlite";
    std::error_code cleanupError;
    fs::remove(path, cleanupError);

    const QString connectionName = "ModelBenchmarkMigrationTest-"
        + QUuid::createUuid().toString(QUuid::WithoutBraces);
    {
        QSqlDatabase database = QSqlDatabase::addDatabase("QSQLITE", connectionName);
        database.setDatabaseName(QString::fromStdString(path.string()));
        ASSERT_TRUE(database.open());
        QSqlQuery query(database);
        ASSERT_TRUE(query.exec("CREATE TABLE library_meta (schema_version INTEGER NOT NULL)"));
        ASSERT_TRUE(query.exec("INSERT INTO library_meta(schema_version) VALUES(1)"));
        ASSERT_TRUE(query.exec(
            "CREATE TABLE models ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, logical_key TEXT NOT NULL, display_name TEXT NOT NULL, "
            "agent_name TEXT, build_name TEXT, stage_name TEXT, version_id TEXT, source_label TEXT, "
            "checkpoint_path TEXT, fingerprint TEXT NOT NULL, latest_timesteps TEXT, "
            "fingerprint_verified INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, "
            "UNIQUE(logical_key, fingerprint))"));
        database.close();
    }
    QSqlDatabase::removeDatabase(connectionName);

    ModelBenchmarkLibrary library;
    QString error;
    ASSERT_TRUE(library.Open(path, &error));
    ModelBenchmarkResult result;
    result.modelNames = { "A", "B" };
    result.games = { ScoredGame(0, 1), ScoredGame(1, 2) };
    ASSERT_TRUE(library.SaveRun(
        "migrated-run",
        result,
        { SavedModel("A", "a", "fingerprint-a"), SavedModel("B", "b", "fingerprint-b") },
        {},
        &error));
    const std::vector<ModelBenchmarkLibraryModel> saved = library.ListModels(&error);
    ASSERT_TRUE(error.isEmpty());
    ASSERT_EQ(saved.size(), size_t(2));
    ASSERT_TRUE(saved[0].branchName == "test-branch");
    ASSERT_TRUE(saved[1].branchName == "test-branch");

    library.Close();
    fs::remove(path, cleanupError);
}

TEST(ModelBenchmarkLibrary, FingerprintTracksWeightsAndTrainingSetup)
{
    const fs::path root = RocketForgeTests::TestTempDir("fingerprint");
    std::error_code cleanupError;
    const fs::path policy = root / "POLICY.lt";
    const fs::path setup = root / "training-setup.json";
    {
        std::ofstream out(policy, std::ios::binary);
        out << "first weights";
    }
    const QString first = ModelBenchmarkLibrary::FingerprintCheckpoint(root);
    ASSERT_FALSE(first.isEmpty());
    {
        std::ofstream out(policy, std::ios::binary | std::ios::trunc);
        out << "continued weights";
    }
    const QString continued = ModelBenchmarkLibrary::FingerprintCheckpoint(root);
    ASSERT_FALSE(continued.isEmpty());
    ASSERT_FALSE(first == continued);
    {
        std::ofstream out(setup, std::ios::binary);
        out << "different agent setup";
    }
    const QString withSetup = ModelBenchmarkLibrary::FingerprintCheckpoint(root, setup);
    ASSERT_FALSE(withSetup.isEmpty());
    ASSERT_FALSE(continued == withSetup);

    const fs::file_time_type preservedWriteTime = fs::last_write_time(policy);
    {
        std::ofstream out(policy, std::ios::binary | std::ios::trunc);
        out << "same-size-A";
    }
    fs::last_write_time(policy, preservedWriteTime);
    const QString sameMetadataA = ModelBenchmarkLibrary::FingerprintCheckpoint(root, setup);
    {
        std::ofstream out(policy, std::ios::binary | std::ios::trunc);
        out << "same-size-B";
    }
    fs::last_write_time(policy, preservedWriteTime);
    const QString sameMetadataB = ModelBenchmarkLibrary::FingerprintCheckpoint(root, setup);
    ASSERT_FALSE(sameMetadataA == sameMetadataB);

    fs::remove_all(root, cleanupError);
}

TEST(ModelBenchmarkLibrary, FingerprintPreservesRawSetupIdentity)
{
    const fs::path root = RocketForgeTests::TestTempDir("raw_setup_identity");
    const fs::path setup = root / "training_setup.json";
    {
        std::ofstream out(root / "POLICY.lt", std::ios::binary);
        out << "unchanged weights";
    }

    auto fingerprint = [&](const std::string& bytes) {
        {
            std::ofstream out(setup, std::ios::binary | std::ios::trunc);
            out << bytes;
        }
        return ModelBenchmarkLibrary::FingerprintCheckpoint(root, setup);
    };

    for (const std::string newline : { "\n", "\r\n" }) {
        const std::string fields = newline + "    \"agent\": \"FixtureA\","
            + newline + "    \"nested\": {\"simulator_version\": 2, \"text\": \"comma, brace}\\\"\"},"
            + newline + "    \"tick_skip\": 8";
        const std::string legacy = "{" + fields + newline + "}";
        const QString original = fingerprint(legacy);
        ASSERT_FALSE(original.isEmpty());

        const std::string history = newline + "    \"simulator_version\": 2," + newline + "    \"simulator_provenance\": \"confirmed\"";
        ASSERT_FALSE(original == fingerprint("{" + fields + "," + history + newline + "}"));
        ASSERT_FALSE(original == fingerprint("{" + history + "," + fields + newline + "}"));
        ASSERT_FALSE(original == fingerprint("{" + fields + "," + newline + "    \"simulator_version\": 4" + newline + "}"));

        std::string changed = legacy;
        changed.replace(changed.find("tick_skip\": 8"), 13, "tick_skip\": 4");
        ASSERT_FALSE(original == fingerprint(changed));

        changed = legacy;
        changed.replace(changed.find("simulator_version\": 2"), 21, "simulator_version\": 3");
        ASSERT_FALSE(original == fingerprint(changed));
        ASSERT_FALSE(original == fingerprint(legacy + "\n"));
    }
    std::error_code cleanupError;
    fs::remove_all(root, cleanupError);
}

TEST(ModelBenchmarkLibrary, ImportsExistingExcelXmlShape)
{
    const fs::path path = RocketForgeTests::TestTempDir("import") / "import.xml";
    {
        std::ofstream out(path);
        out << R"(<?xml version="1.0"?>
<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet" xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">
<Worksheet ss:Name="Summary"><Table>
<Row><Cell><Data>Field</Data></Cell><Cell><Data>Value</Data></Cell></Row>
<Row><Cell><Data>Run Seed</Data></Cell><Cell><Data>42</Data></Cell></Row>
<Row><Cell><Data>Elo Range Confidence</Data></Cell><Cell><Data>95.00%</Data></Cell></Row>
</Table></Worksheet>
<Worksheet ss:Name="Models"><Table>
<Row><Cell><Data>Model Index</Data></Cell><Cell><Data>Display Name</Data></Cell><Cell><Data>Version ID</Data></Cell><Cell><Data>Checkpoint Path</Data></Cell></Row>
<Row><Cell><Data>0</Data></Cell><Cell><Data>A</Data></Cell><Cell><Data>v1.10</Data></Cell><Cell><Data>Z:/missing/a</Data></Cell></Row>
<Row><Cell><Data>1</Data></Cell><Cell><Data>B</Data></Cell><Cell><Data>v1.20</Data></Cell><Cell><Data>Z:/missing/b</Data></Cell></Row>
</Table></Worksheet>
<Worksheet ss:Name="Game Results"><Table>
<Row><Cell><Data>Game ID</Data></Cell><Cell><Data>Snapshot ID</Data></Cell><Cell><Data>Model A Index</Data></Cell><Cell><Data>Model B Index</Data></Cell><Cell><Data>Blue Index</Data></Cell><Cell><Data>Orange Index</Data></Cell><Cell><Data>Mode</Data></Cell><Cell><Data>Goal Time Seconds</Data></Cell><Cell><Data>Goal Timing</Data></Cell><Cell><Data>Goal Scored</Data></Cell><Cell><Data>Winner Index</Data></Cell><Cell><Data>Included In Rating</Data></Cell><Cell><Data>Mirror Completed</Data></Cell></Row>
<Row><Cell><Data>1</Data></Cell><Cell><Data>1</Data></Cell><Cell><Data>0</Data></Cell><Cell><Data>1</Data></Cell><Cell><Data>0</Data></Cell><Cell><Data>1</Data></Cell><Cell><Data>1v1</Data></Cell><Cell><Data>12.5</Data></Cell><Cell><Data>Post-Kickoff</Data></Cell><Cell><Data>Yes</Data></Cell><Cell><Data>0</Data></Cell><Cell><Data>Yes</Data></Cell><Cell><Data>Yes</Data></Cell></Row>
</Table></Worksheet>
</Workbook>)";
    }

    QString error;
    const ModelBenchmarkLibraryImport imported = ModelBenchmarkLibrary::ReadExcelExport(path, &error);
    ASSERT_TRUE(error.isEmpty());
    ASSERT_EQ(imported.models.size(), size_t(2));
    ASSERT_EQ(imported.result.games.size(), size_t(1));
    ASSERT_EQ(imported.result.games[0].mode, 1);
    ASSERT_EQ(imported.result.games[0].winner, 0);
    ASSERT_EQ(imported.result.seed, uint64_t(42));
    ASSERT_EQ(imported.result.rocketSimVersions, std::set<int>({ 2 }));
    ASSERT_FALSE(imported.result.statTotalsComplete);
    ASSERT_FALSE(imported.warnings.isEmpty());

    std::error_code cleanupError;
    fs::remove(path, cleanupError);
}

TEST(ModelBenchmarkLibrary, PrimarySelectionExcludesSecondaryOnlyRunsAndStats)
{
    const fs::path path = RocketForgeTests::TestTempDir("primary") / "library.sqlite";
    std::error_code cleanupError;
    fs::remove(path, cleanupError);

    ModelBenchmarkLibrary library;
    QString error;
    ASSERT_TRUE(library.Open(path, &error));
    const ModelBenchmarkLibraryModel modelA = SavedModel("A", "a", "fingerprint-a");
    const ModelBenchmarkLibraryModel modelB = SavedModel("B", "b", "fingerprint-b");
    const ModelBenchmarkLibraryModel modelC = SavedModel("C", "c", "fingerprint-c");

    ModelBenchmarkResult primaryRun;
    primaryRun.modelNames = { "A", "B" };
    primaryRun.games = { ScoredGame(0, 1), ScoredGame(1, 2) };
    ModelBenchmarkStatTotals totals;
    totals.gameTimeSum = 300.0;
    totals.goals = 1.0;
    totals.goalsConceded = 2.0;
    totals.passes = 2.0;
    totals.passGameTimeSum = 300.0;
    totals.bumps = 3.0;
    totals.boostSteals = 1.0;
    totals.opponentBallTouches = 5.0;
    totals.boostConsumed = 15.0;
    totals.lowAerialTimeSum = 40.0;
    totals.highAerialTimeSum = 20.0;
    totals.offensiveThirdBallSteps = 50.0;
    primaryRun.statTotals = {
        { 0, 1, 1, totals },
        { 1, 0, 1, totals }
    };
    ASSERT_TRUE(library.SaveRun("primary-run", primaryRun, { modelA, modelB }, {}, &error));

    ModelBenchmarkResult secondaryOnlyRun;
    secondaryOnlyRun.modelNames = { "B", "C" };
    secondaryOnlyRun.games = { ScoredGame(0, 3), ScoredGame(1, 4) };
    secondaryOnlyRun.statTotalsComplete = false;
    ASSERT_TRUE(library.SaveRun("secondary-only-run", secondaryOnlyRun, { modelB, modelC }, {}, &error));

    qint64 modelAId = -1;
    qint64 modelBId = -1;
    qint64 modelCId = -1;
    for (const ModelBenchmarkLibraryModel& model : library.ListModels(&error)) {
        if (model.versionId == "a")
            modelAId = model.id;
        else if (model.versionId == "b")
            modelBId = model.id;
        else if (model.versionId == "c")
            modelCId = model.id;
    }
    ASSERT_TRUE(modelAId >= 0 && modelBId >= 0 && modelCId >= 0);
    const ModelBenchmarkLibraryAggregate aggregate = library.LoadAggregate(
        { { modelAId, true }, { modelBId, false }, { modelCId, false } },
        0.95f,
        &error);
    ASSERT_TRUE(error.isEmpty());
    ASSERT_EQ(aggregate.contributingRuns, 1);
    ASSERT_EQ(aggregate.result.scoredGames, 2);
    ASSERT_EQ(aggregate.coverageGames.size(), size_t(4));
    ASSERT_TRUE(aggregate.result.statTotalsComplete);
    ASSERT_EQ(aggregate.result.statTotals.size(), size_t(2));
    ASSERT_EQ(
        aggregate.result.statTotals[0].totals.definitionVersion,
        MODEL_BENCHMARK_STAT_DEFINITION_VERSION);
    ASSERT_TRUE(aggregate.result.statTotals[0].totals.passes == 2.0);
    ASSERT_TRUE(aggregate.result.statTotals[0].totals.goalsConceded == 2.0);
    ASSERT_TRUE(aggregate.result.statTotals[0].totals.opponentBallTouches == 5.0);
    ASSERT_TRUE(aggregate.result.statTotals[0].totals.boostConsumed == 15.0);
    ASSERT_TRUE(aggregate.result.statTotals[0].totals.bumps == 3.0);
    ASSERT_TRUE(aggregate.result.statTotals[0].totals.boostSteals == 1.0);
    ASSERT_TRUE(aggregate.result.statTotals[0].totals.lowAerialTimeSum == 40.0);
    ASSERT_TRUE(aggregate.result.statTotals[0].totals.highAerialTimeSum == 20.0);
    ASSERT_TRUE(aggregate.result.statTotals[0].totals.offensiveThirdBallSteps == 50.0);
    ASSERT_TRUE(aggregate.warnings.join(' ').contains("C"));

    library.Close();
    fs::remove(path, cleanupError);
}

TEST(ModelBenchmarkLibrary, ImportsLosslessStatisticsAndEmbeddedRunIdentity)
{
    const fs::path path = RocketForgeTests::TestTempDir("lossless_import") / "import.xml";
    {
        std::ofstream out(path);
        out << R"(<?xml version="1.0"?>
<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet" xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">
<Worksheet ss:Name="Summary"><Table>
<Row><Cell><Data>Field</Data></Cell><Cell><Data>Value</Data></Cell></Row>
<Row><Cell><Data>Run ID</Data></Cell><Cell><Data>preserved-run-id</Data></Cell></Row>
<Row><Cell><Data>Raw Statistics Complete</Data></Cell><Cell><Data>Yes</Data></Cell></Row>
</Table></Worksheet>
<Worksheet ss:Name="Models"><Table>
<Row><Cell><Data>Model Index</Data></Cell><Cell><Data>Display Name</Data></Cell><Cell><Data>Agent</Data></Cell><Cell><Data>Build</Data></Cell><Cell><Data>Stage</Data></Cell><Cell><Data>Source</Data></Cell><Cell><Data>Version ID</Data></Cell><Cell><Data>Checkpoint Fingerprint</Data></Cell></Row>
<Row><Cell><Data>0</Data></Cell><Cell><Data>A</Data></Cell><Cell><Data>Agent</Data></Cell><Cell><Data>Build</Data></Cell><Cell><Data>Stage</Data></Cell><Cell><Data>RocketForge</Data></Cell><Cell><Data>v1</Data></Cell><Cell><Data>fingerprint-a</Data></Cell></Row>
<Row><Cell><Data>1</Data></Cell><Cell><Data>B</Data></Cell><Cell><Data>Agent</Data></Cell><Cell><Data>Build</Data></Cell><Cell><Data>Stage</Data></Cell><Cell><Data>RocketForge</Data></Cell><Cell><Data>v2</Data></Cell><Cell><Data>fingerprint-b</Data></Cell></Row>
</Table></Worksheet>
<Worksheet ss:Name="Raw Statistics"><Table>
<Row><Cell><Data>Model Index</Data></Cell><Cell><Data>Opponent Index</Data></Cell><Cell><Data>Mode</Data></Cell><Cell><Data>Goals</Data></Cell><Cell><Data>Game Time Sum</Data></Cell><Cell><Data>Step Samples</Data></Cell><Cell><Data>Player Samples</Data></Cell></Row>
<Row><Cell><Data>0</Data></Cell><Cell><Data>1</Data></Cell><Cell><Data>1</Data></Cell><Cell><Data>2.5</Data></Cell><Cell><Data>300</Data></Cell><Cell><Data>120</Data></Cell><Cell><Data>240</Data></Cell></Row>
</Table></Worksheet>
</Workbook>)";
    }

    QString error;
    const ModelBenchmarkLibraryImport imported = ModelBenchmarkLibrary::ReadExcelExport(path, &error);
    ASSERT_TRUE(error.isEmpty());
    ASSERT_TRUE(imported.runId == "preserved-run-id");
    ASSERT_TRUE(imported.result.statTotalsComplete);
    ASSERT_EQ(imported.result.statTotals.size(), size_t(1));
    ASSERT_EQ(imported.result.statTotals[0].mode, 1);
    ASSERT_EQ(imported.result.statTotals[0].totals.definitionVersion, 1);
    ASSERT_EQ(imported.result.statTotals[0].totals.stepSamples, int64_t(120));
    ASSERT_TRUE(imported.result.statTotals[0].totals.goals == 2.5);
    ASSERT_TRUE(imported.warnings.isEmpty());

    std::error_code cleanupError;
    fs::remove(path, cleanupError);
}

TEST(ModelBenchmarkLibrary, TrainingDatesSurviveReopenAndSnapshotConflicts)
{
    QTemporaryDir directory;
    const fs::path path(directory.filePath("training-metadata.sqlite").toStdString());
    QString error;
    ModelBenchmarkLibrary library;
    ASSERT_TRUE(library.Open(path, &error));

    ModelBenchmarkLibraryModel a = SavedModel("A", "a", "a");
    ModelBenchmarkLibraryModel b = SavedModel("B", "b", "b");
    a.latestTimesteps = 123456789012ULL;
    a.lastTrainedAt = "20260927-112233";
    ModelBenchmarkResult run;
    run.modelNames = { "A", "B" };
    run.games = { ScoredGame(0, 1) };
    ASSERT_TRUE(library.SaveRun("first", run, { a, b }, {}, &error));
    library.Close();

    ASSERT_TRUE(library.OpenReadOnly(path, &error));
    std::vector<ModelBenchmarkLibraryModel> models = library.ListModels(&error);
    ASSERT_TRUE(error.isEmpty());
    ASSERT_EQ(models.at(0).latestTimesteps, a.latestTimesteps);
    ASSERT_EQ(models.at(0).lastTrainedAt, a.lastTrainedAt);
    ASSERT_TRUE(models.at(1).lastTrainedAt.isEmpty());
    library.Close();

    ASSERT_TRUE(library.Open(path, &error));
    ModelBenchmarkLibraryModel resumed = a;
    resumed.fingerprint = "new-checkpoint";
    resumed.lastTrainedAt = "20260928-112233";
    const std::vector<ModelBenchmarkLibraryConflict> conflicts = library.FindConflicts({ resumed }, &error);
    ASSERT_EQ(conflicts.at(0).existing.at(0).lastTrainedAt, a.lastTrainedAt);

    b.lastTrainedAt = "20260926-112233";
    const QString recordedDate = a.lastTrainedAt;
    a.lastTrainedAt.clear();
    ASSERT_TRUE(library.SaveRun("second", run, { a, b }, {}, &error));
    models = library.ListModels(&error);
    ASSERT_EQ(models.at(0).lastTrainedAt, recordedDate);
    ASSERT_EQ(models.at(1).lastTrainedAt, b.lastTrainedAt);
}

TEST(ModelBenchmarkLibrary, VersionFourTrainingDateMigrationPreservesResults)
{
    QTemporaryDir directory;
    const fs::path path(directory.filePath("legacy-training.sqlite").toStdString());
    QString error;
    ModelBenchmarkLibrary library;
    ASSERT_TRUE(library.Open(path, &error));

    ModelBenchmarkResult run;
    run.modelNames = { "A", "B" };
    run.games = { ScoredGame(0, 1) };
    ASSERT_TRUE(library.SaveRun("legacy", run, { SavedModel("A", "a", "a"), SavedModel("B", "b", "b") }, {}, &error));
    library.Close();

    const QString connection = QUuid::createUuid().toString();
    {
        QSqlDatabase database = QSqlDatabase::addDatabase("QSQLITE", connection);
        database.setDatabaseName(QString::fromStdString(path.string()));
        ASSERT_TRUE(database.open());
        QSqlQuery query(database);
        ASSERT_TRUE(query.exec("ALTER TABLE models DROP COLUMN last_trained_at"));
        ASSERT_TRUE(query.exec("UPDATE library_meta SET schema_version=4"));
    }
    QSqlDatabase::removeDatabase(connection);

    ASSERT_TRUE(library.OpenReadOnly(path, &error));
    const std::vector<ModelBenchmarkLibraryModel> models = library.ListModels(&error);
    ASSERT_TRUE(error.isEmpty());
    ASSERT_EQ(models.size(), size_t(2));
    ASSERT_TRUE(models[0].lastTrainedAt.isEmpty());
    library.Close();

    ASSERT_TRUE(library.Open(path, &error));
    const std::vector<ModelBenchmarkLibrarySelection> selection = { { models[0].id, true }, { models[1].id, true } };
    const ModelBenchmarkLibraryAggregate aggregate = library.LoadAggregate(selection, 95.0, &error);
    ASSERT_TRUE(error.isEmpty());
    ASSERT_EQ(aggregate.result.scoredGames, 1);
    ASSERT_EQ(library.ListModels(&error).size(), size_t(2));
}

TEST(ModelBenchmarkLibrary, CombinesLiveWithSavedWithoutDoubleCounting)
{
    QTemporaryDir directory;
    QString error;
    ModelBenchmarkLibrary library;
    ASSERT_TRUE(library.Open(fs::path(directory.filePath("live.sqlite").toStdString()), &error));

    ModelBenchmarkResult saved;
    saved.modelNames = { "A", "B" };
    saved.rocketSimVersions = { 3 };
    saved.games = { ScoredGame(0, 1), ScoredGame(1, 2) };
    saved.statTotalsComplete = true;
    ModelBenchmarkStatTotals totals;
    totals.goals = 2;
    totals.gameTimeSum = 40;
    saved.statTotals = { {0, 1, 1, totals} };
    ASSERT_TRUE(library.SaveRun("baseline", saved, { SavedModel("A", "a", "a"), SavedModel("B", "b", "b") }, {}, &error));

    const std::vector<ModelBenchmarkLibraryModel> models = library.ListModels(&error);
    const qint64 a = std::find_if(models.begin(), models.end(), [](const auto& m) { return m.displayName == "A"; })->id;
    const qint64 b = std::find_if(models.begin(), models.end(), [](const auto& m) { return m.displayName == "B"; })->id;
    ModelBenchmarkLibraryModel c = SavedModel("C", "c", "c");
    c.id = -2;
    std::vector<ModelBenchmarkLibrarySelection> selection = { {c.id, true}, {b, true}, {a, true} };
    const float confidence = 0.95f;
    const ModelBenchmarkLibraryAggregate baseline = library.LoadAggregate(selection, confidence, &error, {}, {}, {c});
    ASSERT_TRUE(error.isEmpty());
    ASSERT_EQ(baseline.result.games.size(), size_t(2));

    ModelBenchmarkResult live;
    live.modelNames = { "A", "C" };
    live.rocketSimVersions = { 3 };
    live.games = { ScoredGame(1, 1) };
    live.statTotalsComplete = true;
    totals.goals = 1;
    totals.gameTimeSum = 20;
    live.statTotals = { {0, 1, 1, totals} };
    const std::vector<qint64> ids = {a, c.id};
    ModelBenchmarkLibraryAggregate combined = CombineModelBenchmarkLiveResults(baseline, selection, live, ids, {});
    ASSERT_EQ(combined.result.scoredGames, 3);
    ASSERT_EQ(combined.liveGames, 1);
    ASSERT_EQ(combined.result.games.back().modelA, 2);
    ASSERT_EQ(combined.result.games.back().modelB, 0);
    ASSERT_EQ(combined.result.games.back().winner, 0);
    ASSERT_EQ(combined.result.statTotals.size(), size_t(4));
    ASSERT_TRUE(combined.result.statTotalsComplete);

    const auto liveTotals = std::find_if(combined.result.statTotals.begin(), combined.result.statTotals.end(),
        [](const auto& row) { return row.modelIndex == 2 && row.opponentIndex == 0; });
    ASSERT_TRUE(liveTotals != combined.result.statTotals.end());
    ASSERT_EQ(liveTotals->totals.goals, 1.0);
    ASSERT_TRUE(combined.result.games.back().snapshotId > baseline.result.games.back().snapshotId);

    // Every update combines with the original saved baseline, never the last display.
    live.games.push_back(ScoredGame(0, 2));
    combined = CombineModelBenchmarkLiveResults(baseline, selection, live, ids, {});
    ASSERT_EQ(combined.result.scoredGames, 4);
    ASSERT_EQ(baseline.result.scoredGames, 2);
    ASSERT_EQ(CombineModelBenchmarkLiveResults(baseline, selection, live, ids, {}).result.scoredGames, 4);

    live.statTotalsComplete = false;
    ASSERT_FALSE(CombineModelBenchmarkLiveResults(baseline, selection, live, ids, {}).result.statTotalsComplete);
    ASSERT_TRUE(CombineModelBenchmarkLiveResults(baseline, selection, live, ids, {}).result.optionalStats.empty());

    ModelBenchmarkLibraryFilter filter;
    filter.modeMask = 2;
    ASSERT_EQ(CombineModelBenchmarkLiveResults(baseline, selection, live, ids, filter).liveGames, 0);
    filter.modeMask = 7;
    filter.engineVersion = 2;
    ASSERT_EQ(CombineModelBenchmarkLiveResults(baseline, selection, live, ids, filter).liveGames, 0);

    // Primary filtering excludes new games between two non-primary models.
    selection[0].primary = selection[2].primary = false;
    const ModelBenchmarkLibraryAggregate primaryBaseline = library.LoadAggregate(selection, confidence, &error, {}, {}, {c});
    ASSERT_EQ(CombineModelBenchmarkLiveResults(primaryBaseline, selection, live, ids, {}).liveGames, 0);
    ASSERT_EQ(CombineModelBenchmarkLiveResults(primaryBaseline, selection, live, ids, {}).coverageGames.size(), size_t(4));
}

TEST(ModelBenchmarkLibrary, LiveOnlyIncludesDrawsAndRespectsExcludedModels)
{
    QTemporaryDir directory;
    ModelBenchmarkLibrary library;
    QString error;
    ASSERT_TRUE(library.Open(fs::path(directory.filePath("new-live.sqlite").toStdString()), &error));

    ModelBenchmarkLibraryModel a = SavedModel("A", "a", "a");
    a.id = -2;
    ModelBenchmarkLibraryModel b = SavedModel("B", "b", "b");
    b.id = -3;
    const std::vector<ModelBenchmarkLibrarySelection> selection = {{a.id, true}, {b.id, true}};
    const float confidence = 0.95f;
    const ModelBenchmarkLibraryAggregate baseline = library.LoadAggregate(selection, confidence, &error, {}, {}, {a, b});
    ASSERT_TRUE(error.isEmpty());

    ModelBenchmarkResult live;
    live.modelNames = {"A", "B", "Excluded"};
    live.statTotalsComplete = true;
    live.games = {ScoredGame(0, 1), ScoredGame(1, 2), ScoredGame(0, 3)};
    live.games[1].goalScored = false;
    live.games[1].winner = -1;
    live.games[2].modelB = live.games[2].orangeModel = 2;
    const ModelBenchmarkLibraryAggregate combined = CombineModelBenchmarkLiveResults(baseline, selection, live, {a.id, b.id, -4}, {});
    ASSERT_EQ(combined.contributingRuns, 0);
    ASSERT_EQ(combined.liveGames, 2);
    ASSERT_EQ(combined.result.scoredGames, 1);
    ASSERT_EQ(combined.result.noResultGames, 1);
    ASSERT_EQ(combined.result.games.size(), size_t(2));
    ASSERT_TRUE(combined.result.statTotalsComplete);
    ASSERT_EQ(combined.result.games[1].winner, -1);
}

TEST(ModelBenchmarkLibrary, MixedHistoryPreservesExactEnvironmentCoverage)
{
    QTemporaryDir directory;
    QString error;
    ModelBenchmarkLibrary library;
    const fs::path path(directory.filePath("environments.sqlite").toStdString());
    ASSERT_TRUE(library.Open(path, &error));

    const ModelBenchmarkLibraryModel a = SavedModel("A", "a", "a");
    const ModelBenchmarkLibraryModel b = SavedModel("B", "b", "b");
    ModelBenchmarkResult run;
    run.modelNames = {"A", "B"};
    run.rocketSimVersions = {3};
    run.games = {ScoredGame(0, 1), ScoredGame(1, 2)};
    run.engineFamily = "RocketSim-v3";
    run.engineRevision = "pinned-revision";
    run.engineCarOrder = "insertion-order";
    run.integrationRevision = "integration-a";
    run.predictionConfigurationJson = "{}";
    run.evaluationIdentity = "v3-native-profile";
    ASSERT_TRUE(library.SaveRun("native", run, {a, b}, {}, &error));

    run.predictionConfigurationJson = "{\"A\":2}";
    run.evaluationIdentity = "v3-v2-profile";
    ASSERT_TRUE(library.SaveRun("hybrid", run, {a, b}, {}, &error));

    // An old result stays unidentified, even though its major version is known.
    run.engineFamily.clear();
    run.engineRevision.clear();
    run.integrationRevision.clear();
    run.evaluationIdentity.clear();
    ASSERT_TRUE(library.SaveRun("legacy", run, {a, b}, {}, &error));

    const std::vector<ModelBenchmarkLibraryModel> models = library.ListModels(&error);
    ASSERT_EQ(models.size(), size_t(2));
    const std::vector<ModelBenchmarkLibrarySelection> selection = {{models[0].id, true}, {models[1].id, true}};
    library.Close();

    const float confidence = 0.95f;
    ASSERT_TRUE(library.OpenReadOnly(path, &error));
    ModelBenchmarkLibraryAggregate aggregate = library.LoadAggregate(selection, confidence, &error);
    ASSERT_TRUE(error.isEmpty());
    ASSERT_EQ(aggregate.contributingRuns, 3);
    ASSERT_EQ(aggregate.result.scoredGames, 6);
    ASSERT_EQ(aggregate.environments.size(), size_t(3));
    ASSERT_TRUE(ModelBenchmarkHasMixedEnvironments(aggregate));
    ASSERT_TRUE(aggregate.result.evaluationIdentity.empty());
    ASSERT_TRUE(aggregate.result.engineRevision.empty());
    ASSERT_TRUE(aggregate.result.predictionConfigurationJson.empty());
    for (const auto& [key, environment] : aggregate.environments) {
        ASSERT_EQ(environment.games, qint64(2));
        ASSERT_EQ(environment.modelGames.at(0), qint64(2));
        ASSERT_EQ(environment.modelGames.at(1), qint64(2));
    }

    std::ostringstream stream;
    SpreadsheetXmlWriter writer(stream);
    WriteModelBenchmarkEnvironmentWorksheet(writer, aggregate);
    writer.Finish();
    const QString exported = QString::fromStdString(stream.str());
    QXmlStreamReader xml(exported);
    while (!xml.atEnd())
        xml.readNext();

    ASSERT_FALSE(xml.hasError());
    ASSERT_TRUE(exported.contains("Environment Coverage"));
    ASSERT_TRUE(exported.contains("v3-native-profile"));
    ASSERT_TRUE(exported.contains("v3-v2-profile"));
    ASSERT_TRUE(exported.contains("unrecorded build and settings"));

    ModelBenchmarkLibraryFilter filter;
    filter.engineVersion = 3;
    const std::vector<ModelBenchmarkLibraryCoverage> allCoverage = library.LoadCoverage(filter, &error);
    ASSERT_EQ(allCoverage.size(), size_t(3));
    ASSERT_EQ(std::count_if(allCoverage.begin(), allCoverage.end(), [](const auto& row) {
        return row.evaluationIdentity == "v3-native-profile" && row.games == 2;
    }), 1);

    filter.evaluationIdentity = "v3-native-profile";
    aggregate = library.LoadAggregate(selection, confidence, &error, {}, filter);
    ASSERT_TRUE(error.isEmpty());
    ASSERT_EQ(aggregate.contributingRuns, 1);
    ASSERT_EQ(aggregate.result.scoredGames, 2);
    ASSERT_EQ(aggregate.result.engineFamily, "RocketSim-v3");
    ASSERT_EQ(aggregate.result.engineRevision, "pinned-revision");
    ASSERT_EQ(aggregate.result.integrationRevision, "integration-a");
    ASSERT_EQ(aggregate.result.predictionConfigurationJson, "{}");
    const std::vector<ModelBenchmarkLibraryCoverage> coverage = library.LoadCoverage(filter, &error);
    ASSERT_EQ(coverage.size(), size_t(1));
    ASSERT_EQ(coverage[0].games, qint64(2));

    ModelBenchmarkResult live = aggregate.result;
    live.evaluationIdentity = "v3-v2-profile";
    const std::vector<qint64> liveModelIds = {models[0].id, models[1].id};
    ModelBenchmarkLibraryAggregate combined = CombineModelBenchmarkLiveResults(aggregate, selection, live, liveModelIds, {});
    ASSERT_EQ(combined.liveGames, 2);
    ASSERT_EQ(combined.result.scoredGames, 4);
    ASSERT_EQ(combined.environments.size(), size_t(2));
    ASSERT_TRUE(ModelBenchmarkHasMixedEnvironments(combined));
    ASSERT_TRUE(combined.result.evaluationIdentity.empty());
    ASSERT_FALSE(combined.warnings.empty());

    // A display filter excludes live games without hiding their selectable environment.
    combined = CombineModelBenchmarkLiveResults(aggregate, selection, live, liveModelIds, filter);
    ASSERT_EQ(combined.liveGames, 0);
    ASSERT_EQ(combined.availableEnvironments.size(), size_t(3));

    filter.evaluationIdentity.clear();
    filter.environmentKey = "legacy-major:3,";
    const ModelBenchmarkLibraryAggregate legacy = library.LoadAggregate(selection, confidence, &error, {}, filter);
    ASSERT_TRUE(error.isEmpty());
    ASSERT_EQ(legacy.contributingRuns, 1);
    ASSERT_TRUE(legacy.result.evaluationIdentity.empty());

    filter.environmentKey.clear();
    filter.evaluationIdentity.clear();
    filter.runId = "hybrid";
    aggregate = library.LoadAggregate(selection, confidence, &error, {}, filter);
    ASSERT_TRUE(error.isEmpty());
    ASSERT_EQ(aggregate.contributingRuns, 1);
    ASSERT_EQ(aggregate.result.evaluationIdentity, "v3-v2-profile");
    ASSERT_EQ(ModelBenchmarkRocketSimVersionsText({3}), "v3");
}

TEST(ModelBenchmarkLibrary, PairEnvironmentIgnoresOtherModels)
{
    QJsonObject environment{
        {"physicsFamily", "RocketSim-v3"}, {"physicsRevision", "physics-a"},
        {"integrationSourceRevision", "build-a"}, {"tickRate", 120},
        {"savedPredictionProfiles", QJsonObject{{"C:/models/a", "match"}, {"C:/models/b", "rocketsim-v2"}, {"C:/models/old", "rocketsim-v3"}}},
        {"externalPrediction", QJsonObject{{"C:/models/b", QJsonObject{{"engine", 2}, {"revision", "v2-a"}}},
            {"C:/models/old", QJsonObject{{"engine", 3}, {"revision", "v3-a"}}}}}
    };
    auto encode = [](const QJsonObject& value) { return QString::fromUtf8(QJsonDocument(value).toJson(QJsonDocument::Compact)); };
    const QString baseline = ModelBenchmarkPairEnvironmentKey(encode(environment), "C:/models/a", "C:/models/b");
    ASSERT_FALSE(baseline.isEmpty());

    QJsonObject current = environment;
    QJsonObject profiles = current["savedPredictionProfiles"].toObject();
    profiles.remove("C:/models/old");
    profiles["C:/models/new"] = "rocketsim-v3";
    current["savedPredictionProfiles"] = profiles;
    QJsonObject predictors = current["externalPrediction"].toObject();
    predictors.remove("C:/models/old");
    current["externalPrediction"] = predictors;
    ASSERT_EQ(baseline, ModelBenchmarkPairEnvironmentKey(encode(current), "C:\\models\\a", "C:/models/b"));
    ASSERT_TRUE(baseline != ModelBenchmarkPairEnvironmentKey(encode(current), "C:/models/b", "C:/models/a"));

    for (const char* field : { "physicsRevision", "integrationSourceRevision", "startingProtocol", "futureSharedSetting" }) {
        QJsonObject changed = current;
        changed[field] = "changed";
        ASSERT_TRUE(baseline != ModelBenchmarkPairEnvironmentKey(encode(changed), "C:/models/a", "C:/models/b"));
    }

    profiles["C:/models/b"] = "match";
    QJsonObject changed = current;
    changed["savedPredictionProfiles"] = profiles;
    ASSERT_TRUE(baseline != ModelBenchmarkPairEnvironmentKey(encode(changed), "C:/models/a", "C:/models/b"));

    predictors["C:/models/b"] = QJsonObject{{"engine", 2}, {"revision", "v2-b"}};
    changed = current;
    changed["externalPrediction"] = predictors;
    ASSERT_TRUE(baseline != ModelBenchmarkPairEnvironmentKey(encode(changed), "C:/models/a", "C:/models/b"));
    ASSERT_TRUE(ModelBenchmarkPairEnvironmentKey("not-json", "a", "b").isEmpty());
    ASSERT_TRUE(ModelBenchmarkPairEnvironmentKey(encode(current), "missing", "C:/models/b").isEmpty());

    // LoadCoverage must project using the saved model paths and canonical ID order,
    // even when a later run reverses its roster or uses a smaller model list.
    QTemporaryDir directory;
    ModelBenchmarkLibrary library;
    QString error;
    ASSERT_TRUE(library.Open(directory.filePath("pairs.sqlite").toStdString(), &error));

    ModelBenchmarkLibraryModel a = SavedModel("A", "a", "a");
    ModelBenchmarkLibraryModel b = SavedModel("B", "b", "b");
    ModelBenchmarkLibraryModel old = SavedModel("Old", "old", "old");
    a.checkpointPath = "C:/models/a";
    b.checkpointPath = "C:/models/b";
    old.checkpointPath = "C:/models/old";
    ModelBenchmarkResult run;
    run.modelNames = { "A", "B", "Old" };
    run.games = { ScoredGame(0, 1) };
    run.rocketSimVersions = {3};
    run.evaluationIdentity = encode(environment).toStdString();
    ASSERT_TRUE(library.SaveRun("old-roster", run, { a, b, old }, {}, &error));
    run.modelNames = { "B", "A" };
    run.evaluationIdentity = encode(current).toStdString();
    ASSERT_TRUE(library.SaveRun("reversed-roster", run, { b, a }, {}, &error));

    const std::vector<ModelBenchmarkLibraryCoverage> rows = library.LoadCoverage({}, &error);
    ASSERT_TRUE(error.isEmpty());
    ASSERT_EQ(rows.size(), size_t(2));
    for (const ModelBenchmarkLibraryCoverage& row : rows) {
        ASSERT_EQ(row.pairEvaluationIdentity, baseline);
        ASSERT_EQ(row.games, qint64(1));
    }
}

TEST(ModelBenchmarkLibrary, CompatibilityMarkersNeverInvalidateHistory)
{
    QTemporaryDir directory;
    ModelBenchmarkLibrary library;
    QString error;
    ASSERT_TRUE(library.Open(directory.filePath("compatibility.sqlite").toStdString(), &error));

    const ModelBenchmarkLibraryModel a = SavedModel("A", "a", "a");
    const ModelBenchmarkLibraryModel b = SavedModel("B", "b", "b");
    ModelBenchmarkConfig config;
    ModelBenchmarkResult run;
    run.modelNames = {"A", "B"};
    run.rocketSimVersions = {3};
    run.games = {ScoredGame(0, 1), ScoredGame(1, 2)};

    // Include draws; incomplete mirrors must never satisfy the target.
    run.games[1].winner = -1;
    run.games[1].goalScored = false;
    run.games[1].includedInRating = false;
    run.games.push_back(ScoredGame(0, 3));
    run.games.back().mirrorCompleted = false;
    ASSERT_TRUE(library.SaveRun("legacy", run, {a, b}, {}, &error));
    RecordModelBenchmarkEnvironment(run, config, "RocketSim-v3", "old-physics", "insertion-order", {});
    QJsonObject identity = QJsonDocument::fromJson(QByteArray::fromStdString(run.evaluationIdentity)).object();
    ASSERT_EQ(identity["benchmarkCompatibilityVersion"].toInt(), 1);
    identity.remove("benchmarkCompatibilityVersion");
    identity.remove("savedPredictionProfiles");
    run.evaluationIdentity = QJsonDocument(identity).toJson(QJsonDocument::Compact).toStdString();
    ASSERT_TRUE(library.SaveRun("old-metadata", run, {a, b}, {}, &error));
    RecordModelBenchmarkEnvironment(run, config, "RocketSim-v3", "new-physics", "new-order", {}, false);
    ASSERT_TRUE(library.SaveRun("routine-update", run, {a, b}, {}, &error));

    identity = QJsonDocument::fromJson(QByteArray::fromStdString(run.evaluationIdentity)).object();
    identity["benchmarkCompatibilityVersion"] = 2;
    run.evaluationIdentity = QJsonDocument(identity).toJson(QJsonDocument::Compact).toStdString();
    ASSERT_TRUE(library.SaveRun("manual-break", run, {a, b}, {}, &error));
    run.stochasticKickoff = true;
    run.kickoffTemperature = 0.7f;
    ASSERT_TRUE(library.SaveRun("other-kickoff", run, {a, b}, {}, &error));
    run.stochasticKickoff = false;
    run.rocketSimVersions = {2};
    run.evaluationIdentity.clear();
    ASSERT_TRUE(library.SaveRun("other-simulator", run, {a, b}, {}, &error));
    run.rocketSimVersions = {0};
    run.engineVersion = 0;
    ASSERT_TRUE(library.SaveRun("unknown-simulator", run, {a, b}, {}, &error));

    ModelBenchmarkLibraryFilter filter;
    filter.engineVersion = 3;
    filter.stochasticKickoff = 0;
    const std::vector<ModelBenchmarkLibraryCoverage> coverage = library.LoadCoverage(filter, &error);
    ASSERT_TRUE(error.isEmpty());
    ASSERT_EQ(coverage.size(), size_t(4));

    qint64 baselineGames = 0;
    qint64 nextVersionGames = 0;
    for (const ModelBenchmarkLibraryCoverage& row : coverage) {
        ASSERT_EQ(row.engineVersion, 3);
        ASSERT_EQ(row.games, qint64(2));
        ASSERT_EQ(row.lastSnapshotId, uint64_t(3));
        if (row.compatibilityVersion == 1)
            baselineGames += row.games;

        if (row.compatibilityVersion == 2)
            nextVersionGames += row.games;
    }

    ASSERT_EQ(baselineGames, qint64(6));
    ASSERT_EQ(nextVersionGames, qint64(2));

    // Compatibility is read without assigning invented build provenance to old runs.
    filter.runId = "legacy";
    const std::vector<ModelBenchmarkLibraryCoverage> legacy = library.LoadCoverage(filter, &error);
    ASSERT_EQ(legacy.size(), size_t(1));
    ASSERT_TRUE(legacy[0].evaluationIdentity.isEmpty());
    ASSERT_EQ(legacy[0].compatibilityVersion, 1);
}

TEST(ModelBenchmarkLibrary, CompatibilityMarkerValidation)
{
    ASSERT_EQ(RecordedModelBenchmarkCompatibilityVersion(""), 1);
    ASSERT_EQ(RecordedModelBenchmarkCompatibilityVersion("{}"), 1);
    ASSERT_EQ(RecordedModelBenchmarkCompatibilityVersion(R"({"benchmarkCompatibilityVersion":2})"), 2);

    for (const char* identity : {"not-json", "[]", R"({"benchmarkCompatibilityVersion":0})",
        R"({"benchmarkCompatibilityVersion":-1})", R"({"benchmarkCompatibilityVersion":1.5})",
        R"({"benchmarkCompatibilityVersion":"1"})", R"({"benchmarkCompatibilityVersion":null})",
        R"({"benchmarkCompatibilityVersion":9999999999999999999})"})
        ASSERT_EQ(RecordedModelBenchmarkCompatibilityVersion(identity), 0);

    ASSERT_EQ(CurrentModelBenchmarkCompatibilityVersion(0), 0);
    for (int engine : {2, 3})
        ASSERT_EQ(CurrentModelBenchmarkCompatibilityVersion(engine), 1);
}

TEST(ModelBenchmarkLibrary, CoverageIndexPreservesGroupedCounts)
{
    QTemporaryDir directory;
    const fs::path path(directory.filePath("coverage.sqlite").toStdString());
    ModelBenchmarkLibrary library;
    QString error;
    ASSERT_TRUE(library.Open(path, &error));

    const ModelBenchmarkLibraryModel a = SavedModel("A", "a", "a");
    const ModelBenchmarkLibraryModel b = SavedModel("B", "b", "b");
    ModelBenchmarkResult run;
    run.modelNames = {"A", "B", "A alias"};
    run.rocketSimVersions = {2};
    run.games = {ScoredGame(0, 9), ScoredGame(1, 20), ScoredGame(0, 100), ScoredGame(0, 999)};

    // Two local slots refer to the same saved identity.
    run.games[2].modelA = 2;
    run.games[2].blueModel = 2;
    run.games[2].winner = 2;
    run.games[3].mirrorCompleted = false;
    ASSERT_TRUE(library.SaveRun("counts", run, {a, b, a}, {}, &error));

    const std::vector<ModelBenchmarkLibraryCoverage> before = library.LoadCoverage({}, &error);
    ASSERT_TRUE(error.isEmpty());
    ASSERT_EQ(before.size(), size_t(1));
    ASSERT_EQ(before[0].games, qint64(3));
    ASSERT_EQ(before[0].lastSnapshotId, uint64_t(999));
    ASSERT_TRUE(ModelBenchmarkLibrary::EnsureCoverageIndex(path));
    ASSERT_TRUE(ModelBenchmarkLibrary::EnsureCoverageIndex(path));
    library.Close();

    ASSERT_TRUE(library.OpenReadOnly(path, &error));
    const std::vector<ModelBenchmarkLibraryCoverage> after = library.LoadCoverage({}, &error);
    ASSERT_TRUE(error.isEmpty());
    ASSERT_EQ(after.size(), before.size());
    ASSERT_EQ(after[0].games, before[0].games);
    ASSERT_EQ(after[0].lastSnapshotId, before[0].lastSnapshotId);
    ASSERT_EQ(after[0].modelA, before[0].modelA);
    ASSERT_EQ(after[0].modelB, before[0].modelB);
    ASSERT_EQ(after[0].compatibilityVersion, before[0].compatibilityVersion);
}

TEST(ModelBenchmarkLibrary, PersistentFingerprintCacheAndStrictVerification)
{
    QTemporaryDir directory;
    const fs::path root = fs::path(directory.path().toStdString()) / "Models";
    const fs::path checkpoint = root / "test" / "100";
    fs::create_directories(checkpoint);
    const fs::path policy = checkpoint / "POLICY.lt";
    const fs::path setup = root / "test" / "training_setup.json";
    std::ofstream(policy) << "first policy";
    std::ofstream(setup) << "{\"saved\":true}";

    const bool allowCached = true;
    const QString first = ModelBenchmarkLibrary::FingerprintCheckpoint(checkpoint, setup);
    ASSERT_FALSE(first.isEmpty());
    ASSERT_EQ(first, ModelBenchmarkLibrary::FingerprintCheckpoint(checkpoint, setup, allowCached));

    const fs::path cacheDir = root / ".rocketforge" / "fingerprints";
    ASSERT_TRUE(fs::is_directory(cacheDir));
    const fs::path cachePath = fs::directory_iterator(cacheDir)->path();
    const fs::file_time_type cacheTime = fs::last_write_time(cachePath);
    ASSERT_EQ(first, ModelBenchmarkLibrary::FingerprintCheckpoint(checkpoint, setup, allowCached));
    ASSERT_EQ(cacheTime, fs::last_write_time(cachePath));

    const fs::file_time_type policyTime = fs::last_write_time(policy);
    std::ofstream(policy) << "other policy";
    fs::last_write_time(policy, policyTime);
    const QString changed = ModelBenchmarkLibrary::FingerprintCheckpoint(checkpoint, setup, allowCached);
    ASSERT_FALSE(changed.isEmpty());
    ASSERT_FALSE(changed == first);
    ASSERT_EQ(changed, ModelBenchmarkLibrary::FingerprintCheckpoint(checkpoint, setup));

    // Even a syntactically valid but altered cache must never authorize a run.
    QFile cached(QString::fromStdWString(cachePath.wstring()));
    ASSERT_TRUE(cached.open(QIODevice::ReadOnly));
    QJsonObject record = QJsonDocument::fromJson(cached.readAll()).object();
    cached.close();
    record["sha256"] = QString(64, 'a');
    ASSERT_TRUE(cached.open(QIODevice::WriteOnly | QIODevice::Truncate));
    cached.write(QJsonDocument(record).toJson());
    cached.close();
    ASSERT_EQ(ModelBenchmarkLibrary::FingerprintCheckpoint(checkpoint, setup), changed);

    std::ofstream(cachePath) << "broken JSON";
    ASSERT_EQ(ModelBenchmarkLibrary::FingerprintCheckpoint(checkpoint, setup, allowCached), changed);

    fs::remove(policy);
    ASSERT_FALSE(ModelBenchmarkLibrary::FingerprintCheckpoint(checkpoint, setup, allowCached) == changed);
}

TEST(ModelBenchmarkLibrary, ImportedPolicyFingerprintTracksManifestAndDeclaredWeights) {
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const fs::path root(directory.path().toStdString());
    const fs::path checkpoint = root / "Models" / "Imported Models" / "Fixture";
    fs::create_directories(checkpoint);
    const fs::path manifest = checkpoint / "INFERENCE_POLICY.json";
    const fs::path weights = checkpoint / "policy.pt";
    std::ofstream(manifest) << R"({"artifact":"policy.pt","format":"torchscript"})";
    std::ofstream(weights) << "original weights";

    const bool allowCached = true;
    const QString first = ModelBenchmarkLibrary::FingerprintCheckpoint(checkpoint);
    ASSERT_EQ(first.size(), 64);
    EXPECT_EQ(first, ModelBenchmarkLibrary::FingerprintCheckpoint(checkpoint, {}, allowCached));

    std::ofstream(checkpoint / "unused-original.pt") << "not used for inference";
    EXPECT_EQ(first, ModelBenchmarkLibrary::FingerprintCheckpoint(checkpoint));

    std::ofstream(weights) << "replacement policy weights";
    const QString changedWeights = ModelBenchmarkLibrary::FingerprintCheckpoint(checkpoint, {}, allowCached);
    EXPECT_FALSE(changedWeights.isEmpty());
    EXPECT_NE(first, changedWeights);
    EXPECT_EQ(changedWeights, ModelBenchmarkLibrary::FingerprintCheckpoint(checkpoint));

    std::ofstream(manifest) << R"({"artifact":"policy.pt","format":"torchscript","version":2})";
    const QString changedManifest = ModelBenchmarkLibrary::FingerprintCheckpoint(checkpoint, {}, allowCached);
    EXPECT_FALSE(changedManifest.isEmpty());
    EXPECT_NE(changedWeights, changedManifest);

    fs::remove(weights);
    EXPECT_TRUE(ModelBenchmarkLibrary::FingerprintCheckpoint(checkpoint, {}, allowCached).isEmpty());

    std::ofstream(checkpoint.parent_path() / "outside.pt") << "outside checkpoint";
    std::ofstream(manifest) << R"({"artifact":"../outside.pt"})";
    EXPECT_TRUE(ModelBenchmarkLibrary::FingerprintCheckpoint(checkpoint).isEmpty());

    std::ofstream(manifest) << "invalid manifest";
    EXPECT_TRUE(ModelBenchmarkLibrary::FingerprintCheckpoint(checkpoint).isEmpty());
}

TEST(ModelBenchmarkLibrary, MainHistoryRetainsSimulatorAndUnsignedSnapshotIdentity) {
    ModelBenchmarkLibrary library;
    const fs::path path = RocketForgeTests::TestTempDir() / "main-history.sqlite";
    QString error;
    ASSERT_TRUE(library.Open(path, &error));

    const std::vector<ModelBenchmarkLibraryModel> models{SavedModel("A", "1", "a"), SavedModel("B", "1", "b")};
    ModelBenchmarkResult run;
    run.engineVersion = 3;
    run.modelNames = {"A", "B"};
    run.games = {ScoredGame(0, 1)};
    ASSERT_TRUE(library.SaveRun("embedded", run, models, {}, &error));

    const QString connection = QUuid::createUuid().toString();
    {
        QSqlDatabase db = QSqlDatabase::addDatabase("QSQLITE", connection);
        db.setDatabaseName(QString::fromStdString(path.string()));
        ASSERT_TRUE(db.open());
        QSqlQuery query(db);
        ASSERT_TRUE(query.exec("SELECT result_json FROM runs WHERE run_id='embedded'"));
        ASSERT_TRUE(query.next());
        QJsonObject object = QJsonDocument::fromJson(query.value(0).toByteArray()).object();
        object.remove("rocketSimVersions");
        object["engineVersion"] = 3;
        QJsonObject game{{"gameId", "1"}, {"snapshotId", QString::number(UINT64_MAX - 1)},
            {"modelA", 0}, {"modelB", 1}, {"blueModel", 0}, {"orangeModel", 1}, {"mode", 1},
            {"winner", 0}, {"goalTimeSeconds", 20}, {"goalTiming", 2}, {"goalScored", true},
            {"includedInRating", true}, {"mirrorCompleted", true}};
        object["games"] = QJsonArray{game};
        query.prepare("UPDATE runs SET result_json=? WHERE run_id='embedded'");
        query.addBindValue(QJsonDocument(object).toJson());
        ASSERT_TRUE(query.exec());
        ASSERT_TRUE(query.exec("DELETE FROM games WHERE run_id='embedded'"));
        ASSERT_TRUE(query.exec("UPDATE library_meta SET schema_version=2"));
    }
    QSqlDatabase::removeDatabase(connection);

    library.Close();
    ASSERT_TRUE(library.Open(path, &error));
    ModelBenchmarkLibraryFilter filter;
    filter.restrictEnvironments = true;
    filter.engineVersions = {3, 0};
    std::vector<ModelBenchmarkLibraryCoverage> coverage = library.LoadCoverage(filter, &error);
    ASSERT_TRUE(error.isEmpty()) << error.toStdString();
    ASSERT_EQ(coverage.size(), 1u);
    EXPECT_EQ(coverage[0].engineVersion, 3);
    EXPECT_EQ(coverage[0].games, 1);
    EXPECT_EQ(coverage[0].lastSnapshotId, UINT64_MAX - 1);

    const std::vector<ModelBenchmarkLibraryModel> saved = library.ListModels();
    const std::vector<ModelBenchmarkLibrarySelection> selected{{saved[0].id, true}, {saved[1].id, true}};
    const float confidence = 0.95f;
    const ModelBenchmarkLibraryAggregate aggregate = library.LoadAggregate(selected, confidence, &error, {}, filter);
    ASSERT_TRUE(error.isEmpty()) << error.toStdString();
    EXPECT_EQ(aggregate.result.games.size(), 1u);
    EXPECT_EQ(aggregate.result.engineVersion, 3);

    run.games[0].snapshotId = UINT64_MAX;
    ASSERT_TRUE(library.SaveRun("native", run, models, {}, &error));
    coverage = library.LoadCoverage(filter, &error);
    ASSERT_EQ(coverage.size(), 1u);
    EXPECT_EQ(coverage[0].games, 2);
    EXPECT_EQ(coverage[0].lastSnapshotId, UINT64_MAX);

    filter.engineVersions = {2, 0};
    EXPECT_TRUE(library.LoadCoverage(filter, &error).empty());
}

TEST(ModelBenchmarkLibrary, PredictionObjectProfilesProjectOntoExactPair) {
    const QString identity = R"({"savedPredictionProfiles":{"a":{"engine":2,"tickRate":60,"enabled":true},"b":{"engine":3,"tickRate":120,"enabled":true}},"externalPrediction":{}})";
    const QString key = ModelBenchmarkPairEnvironmentKey(identity, "a", "b");
    EXPECT_FALSE(key.isEmpty());
    EXPECT_NE(key, ModelBenchmarkPairEnvironmentKey(identity, "b", "a"));
}

#else
TEST(ModelBenchmarkLibrary, RequiresQt) { GTEST_SKIP() << "Qt is unavailable"; }
#endif
