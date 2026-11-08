#include "ModelBenchmarkLibrary.h"

#include <ModelBenchmark/CheckpointFingerprint.h>
#include <ModelBenchmark/ModelBenchmarkCompatibility.h>

#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSqlDatabase>
#include <QSqlError>
#include <QSqlQuery>
#include <QUuid>
#include <QXmlStreamReader>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>

using namespace RocketForgeTraining;

namespace
{
    constexpr int LIBRARY_SCHEMA_VERSION = 6;

    QString SqlError(const QSqlQuery& query, const QString& context)
    {
        return context + ": " + query.lastError().text();
    }

    QString LogicalKey(const ModelBenchmarkLibraryModel& model)
    {
        if (!model.logicalKey.isEmpty())
            return model.logicalKey;
        return QStringList{
            model.agentName,
            model.buildName,
            model.stageName,
            model.versionId,
            model.sourceLabel
        }.join('|');
    }

    QJsonObject EncodeTotals(const ModelBenchmarkStatTotals& value)
    {
        QJsonObject out;
        out["definitionVersion"] = value.definitionVersion;
#define WRITE_TOTAL(name) out[#name] = value.name
        WRITE_TOTAL(goals);
        WRITE_TOTAL(shots);
        WRITE_TOTAL(assists);
        WRITE_TOTAL(saves);
        WRITE_TOTAL(goalsConceded);
        WRITE_TOTAL(passes);
        WRITE_TOTAL(passGameTimeSum);
        WRITE_TOTAL(ballTouches);
        WRITE_TOTAL(opponentBallTouches);
        WRITE_TOTAL(flipResets);
        WRITE_TOTAL(bumps);
        WRITE_TOTAL(demos);
        WRITE_TOTAL(boostSteals);
        WRITE_TOTAL(bigBoostPads);
        WRITE_TOTAL(smallBoostPads);
        WRITE_TOTAL(boostConsumed);
        WRITE_TOTAL(gameTimeSum);
        WRITE_TOTAL(boostSum);
        WRITE_TOTAL(speedSum);
        WRITE_TOTAL(aerialTimeSum);
        WRITE_TOTAL(lowAerialTimeSum);
        WRITE_TOTAL(highAerialTimeSum);
        WRITE_TOTAL(lowBoostTimeSum);
        WRITE_TOTAL(supersonicTimeSum);
        WRITE_TOTAL(playerTimeSum);
        WRITE_TOTAL(opponentHalfBallSteps);
        WRITE_TOTAL(offensiveThirdBallSteps);
        WRITE_TOTAL(neutralThirdBallSteps);
        WRITE_TOTAL(defensiveThirdBallSteps);
        WRITE_TOTAL(touchHeightSum);
        WRITE_TOTAL(touchCount);
        WRITE_TOTAL(speedTowardsBallSum);
        WRITE_TOTAL(distanceToBallSum);
#undef WRITE_TOTAL
        out["stepSamples"] = QString::number(value.stepSamples);
        out["playerSamples"] = QString::number(value.playerSamples);
        return out;
    }

    ModelBenchmarkStatTotals DecodeTotals(const QJsonObject& value)
    {
        ModelBenchmarkStatTotals out;
        out.definitionVersion = value.value("definitionVersion").toInt(1);
#define READ_TOTAL(name) out.name = value.value(#name).toDouble()
        READ_TOTAL(goals);
        READ_TOTAL(shots);
        READ_TOTAL(assists);
        READ_TOTAL(saves);
        READ_TOTAL(goalsConceded);
        READ_TOTAL(passes);
        READ_TOTAL(passGameTimeSum);
        READ_TOTAL(ballTouches);
        READ_TOTAL(opponentBallTouches);
        READ_TOTAL(flipResets);
        READ_TOTAL(bumps);
        READ_TOTAL(demos);
        READ_TOTAL(boostSteals);
        READ_TOTAL(bigBoostPads);
        READ_TOTAL(smallBoostPads);
        READ_TOTAL(boostConsumed);
        READ_TOTAL(gameTimeSum);
        READ_TOTAL(boostSum);
        READ_TOTAL(speedSum);
        READ_TOTAL(aerialTimeSum);
        READ_TOTAL(lowAerialTimeSum);
        READ_TOTAL(highAerialTimeSum);
        READ_TOTAL(lowBoostTimeSum);
        READ_TOTAL(supersonicTimeSum);
        READ_TOTAL(playerTimeSum);
        READ_TOTAL(opponentHalfBallSteps);
        READ_TOTAL(offensiveThirdBallSteps);
        READ_TOTAL(neutralThirdBallSteps);
        READ_TOTAL(defensiveThirdBallSteps);
        READ_TOTAL(touchHeightSum);
        READ_TOTAL(touchCount);
        READ_TOTAL(speedTowardsBallSum);
        READ_TOTAL(distanceToBallSum);
#undef READ_TOTAL
        out.stepSamples = value.value("stepSamples").toString().toLongLong();
        out.playerSamples = value.value("playerSamples").toString().toLongLong();
        return out;
    }

    // Legacy revisions remain unidentified,
    // but recorded major families still distinguish incompatible environments for aggregation and historical top-ups.
    std::string AggregationEnvironmentKey(const ModelBenchmarkResult& result)
    {
        if (!result.evaluationIdentity.empty()) return "exact:" + result.evaluationIdentity;

        std::string key = "legacy-major:";
        for (int version : ResolveModelBenchmarkSimulatorVersions(result.rocketSimVersions, result.engineVersion))
            key += std::to_string(version) + ",";

        return key;
    }

    void CopyEnvironment(const ModelBenchmarkResult& source, ModelBenchmarkResult& destination)
    {
        destination.engineFamily = source.engineFamily;
        destination.engineRevision = source.engineRevision;
        destination.engineCarOrder = source.engineCarOrder;
        destination.integrationRevision = source.integrationRevision;
        destination.predictionConfigurationJson = source.predictionConfigurationJson;
        destination.evaluationIdentity = source.evaluationIdentity;
    }

    QByteArray EncodeResult(const ModelBenchmarkResult& result)
    {
        QJsonObject root;
        root["formatVersion"] = 1;
        QJsonArray versions;
        for (int version : ResolveModelBenchmarkSimulatorVersions(result.rocketSimVersions, result.engineVersion))
            versions.append(version);
        root["rocketSimVersions"] = versions;
        root["engineFamily"] = QString::fromStdString(result.engineFamily);
        root["engineRevision"] = QString::fromStdString(result.engineRevision);
        root["engineCarOrder"] = QString::fromStdString(result.engineCarOrder);
        root["integrationRevision"] = QString::fromStdString(result.integrationRevision);
        root["predictionConfigurationJson"] = QString::fromStdString(result.predictionConfigurationJson);
        root["evaluationIdentity"] = QString::fromStdString(result.evaluationIdentity);

        root["seed"] = QString::number(result.seed);
        root["kickoffGoalWindowSeconds"] = result.kickoffGoalWindowSeconds;
        root["stochasticKickoff"] = result.stochasticKickoff;
        root["kickoffTemperature"] = result.kickoffTemperature;
        root["kickoffInferenceSettingsMixed"] = result.kickoffInferenceSettingsMixed;
        root["eloRangeConfidence"] = result.eloRangeConfidence;
        root["statTotalsComplete"] = result.statTotalsComplete;
        root["elapsedSeconds"] = result.elapsedSeconds;
        root["simulationSeconds"] = result.simulationSeconds;

        return QJsonDocument(root).toJson(QJsonDocument::Compact);
    }

    bool DecodeResult(const QByteArray& bytes, ModelBenchmarkResult& result, QString* error)
    {
        QJsonParseError parseError;
        const QJsonDocument doc = QJsonDocument::fromJson(bytes, &parseError);
        if (doc.isNull() || !doc.isObject()) {
            if (error)
                *error = "Invalid stored benchmark JSON: " + parseError.errorString();
            return false;
        }
        const QJsonObject root = doc.object();
        result.rocketSimVersions.clear();
        for (const QJsonValue& version : root.value("rocketSimVersions").toArray())
            result.rocketSimVersions.insert(std::max(0, version.toInt()));
        result.rocketSimVersions = ResolveModelBenchmarkSimulatorVersions(result.rocketSimVersions, root.value("engineVersion").toInt(0));
        result.engineVersion = result.rocketSimVersions.size() == 1 ? *result.rocketSimVersions.begin() : -1;
        result.engineFamily = root.value("engineFamily").toString().toStdString();
        result.engineRevision = root.value("engineRevision").toString().toStdString();
        result.engineCarOrder = root.value("engineCarOrder").toString().toStdString();
        result.integrationRevision = root.value("integrationRevision").toString().toStdString();
        result.predictionConfigurationJson = root.value("predictionConfigurationJson").toString().toStdString();
        result.evaluationIdentity = root.value("evaluationIdentity").toString().toStdString();
        result.seed = root.value("seed").toString().toULongLong();
        result.kickoffGoalWindowSeconds = (float)root.value("kickoffGoalWindowSeconds").toDouble(10.0);
        result.kickoffInferenceSettingsMixed = root.value("kickoffInferenceSettingsMixed").toBool(false);
        result.stochasticKickoff = root.value("stochasticKickoff").toBool(false);
        result.kickoffTemperature = (float)root.value("kickoffTemperature").toDouble(1.0);
        result.eloRangeConfidence = (float)root.value("eloRangeConfidence").toDouble(0.95);
        result.statTotalsComplete = root.value("statTotalsComplete").toBool(false);
        result.elapsedSeconds = root.value("elapsedSeconds").toDouble();
        result.simulationSeconds = root.value("simulationSeconds").toDouble();
        for (const QJsonValue& value : root.value("games").toArray()) {
            const QJsonObject item = value.toObject();
            ModelBenchmarkGameMetadata game;
            game.gameId = item.value("gameId").toString().toULongLong();
            game.snapshotId = item.value("snapshotId").toString().toULongLong();
            game.modelA = item.value("modelA").toInt(-1);
            game.modelB = item.value("modelB").toInt(-1);
            game.blueModel = item.value("blueModel").toInt(-1);
            game.orangeModel = item.value("orangeModel").toInt(-1);
            game.mode = item.value("mode").toInt(1);
            game.winner = item.value("winner").toInt(-1);
            game.goalTimeSeconds = item.value("goalTimeSeconds").toDouble();
            game.goalTiming = (ModelBenchmarkGoalTiming)item.value("goalTiming").toInt();
            game.goalScored = item.value("goalScored").toBool();
            game.includedInRating = item.value("includedInRating").toBool();
            game.mirrorCompleted = item.value("mirrorCompleted").toBool();
            result.games.push_back(game);
        }
        for (const QJsonValue& value : root.value("statTotals").toArray()) {
            const QJsonObject item = value.toObject();
            result.statTotals.push_back({
                item.value("modelIndex").toInt(-1),
                item.value("opponentIndex").toInt(-1),
                item.value("mode").toInt(1),
                DecodeTotals(item.value("totals").toObject())
            });
        }
        return true;
    }

    bool ParseYes(const QString& value)
    {
        return value.compare("Yes", Qt::CaseInsensitive) == 0
            || value.compare("True", Qt::CaseInsensitive) == 0
            || value == "1";
    }

    int HeaderColumn(const QStringList& headers, const QString& name)
    {
        for (int i = 0; i < headers.size(); ++i) {
            if (headers[i].compare(name, Qt::CaseInsensitive) == 0)
                return i;
        }
        return -1;
    }

    QString Cell(const QStringList& row, int column)
    {
        return column >= 0 && column < row.size() ? row[column] : QString();
    }
}

class ModelBenchmarkLibrary::Impl
{
public:
    QString connectionName = "ModelBenchmarkLibrary-" + QUuid::createUuid().toString(QUuid::WithoutBraces);
    std::filesystem::path path;
    int schemaVersion = 0;

    QSqlDatabase Database() const
    {
        return QSqlDatabase::database(connectionName, false);
    }
};

ModelBenchmarkLibrary::ModelBenchmarkLibrary()
    : m_impl(std::make_unique<Impl>())
{
}

ModelBenchmarkLibrary::~ModelBenchmarkLibrary()
{
    Close();
}

bool ModelBenchmarkLibrary::Open(const std::filesystem::path& path, QString* error)
{
    if (error)
        error->clear();
    Close();
    std::error_code dirError;
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path(), dirError);
    if (dirError) {
        if (error)
            *error = "Unable to create benchmark library folder: " + QString::fromStdString(dirError.message());
        return false;
    }

    QSqlDatabase db = QSqlDatabase::addDatabase("QSQLITE", m_impl->connectionName);
    db.setDatabaseName(QString::fromStdString(path.string()));
    if (!db.open()) {
        if (error)
            *error = "Unable to open benchmark library: " + db.lastError().text();
        Close();
        return false;
    }
    QSqlQuery query(db);
    if (!query.exec("PRAGMA foreign_keys = ON")
        // DELETE journaling keeps the user-facing library portable as one file
        // whenever a transaction is not actively being committed.
        || !query.exec("PRAGMA journal_mode = DELETE")
        || !query.exec("CREATE TABLE IF NOT EXISTS library_meta (schema_version INTEGER NOT NULL)")) {
        if (error)
            *error = SqlError(query, "Unable to initialize benchmark library");
        Close();
        return false;
    }
    if (!query.exec("SELECT schema_version FROM library_meta LIMIT 1")) {
        if (error)
            *error = SqlError(query, "Unable to read benchmark library schema");
        Close();
        return false;
    }
    int schemaVersion = LIBRARY_SCHEMA_VERSION;
    if (query.next()) {
        schemaVersion = query.value(0).toInt();
        if (schemaVersion > LIBRARY_SCHEMA_VERSION) {
            if (error)
                *error = QString("This benchmark library uses newer schema version %1 (supported: %2).")
                    .arg(schemaVersion)
                    .arg(LIBRARY_SCHEMA_VERSION);
            Close();
            return false;
        }
    } else {
        QSqlQuery insert(db);
        insert.prepare("INSERT INTO library_meta(schema_version) VALUES(?)");
        insert.addBindValue(LIBRARY_SCHEMA_VERSION);
        if (!insert.exec()) {
            if (error)
                *error = SqlError(insert, "Unable to initialize benchmark library schema");
            Close();
            return false;
        }
    }

    const QStringList schema = {
        "CREATE TABLE IF NOT EXISTS workspace_presets (name TEXT PRIMARY KEY COLLATE NOCASE, settings_json TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS run_setups (run_id TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE, settings_json TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS models ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, logical_key TEXT NOT NULL, display_name TEXT NOT NULL, "
        "agent_name TEXT, build_name TEXT, stage_name TEXT, branch_name TEXT, version_id TEXT, source_label TEXT, "
        "checkpoint_path TEXT, fingerprint TEXT NOT NULL, latest_timesteps TEXT, fingerprint_verified INTEGER NOT NULL DEFAULT 0, last_trained_at TEXT, "
        "created_at TEXT NOT NULL, UNIQUE(logical_key, fingerprint))",
        "CREATE TABLE IF NOT EXISTS runs ("
        "run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, source_kind TEXT NOT NULL DEFAULT 'native', result_json BLOB NOT NULL)",
        "CREATE TABLE IF NOT EXISTS run_models ("
        "run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE, model_index INTEGER NOT NULL, "
        "model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE, PRIMARY KEY(run_id, model_index))",
        "CREATE TABLE IF NOT EXISTS games ("
        "run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE, ordinal INTEGER NOT NULL, "
        "game_id TEXT NOT NULL, snapshot_id TEXT NOT NULL, model_a INTEGER NOT NULL, model_b INTEGER NOT NULL, "
        "blue_model INTEGER NOT NULL, orange_model INTEGER NOT NULL, mode INTEGER NOT NULL, winner INTEGER NOT NULL, "
        "goal_time_seconds REAL NOT NULL, goal_timing INTEGER NOT NULL, goal_scored INTEGER NOT NULL, "
        "included_in_rating INTEGER NOT NULL, mirror_completed INTEGER NOT NULL, PRIMARY KEY(run_id, ordinal))",
        "CREATE TABLE IF NOT EXISTS stat_totals ("
        "run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE, ordinal INTEGER NOT NULL, "
        "model_index INTEGER NOT NULL, opponent_index INTEGER NOT NULL, mode INTEGER NOT NULL, totals_json BLOB NOT NULL, "
        "PRIMARY KEY(run_id, ordinal))",
        "CREATE INDEX IF NOT EXISTS idx_run_models_model ON run_models(model_id)",
        "CREATE TABLE IF NOT EXISTS comparison_sets ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL COLLATE NOCASE UNIQUE, "
        "confidence_percent REAL NOT NULL)",
        "CREATE TABLE IF NOT EXISTS comparison_members ("
        "set_id INTEGER NOT NULL REFERENCES comparison_sets(id) ON DELETE CASCADE, "
        "model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE, "
        "included INTEGER NOT NULL, is_primary INTEGER NOT NULL, PRIMARY KEY(set_id, model_id))"
    };
    for (const QString& sql : schema) {
        if (!query.exec(sql)) {
            if (error)
                *error = SqlError(query, "Unable to create benchmark library tables");
            Close();
            return false;
        }
    }
    if (schemaVersion < 2) {
        bool hasBranchColumn = false;
        QSqlQuery columns(db);
        if (!columns.exec("PRAGMA table_info(models)")) {
            if (error)
                *error = SqlError(columns, "Unable to inspect benchmark library schema");
            Close();
            return false;
        }
        while (columns.next()) {
            if (columns.value(1).toString() == "branch_name") {
                hasBranchColumn = true;
                break;
            }
        }
        columns.finish();
        if (!hasBranchColumn && !query.exec("ALTER TABLE models ADD COLUMN branch_name TEXT")) {
            if (error)
                *error = SqlError(query, "Unable to add model branch metadata");
            Close();
            return false;
        }
    }

    if (schemaVersion < 5) {
        bool hasTrainingDate = false;
        QSqlQuery columns(db);
        if (!columns.exec("PRAGMA table_info(models)")) {
            if (error) *error = SqlError(columns, "Unable to inspect model metadata");
            Close();
            return false;
        }
        while (columns.next())
            hasTrainingDate |= columns.value(1).toString() == "last_trained_at";
        columns.finish();

        if (!hasTrainingDate && !query.exec("ALTER TABLE models ADD COLUMN last_trained_at TEXT")) {
            if (error) *error = SqlError(query, "Unable to add model training dates");
            Close();
            return false;
        }
    }

    if (schemaVersion < LIBRARY_SCHEMA_VERSION) {
        // Only simulator metadata changes. Keep unknown JSON fields, game rows, and setup snapshots intact.
        if (!db.transaction()) {
            if (error) *error = "Unable to begin simulator history migration: " + db.lastError().text();
            Close();
            return false;
        }

        QSqlQuery history(db);
        bool migrated = history.exec("SELECT run_id, result_json FROM runs");
        QString migrationError = migrated ? QString() : SqlError(history, "Unable to read simulator history");
        std::vector<std::pair<QString, QByteArray>> updates;
        while (migrated && history.next()) {
            const QByteArray bytes = history.value(1).toByteArray();
            ModelBenchmarkResult result;
            if (!DecodeResult(bytes, result, &migrationError)) {
                migrated = false;
                break;
            }

            QJsonObject object = QJsonDocument::fromJson(bytes).object();
            QJsonArray versions;
            for (int version : result.rocketSimVersions) versions.append(version);
            if (object.value("rocketSimVersions") == versions
                && (!object.contains("engineVersion") || object.value("engineVersion").toInt() == result.engineVersion)) continue;

            object["rocketSimVersions"] = versions;
            object["engineVersion"] = result.engineVersion;
            updates.emplace_back(history.value(0).toString(), QJsonDocument(object).toJson(QJsonDocument::Compact));
        }
        history.finish();

        for (const auto& [id, bytes] : updates) {
            if (!migrated) break;

            QSqlQuery update(db);
            update.prepare("UPDATE runs SET result_json=? WHERE run_id=?");
            update.addBindValue(bytes);
            update.addBindValue(id);
            if (!update.exec()) {
                migrated = false;
                migrationError = SqlError(update, "Unable to assign historical simulator");
            }
        }

        if (!migrated) {
            db.rollback();
            if (error) *error = migrationError;
            Close();
            return false;
        }

        QSqlQuery updateVersion(db);
        updateVersion.prepare("UPDATE library_meta SET schema_version=?");
        updateVersion.addBindValue(LIBRARY_SCHEMA_VERSION);
        if (!updateVersion.exec() || !db.commit()) {
            db.rollback();
            if (error) *error = "Unable to commit simulator history migration: " + db.lastError().text();
            Close();
            return false;
        }
    }

    m_impl->path = path;
    m_impl->schemaVersion = LIBRARY_SCHEMA_VERSION;

    return true;
}

bool ModelBenchmarkLibrary::OpenReadOnly(const std::filesystem::path& path, QString* error)
{
    if (error)
        error->clear();
    Close();
    if (!std::filesystem::is_regular_file(path)) {
        if (error)
            *error = "Benchmark library does not exist.";
        return false;
    }

    QSqlDatabase db = QSqlDatabase::addDatabase("QSQLITE", m_impl->connectionName);
    db.setConnectOptions("QSQLITE_OPEN_READONLY;QSQLITE_BUSY_TIMEOUT=1000");
    db.setDatabaseName(QString::fromStdString(path.string()));
    if (!db.open()) {
        if (error)
            *error = "Unable to open benchmark library for reading: " + db.lastError().text();
        Close();
        return false;
    }

    QSqlQuery query(db);
    if (!query.exec("SELECT schema_version FROM library_meta LIMIT 1") || !query.next()) {
        if (error)
            *error = SqlError(query, "Unable to read benchmark library schema");
        Close();
        return false;
    }
    const int schemaVersion = query.value(0).toInt();
    if (schemaVersion > LIBRARY_SCHEMA_VERSION) {
        if (error) {
            *error = QString("This benchmark library uses newer schema version %1 (supported: %2).")
                .arg(schemaVersion)
                .arg(LIBRARY_SCHEMA_VERSION);
        }
        Close();
        return false;
    }

    m_impl->path = path;
    m_impl->schemaVersion = schemaVersion;

    return true;
}

void ModelBenchmarkLibrary::Close()
{
    if (!m_impl)
        return;
    {
        QSqlDatabase db = QSqlDatabase::database(m_impl->connectionName, false);
        if (db.isValid())
            db.close();
    }
    QSqlDatabase::removeDatabase(m_impl->connectionName);
    m_impl->path.clear();
}

bool ModelBenchmarkLibrary::IsOpen() const
{
    return m_impl && m_impl->Database().isOpen();
}

std::filesystem::path ModelBenchmarkLibrary::Path() const
{
    return m_impl ? m_impl->path : std::filesystem::path();
}

std::vector<ModelBenchmarkComparisonSet> ModelBenchmarkLibrary::ListComparisonSets(QString* error) const
{
    if (error)
        error->clear();

    std::vector<ModelBenchmarkComparisonSet> comparisons;
    if (!IsOpen()) {
        if (error)
            *error = "No benchmark library is open.";
        return comparisons;
    }

    QSqlQuery query(m_impl->Database());
    if (!query.exec(
        "SELECT s.id,s.name,s.confidence_percent,m.model_id,m.included,m.is_primary "
        "FROM comparison_sets s LEFT JOIN comparison_members m ON m.set_id=s.id "
        "ORDER BY s.name COLLATE NOCASE,s.id,m.model_id")) {
        if (error)
            *error = SqlError(query, "Unable to load comparison sets");
        return comparisons;
    }

    while (query.next()) {
        const qint64 id = query.value(0).toLongLong();
        if (comparisons.empty() || comparisons.back().id != id)
            comparisons.push_back({ id, query.value(1).toString(), query.value(2).toDouble(), {} });
        if (!query.value(3).isNull())
            comparisons.back().members.push_back({ query.value(3).toLongLong(), query.value(4).toBool(), query.value(5).toBool() });
    }

    return comparisons;
}

bool ModelBenchmarkLibrary::SaveComparisonSet(ModelBenchmarkComparisonSet& comparison, QString* error)
{
    if (error)
        error->clear();

    const QString name = comparison.name.trimmed();
    if (!IsOpen() || name.isEmpty() || name.size() > 120
        || !std::isfinite(comparison.confidencePercent)
        || comparison.confidencePercent < 50.0 || comparison.confidencePercent > 99.99) {
        if (error)
            *error = "Open a library, enter a name of 1 to 120 characters, and use a confidence between 50% and 99.99%.";
        return false;
    }

    QSqlDatabase db = m_impl->Database();
    if (!db.transaction()) {
        if (error)
            *error = "Unable to begin comparison save: " + db.lastError().text();
        return false;
    }

    auto fail = [&](const QString& message) {
        db.rollback();
        if (error)
            *error = message;
        return false;
    };

    QSqlQuery duplicate(db);
    duplicate.prepare("SELECT id FROM comparison_sets WHERE name=? AND id<>?");
    duplicate.addBindValue(name);
    duplicate.addBindValue(comparison.id);
    if (!duplicate.exec())
        return fail(SqlError(duplicate, "Unable to check comparison name"));
    if (duplicate.next())
        return fail("A comparison with this name already exists. Choose another name, or load that comparison to update it.");
    duplicate.finish();

    QSqlQuery save(db);
    save.prepare(comparison.id < 0
        ? "INSERT INTO comparison_sets(name,confidence_percent) VALUES(?,?)"
        : "UPDATE comparison_sets SET name=?,confidence_percent=? WHERE id=?");
    save.addBindValue(name);
    save.addBindValue(comparison.confidencePercent);
    if (comparison.id >= 0)
        save.addBindValue(comparison.id);
    if (!save.exec())
        return fail(SqlError(save, "Unable to save comparison"));
    if (save.numRowsAffected() != 1)
        return fail("This comparison no longer exists. Use Save As to create it again.");

    const qint64 id = comparison.id < 0 ? save.lastInsertId().toLongLong() : comparison.id;

    // Replace membership in the same transaction as the name and settings,
    // so a failed save never leaves a previously saved comparison empty or partially updated.
    QSqlQuery remove(db);
    remove.prepare("DELETE FROM comparison_members WHERE set_id=?");
    remove.addBindValue(id);
    if (!remove.exec())
        return fail(SqlError(remove, "Unable to update comparison members"));

    QSqlQuery member(db);
    member.prepare("INSERT INTO comparison_members(set_id,model_id,included,is_primary) "
        "SELECT ?,?,?,? WHERE EXISTS(SELECT 1 FROM run_models WHERE model_id=?)");
    for (const ModelBenchmarkComparisonMember& entry : comparison.members) {
        member.bindValue(0, id);
        member.bindValue(1, entry.modelId);
        member.bindValue(2, entry.included);
        member.bindValue(3, entry.primary);
        member.bindValue(4, entry.modelId);
        if (!member.exec())
            return fail(SqlError(member, "Unable to save comparison members"));
        if (member.numRowsAffected() != 1)
            return fail("A model in this comparison no longer has saved benchmark data. Reload the library and try again.");
    }

    if (!db.commit())
        return fail("Unable to commit comparison save: " + db.lastError().text());

    comparison.id = id;
    comparison.name = name;

    return true;
}

std::vector<ModelBenchmarkLibraryModel> ModelBenchmarkLibrary::ListModels(QString* error) const
{
    if (error)
        error->clear();
    std::vector<ModelBenchmarkLibraryModel> models;
    if (!IsOpen()) {
        if (error)
            *error = "No benchmark library is open.";
        return models;
    }
    QSqlQuery query(m_impl->Database());
    if (!query.exec(
        "SELECT m.id,m.logical_key,m.display_name,m.agent_name,m.build_name,m.stage_name,m.branch_name,m.version_id,"
        "m.source_label,m.checkpoint_path,m.fingerprint,m.latest_timesteps,m.fingerprint_verified,"
        + QString(m_impl->schemaVersion >= 5 ? "m.last_trained_at " : "NULL ") +
        "FROM models m WHERE EXISTS(SELECT 1 FROM run_models rm WHERE rm.model_id=m.id) "
        "ORDER BY m.agent_name,m.build_name,m.stage_name,m.version_id,CAST(m.latest_timesteps AS INTEGER)")) {
        if (error)
            *error = SqlError(query, "Unable to list saved models");
        return models;
    }
    std::map<QString, int> logicalCounts;
    while (query.next()) {
        ModelBenchmarkLibraryModel model;
        model.id = query.value(0).toLongLong();
        model.logicalKey = query.value(1).toString();
        model.displayName = query.value(2).toString();
        model.agentName = query.value(3).toString();
        model.buildName = query.value(4).toString();
        model.stageName = query.value(5).toString();
        model.branchName = query.value(6).toString();
        model.versionId = query.value(7).toString();
        model.sourceLabel = query.value(8).toString();
        model.checkpointPath = query.value(9).toString();
        model.fingerprint = query.value(10).toString();
        model.latestTimesteps = query.value(11).toString().toULongLong();
        model.fingerprintVerified = query.value(12).toBool();
        model.lastTrainedAt = query.value(13).toString();
        std::error_code checkpointError;
        model.availableLocally = !model.checkpointPath.isEmpty()
            && std::filesystem::exists(
                std::filesystem::path(model.checkpointPath.toStdString()),
                checkpointError)
            && !checkpointError;
        logicalCounts[model.logicalKey]++;
        models.push_back(std::move(model));
    }
    for (ModelBenchmarkLibraryModel& model : models) {
        if (logicalCounts[model.logicalKey] > 1) {
            const QString suffix = model.latestTimesteps > 0
                ? QString("%1 TS · %2").arg(model.latestTimesteps).arg(model.fingerprint.left(8))
                : model.fingerprint.left(8);
            model.displayName += " @ " + suffix;
        }
    }
    return models;
}

std::vector<ModelBenchmarkLibraryConflict> ModelBenchmarkLibrary::FindConflicts(
    const std::vector<ModelBenchmarkLibraryModel>& models,
    QString* error) const
{
    if (error)
        error->clear();
    std::vector<ModelBenchmarkLibraryConflict> conflicts;
    if (!IsOpen()) {
        if (error)
            *error = "No benchmark library is open.";
        return conflicts;
    }
    for (int incomingIndex = 0; incomingIndex < (int)models.size(); ++incomingIndex) {
        const ModelBenchmarkLibraryModel& incoming = models[incomingIndex];
        QSqlQuery exact(m_impl->Database());
        exact.prepare("SELECT 1 FROM models WHERE logical_key=? AND fingerprint=? LIMIT 1");
        exact.addBindValue(LogicalKey(incoming));
        exact.addBindValue(incoming.fingerprint);
        if (!exact.exec()) {
            if (error)
                *error = SqlError(exact, "Unable to inspect saved model identity");
            return {};
        }
        if (exact.next())
            continue;
        QSqlQuery query(m_impl->Database());
        query.prepare(
            "SELECT id,display_name,checkpoint_path,fingerprint,latest_timesteps,fingerprint_verified,"
            + QString(m_impl->schemaVersion >= 5 ? "last_trained_at " : "NULL ") +
            "FROM models WHERE logical_key=? AND fingerprint<>?");
        query.addBindValue(LogicalKey(incoming));
        query.addBindValue(incoming.fingerprint);
        if (!query.exec()) {
            if (error)
                *error = SqlError(query, "Unable to inspect model conflicts");
            return {};
        }
        ModelBenchmarkLibraryConflict conflict;
        conflict.incomingModelIndex = incomingIndex;
        conflict.incoming = incoming;
        while (query.next()) {
            ModelBenchmarkLibraryModel existing = incoming;
            existing.id = query.value(0).toLongLong();
            existing.displayName = query.value(1).toString();
            existing.checkpointPath = query.value(2).toString();
            existing.fingerprint = query.value(3).toString();
            existing.latestTimesteps = query.value(4).toString().toULongLong();
            existing.fingerprintVerified = query.value(5).toBool();
            existing.lastTrainedAt = query.value(6).toString();
            conflict.existing.push_back(std::move(existing));
        }
        if (!conflict.existing.empty())
            conflicts.push_back(std::move(conflict));
    }
    return conflicts;
}

bool ModelBenchmarkLibrary::SaveRun(
    const QString& runId,
    const ModelBenchmarkResult& result,
    const std::vector<ModelBenchmarkLibraryModel>& models,
    const std::set<qint64>& replaceModelIds,
    QString* error,
    const QString& settingsJson)
{
    if (error)
        error->clear();
    const bool missingFingerprint = std::any_of(models.begin(), models.end(), [](const auto& model) {
        return model.fingerprint.isEmpty();
    });
    const int modelCount = (int)models.size();
    auto validModelIndex = [&](int index) { return index >= 0 && index < modelCount; };
    const bool invalidGame = std::any_of(result.games.begin(), result.games.end(), [&](const auto& game) {
        return !validModelIndex(game.modelA)
            || !validModelIndex(game.modelB)
            || game.modelA == game.modelB
            || !validModelIndex(game.blueModel)
            || !validModelIndex(game.orangeModel)
            || game.blueModel == game.orangeModel
            || (game.blueModel != game.modelA && game.blueModel != game.modelB)
            || (game.orangeModel != game.modelA && game.orangeModel != game.modelB)
            || game.mode < 1
            || game.mode > 3
            || (game.goalScored && !validModelIndex(game.winner));
    });
    const bool invalidTotals = std::any_of(result.statTotals.begin(), result.statTotals.end(), [&](const auto& row) {
        return !validModelIndex(row.modelIndex)
            || !validModelIndex(row.opponentIndex)
            || row.modelIndex == row.opponentIndex
            || row.mode < 1
            || row.mode > 3;
    });
    if (!IsOpen()
        || runId.isEmpty()
        || models.size() != result.modelNames.size()
        || missingFingerprint
        || (result.games.empty() && result.statTotals.empty())
        || invalidGame
        || invalidTotals) {
        if (error)
            *error = "The benchmark run or its model metadata is incomplete.";
        return false;
    }
    QSqlDatabase db = m_impl->Database();
    if (!db.transaction()) {
        if (error)
            *error = "Unable to begin benchmark library transaction: " + db.lastError().text();
        return false;
    }
    auto fail = [&](const QString& message) {
        db.rollback();
        if (error)
            *error = message;
        return false;
    };

    QSqlQuery duplicate(db);
    duplicate.prepare("SELECT 1 FROM runs WHERE run_id=?");
    duplicate.addBindValue(runId);
    if (!duplicate.exec())
        return fail(SqlError(duplicate, "Unable to check saved run identity"));
    if (duplicate.next())
        return fail("This benchmark run is already saved in the active library.");

    std::set<QString> incomingLogicalKeys;
    for (const ModelBenchmarkLibraryModel& model : models)
        incomingLogicalKeys.insert(LogicalKey(model));
    for (qint64 id : replaceModelIds) {
        QSqlQuery inspect(db);
        inspect.prepare("SELECT logical_key FROM models WHERE id=?");
        inspect.addBindValue(id);
        if (!inspect.exec())
            return fail(SqlError(inspect, "Unable to validate old model snapshot"));
        if (!inspect.next() || !incomingLogicalKeys.contains(inspect.value(0).toString()))
            return fail("A requested replacement is not an older snapshot of an incoming model.");
        QSqlQuery remove(db);
        remove.prepare("DELETE FROM models WHERE id=?");
        remove.addBindValue(id);
        if (!remove.exec())
            return fail(SqlError(remove, "Unable to replace old model snapshot"));
    }
    QSqlQuery cleanup(db);
    if (!cleanup.exec(
        "DELETE FROM runs WHERE (SELECT COUNT(DISTINCT model_id) FROM run_models "
        "WHERE run_models.run_id=runs.run_id) < 2")) {
        return fail(SqlError(cleanup, "Unable to clean replaced benchmark data"));
    }

    std::vector<qint64> modelIds;
    for (const ModelBenchmarkLibraryModel& model : models) {
        const QString logicalKey = LogicalKey(model);
        QSqlQuery find(db);
        find.prepare("SELECT id FROM models WHERE logical_key=? AND fingerprint=?");
        find.addBindValue(logicalKey);
        find.addBindValue(model.fingerprint);
        if (!find.exec())
            return fail(SqlError(find, "Unable to match saved model snapshot"));
        if (find.next()) {
            const qint64 modelId = find.value(0).toLongLong();
            if (!model.branchName.isEmpty()) {
                QSqlQuery updateBranch(db);
                updateBranch.prepare(
                    "UPDATE models SET branch_name=? WHERE id=? AND (branch_name IS NULL OR branch_name='')");
                updateBranch.addBindValue(model.branchName);
                updateBranch.addBindValue(modelId);
                if (!updateBranch.exec())
                    return fail(SqlError(updateBranch, "Unable to update saved model branch"));
            }

            if (!model.lastTrainedAt.isEmpty()) {
                QSqlQuery updateDate(db);
                updateDate.prepare("UPDATE models SET last_trained_at=? WHERE id=? "
                    "AND (last_trained_at IS NULL OR last_trained_at='')");
                updateDate.addBindValue(model.lastTrainedAt);
                updateDate.addBindValue(modelId);
                if (!updateDate.exec())
                    return fail(SqlError(updateDate, "Unable to update model training date"));
            }

            modelIds.push_back(modelId);
            continue;
        }

        QSqlQuery insert(db);
        insert.prepare(
            "INSERT INTO models(logical_key,display_name,agent_name,build_name,stage_name,branch_name,version_id,source_label,"
            "checkpoint_path,fingerprint,latest_timesteps,fingerprint_verified,created_at,last_trained_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)");
        insert.addBindValue(logicalKey);
        insert.addBindValue(model.displayName);
        insert.addBindValue(model.agentName);
        insert.addBindValue(model.buildName);
        insert.addBindValue(model.stageName);
        insert.addBindValue(model.branchName);
        insert.addBindValue(model.versionId);
        insert.addBindValue(model.sourceLabel);
        insert.addBindValue(model.checkpointPath);
        insert.addBindValue(model.fingerprint);
        insert.addBindValue(QString::number(model.latestTimesteps));
        insert.addBindValue(model.fingerprintVerified);
        insert.addBindValue(QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
        insert.addBindValue(model.lastTrainedAt);
        if (!insert.exec())
            return fail(SqlError(insert, "Unable to save model snapshot"));
        modelIds.push_back(insert.lastInsertId().toLongLong());
    }

    QSqlQuery run(db);
    run.prepare("INSERT INTO runs(run_id,created_at,source_kind,result_json) VALUES(?,?,?,?)");
    run.addBindValue(runId);
    run.addBindValue(QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    run.addBindValue(runId.startsWith("excel-") ? "excel" : "native");
    run.addBindValue(EncodeResult(result));
    if (!run.exec())
        return fail(SqlError(run, "Unable to save benchmark run"));

    for (int index = 0; index < (int)modelIds.size(); ++index) {
        QSqlQuery mapping(db);
        mapping.prepare("INSERT INTO run_models(run_id,model_index,model_id) VALUES(?,?,?)");
        mapping.addBindValue(runId);
        mapping.addBindValue(index);
        mapping.addBindValue(modelIds[index]);
        if (!mapping.exec())
            return fail(SqlError(mapping, "Unable to save benchmark model mapping"));
    }
    QSqlQuery insertGame(db);
    insertGame.prepare(
        "INSERT INTO games(run_id,ordinal,game_id,snapshot_id,model_a,model_b,blue_model,orange_model,mode,winner,"
        "goal_time_seconds,goal_timing,goal_scored,included_in_rating,mirror_completed) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)");
    for (int index = 0; index < (int)result.games.size(); ++index) {
        const ModelBenchmarkGameMetadata& game = result.games[index];
        insertGame.bindValue(0, runId);
        insertGame.bindValue(1, index);
        insertGame.bindValue(2, QString::number(game.gameId));
        insertGame.bindValue(3, QString::number(game.snapshotId));
        insertGame.bindValue(4, game.modelA);
        insertGame.bindValue(5, game.modelB);
        insertGame.bindValue(6, game.blueModel);
        insertGame.bindValue(7, game.orangeModel);
        insertGame.bindValue(8, game.mode);
        insertGame.bindValue(9, game.winner);
        insertGame.bindValue(10, game.goalTimeSeconds);
        insertGame.bindValue(11, (int)game.goalTiming);
        insertGame.bindValue(12, game.goalScored);
        insertGame.bindValue(13, game.includedInRating);
        insertGame.bindValue(14, game.mirrorCompleted);
        if (!insertGame.exec())
            return fail(SqlError(insertGame, "Unable to save benchmark game"));
    }
    QSqlQuery insertTotals(db);
    insertTotals.prepare(
        "INSERT INTO stat_totals(run_id,ordinal,model_index,opponent_index,mode,totals_json) VALUES(?,?,?,?,?,?)");
    for (int index = 0; index < (int)result.statTotals.size(); ++index) {
        const ModelBenchmarkStatTotalsRow& row = result.statTotals[index];
        insertTotals.bindValue(0, runId);
        insertTotals.bindValue(1, index);
        insertTotals.bindValue(2, row.modelIndex);
        insertTotals.bindValue(3, row.opponentIndex);
        insertTotals.bindValue(4, row.mode);
        insertTotals.bindValue(5, QJsonDocument(EncodeTotals(row.totals)).toJson(QJsonDocument::Compact));
        if (!insertTotals.exec())
            return fail(SqlError(insertTotals, "Unable to save benchmark statistic totals"));
    }

    if (!settingsJson.isEmpty()) {
        QSqlQuery setup(db);
        setup.prepare("INSERT OR IGNORE INTO run_setups(run_id,settings_json) VALUES(?,?)");
        setup.addBindValue(runId);
        setup.addBindValue(settingsJson);
        if (!setup.exec()) return fail(SqlError(setup, "Unable to save run configuration"));
    }

    if (!db.commit())
        return fail("Unable to commit benchmark library transaction: " + db.lastError().text());
    return true;
}

ModelBenchmarkLibraryAggregate ModelBenchmarkLibrary::LoadAggregate(
    const std::vector<ModelBenchmarkLibrarySelection>& selection,
    float eloRangeConfidence,
    QString* error,
    const std::function<bool()>& isCancelled,
    const ModelBenchmarkLibraryFilter& filter,
    const std::vector<ModelBenchmarkLibraryModel>& additionalModels) const
{
    if (error)
        error->clear();
    ModelBenchmarkLibraryAggregate aggregate;
    auto cancelled = [&]() {
        return isCancelled && isCancelled();
    };
    if (cancelled())
        return aggregate;
    const bool hasPrimary = std::any_of(selection.begin(), selection.end(), [](const auto& model) {
        return model.primary;
    });
    if (!IsOpen()) {
        if (error)
            *error = "No benchmark library is open.";
        return aggregate;
    }
    if (selection.size() < 2 || !hasPrimary) {
        if (error)
            *error = selection.size() < 2
                ? "Select at least two models with saved benchmark data."
                : "Mark at least one selected saved model as Primary.";
        return aggregate;
    }

    const std::vector<ModelBenchmarkLibraryModel> allModels = ListModels(error);
    if (error && !error->isEmpty())
        return aggregate;
    if (cancelled())
        return aggregate;
    std::map<qint64, ModelBenchmarkLibraryModel> modelById;
    for (const auto& model : allModels)
        modelById[model.id] = model;
    for (const ModelBenchmarkLibraryModel& model : additionalModels) {
        // Temporary negative IDs never replace an existing saved snapshot.
        if (model.id < -1) modelById[model.id] = model;
    }

    std::map<qint64, int> selectedIndex;
    ModelBenchmarkConfig config;
    config.eloRangeConfidence = eloRangeConfidence;
    config.collectOptionalStats = true;
    for (const ModelBenchmarkLibrarySelection& selected : selection) {
        const auto itr = modelById.find(selected.modelId);
        if (itr == modelById.end()) {
            if (error)
                *error = "A selected saved model no longer exists in the library.";
            return {};
        }
        selectedIndex[selected.modelId] = (int)config.models.size();
        const auto& model = itr->second;
        ModelBenchmarkModel configModel;
        configModel.displayName = model.displayName.toStdString();
        configModel.agentName = model.agentName.toStdString();
        configModel.stageName = model.stageName.toStdString();
        configModel.versionId = model.versionId.toStdString();
        configModel.checkpointPath = model.checkpointPath.toStdString();
        configModel.primary = selected.primary;
        config.models.push_back(std::move(configModel));
    }

    std::vector<ModelBenchmarkGameMetadata> games;
    std::vector<ModelBenchmarkStatTotalsRow> totals;
    std::set<int> modelsWithGames;
    bool completeStats = true;
    uint64_t nextGameId = 1;
    uint64_t nextSnapshotId = 1;
    QSqlQuery runs(m_impl->Database());
    QStringList selectedPlaceholders;
    selectedPlaceholders.fill("?", (qsizetype)selection.size());
    runs.prepare(
        "SELECT r.run_id,r.created_at,r.result_json FROM runs r "
        "WHERE (SELECT COUNT(DISTINCT rm.model_id) FROM run_models rm WHERE rm.run_id=r.run_id "
        "AND rm.model_id IN (" + selectedPlaceholders.join(',') + ")) >= 2 "
        "ORDER BY r.created_at");
    for (const ModelBenchmarkLibrarySelection& selected : selection)
        runs.addBindValue(selected.modelId);
    if (!runs.exec()) {
        if (error)
            *error = SqlError(runs, "Unable to read saved benchmark runs");
        return {};
    }
    aggregate.loadQueryCount++;

    struct CandidateRun {
        QString id;
        QString createdAt;
        QByteArray resultJson;
    };

    struct StoredTotal {
        int modelIndex = -1;
        int opponentIndex = -1;
        int mode = 1;
        QByteArray json;
    };

    std::vector<CandidateRun> candidateRuns;
    while (runs.next()) {
        if (cancelled())
            return {};
        candidateRuns.push_back({ runs.value(0).toString(), runs.value(1).toString(), runs.value(2).toByteArray() });
    }

    std::map<QString, std::map<int, qint64>> mappingsByRun;
    std::map<QString, std::vector<ModelBenchmarkGameMetadata>> gamesByRun;
    std::map<QString, std::vector<StoredTotal>> totalsByRun;
    constexpr int BulkRunChunkSize = 400;
    for (size_t offset = 0; offset < candidateRuns.size(); offset += BulkRunChunkSize) {
        if (cancelled())
            return {};
        const size_t count = std::min(candidateRuns.size() - offset, (size_t)BulkRunChunkSize);
        QStringList placeholders;
        placeholders.fill("?", (qsizetype)count);
        const QString inClause = placeholders.join(',');
        auto bindRunIds = [&](QSqlQuery& query) {
            for (size_t index = 0; index < count; ++index)
                query.addBindValue(candidateRuns[offset + index].id);
        };

        QSqlQuery mappings(m_impl->Database());
        mappings.prepare(
            "SELECT run_id,model_index,model_id FROM run_models WHERE run_id IN ("
            + inClause + ") ORDER BY run_id,model_index");
        bindRunIds(mappings);
        if (!mappings.exec()) {
            if (error)
                *error = SqlError(mappings, "Unable to read saved benchmark model mappings");
            return {};
        }
        aggregate.loadQueryCount++;

        while (mappings.next()) {
            if (cancelled())
                return {};
            mappingsByRun[mappings.value(0).toString()][mappings.value(1).toInt()] = mappings.value(2).toLongLong();
        }

        QSqlQuery storedGames(m_impl->Database());
        storedGames.prepare(
            "SELECT run_id,game_id,snapshot_id,model_a,model_b,blue_model,orange_model,mode,winner,goal_time_seconds,"
            "goal_timing,goal_scored,included_in_rating,mirror_completed FROM games WHERE run_id IN ("
            + inClause + ") ORDER BY run_id,ordinal");
        bindRunIds(storedGames);
        if (!storedGames.exec()) {
            if (error)
                *error = SqlError(storedGames, "Unable to read stored benchmark games");
            return {};
        }
        aggregate.loadQueryCount++;

        while (storedGames.next()) {
            if (cancelled())
                return {};
            ModelBenchmarkGameMetadata game;
            game.gameId = storedGames.value(1).toString().toULongLong();
            game.snapshotId = storedGames.value(2).toString().toULongLong();
            game.modelA = storedGames.value(3).toInt();
            game.modelB = storedGames.value(4).toInt();
            game.blueModel = storedGames.value(5).toInt();
            game.orangeModel = storedGames.value(6).toInt();
            game.mode = storedGames.value(7).toInt();
            game.winner = storedGames.value(8).toInt();
            game.goalTimeSeconds = storedGames.value(9).toDouble();
            game.goalTiming = (ModelBenchmarkGoalTiming)storedGames.value(10).toInt();
            game.goalScored = storedGames.value(11).toBool();
            game.includedInRating = storedGames.value(12).toBool();
            game.mirrorCompleted = storedGames.value(13).toBool();
            gamesByRun[storedGames.value(0).toString()].push_back(game);
        }

        QSqlQuery storedTotals(m_impl->Database());
        storedTotals.prepare(
            "SELECT run_id,model_index,opponent_index,mode,totals_json FROM stat_totals WHERE run_id IN ("
            + inClause + ") ORDER BY run_id,ordinal");
        bindRunIds(storedTotals);
        if (!storedTotals.exec()) {
            if (error)
                *error = SqlError(storedTotals, "Unable to read stored benchmark statistics");
            return {};
        }
        aggregate.loadQueryCount++;

        while (storedTotals.next()) {
            if (cancelled())
                return {};
            totalsByRun[storedTotals.value(0).toString()].push_back({
                storedTotals.value(1).toInt(),
                storedTotals.value(2).toInt(),
                storedTotals.value(3).toInt(),
                storedTotals.value(4).toByteArray()
            });
        }
    }

    // Build a roster-specific catalog before applying the display filters,
    // so alternative environments stay selectable even when one is currently empty.
    ModelBenchmarkLibraryFilter catalogFilter = filter;
    catalogFilter.engineVersion = 0;
    catalogFilter.restrictEnvironments = false;
    catalogFilter.evaluationIdentity.clear();
    catalogFilter.environmentKey.clear();

    for (const CandidateRun& candidate : candidateRuns) {
        if (cancelled()) return {};
        if (!filter.runId.isEmpty() && filter.runId != candidate.id) continue;

        ModelBenchmarkResult source;
        QString decodeError;
        if (!DecodeResult(candidate.resultJson, source, &decodeError) || !catalogFilter.Matches(source)) continue;

        const std::map<int, qint64>& mapping = mappingsByRun[candidate.id];
        const std::string environmentKey = AggregationEnvironmentKey(source);
        for (const ModelBenchmarkGameMetadata& game : gamesByRun[candidate.id].empty() ? source.games : gamesByRun[candidate.id]) {
            if (game.mode < 1 || game.mode > 3 || !(filter.modeMask & (1 << (game.mode - 1)))) continue;

            const auto a = mapping.find(game.modelA), b = mapping.find(game.modelB);
            if (a == mapping.end() || b == mapping.end()) continue;

            const auto selectedA = selectedIndex.find(a->second), selectedB = selectedIndex.find(b->second);
            if (selectedA == selectedIndex.end() || selectedB == selectedIndex.end()
                || (!config.models[selectedA->second].primary && !config.models[selectedB->second].primary)) continue;

            auto [entry, inserted] = aggregate.availableEnvironments.try_emplace(environmentKey);
            ModelBenchmarkLibraryEnvironment& environment = entry->second;
            if (inserted) {
                CopyEnvironment(source, environment.provenance);
                environment.provenance.rocketSimVersions = source.rocketSimVersions;
            }

            if (game.mirrorCompleted) {
                ++environment.games;
                ++environment.modelGames[selectedA->second];
                ++environment.modelGames[selectedB->second];
            }
        }
    }

    ModelBenchmarkResult sourceEnvironment;
    bool kickoffInferenceSettingsMixed = false;
    std::set<int> rocketSimVersions;
    for (const CandidateRun& candidate : candidateRuns) {
        if (cancelled())
            return {};
        const QString& runId = candidate.id;
        if (!filter.runId.isEmpty() && filter.runId != runId) continue;
        const QString& runAt = candidate.createdAt;
        const std::map<int, qint64>& sourceModelIds = mappingsByRun[runId];

        ModelBenchmarkResult source;
        QString decodeError;
        if (!DecodeResult(candidate.resultJson, source, &decodeError)) {
            aggregate.warnings << QString("Skipped run %1: %2").arg(runId, decodeError);
            continue;
        }
        if (!filter.Matches(source)) continue;
        if (!gamesByRun[runId].empty()) source.games = gamesByRun.at(runId);

        for (const StoredTotal& storedTotal : totalsByRun[runId]) {
            if (cancelled())
                return {};
            QJsonParseError totalsError;
            const QJsonDocument totalsDoc = QJsonDocument::fromJson(storedTotal.json, &totalsError);
            if (!totalsDoc.isObject()) {
                aggregate.warnings << QString("Skipped invalid statistic totals in run %1.").arg(runId);
                continue;
            }
            source.statTotals.push_back({
                storedTotal.modelIndex,
                storedTotal.opponentIndex,
                storedTotal.mode,
                DecodeTotals(totalsDoc.object())
            });
        }

        bool contributed = false;
        ModelBenchmarkLibraryEnvironment environment;
        CopyEnvironment(source, environment.provenance);
        environment.provenance.rocketSimVersions = source.rocketSimVersions;
        for (ModelBenchmarkGameMetadata game : source.games) {
            if (game.mode < 1 || game.mode > 3 || !(filter.modeMask & (1 << (game.mode - 1)))) continue;
            if (cancelled())
                return {};
            const auto modelAId = sourceModelIds.find(game.modelA);
            const auto modelBId = sourceModelIds.find(game.modelB);
            if (modelAId == sourceModelIds.end() || modelBId == sourceModelIds.end())
                continue;
            const auto newA = selectedIndex.find(modelAId->second);
            const auto newB = selectedIndex.find(modelBId->second);
            if (newA == selectedIndex.end() || newB == selectedIndex.end())
                continue;
            auto remap = [&](int sourceIndex) {
                const auto sourceId = sourceModelIds.find(sourceIndex);
                if (sourceId == sourceModelIds.end())
                    return -1;
                const auto selected = selectedIndex.find(sourceId->second);
                return selected == selectedIndex.end() ? -1 : selected->second;
            };
            game.modelA = newA->second;
            game.modelB = newB->second;
            game.blueModel = remap(game.blueModel);
            game.orangeModel = remap(game.orangeModel);
            game.winner = remap(game.winner);
            aggregate.coverageGames.push_back(game);
            if (!config.models[newA->second].primary && !config.models[newB->second].primary)
                continue;
            game.gameId = nextGameId++;
            game.snapshotId = nextSnapshotId++;
            games.push_back(game);
            if (game.mirrorCompleted) {
                ++environment.games;
                ++environment.modelGames[game.modelA];
                ++environment.modelGames[game.modelB];
            }
            modelsWithGames.insert(game.modelA);
            modelsWithGames.insert(game.modelB);
            contributed = true;
        }

        for (const ModelBenchmarkStatTotalsRow& row : source.statTotals) {
            if (!(filter.modeMask & (1 << (row.mode - 1)))) continue;
            if (cancelled())
                return {};
            const auto modelId = sourceModelIds.find(row.modelIndex);
            const auto opponentId = sourceModelIds.find(row.opponentIndex);
            if (modelId == sourceModelIds.end() || opponentId == sourceModelIds.end())
                continue;
            const auto model = selectedIndex.find(modelId->second);
            const auto opponent = selectedIndex.find(opponentId->second);
            if (model == selectedIndex.end() || opponent == selectedIndex.end())
                continue;
            if (!config.models[model->second].primary && !config.models[opponent->second].primary)
                continue;
            totals.push_back({ model->second, opponent->second, row.mode, row.totals });
            contributed = true;
        }

        if (contributed) {
            ModelBenchmarkLibraryEnvironment& combinedEnvironment = aggregate.environments[AggregationEnvironmentKey(source)];
            combinedEnvironment.provenance = environment.provenance;
            combinedEnvironment.games += environment.games;
            for (const auto& [model, count] : environment.modelGames)
                combinedEnvironment.modelGames[model] += count;

            rocketSimVersions.insert(source.rocketSimVersions.begin(), source.rocketSimVersions.end());
            if (aggregate.contributingRuns == 0) {
                CopyEnvironment(source, sourceEnvironment);
                config.stochasticKickoff = source.stochasticKickoff;
                config.kickoffTemperature = source.kickoffTemperature;
            } else if (config.stochasticKickoff != source.stochasticKickoff
                || (source.stochasticKickoff && config.kickoffTemperature != source.kickoffTemperature)) {
                kickoffInferenceSettingsMixed = true;
            }

            kickoffInferenceSettingsMixed = kickoffInferenceSettingsMixed || source.kickoffInferenceSettingsMixed;
            aggregate.contributingRuns++;
            if (aggregate.firstRunAt.isEmpty() || runAt < aggregate.firstRunAt)
                aggregate.firstRunAt = runAt;
            if (aggregate.latestRunAt.isEmpty() || runAt > aggregate.latestRunAt)
                aggregate.latestRunAt = runAt;
            completeStats = completeStats && source.statTotalsComplete;
        }
    }

    for (int modelIndex = 0; modelIndex < (int)config.models.size(); ++modelIndex) {
        if (!modelsWithGames.contains(modelIndex)) {
            aggregate.warnings << QString::fromStdString(config.models[modelIndex].displayName)
                + ": no eligible saved games match the current Primary selection.";
        }
    }

    if (cancelled())
        return {};
    try {
        aggregate.result = SummarizeModelBenchmarkData(config, games, totals, completeStats);

        if (ModelBenchmarkHasMixedEnvironments(aggregate)) {
            aggregate.result.engineFamily = "Mixed environments";
            aggregate.warnings << "Mixed environments: pooled Elo describes the included historical games. Unequal simulator coverage can affect rankings; choose environments in Test Setup for controlled comparisons.";
        } else {
            CopyEnvironment(sourceEnvironment, aggregate.result);
        }

        aggregate.result.kickoffInferenceSettingsMixed = kickoffInferenceSettingsMixed;
        if (!rocketSimVersions.empty())
            aggregate.result.rocketSimVersions = std::move(rocketSimVersions);

        std::set<int> knownEngines = aggregate.result.rocketSimVersions;
        knownEngines.erase(0);
        aggregate.result.engineVersion = knownEngines.empty() ? 0 : knownEngines.size() == 1 ? *knownEngines.begin() : -1;
        aggregate.result.elapsedSeconds = 0.0;
        aggregate.result.simulationSeconds = 0.0;
        aggregate.result.gamesPerSecond = 0.0;
        if (!completeStats) {
            aggregate.result.optionalStats.clear();
            aggregate.result.optionalStatsByMode.clear();
            aggregate.warnings << "Optional statistics are unavailable because at least one contributing run lacks lossless per-matchup totals.";
        }
    } catch (const std::exception& e) {
        if (error)
            *error = QString::fromStdString(e.what());
    }
    return aggregate;
}

bool ModelBenchmarkLibraryFilter::MatchesEnvironmentSelection(int engine, const QString& key) const
{
    if (!restrictEnvironments) return true;
    if (engine == -1) return MatchesEnvironmentSelection(2, key) && MatchesEnvironmentSelection(3, key);
    if (engine == 0) engine = 2; // Pre-versioned single-simulator coverage.
    if (!engineVersions.contains(engine)) return false;

    const auto found = environmentKeys.find(engine);
    return found == environmentKeys.end()
        || std::find(found->second.begin(), found->second.end(), key.toStdString()) != found->second.end();
}

bool ModelBenchmarkLibraryFilter::Matches(const ModelBenchmarkResult& result) const
{
    const std::set<int> versions = ResolveModelBenchmarkSimulatorVersions(result.rocketSimVersions, result.engineVersion);

    // Mixed runs require every contributing simulator and its environment to be selected.
    for (int engine : versions)
        if (!MatchesEnvironmentSelection(engine, QString::fromStdString(AggregationEnvironmentKey(result)))) return false;

    if (!evaluationIdentity.isEmpty() && result.evaluationIdentity != evaluationIdentity.toStdString()) return false;
    if (!environmentKey.isEmpty() && AggregationEnvironmentKey(result) != environmentKey.toStdString()) return false;
    if (engineVersion != 0 && versions != std::set<int>{ engineVersion == -1 ? 2 : engineVersion })
        return false;
    if (stochasticKickoff >= 0 && (result.kickoffInferenceSettingsMixed
        || result.stochasticKickoff != (stochasticKickoff != 0)
        || (result.stochasticKickoff && std::abs(result.kickoffTemperature - kickoffTemperature) > 0.0001f)))
        return false;

    return true;
}

QString ModelBenchmarkPairEnvironmentKey(const QString& identity, const QString& checkpointA, const QString& checkpointB)
{
    const QJsonDocument document = QJsonDocument::fromJson(identity.toUtf8());
    if (!document.isObject()) return {};

    QJsonObject environment = document.object();
    if (!environment.value("savedPredictionProfiles").isObject() || !environment.value("externalPrediction").isObject()) return {};

    const QJsonObject profiles = environment.value("savedPredictionProfiles").toObject();
    const QJsonObject predictors = environment.value("externalPrediction").toObject();
    QJsonArray pairProfiles, pairPredictors;
    for (const QString& checkpoint : { checkpointA, checkpointB }) {
        // Library model IDs verify the snapshots.
        // Paths only locate their settings in the historical roster; they are not part of the comparison itself.
        auto normalize = [](QString path) { return QDir::cleanPath(path.replace('\\', '/')); };
        QString key;
        bool found = false;
        for (auto it = profiles.begin(); it != profiles.end(); ++it) {
            if (normalize(it.key()) != normalize(checkpoint)) continue;
            if (found) return {}; // Ambiguous historical metadata cannot establish compatibility.
            key = it.key();
            found = true;
        }

        if (!found || (!profiles.value(key).isString() && !profiles.value(key).isObject())) return {};

        pairProfiles.append(profiles.value(key));
        pairPredictors.append(predictors.value(key).isUndefined() ? QJsonValue(QJsonValue::Null) : predictors.value(key));
    }

    // Preserve exact shared provenance for inspection.
    // Scheduling uses the user-selected simulators and environments; legacy markers are provenance only.
    environment["savedPredictionProfiles"] = pairProfiles;
    environment["externalPrediction"] = pairPredictors;

    return QString::fromUtf8(QJsonDocument(environment).toJson(QJsonDocument::Compact));
}

bool ModelBenchmarkLibrary::EnsureCoverageIndex(const std::filesystem::path& path)
{
    const QString name = "CoverageIndex-" + QUuid::createUuid().toString(QUuid::WithoutBraces);
    bool ready = false;
    {
        QSqlDatabase db = QSqlDatabase::addDatabase("QSQLITE", name);
        db.setDatabaseName(QString::fromStdString(path.string()));
        db.setConnectOptions("QSQLITE_BUSY_TIMEOUT=1000");
        if (std::filesystem::exists(path) && db.open()) {
            QSqlQuery query(db);

            // An additive index needs no schema/benchmark compatibility change.
            // Read-only libraries and concurrent writers can use the faster query without this index.
            ready = query.exec("CREATE INDEX IF NOT EXISTS idx_games_coverage_unsigned "
                "ON games(run_id,model_a,model_b,mode,snapshot_id,mirror_completed)");
        }
        db.close();
    }

    QSqlDatabase::removeDatabase(name);

    return ready;
}

std::vector<ModelBenchmarkLibraryCoverage> ModelBenchmarkLibrary::LoadCoverage(
    const ModelBenchmarkLibraryFilter& filter, QString* error) const
{
    if (error) error->clear();

    std::vector<ModelBenchmarkLibraryCoverage> coverage;
    if (!IsOpen()) {
        if (error) *error = "Open a benchmark library first.";
        return coverage;
    }

    QSqlQuery query(m_impl->Database());
    if (!query.exec(
        // Aggregate narrow game rows before attaching run JSON and model paths.
        // Joining those wide rows first multiplies lookup and temporary-sort work by every game.
        "WITH all_games AS (SELECT run_id,model_a,model_b,mode,snapshot_id,mirror_completed FROM games "
        "UNION ALL SELECT r.run_id,json_extract(j.value,'$.modelA'),json_extract(j.value,'$.modelB'),"
        "json_extract(j.value,'$.mode'),json_extract(j.value,'$.snapshotId'),"
        "COALESCE(json_extract(j.value,'$.mirrorCompleted'),1) FROM runs r,json_each(r.result_json,'$.games') j "
        "WHERE NOT EXISTS(SELECT 1 FROM games g WHERE g.run_id=r.run_id)),"
        "game_counts AS (SELECT run_id,model_a,model_b,mode,SUM(mirror_completed=1) AS n,"
        "MAX(printf('%020s',snapshot_id)) AS latest FROM all_games "
        "GROUP BY run_id,model_a,model_b,mode) "
        "SELECT r.run_id,r.result_json,a.model_id,b.model_id,g.mode,SUM(g.n),MAX(g.latest),ma.checkpoint_path,mb.checkpoint_path "
        "FROM game_counts g JOIN runs r ON r.run_id=g.run_id "
        "JOIN run_models a ON a.run_id=g.run_id AND a.model_index=g.model_a "
        "JOIN run_models b ON b.run_id=g.run_id AND b.model_index=g.model_b "
        "JOIN models ma ON ma.id=a.model_id JOIN models mb ON mb.id=b.model_id "
        "GROUP BY r.run_id,a.model_id,b.model_id,g.mode")) {
        if (error) *error = SqlError(query, "Unable to read matchup coverage");
        return coverage;
    }

    std::map<QString, bool> acceptedRuns;
    std::map<QString, std::string> environmentsByRun;

    // Preserve each environment's coverage rather than dropping a mixed pair.
    using CoverageKey = std::tuple<qint64, qint64, int, std::string>;
    std::map<CoverageKey, qint64> counts;
    std::map<CoverageKey, uint64_t> lastSnapshots;
    std::map<CoverageKey, QString> pairIdentities;
    std::map<QString, std::pair<int, int>> compatibilityByRun;
    std::map<CoverageKey, std::pair<int, int>> compatibility;
    while (query.next()) {
        const QString runId = query.value(0).toString();
        if (!filter.runId.isEmpty() && filter.runId != runId) continue;
        const int mode = query.value(4).toInt();
        if (mode < 1 || mode > 3 || !(filter.modeMask & (1 << (mode - 1)))) continue;

        if (!acceptedRuns.contains(runId)) {
            ModelBenchmarkResult result;
            QString decodeError;
            if (!DecodeResult(query.value(1).toByteArray(), result, &decodeError)) {
                if (error) *error = "Unable to inspect saved run " + runId + ": " + decodeError;
                return {};
            }
            acceptedRuns[runId] = filter.Matches(result);
            environmentsByRun[runId] = AggregationEnvironmentKey(result);
            compatibilityByRun[runId] = {
                result.engineVersion,
                RecordedModelBenchmarkCompatibilityVersion(result.evaluationIdentity)
            };
        }

        if (!acceptedRuns.at(runId)) continue;

        const qint64 a = query.value(2).toLongLong(), b = query.value(3).toLongLong();
        const CoverageKey key = std::make_tuple(std::min(a, b), std::max(a, b), mode, environmentsByRun[runId]);
        const std::string& environment = environmentsByRun[runId];
        if (environment.rfind("exact:", 0) == 0)
            pairIdentities[key] = ModelBenchmarkPairEnvironmentKey(QString::fromStdString(environment.substr(6)),
                query.value(a <= b ? 7 : 8).toString(), query.value(a <= b ? 8 : 7).toString());
        compatibility[key] = compatibilityByRun.at(runId);
        counts[key] += query.value(5).toLongLong();
        lastSnapshots[key] = std::max(lastSnapshots[key], (uint64_t)query.value(6).toULongLong());
    }

    for (const auto& [key, games] : counts) {
        const std::string& environment = std::get<3>(key);
        const QString identity = environment.rfind("exact:", 0) == 0 ? QString::fromStdString(environment.substr(6)) : QString();
        coverage.push_back({ std::get<0>(key), std::get<1>(key), std::get<2>(key), games, lastSnapshots[key], identity, pairIdentities[key],
            compatibility.at(key).first, compatibility.at(key).second, QString::fromStdString(environment) });
    }

    return coverage;
}

std::map<std::string, ModelBenchmarkLibraryEnvironment> ModelBenchmarkLibrary::ListEnvironments(QString* error) const
{
    if (error) error->clear();

    std::map<std::string, ModelBenchmarkLibraryEnvironment> environments;
    if (!IsOpen()) {
        if (error) *error = "Open a benchmark library first.";
        return environments;
    }

    QSqlQuery query(m_impl->Database());
    if (!query.exec("SELECT result_json FROM runs")) {
        if (error) *error = SqlError(query, "Unable to list saved environments");
        return {};
    }

    while (query.next()) {
        ModelBenchmarkLibraryEnvironment environment;
        if (!DecodeResult(query.value(0).toByteArray(), environment.provenance, error)) return {};

        const std::string key = AggregationEnvironmentKey(environment.provenance);
        environments.try_emplace(key, std::move(environment));
    }

    return environments;
}

std::vector<ModelBenchmarkWorkspaceRecord> ModelBenchmarkLibrary::ListPresets(QString* error) const
{
    if (error) error->clear();

    std::vector<ModelBenchmarkWorkspaceRecord> records;
    if (!IsOpen()) return records;

    QSqlQuery query(m_impl->Database());
    if (!query.exec("SELECT name,settings_json FROM workspace_presets ORDER BY name COLLATE NOCASE")) {
        if (error) *error = SqlError(query, "Unable to list benchmark presets");
        return records;
    }

    while (query.next()) records.push_back({ query.value(0).toString(), query.value(0).toString(), query.value(1).toString() });

    return records;
}

std::vector<ModelBenchmarkWorkspaceRecord> ModelBenchmarkLibrary::ListRuns(QString* error) const
{
    if (error) error->clear();

    std::vector<ModelBenchmarkWorkspaceRecord> records;
    if (!IsOpen()) return records;

    QSqlQuery query(m_impl->Database());
    if (!query.exec(
        "SELECT r.run_id,r.created_at,s.settings_json,"
        "(SELECT GROUP_CONCAT(m.display_name, ', ') FROM run_models rm JOIN models m ON m.id=rm.model_id WHERE rm.run_id=r.run_id) "
        "FROM runs r LEFT JOIN run_setups s ON s.run_id=r.run_id ORDER BY r.created_at DESC")) {
        if (error) *error = SqlError(query, "Unable to list benchmark runs");
        return records;
    }

    while (query.next()) {
        const QString label = query.value(1).toString() + "  —  " + query.value(3).toString();
        records.push_back({ query.value(0).toString(), label, query.value(2).toString() });
    }

    return records;
}

bool ModelBenchmarkLibrary::RenamePreset(const QString& oldName, const QString& newName, QString* error)
{
    if (error) error->clear();

    const QString name = newName.trimmed();
    if (!IsOpen() || oldName.isEmpty() || name.isEmpty()) {
        if (error) *error = "Select a saved preset and enter a nonempty name.";
        return false;
    }

    QSqlQuery query(m_impl->Database());
    query.prepare("SELECT settings_json FROM workspace_presets WHERE name=?");
    query.addBindValue(oldName);
    if (!query.exec() || !query.next()) {
        if (error) *error = "The selected preset could not be found.";
        return false;
    }

    const QString original = query.value(0).toString();
    QJsonDocument document = QJsonDocument::fromJson(original.toUtf8());
    if (!document.isObject()) {
        if (error) *error = "The saved preset configuration could not be read.";
        return false;
    }

    QJsonObject object = document.object();
    object["presetName"] = name;
    document.setObject(object);

    // Update in place; the unique name constraint rejects collisions without replacing a preset.
    query.prepare("UPDATE workspace_presets SET name=?,settings_json=? WHERE name=? AND settings_json=?");
    query.addBindValue(name);
    query.addBindValue(QString::fromUtf8(document.toJson(QJsonDocument::Compact)));
    query.addBindValue(oldName);
    query.addBindValue(original);
    if (!query.exec()) {
        if (error) *error = SqlError(query, "Unable to rename preset (the name may already be in use)");
        return false;
    }

    if (query.numRowsAffected() != 1) {
        if (error) *error = "The preset changed while renaming. Please try again.";
        return false;
    }

    return true;
}

bool ModelBenchmarkLibrary::CreatePreset(const QString& name, const QString& settingsJson, QString* error)
{
    if (error) error->clear();
    if (!IsOpen() || name.trimmed().isEmpty() || settingsJson.isEmpty()) {
        if (error) *error = "A benchmark preset needs a name, configuration, and open library.";
        return false;
    }

    QSqlQuery query(m_impl->Database());
    query.prepare("INSERT INTO workspace_presets(name,settings_json) VALUES(?,?)");
    query.addBindValue(name.trimmed());
    query.addBindValue(settingsJson);
    if (!query.exec()) {
        if (error) *error = SqlError(query, "Unable to create preset (the name may already be in use)");
        return false;
    }

    return true;
}

bool ModelBenchmarkLibrary::SavePreset(const QString& name, const QString& settingsJson, QString* error)
{
    if (error) error->clear();
    if (!IsOpen() || name.trimmed().isEmpty() || settingsJson.isEmpty()) {
        if (error) *error = "A benchmark preset needs a name, configuration, and open library.";
        return false;
    }

    QSqlQuery query(m_impl->Database());
    query.prepare("INSERT INTO workspace_presets(name,settings_json) VALUES(?,?) "
        "ON CONFLICT(name) DO UPDATE SET settings_json=excluded.settings_json");
    query.addBindValue(name.trimmed());
    query.addBindValue(settingsJson);
    if (!query.exec()) {
        if (error) *error = SqlError(query, "Unable to save benchmark preset");
        return false;
    }

    return true;
}

QString ModelBenchmarkLibrary::FingerprintCheckpoint(
    const std::filesystem::path& checkpointPath,
    const std::filesystem::path& trainingSetupPath, bool allowCached)
{
    return QString::fromStdString(RocketForgeTraining::FingerprintCheckpoint(checkpointPath, trainingSetupPath, allowCached));
}

ModelBenchmarkLibraryImport ModelBenchmarkLibrary::ReadExcelExport(
    const std::filesystem::path& path,
    QString* error)
{
    if (error)
        error->clear();
    ModelBenchmarkLibraryImport imported;
    QFile file(QString::fromStdString(path.string()));
    if (!file.open(QIODevice::ReadOnly)) {
        if (error)
            *error = "Unable to open Excel XML export.";
        return imported;
    }
    QCryptographicHash fileHash(QCryptographicHash::Sha256);
    fileHash.addData(&file);
    imported.runId = "excel-" + QString::fromLatin1(fileHash.result().toHex());
    file.seek(0);

    QXmlStreamReader xml(&file);
    QString worksheet;
    QStringList headers;
    QStringList row;
    bool firstRow = true;
    bool sawRawStatistics = false;
    std::map<QString, QString> summary;
    auto processRow = [&]() {
        if (firstRow) {
            headers = row;
            firstRow = false;
            return;
        }
        if (worksheet == "Summary") {
            if (row.size() >= 2)
                summary[row[0]] = row[1];
            return;
        }
        if (worksheet == "Models") {
            const int index = Cell(row, HeaderColumn(headers, "Model Index")).toInt();
            if (index < 0)
                return;
            if ((int)imported.models.size() <= index)
                imported.models.resize(index + 1);
            auto& model = imported.models[index];
            model.displayName = Cell(row, HeaderColumn(headers, "Display Name"));
            model.versionId = Cell(row, HeaderColumn(headers, "Version ID"));
            const int primaryColumn = HeaderColumn(headers, "Primary");
            const QString primaryValue = Cell(row, primaryColumn);
            model.primary = primaryColumn < 0 || primaryValue.isEmpty() || ParseYes(primaryValue);
            model.checkpointPath = Cell(row, HeaderColumn(headers, "Checkpoint Path"));
            model.agentName = Cell(row, HeaderColumn(headers, "Agent"));
            model.buildName = Cell(row, HeaderColumn(headers, "Build"));
            model.stageName = Cell(row, HeaderColumn(headers, "Stage"));
            model.branchName = Cell(row, HeaderColumn(headers, "Branch"));
            model.sourceLabel = Cell(row, HeaderColumn(headers, "Source"));
            model.latestTimesteps = Cell(row, HeaderColumn(headers, "Latest Timesteps")).toULongLong();
            model.lastTrainedAt = Cell(row, HeaderColumn(headers, "Last Trained"));
            if (model.agentName.isEmpty())
                model.agentName = "Imported";
            if (model.sourceLabel.isEmpty())
                model.sourceLabel = "Excel";
            model.logicalKey = HeaderColumn(headers, "Agent") >= 0
                ? QStringList{
                    model.agentName,
                    model.buildName,
                    model.stageName,
                    model.versionId,
                    model.sourceLabel
                }.join('|')
                : QStringList{
                    model.agentName,
                    model.displayName,
                    model.versionId,
                    model.checkpointPath
                }.join('|');
            const std::filesystem::path checkpoint(model.checkpointPath.toStdString());
            model.fingerprint = Cell(row, HeaderColumn(headers, "Checkpoint Fingerprint"));
            model.fingerprintVerified = !model.fingerprint.isEmpty();
            if (model.fingerprint.isEmpty()) {
                model.fingerprint = FingerprintCheckpoint(checkpoint);
            }
            if (model.fingerprint.isEmpty()) {
                model.fingerprint = "unverified:" + QString::fromLatin1(
                    QCryptographicHash::hash(model.logicalKey.toUtf8(), QCryptographicHash::Sha256).toHex());
            }
            std::error_code checkpointError;
            model.availableLocally = std::filesystem::exists(checkpoint, checkpointError) && !checkpointError;
            return;
        }
        if (worksheet == "Raw Statistics") {
            ModelBenchmarkStatTotalsRow totalsRow;
            totalsRow.modelIndex = Cell(row, HeaderColumn(headers, "Model Index")).toInt();
            totalsRow.opponentIndex = Cell(row, HeaderColumn(headers, "Opponent Index")).toInt();
            totalsRow.mode = Cell(row, HeaderColumn(headers, "Mode")).toInt();
            auto number = [&](const QString& name) {
                return Cell(row, HeaderColumn(headers, name)).toDouble();
            };
            auto& s = totalsRow.totals;
            s.definitionVersion = Cell(row, HeaderColumn(headers, "Definition Version")).toInt();
            if (s.definitionVersion <= 0)
                s.definitionVersion = 1;
            s.goals = number("Goals");
            s.shots = number("Shots");
            s.assists = number("Assists");
            s.saves = number("Saves");
            s.goalsConceded = number("Goals Conceded");
            s.passes = number("Passes");
            s.passGameTimeSum = number("Pass Game Time Sum");
            s.ballTouches = number("Ball Touches");
            s.opponentBallTouches = number("Opponent Ball Touches");
            s.flipResets = number("Flip Resets");
            s.bumps = number("Bumps");
            s.demos = number("Demos");
            s.boostSteals = number("Boost Steals");
            s.bigBoostPads = number("Big Boost Pads");
            s.smallBoostPads = number("Small Boost Pads");
            s.boostConsumed = number("Boost Consumed");
            s.gameTimeSum = number("Game Time Sum");
            s.boostSum = number("Boost Sum");
            s.speedSum = number("Speed Sum");
            s.aerialTimeSum = number("Aerial Time Sum");
            s.lowAerialTimeSum = number("Low Aerial Time Sum");
            s.highAerialTimeSum = number("High Aerial Time Sum");
            s.lowBoostTimeSum = number("Low Boost Time Sum");
            s.supersonicTimeSum = number("Supersonic Time Sum");
            s.playerTimeSum = number("Player Time Sum");
            s.opponentHalfBallSteps = number("Opponent Half Ball Steps");
            s.offensiveThirdBallSteps = number("Offensive Third Ball Steps");
            s.neutralThirdBallSteps = number("Neutral Third Ball Steps");
            s.defensiveThirdBallSteps = number("Defensive Third Ball Steps");
            s.touchHeightSum = number("Touch Height Sum");
            s.touchCount = number("Touch Count");
            s.speedTowardsBallSum = number("Speed Towards Ball Sum");
            s.distanceToBallSum = number("Distance To Ball Sum");
            s.stepSamples = Cell(row, HeaderColumn(headers, "Step Samples")).toLongLong();
            s.playerSamples = Cell(row, HeaderColumn(headers, "Player Samples")).toLongLong();
            imported.result.statTotals.push_back(totalsRow);
            return;
        }
        if (!worksheet.startsWith("Game Results"))
            return;
        ModelBenchmarkGameMetadata game;
        game.gameId = Cell(row, HeaderColumn(headers, "Game ID")).toULongLong();
        game.snapshotId = Cell(row, HeaderColumn(headers, "Snapshot ID")).toULongLong();
        game.modelA = Cell(row, HeaderColumn(headers, "Model A Index")).toInt();
        game.modelB = Cell(row, HeaderColumn(headers, "Model B Index")).toInt();
        game.blueModel = Cell(row, HeaderColumn(headers, "Blue Index")).toInt();
        game.orangeModel = Cell(row, HeaderColumn(headers, "Orange Index")).toInt();
        QString mode = Cell(row, HeaderColumn(headers, "Mode"));
        const qsizetype versusIndex = mode.indexOf(QChar('v'), 0, Qt::CaseInsensitive);
        game.mode = (versusIndex > 0 ? mode.left(versusIndex) : mode).toInt();
        game.goalTimeSeconds = Cell(row, HeaderColumn(headers, "Goal Time Seconds")).toDouble();
        const QString timing = Cell(row, HeaderColumn(headers, "Goal Timing"));
        game.goalTiming = timing.startsWith("Kickoff", Qt::CaseInsensitive)
            ? ModelBenchmarkGoalTiming::Kickoff
            : (timing.startsWith("Post", Qt::CaseInsensitive)
                ? ModelBenchmarkGoalTiming::PostKickoff
                : ModelBenchmarkGoalTiming::NoGoal);
        game.goalScored = ParseYes(Cell(row, HeaderColumn(headers, "Goal Scored")));
        const QString winner = Cell(row, HeaderColumn(headers, "Winner Index"));
        game.winner = winner == "--" || winner.isEmpty() ? -1 : winner.toInt();
        game.includedInRating = ParseYes(Cell(row, HeaderColumn(headers, "Included In Rating")));
        game.mirrorCompleted = ParseYes(Cell(row, HeaderColumn(headers, "Mirror Completed")));
        imported.result.games.push_back(game);
    };

    while (!xml.atEnd()) {
        xml.readNext();
        if (xml.isStartElement() && xml.name() == QStringLiteral("Worksheet")) {
            worksheet.clear();
            for (const QXmlStreamAttribute& attribute : xml.attributes()) {
                if (attribute.name() == QStringLiteral("Name")) {
                    worksheet = attribute.value().toString();
                    break;
                }
            }
            headers.clear();
            firstRow = true;
            if (worksheet == "Raw Statistics")
                sawRawStatistics = true;
        } else if (xml.isStartElement() && xml.name() == QStringLiteral("Row")) {
            row.clear();
        } else if (xml.isStartElement() && xml.name() == QStringLiteral("Data")) {
            row << xml.readElementText();
        } else if (xml.isEndElement() && xml.name() == QStringLiteral("Row")) {
            processRow();
        }
    }
    if (xml.hasError()) {
        if (error)
            *error = "Unable to parse Excel XML export: " + xml.errorString();
        return {};
    }
    if (imported.models.size() < 2) {
        if (error)
            *error = "The workbook does not contain a valid Models worksheet.";
        return {};
    }
    for (const auto& model : imported.models) {
        imported.result.modelNames.push_back(model.displayName.toStdString());
        imported.result.modelVersionIds.push_back(model.versionId.toStdString());
        imported.result.modelCheckpointPaths.push_back(model.checkpointPath.toStdString());
        if (!model.fingerprintVerified)
            imported.warnings << model.displayName + ": checkpoint identity could not be verified.";
    }

    // Older exports predate simulator selection and therefore used v2, regardless of this build.
    imported.result.rocketSimVersions.clear();
    for (QString version : summary["RocketSim Versions"].split(',', Qt::SkipEmptyParts)) {
        version = version.trimmed();
        if (version.startsWith('v', Qt::CaseInsensitive))
            version.remove(0, 1);
        imported.result.rocketSimVersions.insert(std::max(0, version.toInt()));
    }
    imported.result.rocketSimVersions = ResolveModelBenchmarkSimulatorVersions(
        imported.result.rocketSimVersions, summary["Engine Version"].toInt());
    imported.result.engineVersion = imported.result.rocketSimVersions.size() == 1 ? *imported.result.rocketSimVersions.begin() : -1;
    imported.result.engineFamily = summary["Engine Family"].toStdString();
    imported.result.engineRevision = summary["Engine Revision"].toStdString();
    imported.result.engineCarOrder = summary["Engine Car Order"].toStdString();
    imported.result.integrationRevision = summary["Integration Source Revision"].toStdString();
    imported.result.predictionConfigurationJson = summary["Prediction Configuration"].toStdString();
    imported.result.evaluationIdentity = summary["Evaluation Identity"].toStdString();
    imported.result.seed = summary["Run Seed"].toULongLong();
    const QString exportedRunId = summary["Run ID"].trimmed();
    if (!exportedRunId.isEmpty())
        imported.runId = exportedRunId;
    QString confidence = summary["Elo Range Confidence"];
    confidence.remove('%');
    imported.result.eloRangeConfidence = confidence.isEmpty() ? 0.95f : (float)(confidence.toDouble() / 100.0);
    imported.result.kickoffInferenceSettingsMixed = summary["Stochastic Kickoff"] == "Mixed";
    imported.result.stochasticKickoff = summary["Stochastic Kickoff"] == "Yes";
    imported.result.kickoffTemperature = summary.contains("Kickoff Temperature") ? summary["Kickoff Temperature"].toFloat() : 1.0f;
    imported.result.kickoffGoalWindowSeconds = summary["Kickoff Goal Window Seconds"].toFloat();
    if (imported.result.kickoffGoalWindowSeconds <= 0.0f)
        imported.result.kickoffGoalWindowSeconds = 10.0f;
    const QString rawStatisticsComplete = summary["Raw Statistics Complete"].trimmed();
    imported.result.statTotalsComplete = sawRawStatistics
        && (rawStatisticsComplete.isEmpty() || ParseYes(rawStatisticsComplete));
    if (!imported.result.statTotalsComplete) {
        imported.warnings << "The workbook lacks complete lossless per-matchup statistics; saved Elo and records will be rebuilt from game rows, but optional statistics will be unavailable.";
    }
    return imported;
}

ModelBenchmarkLibraryAggregate CombineModelBenchmarkLiveResults(
    const ModelBenchmarkLibraryAggregate& saved,
    const std::vector<ModelBenchmarkLibrarySelection>& selection,
    const ModelBenchmarkResult& live,
    const std::vector<qint64>& liveModelIds,
    const ModelBenchmarkLibraryFilter& filter)
{
    if (!filter.runId.isEmpty()) return saved;
    if (selection.size() != saved.result.modelNames.size() || liveModelIds.size() != live.modelNames.size())
        throw std::runtime_error("Live comparison model mapping is incomplete.");

    ModelBenchmarkLibraryAggregate combined = saved;

    // Live environments stay discoverable even while a different display filter is active.
    std::map<qint64, int> catalogIndex;
    for (int i = 0; i < (int)selection.size(); ++i) catalogIndex[selection[i].modelId] = i;
    const std::string liveEnvironmentKey = AggregationEnvironmentKey(live);
    for (const ModelBenchmarkGameMetadata& game : live.games) {
        if (game.mode < 1 || game.mode > 3 || !(filter.modeMask & (1 << (game.mode - 1)))
            || game.modelA < 0 || game.modelB < 0
            || game.modelA >= (int)liveModelIds.size() || game.modelB >= (int)liveModelIds.size()) continue;
        const auto a = catalogIndex.find(liveModelIds[game.modelA]), b = catalogIndex.find(liveModelIds[game.modelB]);
        if (a == catalogIndex.end() || b == catalogIndex.end() || a == b
            || (!selection[a->second].primary && !selection[b->second].primary)) continue;

        auto [entry, inserted] = combined.availableEnvironments.try_emplace(liveEnvironmentKey);
        ModelBenchmarkLibraryEnvironment& environment = entry->second;
        if (inserted) {
            CopyEnvironment(live, environment.provenance);
            environment.provenance.rocketSimVersions = live.rocketSimVersions;
        }

        if (game.mirrorCompleted) {
            ++environment.games;
            ++environment.modelGames[a->second];
            ++environment.modelGames[b->second];
        }
    }

    if (!filter.Matches(live)) return combined;
    if (live.games.empty()) {
        if (live.scoredGames + live.noResultGames > 0)
            combined.warnings << "This runner does not provide the game data needed for live combined results. Rebuild its benchmark helper to include completed games in combined results while the run is active.";
        return combined;
    }

    ModelBenchmarkConfig config;
    config.eloRangeConfidence = saved.result.eloRangeConfidence;
    config.collectOptionalStats = true;
    config.kickoffGoalWindowSeconds = live.kickoffGoalWindowSeconds;
    config.stochasticKickoff = live.stochasticKickoff;
    config.kickoffTemperature = live.kickoffTemperature;

    std::map<qint64, int> selectedIndex;
    for (int i = 0; i < (int)selection.size(); ++i) {
        ModelBenchmarkModel model;
        model.displayName = saved.result.modelNames[i];
        if (i < (int)saved.result.modelVersionIds.size()) model.versionId = saved.result.modelVersionIds[i];
        if (i < (int)saved.result.modelCheckpointPaths.size()) model.checkpointPath = saved.result.modelCheckpointPaths[i];
        model.primary = selection[i].primary;
        config.models.push_back(std::move(model));
        selectedIndex[selection[i].modelId] = i;
    }

    auto remap = [&](int index) {
        if (index < 0 || index >= (int)liveModelIds.size()) return -1;
        const auto found = selectedIndex.find(liveModelIds[index]);
        return found == selectedIndex.end() ? -1 : found->second;
    };
    auto acceptedMode = [&](int mode) { return mode >= 1 && mode <= 3 && (filter.modeMask & (1 << (mode - 1))); };
    auto rated = [&](int a, int b) { return config.models[a].primary || config.models[b].primary; };

    std::vector<ModelBenchmarkGameMetadata> games = saved.result.games;
    std::vector<ModelBenchmarkStatTotalsRow> totals = saved.result.statTotals;
    uint64_t gameId = 0, snapshotId = 0;
    for (const ModelBenchmarkGameMetadata& game : games) {
        gameId = std::max(gameId, game.gameId);
        snapshotId = std::max(snapshotId, game.snapshotId);
    }

    std::map<uint64_t, uint64_t> snapshots;
    ModelBenchmarkLibraryEnvironment liveEnvironment;
    CopyEnvironment(live, liveEnvironment.provenance);
    liveEnvironment.provenance.rocketSimVersions = live.rocketSimVersions;
    for (ModelBenchmarkGameMetadata game : live.games) {
        if (!acceptedMode(game.mode)) continue;

        const int a = remap(game.modelA), b = remap(game.modelB);
        if (a < 0 || b < 0 || a == b) continue;

        game.modelA = a;
        game.modelB = b;
        game.blueModel = remap(game.blueModel);
        game.orangeModel = remap(game.orangeModel);
        game.winner = remap(game.winner);
        combined.coverageGames.push_back(game);
        if (!rated(a, b)) continue;

        // Mirrored games retain a shared snapshot ID, disjoint from saved runs.
        auto [position, inserted] = snapshots.try_emplace(game.snapshotId, 0);
        if (inserted) position->second = ++snapshotId;
        game.snapshotId = position->second;
        game.gameId = ++gameId;
        games.push_back(game);
        ++combined.liveGames;
        if (game.mirrorCompleted) {
            ++liveEnvironment.games;
            ++liveEnvironment.modelGames[a];
            ++liveEnvironment.modelGames[b];
        }
    }

    for (const ModelBenchmarkStatTotalsRow& row : live.statTotals) {
        if (!acceptedMode(row.mode)) continue;
        const int a = remap(row.modelIndex), b = remap(row.opponentIndex);
        if (a >= 0 && b >= 0 && a != b && rated(a, b))
            totals.push_back({a, b, row.mode, row.totals});
    }

    if (combined.liveGames == 0) return combined;

    const bool complete = (saved.contributingRuns == 0 || saved.result.statTotalsComplete) && live.statTotalsComplete;
    combined.result = SummarizeModelBenchmarkData(config, games, totals, complete);

    ModelBenchmarkLibraryEnvironment& environment = combined.environments[AggregationEnvironmentKey(live)];
    environment.provenance = liveEnvironment.provenance;
    environment.games += liveEnvironment.games;
    for (const auto& [model, count] : liveEnvironment.modelGames)
        environment.modelGames[model] += count;

    if (ModelBenchmarkHasMixedEnvironments(combined)) {
        combined.result.engineFamily = "Mixed environments";
        const QString warning = "Mixed environments: pooled Elo describes the included historical games. Unequal simulator coverage can affect rankings; choose environments in Test Setup for controlled comparisons.";
        if (!combined.warnings.contains(warning)) combined.warnings << warning;
    } else {
        CopyEnvironment(live, combined.result);
    }

    combined.result.rocketSimVersions = live.rocketSimVersions;
    if (saved.contributingRuns > 0)
        combined.result.rocketSimVersions.insert(saved.result.rocketSimVersions.begin(), saved.result.rocketSimVersions.end());
    combined.result.kickoffInferenceSettingsMixed = live.kickoffInferenceSettingsMixed
        || (saved.contributingRuns > 0 && (saved.result.kickoffInferenceSettingsMixed
            || saved.result.stochasticKickoff != live.stochasticKickoff
            || (live.stochasticKickoff && saved.result.kickoffTemperature != live.kickoffTemperature)));

    combined.result.stopReason = "Saved results plus unsaved current run";
    combined.result.timingSummary = "Includes completed games from the current run; these are not saved until the run ends.";
    combined.warnings.removeIf([](const QString& warning) {
        return warning.endsWith(": no eligible saved games match the current Primary selection.");
    });

    std::set<int> modelsWithGames;
    for (const ModelBenchmarkGameMetadata& game : combined.result.games) {
        modelsWithGames.insert(game.modelA);
        modelsWithGames.insert(game.modelB);
    }

    for (int i = 0; i < (int)config.models.size(); ++i) {
        if (!modelsWithGames.contains(i))
            combined.warnings << QString::fromStdString(config.models[i].displayName)
                + ": no eligible saved or live games match the current Primary selection.";
    }

    if (!complete) {
        combined.result.optionalStats.clear();
        combined.result.optionalStatsByMode.clear();
        const QString warning = "Optional statistics are unavailable because at least one contributing run lacks lossless per-matchup totals.";
        if (!combined.warnings.contains(warning)) combined.warnings << warning;
    }

    return combined;
}

QString ModelBenchmarkEnvironmentLabel(const ModelBenchmarkLibraryEnvironment& environment)
{
    const ModelBenchmarkResult& source = environment.provenance;
    const QString simulator = QString::fromStdString(ModelBenchmarkRocketSimVersionsText(source.rocketSimVersions));
    if (source.evaluationIdentity.empty()) return simulator + " / unrecorded build and settings";

    // Only legacy records carry an integration build; newer runs identify themselves by settings alone.
    const QString build = source.integrationRevision.empty() ? QString()
        : " / build " + QString::fromStdString(source.integrationRevision).left(12);

    return simulator + build
        + " / settings " + QString::fromLatin1(QCryptographicHash::hash(
            QByteArray::fromStdString(source.evaluationIdentity), QCryptographicHash::Sha256).toHex().left(8));
}

bool ModelBenchmarkHasMixedEnvironments(const ModelBenchmarkLibraryAggregate& aggregate)
{
    if (aggregate.environments.size() > 1) return true;
    for (const auto& [key, environment] : aggregate.environments)
        if (environment.provenance.rocketSimVersions.size() > 1
            || environment.provenance.engineFamily == "Mixed environments") return true;

    return false;
}

void WriteModelBenchmarkEnvironmentWorksheet(SpreadsheetXmlWriter& writer, const ModelBenchmarkLibraryAggregate& aggregate)
{
    writer.OpenWorksheet("Environment Coverage");

    auto row = [&](const QStringList& cells) {
        std::vector<std::string> values;
        for (const QString& cell : cells)
            values.push_back(cell.toStdString());
        writer.Row(values);
    };
    row({"Environment", "Simulator", "Games", "Model", "Model Games", "Physics Revision",
        "Integration Source Revision", "Prediction Configuration", "Evaluation Identity"});

    for (const auto& [key, environment] : aggregate.environments) {
        const ModelBenchmarkResult& source = environment.provenance;
        auto write = [&](const QString& model, qint64 count) {
            row({ModelBenchmarkEnvironmentLabel(environment),
                QString::fromStdString(ModelBenchmarkRocketSimVersionsText(source.rocketSimVersions)),
                QString::number(environment.games), model, QString::number(count),
                QString::fromStdString(source.engineRevision), QString::fromStdString(source.integrationRevision),
                QString::fromStdString(source.predictionConfigurationJson), QString::fromStdString(source.evaluationIdentity)});
        };

        // Emit zeroes too: absent simulator coverage is important when judging a pooled ranking.
        for (int model = 0; model < (int)aggregate.result.modelNames.size(); ++model) {
            const auto found = environment.modelGames.find(model);
            write(QString::fromStdString(aggregate.result.modelNames[model]), found == environment.modelGames.end() ? 0 : found->second);
        }
    }

    writer.CloseWorksheet();
}
