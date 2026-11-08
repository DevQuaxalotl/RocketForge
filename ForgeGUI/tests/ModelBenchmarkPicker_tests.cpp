#include "test_runner.h"
#include <ModelBenchmark/ModelBenchmarkEngine.h>
#include "ModelBenchmark/ModelBenchmarkWidget.h"
#include "ModelBenchmark/ModelBenchmarkWorker.h"
#include <ModelBenchmark/ModelBenchmarkCompatibility.h>
#include "ModelBenchmark/ModelBenchmarkResultTables.h"
#include "ModelBenchmark/ModelBenchmarkSavedResultsWidget.h"
#include "Widgets/AgentComboBox.h"

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QElapsedTimer>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QMouseEvent>
#include <QPushButton>
#include <QMenu>
#include <QAction>
#include <QSignalBlocker>
#include <QTableWidget>
#include <QTabWidget>
#include <QTimer>
#include <QThread>
#include <QDialog>
#include <QDir>
#include <QFile>
#include <QLabel>
#include <QHeaderView>
#include <QInputDialog>
#include <QMessageBox>
#include <QListWidget>
#include <QPlainTextEdit>
#include <QDialogButtonBox>
#include <QScreen>
#include <QSpinBox>
#include <QTemporaryDir>

#include <iostream>
#include <GigaLearnInfer/AgentRegistry.h>
#include <TrainingSetup/TrainingSetupSnapshot.h>
#include <fstream>
#include <QJsonDocument>
#include <QJsonObject>

namespace {
    constexpr bool PERSIST_SETUP = false;
    constexpr bool SHOW_ERRORS = false;
    constexpr bool SHOW_PREVIEW = false;
    constexpr bool REQUIRE_RUNNABLE = true;
}

class ModelBenchmarkPickerTests {
public:
    static void PublicModelHistory() {
        using namespace RocketForgeTraining;
        AgentRegistry::Register("HistoryFixture", [] { return std::make_unique<Agent>(); });
        QTemporaryDir directory;
        ASSERT_TRUE(directory.isValid());
        ModelBenchmarkWidget widget(nullptr, PERSIST_SETUP);
        widget.m_loadingPersistedSetup = true;
        ASSERT_TRUE(widget.m_savedResultsWidget->OpenLibrary(directory.filePath("history.sqlite").toStdString(), SHOW_ERRORS));
        widget.m_engineCombo->setCurrentIndex(widget.m_engineCombo->findData(static_cast<int>(RocketSim::EngineId::V2)));
        widget.m_maxGamesSpin->setValue(1);
        widget.m_mode1v1Check->setChecked(true);
        widget.m_mode2v2Check->setChecked(true);
        widget.m_mode3v3Check->setChecked(true);

        for (int i = 0; i < 5; ++i) {
            ModelBenchmarkWidget::BenchmarkVersionEntry entry;
            entry.agentName = "HistoryFixture";
            entry.buildName = i < 3 ? "Native" : "RLBotPack";
            entry.sourceLabel = i < 3 ? "RocketForge" : "Imported Model";
            entry.agentSetupSource = ModelBenchmarkAgentSetupSource::CurrentAgent;
            entry.version.id = std::to_string(i);
            entry.primary = i < 3;
            entry.checkpointPath = directory.filePath(QString::number(i)).toStdString();
            ASSERT_TRUE(QDir().mkpath(QString::fromStdString(entry.checkpointPath.string())));
            QFile policy(QString::fromStdString((entry.checkpointPath / (i < 3 ? "POLICY.lt" : "model.pt")).string()));
            ASSERT_TRUE(policy.open(QIODevice::WriteOnly));
            policy.write("fixture policy");
            widget.m_selectedVersions.push_back(entry);
        }

        const std::vector<ModelBenchmarkLibraryModel> models = widget.BuildLibraryModels(widget.m_selectedVersions);
        for (int i = 0; i < 5; ++i) {
            ASSERT_EQ(models[i].fingerprintVerified, i < 3);
            if (i >= 3)
                ASSERT_EQ(models[i].fingerprint, "unverified:" + models[i].logicalKey + "|" + models[i].checkpointPath);
        }

        // Reproduce three targets, two public opponents, and all three team sizes.
        ModelBenchmarkResult result;
        result.rocketSimVersions = {static_cast<int>(RocketSim::EngineId::V2)};
        result.stopReason = "Max games per mode reached";
        for (const ModelBenchmarkLibraryModel& model : models)
            result.modelNames.push_back(model.displayName.toStdString());

        for (int a = 0; a < 3; ++a) {
            for (int b = a + 1; b < 5; ++b) {
                for (int mode : {1, 2, 3}) {
                    ModelBenchmarkGameMetadata game;
                    game.modelA = game.blueModel = a;
                    game.modelB = game.orangeModel = b;
                    game.winner = a;
                    game.mode = mode;
                    game.gameId = game.snapshotId = result.games.size() + 1;
                    game.goalTiming = ModelBenchmarkGoalTiming::PostKickoff;
                    game.goalTimeSeconds = 20.0;
                    game.goalScored = game.includedInRating = game.mirrorCompleted = true;
                    result.games.push_back(game);
                }
            }
        }

        result.scoredGames = result.maxGames = result.games.size();
        widget.m_activeRunVersions = widget.m_selectedVersions;
        widget.m_activeRunLibraryModels = models;
        widget.SetRunning(true);
        widget.OnBenchmarkComplete(result);
        ASSERT_TRUE(widget.m_lastResultSavedToActiveLibrary);
        ASSERT_FALSE(widget.IsRunning());
        for (int i = 0; i < 5; ++i) {
            auto* role = qobject_cast<QComboBox*>(widget.m_selectedTable->cellWidget(i, 1));
            ASSERT_TRUE(role && role->isEnabled());
            ASSERT_EQ(role->currentIndex(), i < 3 ? 0 : 1);
        }

        widget.m_loadingPersistedSetup = false;
        auto refresh = [&]() {
            widget.m_workspaceRefreshTimer->stop();
            widget.RefreshWorkspace();

            QElapsedTimer timeout;
            timeout.start();
            auto* tabs = widget.m_savedResultsWidget->findChild<QTabWidget*>("benchmarkResultTabs");
            auto* ranking = qobject_cast<QTableWidget*>(tabs->widget(0));
            while ((widget.m_workspaceScan.valid() || ranking->rowCount() != 5) && timeout.elapsed() < 10000) {
                QApplication::processEvents();
                QThread::msleep(5);
            }

            ASSERT_FALSE(widget.m_workspaceScan.valid());
            ASSERT_TRUE(widget.m_coverageError.isEmpty());
            if (ranking->rowCount() != 5) {
                std::cerr << widget.m_scheduleSummaryLabel->text().toStdString() << '\n';
                for (int column = 0; column < ranking->columnCount(); ++column) {
                    if (QTableWidgetItem* item = ranking->item(0, column))
                        std::cerr << item->text().toStdString() << '\n';
                }
            }

            ASSERT_EQ(ranking->rowCount(), 5);
            std::vector<ModelBenchmarkLibrarySelection> selection;
            for (int i = 0; i < 5; ++i) {
                const ModelBenchmarkWidget::BenchmarkVersionEntry& entry = widget.m_selectedVersions[i];
                ASSERT_TRUE(entry.libraryModelId >= 0);
                ASSERT_EQ(entry.fingerprint, models[i].fingerprint);
                selection.push_back({entry.libraryModelId, true});
            }

            ASSERT_TRUE(widget.PrepareSchedule(SHOW_PREVIEW).pairBudgets.empty());
            ASSERT_TRUE(widget.m_scheduleSummaryLabel->text().startsWith("0 new games"));
            const ModelBenchmarkLibraryAggregate aggregate = widget.m_savedResultsWidget->Library().LoadAggregate(
                selection, 0.95f, nullptr, {}, widget.WorkspaceFilter());
            ASSERT_EQ(aggregate.result.games.size(), size_t(27));
        };
        refresh();

        // Older presets stored empty fingerprints and unresolved IDs for public bots.
        ModelBenchmarkSettings restored = widget.BuildPersistedSetup();
        for (int i = 3; i < 5; ++i) {
            restored.selectedModels[i].fingerprint.clear();
            restored.selectedModels[i].libraryModelId = -1;
        }

        widget.ApplyPersistedSetup(restored);
        refresh();
        widget.m_loadingPersistedSetup = true;
        widget.m_workspaceRefreshTimer->stop();
        widget.m_setupSaveTimer->stop();
        std::cout << "Public-model completion and legacy preset reload retain all 27 games, five rankings, and zero scheduled games\n";
    }

    static void RoleUnlock() {
        ModelBenchmarkWidget widget(nullptr, PERSIST_SETUP);
        widget.m_loadingPersistedSetup = true;
        for (int role = 0; role < 3; ++role) {
            ModelBenchmarkWidget::BenchmarkVersionEntry entry;
            entry.primary = role == 0;
            entry.resultsOnly = role == 2;
            widget.m_selectedVersions.push_back(entry);
        }

        for (int ending = 0; ending < 3; ++ending) {
            widget.SetRunning(true);
            widget.PopulateSelectedTable();
            for (int row = 0; row < 3; ++row)
                ASSERT_FALSE(widget.m_selectedTable->cellWidget(row, 1)->isEnabled());

            if (ending == 0) {
                widget.OnBenchmarkComplete({});
            } else if (ending == 1) {
                widget.OnBenchmarkCancelled();
            } else {
                QTimer::singleShot(0, &widget, []() {
                    if (auto* prompt = qobject_cast<QMessageBox*>(QApplication::activeModalWidget()))
                        prompt->accept();
                });
                widget.OnBenchmarkFailed("Synthetic failure", false);
            }

            for (int row = 0; row < 3; ++row) {
                auto* role = qobject_cast<QComboBox*>(widget.m_selectedTable->cellWidget(row, 1));
                ASSERT_TRUE(role && role->isEnabled());
                ASSERT_EQ(role->currentIndex(), row);
            }
        }

        auto* role = qobject_cast<QComboBox*>(widget.m_selectedTable->cellWidget(1, 1));
        role->setCurrentIndex(0);
        ASSERT_TRUE(widget.m_selectedVersions[1].primary);
        std::cout << "Next-run roles unlock after completion, cancellation, and failure without changing selections\n";
    }

    static void WorkerCoalescesCompleteLiveSnapshots() {
        using namespace RocketForgeTraining;
        ModelBenchmarkWorker worker;
        int notifications = 0;
        QObject::connect(&worker, &ModelBenchmarkWorker::ProgressUpdated, [&]() { ++notifications; });
        for (int i = 0; i < 3; ++i) {
            std::shared_ptr<ModelBenchmarkResult> result = std::make_shared<ModelBenchmarkResult>();
            ModelBenchmarkGameMetadata game;
            game.gameId = i + 1;
            game.mirrorCompleted = true;
            result->games.push_back(game);

            ModelBenchmarkProgress progress;
            progress.liveGamesOffset = i;
            progress.liveResult = result;
            worker.PublishProgress(progress);
        }

        worker.PublishProgress(ModelBenchmarkProgress{});
        ASSERT_EQ(notifications, 1);

        std::optional<ModelBenchmarkProgress> latest = worker.TakeLatestProgress();
        ASSERT_TRUE(latest && latest->liveResult);
        ASSERT_EQ(latest->liveResult->games.size(), size_t(3));
        ASSERT_EQ(latest->liveResult->games.back().gameId, uint64_t(3));
        ASSERT_FALSE(worker.TakeLatestProgress().has_value());

        worker.PublishProgress(ModelBenchmarkProgress{});
        ASSERT_EQ(notifications, 2);
    }

    static void StoppedRunsPreserveGames() {
        using namespace RocketForgeTraining;
        AgentRegistry::Register("StoppedFixture", [] { return std::make_unique<Agent>(); });

        // Exercise normal cancellation and both missing-final-result outcomes through the real autosave path.
        for (int scenario = 0; scenario < 3; ++scenario) {
            QTemporaryDir directory;
            ASSERT_TRUE(directory.isValid());
            ModelBenchmarkWidget widget(nullptr, PERSIST_SETUP);
            widget.m_loadingPersistedSetup = true;
            const std::string libraryPath = directory.filePath("stopped.sqlite").toStdString();
            ASSERT_TRUE(widget.m_savedResultsWidget->OpenLibrary(libraryPath, SHOW_ERRORS));
            widget.m_engineCombo->setCurrentIndex(widget.m_engineCombo->findData(static_cast<int>(RocketSim::EngineId::V2)));
            widget.m_maxGamesSpin->setValue(4);
            widget.m_mode1v1Check->setChecked(true);
            widget.m_mode2v2Check->setChecked(false);
            widget.m_mode3v3Check->setChecked(false);

            for (int i = 0; i < 2; ++i) {
                ModelBenchmarkWidget::BenchmarkVersionEntry entry;
                entry.agentName = "StoppedFixture";
                entry.buildName = "Fixture";
                entry.sourceLabel = "Imported Model";
                entry.version.id = std::to_string(i);
                entry.agentSetupSource = ModelBenchmarkAgentSetupSource::CurrentAgent;
                entry.checkpointPath = directory.filePath(QString("Models/Imported Models/%1").arg(i)).toStdString();
                std::filesystem::create_directories(entry.checkpointPath);
                std::ofstream(entry.checkpointPath / "INFERENCE_POLICY.json") << R"({"artifact":"policy.pt"})";
                std::ofstream(entry.checkpointPath / "policy.pt") << "fixture weights " << i;
                widget.m_selectedVersions.push_back(entry);
            }

            widget.m_activeRunVersions = widget.m_selectedVersions;
            widget.m_activeRunLibraryModels = widget.BuildLibraryModels(widget.m_activeRunVersions);
            for (const ModelBenchmarkLibraryModel& model : widget.m_activeRunLibraryModels)
                ASSERT_TRUE(model.fingerprintVerified);

            ModelBenchmarkResult result;
            result.engineVersion = static_cast<int>(RocketSim::EngineId::V2);
            result.rocketSimVersions = {static_cast<int>(RocketSim::EngineId::V2)};
            result.modelNames = {"First", "Second"};
            result.modelVersionIds = {"0", "1"};
            result.seed = 12345;
            result.maxGames = 4;
            result.stopReason = "Cancelled";
            for (int i = 0; i < 3; ++i) {
                ModelBenchmarkGameMetadata game;
                game.gameId = i + 1;
                game.snapshotId = i / 2 + 1;
                game.modelA = 0;
                game.modelB = 1;
                game.mode = 1;
                game.blueModel = i % 2;
                game.orangeModel = 1 - i % 2;
                game.winner = i % 2;
                game.goalScored = true;
                game.goalTimeSeconds = 20;
                game.goalTiming = ModelBenchmarkGoalTiming::PostKickoff;
                game.mirrorCompleted = i < 2;
                game.includedInRating = i < 2;
                result.games.push_back(game);
            }

            int completed = 0;
            int cancelled = 0;
            int failed = 0;
            ModelBenchmarkWorker worker;
            QObject::connect(&worker, &ModelBenchmarkWorker::Completed, [&](const ModelBenchmarkResult& saved) {
                ++completed;
                ASSERT_EQ(saved.seed, uint64_t(12345));
                ASSERT_EQ(saved.games.size(), size_t(2));
                ASSERT_EQ(saved.stopReason, scenario == 1 ? std::string("Cancelled") : std::string("Terminated early: transport failed"));
                widget.OnBenchmarkComplete(saved);
            });
            QObject::connect(&worker, &ModelBenchmarkWorker::Cancelled, [&]() { ++cancelled; });
            QObject::connect(&worker, &ModelBenchmarkWorker::Failed, [&]() { ++failed; });

            if (scenario == 0) {
                widget.OnBenchmarkComplete(result);
            } else {
                for (int i = 0; i < 2; ++i) {
                    std::shared_ptr<ModelBenchmarkResult> delta = std::make_shared<ModelBenchmarkResult>(result);
                    delta->games = {result.games[i]};
                    ModelBenchmarkProgress progress;
                    progress.liveGamesOffset = i;
                    progress.liveResult = delta;
                    worker.PublishProgress(progress);
                    ASSERT_TRUE(worker.TakeLatestProgress().has_value());
                }

                worker.PublishProgress(ModelBenchmarkProgress{});
                worker.TakeLatestProgress();
                worker.FinishWithoutResult(scenario == 1 ? "Cancelled" : "transport failed", scenario == 2);
                ASSERT_EQ(completed, 1);
                ASSERT_EQ(cancelled, 0);
                ASSERT_EQ(failed, 0);
            }

            ASSERT_TRUE(widget.m_lastResultSavedToActiveLibrary);
            ASSERT_EQ(widget.m_lastResult.games.size(), size_t(scenario == 0 ? 3 : 2));
            ASSERT_TRUE(widget.m_consoleText->toPlainText().contains("2 count toward results and future runs"));

            // Reopen the library and reconstruct the selection as it appears after restarting the GUI.
            for (ModelBenchmarkWidget::BenchmarkVersionEntry& entry : widget.m_selectedVersions) {
                entry.fingerprint.clear();
                entry.libraryModelId = -1;
            }

            widget.m_workspaceRefreshTimer->stop();
            widget.m_loadingPersistedSetup = false;
            widget.RefreshWorkspace();
            QElapsedTimer timeout;
            timeout.start();
            while (widget.m_workspaceScan.valid() && timeout.elapsed() < 10000) {
                QApplication::processEvents();
                QThread::msleep(5);
            }

            ASSERT_FALSE(widget.m_workspaceScan.valid());
            for (const ModelBenchmarkWidget::BenchmarkVersionEntry& entry : widget.m_selectedVersions)
                ASSERT_TRUE(entry.libraryModelId > 0);

            ModelBenchmarkLibrary reopened;
            QString error;
            ASSERT_TRUE(reopened.OpenReadOnly(libraryPath, &error));
            const std::vector<ModelBenchmarkLibraryModel> models = reopened.ListModels(&error);
            ASSERT_EQ(models.size(), size_t(2));

            ModelBenchmarkLibraryAggregate aggregate = reopened.LoadAggregate({{models[0].id, true}, {models[1].id, true}}, .95f, &error);
            ASSERT_TRUE(error.isEmpty());
            ASSERT_EQ(aggregate.contributingRuns, 1);
            ASSERT_EQ(aggregate.coverageGames.size(), size_t(scenario == 0 ? 3 : 2));
            ASSERT_EQ(std::count_if(aggregate.coverageGames.begin(), aggregate.coverageGames.end(),
                [](const ModelBenchmarkGameMetadata& game) { return game.mirrorCompleted; }), 2);

            const std::vector<ModelBenchmarkLibraryCoverage> coverage = reopened.LoadCoverage({}, &error);
            ASSERT_EQ(coverage.size(), size_t(1));
            ASSERT_EQ(coverage[0].games, qint64(2));

            const ModelBenchmarkConfig schedule = widget.PrepareSchedule(SHOW_PREVIEW);
            ASSERT_EQ(schedule.pairBudgets.size(), size_t(1));
            ASSERT_EQ(schedule.pairBudgets[0].games, 2);
            ASSERT_TRUE(widget.m_scheduleSummaryLabel->text().contains("2 covered games"));
            widget.m_loadingPersistedSetup = true;
            widget.m_setupSaveTimer->stop();
            widget.m_workspaceRefreshTimer->stop();
        }

        ModelBenchmarkWorker empty;
        int cancelled = 0;
        int failed = 0;
        int completed = 0;
        QObject::connect(&empty, &ModelBenchmarkWorker::Cancelled, [&]() { ++cancelled; });
        QObject::connect(&empty, &ModelBenchmarkWorker::Failed, [&]() { ++failed; });
        QObject::connect(&empty, &ModelBenchmarkWorker::Completed, [&]() { ++completed; });

        const bool reportCancelled = false;
        const bool reportFailed = true;
        empty.FinishWithoutResult("Cancelled", reportCancelled);
        empty.FinishWithoutResult("startup failed", reportFailed);
        ASSERT_EQ(cancelled, 1);
        ASSERT_EQ(failed, 1);
        ASSERT_EQ(completed, 0);
        std::cout << "Stopped runs retain completed games, reconnect imported policies, and resume only missing coverage\n";
    }

    static void CatalogPreservesAutosavesAndSimulation() {
        using namespace RocketForgeTraining;
        namespace fs = std::filesystem;
        QTemporaryDir directory;
        const fs::path modelRoot(directory.path().toStdString());
        AgentRegistry::Register("CatalogPublicFixture", [modelRoot] {
            std::unique_ptr<Agent> agent = std::make_unique<Agent>();
            agent->modelPath = modelRoot;
            return agent;
        });

        const fs::path segmentRoot = modelRoot / "Build" / "segments" / "1.1";
        fs::create_directories(segmentRoot / "checkpoints" / "100");
        fs::create_directories(segmentRoot / "temp_checkpoints" / "200");

        BuildMetadata build;
        build.buildName = "Build";
        build.folderName = "Build";
        ASSERT_TRUE(GGL::JsonSaveToFile(build, modelRoot / "Build" / "build.json"));

        TrainingSegmentMetadata segment;
        segment.segmentId = "1.1";
        segment.folderName = "1.1";
        segment.buildName = "Build";
        ASSERT_TRUE(GGL::JsonSaveToFile(segment, segmentRoot / "segment.json"));

        Agent agent;
        nlohmann::json setup = RocketForge::AgentSetupSnapshotToJson(RocketForge::CaptureAgentSetupSnapshot(agent));
        setup.erase("training_engine_version");
        setup.erase("ball_prediction_engine_version");
        { std::ofstream out(segmentRoot / "training_setup.json");
          out << nlohmann::json{{"engine_version",2},{"agent_setup",setup}}; }

        ModelCatalog& catalog = ModelCatalog::Instance();
        catalog.Get("CatalogPublicFixture");
        QElapsedTimer timeout;
        timeout.start();
        while (catalog.Get("CatalogPublicFixture")->models.empty() && timeout.elapsed() < 10000) {
            QApplication::processEvents();
            QThread::msleep(10);
        }

        const auto snapshot = catalog.Get("CatalogPublicFixture");
        ASSERT_EQ(snapshot->models.size(), size_t(1));
        const CatalogModel& model = snapshot->models.front();
        ASSERT_EQ(model.version.latestTimesteps, uint64_t(200));
        ASSERT_TRUE(model.version.latestIsAutosave);
        ASSERT_EQ(model.version.modelCount, 1);
        ASSERT_EQ(fs::path(model.checkpointPath).filename().string(), std::string("200"));
        std::optional<RocketForge::AgentSetupSnapshot> saved = RocketForge::AgentSetupSnapshotFromJson(model.agentSetup);
        ASSERT_TRUE(saved.has_value()); ASSERT_EQ(saved->trainingEngineVersion, 2); ASSERT_EQ(saved->ballPredictionEngineVersion, 2);
        ModelBenchmarkWidget::VersionCatalog rows = ModelBenchmarkWidget::ScanVersionCatalog("CatalogPublicFixture", snapshot);
        ASSERT_EQ(rows.entries.size(), size_t(1));

        // A segment with only an autosave must still be selectable for new games.
        auto autosaveOnly = std::make_shared<ModelCatalogSnapshot>(*snapshot);
        autosaveOnly->models.front().version.modelCount = 0;
        const auto autosaveRows = ModelBenchmarkWidget::ScanVersionCatalog("CatalogPublicFixture", autosaveOnly);
        ASSERT_EQ(autosaveRows.entries.size(), size_t(1));
        ASSERT_TRUE(autosaveRows.entries[0].version.modelCount > 0);
        ASSERT_FALSE(autosaveRows.entries[0].checkpointPath.empty());

        const auto hasPredictionDifference = [](const auto& entry) {
            return std::any_of(entry.agentSetupDiffs.begin(), entry.agentSetupDiffs.end(),
                [](const auto& diff) { return diff.field == "Ball Prediction Simulator"; });
        };
        ASSERT_TRUE(hasPredictionDifference(rows.entries[0]));

        ModelBenchmarkWidget widget(nullptr, PERSIST_SETUP);
        ModelBenchmarkWidget::BenchmarkVersionEntry refreshed = rows.entries[0];
        widget.RefreshAgentSetupStatus(refreshed);
        ASSERT_TRUE(hasPredictionDifference(refreshed));
        const std::optional<RocketForge::AgentSetupSnapshot> current =
            RocketForge::CaptureAgentSetupSnapshot(*AgentRegistry::Create("CatalogPublicFixture"));
        widget.RefreshAgentSetupStatus(refreshed, &current);
        ASSERT_TRUE(hasPredictionDifference(refreshed));

        // A captured setup must be visible and replaceable even without its training snapshot.
        for (bool hasSavedSetup : {true, false}) {
            ModelBenchmarkWidget restored(nullptr, PERSIST_SETUP);
            ModelBenchmarkWidget::BenchmarkVersionEntry entry = rows.entries[0];
            entry.agentSetupSource = ModelBenchmarkAgentSetupSource::CurrentAgent;
            entry.agentSetupJson = model.agentSetup.dump();
            entry.hasSavedAgentSetup = hasSavedSetup;
            if (!hasSavedSetup)
                entry.trainingSetupPath.clear();

            restored.m_selectedVersions = {entry};
            restored.PopulateSelectedTable();
            QComboBox* choice = nullptr;
            for (QComboBox* combo : restored.m_selectedTable->findChildren<QComboBox*>()) {
                if (combo->currentText() == "Current (saved version)")
                    choice = combo;
            }

            ASSERT_TRUE(choice && choice->isEnabled());
            ASSERT_TRUE(choice->toolTip().contains("Ball Prediction Simulator: 2 -> 3"));
            ASSERT_EQ(choice->findText("Previous run"), -1);
            ASSERT_EQ(restored.CaptureModelSetup(restored.m_selectedVersions[0]), entry.agentSetupJson);
            choice->setCurrentIndex(choice->findData((int)ModelBenchmarkAgentSetupSource::CurrentAgent));
            ASSERT_TRUE(restored.m_selectedVersions[0].agentSetupJson.empty());
            const std::optional<RocketForge::AgentSetupSnapshot> selected = RocketForge::AgentSetupSnapshotFromJson(nlohmann::json::parse(
                restored.CaptureModelSetup(restored.m_selectedVersions[0])));
            ASSERT_TRUE(selected.has_value());
            ASSERT_TRUE(RocketForge::SameAgentSetupSnapshot(*current, *selected));
        }

        // Loading a run preserves each model's own source instead of replacing it with a run label.
        for (auto source : {ModelBenchmarkAgentSetupSource::SavedSegment, ModelBenchmarkAgentSetupSource::CurrentAgent}) {
            ModelBenchmarkWidget restored(nullptr, PERSIST_SETUP);
            ModelBenchmarkWidget::BenchmarkVersionEntry entry = rows.entries[0];
            entry.agentSetupSource = source;
            entry.agentSetupJson = source == ModelBenchmarkAgentSetupSource::SavedSegment ? model.agentSetup.dump()
                : RocketForge::AgentSetupSnapshotToJson(*current).dump();
            entry.fingerprint = "recorded-fingerprint";
            entry.libraryModelId = 123;
            restored.m_selectedVersions = {entry};
            restored.PopulateSelectedTable();
            QComboBox* choice = nullptr;
            for (QComboBox* combo : restored.m_selectedTable->findChildren<QComboBox*>()) {
                if (combo->findText("Trained") >= 0 && combo->findText("Current") >= 0)
                    choice = combo;
            }

            ASSERT_TRUE(choice);
            ASSERT_EQ(choice->count(), 2);
            ASSERT_EQ(choice->currentText(), source == ModelBenchmarkAgentSetupSource::SavedSegment ? QString("Trained") : QString("Current"));
            ASSERT_EQ(restored.CaptureModelSetup(restored.m_selectedVersions[0]), entry.agentSetupJson);

            // Reselecting the same visible option must apply the live source explicitly.
            choice->activated(choice->currentIndex());
            ASSERT_TRUE(restored.m_selectedVersions[0].agentSetupJson.empty());
            ASSERT_TRUE(restored.m_selectedVersions[0].fingerprint.isEmpty());
            ASSERT_EQ(restored.m_selectedVersions[0].libraryModelId, qint64(-1));
            ASSERT_EQ(restored.m_selectedVersions[0].agentSetupSource, source);
        }

        ModelBenchmarkWidget::BenchmarkVersionEntry missing = rows.entries[0];
        missing.agentSetupJson = model.agentSetup.dump();
        missing.agentSetupSource = ModelBenchmarkAgentSetupSource::SavedSegment;
        missing.trainingSetupPath.clear();
        missing.sourceLabel = "Imported Model";
        widget.RefreshAgentSetupStatus(missing);
        ASSERT_EQ(missing.agentSetupSource, ModelBenchmarkAgentSetupSource::SavedSegment);
        QString missingLabel;
        widget.BuildSetupStatusTooltip(missing, &missingLabel);
        ASSERT_EQ(missingLabel, QString("Trained (saved version)"));
        ASSERT_EQ(widget.CaptureModelSetup(missing), model.agentSetup.dump());
    }

    static void PersistenceDisabled() {
        QTemporaryDir directory;
        QFile settings(directory.filePath("last-used-settings.json"));
        const QByteArray original(R"({"presetName":"User preset"})");
        ASSERT_TRUE(settings.open(QIODevice::WriteOnly));
        ASSERT_EQ(settings.write(original), qint64(original.size()));
        settings.close();

        {
            ModelBenchmarkWidget widget(nullptr, PERSIST_SETUP);

            // Redirect even a regressed writer to a disposable file, never user data.
            widget.m_settingsPath = settings.fileName().toStdString();
            widget.ReloadPersistedSetup();
            ASSERT_TRUE(widget.m_savedResultsWidget->LibraryPath().empty());
            widget.m_maxGamesSpin->setValue(123);
            widget.m_workspaceRefreshTimer->stop();
            widget.SavePersistedSetup();
            QThread::msleep(300);
            QApplication::processEvents();
            ASSERT_FALSE(widget.m_setupSaveTimer->isActive());

            // Also cover the destructor's pending save.
            widget.m_setupSaveTimer->start();
        }

        ASSERT_TRUE(settings.open(QIODevice::ReadOnly));
        ASSERT_EQ(settings.readAll(), original);
    }

    static void Compatibility() {
        using namespace RocketForgeTraining;
        AgentRegistry::Register("ReuseFixture", [] { return std::make_unique<Agent>(); });
        QTemporaryDir directory;
        ModelBenchmarkWidget widget(nullptr, PERSIST_SETUP);
        widget.m_loadingPersistedSetup = true;
        ASSERT_TRUE(widget.m_savedResultsWidget->OpenLibrary(directory.filePath("compatibility.sqlite").toStdString(), SHOW_ERRORS));
        ModelBenchmarkLibrary& library = widget.m_savedResultsWidget->Library();
        std::vector<ModelBenchmarkLibraryModel> models(2);
        for (int i = 0; i < 2; ++i) {
            models[i].displayName = models[i].logicalKey = models[i].fingerprint = QString::number(i);
            models[i].checkpointPath = directory.filePath(QString::number(i));
            ASSERT_TRUE(QDir().mkpath(models[i].checkpointPath));
        }

        for (int environment = 0; environment < 3; ++environment) {
            ModelBenchmarkResult result;
            result.modelNames = { "0", "1" };
            result.rocketSimVersions = { environment == 2 ? 3 : 2 };
            result.evaluationIdentity = "{\"environment\":" + std::to_string(environment) + ",\"benchmarkCompatibilityVersion\":99}";
            result.stochasticKickoff = environment == 1;
            result.kickoffTemperature = 2.0f;

            ModelBenchmarkGameMetadata game;
            game.modelA = game.blueModel = 0;
            game.modelB = game.orangeModel = 1;
            game.mode = 1;
            game.gameId = game.snapshotId = environment + 1;
            game.mirrorCompleted = true;
            result.games = { game };
            ASSERT_TRUE(library.SaveRun(QString::number(environment), result, models, {}));
        }

        ASSERT_EQ(library.ListEnvironments().size(), size_t(3));
        widget.m_coverage = library.LoadCoverage({});
        ASSERT_EQ(widget.m_coverage.size(), size_t(3));
        for (const ModelBenchmarkLibraryModel& model : library.ListModels()) {
            ModelBenchmarkWidget::BenchmarkVersionEntry entry;
            entry.agentName = "ReuseFixture";
            entry.agentSetupSource = ModelBenchmarkAgentSetupSource::CurrentAgent;
            entry.version.id = model.displayName.toStdString();
            entry.libraryModelId = model.id;
            entry.checkpointPath = model.checkpointPath.toStdString();
            widget.m_selectedVersions.push_back(entry);
        }

        widget.m_maxGamesSpin->setValue(5);
        widget.m_mode2v2Check->setChecked(false);
        widget.m_mode3v3Check->setChecked(false);
        widget.m_engineCombo->setCurrentIndex(widget.m_engineCombo->findData(2));
        auto needed = [&]() { return widget.PrepareSchedule(SHOW_PREVIEW).pairBudgets.at(0).games; };
        auto resultGames = [&]() {
            std::vector<ModelBenchmarkLibrarySelection> selection;
            for (const ModelBenchmarkWidget::BenchmarkVersionEntry& entry : widget.m_selectedVersions)
                selection.push_back({entry.libraryModelId, true});

            QString error;
            const ModelBenchmarkLibraryAggregate result = library.LoadAggregate(selection, 0.95f, &error, {}, widget.WorkspaceFilter());
            ASSERT_TRUE(error.isEmpty());

            qint64 count = 0;
            for (const auto& [key, environment] : result.environments)
                count += environment.games;

            return count;
        };

        // Both v2 environments count, including different kickoff and compatibility markers.
        ASSERT_EQ(needed(), 3);
        ASSERT_EQ(resultGames(), qint64(2));
        for (const auto& [engine, check] : widget.m_reuseEngineChecks) {
            ASSERT_FALSE(check->isEnabled());
            ASSERT_FALSE(widget.m_reuseEnvironmentButtons.at(engine)->isEnabled());
        }

        widget.m_engineCombo->setCurrentIndex(widget.m_engineCombo->findData(3));
        ASSERT_EQ(needed(), 4);

        // Explicit cross-simulator reuse.
        widget.m_reuseSelectedEngineCheck->setChecked(false);
        widget.m_reuseEngineChecks.at(2)->setChecked(true);
        ASSERT_EQ(needed(), 2);
        ASSERT_EQ(resultGames(), qint64(3));
        widget.m_engineCombo->setCurrentIndex(widget.m_engineCombo->findData(2));
        ASSERT_TRUE(widget.m_reuseEngineChecks.at(3)->isChecked());

        bool popupWorked = false;
        std::string selectedKey;
        QTimer::singleShot(50, &widget, [&]() {
            auto* dialog = widget.m_testSetupDialog->findChild<QDialog*>("benchmarkCompatibilityDialog");
            if (!dialog)
                return;

            auto* all = dialog->findChild<QCheckBox*>("benchmarkAllEnvironments");
            auto* table = dialog->findChild<QTableWidget*>("benchmarkCompatibilityEnvironments");
            if (all && table && table->rowCount() == 2) {
                all->setChecked(false);
                table->item(0, 0)->setCheckState(Qt::Checked);
                table->item(1, 0)->setCheckState(Qt::Unchecked);
                selectedKey = table->item(0, 0)->data(Qt::UserRole).toString().toStdString();
                popupWorked = dialog->width() <= dialog->screen()->availableGeometry().width();
                const QString output = qEnvironmentVariable("CODEX_BENCHMARK_UI_OUTPUT");
                if (!output.isEmpty())
                    dialog->grab().save(output + "/compatibility-environments.png");
            }

            dialog->accept();
        });
        widget.ShowCompatibilityEnvironments(2);
        ASSERT_TRUE(popupWorked);
        ASSERT_EQ(needed(), 3);
        ASSERT_EQ(widget.m_reuseEnvironmentKeys.at(2).size(), size_t(1));
        ASSERT_EQ(resultGames(), qint64(2));

        const ModelBenchmarkSettings saved = widget.BuildPersistedSetup();
        const ModelBenchmarkSettingsLoadResult decoded = DeserializeModelBenchmarkSettings(SerializeModelBenchmarkSettings(saved));
        ASSERT_TRUE(decoded.ok);
        ASSERT_FALSE(decoded.value.reuseSelectedEngine);
        ASSERT_EQ(decoded.value.reuseEngineVersions.size(), size_t(2));
        ASSERT_EQ(decoded.value.reuseEnvironmentKeys.at(2).at(0), selectedKey);

        widget.m_reuseEnvironmentKeys[2].clear();
        ASSERT_EQ(needed(), 4);
        widget.m_reuseEngineChecks.at(3)->setChecked(false);
        ASSERT_EQ(needed(), 5);
        ASSERT_EQ(resultGames(), qint64(0));

        // The default ignores stored custom restrictions and disables every override.
        widget.m_reuseSelectedEngineCheck->setChecked(true);
        ASSERT_EQ(needed(), 3);
        ASSERT_EQ(resultGames(), qint64(2));
        for (const auto& [engine, check] : widget.m_reuseEngineChecks) {
            ASSERT_FALSE(check->isEnabled());
            ASSERT_FALSE(widget.m_reuseEnvironmentButtons.at(engine)->isEnabled());
        }

        widget.m_engineCombo->setCurrentIndex(widget.m_engineCombo->findData(3));
        ASSERT_EQ(resultGames(), qint64(1));
        widget.m_engineCombo->setCurrentIndex(widget.m_engineCombo->findData(2));
        widget.m_reuseSelectedEngineCheck->setChecked(false);
        ASSERT_TRUE(widget.m_reuseEnvironmentButtons.at(2)->isEnabled());
        ASSERT_EQ(needed(), 5);
        ASSERT_EQ(resultGames(), qint64(0));

        // Canceling Test Setup restores compatibility choices as well as ordinary settings.
        QTimer::singleShot(50, &widget, [&]() {
            const QString output = qEnvironmentVariable("CODEX_BENCHMARK_UI_OUTPUT");
            if (!output.isEmpty())
                widget.m_testSetupDialog->grab().save(output + "/compatibility-setup.png");

            widget.m_reuseSelectedEngineCheck->setChecked(true);
            if (!output.isEmpty())
                widget.m_testSetupDialog->grab().save(output + "/compatibility-default.png");

            widget.m_reuseEnvironmentKeys.clear();
            widget.m_testSetupDialog->reject();
        });
        widget.ShowTestSetup();
        ASSERT_FALSE(widget.m_reuseSelectedEngineCheck->isChecked());
        ASSERT_TRUE(widget.m_reuseEnvironmentKeys.contains(2));
        ASSERT_TRUE(widget.m_reuseEnvironmentKeys.at(2).empty());

        widget.m_reuseEnvironmentKeys[2] = { "missing-historical-environment" };
        bool retainedMissing = false;
        QTimer::singleShot(0, &widget, [&]() {
            auto* dialog = widget.m_testSetupDialog->findChild<QDialog*>("benchmarkCompatibilityDialog");
            if (!dialog)
                return;

            auto* table = dialog->findChild<QTableWidget*>("benchmarkCompatibilityEnvironments");
            if (table && table->rowCount() == 3) {
                retainedMissing = table->item(2, 0)->checkState() == Qt::Checked;
                table->item(2, 0)->setCheckState(Qt::Unchecked);
            }

            dialog->reject();
        });
        widget.ShowCompatibilityEnvironments(2);
        ASSERT_TRUE(retainedMissing);
        ASSERT_EQ(widget.m_reuseEnvironmentKeys.at(2).at(0), std::string("missing-historical-environment"));

        widget.m_reuseEnvironmentKeys.erase(2);
        ModelBenchmarkLibraryCoverage future = *std::find_if(widget.m_coverage.begin(), widget.m_coverage.end(),
            [](const ModelBenchmarkLibraryCoverage& row) { return row.engineVersion == 2; });
        future.environmentKey = "future-environment";
        future.compatibilityVersion = 999;
        widget.m_coverage.push_back(future);
        ASSERT_EQ(needed(), 2);
        widget.m_reuseEngineChecks.at(2)->setChecked(false);
        ASSERT_EQ(needed(), 5);
        widget.m_loadingPersistedSetup = true;
        widget.m_setupSaveTimer->stop();
        widget.m_workspaceRefreshTimer->stop();

        const ModelBenchmarkSettingsLoadResult legacy = DeserializeModelBenchmarkSettings("{}");
        ASSERT_TRUE(legacy.ok);
        ASSERT_TRUE(legacy.value.reuseSelectedEngine);
        ASSERT_TRUE(legacy.value.reuseEnvironmentKeys.empty());

        const ModelBenchmarkSettingsLoadResult oldSelection = DeserializeModelBenchmarkSettings(R"({"reuseSelectedEngine":false,"reuseEngineVersions":[0,3],"reuseEnvironmentKeys":{"0":["legacy-major:0,"],"2":["unused-old-selection"],"3":["exact:v3"]}})");
        ASSERT_TRUE(oldSelection.ok);
        ASSERT_EQ(oldSelection.value.reuseEngineVersions, (std::vector<int>{2, 3}));
        ASSERT_EQ(oldSelection.value.reuseEnvironmentKeys.at(2), (std::vector<std::string>{"legacy-major:2,"}));
        ASSERT_EQ(oldSelection.value.reuseEnvironmentKeys.at(3), (std::vector<std::string>{"exact:v3"}));
        ASSERT_FALSE(oldSelection.value.reuseEnvironmentKeys.contains(0));

        const ModelBenchmarkSettingsLoadResult oldAll = DeserializeModelBenchmarkSettings(R"({"reuseSelectedEngine":false,"reuseEngineVersions":[0,2],"reuseEnvironmentKeys":{"2":["exact:v2"]}})");
        ASSERT_TRUE(oldAll.ok);
        ASSERT_FALSE(oldAll.value.reuseEnvironmentKeys.contains(2));

        const ModelBenchmarkSettingsLoadResult oldNarrow = DeserializeModelBenchmarkSettings(R"({"reuseSelectedEngine":false,"reuseEngineVersions":[0,2],"reuseEnvironmentKeys":{"0":["legacy-major:0,"],"2":["exact:v2"]}})");
        ASSERT_TRUE(oldNarrow.ok);
        ASSERT_EQ(oldNarrow.value.reuseEnvironmentKeys.at(2), (std::vector<std::string>{"exact:v2", "legacy-major:2,"}));
    }

    static void NewPreset() {
        using namespace RocketForgeTraining;
        QTemporaryDir directory;
        ModelBenchmarkWidget widget(nullptr, PERSIST_SETUP);
        ASSERT_TRUE(widget.m_savedResultsWidget->OpenLibrary(directory.filePath("new.sqlite").toStdString(), SHOW_ERRORS));
        ModelBenchmarkLibrary& library = widget.m_savedResultsWidget->Library();
        ModelBenchmarkSettings original = widget.BuildPersistedSetup();
        original.presetName = "Original";
        original.maxGamesPerMode = 123;
        ASSERT_TRUE(library.CreatePreset("Original", QString::fromStdString(SerializeModelBenchmarkSettings(original))));
        widget.RefreshPresetList();
        widget.m_presetCombo->setCurrentIndex(widget.m_presetCombo->findData("Original"));
        widget.LoadWorkspacePreset();
        widget.m_maxGamesSpin->setValue(456);

        QTimer responder;
        QObject::connect(&responder, &QTimer::timeout, &widget, []() {
            if (auto* input = qobject_cast<QInputDialog*>(QApplication::activeModalWidget())) {
                input->setTextValue(" Fresh ");
                input->accept();
            } else if (auto* prompt = qobject_cast<QMessageBox*>(QApplication::activeModalWidget())) {
                prompt->button(QMessageBox::Discard)->click();
            }
        });
        responder.start(10);
        auto* button = widget.findChild<QPushButton*>("benchmarkNewPreset");
        ASSERT_TRUE(button != nullptr);
        button->click();
        responder.stop();

        ASSERT_EQ(widget.m_presetName, QString("Fresh"));
        ASSERT_EQ(widget.m_presetCombo->currentData().toString(), QString("Fresh"));
        ASSERT_TRUE(widget.m_selectedVersions.empty());
        ASSERT_EQ(widget.m_maxGamesSpin->value(), ModelBenchmarkSettings().maxGamesPerMode);
        ASSERT_EQ(widget.m_presetStatusLabel->text(), QString("Preset saved"));
        ASSERT_EQ(library.ListPresets().size(), size_t(2));
        for (const ModelBenchmarkWorkspaceRecord& preset : library.ListPresets()) {
            const ModelBenchmarkSettingsLoadResult decoded = DeserializeModelBenchmarkSettings(preset.settingsJson.toStdString());
            ASSERT_TRUE(decoded.ok);
            ASSERT_EQ(decoded.value.maxGamesPerMode, preset.id == "Original" ? 123 : ModelBenchmarkSettings().maxGamesPerMode);
        }

        QString error;
        ASSERT_FALSE(library.CreatePreset("fresh", "{}", &error));
        ASSERT_FALSE(library.CreatePreset("  ", "{}", &error));

        QTimer::singleShot(0, &widget, []() {
            if (auto* input = qobject_cast<QInputDialog*>(QApplication::activeModalWidget()))
                input->reject();
        });
        button->click();
        ASSERT_EQ(widget.m_presetName, QString("Fresh"));
        ASSERT_EQ(library.ListPresets().size(), size_t(2));
    }

    static void RenamePreset() {
        using namespace RocketForgeTraining;
        QTemporaryDir directory;
        ModelBenchmarkWidget widget(nullptr, PERSIST_SETUP);
        widget.m_loadingPersistedSetup = true;
        ASSERT_TRUE(widget.m_savedResultsWidget->OpenLibrary(directory.filePath("rename.sqlite").toStdString(), SHOW_ERRORS));
        ModelBenchmarkLibrary& library = widget.m_savedResultsWidget->Library();
        ModelBenchmarkSettings settings = widget.BuildPersistedSetup();
        settings.presetName = "Original";
        settings.maxGamesPerMode = 100;
        ASSERT_TRUE(library.SavePreset("Original", QString::fromStdString(SerializeModelBenchmarkSettings(settings))));
        ASSERT_TRUE(library.SavePreset("Other", QString::fromStdString(SerializeModelBenchmarkSettings(settings))));

        widget.RefreshPresetList();
        ASSERT_FALSE(widget.m_renamePresetButton->isEnabled());
        widget.m_presetCombo->setCurrentIndex(widget.m_presetCombo->findData("Original"));
        widget.LoadWorkspacePreset();
        ASSERT_TRUE(widget.m_renamePresetButton->isEnabled());
        widget.m_maxGamesSpin->setValue(123);

        QTimer::singleShot(0, &widget, []() {
            auto* prompt = qobject_cast<QInputDialog*>(QApplication::activeModalWidget());
            if (prompt) {
                prompt->setTextValue("Renamed");
                prompt->accept();
            }
        });
        widget.m_renamePresetButton->click();
        ASSERT_EQ(widget.m_presetName, QString("Renamed"));
        ASSERT_EQ(widget.m_presetCombo->currentData().toString(), QString("Renamed"));
        ASSERT_EQ(widget.m_maxGamesSpin->value(), 123);
        widget.UpdateWorkspaceSummary();
        ASSERT_EQ(widget.m_presetStatusLabel->text(), QString("Unsaved changes"));
        ASSERT_EQ(library.ListPresets().size(), size_t(2));
        for (const ModelBenchmarkWorkspaceRecord& preset : library.ListPresets()) {
            ASSERT_TRUE(preset.id != "Original");
            if (preset.id == "Renamed") {
                const ModelBenchmarkSettingsLoadResult saved = DeserializeModelBenchmarkSettings(preset.settingsJson.toStdString());
                ASSERT_TRUE(saved.ok);
                ASSERT_EQ(saved.value.presetName, std::string("Renamed"));
                ASSERT_EQ(saved.value.maxGamesPerMode, 100);
            }
        }

        QString error;
        ASSERT_FALSE(library.RenamePreset("Renamed", "other", &error));
        ASSERT_FALSE(error.isEmpty());
        ASSERT_FALSE(library.RenamePreset("Renamed", "  ", &error));
        ASSERT_TRUE(library.RenamePreset("Renamed", "RENAMED", &error));
        ASSERT_EQ(library.ListPresets().size(), size_t(2));
    }

    static void ComparisonSort() {
        ModelBenchmarkWidget widget(nullptr, PERSIST_SETUP);
        widget.m_loadingPersistedSetup = true;
        auto add = [&](const char* agent, const char* build, int version) {
            ModelBenchmarkWidget::BenchmarkVersionEntry entry;
            entry.agentName = agent;
            entry.buildName = build;
            entry.version.major = version;
            entry.version.id = "v" + std::to_string(version);
            widget.m_selectedVersions.push_back(entry);
        };

        add("FixtureB", "A", 1);
        add("FixtureA", "B", 1);
        add("FixtureA", "A", 10);
        add("FixtureA", "A", 2);
        widget.SortSelectedVersions();
        widget.PopulateSelectedTable();
        ASSERT_EQ(widget.m_selectedVersions[0].agentName, std::string("FixtureA"));
        ASSERT_EQ(widget.m_selectedVersions[0].buildName, std::string("A"));
        ASSERT_EQ(widget.m_selectedVersions[0].version.major, 2);
        ASSERT_EQ(widget.m_selectedVersions[1].version.major, 10);
        ASSERT_EQ(widget.m_selectedVersions[2].buildName, std::string("B"));
        ASSERT_EQ(widget.m_selectedVersions[3].agentName, std::string("FixtureB"));

        widget.m_selectedTable->horizontalHeader()->sectionClicked(3);
        ASSERT_EQ(widget.m_selectedVersions[0].agentName, std::string("FixtureB"));
        widget.m_selectedTable->horizontalHeader()->sectionClicked(4);
        ASSERT_EQ(widget.m_selectedVersions.back().buildName, std::string("B"));

        add("Alpha", "Z", 1);
        widget.SortSelectedVersions();
        ASSERT_EQ(widget.m_selectedVersions.back().buildName, std::string("Z"));
    }

    static void KickoffSettings() {
        using namespace RocketForgeTraining;
        QTemporaryDir directory;
        ASSERT_TRUE(directory.isValid());
        ModelBenchmarkWidget widget(nullptr, PERSIST_SETUP);
        widget.m_loadingPersistedSetup = true;
        ASSERT_TRUE(widget.m_savedResultsWidget->OpenLibrary(directory.filePath("kickoff.sqlite").toStdString(), SHOW_ERRORS));
        widget.UpdateInputState();
        ASSERT_FALSE(widget.m_stochasticKickoffCheck->isChecked());
        ASSERT_FALSE(widget.m_kickoffTemperatureSpin->isEnabled());

        widget.m_stochasticKickoffCheck->setChecked(true);
        widget.m_kickoffTemperatureSpin->setValue(0.65);
        ASSERT_TRUE(widget.m_kickoffTemperatureSpin->isEnabled());
        ASSERT_TRUE(widget.BuildConfig().stochasticKickoff);
        ASSERT_EQ(widget.BuildConfig().kickoffTemperature, 0.65f);

        const auto saved = DeserializeModelBenchmarkSettings(SerializeModelBenchmarkSettings(widget.BuildPersistedSetup()));
        ASSERT_TRUE(saved.ok);
        widget.m_stochasticKickoffCheck->setChecked(false);
        widget.ApplyPersistedSetup(saved.value);
        widget.UpdateInputState();
        ASSERT_TRUE(widget.m_stochasticKickoffCheck->isChecked());
        ASSERT_EQ(widget.BuildConfig().kickoffTemperature, 0.65f);

        QTimer::singleShot(0, &widget, [&]() {
            widget.m_stochasticKickoffCheck->setChecked(false);
            widget.m_kickoffTemperatureSpin->setValue(2.0);
            widget.m_testSetupDialog->reject();
        });
        widget.ShowTestSetup();
        ASSERT_TRUE(widget.BuildConfig().stochasticKickoff);
        ASSERT_EQ(widget.BuildConfig().kickoffTemperature, 0.65f);

        QTimer::singleShot(0, &widget, [&]() {
            widget.m_kickoffTemperatureSpin->setValue(2.0);
            widget.m_testSetupDialog->accept();
        });
        widget.ShowTestSetup();
        ASSERT_EQ(widget.BuildConfig().kickoffTemperature, 2.0f);

        widget.SetRunning(true);
        ASSERT_FALSE(widget.m_stochasticKickoffCheck->isEnabled());
        ASSERT_FALSE(widget.m_kickoffTemperatureSpin->isEnabled());
        widget.SetRunning(false);
        ASSERT_TRUE(widget.m_kickoffTemperatureSpin->isEnabled());
    }

    static void FixedRunSettings() {
        QTemporaryDir directory;
        ModelBenchmarkWidget widget(nullptr, PERSIST_SETUP);
        widget.m_loadingPersistedSetup = true;
        ASSERT_TRUE(widget.m_savedResultsWidget->OpenLibrary(directory.filePath("fixed-settings.sqlite").toStdString(), SHOW_ERRORS));
        RocketForgeTraining::ModelBenchmarkSettings legacy = widget.BuildPersistedSetup();
        legacy.presetName = "Legacy early stop";
        legacy.runUntilMaxGames = false;
        legacy.stopWhenRankingKnown = true;
        legacy.betterConfidence = 0.8f;
        legacy.eloRangeConfidence = 0.9f;
        legacy.maxEloUncertaintyRange = 999.0f;
        legacy.collectOptionalStats = false;
        ASSERT_TRUE(widget.m_savedResultsWidget->Library().SavePreset("Legacy early stop",
            QString::fromStdString(RocketForgeTraining::SerializeModelBenchmarkSettings(legacy))));

        widget.ApplyWorkspaceSettings(legacy);
        widget.m_setupSaveTimer->stop();
        widget.m_workspaceRefreshTimer->stop();
        widget.UpdateWorkspaceSummary();
        ASSERT_EQ(widget.m_presetStatusLabel->text(), QString("Preset saved"));

        for (int mode : { 0, 1 }) {
            widget.m_scheduleModeCombo->setCurrentIndex(mode);
            const RocketForgeTraining::ModelBenchmarkConfig config = widget.BuildConfig();
            ASSERT_TRUE(config.runUntilMaxGames);
            ASSERT_FALSE(config.stopWhenRankingKnown);
            ASSERT_EQ(config.eloRangeConfidence, 0.95f);
            ASSERT_TRUE(config.collectOptionalStats);

            const RocketForgeTraining::ModelBenchmarkSettings saved = widget.BuildPersistedSetup();
            ASSERT_TRUE(saved.runUntilMaxGames);
            ASSERT_EQ(saved.eloRangeConfidence, 0.95f);
            ASSERT_TRUE(saved.collectOptionalStats);
        }

        for (QCheckBox* check : widget.m_testSetupDialog->findChildren<QCheckBox*>()) {
            ASSERT_FALSE(check->text().contains("Run to the game limit"));
            ASSERT_FALSE(check->text().contains("Stop once model ranking"));
            ASSERT_FALSE(check->text().contains("optional statistics"));
        }

        for (QLabel* label : widget.m_testSetupDialog->findChildren<QLabel*>()) {
            ASSERT_FALSE(label->text().contains("Benchmark ranking confidence:"));
            ASSERT_FALSE(label->text().contains("Elo range confidence:"));
            ASSERT_FALSE(label->text().contains("Max Elo range:"));
        }
    }

    static void PreviousRunPicker() {
        using namespace RocketForgeTraining;
        QTemporaryDir directory;
        ModelBenchmarkWidget widget(nullptr, PERSIST_SETUP);
        widget.m_loadingPersistedSetup = true;
        ASSERT_TRUE(widget.m_savedResultsWidget->OpenLibrary(directory.filePath("previous.sqlite").toStdString(), SHOW_ERRORS));

        ModelBenchmarkResult result;
        result.modelNames = { std::string(3000, 'A'), "Second model" };
        ModelBenchmarkGameMetadata game;
        game.modelA = game.blueModel = 0;
        game.modelB = game.orangeModel = 1;
        game.mode = 1;
        result.games = { game };

        std::vector<ModelBenchmarkLibraryModel> models(2);
        for (int i = 0; i < 2; ++i) {
            models[i].displayName = QString::fromStdString(result.modelNames[i]);
            models[i].logicalKey = models[i].fingerprint = QString::number(i);
        }

        ModelBenchmarkSettings settings = widget.BuildPersistedSetup();
        settings.maxGamesPerMode = 321;
        ASSERT_TRUE(widget.m_savedResultsWidget->Library().SaveRun("long-roster", result, models, {}, nullptr,
            QString::fromStdString(SerializeModelBenchmarkSettings(settings))));

        bool readable = false;
        bool filtered = false;
        QTimer::singleShot(0, &widget, [&]() {
            auto* dialog = widget.findChild<QDialog*>("previousBenchmarkRunDialog");
            if (!dialog)
                return;

            auto* list = dialog->findChild<QListWidget*>("previousBenchmarkRunList");
            auto* details = dialog->findChild<QPlainTextEdit*>("previousBenchmarkRunDetails");
            auto* search = dialog->findChild<QLineEdit*>();
            auto* buttons = dialog->findChild<QDialogButtonBox*>();
            readable = dialog->width() <= dialog->screen()->availableGeometry().width()
                && details && details->toPlainText().contains(QString(3000, 'A'));
            if (list && search && buttons) {
                search->setText("no matching run");
                filtered = list->item(0)->isHidden() && !buttons->button(QDialogButtonBox::Ok)->isEnabled();
                search->clear();
                filtered = filtered && !list->item(0)->isHidden() && buttons->button(QDialogButtonBox::Ok)->isEnabled();
            }

            dialog->accept();
        });
        widget.LoadPreviousRun();
        ASSERT_TRUE(readable);
        ASSERT_TRUE(filtered);
        ASSERT_EQ(widget.m_maxGamesSpin->value(), 321);
    }

    static void PresetSwitching() {
        QTemporaryDir directory;
        ModelBenchmarkWidget widget(nullptr, PERSIST_SETUP);
        ASSERT_TRUE(widget.m_savedResultsWidget->OpenLibrary(directory.filePath("presets.sqlite").toStdString(), SHOW_ERRORS));
        RocketForgeTraining::ModelBenchmarkSettings settings = widget.BuildPersistedSetup();
        for (const char* name : { "First", "Second" }) {
            settings.presetName = name;
            settings.maxGamesPerMode = name == std::string("First") ? 100 : 200;
            ASSERT_TRUE(widget.m_savedResultsWidget->Library().SavePreset(name,
                QString::fromStdString(RocketForgeTraining::SerializeModelBenchmarkSettings(settings))));
        }

        widget.RefreshPresetList();
        auto select = [&](const QString& name) {
            const int index = widget.m_presetCombo->findData(name);
            ASSERT_TRUE(index >= 0);
            widget.m_presetCombo->setCurrentIndex(index);
            widget.m_presetCombo->activated(index);
        };
        auto answer = [&](QMessageBox::StandardButton button) {
            QTimer::singleShot(0, &widget, [button]() {
                auto* prompt = qobject_cast<QMessageBox*>(QApplication::activeModalWidget());
                if (prompt)
                    prompt->button(button)->click();
            });
        };

        select("First");
        ASSERT_EQ(widget.m_maxGamesSpin->value(), 100);
        widget.m_maxGamesSpin->setValue(123);
        answer(QMessageBox::Cancel);
        select("Second");
        ASSERT_EQ(widget.m_presetCombo->currentData().toString(), QString("First"));
        ASSERT_EQ(widget.m_maxGamesSpin->value(), 123);
        answer(QMessageBox::Save);
        select("Second");
        ASSERT_EQ(widget.m_presetName, QString("Second"));
        ASSERT_EQ(widget.m_maxGamesSpin->value(), 200);
        select("First");
        ASSERT_EQ(widget.m_maxGamesSpin->value(), 123);
        widget.m_maxGamesSpin->setValue(456);
        answer(QMessageBox::Discard);
        select("Second");
        select("First");
        ASSERT_EQ(widget.m_maxGamesSpin->value(), 123);
    }

    static void AsynchronousWorkspace() {
        QTemporaryDir directory;
        ModelBenchmarkWidget widget(nullptr, PERSIST_SETUP);
        widget.m_loadingPersistedSetup = true;
        ASSERT_TRUE(widget.m_savedResultsWidget->OpenLibrary(directory.filePath("async.sqlite").toStdString(), SHOW_ERRORS));
        widget.m_workspaceRefreshTimer->stop();
        widget.m_loadingPersistedSetup = false;

        std::promise<ModelBenchmarkWidget::WorkspaceData> pending;
        widget.m_workspaceScan = pending.get_future();
        widget.m_scanningWorkspaceGeneration = widget.m_workspaceGeneration;
        ModelBenchmarkWidget::BenchmarkVersionEntry selected;
        selected.version.id = "new selection";
        selected.resultsOnly = true;
        widget.m_selectedVersions.push_back(selected);
        widget.OnSetupChanged();
        widget.m_setupSaveTimer->stop();
        widget.m_workspaceRefreshTimer->stop();

        ModelBenchmarkWidget::WorkspaceData stale;
        pending.set_value(std::move(stale));
        widget.FinishWorkspaceRefresh();
        ASSERT_EQ(widget.m_selectedVersions.size(), size_t(1));
        ASSERT_EQ(widget.m_selectedVersions[0].version.id, std::string("new selection"));
        ASSERT_TRUE(widget.m_workspaceScan.valid());
        ASSERT_FALSE(widget.m_runButton->isEnabled());

        bool dispatched = false;
        QTimer::singleShot(0, &widget, [&]() { dispatched = true; });
        QElapsedTimer timeout;
        timeout.start();
        while (widget.m_workspaceScan.valid() && timeout.elapsed() < 10000) {
            QApplication::processEvents();
            QThread::msleep(5);
        }

        ASSERT_TRUE(dispatched);
        ASSERT_FALSE(widget.m_workspaceScan.valid());
        ASSERT_TRUE(widget.m_coverageError.isEmpty());
        ASSERT_FALSE(widget.m_coverageDirty);
        ASSERT_EQ(widget.m_selectedVersions[0].version.id, std::string("new selection"));
        widget.m_loadingPersistedSetup = true;
        widget.m_setupSaveTimer->stop();
        widget.m_workspaceRefreshTimer->stop();
        std::cout << "Asynchronous coverage keeps the event loop live and rejects obsolete selections\n";
    }

    static void Schedule() {
        using namespace RocketForgeTraining;
        AgentRegistry::Register("ScheduleFixture", [] {
            std::unique_ptr<Agent> agent = std::make_unique<Agent>();
            agent->maxTeamSize = 3;
            return agent;
        });
        AgentRegistry::Register("ScheduleZ1v1", [] { return std::make_unique<Agent>(); });
        QTemporaryDir directory;
        ASSERT_TRUE(directory.isValid());
        ModelBenchmarkWidget widget(nullptr, PERSIST_SETUP);
        widget.m_loadingPersistedSetup = true;
        widget.m_engineCombo->setCurrentIndex(widget.m_engineCombo->findData(static_cast<int>(RocketSim::EngineId::V2)));
        widget.m_makeAllButton->menu()->actions()[0]->trigger();
        widget.m_maxGamesSpin->setValue(500);
        widget.m_mode1v1Check->setChecked(true);
        widget.m_mode2v2Check->setChecked(true);
        widget.m_mode3v3Check->setChecked(true);

        for (int i = 0; i < 3; ++i) {
            ModelBenchmarkWidget::BenchmarkVersionEntry entry;
            entry.agentName = i == 2 ? "ScheduleZ1v1" : "ScheduleFixture";
            entry.agentSetupSource = ModelBenchmarkAgentSetupSource::CurrentAgent;
            entry.version.id = std::to_string(i);
            entry.libraryModelId = i + 1;
            entry.checkpointPath = directory.filePath(QString::number(i)).toStdString();
            ASSERT_TRUE(QDir().mkpath(QString::fromStdString(entry.checkpointPath.string())));
            widget.m_selectedVersions.push_back(entry);
        }

        const auto identity = DescribeModelBenchmarkEnvironment(widget.BuildConfig()).evaluationIdentity;
        QJsonObject old = QJsonDocument::fromJson(QByteArray::fromStdString(identity)).object();
        QJsonObject profiles = old["savedPredictionProfiles"].toObject();
        profiles.remove(QString::fromStdString(widget.m_selectedVersions[2].checkpointPath.generic_string()));
        profiles["retired-model"] = "rocketsim-v3";
        old["savedPredictionProfiles"] = profiles;
        old["externalPrediction"] = QJsonObject{{"retired-model", QJsonObject{{"engine", 3}, {"revision", "old"}}}};

        const QString historical = QString::fromUtf8(QJsonDocument(old).toJson(QJsonDocument::Compact));
        widget.m_coverage = {{1, 2, 1, 500, 10, historical}, {1, 2, 2, 300, 20, historical},
            {1, 2, 3, 600, 30, historical}, {80, 90, 1, 900000, 999999, "unrelated"},
            {1, 2, 2, 50, 99999, "different-build"}};
        for (ModelBenchmarkLibraryCoverage& row : widget.m_coverage) {
            row.engineVersion = static_cast<int>(RocketSim::EngineId::V2);
            row.compatibilityVersion = RecordedModelBenchmarkCompatibilityVersion(row.evaluationIdentity.toStdString());
        }
        widget.m_coverage.back().engineVersion = static_cast<int>(RocketSim::EngineId::V2) == 2 ? 3 : 2;
        widget.m_seedEdit->setText("123");
        auto verify = [&]() {
            const ModelBenchmarkConfig config = widget.PrepareSchedule(SHOW_PREVIEW);
            qint64 budget = 0;
            qint64 tableBudget = 0;
            int scheduledRows = 0;
            for (const ModelBenchmarkPairBudget& pair : config.pairBudgets) {
                budget += pair.games;
                ASSERT_EQ(ModelBenchmarkPairGameTarget(config, pair.modelA, pair.modelB, pair.mode), pair.games);
                ASSERT_TRUE(pair.modelA >= 0 && pair.modelB < (int)config.models.size());
                ASSERT_FALSE(config.models[pair.modelA].agentSetupJson.empty());
            }

            for (int row = 0; row < widget.m_scheduleTable->rowCount(); ++row) {
                if (widget.m_scheduleTable->item(row, 6)->text() != "Scheduled")
                    continue;

                tableBudget += widget.m_scheduleTable->item(row, 5)->text().toLongLong();
                scheduledRows += widget.m_scheduleTable->item(row, 2)->text().split(", ").size();
            }

            for (int row = 0; row < widget.m_scheduleTable->rowCount(); ++row) {
                for (int column = 0; column < widget.m_scheduleTable->columnCount(); ++column)
                    ASSERT_EQ(widget.m_scheduleTable->item(row, column)->textAlignment(), Qt::AlignCenter);
            }

            ASSERT_EQ(budget, tableBudget);
            ASSERT_EQ(scheduledRows, (int)config.pairBudgets.size());
            ASSERT_TRUE(widget.m_scheduleSummaryLabel->text().contains(QString("%1 new games").arg(budget)));
            ASSERT_TRUE(widget.m_scheduleSummaryLabel->text().contains(QString("%1 new matchups to run").arg(scheduledRows)));
            ASSERT_FALSE(widget.m_scheduleSummaryLabel->text().contains("900000"));
            return config;
        };

        ModelBenchmarkConfig config = verify();
        ASSERT_EQ(config.pairBudgets.size(), size_t(3));
        ASSERT_EQ(config.snapshotOffset, uint64_t(30));
        ASSERT_TRUE(widget.m_scheduleSummaryLabel->text().contains("1400 covered games"));
        ASSERT_EQ(widget.m_scheduleTable->rowCount(), 4);
        ASSERT_EQ(widget.m_scheduleTable->item(0, 3)->text(), QString("1000"));
        ASSERT_EQ(widget.m_scheduleTable->item(1, 3)->text(), QString("500"));

        bool filterWorked = false;
        QTimer::singleShot(0, [&]() {
            auto* dialog = widget.findChild<QDialog*>("benchmarkScheduleDialog");
            auto* filter = dialog ? dialog->findChild<QCheckBox*>("hideCoveredMatchupsCheck") : nullptr;
            if (filter) {
                filterWorked = filter->isChecked();
                filterWorked = filterWorked && widget.m_scheduleTable->isRowHidden(0) && !widget.m_scheduleTable->isRowHidden(1);
                filter->setChecked(false);
                filterWorked = filterWorked && !widget.m_scheduleTable->isRowHidden(0);
            }

            if (dialog)
                dialog->reject();
        });
        const bool showPreview = true;
        widget.PrepareSchedule(showPreview);
        ASSERT_TRUE(filterWorked);
        ASSERT_EQ(widget.m_scheduleTable->item(0, 2)->text(), QString("1v1, 3v3"));
        ASSERT_EQ(widget.m_scheduleTable->item(0, 4)->text(), QString("1100"));
        ASSERT_EQ(widget.m_scheduleTable->item(0, 6)->text(), QString("Covered"));
        ASSERT_EQ(widget.m_scheduleTable->item(1, 2)->text(), QString("2v2"));
        ASSERT_EQ(widget.m_scheduleTable->item(1, 5)->text(), QString("200"));
        ASSERT_TRUE(widget.m_scheduleSummaryLabel->text().startsWith("1200 new games"));
        ASSERT_TRUE(widget.m_scheduleSummaryLabel->text().contains("2 covered matchups"));
        ASSERT_TRUE(widget.m_scheduleSummaryLabel->text().contains("50 saved games excluded by Compatibility"));

        const std::filesystem::path checkpoint = widget.m_selectedVersions[2].checkpointPath;
        widget.m_selectedVersions[2].checkpointPath /= "missing";
        config = widget.PrepareSchedule(SHOW_PREVIEW);
        ASSERT_EQ(config.pairBudgets.size(), size_t(1));
        ASSERT_TRUE(widget.m_scheduleSummaryLabel->text().startsWith("200 new games"));
        ASSERT_TRUE(widget.m_scheduleSummaryLabel->text().contains("2 matchups blocked (1000 games)"));
        ASSERT_FALSE(widget.m_runButton->isEnabled());

        bool refused = false;
        try {
            widget.PrepareSchedule(SHOW_PREVIEW, REQUIRE_RUNNABLE);
        } catch (const std::exception&) {
            refused = true;
        }

        ASSERT_TRUE(refused);
        widget.m_selectedVersions[2].checkpointPath = checkpoint;
        std::reverse(widget.m_selectedVersions.begin(), widget.m_selectedVersions.end());
        verify();
        ASSERT_TRUE(widget.m_scheduleSummaryLabel->text().startsWith("1200 new games"));

        widget.m_selectedVersions[0].resultsOnly = true;
        config = verify();
        ASSERT_EQ(config.models.size(), size_t(2));
        ASSERT_EQ(config.pairBudgets.size(), size_t(1));
        ASSERT_EQ(config.pairBudgets[0].games, 200);

        widget.m_selectedVersions[0].resultsOnly = false;
        widget.m_selectedVersions[0].included = false;
        config = verify();
        ASSERT_EQ(config.pairBudgets.size(), size_t(3));
        ASSERT_EQ(config.models.size(), size_t(3));

        widget.m_selectedVersions[0].included = true;
        for (ModelBenchmarkWidget::BenchmarkVersionEntry& entry : widget.m_selectedVersions)
            entry.primary = false;

        ASSERT_TRUE(widget.PrepareSchedule(SHOW_PREVIEW).pairBudgets.empty());
        ASSERT_EQ(widget.m_scheduleTable->rowCount(), 0);
        ASSERT_TRUE(widget.m_scheduleSummaryLabel->text().contains("Choose at least one Target"));

        for (ModelBenchmarkWidget::BenchmarkVersionEntry& entry : widget.m_selectedVersions)
            entry.primary = entry.libraryModelId == 3;

        config = verify();
        ASSERT_EQ(config.pairBudgets.size(), size_t(2));
        ASSERT_EQ(config.snapshotOffset, uint64_t(0));
        const QString targetName = QString::fromStdString(widget.MakeDisplayName(widget.m_selectedVersions[0]));
        for (int row = 0; row < widget.m_scheduleTable->rowCount(); ++row) {
            ASSERT_EQ(widget.m_scheduleTable->item(row, 0)->text(), targetName);
            if (row > 0)
                ASSERT_TRUE(QString::compare(widget.m_scheduleTable->item(row - 1, 1)->text(),
                    widget.m_scheduleTable->item(row, 1)->text(), Qt::CaseInsensitive) <= 0);
        }

        ASSERT_FALSE(widget.m_scheduleSummaryLabel->text().contains("saved games excluded by Compatibility"));
        widget.m_makeAllButton->menu()->actions()[0]->trigger();
        widget.m_mode1v1Check->setChecked(false);
        config = verify();
        ASSERT_EQ(config.pairBudgets.size(), size_t(1));
        ASSERT_EQ(config.models.size(), size_t(2));

        widget.m_scheduleModeCombo->setCurrentIndex(1);
        config = verify();
        ASSERT_EQ(config.pairBudgets.size(), size_t(2));
        ASSERT_TRUE(widget.m_scheduleSummaryLabel->text().startsWith("1000 new games"));
        ASSERT_EQ(widget.m_scheduleTable->rowCount(), 1);
        ASSERT_EQ(widget.m_scheduleTable->item(0, 2)->text(), QString("2v2, 3v3"));
        ASSERT_EQ(widget.m_scheduleTable->item(0, 4)->text(), QString("900"));
        ASSERT_EQ(widget.m_scheduleTable->item(0, 5)->text(), QString("1000"));

        widget.m_mode2v2Check->setChecked(false);
        widget.m_mode3v3Check->setChecked(false);
        ASSERT_TRUE(widget.PrepareSchedule(SHOW_PREVIEW).pairBudgets.empty());
        ASSERT_EQ(widget.m_scheduleSummaryLabel->text(), QString("Choose a game mode in Test Setup"));

        widget.m_loadingPersistedSetup = false;
        widget.OnSetupChanged();
        ASSERT_TRUE(widget.m_scheduleSummaryLabel->text().startsWith("Updating schedule"));
        ASSERT_FALSE(widget.m_runButton->isEnabled());
        widget.m_loadingPersistedSetup = true;
        widget.m_setupSaveTimer->stop();
        widget.m_workspaceRefreshTimer->stop();
        std::cout << "Roster changes, scoped coverage, eligibility, budgets, remapping, early stopping and stale-summary checks passed\n";
    }

    static void Workspace() {
        AgentRegistry::Register("FixtureA", [] {
            std::unique_ptr<Agent> agent = std::make_unique<Agent>();
            agent->maxTeamSize = 3;
            return agent;
        });
        QTemporaryDir directory;
        ModelBenchmarkWidget widget(nullptr, PERSIST_SETUP);
        widget.m_loadingPersistedSetup = true;
        ASSERT_TRUE(widget.m_savedResultsWidget->OpenLibrary(directory.filePath("workspace.sqlite").toStdString(), SHOW_ERRORS));
        ASSERT_TRUE(widget.m_testSetupDialog->isHidden());
        ASSERT_TRUE(widget.m_modelPickerDialog->isHidden());

        widget.m_presetName = "FixtureA release comparison";
        for (const char* version : { "7.34", "7.33", "7.32" }) {
            ModelBenchmarkWidget::BenchmarkVersionEntry entry;
            entry.agentName = "FixtureA";
            entry.agentSetupSource = RocketForgeTraining::ModelBenchmarkAgentSetupSource::CurrentAgent;
            entry.buildName = "FixtureA";
            entry.version.id = version;
            entry.version.latestTimesteps = 1234567890;
            if (entry.version.id == "7.34")
                entry.version.lastTrainedAt = "20260927-112233";

            entry.included = entry.version.id != "7.32";
            entry.primary = entry.version.id == "7.34";
            entry.resultsOnly = entry.version.id == "7.33";
            widget.m_selectedVersions.push_back(entry);
        }

        widget.PopulateSelectedTable();
        widget.UpdateWorkspaceSummary();
        RocketForgeTraining::ModelBenchmarkSettings settings = widget.BuildPersistedSetup();
        ASSERT_TRUE(settings.selectedModels[1].resultsOnly);
        ASSERT_EQ(settings.selectedModels[0].lastTrainedAt, std::string("20260927-112233"));
        ASSERT_FALSE(widget.m_selectedTable->isColumnHidden(6));
        ASSERT_FALSE(widget.m_selectedTable->isColumnHidden(7));
        ASSERT_EQ(widget.m_selectedTable->item(0, 7)->text(), QString("2026-09-27 11:22"));
        ASSERT_EQ(widget.m_selectedTable->item(1, 7)->text(), QString::fromUtf8("—"));
        ASSERT_TRUE(widget.m_selectedTable->item(0, 6)->toolTip().contains("1234567890"));
        ASSERT_FALSE(settings.selectedModels[2].included);
        ASSERT_FALSE(widget.BuildPersistedSetup().allMatchups);

        // Old presets must not override the visible roles when scheduling new games.
        settings.allMatchups = true;
        settings.ratingTargetsOnly = true;
        widget.ApplyPersistedSetup(settings);
        ASSERT_FALSE(widget.BuildPersistedSetup().allMatchups);
        ASSERT_TRUE(widget.BuildConfig().models[0].primary);

        // Show controls results visibility, not participation in the next run.
        ASSERT_EQ(widget.BuildConfig().models.size(), size_t(2));
        widget.m_selectedVersions[1].resultsOnly = false;
        ASSERT_FALSE(widget.BuildConfig().models[1].primary);
        widget.m_selectedVersions[1].resultsOnly = true;

        const std::vector<ModelBenchmarkWidget::BenchmarkVersionEntry> originalRoster = widget.m_selectedVersions;
        widget.SetRunning(true);
        ASSERT_FALSE(widget.m_makeAllButton->isEnabled());
        widget.m_makeAllButton->menu()->actions()[0]->trigger();
        ASSERT_TRUE(widget.m_selectedVersions[1].resultsOnly);
        widget.SetRunning(false);
        widget.m_makeAllButton->menu()->actions()[0]->trigger();
        for (int i = 0; i < (int)widget.m_selectedVersions.size(); ++i) {
            ASSERT_TRUE(widget.m_selectedVersions[i].primary);
            ASSERT_FALSE(widget.m_selectedVersions[i].resultsOnly);
            ASSERT_EQ(widget.m_selectedVersions[i].included, originalRoster[i].included);
            auto* targetRole = qobject_cast<QComboBox*>(widget.m_selectedTable->cellWidget(i, 1));
            ASSERT_EQ(targetRole->currentIndex(), 0);
        }

        ASSERT_EQ(widget.m_makeAllButton->menu()->actions().size(), 3);
        for (int roleIndex : { 1, 2 }) {
            widget.m_makeAllButton->menu()->actions()[roleIndex]->trigger();
            for (int i = 0; i < (int)widget.m_selectedVersions.size(); ++i) {
                ASSERT_FALSE(widget.m_selectedVersions[i].primary);
                ASSERT_EQ(widget.m_selectedVersions[i].resultsOnly, roleIndex == 2);
                ASSERT_EQ(widget.m_selectedVersions[i].included, originalRoster[i].included);
                auto* roleCombo = qobject_cast<QComboBox*>(widget.m_selectedTable->cellWidget(i, 1));
                ASSERT_EQ(roleCombo->currentIndex(), roleIndex);
            }
        }

        ASSERT_FALSE(widget.BuildPersistedSetup().ratingTargetsOnly);
        widget.m_selectedVersions = originalRoster;
        widget.PopulateSelectedTable();

        // Changing run targets cannot narrow the saved-game rating scope.
        auto* role = qobject_cast<QComboBox*>(widget.m_selectedTable->cellWidget(0, 1));
        ASSERT_TRUE(role != nullptr);
        role->setCurrentIndex(1);
        ASSERT_FALSE(widget.BuildPersistedSetup().ratingTargetsOnly);
        role->setCurrentIndex(0);

        widget.resize(1200, 800);
        widget.show();
        QApplication::processEvents();
        ASSERT_TRUE(widget.m_selectedTable->isVisible());
        ASSERT_TRUE(widget.m_savedResultsWidget->isVisible());
        ASSERT_FALSE(widget.m_versionTable->isVisible());

        const int originalTarget = widget.m_maxGamesSpin->value();
        QTimer::singleShot(0, &widget, [&]() {
            widget.m_maxGamesSpin->setValue(2345);
            widget.m_testSetupDialog->reject();
        });
        widget.ShowTestSetup();
        ASSERT_EQ(widget.m_maxGamesSpin->value(), originalTarget);
        widget.m_loadingPersistedSetup = true;
        widget.m_setupSaveTimer->stop();
        widget.m_workspaceRefreshTimer->stop();

        QTimer::singleShot(0, &widget, [&]() {
            widget.m_maxGamesSpin->setValue(1500);
            widget.m_testSetupDialog->accept();
        });
        widget.ShowTestSetup();
        ASSERT_EQ(widget.m_maxGamesSpin->value(), 1500);
        widget.m_loadingPersistedSetup = true;
        widget.m_setupSaveTimer->stop();
        widget.m_workspaceRefreshTimer->stop();

        // Coverage is useful even when old checkpoints are unavailable:
        // only missing pair-modes require a runnable model, and already-covered pairs stay unscheduled.
        widget.m_selectedVersions.resize(2);
        widget.m_selectedVersions[0].libraryModelId = 1;
        widget.m_selectedVersions[1].libraryModelId = 2;
        widget.m_selectedVersions[1].resultsOnly = false;
        widget.m_maxGamesSpin->setValue(1000);
        widget.m_coverage = { { 1, 2, 1, 650 }, { 1, 2, 2, 1000 }, { 1, 2, 3, 1250 } };

        // Known-simulator legacy coverage remains useful even without build metadata or runnable checkpoints.
        // Legacy compatibility markers do not invalidate it.
        widget.m_engineCombo->setCurrentIndex(widget.m_engineCombo->findData(static_cast<int>(RocketSim::EngineId::V2)));
        const auto compiled=RocketForgeTraining::DescribeModelBenchmarkEnvironment(widget.BuildConfig());
        for (auto& pair : widget.m_coverage) {
            pair.engineVersion = static_cast<int>(RocketSim::EngineId::V2);
            pair.compatibilityVersion = 1;
        }

        bool needsCheckpoint = false;
        try {
            widget.PrepareSchedule(SHOW_PREVIEW, REQUIRE_RUNNABLE);
        } catch (const std::exception&) {
            needsCheckpoint = true;
        }

        ASSERT_TRUE(needsCheckpoint);
        ASSERT_EQ(widget.m_scheduleTable->rowCount(), 2);
        ASSERT_EQ(widget.m_scheduleTable->item(0, 5)->text(), QString("350"));
        ASSERT_EQ(widget.m_scheduleTable->item(1, 5)->text(), QString("0"));
        ASSERT_EQ(widget.m_scheduleTable->item(1, 2)->text(), QString("2v2, 3v3"));
        ASSERT_EQ(widget.m_scheduleTable->item(1, 4)->text(), QString("2250"));

        // Other simulators for the same pair must neither erase matching coverage nor fill its remaining budget, regardless of row order.
        ModelBenchmarkLibraryCoverage otherEnvironment = widget.m_coverage[0];
        otherEnvironment.games = 9000;
        otherEnvironment.engineVersion = static_cast<int>(RocketSim::EngineId::V2) == 2 ? 3 : 2;
        widget.m_coverage.push_back(otherEnvironment);
        needsCheckpoint = false;
        try {
            widget.PrepareSchedule(SHOW_PREVIEW, REQUIRE_RUNNABLE);
        } catch (const std::exception&) {
            needsCheckpoint = true;
        }

        ASSERT_TRUE(needsCheckpoint);
        ASSERT_EQ(widget.m_scheduleTable->item(0, 4)->text(), QString("650"));
        ASSERT_EQ(widget.m_scheduleTable->item(0, 5)->text(), QString("350"));

        widget.m_coverage.pop_back();
        widget.m_coverage[0].games = 1000;
        ASSERT_TRUE(widget.PrepareSchedule(SHOW_PREVIEW, REQUIRE_RUNNABLE).pairBudgets.empty());
        ASSERT_EQ(widget.m_scheduleTable->rowCount(), 1);
        ASSERT_EQ(widget.m_scheduleTable->item(0, 2)->text(), QString("1v1, 2v2, 3v3"));
        ASSERT_EQ(widget.m_scheduleTable->item(0, 4)->text(), QString("3250"));
        ASSERT_TRUE(widget.m_scheduleSummaryLabel->text().contains("3250 covered games"));
        ASSERT_TRUE(widget.m_scheduleSummaryLabel->text().contains("3 covered matchups"));

        // Different exact revisions (or none) still count within version 1.
        widget.m_coverage[0].evaluationIdentity = QString::fromStdString(compiled.evaluationIdentity);
        ASSERT_TRUE(widget.PrepareSchedule(SHOW_PREVIEW, REQUIRE_RUNNABLE).pairBudgets.empty());
        widget.m_coverage[0].evaluationIdentity.clear();
        ASSERT_TRUE(widget.PrepareSchedule(SHOW_PREVIEW, REQUIRE_RUNNABLE).pairBudgets.empty());
        ASSERT_FALSE(widget.m_scheduleSummaryLabel->text().contains("could not be verified"));
        widget.m_coverage[0].compatibilityVersion = 2;
        ASSERT_TRUE(widget.PrepareSchedule(SHOW_PREVIEW, REQUIRE_RUNNABLE).pairBudgets.empty());
        widget.m_coverage[0].compatibilityVersion = 0;
        ASSERT_TRUE(widget.PrepareSchedule(SHOW_PREVIEW, REQUIRE_RUNNABLE).pairBudgets.empty());

        widget.m_mode2v2Check->setChecked(false);
        widget.m_mode3v3Check->setChecked(false);
        widget.m_coverage[0].engineVersion = 0;
        widget.m_engineCombo->setCurrentIndex(widget.m_engineCombo->findData(2));
        ASSERT_TRUE(widget.PrepareSchedule(SHOW_PREVIEW, REQUIRE_RUNNABLE).pairBudgets.empty());
        ASSERT_EQ(widget.m_reuseEngineChecks.size(), size_t(2));
        ASSERT_TRUE(widget.m_testSetupDialog->findChild<QCheckBox*>("benchmarkReuseEngine0") == nullptr);
        widget.m_engineCombo->setCurrentIndex(widget.m_engineCombo->findData(3));
        needsCheckpoint = false;
        try {
            widget.PrepareSchedule(SHOW_PREVIEW, REQUIRE_RUNNABLE);
        } catch (const std::exception&) {
            needsCheckpoint = true;
        }

        ASSERT_TRUE(needsCheckpoint);
        ASSERT_EQ(widget.m_scheduleTable->item(0, 5)->text(), QString("1000"));
        widget.m_engineCombo->setCurrentIndex(widget.m_engineCombo->findData(static_cast<int>(RocketSim::EngineId::V2)));
        widget.m_reuseSelectedEngineCheck->setChecked(true);
        widget.m_coverage[0].engineVersion = static_cast<int>(RocketSim::EngineId::V2);
        widget.m_mode2v2Check->setChecked(true);
        widget.m_mode3v3Check->setChecked(true);
        // A covered alternate-simulator roster needs no environment query.
        const int alternateEngine = static_cast<int>(RocketSim::EngineId::V2) == 2 ? 3 : 2;
        widget.m_engineCombo->setCurrentIndex(widget.m_engineCombo->findData(alternateEngine));
        for (ModelBenchmarkLibraryCoverage& pair : widget.m_coverage)
            pair.engineVersion = alternateEngine;

        ASSERT_TRUE(widget.PrepareSchedule(SHOW_PREVIEW, REQUIRE_RUNNABLE).pairBudgets.empty());
        ASSERT_FALSE(widget.m_scheduleSummaryLabel->text().contains("could not be verified"));
        widget.m_engineCombo->setCurrentIndex(widget.m_engineCombo->findData(static_cast<int>(RocketSim::EngineId::V2)));
        for (auto& pair : widget.m_coverage) pair.engineVersion = static_cast<int>(RocketSim::EngineId::V2);
        widget.m_selectedVersions[1].included = false;

        ModelBenchmarkLibraryModel duplicate;
        duplicate.id = 2;
        widget.AddSavedModel(duplicate);
        ASSERT_EQ(widget.m_selectedVersions.size(), size_t(2));
        ASSERT_FALSE(widget.m_selectedVersions[1].included);
        widget.m_selectedVersions[1].included = true;

        const bool saveAs = false;
        widget.SaveWorkspacePreset(saveAs);
        const std::vector<ModelBenchmarkWorkspaceRecord> presets = widget.m_savedResultsWidget->Library().ListPresets();
        ASSERT_EQ(presets.size(), size_t(1));
        RocketForgeTraining::ModelBenchmarkSettingsLoadResult restored =
            RocketForgeTraining::DeserializeModelBenchmarkSettings(presets[0].settingsJson.toStdString());
        ASSERT_TRUE(restored.ok);
        ASSERT_EQ(restored.value.maxGamesPerMode, 1000);
        ASSERT_FALSE(restored.value.allMatchups);

        widget.PopulateSelectedTable();
        widget.UpdateWorkspaceSummary();
        const QString renderDirectory = qEnvironmentVariable("FORGE_BENCHMARK_RENDER_DIR");
        if (!renderDirectory.isEmpty()) {
            QDir().mkpath(renderDirectory);
            for (const QSize& size : { QSize(1200, 800), QSize(1000, 700) }) {
                widget.resize(size);
                QApplication::processEvents();
                ASSERT_TRUE(widget.width() <= size.width());
                ASSERT_TRUE(widget.grab().save(QDir(renderDirectory).filePath(
                    QString("workspace-%1x%2.png").arg(size.width()).arg(size.height()))));
            }

            widget.m_testSetupDialog->show();
            QApplication::processEvents();
            ASSERT_TRUE(widget.m_testSetupDialog->grab().save(QDir(renderDirectory).filePath("test-setup.png")));
            widget.m_testSetupDialog->hide();
            widget.SetRunning(true);
            QApplication::processEvents();
            ASSERT_FALSE(widget.m_hasLastResult);
            ASSERT_TRUE(widget.m_savedResultsWidget->isVisible());
            ASSERT_TRUE(widget.grab().save(QDir(renderDirectory).filePath("combined-running-1000x700.png")));
            widget.SetRunning(false);
        }
    }

    static void Run() {
        // Keep persistence and snapshots isolated from the user's benchmark library.
        QTemporaryDir directory;
        ModelBenchmarkWidget widget(nullptr, PERSIST_SETUP);
        widget.m_loadingPersistedSetup = true;
        ASSERT_TRUE(widget.m_savedResultsWidget->OpenLibrary(directory.filePath("picker.sqlite").toStdString(), SHOW_ERRORS));
        {
            QSignalBlocker blocker(widget.m_agentCombo);
            widget.m_agentCombo->clear();
            widget.m_agentCombo->addItems({ "Picker A", "Picker B" });
        }

        RocketForgeTraining::ModelBenchmarkResult drawResult;
        drawResult.rankings.resize(1);
        drawResult.rankings[0].totalRecord = { 2, 3, 7 };
        drawResult.pairwise.resize(1);
        drawResult.pairwise[0].combined = { 2, 3, 7 };
        drawResult.pairwise[0].kickoff = { 1, 1, 0 };
        drawResult.pairwise[0].postKickoff = { 1, 2, 0 };

        QTableWidget rankingTable;
        QTableWidget pairwiseTable;
        ForgeGUI::ModelBenchmarkTables::PopulateRankingTable(&rankingTable, drawResult);
        ForgeGUI::ModelBenchmarkTables::PopulatePairwiseTable(&pairwiseTable, drawResult);
        drawResult.rankings[0].betterThanSummary = "Opponent (95.0%)";
        QTableWidgetItem* liveCell = rankingTable.item(0, 8);
        const bool refreshLayout = false;
        ForgeGUI::ModelBenchmarkTables::PopulateRankingTable(&rankingTable, drawResult, refreshLayout);
        ASSERT_TRUE(rankingTable.item(0, 8) == liveCell);
        ASSERT_EQ(liveCell->text(), QString("Opponent (95.0%)"));
        ASSERT_EQ(rankingTable.horizontalScrollBarPolicy(), Qt::ScrollBarAsNeeded);
        ASSERT_EQ(rankingTable.columnCount(), 9);
        ASSERT_EQ(rankingTable.horizontalHeaderItem(8)->text(), QString("Better Than"));
        ASSERT_EQ(rankingTable.item(0, 7)->text(), QString("2-3 (40.0%)"));
        ASSERT_EQ(pairwiseTable.columnCount(), 6);
        ASSERT_EQ(pairwiseTable.item(0, 3)->text(), QString("2/3/7 (40.0%)"));
        ASSERT_EQ(pairwiseTable.item(0, 4)->text(), QString("1/1/0 (50.0%)"));
        ASSERT_EQ(pairwiseTable.item(0, 5)->text(), QString("1/2/0 (33.3%)"));

        auto entry = [](const char* build, const char* stage, const char* id) {
            ModelBenchmarkWidget::BenchmarkVersionEntry result;
            result.agentName = "Picker A";
            result.buildName = build;
            result.stageName = stage;
            result.sourceLabel = "RocketForge";
            result.version.id = id;
            result.version.modelCount = 1;
            result.version.notes = "aerial control";
            result.checkpointPath = std::filesystem::path("nonexistent-picker-fixture") / build / id;
            return result;
        };

        ModelBenchmarkWidget::BenchmarkVersionEntry first = entry("Build A", "Stage A", "v1.0");
        ModelBenchmarkWidget::BenchmarkVersionEntry second = entry("Build B", "Stage B", "v2.0");
        second.version.branchName = "experiment";
        ModelBenchmarkWidget::BenchmarkVersionEntry missing = entry("Build A", "Stage A", "v3.0");
        missing.version.modelCount = 0;
        missing.checkpointPath.clear();
        ModelBenchmarkWidget::BenchmarkVersionEntry imported = entry("Imported", "", "imported");
        imported.sourceLabel = "Imported Model";

        widget.m_versionCatalogs["Picker A"] = { { "Build A", "Build B" }, { first, second, missing, imported } };
        widget.m_versionCatalogs["Picker B"] = {};
        widget.ShowVersionCatalog();
        ASSERT_EQ(widget.m_versionTable->rowCount(), 4);
        ASSERT_EQ(widget.m_buildCombo->count(), 3);
        ASSERT_EQ(widget.m_stageCombo->count(), 3);

        auto rowFor = [&](const std::string& id) {
            for (int row = 0; row < (int)widget.m_availableVersions.size(); ++row) {
                if (widget.m_availableVersions[row].version.id == id)
                    return row;
            }

            return -1;
        };
        QTableWidgetItem* originalItem = widget.m_versionTable->item(rowFor("v1.0"), 4);
        widget.m_buildCombo->setCurrentText("Build A");
        widget.RefreshStageSelector();
        widget.ApplyAvailableRowFilters();
        ASSERT_EQ(widget.m_stageCombo->count(), 2);
        ASSERT_TRUE(widget.m_versionTable->isRowHidden(rowFor("v2.0")));
        ASSERT_FALSE(widget.m_versionTable->isRowHidden(rowFor("imported")));
        widget.m_hideImportedModelsCheck->setChecked(true);
        ASSERT_TRUE(widget.m_versionTable->isRowHidden(rowFor("imported")));
        widget.m_availableSearchEdit->setText("v1.0 aerial");
        ASSERT_FALSE(widget.m_versionTable->isRowHidden(rowFor("v1.0")));
        ASSERT_TRUE(widget.m_versionTable->isRowHidden(rowFor("v3.0")));
        ASSERT_EQ(widget.m_versionTable->item(rowFor("v1.0"), 4), originalItem);
        ASSERT_FALSE(widget.m_catalogScan.valid());

        widget.m_loadingPersistedSetup = false;
        widget.m_loadedAgentName = "Picker A";
        widget.m_buildCombo->setCurrentText("Build B");
        ASSERT_EQ(widget.m_stageCombo->count(), 2);
        ASSERT_EQ(widget.m_stageCombo->itemText(1), QString("Stage B"));
        ASSERT_FALSE(widget.m_catalogScan.valid());
        ASSERT_EQ(widget.m_versionTable->item(rowFor("v1.0"), 4), originalItem);
        widget.m_buildCombo->setCurrentText("Build A");
        widget.m_setupSaveTimer->stop();
        widget.m_loadingPersistedSetup = true;

        // Exercise actual table mouse events, including the checkbox delegate's signal order.
        widget.resize(1600, 1000);
        widget.show();
        widget.m_modelPickerDialog->show();
        QApplication::processEvents();
        auto clickCell = [&](int row, int column) {
            QTableWidget* table = widget.m_versionTable;
            table->scrollToItem(table->item(row, column));
            const QPointF local = table->visualItemRect(table->item(row, column)).center();
            const QPointF global = table->viewport()->mapToGlobal(local.toPoint());
            QMouseEvent press(QEvent::MouseButtonPress, local, global, Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
            QMouseEvent release(QEvent::MouseButtonRelease, local, global, Qt::LeftButton, Qt::NoButton, Qt::NoModifier);
            QApplication::sendEvent(table->viewport(), &press);
            QApplication::sendEvent(table->viewport(), &release);
            QApplication::processEvents();
        };

        clickCell(rowFor("v1.0"), 4);
        ASSERT_EQ(widget.m_selectedVersions.size(), size_t(1));
        ASSERT_EQ(widget.m_versionTable->item(rowFor("v1.0"), 0)->checkState(), Qt::Checked);
        clickCell(rowFor("v1.0"), 4);
        ASSERT_EQ(widget.m_selectedVersions.size(), size_t(1));
        ASSERT_EQ(widget.m_versionTable->item(rowFor("v1.0"), 4), originalItem);

        widget.OnClearSelectedClicked();
        clickCell(rowFor("v1.0"), 0);
        ASSERT_EQ(widget.m_selectedVersions.size(), size_t(1));

        widget.OnClearSelectedClicked();
        widget.m_availableSearchEdit->clear();
        widget.OnAddVersionsClicked();
        ASSERT_EQ(widget.m_selectedVersions.size(), size_t(1));
        ASSERT_EQ(widget.m_selectedVersions.front().version.id, std::string("v1.0"));
        widget.AddAvailableVersion(rowFor("v3.0"));
        ASSERT_EQ(widget.m_selectedVersions.size(), size_t(1));
        widget.SetRunning(true);
        widget.AddAvailableVersion(rowFor("imported"));
        widget.OnClearSelectedClicked();
        ASSERT_EQ(widget.m_selectedVersions.size(), size_t(1));
        widget.SetRunning(false);

        widget.m_buildCombo->setCurrentIndex(0);
        widget.RefreshStageSelector();
        widget.m_availableBranchCombo->setCurrentText("experiment");
        ASSERT_FALSE(widget.m_versionTable->isRowHidden(rowFor("v2.0")));
        ASSERT_TRUE(widget.m_versionTable->isRowHidden(rowFor("v1.0")));

        // Completion for a previously selected agent must never replace the new agent's rows.
        std::promise<ModelBenchmarkWidget::VersionCatalog> completed;
        widget.m_catalogScan = completed.get_future();
        widget.m_scanningAgent = "Picker A";
        completed.set_value(widget.m_versionCatalogs.at("Picker A"));
        widget.m_agentCombo->setCurrentText("Picker B");
        widget.FinishVersionCatalogScan();
        ASSERT_EQ(widget.m_displayedCatalogAgent, QString("Picker B"));
        ASSERT_EQ(widget.m_versionTable->rowCount(), 0);
        ASSERT_EQ(widget.m_selectedVersions.size(), size_t(1));
        ASSERT_FALSE(widget.m_catalogScan.valid());

        widget.m_agentCombo->setCurrentText("Picker A");
        widget.m_pendingStageFilter = "Stage B";
        widget.EnsureVersionCatalog();
        ASSERT_EQ(widget.m_stageCombo->currentText(), QString("Stage B"));
        ASSERT_EQ(widget.m_versionTable->rowCount(), 4);
        ASSERT_FALSE(widget.m_catalogScan.valid());
        ASSERT_FALSE(widget.m_setupSaveTimer->isActive());

        ModelBenchmarkWidget::BenchmarkVersionEntry legacyImported = imported;
        legacyImported.stageName = "Stage A";
        ASSERT_EQ(widget.MakeEntryKey(imported), widget.MakeEntryKey(legacyImported));

        std::promise<ModelBenchmarkWidget::VersionCatalog> failed;
        widget.m_catalogScan = failed.get_future();
        widget.m_scanningAgent = "Picker A";
        failed.set_exception(std::make_exception_ptr(std::runtime_error("Synthetic discovery failure")));
        widget.FinishVersionCatalogScan();
        ASSERT_EQ(widget.m_versionTable->rowCount(), 4);
        ASSERT_EQ(widget.m_selectedVersions.size(), size_t(1));
        ASSERT_TRUE(widget.m_refreshVersionsButton->isEnabled());

        // Measure filtering on a larger synthetic catalog without loading actual models.
        widget.m_availableVersions.clear();
        for (int i = 0; i < 2000; ++i) {
            ModelBenchmarkWidget::BenchmarkVersionEntry model = i % 2 == 0 ? first : second;
            model.version.id = "v" + std::to_string(i) + ".0";
            widget.m_availableVersions.push_back(std::move(model));
        }

        widget.m_buildCombo->setCurrentIndex(0);
        widget.m_stageCombo->setCurrentIndex(0);
        widget.m_availableBranchCombo->setCurrentIndex(0);
        const bool preserveColumnWidths = true;
        widget.PopulateAvailableTable(preserveColumnWidths);

        QElapsedTimer elapsed;
        elapsed.start();
        for (int i = 0; i < 100; ++i)
            widget.m_availableSearchEdit->setText(i % 2 == 0 ? "aerial Build A" : "aerial Build B");

        std::cout << "2000 synthetic models: " << elapsed.elapsed() / 100.0 << " ms per search filter\n";
        ASSERT_FALSE(widget.m_catalogScan.valid());
        std::cout << "Picker filtering, row/checkbox clicks, bulk add, duplicates, missing checkpoints, run lock, and stale discovery checks passed\n";
    }
};

int main(int argc, char** argv) {
#ifdef Q_OS_WIN
    if (qEnvironmentVariable("QT_QPA_PLATFORM") == "offscreen" && qEnvironmentVariableIsEmpty("QT_QPA_FONTDIR"))
        qputenv("QT_QPA_FONTDIR", qgetenv("WINDIR") + "/Fonts");
#endif
    QApplication app(argc, argv);
    QFile theme(":/theme.qss");
    if (theme.open(QIODevice::ReadOnly))
        app.setStyleSheet(QString::fromUtf8(theme.readAll()));

    try {
        ModelBenchmarkPickerTests::PersistenceDisabled();
        ModelBenchmarkPickerTests::KickoffSettings();
        if (app.arguments().contains("--completed-history")) {
            ModelBenchmarkPickerTests::RoleUnlock();
            ModelBenchmarkPickerTests::PublicModelHistory();
            return 0;
        }

        ModelBenchmarkPickerTests::CatalogPreservesAutosavesAndSimulation();
        ModelBenchmarkPickerTests::WorkerCoalescesCompleteLiveSnapshots();
        ModelBenchmarkPickerTests::StoppedRunsPreserveGames();
        if (app.arguments().contains("--stop-preservation"))
            return 0;

        if (app.arguments().contains("--compatibility")) {
            ModelBenchmarkPickerTests::Compatibility();
            ModelBenchmarkPickerTests::Schedule();
            return 0;
        }

        if (app.arguments().contains("--workspace")) {
            ModelBenchmarkPickerTests::FixedRunSettings();
            ModelBenchmarkPickerTests::Workspace();
            ModelBenchmarkPickerTests::AsynchronousWorkspace();
            return 0;
        }

        if (app.arguments().contains("--schedule")) {
            ModelBenchmarkPickerTests::Schedule();
            return 0;
        }

        if (app.arguments().contains("--new-preset")) {
            ModelBenchmarkPickerTests::NewPreset();
            return 0;
        }

        if (app.arguments().contains("--rename-preset")) {
            ModelBenchmarkPickerTests::RenamePreset();
            return 0;
        }

        if (app.arguments().contains("--comparison-sort")) {
            ModelBenchmarkPickerTests::ComparisonSort();
            return 0;
        }

        ModelBenchmarkPickerTests::ComparisonSort();

        if (app.arguments().contains("--previous-run")) {
            ModelBenchmarkPickerTests::PreviousRunPicker();
            return 0;
        }

        ModelBenchmarkPickerTests::RenamePreset();
        ModelBenchmarkPickerTests::NewPreset();
        ModelBenchmarkPickerTests::Compatibility();
        ModelBenchmarkPickerTests::PresetSwitching();
        ModelBenchmarkPickerTests::PreviousRunPicker();
        ModelBenchmarkPickerTests::FixedRunSettings();
        ModelBenchmarkPickerTests::Run();
        ModelBenchmarkPickerTests::Workspace();
        ModelBenchmarkPickerTests::Schedule();
        ModelBenchmarkPickerTests::AsynchronousWorkspace();
        ModelBenchmarkPickerTests::RoleUnlock();
        ModelBenchmarkPickerTests::PublicModelHistory();
        return 0;
    } catch (const RocketForgeTests::AssertionFailure& error) {
        std::cerr << error.message << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
    }

    return 1;
}
