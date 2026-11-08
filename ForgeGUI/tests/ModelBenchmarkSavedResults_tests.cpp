#include "test_runner.h"
#include <ModelBenchmark/ModelBenchmarkEngine.h>
#include "ModelBenchmark/ModelBenchmarkSavedResultsWidget.h"
#include "ModelBenchmark/ModelBenchmarkResultTables.h"

#include <QApplication>
#include <QComboBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QElapsedTimer>
#include <QFile>
#include <QInputDialog>
#include <QHeaderView>
#include <QScrollBar>
#include <QLabel>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPixmap>
#include <QPushButton>
#include <QTableWidget>
#include <QTabWidget>
#include <QTemporaryDir>
#include <QThread>
#include <QTimer>

#include <algorithm>
#include <fstream>
#include <iostream>

using namespace RocketForgeTraining;

class ModelBenchmarkSavedResultsTests
{
    static void ClickCell(QTableWidget* table, int row, int column)
    {
        table->scrollToItem(table->item(row, column));
        const QPointF local = table->visualItemRect(table->item(row, column)).center();
        const QPointF global = table->viewport()->mapToGlobal(local.toPoint());
        QMouseEvent press(QEvent::MouseButtonPress, local, global, Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
        QMouseEvent release(QEvent::MouseButtonRelease, local, global, Qt::LeftButton, Qt::NoButton, Qt::NoModifier);
        QApplication::sendEvent(table->viewport(), &press);
        QApplication::sendEvent(table->viewport(), &release);
        QApplication::processEvents();
    }

    static void NameNextSave(const QString& name)
    {
        QTimer::singleShot(0, [name]() {
            auto* dialog = qobject_cast<QInputDialog*>(QApplication::activeModalWidget());
            ASSERT_TRUE(dialog);
            dialog->setTextValue(name);
            dialog->accept();
        });
    }

    static void AnswerNextPrompt(QMessageBox::StandardButton answer)
    {
        QTimer::singleShot(0, [answer]() {
            auto* dialog = qobject_cast<QMessageBox*>(QApplication::activeModalWidget());
            ASSERT_TRUE(dialog);
            dialog->button(answer)->click();
        });
    }

    static void WaitForResults(ModelBenchmarkSavedResultsWidget& widget)
    {
        const qint64 timeoutMs = 10000;
        const qint64 settleMs = 50;
        QElapsedTimer timer;
        timer.start();
        QElapsedTimer idle;
        idle.start();

        for (;;) {
            ASSERT_TRUE(timer.elapsed() < timeoutMs);
            QApplication::processEvents();

            // A completed worker can queue a zero-delay refresh; drain that callback before deciding that aggregation is idle.
            QApplication::processEvents();
            if (widget.m_aggregateThread || widget.m_aggregateRefreshPending
                || widget.m_selectionRefreshQueued || widget.m_resultsRefreshQueued
                || widget.m_fingerprintThread || widget.m_fingerprintRefreshPending
                || widget.m_liveRefreshQueued || widget.m_liveRefreshPending)
                idle.restart();
            else if (idle.elapsed() >= settleMs)
                break;

            QThread::msleep(1);
        }
    }

public:

    static void LiveResultCellsNotifyViews()
    {
        QTableWidget table;
        ModelBenchmarkResult result;
        result.overallEloAvailable = true;
        ModelBenchmarkRankingRow row;
        row.modelIndex = 0;
        row.modelName = "First bot";
        row.rank = 1;
        row.betterThanSummary = "Second bot (60.0%)";
        result.rankings.push_back(row);
        ForgeGUI::ModelBenchmarkTables::PopulateRankingTable(&table, result);

        QTableWidgetItem* nameItem = table.item(0, 1);
        QTableWidgetItem* confidenceItem = table.item(0, 8);
        std::set<int> notifiedColumns;
        QObject::connect(table.model(), &QAbstractItemModel::dataChanged, &table,
            [&](const QModelIndex& first, const QModelIndex& last) {
                for (int column = first.column(); column <= last.column(); ++column)
                    notifiedColumns.insert(column);
            });

        result.rankings[0].modelName = "Second bot";
        result.rankings[0].betterThanSummary = "First bot (80.0%)";
        const bool refreshLayout = false;
        ForgeGUI::ModelBenchmarkTables::PopulateRankingTable(&table, result, refreshLayout);
        ASSERT_TRUE(table.item(0, 1) == nameItem);
        ASSERT_TRUE(table.item(0, 8) == confidenceItem);
        ASSERT_EQ(nameItem->text(), QString("Second bot"));
        ASSERT_EQ(confidenceItem->text(), QString("First bot (80.0%)"));

        // Reading new values is insufficient:
        // the view must be told to repaint cells whose identity, text, or mention colors changed in place.
        ASSERT_TRUE(notifiedColumns.contains(8));
        ASSERT_TRUE(notifiedColumns.contains(1));
        std::cout << "Live model names and Better Than values notify the view after in-place updates\n";
    }

    static void RankChangesKeepModelDataTogether()
    {
        ModelBenchmarkSavedResultsWidget widget;
        widget.SetWorkspaceMode();
        widget.setAttribute(Qt::WA_DontShowOnScreen);
        widget.resize(1600, 700);
        widget.show();
        widget.m_liveComparison = true;

        ModelBenchmarkLibraryAggregate aggregate;
        aggregate.liveGames = 1;
        ModelBenchmarkResult& result = aggregate.result;
        result.overallEloAvailable = true;
        for (int model = 0; model < 12; ++model)
            result.modelNames.push_back("Bot / v" + std::to_string(model));

        for (int update = 0; update < 60; ++update) {
            result.rankings.clear();
            result.breakdown.clear();
            result.pairwise.clear();
            result.optionalStats.clear();

            // Visit every position with different values on each tick.
            // Auxiliary rows deliberately arrive in a different order from the ranking.
            for (int rank = 0; rank < 12; ++rank) {
                const int model = (rank * 5 + update) % 12;
                const int signature = model * 100 + update;

                ModelBenchmarkRankingRow ranking;
                ranking.modelIndex = model;
                ranking.modelName = result.modelNames[model];
                ranking.rank = rank + 1;
                ranking.relativeElo = 1000 - rank * 13 + update;
                ranking.rangeLow = ranking.relativeElo - 3;
                ranking.rangeHigh = ranking.relativeElo + 4;
                ranking.elo1v1 = signature + 1;
                ranking.elo2v2 = signature + 2;
                ranking.elo3v3 = signature + 3;
                ranking.elo1v1Available = ranking.elo2v2Available = ranking.elo3v3Available = true;
                ranking.totalRecord = { signature + 4, signature + 5, 0 };
                ranking.betterThanSummary = result.modelNames[(model + 1) % 12] + " (75.0%)";
                result.rankings.push_back(ranking);

                ModelBenchmarkBreakdownRow breakdown;
                breakdown.modelIndex = model;
                breakdown.modelName = ranking.modelName;
                breakdown.combinedElo = ranking.relativeElo;
                breakdown.kickoffGames = signature + 6;
                breakdown.postKickoffGames = signature + 7;
                result.breakdown.insert(result.breakdown.begin(), breakdown);

                ModelBenchmarkPairwiseRow pairwise;
                pairwise.modelIndex = model;
                pairwise.opponentIndex = (model + 1) % 12;
                pairwise.modelName = ranking.modelName;
                pairwise.opponentName = result.modelNames[pairwise.opponentIndex];
                pairwise.modeName = "1v1";
                pairwise.combined = ranking.totalRecord;
                result.pairwise.insert(result.pairwise.begin(), pairwise);

                ModelBenchmarkStatRow stats;
                stats.modelIndex = model;
                stats.modelName = ranking.modelName;
                stats.stats.goals = signature + 8;
                stats.stats.shots = signature + 9;
                result.optionalStats.insert(result.optionalStats.begin(), stats);
            }

            // Alternate visible and hidden updates, then open every affected tab.
            widget.m_resultTabs->setCurrentIndex(update % 4);
            widget.PopulateResults(aggregate);
            widget.m_resultTabs->setCurrentWidget(widget.m_rankingTable);
            QApplication::processEvents();
            for (int row = 0; row < 12; ++row) {
                const int model = (row * 5 + update) % 12;
                const int signature = model * 100 + update;
                const QTableWidget* table = widget.m_rankingTable;
                ASSERT_EQ(table->item(row, 0)->text(), QString::number(row + 1));
                ASSERT_EQ(table->item(row, 1)->text(), QString::fromStdString(result.modelNames[model]));
                ASSERT_EQ(table->item(row, 2)->text(), QString("+%1").arg(1000 - row * 13 + update));
                ASSERT_EQ(table->item(row, 3)->text(), QString("[+%1, +%2]").arg(997 - row * 13 + update).arg(1004 - row * 13 + update));
                for (int mode = 1; mode <= 3; ++mode)
                    ASSERT_EQ(table->item(row, 3 + mode)->text(), QString("+%1").arg(signature + mode));
                ASSERT_EQ(table->item(row, 7)->text(), QString("%1-%2 (%3%)")
                    .arg(signature + 4).arg(signature + 5)
                    .arg(100.0 * (signature + 4) / (2 * signature + 9), 0, 'f', 1));
                ASSERT_EQ(table->item(row, 8)->text(), QString::fromStdString(result.modelNames[(model + 1) % 12]) + " (75.0%)");
            }

            widget.m_resultTabs->setCurrentWidget(widget.m_breakdownTable);
            for (int row = 0; row < 12; ++row) {
                const int model = (row * 5 + update) % 12;
                ASSERT_EQ(widget.m_breakdownTable->item(row, 0)->text(), QString::fromStdString(result.modelNames[model]));
                ASSERT_EQ(widget.m_breakdownTable->item(row, 1)->text(), QString("+%1").arg(1000 - row * 13 + update));
                ASSERT_EQ(widget.m_breakdownTable->item(row, 3)->text(), QString::number(model * 100 + update + 6));
                ASSERT_EQ(widget.m_breakdownTable->item(row, 5)->text(), QString::number(model * 100 + update + 7));
            }

            widget.m_resultTabs->setCurrentWidget(widget.m_pairwiseTable);
            for (int row = 0; row < 12; ++row) {
                const int model = (row * 5 + update) % 12;
                const int signature = model * 100 + update;
                ASSERT_EQ(widget.m_pairwiseTable->item(row, 0)->text(), QString::fromStdString(result.modelNames[model]));
                ASSERT_EQ(widget.m_pairwiseTable->item(row, 1)->text(), QString::fromStdString(result.modelNames[(model + 1) % 12]));
                ASSERT_EQ(widget.m_pairwiseTable->item(row, 3)->text(), QString("%1/%2/0 (%3%)")
                    .arg(signature + 4).arg(signature + 5)
                    .arg(100.0 * (signature + 4) / (2 * signature + 9), 0, 'f', 1));
            }

            widget.m_resultTabs->setCurrentIndex(3);
            for (int row = 0; row < 12; ++row) {
                const int model = (row * 5 + update) % 12;
                ASSERT_EQ(widget.m_statsTable->item(row, 0)->text(), QString::fromStdString(result.modelNames[model]));
                ASSERT_EQ(widget.m_statsTable->item(row, 2)->text(), QString::number(model * 100 + update + 8, 'f', 2));
                ASSERT_EQ(widget.m_statsTable->item(row, 3)->text(), QString::number(model * 100 + update + 9, 'f', 2));
            }
        }

        std::cout << "Names, Elo, records, opponents, and statistics stay together across 60 live rank permutations\n";
    }

    static void LiveTabRefresh()
    {
        ModelBenchmarkSavedResultsWidget widget;
        widget.SetWorkspaceMode();
        widget.setAttribute(Qt::WA_DontShowOnScreen);
        widget.resize(1600, 500);
        widget.show();
        widget.m_liveComparison = true;

        ModelBenchmarkLibraryAggregate aggregate;
        aggregate.liveGames = 1;
        ModelBenchmarkResult& result = aggregate.result;
        result.overallEloAvailable = true;
        for (int i = 0; i < 12; ++i) {
            const std::string name = "Model " + std::to_string(i);
            result.modelNames.push_back(name);

            ModelBenchmarkRankingRow ranking;
            ranking.modelIndex = i;
            ranking.rank = i + 1;
            ranking.modelName = name;
            result.rankings.push_back(ranking);

            ModelBenchmarkBreakdownRow breakdown;
            breakdown.modelIndex = i;
            breakdown.modelName = name;
            result.breakdown.push_back(breakdown);

            ModelBenchmarkStatRow stats;
            stats.modelIndex = i;
            stats.modelName = name;
            result.optionalStats.push_back(stats);
        }

        widget.PopulateResults(aggregate);
        widget.m_resultTabs->setCurrentWidget(widget.m_breakdownTable);
        QApplication::processEvents();

        int hiddenChanges = 0;
        const QMetaObject::Connection connection = QObject::connect(widget.m_rankingTable->model(),
            &QAbstractItemModel::dataChanged, &widget, [&]() { ++hiddenChanges; });
        for (int update = 1; update <= 40; ++update) {
            result.rankings[0].relativeElo = update;
            result.rankings[0].betterThanSummary.assign(update * 5, 'x');
            result.breakdown[0].kickoffGames = update;
            widget.PopulateResults(aggregate);
            ASSERT_EQ(widget.m_breakdownTable->item(0, 3)->text(), QString::number(update));

            bool heartbeat = false;
            QTimer::singleShot(0, &widget, [&]() { heartbeat = true; });
            QApplication::processEvents();
            ASSERT_TRUE(heartbeat);
        }

        // Hidden tables receive only the newest snapshot when opened, not every tick.
        ASSERT_EQ(hiddenChanges, 0);
        widget.m_resultTabs->setCurrentWidget(widget.m_rankingTable);
        ASSERT_EQ(widget.m_rankingTable->item(0, 2)->text(), QString("+40"));
        ASSERT_TRUE(hiddenChanges > 0);
        QObject::disconnect(connection);

        widget.m_resultTabs->setCurrentIndex(3);
        ASSERT_EQ(widget.m_statsTable->rowCount(), 12);
        widget.m_resultTabs->setCurrentIndex(4);
        ASSERT_EQ(widget.m_coverageTable->rowCount(), 12);

        // Re-entering an empty view must not restore the previous run on tab switches.
        widget.SetEmptyResults("No matching games");
        widget.m_resultTabs->setCurrentWidget(widget.m_breakdownTable);
        ASSERT_EQ(widget.m_breakdownTable->columnCount(), 1);
        widget.PopulateResults(aggregate);
        ASSERT_EQ(widget.m_breakdownTable->rowCount(), 12);

        QTableWidget* table = widget.m_breakdownTable;
        QApplication::processEvents();
        const int fitHeight = widget.height() - table->viewport()->height() + table->verticalHeader()->length();
        for (int offset = -16; offset <= 16; ++offset) {
            widget.resize(1600, fitHeight + offset);
            for (int i = 0; i < 10; ++i)
                QApplication::processEvents(QEventLoop::AllEvents, 5);

            ASSERT_EQ(table->horizontalScrollBar()->maximum(), 0);
            ASSERT_TRUE(table->horizontalHeader()->length() <= table->viewport()->width());
        }

        std::cout << "Live tabs coalesce hidden updates, show the latest snapshot, and settle at scrollbar boundaries\n";
    }

    static void MixedEnvironments(const QString& renderDirectory)
    {
        const bool showErrors = false;
        const bool notify = false;

        QTemporaryDir directory;
        ModelBenchmarkSavedResultsWidget widget;
        widget.SetWorkspaceMode();
        ASSERT_TRUE(widget.OpenLibrary(std::filesystem::path(directory.filePath("mixed.sqlite").toStdString()), showErrors));

        ModelBenchmarkResult run;
        run.modelNames = {"A", "B"};
        run.rocketSimVersions = {2};
        run.evaluationIdentity = "v2-exact";
        run.engineRevision = "physics-v2";
        run.integrationRevision = "build-one";

        ModelBenchmarkGameMetadata game;
        game.gameId = game.snapshotId = 1;
        game.modelA = game.blueModel = game.winner = 0;
        game.modelB = game.orangeModel = 1;
        game.goalScored = game.includedInRating = game.mirrorCompleted = true;
        game.goalTiming = ModelBenchmarkGoalTiming::PostKickoff;
        game.goalTimeSeconds = 20.0;
        run.games = {game};

        std::vector<ModelBenchmarkLibraryModel> models(2);
        for (int i = 0; i < 2; ++i) {
            models[i].displayName = QString::fromStdString(run.modelNames[i]);
            models[i].logicalKey = models[i].fingerprint = models[i].displayName;
        }

        ASSERT_TRUE(widget.SaveRun("v2", run, models, notify));
        run.rocketSimVersions = {3};
        run.evaluationIdentity = "v3-exact";
        run.engineRevision = "physics-v3";
        ASSERT_TRUE(widget.SaveRun("v3", run, models, notify));
        run.evaluationIdentity.clear();
        ASSERT_TRUE(widget.SaveRun("legacy-v3", run, models, notify));
        WaitForResults(widget);

        const std::vector<ModelBenchmarkLibraryModel> saved = widget.Library().ListModels();
        const std::vector<ModelBenchmarkLibrarySelection> selection{{saved[0].id, true}, {saved[1].id, true}};
        ModelBenchmarkLibraryFilter setup;
        setup.restrictEnvironments = true;
        setup.engineVersions = {3};
        auto applySetup = [&]() {
            widget.SetWorkspaceSelection(selection, setup);
            WaitForResults(widget);
        };

        applySetup();
        ASSERT_TRUE(widget.m_hasDisplayedResult);
        ASSERT_EQ(widget.m_displayedResult.eloRangeConfidence, 0.95f);
        ASSERT_EQ(widget.m_displayedResult.scoredGames, 2);
        ASSERT_EQ(widget.m_environmentCoverageTable->item(0, 1)->text(), QString("2"));
        ASSERT_EQ(widget.m_displayedEnvironments.size(), size_t(2));

        setup.engineVersions = {2, 3};
        applySetup();
        ASSERT_EQ(widget.m_displayedResult.scoredGames, 3);
        ASSERT_TRUE(widget.m_environmentSummaryLabel->text().contains("Mixed environments"));
        ASSERT_EQ(widget.m_environmentCoverageTable->columnCount(), 5);

        widget.resize(1440, 850);
        widget.show();
        QApplication::processEvents();
        if (!renderDirectory.isEmpty()) {
            ASSERT_TRUE(QDir().mkpath(renderDirectory));
            ASSERT_TRUE(widget.grab().save(QDir(renderDirectory).filePath("mixed-environments.png")));
            widget.m_resultTabs->setCurrentWidget(widget.m_environmentCoverageTable);
            QApplication::processEvents();
            ASSERT_TRUE(widget.grab().save(QDir(renderDirectory).filePath("environment-coverage.png")));
        }

        setup.environmentKeys[3] = {"exact:v3-exact"};
        applySetup();
        ASSERT_EQ(widget.m_displayedResult.scoredGames, 2);

        setup.engineVersions = {3};
        applySetup();
        ASSERT_EQ(widget.m_displayedResult.scoredGames, 1);
        ASSERT_EQ(widget.m_displayedResult.evaluationIdentity, "v3-exact");
        ASSERT_FALSE(widget.m_environmentSummaryLabel->text().contains("Mixed environments"));

        bool exported = false;
        const QMetaObject::Connection exportConnection = QObject::connect(
            &widget, &ModelBenchmarkSavedResultsWidget::ExportRequested, &widget, [&](ModelBenchmarkLibraryAggregate aggregate) {
                exported = true;
                ASSERT_EQ(aggregate.result.scoredGames, 1);
                ASSERT_EQ(aggregate.environments.size(), size_t(1));
                ASSERT_EQ(aggregate.result.evaluationIdentity, "v3-exact");
            });
        widget.ExportDisplayedResults();
        ASSERT_TRUE(exported);
        QObject::disconnect(exportConnection);

        setup.environmentKeys[3].clear();
        applySetup();
        ASSERT_FALSE(widget.m_hasDisplayedResult);
        ASSERT_FALSE(widget.m_exportButton->isEnabled());

        setup.environmentKeys.clear();
        setup.engineVersions.clear();
        applySetup();
        ASSERT_FALSE(widget.m_hasDisplayedResult);

        setup.engineVersions = {0};
        applySetup();
        ASSERT_FALSE(widget.m_hasDisplayedResult);

        // Live games obey the same simulator and exact-environment selection.
        run.rocketSimVersions = {2};
        run.evaluationIdentity = "live-v2";
        auto applyLiveSetup = [&]() {
            widget.BeginLiveComparison(selection, {}, {saved[0].id, saved[1].id}, setup);
            widget.UpdateLiveComparison(std::make_shared<const ModelBenchmarkResult>(run));
            WaitForResults(widget);
        };

        setup.engineVersions = {3};
        applyLiveSetup();
        ASSERT_EQ(widget.m_displayedResult.scoredGames, 2);

        setup.engineVersions = {2, 3};
        setup.environmentKeys[2] = {"exact:other-v2"};
        applyLiveSetup();
        ASSERT_EQ(widget.m_displayedResult.scoredGames, 2);

        setup.environmentKeys[2] = {"exact:live-v2"};
        applyLiveSetup();
        ASSERT_EQ(widget.m_displayedResult.scoredGames, 3);

        ASSERT_TRUE(widget.SaveRun("live-v2", run, models, notify));
        WaitForResults(widget);
        ASSERT_EQ(widget.m_displayedResult.scoredGames, 3);
        ASSERT_EQ(widget.m_environmentCoverageTable->item(0, 1)->text(), QString("3"));
        std::cout << "Setup selections control saved/live results, environment coverage, exports, and empty views without double-counting saved live games\n";
    }

    static void LiveCombined()
    {
        const bool showErrors = false;
        const bool notify = false;

        QTemporaryDir directory;
        const std::filesystem::path path(directory.filePath("live-ui.sqlite").toStdString());
        ModelBenchmarkSavedResultsWidget widget;
        ASSERT_TRUE(widget.OpenLibrary(path, showErrors));

        ModelBenchmarkResult run;
        run.modelNames = { "A", "B" };
        run.rocketSimVersions = {3};

        ModelBenchmarkGameMetadata game;
        game.gameId = game.snapshotId = 1;
        game.modelA = game.blueModel = game.winner = 0;
        game.modelB = game.orangeModel = 1;
        game.mode = 1;
        game.goalTimeSeconds = 20.0;
        game.goalTiming = ModelBenchmarkGoalTiming::PostKickoff;
        game.goalScored = game.includedInRating = game.mirrorCompleted = true;
        run.games = {game};

        std::vector<ModelBenchmarkLibraryModel> models(2);
        for (int i = 0; i < 2; ++i) {
            models[i].displayName = QString::fromStdString(run.modelNames[i]);
            models[i].agentName = models[i].displayName;
            models[i].logicalKey = models[i].fingerprint = models[i].displayName;
        }

        ASSERT_TRUE(widget.SaveRun("saved", run, models, notify));
        WaitForResults(widget);

        std::vector<ModelBenchmarkLibraryModel> rows = widget.m_library.ListModels();
        const auto hasName = [](const QString& name) {
            return [name](const ModelBenchmarkLibraryModel& model) { return model.displayName == name; };
        };
        const qint64 idA = std::find_if(rows.begin(), rows.end(), hasName("A"))->id;
        const qint64 idB = std::find_if(rows.begin(), rows.end(), hasName("B"))->id;
        widget.SetWorkspaceMode();

        // Legacy target-only flags cannot hide games between included models.
        widget.SetWorkspaceSelection({{idA, false}, {idB, false}}, {});
        WaitForResults(widget);
        ASSERT_TRUE(widget.m_hasDisplayedResult);
        ASSERT_EQ(widget.m_displayedResult.scoredGames, 1);

        ModelBenchmarkLibraryModel modelC = models[0];
        modelC.id = -2;
        modelC.displayName = modelC.agentName = modelC.logicalKey = modelC.fingerprint = "C";
        widget.BeginLiveComparison({{idA, false}, {idB, false}, {modelC.id, false}}, {modelC}, {idA, modelC.id}, {});

        ModelBenchmarkResult live = run;
        live.modelNames[1] = "C";
        widget.UpdateLiveComparison(std::make_shared<const ModelBenchmarkResult>(live));
        WaitForResults(widget);
        if (!widget.m_hasDisplayedResult && widget.m_detailsTable->item(0, 0))
            std::cerr << "Live comparison failed: " << widget.m_detailsTable->item(0, 0)->text().toStdString() << '\n';

        ASSERT_TRUE(widget.m_hasDisplayedResult);
        ASSERT_EQ(widget.m_displayedResult.scoredGames, 2);

        const auto baseline = widget.m_savedBaseline;
        ASSERT_TRUE(baseline != nullptr);
        for (int i = 2; i <= 10; ++i) {
            game.gameId = game.snapshotId = i;
            live.games.push_back(game);
            widget.UpdateLiveComparison(std::make_shared<const ModelBenchmarkResult>(live));
        }

        WaitForResults(widget);
        ASSERT_EQ(widget.m_displayedResult.scoredGames, 11);
        ASSERT_TRUE(widget.m_savedBaseline == baseline);
        widget.SetKnownModels(models);
        WaitForResults(widget);
        ASSERT_EQ(widget.m_displayedResult.scoredGames, 11);
        ASSERT_EQ(widget.m_displayedResult.modelNames.size(), size_t(3));

        // Saving the cumulative run clears its overlay before reading it back from disk.
        ASSERT_TRUE(widget.SaveRun("live", live, {models[0], modelC}, notify));
        rows = widget.m_library.ListModels();
        std::vector<ModelBenchmarkLibrarySelection> selection;
        for (const ModelBenchmarkLibraryModel& row : rows)
            selection.push_back({row.id, true});

        widget.SetWorkspaceSelection(selection, {});
        widget.UpdateLiveComparison(std::make_shared<const ModelBenchmarkResult>(live));
        WaitForResults(widget);
        ASSERT_FALSE(widget.m_liveComparison);
        ASSERT_EQ(widget.m_displayedResult.scoredGames, 11);

        // Excluding B removes its saved A-B game, but keeps all ten A-C games.
        std::vector<ModelBenchmarkLibrarySelection> withoutB = selection;
        std::erase_if(withoutB, [idB](const ModelBenchmarkLibrarySelection& member) { return member.modelId == idB; });
        widget.SetWorkspaceSelection(withoutB, {});
        WaitForResults(widget);
        ASSERT_EQ(widget.m_displayedResult.scoredGames, 10);
        ASSERT_EQ(widget.m_displayedResult.modelNames.size(), size_t(2));
        widget.SetWorkspaceSelection(selection, {});
        WaitForResults(widget);
        ASSERT_EQ(widget.m_displayedResult.scoredGames, 11);

        // A library switch invalidates delayed updates from the previous session.
        widget.BeginLiveComparison(selection, {}, {idA, idB}, {});
        widget.UpdateLiveComparison(std::make_shared<const ModelBenchmarkResult>(run));
        ASSERT_TRUE(widget.OpenLibrary(std::filesystem::path(directory.filePath("empty.sqlite").toStdString()), showErrors));
        WaitForResults(widget);
        ASSERT_FALSE(widget.m_liveComparison);
        ASSERT_FALSE(widget.m_hasDisplayedResult);
        std::cout << "Live combined results, cumulative updates, saving, and library isolation passed\n";
    }

    static void Run(const QString& renderDirectory)
    {
        const bool showErrors = false;

        QTemporaryDir directory;
        ASSERT_TRUE(directory.isValid());
        const std::filesystem::path path(directory.filePath("comparison-tests.sqlite").toStdString());
        QString error;
        ModelBenchmarkLibrary library;
        ASSERT_TRUE(library.Open(path, &error));

        ModelBenchmarkResult run;
        run.rocketSimVersions = { 2, 3 };
        run.modelNames = { "FixtureC v42", "FixtureC v41", "FixtureA v720", "FixtureB v107", "Nexto" };
        std::vector<ModelBenchmarkLibraryModel> models;
        for (int i = 0; i < (int)run.modelNames.size(); ++i) {
            ModelBenchmarkLibraryModel model;
            model.displayName = QString::fromStdString(run.modelNames[i]);
            model.agentName = model.displayName.section(' ', 0, 0);
            model.buildName = i < 2 ? "Main" : "Baseline";
            model.stageName = "3v3";
            model.branchName = "main";
            model.versionId = QString::number(i + 1);
            model.fingerprint = "fixture-" + model.versionId;
            model.logicalKey = model.displayName;
            model.latestTimesteps = (i + 1) * 100000000ULL;
            models.push_back(model);
        }

        const std::filesystem::path checkpoint(directory.filePath("local-checkpoint").toStdString());
        std::filesystem::create_directories(checkpoint);
        const std::filesystem::path policy = checkpoint / "POLICY.lt";
        {
            std::ofstream out(policy, std::ios::binary);
            out << "same-size-A";
        }

        const std::filesystem::file_time_type originalWriteTime = std::filesystem::last_write_time(policy);
        models[0].checkpointPath = QString::fromStdString(checkpoint.string());
        models[0].fingerprint = ModelBenchmarkLibrary::FingerprintCheckpoint(checkpoint);
        for (int a = 0; a < 5; ++a) {
            for (int b = a + 1; b < 5; ++b) {
                for (int match = 0; match < 20; ++match) {
                    ModelBenchmarkGameMetadata game;
                    game.gameId = run.games.size() + 1;
                    game.snapshotId = game.gameId;
                    game.modelA = game.blueModel = a;
                    game.modelB = game.orangeModel = b;
                    game.winner = match < 14 ? a : b;
                    game.mode = 3;
                    game.goalTimeSeconds = 20.0;
                    game.goalTiming = ModelBenchmarkGoalTiming::PostKickoff;
                    game.goalScored = game.includedInRating = game.mirrorCompleted = true;
                    run.games.push_back(game);
                }
            }
        }

        ASSERT_TRUE(library.SaveRun("comparison-fixture", run, models, {}, &error));
        library.Close();

        ModelBenchmarkSavedResultsWidget widget;
        widget.resize(1440, 900);
        widget.show();
        ASSERT_FALSE(widget.m_saveComparisonButton->isEnabled());
        ASSERT_FALSE(widget.m_primaryAllButton->isEnabled());
        ASSERT_TRUE(widget.OpenLibrary(path, showErrors));

        ModelBenchmarkLibraryModel known = models[0];
        known.availableLocally = true;
        known.fingerprint.clear();
        known.fingerprintVerified = false;
        widget.SetKnownModels({ known });
        WaitForResults(widget);
        auto savedLocal = std::find_if(widget.m_rows.begin(), widget.m_rows.end(), [&](const ModelBenchmarkLibraryModel& row) {
            return row.id >= 0 && row.logicalKey == known.logicalKey;
        });
        ASSERT_TRUE(savedLocal != widget.m_rows.end());
        ASSERT_TRUE(savedLocal->availableLocally);

        // Content verification must detect a replacement even when size and mtime are unchanged.
        {
            std::ofstream out(policy, std::ios::binary | std::ios::trunc);
            out << "same-size-B";
        }

        std::filesystem::last_write_time(policy, originalWriteTime);
        widget.SetKnownModels({ known });
        WaitForResults(widget);
        savedLocal = std::find_if(widget.m_rows.begin(), widget.m_rows.end(), [&](const ModelBenchmarkLibraryModel& row) {
            return row.id >= 0 && row.logicalKey == known.logicalKey;
        });
        ASSERT_TRUE(savedLocal != widget.m_rows.end());
        ASSERT_FALSE(savedLocal->availableLocally);
        ASSERT_TRUE(std::any_of(widget.m_rows.begin(), widget.m_rows.end(), [&](const ModelBenchmarkLibraryModel& row) {
            return row.id < 0 && row.logicalKey == known.logicalKey && row.fingerprintVerified;
        }));

        // A newer refresh generation wins if verification is already in flight.
        {
            std::ofstream out(policy, std::ios::binary | std::ios::trunc);
            out << "same-size-A";
        }

        std::filesystem::last_write_time(policy, originalWriteTime);
        widget.SetKnownModels({ known });
        widget.SetKnownModels({ known });
        WaitForResults(widget);
        savedLocal = std::find_if(widget.m_rows.begin(), widget.m_rows.end(), [&](const ModelBenchmarkLibraryModel& row) {
            return row.id >= 0 && row.logicalKey == known.logicalKey;
        });
        ASSERT_TRUE(savedLocal != widget.m_rows.end());
        ASSERT_TRUE(savedLocal->availableLocally);

        QApplication::processEvents();
        for (int i = 0; i < 3; ++i)
            ClickCell(widget.m_modelTable, 0, 0);

        ASSERT_EQ(widget.m_poolTable->rowCount(), 3);
        WaitForResults(widget);
        ASSERT_EQ(widget.m_detailsTable->item(5, 0)->text(), QString("RocketSim Versions"));
        ASSERT_EQ(widget.m_detailsTable->item(5, 1)->text(), QString("v2, v3"));
        ASSERT_TRUE(widget.m_saveComparisonButton->isEnabled());

        // Bulk primary changes also reach excluded members without including them again.
        ClickCell(widget.m_poolTable, 0, 0);
        widget.m_primaryNoneButton->click();
        for (const ModelBenchmarkComparisonMember& member : widget.CurrentComparison().members)
            ASSERT_FALSE(member.primary);

        ASSERT_FALSE(widget.CurrentComparison().members[0].included);
        ASSERT_FALSE(widget.m_hasDisplayedResult);
        widget.m_primaryAllButton->click();
        for (const ModelBenchmarkComparisonMember& member : widget.CurrentComparison().members)
            ASSERT_TRUE(member.primary);

        ASSERT_FALSE(widget.CurrentComparison().members[0].included);
        ClickCell(widget.m_poolTable, 1, 1);

        NameNextSave("  Weekly contenders  ");
        widget.m_saveComparisonButton->click();
        const ModelBenchmarkComparisonSet first = widget.CurrentComparison();
        ASSERT_EQ(first.name, QString("Weekly contenders"));
        ASSERT_TRUE(first.id >= 0);
        ASSERT_FALSE(widget.IsComparisonModified());
        ASSERT_FALSE(widget.m_saveComparisonButton->isEnabled());

        // Remove a bot, add a different one, save, and reload exactly those members.
        widget.m_poolTable->selectRow(1);
        const qint64 removed = widget.m_poolTable->item(1, 2)->data(Qt::UserRole).toLongLong();
        widget.m_removeButton->click();
        ASSERT_FALSE(widget.m_pooledModelIds.contains(removed));
        for (int row = 0; row < widget.m_modelTable->rowCount(); ++row) {
            if (widget.m_modelTable->item(row, 0)->data(Qt::UserRole).toLongLong() != removed) {
                ClickCell(widget.m_modelTable, row, 0);
                break;
            }
        }

        widget.m_saveComparisonButton->click();
        const ModelBenchmarkComparisonSet updated = widget.CurrentComparison();
        ASSERT_EQ(updated.members.size(), size_t(3));
        ASSERT_FALSE(updated.members == first.members);

        NameNextSave("All-round baselines");
        widget.m_saveComparisonAsButton->click();
        const qint64 secondId = widget.m_savedComparison.id;
        ASSERT_TRUE(secondId != first.id);
        ASSERT_EQ(widget.m_comparisonCombo->count(), 3);

        widget.m_excludeAllButton->click();
        for (const ModelBenchmarkComparisonMember& member : widget.CurrentComparison().members)
            ASSERT_FALSE(member.included);

        widget.m_includeAllButton->click();
        for (const ModelBenchmarkComparisonMember& member : widget.CurrentComparison().members)
            ASSERT_TRUE(member.included);

        // Cancel preserves the draft; Save before switching updates the current set.
        widget.m_comparisonCombo->setCurrentIndex(widget.m_comparisonCombo->findData(first.id));
        AnswerNextPrompt(QMessageBox::Cancel);
        widget.m_loadComparisonButton->click();
        ASSERT_EQ(widget.m_savedComparison.id, secondId);
        ASSERT_TRUE(widget.IsComparisonModified());

        AnswerNextPrompt(QMessageBox::Save);
        widget.m_loadComparisonButton->click();
        ASSERT_EQ(widget.m_savedComparison.id, first.id);
        ASSERT_EQ(widget.CurrentComparison().members, updated.members);
        ASSERT_EQ(widget.CurrentComparison().confidencePercent, 95.0);
        ASSERT_FALSE(widget.IsComparisonModified());

        widget.m_newComparisonButton->click();
        ASSERT_EQ(widget.m_poolTable->rowCount(), 0);
        ASSERT_FALSE(widget.m_removeButton->isEnabled());
        ASSERT_FALSE(widget.m_primaryNoneButton->isEnabled());
        ASSERT_EQ(widget.CurrentComparison().confidencePercent, 95.0);

        // Old comparison sets cannot restore a different Elo confidence.
        ModelBenchmarkComparisonSet legacyComparison = updated;
        legacyComparison.confidencePercent = 90.0;
        ASSERT_TRUE(widget.Library().SaveComparisonSet(legacyComparison));

        // Reopen from disk to exercise persistence beyond one widget/library connection.
        WaitForResults(widget);
        ASSERT_TRUE(widget.OpenLibrary(path, showErrors));
        widget.m_comparisonCombo->setCurrentIndex(widget.m_comparisonCombo->findData(first.id));
        widget.m_loadComparisonButton->click();
        ASSERT_EQ(widget.CurrentComparison().members, updated.members);
        ASSERT_EQ(widget.CurrentComparison().confidencePercent, 95.0);
        widget.m_primaryNoneButton->click();
        AnswerNextPrompt(QMessageBox::Discard);
        widget.m_loadComparisonButton->click();
        ASSERT_EQ(widget.CurrentComparison().members, updated.members);

        // Multiple selected rows can be removed together and an empty existing set saved.
        widget.m_poolTable->selectAll();
        widget.m_removeButton->click();
        ASSERT_EQ(widget.m_poolTable->rowCount(), 0);
        ASSERT_TRUE(widget.m_saveComparisonButton->isEnabled());
        widget.m_saveComparisonButton->click();
        widget.m_loadComparisonButton->click();
        ASSERT_EQ(widget.m_poolTable->rowCount(), 0);

        widget.m_comparisonCombo->setCurrentIndex(widget.m_comparisonCombo->findData(secondId));
        widget.m_loadComparisonButton->click();
        ASSERT_EQ(widget.m_poolTable->rowCount(), 3);
        for (const ModelBenchmarkComparisonMember& member : widget.CurrentComparison().members)
            ASSERT_TRUE(member.included);

        WaitForResults(widget);
        if (!widget.m_hasDisplayedResult && widget.m_rankingTable->item(0, 0))
            std::cerr << "Displayed result: " << widget.m_rankingTable->item(0, 0)->text().toStdString() << '\n';

        ASSERT_TRUE(widget.m_hasDisplayedResult);

        QTableWidgetItem* firstRankingItem = widget.m_rankingTable->item(0, 0);
        ASSERT_TRUE(firstRankingItem);
        ModelBenchmarkResult updatedResult = widget.m_displayedResult;
        updatedResult.rankings[0].relativeElo += 1.0;
        const bool refreshLayout = false;
        ForgeGUI::ModelBenchmarkTables::PopulateRankingTable(widget.m_rankingTable, updatedResult, refreshLayout);
        ASSERT_TRUE(widget.m_rankingTable->item(0, 0) == firstRankingItem);

        if (!renderDirectory.isEmpty()) {
            ASSERT_TRUE(QDir().mkpath(renderDirectory));
            widget.m_poolTable->clearSelection();
            for (const QSize size : { QSize(1440, 900), QSize(1280, 800) }) {
                widget.resize(size);
                QApplication::processEvents();
                ASSERT_EQ(widget.size(), size);
                for (QPushButton* button : widget.findChildren<QPushButton*>()) {
                    if (!button->isVisible())
                        continue;

                    ASSERT_TRUE(button->width() >= button->minimumSizeHint().width());
                    ASSERT_TRUE(widget.rect().contains(QRect(button->mapTo(&widget, QPoint()), button->size())));
                }

                const QString output = QDir(renderDirectory).filePath(
                    QString("saved-comparisons-%1x%2.png").arg(size.width()).arg(size.height()));
                ASSERT_TRUE(widget.grab().save(output));
                std::cout << "Rendered " << output.toStdString() << '\n';
            }

            QTimer::singleShot(0, [renderDirectory]() {
                auto* dialog = qobject_cast<QInputDialog*>(QApplication::activeModalWidget());
                ASSERT_TRUE(dialog);
                ASSERT_TRUE(dialog->grab().save(QDir(renderDirectory).filePath("save-comparison-dialog.png")));
                dialog->reject();
            });
            widget.m_saveComparisonAsButton->click();
            ASSERT_FALSE(widget.IsComparisonModified());
        }

        WaitForResults(widget);
        ASSERT_TRUE(widget.OpenLibrary(std::filesystem::path(directory.filePath("other.sqlite").toStdString()), showErrors));
        ASSERT_EQ(widget.m_comparisonCombo->count(), 1);
        ASSERT_FALSE(widget.m_loadComparisonButton->isEnabled());
        ASSERT_EQ(widget.m_poolTable->rowCount(), 0);
        std::cout << "Comparison controls, bulk flags, save/load, add/remove, copies, unsaved changes, and library isolation passed\n";
    }
};

int main(int argc, char** argv)
{
#ifdef Q_OS_WIN
    // The offscreen platform needs an explicit font directory on Windows.
    if (qEnvironmentVariable("QT_QPA_PLATFORM") == "offscreen" && qEnvironmentVariableIsEmpty("QT_QPA_FONTDIR"))
        qputenv("QT_QPA_FONTDIR", qgetenv("WINDIR") + "/Fonts");
#endif
    QApplication app(argc, argv);
    QFile theme(":/theme.qss");
    if (theme.open(QIODevice::ReadOnly))
        app.setStyleSheet(QString::fromUtf8(theme.readAll()));

    const QString renderDirectory = argc > 1 ? QString::fromLocal8Bit(argv[1]) : QString();
    try {
        ModelBenchmarkSavedResultsTests::LiveResultCellsNotifyViews();
        ModelBenchmarkSavedResultsTests::RankChangesKeepModelDataTogether();
        if (renderDirectory == "--row-identity")
            return 0;

        ModelBenchmarkSavedResultsTests::LiveTabRefresh();
        ModelBenchmarkSavedResultsTests::MixedEnvironments(renderDirectory);
        ModelBenchmarkSavedResultsTests::LiveCombined();
        ModelBenchmarkSavedResultsTests::Run(renderDirectory);
        return 0;
    } catch (const RocketForgeTests::AssertionFailure& error) {
        std::cerr << error.message << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
    }

    return 1;
}
