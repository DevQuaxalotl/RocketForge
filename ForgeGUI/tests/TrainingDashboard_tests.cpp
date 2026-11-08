#include "test_runner.h"
#include "MainWindow/mainwindow.h"
#include "MainWindow/SkillRatingsPanel.h"
#include "VersionManagementDialog.h"
#include "ParamTesting/ParamTestingWidget.h"
#include "Widgets/AgentComboBox.h"
#include <GigaLearnInfer/AgentRegistry.h>
#include <QComboBox>
#include <QDialog>
#include <QElapsedTimer>
#include <QTableWidget>
#include <QThread>
#include "ui_mainwindow.h"
#include "Widgets/TrainingMonitorWindow.h"
#include <Storage/BuildStorage.h>
#include <GigaLearnCPP/Util/Report.h>
#include <QApplication>
#include <QDir>
#include <QFile>
#include <QGroupBox>
#include <QLabel>
#include <QRegularExpression>
#include <QSettings>
#include <QTemporaryDir>
#include <QTimer>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace RocketForgeTraining;
namespace fs = std::filesystem;

class TrainingDashboardTests {
public:
    static void CatalogReuse() {
        QTemporaryDir directory;
        ASSERT_TRUE(directory.isValid());
        const fs::path root(directory.path().toStdString());
        const QString agentName = "StorageCatalogFixture";
        AgentRegistry::Register(agentName.toStdString(), [root] {
            std::unique_ptr<Agent> agent = std::make_unique<Agent>();
            agent->modelPath = root;
            return agent;
        });

        BuildStorage storage(root, "Learning");
        storage.EnsureBuildExists();

        TrainingSegmentMetadata segment;
        segment.segmentId = "v1.00";
        segment.folderName = "v1.00_starter";
        segment.buildName = "Learning";
        segment.stageName = "starter";
        segment.skillStartRatings = {{"1v1", 1234}};
        storage.SaveSegment(segment);
        fs::create_directories(storage.SegmentCheckpointsRoot(segment.folderName) / "100");
        (void)storage.ListSegments();

        TrainingSegmentMetadata inactive = segment;
        inactive.segmentId = "v2.00";
        inactive.folderName = "v2.00_other";
        inactive.stageName = "other";
        inactive.status = "inactive";
        storage.SaveSegment(inactive);

        ModelCatalog& catalog = ModelCatalog::Instance();
        const auto waitUntil = [&](auto condition) {
            const qint64 timeoutMs = 10000;
            QElapsedTimer timer;
            timer.start();
            while (!condition() && timer.elapsed() < timeoutMs) {
                QApplication::processEvents();
                QThread::msleep(10);
            }

            ASSERT_TRUE(condition());
        };

        // Start cold: the requested Build must survive the asynchronous first scan.
        const bool trainingActive = false;
        VersionManagementDialog versionDialog(agentName, "Learning", {}, trainingActive, {}, {}, {}, {});
        ParamTestingWidget paramTesting;
        paramTesting.FollowTrainingAgent(agentName);
        waitUntil([&] { return versionDialog.m_versions.size() == 2 && paramTesting.m_baselineVersionCombo->count() == 1; });
        ASSERT_EQ(versionDialog.CurrentBuildName(), QString("Learning"));
        ASSERT_EQ(paramTesting.CurrentBaselineSegmentId(), QString("v1.00"));
        ASSERT_TRUE(versionDialog.m_catalog == paramTesting.m_catalog);
        ASSERT_EQ(versionDialog.m_catalog->Versions("Learning").size(), size_t(1));
        ASSERT_EQ(versionDialog.m_catalog->TrainingStages("Learning"), std::vector<std::string>{"starter"});
        ASSERT_EQ(versionDialog.m_catalog->storage.segmentsByBuild.at("Learning").front().skillStartRatings.at("1v1"), 1234);
        waitUntil([&] { return !catalog.IsRefreshing(agentName); });

        int jsonReads = -1;
        int directoryReads = -1;
        const QMetaObject::Connection connection = QObject::connect(&catalog, &ModelCatalog::RefreshCompleted, &versionDialog,
            [&](const QString& agent, int jsonCount, int directoryCount, qint64) {
                if (agent == agentName) {
                    jsonReads = jsonCount;
                    directoryReads = directoryCount;
                }
            });
        catalog.Refresh(agentName);
        waitUntil([&] { return jsonReads >= 0 && !catalog.IsRefreshing(agentName); });
        ASSERT_EQ(jsonReads, 0);
        ASSERT_EQ(directoryReads, 0);

        versionDialog.m_table->setColumnWidth(0, 180);
        const auto snapshot = versionDialog.m_catalog;
        versionDialog.m_stageCombo->setCurrentText("starter");
        ASSERT_EQ(versionDialog.m_versions.size(), size_t(1));
        ASSERT_TRUE(versionDialog.m_catalog == snapshot);
        versionDialog.m_table->selectRow(0);
        const std::string selected = versionDialog.SelectedVersionId();

        // File changes are validated in the worker and both screens receive one snapshot.
        const fs::path autosave = storage.SegmentTempCheckpointsRoot(segment.folderName) / "200";
        fs::create_directories(autosave);
        ASSERT_TRUE(storage.RecordSavedCheckpoint(segment.segmentId, segment.folderName, 200, autosave));
        catalog.Refresh(agentName);
        waitUntil([&] { return versionDialog.m_versions.front().latestTimesteps == 200; });
        ASSERT_TRUE(versionDialog.m_versions.front().latestIsAutosave);
        ASSERT_EQ(versionDialog.SelectedVersionId(), selected);
        ASSERT_EQ(versionDialog.m_table->columnWidth(0), 180);
        ASSERT_TRUE(versionDialog.m_catalog == paramTesting.m_catalog);
        ASSERT_TRUE(paramTesting.m_baselineVersionCombo->currentText().contains("200"));
        ASSERT_EQ(paramTesting.CurrentBaselineSegmentId(), selected.c_str());

        // A confirmation dialog must not have its selected row replaced under it.
        QDialog modal;
        modal.setModal(true);
        modal.show();
        TrainingSegmentMetadata updated = storage.LoadSegmentByFolderName(segment.folderName).value();
        updated.notes = "new note";
        storage.SaveSegment(updated);
        catalog.Refresh(agentName);
        waitUntil([&] { return catalog.Get(agentName)->Versions("Learning").front().notes == "new note"; });
        ASSERT_TRUE(versionDialog.m_versions.front().notes != "new note");
        modal.close();
        waitUntil([&] { return versionDialog.m_versions.front().notes == "new note"; });
        QObject::disconnect(connection);

        std::cout << "Shared catalog, cold selection, warm zero-read refresh, live autosaves, and modal selection checks passed.\n";
    }

    static void Run(const QString& output) {
        const bool loadSettings = false;
        MainWindow window(nullptr, loadSettings);
        window.m_saveDebounceTimer->stop();

        const bool savePreference = false;
        window.SetColorsEnabled(true, savePreference);

        auto line = [&](const QString& value) { window.m_parser->ProcessLine(value); };
        auto report = [&](GGL::Report values, const std::vector<std::string>& keys) {
            std::ostringstream text;
            std::streambuf* capture = std::cout.rdbuf(text.rdbuf());
            values.Display(keys);
            std::cout.rdbuf(capture);
            ASSERT_TRUE(text.str().find("[Training metrics]") != std::string::npos);

            for (const QString& value : QString::fromStdString(text.str()).split('\n'))
                line(value);
        };

        QTemporaryDir temporary;
        ASSERT_TRUE(temporary.isValid());
        BuildStorage storage(fs::path(temporary.path().toStdString()), "Learning");
        TrainingSegmentMetadata segment;
        segment.segmentId = "v1.01";
        segment.folderName = "v1.01_starter";
        segment.startTimestep = 100;
        segment.sourceCheckpointPath = "removed/100";
        segment.skillStartRatings = {{"1v1", 2100}, {"2v2", 2500}};
        storage.SaveSegment(segment);

        const fs::path saved = storage.SegmentTempCheckpointsRoot(segment.folderName) / "500";
        fs::create_directories(saved);
        std::ofstream(saved / "RUNNING_STATS.json") << R"({"skill_ratings":{"1v1":{"mu":2250},"2v2":{"mu":2520}}})";

        const std::optional<TrainingSegmentMetadata> loaded = storage.LoadSegmentById(segment.segmentId);
        ASSERT_TRUE(loaded.has_value());
        window.m_skillRatings->SeedFromSegment(storage, *loaded);
        ASSERT_EQ(window.m_skillRatings->m_valueLabels["1v1"]->text().toStdString(), "2250 (+150)");
        line("SkillTest/Rating/1v1: 2,300.5");
        ASSERT_EQ(window.m_skillRatings->m_valueLabels["1v1"]->text().toStdString(), "2300.5 (+200.5)");
        window.ResetSkillRatings();
        window.m_skillRatings->SeedFromSegment(storage, *loaded);
        ASSERT_EQ(window.m_skillRatings->m_valueLabels["1v1"]->text().toStdString(), "2250 (+150)");

        GGL::Report first;
        first["Average Step Reward"] = 0.1234567;
        first["LossAndEntropy/Policy Loss"] = 0.0000002;
        first["Steps Per Second/Overall"] = 12345.678;
        first["Iteration Time"] = 2.4;
        report(first, {"Average Step Reward", "LossAndEntropy/Policy Loss", "Steps Per Second/Overall", "Iteration Time"});
        ASSERT_EQ(window.ui->collectAvgRewardValueLabel->text().toStdString(), "0.1235");
        ASSERT_TRUE(window.ui->lossPolicyValueLabel->text().contains("e-07"));
        ASSERT_EQ(window.ui->throughputOverallValueLabel->text().toStdString(), "12,345.7");

        GGL::Report second;
        second["Collected Timesteps"] = 1000;
        report(second, {"Collected Timesteps"});
        ASSERT_EQ(window.ui->lossPolicyValueLabel->text().toStdString(), "N/A");
        ASSERT_EQ(window.ui->timingIterationValueLabel->text().toStdString(), "N/A");
        ASSERT_EQ(window.m_skillRatings->m_valueLabels["1v1"]->text().toStdString(), "2250 (+150)");

        // A tailed log can split a report anywhere; the unread remainder must keep its previous values.
        report(first, {"Average Step Reward", "LossAndEntropy/Policy Loss", "Steps Per Second/Overall", "Iteration Time"});
        const QString firstIterationTime = window.ui->timingIterationValueLabel->text();
        GGL::Report third;
        third["Average Step Reward"] = 0.5;
        third["Iteration Time"] = 3.1;
        std::ostringstream split;
        auto* splitCapture = std::cout.rdbuf(split.rdbuf());
        third.Display({"Average Step Reward", "Iteration Time"});
        std::cout.rdbuf(splitCapture);
        const QStringList splitLines = QString::fromStdString(split.str()).split('\n');
        const qsizetype iterationLine = splitLines.indexOf(QRegularExpression("^Iteration Time:.*"));
        ASSERT_TRUE(iterationLine > 0);
        for (qsizetype i = 0; i < iterationLine; ++i) line(splitLines[i]);
        ASSERT_TRUE(window.ui->collectAvgRewardValueLabel->text().startsWith("0.5"));
        ASSERT_EQ(window.ui->timingIterationValueLabel->text(), firstIterationTime);
        ASSERT_TRUE(window.ui->lossPolicyValueLabel->text().contains("e-07"));
        for (qsizetype i = iterationLine; i < splitLines.size(); ++i) line(splitLines[i]);
        ASSERT_TRUE(window.ui->timingIterationValueLabel->text() != firstIterationTime);
        ASSERT_EQ(window.ui->lossPolicyValueLabel->text().toStdString(), "N/A");
        ASSERT_EQ(window.ui->throughputOverallValueLabel->text().toStdString(), "N/A");

        line("LossAndEntropy/Policy Loss: nan");
        ASSERT_EQ(window.ui->lossPolicyValueLabel->text().toStdString(), "N/A");

        line("Total Timesteps: 9,007,199,254,740,993");
        ASSERT_EQ(window.ui->totalTimestepsLabel->text().toStdString(), "9,007,199,254,740,993");

        // A current rating must not fabricate a historical baseline, including overall skill.
        window.ResetSkillRatings();
        window.m_skillRatings->m_startRatings = {{"1v1", 2100}};
        line("SkillTest/Rating/1v1: 2300");
        line("SkillTest/Rating/2v2: 2500");
        ASSERT_EQ(window.m_skillRatings->m_startRatings.size(), 1);
        ASSERT_TRUE(window.m_skillRatings->m_valueLabels["2v2"]->text().contains(QString::fromUtf8("—")));
        ASSERT_TRUE(window.m_skillRatings->m_valueLabels["Overall"]->text().startsWith("2400"));
        ASSERT_TRUE(window.m_skillRatings->m_valueLabels["Overall"]->text().contains(QString::fromUtf8("—")));

        window.m_currentAgent.clear();
        window.OnStageStarted(0, "starter", "v1.02");
        ASSERT_TRUE(window.m_skillRatings->m_startRatings.isEmpty());
        ASSERT_TRUE(window.m_skillRatings->m_currentRatings.isEmpty());
        ASSERT_EQ(window.ui->totalIterationsLabel->text().toStdString(), "N/A");
        window.m_trainingDurationRefreshTimer->stop();

        // Populate public metrics for a real layout check and optional screenshot.
        window.m_skillRatings->m_startRatings = {{"1v1", 2100}, {"2v2", 2500}, {"3v3", 2300}};
        line("SkillTest/Rating/1v1: 2240.5");
        line("SkillTest/Rating/2v2: 2520");
        line("SkillTest/Rating/3v3: 2290");
        line("Total Timesteps: 125000000");
        line("Total Iterations: 500");
        line("Collected Timesteps: 250000");
        for (const char* text : {"Steps Per Second/Overall: 12345.6", "Average Step Reward: 0.0251",
                "LossAndEntropy/Policy Entropy: 0.38", "LossAndEntropy/Policy Loss: 0.0000002",
                "LossAndEntropy/Critic Loss: 0.052", "Iteration Time: 20.1", "Collection Time: 15.7",
                "Consumption Time: 4.4", "LossAndEntropy/Policy Update Magnitude: 0.0028"})
            line(text);

        const QString path = QString::fromStdString(saved.string());
        line("\tCheckpoint Save/Load Dir: \"" + path + "\"");
        ASSERT_EQ(window.ui->checkpointValueLabel->toolTip(), "\"" + path + "\"");

        window.show();
        for (const QSize size : {QSize(1800, 1000), QSize(1744, 840)}) {
            window.resize(size);
            QApplication::processEvents();

            const QList<QGroupBox*> cards = {window.ui->trainingStatsGroup, window.m_progressMetricsGroup,
                window.ui->skillRatingsGroup, window.ui->collectMetricsGroup, window.ui->gaeMetricsGroup,
                window.ui->lossMetricsGroup, window.ui->timingMetricsGroup, window.ui->throughputMetricsGroup,
                window.ui->updateMetricsGroup};
            for (int i = 0; i < cards.size(); ++i) {
                const QRect card(cards[i]->mapTo(&window, QPoint()), cards[i]->size());
                const QRect container(window.ui->metricsRowsWidget->mapTo(&window, QPoint()), window.ui->metricsRowsWidget->size());
                ASSERT_TRUE(container.contains(card));

                const QRect console(window.ui->consoleGroup->mapTo(&window, QPoint()), window.ui->consoleGroup->size());
                ASSERT_FALSE(card.intersects(console));

                for (int j = i + 1; j < cards.size(); ++j) {
                    const QRect otherCard(cards[j]->mapTo(&window, QPoint()), cards[j]->size());
                    ASSERT_FALSE(card.intersects(otherCard));
                }
            }

            if (!output.isEmpty()) {
                QDir().mkpath(output);
                ASSERT_TRUE(window.grab().save(output + QString("/dashboard-%1x%2.png").arg(size.width()).arg(size.height())));
            }
        }

        const TrainingMonitorSnapshot monitor = window.BuildTrainingMonitorSnapshot();
        ASSERT_EQ(monitor.eta, window.m_progressEtaValueLabel->text());
        ASSERT_EQ(monitor.segmentTime, window.m_progressSegmentTimeValueLabel->text());
    }
};

int main(int argc, char** argv) {
#ifdef Q_OS_WIN
    if (qEnvironmentVariable("QT_QPA_PLATFORM") == "offscreen" && qEnvironmentVariableIsEmpty("QT_QPA_FONTDIR"))
        qputenv("QT_QPA_FONTDIR", qgetenv("WINDIR") + "/Fonts");
#endif
    QApplication app(argc, argv);
    QTemporaryDir settings;
    if (!settings.isValid())
        return 1;

    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, settings.path());
    QSettings::setPath(QSettings::NativeFormat, QSettings::UserScope, settings.path());
    QDir::setCurrent(settings.path());

    std::streambuf* original = std::cout.rdbuf();
    try {
        const QString output = app.arguments().size() > 1 ? app.arguments().at(1) : QString();
        TrainingDashboardTests::CatalogReuse();
        TrainingDashboardTests::Run(output);
        std::cout.rdbuf(original);
        std::cout.clear();
        std::cout << "Training dashboard checks passed.\n";
        return 0;
    } catch (const RocketForgeTests::AssertionFailure& error) {
        std::cerr << error.message << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
    }

    std::cout.rdbuf(original);
    std::cout.clear();
    return 1;
}
