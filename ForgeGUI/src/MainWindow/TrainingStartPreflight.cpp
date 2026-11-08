#include "MainWindow/TrainingStartPreflight.h"

#include <AgentModelPaths.h>
#include <ForgeTraining.h>
#include <TrainingStage.h>
#include <GigaLearnCPP/TrainingStorage.h>
#include <StaticOpponents/StaticOpponentSelection.h>
#include <Storage/BuildStorage.h>
#include <TrainingSetup/TrainingSetupSnapshot.h>

#include <QDialog>
#include <QDialogButtonBox>
#include <QHeaderView>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QStringList>
#include <QTableWidget>
#include <QVBoxLayout>

#include <algorithm>
#include <exception>
#include <optional>
#include <set>
#include <string>
#include <vector>

using namespace RocketForge;
using namespace RocketForgeTraining;

namespace
{
    enum class AgentSetupMismatchChoice
    {
        UseSaved,
        UseCurrent,
        Cancel,
    };

    QTableWidgetItem* SetupDiffItem(const QString& text)
    {
        auto* item = new QTableWidgetItem(text);
        item->setFlags(item->flags() & ~Qt::ItemIsEditable);
        item->setToolTip(text);

        return item;
    }

    AgentSetupMismatchChoice PromptAgentSetupMismatch(
        QWidget* parent,
        int stageIndex,
        const QString& stageName,
        const QString& segmentId,
        bool startNewFromSource,
        const std::vector<AgentSetupDiff>& diffs)
    {
        QDialog dialog(parent);
        dialog.setWindowTitle("Agent Setup Changed");

        auto* layout = new QVBoxLayout(&dialog);
        const QString intro = startNewFromSource
            ? QString(
                "Stage %1 (%2) is starting a new segment from checkpoint %3, but the current Agent setup "
                "does not match the setup saved with that source segment.")
                .arg(stageIndex + 1)
                .arg(stageName)
                .arg(segmentId)
            : QString(
                "Stage %1 (%2) is continuing segment %3, but the current Agent setup "
                "does not match the setup saved with that segment.")
                .arg(stageIndex + 1)
                .arg(stageName)
                .arg(segmentId);
        auto* label = new QLabel(intro, &dialog);
        label->setWordWrap(true);
        layout->addWidget(label);

        auto* table = new QTableWidget((int)diffs.size(), 3, &dialog);
        table->setHorizontalHeaderLabels({ "Field", "Saved", "Current" });
        table->setEditTriggers(QAbstractItemView::NoEditTriggers);
        table->setSelectionMode(QAbstractItemView::NoSelection);
        table->verticalHeader()->setVisible(false);
        table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
        table->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
        table->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);

        for (int row = 0; row < (int)diffs.size(); ++row) {
            const AgentSetupDiff& diff = diffs[row];
            table->setItem(row, 0, SetupDiffItem(QString::fromStdString(diff.field)));
            table->setItem(row, 1, SetupDiffItem(QString::fromStdString(diff.savedValue)));
            table->setItem(row, 2, SetupDiffItem(QString::fromStdString(diff.currentValue)));
        }

        table->resizeRowsToContents();
        layout->addWidget(table);

        const QString detailText = startNewFromSource
            ? "Use Saved Agent Setup keeps the new segment compatible with the source checkpoint. "
              "Use Current Agent Setup trains the new segment with the current Agent values."
            : "Use Saved Agent Setup keeps training compatible with the existing segment. "
              "Use Current Agent Setup updates the segment metadata to the current Agent values.";
        auto* detail = new QLabel(detailText, &dialog);
        detail->setWordWrap(true);
        layout->addWidget(detail);

        AgentSetupMismatchChoice choice = AgentSetupMismatchChoice::Cancel;
        auto* buttons = new QDialogButtonBox(&dialog);
        QPushButton* savedButton = buttons->addButton("Use Saved Agent Setup", QDialogButtonBox::AcceptRole);
        QPushButton* currentButton = buttons->addButton("Use Current Agent Setup", QDialogButtonBox::ActionRole);
        QPushButton* cancelButton = buttons->addButton(QDialogButtonBox::Cancel);

        QObject::connect(savedButton, &QPushButton::clicked, [&]() {
            choice = AgentSetupMismatchChoice::UseSaved;
            dialog.accept();
        });
        QObject::connect(currentButton, &QPushButton::clicked, [&]() {
            choice = AgentSetupMismatchChoice::UseCurrent;
            dialog.accept();
        });
        QObject::connect(cancelButton, &QPushButton::clicked, [&]() {
            choice = AgentSetupMismatchChoice::Cancel;
            dialog.reject();
        });

        layout->addWidget(buttons);
        dialog.resize(760, 420);

        return dialog.exec() == QDialog::Accepted ? choice : AgentSetupMismatchChoice::Cancel;
    }

    bool PromptAssignMissingAgentSetup(
        QWidget* parent,
        int stageIndex,
        const QString& stageName,
        const QString& segmentId,
        bool startNewFromSource,
        const AgentSetupSnapshot& currentSetup)
    {
        QMessageBox message(parent);
        message.setIcon(QMessageBox::Warning);
        message.setWindowTitle("Agent Setup Missing");
        message.setText(startNewFromSource
            ? QString(
                "Stage %1 (%2) is starting a new segment from checkpoint %3, but that source segment "
                "does not have a complete saved Agent setup.")
                .arg(stageIndex + 1)
                .arg(stageName)
                .arg(segmentId)
            : QString(
                "Stage %1 (%2) is continuing segment %3, but that segment does not have a complete saved Agent setup.")
                .arg(stageIndex + 1)
                .arg(stageName)
                .arg(segmentId));
        message.setInformativeText(QString(
            "This includes older setup snapshots with no Agent setup and snapshots missing "
            "Observation Builder or Action Parser.\n\n"
            "Assign the current active Agent setup to this segment?\n\n"
            "Observation Builder: %1\n"
            "Action Parser: %2")
            .arg(QString::fromStdString(currentSetup.obsBuilderName))
            .arg(QString::fromStdString(currentSetup.actionParserName)));
        QPushButton* assignButton =
            message.addButton("Assign Current Agent Setup", QMessageBox::AcceptRole);
        message.addButton(QMessageBox::Cancel);
        message.setDefaultButton(assignButton);
        message.exec();

        return message.clickedButton() == assignButton;
    }

    bool AssignCurrentAgentSetupToSegment(
        BuildStorage& storage,
        TrainingSegmentMetadata& segment,
        TrainingSetupSnapshot snapshot,
        const AgentSetupSnapshot& currentSetup,
        std::string* error)
    {
        if (snapshot.agentName.empty())
            snapshot.agentName = currentSetup.agentName;
        if (snapshot.stageName.empty())
            snapshot.stageName = segment.stageName;
        if (snapshot.versionId.empty())
            snapshot.versionId = segment.segmentId;

        snapshot.savedAt = BuildStorage::MakeTimestamp();
        snapshot.agentSetup = currentSetup;

        const std::string setupHash = BuildStorage::MakeSetupHash(snapshot);
        const std::filesystem::path setupPath =
            storage.SegmentRoot(segment.folderName) / TrainingSetupSnapshot::FILE_NAME;

        if (!GGL::JsonSaveToFile(snapshot, setupPath, error))
            return false;

        segment.setupHash = setupHash;
        segment.updatedAt = BuildStorage::MakeTimestamp();
        storage.SaveSegment(segment);

        return true;
    }

    QString FormatSetupDiffText(
        const std::vector<TrainingSetupDiff>& diffs,
        const std::string& savedHash,
        const std::string& currentHash)
    {
        constexpr int MaxDiffLines = 12;

        if (diffs.empty()) {
            return QString(
                "- Saved setup hash: %1\n"
                "- Current setup hash: %2")
                .arg(QString::fromStdString(savedHash),
                     QString::fromStdString(currentHash));
        }

        QStringList lines;
        const int visibleCount = std::min(MaxDiffLines, static_cast<int>(diffs.size()));

        for (int i = 0; i < visibleCount; ++i) {
            const TrainingSetupDiff& diff = diffs[i];
            lines << QString("- %1: %2 -> %3")
                .arg(QString::fromStdString(diff.field),
                     QString::fromStdString(diff.savedValue),
                     QString::fromStdString(diff.currentValue));
        }

        const int remainingCount = static_cast<int>(diffs.size()) - visibleCount;
        if (remainingCount > 0)
            lines << QString("- ... and %1 more").arg(remainingCount);

        return lines.join('\n');
    }

    QString BuildSegmentSetupDiffText(
        const BuildStorage& storage,
        const TrainingSegmentMetadata& segment,
        const TrainingSetupSnapshot& currentSnapshot,
        const std::string& currentHash)
    {
        const std::filesystem::path setupPath =
            storage.SegmentRoot(segment.folderName) / TrainingSetupSnapshot::FILE_NAME;
        const std::optional<TrainingSetupSnapshot> savedSnapshot = LoadTrainingSetupSnapshot(setupPath);

        if (!savedSnapshot.has_value()) {
            return QString(
                "- Saved training setup snapshot is missing or unreadable.\n"
                "- Saved setup hash: %1\n"
                "- Current setup hash: %2")
                .arg(QString::fromStdString(segment.setupHash),
                     QString::fromStdString(currentHash));
        }

        return FormatSetupDiffText(
            DiffTrainingSetupSnapshots(savedSnapshot.value(), currentSnapshot),
            segment.setupHash,
            currentHash);
    }

    CheckpointInfo InspectSegmentForPreflight(const BuildStorage& storage, const std::string& segmentId)
    {
        CheckpointInfo info;

        if (segmentId.empty())
            return info;

        std::optional<TrainingSegmentMetadata> segment = storage.LoadSegmentById(segmentId);
        info.segmentExists = segment.has_value();

        if (!segment.has_value())
            return info;
        info.stageName = segment->stageName;
        info.parentSegmentId = segment->parentSegmentId;

        const std::set<int64_t> checkpoints =
            GGL::TrainingStorage::FindNumberedDirs(storage.SegmentCheckpointsRoot(segment->folderName));
        info.hasCheckpoints = !checkpoints.empty();
        if (info.hasCheckpoints)
            info.latestTimesteps = static_cast<uint64_t>(*checkpoints.rbegin());
        return info;
    }
}

namespace ForgeGUI
{
    std::filesystem::path TrainingBuildRoot(const ForgeSettings& settings)
    {
        const std::string buildName = settings.buildName.empty() ? std::string(GGL::DEFAULT_BUILD_NAME) : settings.buildName;
        BuildStorage storage(ResolveAgentModelsFolder(settings.agentName), buildName);

        return std::filesystem::absolute(storage.BuildRoot());
    }

    QString StaticOpponentSelectionWarning(const ForgeSettings& settings)
    {
        QStringList enabledStages;

        for (int i = 0; i < (int)settings.stages.size(); ++i) {
            const StageSettings& stage = settings.stages[i];

            if (!stage.staticOpponentsEnabled)
                continue;

            const QString stageName = QString::fromStdString(stage.stageName.empty()
                ? std::string("<unnamed>")
                : stage.stageName);
            enabledStages.push_back(QString("%1. %2").arg(i + 1).arg(stageName));
        }

        if (enabledStages.empty())
            return {};

        const std::string buildName = settings.buildName.empty()
            ? std::string(GGL::DEFAULT_BUILD_NAME)
            : settings.buildName;

        try {
            const std::vector<StaticOpponentSelectionEntry> entries = LoadStaticOpponentSelection(settings.agentName, buildName);

            if (entries.empty()) {
                return QString(
                    "Static Opponents is enabled for one or more stages, but build \"%1\" has no selected static opponents.\n\n"
                    "Stages:\n%2\n\n"
                    "Open Static Opponents... and select at least one model, or uncheck Static Opponents for those stages.")
                    .arg(QString::fromStdString(buildName), enabledStages.join("\n"));
            }

            SyncStaticOpponentSelection(settings.agentName, buildName);
        } catch (const std::exception& e) {
            return QString(
                "Static Opponents is enabled for one or more stages, but build \"%1\" has an invalid static opponent selection.\n\n"
                "Stages:\n%2\n\n"
                "%3\n\n"
                "Open Static Opponents... and update the selection before starting training.")
                .arg(
                    QString::fromStdString(buildName),
                    enabledStages.join("\n"),
                    QString::fromStdString(e.what()));
        } catch (...) {
            return QString(
                "Static Opponents is enabled for one or more stages, but build \"%1\" has an invalid static opponent selection.\n\n"
                "Stages:\n%2\n\n"
                "Open Static Opponents... and update the selection before starting training.")
                .arg(QString::fromStdString(buildName), enabledStages.join("\n"));
        }

        return {};
    }

    void PrepareSettingsForQueuePreflight(ForgeSettings& settings, BuildStorage& storage)
    {
        if (!settings.stages.empty() &&
            settings.stages.front().bootstrap.mode == BootstrapConfig::Mode::ContinueLatest) {
            const BuildMetadata build = storage.LoadBuild();
            settings.stages.front().bootstrap.selectedExistingVersionId = build.activeSegmentId;
        }

        PrepareQueueLineage(settings, MakePreflightInspector(storage));
    }

    SegmentInspector MakePreflightInspector(const BuildStorage& storage)
    {
        return [&storage](const std::string&, const std::string&, const std::string& segmentId) {
            return InspectSegmentForPreflight(storage, segmentId);
        };
    }

    bool ResolveAgentSetupOverridesForContinuation(QWidget* parent, ForgeSettings& settings)
    {
        if (settings.stages.empty() || settings.agentName.empty() || settings.buildName.empty())
            return true;

        BuildStorage storage(ResolveAgentModelsFolder(settings.agentName), settings.buildName);
        PrepareSettingsForQueuePreflight(settings, storage);
        const SegmentInspector inspector = MakePreflightInspector(storage);

        for (int i = 0; i < (int)settings.stages.size(); ++i) {
            StageSettings& entry = settings.stages[i];
            entry.agentSetupOverride.reset();

            const StageAction action = DecideAction(settings, (size_t)i, inspector);

            if (action.kind == ActionKind::Skip)
                continue;

            bool startNewFromSource = false;
            std::string segmentId;

            if (action.kind == ActionKind::Continue) {
                segmentId = action.continueSegmentId;
            } else if (action.kind == ActionKind::StartNew && !action.startNewSourceSegmentId.empty()) {
                segmentId = action.startNewSourceSegmentId;
                startNewFromSource = true;
            } else {
                continue;
            }

            if (segmentId.empty())
                continue;

            std::optional<TrainingSegmentMetadata> segment = storage.LoadSegmentById(segmentId);

            if (!segment.has_value())
                continue;

            const std::filesystem::path setupPath = storage.SegmentRoot(segment->folderName) / TrainingSetupSnapshot::FILE_NAME;

            // A setup that exists but can't be read must stop the run, never be offered for overwriting as "missing".
            std::optional<TrainingSetupSnapshot> savedSnapshot;

            try {
                savedSnapshot = LoadExistingTrainingSetupSnapshot(setupPath);
            } catch (const std::exception& error) {
                QMessageBox::warning(parent, "Cannot start training", QString::fromUtf8(error.what()));
                return false;
            }

            StageSettings currentEntry = entry;
            currentEntry.agentSetupOverride.reset();
            auto currentStage = CreateTrainingStage(settings.agentName, entry.stageName, settings.globals, currentEntry);

            if (!currentStage || !currentStage->agent)
                continue;

            AgentSetupSnapshot currentSetup = CaptureAgentSetupSnapshot(*currentStage->agent);

            if (savedSnapshot && savedSnapshot->agentSetup) {
                // Simulator and predictor follow the checkpoint automatically,
                // even if the user chooses current network/training settings in the existing setup comparison.
                CopyAgentSimulation(currentSetup, *savedSnapshot->agentSetup);
                entry.agentSetupOverride = currentSetup;
            }

            if (!savedSnapshot.has_value() || !savedSnapshot->agentSetup.has_value()) {
                try {
                    currentSetup = ResolveSavedAgentSetup(savedSnapshot, currentSetup);
                } catch (const std::exception& error) {
                    QMessageBox::warning(parent, "Cannot start training", QString::fromUtf8(error.what()));
                    return false;
                }

                entry.agentSetupOverride = currentSetup;

                const QString stageName = QString::fromStdString(entry.stageName);
                const QString savedSegmentId = QString::fromStdString(segment->segmentId);

                if (!PromptAssignMissingAgentSetup(parent, i, stageName, savedSegmentId, startNewFromSource, currentSetup))
                    return false;

                TrainingSetupSnapshot snapshotToSave = savedSnapshot.value_or(TrainingSetupSnapshot{});
                std::string error = {};

                if (!AssignCurrentAgentSetupToSegment(storage, segment.value(), snapshotToSave, currentSetup, &error)) {
                    QMessageBox::warning(
                        parent,
                        "Agent Setup Missing",
                        QString("Failed to assign current Agent setup to segment %1.\n\n%2")
                            .arg(savedSegmentId)
                            .arg(QString::fromStdString(error)));
                    return false;
                }

                continue;
            }

            const std::vector<AgentSetupDiff> diffs = DiffAgentSetupSnapshots(savedSnapshot->agentSetup.value(), currentSetup);

            if (diffs.empty())
                continue;

            const AgentSetupMismatchChoice choice = PromptAgentSetupMismatch(
                parent,
                i,
                QString::fromStdString(entry.stageName),
                QString::fromStdString(segment->segmentId),
                startNewFromSource,
                diffs);

            if (choice == AgentSetupMismatchChoice::Cancel)
                return false;

            if (choice == AgentSetupMismatchChoice::UseSaved) {
                entry.agentSetupOverride = savedSnapshot->agentSetup.value();
            } else if (action.kind == ActionKind::Continue) {
                entry.allowSegmentSetupMismatch = true;
            }
        }

        return true;
    }

    bool ConfirmContinueSegmentMetadataOverwrite(QWidget* parent, ForgeSettings& settings)
    {
        if (settings.stages.empty() || settings.buildName.empty())
            return true;

        const std::string& buildName = settings.buildName;
        BuildStorage storage(ResolveAgentModelsFolder(settings.agentName), buildName);
        PrepareSettingsForQueuePreflight(settings, storage);
        const SegmentInspector inspector = MakePreflightInspector(storage);

        for (int i = 0; i < (int)settings.stages.size(); ++i) {
            StageSettings& entry = settings.stages[i];
            const StageAction action = DecideAction(settings, (size_t)i, inspector);

            if (action.kind != ActionKind::Continue)
                continue;
            if (entry.allowSegmentSetupMismatch)
                continue;

            const std::string segmentId = action.continueSegmentId;

            if (segmentId.empty())
                continue;

            std::optional<TrainingSegmentMetadata> segment = storage.LoadSegmentById(segmentId);

            if (!segment.has_value())
                continue;

            auto stage = CreateTrainingStage(settings.agentName, entry.stageName, settings.globals, entry);

            if (!stage)
                continue;

            stage->ApplySegmentId(segment->segmentId, entry.versionNotes);
            GGL::LearnerRuntimeConfig& runtimeConfig = stage->learnerConfig.runtime;
            runtimeConfig.buildName = buildName;
            runtimeConfig.segmentId = segment->segmentId;
            runtimeConfig.trainingStageName = segment->stageName;
            runtimeConfig.staticOpponentsRegistryPath = storage.StaticOpponentsPath();
            runtimeConfig.skillReferencesRegistryPath = storage.SkillReferencesPath();
            TrainingSetupSnapshot snapshot = MakeTrainingSetupSnapshot(stage.get(), segment->segmentId);
            const std::string setupHash = BuildStorage::MakeSetupHash(snapshot);

            if (storage.SegmentSetupHashMatches(segment.value(), snapshot, setupHash))
                continue;

            const QString differences = BuildSegmentSetupDiffText(storage, segment.value(), snapshot, setupHash);
            const QString message = QString(
                "Segment %1 already has saved metadata for training stage \"%2\".\n\n"
                "The current row uses training stage \"%3\" and would overwrite these saved setup differences:\n"
                "%4\n\n"
                "Continuing without a revision increment will overwrite that segment's metadata.\n\n"
                "Recommended: choose Next Minor or Next Major to preserve the existing segment history.\n\n"
                "Continue and overwrite metadata for this segment?")
                .arg(QString::fromStdString(segment->segmentId),
                     QString::fromStdString(segment->stageName),
                     QString::fromStdString(entry.stageName),
                     differences);

            const QMessageBox::StandardButton choice = QMessageBox::warning(
                parent,
                "Overwrite Segment Metadata?",
                message,
                QMessageBox::Yes | QMessageBox::No,
                QMessageBox::No);

            if (choice != QMessageBox::Yes)
                return false;

            entry.allowSegmentSetupMismatch = true;
        }

        return true;
    }
}
