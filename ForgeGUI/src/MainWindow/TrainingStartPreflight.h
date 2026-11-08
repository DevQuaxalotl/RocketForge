#pragma once

#include <QString>

#include <Settings/ForgeSettings.h>
#include <StageQueue/StageQueueRunner.h>

#include <filesystem>

class QWidget;
namespace RocketForgeTraining { class BuildStorage; }

/**
 * Checks the Training tab makes before it dispatches a queue run,
 * prompting the user where a saved segment and the current setup disagree.
 */
namespace ForgeGUI
{
    /** Resolves the build folder the queue's learner writes, without creating a segment. */
    std::filesystem::path TrainingBuildRoot(const RocketForgeTraining::ForgeSettings& settings);

    /** Returns why the queue's Static Opponents selection can't be used, or an empty string when it can. */
    QString StaticOpponentSelectionWarning(const RocketForgeTraining::ForgeSettings& settings);

    /** Resolves the queue's lineage the same way the queue runner will, so each stage's action can be predicted. */
    void PrepareSettingsForQueuePreflight(RocketForgeTraining::ForgeSettings& settings, RocketForgeTraining::BuildStorage& storage);

    /** Inspects segments of the given build for DecideAction. The storage must outlive the inspector. */
    RocketForgeTraining::SegmentInspector MakePreflightInspector(const RocketForgeTraining::BuildStorage& storage);

    /**
     * Compares each continued or source segment's saved Agent setup with the current one,
     * and asks the user which to train with. Sets each stage's agentSetupOverride accordingly.
     * Returns false when the user cancels or a saved setup can't be read.
     */
    bool ResolveAgentSetupOverridesForContinuation(QWidget* parent, RocketForgeTraining::ForgeSettings& settings);

    /** Asks before continuing a segment whose saved training setup the current stage would overwrite. Returns false on No. */
    bool ConfirmContinueSegmentMetadataOverwrite(QWidget* parent, RocketForgeTraining::ForgeSettings& settings);
}
