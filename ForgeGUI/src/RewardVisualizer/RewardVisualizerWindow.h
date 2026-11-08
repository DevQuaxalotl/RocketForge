#pragma once

#include <QWidget>
#include <QStringList>
#include <string>
#include <vector>
#include <ForgeDebugger.h>

class DebugBridge;
class RewardVisualizerWidget;
class QComboBox;
class QCheckBox;
class QSpinBox;

/**
 * RewardVisualizerWindow
 *
 * Floating panel that shows a live per-reward breakdown for the currently
 * observed player. Driven by RewardDebugSnapshot signals from DebugBridge.
 *
 * The window owns its own reward-viz controls (normalization mode, display
 * mode, player index, force-debug-all). MainWindow calls InitFromSettings
 * and ReadIntoSettings to sync with persisted GlobalSettings.
 **/
class RewardVisualizerWindow : public QWidget
{
    Q_OBJECT

public:
    explicit RewardVisualizerWindow(DebugBridge* bridge, QWidget* parent = nullptr);

    /** Initialize controls from persisted settings (call before first snapshot). **/
    void InitFromSettings(const ForgeDebug::DebugSettings& settings);

    /** Write current reward-viz control values back into settings for saving. **/
    void ReadIntoSettings(ForgeDebug::DebugSettings& settings) const;

signals:
    /** Emitted when the per-reward debug selection changes (names of enabled rewards). */
    void SelectionChanged(QStringList enabledRewardNames);

    /** Emitted when a persistable reward-viz control (norm/display/force-all) changes. */
    void RewardVizConfigChanged(int normMode, int displayMode, bool forceDebugAll);

    /** Emitted when the critic head selection changes (names of shown heads). */
    void CriticHeadSelectionChanged(QStringList enabledHeadNames);

    /** Emitted when the "Critic Heads" toggle changes. */
    void ShowCriticHeadsChanged(bool show);

public slots:
    void OnSnapshotReady(const ForgeDebug::RewardDebugSnapshot& snapshot);

protected:
    void showEvent(QShowEvent* event) override;

private:
    void BuildControls(DebugBridge* bridge);
    void UpdateSelectionFromWidgetToggle(const QString& rewardName, bool enabled);
    void UpdateCriticHeadSelection(const QString& headName, bool enabled);
    void EmitVizConfig();

    RewardVisualizerWidget* _widget         = nullptr;
    QComboBox*              _normCombo      = nullptr;
    QComboBox*              _displayCombo   = nullptr;
    QComboBox*              _sortCombo      = nullptr;
    QComboBox*              _sortOrderCombo = nullptr;
    QCheckBox*              _forceDebugCheck= nullptr;
    QSpinBox*               _playerSpin     = nullptr;
    QCheckBox*              _criticHeadsCheck = nullptr;
    bool                    _sizedToContent = false;

    std::vector<std::string> _criticHeadSelection;
    bool _hasCriticHeadSelection = false;
    std::vector<std::string> _lastCriticHeadNames;

    /**
     * Desired selection, updated each snapshot and on toggle.
     * Names absent from the current roster are kept, since the debug env's reward list can change between episodes.
     */
    QStringList _desiredSelection;
};
