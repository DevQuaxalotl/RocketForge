#pragma once

#include <QWidget>
#include <vector>

#include <ForgeDebugger.h>
#include <TrainingStage.h>

class DebugBridge;
class QCheckBox;
class QDoubleSpinBox;
class QComboBox;
class QLabel;
class QTimer;

/**
 * DebugOptionsWidget
 *
 * Collapsible panel of debug controls placed below the render checkbox.
 * Visibility is managed externally (MainWindow calls setVisible).
 * Control changes are forwarded to DebugBridge::Request*() and emit Changed()
 * so MainWindow triggers the auto-save debounce.
 **/
class DebugOptionsWidget : public QWidget
{
    Q_OBJECT

public:
    explicit DebugOptionsWidget(DebugBridge* bridge, QWidget* parent = nullptr);

    /** Load persisted values. showRewards lives in ForgeDebugSettings, not DebugSettings. **/
    void LoadSettings(const ForgeDebug::DebugSettings& settings, bool showRewards);

    /** Read control values back for saving. **/
    ForgeDebug::DebugSettings ToDebugSettings() const;

    /** Whether the Show Rewards checkbox is checked (saved to ForgeDebugSettings::showRewards). **/
    bool IsShowRewardsChecked() const;

    /** Repopulate the team size combo from stage data. **/
    void SetTeamSizes(const std::vector<RocketForge::WeightedTeamSize>& sizes);

    /** Repopulate force-setter choices from stage data. **/
    void SetStateSetterOptions(const std::vector<RocketForge::DebugStateSetterOption>& options);

    /** Enable/disable controls that only apply during active training. **/
    void SetTrainingActive(bool active);

signals:
    void Changed();

private:
    void BuildUI(DebugBridge* bridge);
    void ApplyForceSetterSelectionToCombo(bool keepMissingSelection);
    void UpdateForceSetterControls();
    void SyncGameSpeedFromBridge();

    DebugBridge*     _bridge             = nullptr;
    QCheckBox*      _showRewardsCheck    = nullptr;
    QDoubleSpinBox* _gameSpeedSpin       = nullptr;
    QComboBox*      _teamSizeCombo       = nullptr;
    QCheckBox*      _forceSetterCheck    = nullptr;
    QLabel*         _forceSetterLabel    = nullptr;
    QComboBox*      _forceSetterCombo    = nullptr;
    QCheckBox*      _maxTimeEnabledCheck = nullptr;
    QDoubleSpinBox* _maxTimeSpin         = nullptr;
    QTimer*         _gameSpeedSyncTimer  = nullptr;

    std::vector<RocketForge::WeightedTeamSize> _lastTeamSizes;
    int _forceSetterSelectionIndex = -1;
};
