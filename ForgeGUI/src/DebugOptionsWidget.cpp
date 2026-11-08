#include "DebugOptionsWidget.h"
#include "DebugBridge.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QSignalBlocker>
#include <QTimer>
#include <cmath>

namespace
{
    // How often the game-speed spinbox polls the render loop's live value while training is active.
    constexpr int GAME_SPEED_SYNC_INTERVAL_MS = 200;

    // Absorbs float round-trip noise between the spinbox and the loop's atomic game speed.
    constexpr double GAME_SPEED_SYNC_EPSILON = 1e-3;
}

DebugOptionsWidget::DebugOptionsWidget(DebugBridge* bridge, QWidget* parent)
    : QWidget(parent)
{
    BuildUI(bridge);
}

void DebugOptionsWidget::BuildUI(DebugBridge* bridge)
{
    _bridge = bridge;

    auto* layout = new QFormLayout(this);
    layout->setContentsMargins(8, 4, 8, 4);

    // Show Rewards
    _showRewardsCheck = new QCheckBox("Show Rewards");
    layout->addRow(_showRewardsCheck);

    // Game Speed
    _gameSpeedSpin = new QDoubleSpinBox;
    _gameSpeedSpin->setRange(0.1, 10.0);
    _gameSpeedSpin->setSingleStep(0.1);
    _gameSpeedSpin->setValue(1.0);
    _gameSpeedSpin->setDecimals(2);
    _gameSpeedSpin->setEnabled(false); // enabled only while training
    layout->addRow("Game Speed:", _gameSpeedSpin);

    // The render loop also writes the speed (viewer keybinds), so mirror it back while running.
    _gameSpeedSyncTimer = new QTimer(this);
    _gameSpeedSyncTimer->setInterval(GAME_SPEED_SYNC_INTERVAL_MS);

    // Team Size
    _teamSizeCombo = new QComboBox;
    _teamSizeCombo->addItem("Default (weighted)", -1);
    layout->addRow("Team Size:", _teamSizeCombo);

    // Force State Setter
    _forceSetterCheck = new QCheckBox("Force Single State Setter");
    layout->addRow(_forceSetterCheck);

    _forceSetterLabel = new QLabel("State Setter:");
    _forceSetterCombo = new QComboBox;
    layout->addRow(_forceSetterLabel, _forceSetterCombo);
    UpdateForceSetterControls();

    // Max Episode Time
    auto* maxRow = new QHBoxLayout;
    _maxTimeEnabledCheck = new QCheckBox;
    _maxTimeSpin = new QDoubleSpinBox;
    _maxTimeSpin->setRange(0.5, 300.0);
    _maxTimeSpin->setSingleStep(0.5);
    _maxTimeSpin->setValue(30.0);
    _maxTimeSpin->setDecimals(1);
    _maxTimeSpin->setSuffix(" s");
    _maxTimeSpin->setEnabled(false);
    maxRow->addWidget(_maxTimeEnabledCheck);
    maxRow->addWidget(_maxTimeSpin);
    maxRow->addStretch();
    layout->addRow("Max Episode Time:", maxRow);

    // Connections
    connect(_showRewardsCheck, &QCheckBox::toggled,
            this, &DebugOptionsWidget::Changed);

    connect(_gameSpeedSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [this, bridge](double val) {
                if (bridge) bridge->RequestGameSpeed(static_cast<float>(val));
                emit Changed();
            });

    connect(_gameSpeedSyncTimer, &QTimer::timeout, this, &DebugOptionsWidget::SyncGameSpeedFromBridge);

    connect(_teamSizeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this, bridge](int) {
                if (bridge) bridge->RequestTeamSizeOverride(_teamSizeCombo->currentData().toInt());
                emit Changed();
            });

    connect(_forceSetterCheck, &QCheckBox::toggled,
            this, [this, bridge](bool checked) {
                UpdateForceSetterControls();
                if (bridge) bridge->RequestForceSingleStateSetter(checked);
                emit Changed();
            });

    connect(_forceSetterCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this, bridge](int) {
                _forceSetterSelectionIndex = _forceSetterCombo->currentData().toInt();
                if (bridge) bridge->RequestForceStateSetterIndex(_forceSetterSelectionIndex);
                emit Changed();
            });

    connect(_maxTimeEnabledCheck, &QCheckBox::toggled,
            this, [this, bridge](bool checked) {
                _maxTimeSpin->setEnabled(checked);
                if (bridge) bridge->RequestMaxEpisodeTimeEnabled(checked);
                emit Changed();
            });

    connect(_maxTimeSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [this, bridge](double val) {
                if (bridge) bridge->RequestMaxEpisodeTime(static_cast<float>(val));
                emit Changed();
            });
}

void DebugOptionsWidget::LoadSettings(const ForgeDebug::DebugSettings& s, bool showRewards)
{
    QSignalBlocker b1(_showRewardsCheck), b2(_gameSpeedSpin), b3(_teamSizeCombo),
                   b4(_forceSetterCheck), b5(_maxTimeEnabledCheck), b6(_maxTimeSpin),
                   b7(_forceSetterCombo);

    _showRewardsCheck->setChecked(showRewards);
    _gameSpeedSpin->setValue(s.gameSpeed);
    _forceSetterCheck->setChecked(s.forceSingleStateSetter);
    _forceSetterSelectionIndex = s.forceStateSetterIndex;
    ApplyForceSetterSelectionToCombo(true);
    UpdateForceSetterControls();
    _maxTimeEnabledCheck->setChecked(s.maxEpisodeTimeEnabled);
    _maxTimeSpin->setValue(s.maxEpisodeTime);
    _maxTimeSpin->setEnabled(s.maxEpisodeTimeEnabled);

    // Restore team size selection by stored index (-1 = first entry "Default")
    int comboIdx = 0;
    for (int i = 0; i < _teamSizeCombo->count(); ++i) {
        if (_teamSizeCombo->itemData(i).toInt() == s.teamSizeOverrideIndex) {
            comboIdx = i;
            break;
        }
    }
    _teamSizeCombo->setCurrentIndex(comboIdx);
}

ForgeDebug::DebugSettings DebugOptionsWidget::ToDebugSettings() const
{
    ForgeDebug::DebugSettings s;
    s.gameSpeed              = static_cast<float>(_gameSpeedSpin->value());
    s.teamSizeOverrideIndex  = _teamSizeCombo->currentData().toInt();
    s.forceSingleStateSetter = _forceSetterCheck->isChecked();
    s.forceStateSetterIndex  = _forceSetterSelectionIndex;
    s.maxEpisodeTimeEnabled  = _maxTimeEnabledCheck->isChecked();
    s.maxEpisodeTime         = static_cast<float>(_maxTimeSpin->value());
    // normMode/displayMode/playerIndex/forceDebugAll managed by RewardVisualizerWindow
    return s;
}

bool DebugOptionsWidget::IsShowRewardsChecked() const
{
    return _showRewardsCheck->isChecked();
}

void DebugOptionsWidget::SetTeamSizes(const std::vector<RocketForge::WeightedTeamSize>& sizes)
{
    // Skip rebuild if the available team sizes haven't changed so the user's
    // selection is not disturbed when a new stage starts with the same layout.
    if (sizes.size() == _lastTeamSizes.size()) {
        bool same = true;
        for (size_t i = 0; i < sizes.size(); ++i) {
            if (sizes[i].teamSizeA != _lastTeamSizes[i].teamSizeA ||
                sizes[i].teamSizeB != _lastTeamSizes[i].teamSizeB) {
                same = false;
                break;
            }
        }
        if (same) return;
    }
    _lastTeamSizes = sizes;

    QSignalBlocker blocker(_teamSizeCombo);
    int prevIndex = _teamSizeCombo->currentData().toInt();

    _teamSizeCombo->clear();
    _teamSizeCombo->addItem("Default (weighted)", -1);
    for (int i = 0; i < (int)sizes.size(); ++i) {
        const auto& ts = sizes[i];
        _teamSizeCombo->addItem(
            QString("%1v%2").arg(ts.teamSizeA).arg(ts.teamSizeB), i);
    }

    // Restore previous selection if still valid
    int restoreIdx = 0;
    for (int i = 0; i < _teamSizeCombo->count(); ++i) {
        if (_teamSizeCombo->itemData(i).toInt() == prevIndex) {
            restoreIdx = i;
            break;
        }
    }
    _teamSizeCombo->setCurrentIndex(restoreIdx);
}

void DebugOptionsWidget::SetStateSetterOptions(const std::vector<RocketForge::DebugStateSetterOption>& options)
{
    QSignalBlocker blocker(_forceSetterCombo);
    const int previousSelection = _forceSetterSelectionIndex;

    _forceSetterCombo->clear();
    for (const auto& option : options)
        _forceSetterCombo->addItem(QString::fromStdString(option.label), option.index);

    ApplyForceSetterSelectionToCombo(false);
    if (_forceSetterSelectionIndex != previousSelection) {
        if (_bridge)
            _bridge->RequestForceStateSetterIndex(_forceSetterSelectionIndex);
        emit Changed();
    }
    UpdateForceSetterControls();
}

void DebugOptionsWidget::ApplyForceSetterSelectionToCombo(bool keepMissingSelection)
{
    if (_forceSetterCombo->count() == 0)
        return;

    int restoreIdx = -1;
    for (int i = 0; i < _forceSetterCombo->count(); ++i) {
        if (_forceSetterCombo->itemData(i).toInt() == _forceSetterSelectionIndex) {
            restoreIdx = i;
            break;
        }
    }

    if (restoreIdx < 0) {
        restoreIdx = 0;
        if (!keepMissingSelection && _forceSetterCombo->count() > 0)
            _forceSetterSelectionIndex = _forceSetterCombo->itemData(0).toInt();
    }

    if (restoreIdx >= 0)
        _forceSetterCombo->setCurrentIndex(restoreIdx);
}

void DebugOptionsWidget::UpdateForceSetterControls()
{
    const bool visible = _forceSetterCheck && _forceSetterCheck->isChecked();
    if (_forceSetterLabel)
        _forceSetterLabel->setVisible(visible);
    if (_forceSetterCombo) {
        _forceSetterCombo->setVisible(visible);
        _forceSetterCombo->setEnabled(visible && _forceSetterCheck->isEnabled());
    }
}

void DebugOptionsWidget::SyncGameSpeedFromBridge()
{
    if (!_bridge)
        return;

    const double loopSpeed = _bridge->GameSpeedAtomic().load(std::memory_order_relaxed);
    if (std::abs(loopSpeed - _gameSpeedSpin->value()) < GAME_SPEED_SYNC_EPSILON)
        return;

    {
        QSignalBlocker blocker(_gameSpeedSpin);
        _gameSpeedSpin->setValue(loopSpeed);
    }

    emit Changed();
}

void DebugOptionsWidget::SetTrainingActive(bool active)
{
    _gameSpeedSpin->setEnabled(active);
    _showRewardsCheck->setEnabled(!active);
    _teamSizeCombo->setEnabled(!active);
    _forceSetterCheck->setEnabled(!active);
    UpdateForceSetterControls();
    _maxTimeEnabledCheck->setEnabled(!active);
    _maxTimeSpin->setEnabled(!active && _maxTimeEnabledCheck->isChecked());

    if (active)
        _gameSpeedSyncTimer->start();
    else
        _gameSpeedSyncTimer->stop();
}
