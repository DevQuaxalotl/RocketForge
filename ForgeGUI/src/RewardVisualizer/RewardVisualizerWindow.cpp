#include "RewardVisualizerWindow.h"

#include "../DebugBridge.h"
#include "CriticHeadRows.h"
#include "RewardVisualizerWidget.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QComboBox>
#include <QCheckBox>
#include <QSpinBox>
#include <QLabel>
#include <QSignalBlocker>
#include <QShowEvent>
#include <QSize>

#include <algorithm>

RewardVisualizerWindow::RewardVisualizerWindow(DebugBridge* bridge, QWidget* parent)
    : QWidget(parent, Qt::Window)
{
    setWindowTitle("Reward Debugger");
    setMinimumSize(760, 380);

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(6, 6, 6, 4);
    root->setSpacing(4);

    _widget = new RewardVisualizerWidget(this);

    BuildControls(bridge);

    root->addWidget(_widget, /*stretch=*/1);
}

void RewardVisualizerWindow::showEvent(QShowEvent* event)
{
    QWidget::showEvent(event);

    // Sized on first show rather than in the constructor so the hint includes the application style sheet's fonts and padding.
    if (_sizedToContent)
        return;

    _sizedToContent = true;
    const int initialWidth = std::max(860, layout()->sizeHint().width() + 24);
    resize(initialWidth, minimumHeight());
}

void RewardVisualizerWindow::BuildControls(DebugBridge* bridge)
{
    auto* bar    = new QHBoxLayout;
    auto* layout = static_cast<QVBoxLayout*>(this->layout());
    layout->addLayout(bar);

    // Norm mode
    bar->addWidget(new QLabel("Norm:", this));
    _normCombo = new QComboBox(this);
    _normCombo->addItem("Raw x Weight",   0);
    _normCombo->addItem("Per Reward",     1);
    _normCombo->addItem("By Category",    2);
    _normCombo->addItem("Proportional",   3);
    _normCombo->setCurrentIndex(1); // default matches LearnerConfig default
    _normCombo->setToolTip("How reward values are normalised for the bar visualisation");
    bar->addWidget(_normCombo);

    bar->addSpacing(8);

    // Display mode
    bar->addWidget(new QLabel("Display:", this));
    _displayCombo = new QComboBox(this);
    _displayCombo->addItem("Earned",   0);
    _displayCombo->addItem("Training", 1);
    _displayCombo->setToolTip("Earned: raw reward function output\nTraining: post-wrapper values (e.g. after ZeroSum)");
    bar->addWidget(_displayCombo);

    bar->addSpacing(8);

    // Sort mode
    bar->addWidget(new QLabel("Sort:", this));
    _sortCombo = new QComboBox(this);
    _sortCombo->addItem("Abs Sum",  2);
    _sortCombo->addItem("Weighted", 1);
    _sortCombo->addItem("Norm",     3);
    _sortCombo->addItem("Name",     0);
    _sortCombo->setToolTip("Sort rewards within each category by episode absolute sum, current value, normalized value, or name");
    bar->addWidget(_sortCombo);

    _sortOrderCombo = new QComboBox(this);
    _sortOrderCombo->addItem("High to Low", true);
    _sortOrderCombo->addItem("Low to High", false);
    _sortOrderCombo->setToolTip("High to Low: largest values first, or Z to A by name\nLow to High: smallest values first, or A to Z by name");
    bar->addWidget(_sortOrderCombo);

    bar->addSpacing(8);

    // Force debug all
    _forceDebugCheck = new QCheckBox("Force Debug All", this);
    _forceDebugCheck->setToolTip("Enable debug output for all rewards, not just those with debug enabled");
    bar->addWidget(_forceDebugCheck);

    bar->addSpacing(8);

    // Player index
    bar->addWidget(new QLabel("Player:", this));
    _playerSpin = new QSpinBox(this);
    _playerSpin->setRange(0, 7);
    _playerSpin->setToolTip("Index of the player whose rewards are displayed");
    bar->addWidget(_playerSpin);

    bar->addSpacing(8);

    _criticHeadsCheck = new QCheckBox("Critic Heads", this);
    _criticHeadsCheck->setChecked(true);
    _criticHeadsCheck->setEnabled(false);
    _criticHeadsCheck->setToolTip("List each decomposed-critic value head under the critic's total value");
    bar->addWidget(_criticHeadsCheck);

    bar->addStretch();

    // Wire controls to bridge request slots
    connect(_normCombo, qOverload<int>(&QComboBox::currentIndexChanged),
            bridge, &DebugBridge::RequestNormMode);

    connect(_displayCombo, qOverload<int>(&QComboBox::currentIndexChanged),
            bridge, &DebugBridge::RequestDisplayMode);

    connect(_sortCombo, qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this](int index) {
                if (_widget)
                    _widget->SetSortColumn(_sortCombo->itemData(index).toInt());
            });

    connect(_sortOrderCombo, qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this](int index) {
                if (_widget)
                    _widget->SetSortDescending(_sortOrderCombo->itemData(index).toBool());
            });

    connect(_forceDebugCheck, &QCheckBox::checkStateChanged, this,
            [bridge](Qt::CheckState state) {
                bridge->RequestForceDebugAll(state == Qt::Checked);
            });

    connect(_playerSpin, qOverload<int>(&QSpinBox::valueChanged),
            bridge, &DebugBridge::RequestPlayerIndex);

    // Wire reward toggles from the widget to bridge and emit selection changes
    connect(_widget, &RewardVisualizerWidget::RewardToggled, this,
            [this, bridge](const QString& name, bool enabled) {
                bridge->RequestRewardDebugEnabled(name.toStdString(), enabled);
                UpdateSelectionFromWidgetToggle(name, enabled);
            });

    // Emit RewardVizConfigChanged when norm/display/force-all change
    connect(_normCombo, qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this](int) { EmitVizConfig(); });
    connect(_displayCombo, qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this](int) { EmitVizConfig(); });
    connect(_forceDebugCheck, &QCheckBox::checkStateChanged, this,
            [this](Qt::CheckState) { EmitVizConfig(); });

    connect(_criticHeadsCheck, &QCheckBox::checkStateChanged, this,
            [this](Qt::CheckState state) {
                const bool show = state == Qt::Checked;
                _widget->SetShowCriticHeads(show);
                emit ShowCriticHeadsChanged(show);
            });

    connect(_widget, &RewardVisualizerWidget::CriticHeadToggled,
            this, &RewardVisualizerWindow::UpdateCriticHeadSelection);
}

void RewardVisualizerWindow::InitFromSettings(const ForgeDebug::DebugSettings& s)
{
    QSignalBlocker b1(_normCombo), b2(_displayCombo),
                   b3(_forceDebugCheck), b4(_playerSpin), b5(_criticHeadsCheck);
    _normCombo->setCurrentIndex((int)s.normMode);
    _displayCombo->setCurrentIndex((int)s.displayMode);
    _forceDebugCheck->setChecked(s.forceDebugAll);
    _playerSpin->setValue(s.playerIndex);

    _criticHeadsCheck->setChecked(s.showCriticHeads);
    _criticHeadSelection    = s.debugEnabledCriticHeads;
    _hasCriticHeadSelection = s.hasCriticHeadSelection;
    _widget->SetShowCriticHeads(s.showCriticHeads);
    _widget->SetCriticHeadSelection(_criticHeadSelection, _hasCriticHeadSelection);
    _widget->ResetSession();
}

void RewardVisualizerWindow::ReadIntoSettings(ForgeDebug::DebugSettings& s) const
{
    s.normMode      = (ForgeDebug::NormalizationMode)_normCombo->currentIndex();
    s.displayMode   = (ForgeDebug::RewardDisplayMode)_displayCombo->currentIndex();
    s.forceDebugAll = _forceDebugCheck->isChecked();
    s.playerIndex   = _playerSpin->value();

    s.showCriticHeads         = _criticHeadsCheck->isChecked();
    s.debugEnabledCriticHeads = _criticHeadSelection;
    s.hasCriticHeadSelection  = _hasCriticHeadSelection;
}

void RewardVisualizerWindow::OnSnapshotReady(const ForgeDebug::RewardDebugSnapshot& snapshot)
{
    // Update the table
    _widget->UpdateSnapshot(snapshot);

    _criticHeadsCheck->setEnabled(snapshot.critic && snapshot.critic->isDecomposed);

    _lastCriticHeadNames.clear();
    if (snapshot.critic)
        for (const ForgeDebug::CriticHeadDebugData& head : snapshot.critic->heads)
            _lastCriticHeadNames.push_back(head.name);

    // Sync _desiredSelection from the roster only when no user toggles are
    // pending. While pending, the roster lags by one trainer round-trip; blindly
    // applying it here would revert an optimistic click just registered above.
    if (!_widget->HasPendingOverrides()) {
        QStringList sel;
        for (const auto& e : snapshot.roster)
            if (e.desiredEnabled)
                sel.append(QString::fromStdString(e.name));

        for (const QString& name : _desiredSelection) {
            const bool inRoster = std::any_of(snapshot.roster.begin(), snapshot.roster.end(),
                [&](const ForgeDebug::RewardRosterEntry& e) { return QString::fromStdString(e.name) == name; });
            if (!inRoster)
                sel.append(name);
        }

        if (sel != _desiredSelection) {
            _desiredSelection = sel;
            emit SelectionChanged(_desiredSelection);
        }
    }

    // Sync controls to the actual collector state reported in the snapshot.
    // QSignalBlocker prevents these updates from firing Request* slots back.
    {
        QSignalBlocker b1(_normCombo);
        QSignalBlocker b2(_displayCombo);
        QSignalBlocker b3(_playerSpin);

        _normCombo->setCurrentIndex(static_cast<int>(snapshot.normMode));
        _displayCombo->setCurrentIndex(static_cast<int>(snapshot.rewardDisplayMode));
        _playerSpin->setValue(snapshot.playerIndex);
    }
}

void RewardVisualizerWindow::UpdateSelectionFromWidgetToggle(const QString& name, bool enabled)
{
    if (enabled) {
        if (!_desiredSelection.contains(name))
            _desiredSelection.append(name);
    } else {
        _desiredSelection.removeAll(name);
    }
    emit SelectionChanged(_desiredSelection);
}

void RewardVisualizerWindow::UpdateCriticHeadSelection(const QString& headName, bool enabled)
{
    _criticHeadSelection = ForgeGUI::ToggleCriticHeadSelection(
        _criticHeadSelection, _hasCriticHeadSelection, _lastCriticHeadNames, headName.toStdString(), enabled);
    _hasCriticHeadSelection = true;
    _widget->SetCriticHeadSelection(_criticHeadSelection, _hasCriticHeadSelection);

    QStringList names;
    for (const std::string& name : _criticHeadSelection)
        names.append(QString::fromStdString(name));
    emit CriticHeadSelectionChanged(names);
}

void RewardVisualizerWindow::EmitVizConfig()
{
    emit RewardVizConfigChanged(_normCombo->currentIndex(),
                                _displayCombo->currentIndex(),
                                _forceDebugCheck->isChecked());
}
