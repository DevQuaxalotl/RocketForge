#pragma once

#include <QWidget>
#include <unordered_map>
#include <string>
#include <vector>

#include <ForgeDebugger.h>
#include "RewardBoxLayout.h"
#include "RewardDisplayRow.h"
#include "ValueTrendTracker.h"

class QLabel;
class CategoryRewardTable;
class CriticTotalStrip;

/**
 * RewardVisualizerWidget
 *
 * Coordinator for the reward debugger. Splits each RewardDebugSnapshot into two
 * category boxes (Continuous, Events), sorts each, and divides the available
 * vertical space so each box shows its enabled rewards (disabled rewards scroll).
 * Below them, a Critic strip shows the critic's total value,
 * and for a decomposed critic a Critic Heads box lists each head.
 * A footer shows the continuous/event/total weighted sums.
 *
 * UpdateSnapshot() must be called on the main/GUI thread.
 */
class RewardVisualizerWidget : public QWidget {
    Q_OBJECT
public:
    explicit RewardVisualizerWidget(QWidget* parent = nullptr);

    /** True while any user checkbox toggle has not yet been confirmed by the trainer. */
    bool HasPendingOverrides() const;

    void SetShowCriticHeads(bool show);
    void SetCriticHeadSelection(const std::vector<std::string>& names, bool hasSelection);

    /** Clears trends and the total bar's running scale; call when a new render session starts. */
    void ResetSession();

signals:
    void RewardToggled(const QString& name, bool enabled);
    void CriticHeadToggled(const QString& headName, bool enabled);

public slots:
    void UpdateSnapshot(const ForgeDebug::RewardDebugSnapshot& snapshot);
    void SetSortColumn(int sortColumn);
    void SetSortDescending(bool descending);

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    void Relayout();
    void UpdateCritic(const ForgeDebug::RewardDebugSnapshot& snapshot);

    CategoryRewardTable* _continuousBox = nullptr;
    CategoryRewardTable* _eventBox      = nullptr;
    CriticTotalStrip*    _criticStrip     = nullptr;
    CategoryRewardTable* _criticHeadsBox  = nullptr;
    QLabel*              _footerLabel   = nullptr;

    ForgeGUI::ValueTrendTracker _rewardTrends;
    ForgeGUI::ValueTrendTracker _totalTrend;
    ForgeGUI::ValueTrendTracker _headTrends;
    float _maxAbsCriticTotal = 0.f;
    float _peakAbsHeadContribution = 0.f;
    bool _headsForceDebugAll = false;
    int _focusPlayerIndex = -1;

    bool _showCriticHeads = true;
    std::vector<std::string> _criticHeadSelection;
    bool _hasCriticHeadSelection = false;
    ForgeGUI::RewardBoxMetrics _lastCriticMetrics;

    struct HeldValues { float norm, weighted; };
    std::unordered_map<std::string, HeldValues> _heldEventValues;

    ForgeGUI::RewardBoxMetrics _lastContMetrics;
    ForgeGUI::RewardBoxMetrics _lastEvtMetrics;

    bool _hasSnapshot = false;
    ForgeDebug::RewardDebugSnapshot _lastSnapshot;

    ForgeGUI::RewardSortKey _sortKey = ForgeGUI::RewardSortKey::AbsSum;
    bool _sortDescending = true;
};
