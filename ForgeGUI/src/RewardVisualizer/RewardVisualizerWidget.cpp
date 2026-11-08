#include "RewardVisualizerWidget.h"
#include "CategoryRewardTable.h"
#include "CriticHeadRows.h"
#include "CriticTotalStrip.h"
#include "ValueDisplay.h"
#include "ThemeUtils.h"

#include <QVBoxLayout>
#include <QLabel>
#include <QResizeEvent>

#include <algorithm>
#include <cmath>

using namespace ForgeGUI;

RewardVisualizerWidget::RewardVisualizerWidget(QWidget* parent)
    : QWidget(parent)
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(4);

    _continuousBox = new CategoryRewardTable("Continuous", this);
    _eventBox      = new CategoryRewardTable("Events", this);
    layout->addWidget(_continuousBox);
    layout->addWidget(_eventBox);

    _criticStrip    = new CriticTotalStrip(this);
    _criticHeadsBox = new CategoryRewardTable("Critic Heads", this);
    _criticStrip->setVisible(false);
    _criticHeadsBox->setVisible(false);
    layout->addWidget(_criticStrip);
    layout->addWidget(_criticHeadsBox);
    layout->addStretch(1);   // absorbs slack so the boxes stay adjacent

    _footerLabel = new QLabel(this);
    _footerLabel->setProperty(ForgeGUI::TextRoleProperty, "caption");
    layout->addWidget(_footerLabel);

    connect(_continuousBox, &CategoryRewardTable::RewardToggled,
            this, &RewardVisualizerWidget::RewardToggled);
    connect(_eventBox, &CategoryRewardTable::RewardToggled,
            this, &RewardVisualizerWidget::RewardToggled);
    connect(_criticHeadsBox, &CategoryRewardTable::RewardToggled,
            this, &RewardVisualizerWidget::CriticHeadToggled);
}

bool RewardVisualizerWidget::HasPendingOverrides() const {
    return _continuousBox->HasPendingOverrides() || _eventBox->HasPendingOverrides();
}

void RewardVisualizerWidget::SetShowCriticHeads(bool show) {
    _showCriticHeads = show;
    if (_hasSnapshot) UpdateSnapshot(_lastSnapshot);
}

void RewardVisualizerWidget::SetCriticHeadSelection(const std::vector<std::string>& names, bool hasSelection) {
    _criticHeadSelection = names;
    _hasCriticHeadSelection = hasSelection;
    _peakAbsHeadContribution = 0.f;
    if (_hasSnapshot) UpdateSnapshot(_lastSnapshot);
}

void RewardVisualizerWidget::ResetSession() {
    _rewardTrends.Clear();
    _totalTrend.Clear();
    _headTrends.Clear();
    _maxAbsCriticTotal = 0.f;
    _peakAbsHeadContribution = 0.f;
    _focusPlayerIndex = -1;

    _hasSnapshot = false;
    _criticStrip->setVisible(false);
    _criticHeadsBox->setVisible(false);
}

void RewardVisualizerWidget::UpdateCritic(const ForgeDebug::RewardDebugSnapshot& snapshot) {
    if (!snapshot.critic) {
        _criticStrip->setVisible(false);
        _criticHeadsBox->setVisible(false);
        return;
    }

    const ForgeDebug::CriticDebugData& critic = *snapshot.critic;

    // The head bars share a peak over the enabled heads, so it restarts whenever that set changes.
    if (snapshot.forceDebugAll != _headsForceDebugAll) {
        _peakAbsHeadContribution = 0.f;
        _headsForceDebugAll = snapshot.forceDebugAll;
    }

    _totalTrend.ObserveEpisode(snapshot.episodeIndex);
    _headTrends.ObserveEpisode(snapshot.episodeIndex);
    _totalTrend.Record("total", snapshot.gameTimeSeconds, critic.total);
    for (const ForgeDebug::CriticHeadDebugData& head : critic.heads)
        _headTrends.Record(head.name, snapshot.gameTimeSeconds, head.contribution);

    _maxAbsCriticTotal = std::max(_maxAbsCriticTotal, std::abs(critic.total));
    const float totalNorm = _maxAbsCriticTotal > 0.f ? critic.total / _maxAbsCriticTotal : 0.f;
    _criticStrip->SetTotal(critic.total, totalNorm, _totalTrend.Get("total"));
    _criticStrip->setVisible(true);

    const bool showHeads = critic.isDecomposed && _showCriticHeads;
    _criticHeadsBox->setVisible(showHeads);
    if (!showHeads)
        return;

    std::vector<RewardDisplayRow> rows = BuildCriticHeadRows(
        critic,
        _criticHeadSelection,
        _hasCriticHeadSelection,
        snapshot.forceDebugAll,
        _headTrends,
        _peakAbsHeadContribution);
    SortRewardDisplayRows(rows, _sortKey, _sortDescending);

    const int enabledCount = (int)std::count_if(rows.begin(), rows.end(), [](const RewardDisplayRow& r) { return r.enabled; });
    _lastCriticMetrics = { enabledCount, (int)rows.size() };
    _criticHeadsBox->SetRows(rows);
}

void RewardVisualizerWidget::SetSortColumn(int sortColumn) {
    _sortKey = static_cast<RewardSortKey>(std::clamp(sortColumn, (int)RewardSortKey::Name, (int)RewardSortKey::Norm));
    if (_hasSnapshot) UpdateSnapshot(_lastSnapshot);
}

void RewardVisualizerWidget::SetSortDescending(bool descending) {
    _sortDescending = descending;
    if (_hasSnapshot) UpdateSnapshot(_lastSnapshot);
}

void RewardVisualizerWidget::UpdateSnapshot(const ForgeDebug::RewardDebugSnapshot& snapshot) {
    _lastSnapshot = snapshot;
    _hasSnapshot = true;

    // Trends and running scales are per player, so a focus change must not compare against the previous player's values.
    if (snapshot.playerIndex != _focusPlayerIndex) {
        _rewardTrends.Clear();
        _totalTrend.Clear();
        _headTrends.Clear();
        _peakAbsHeadContribution = 0.f;
        _focusPlayerIndex = snapshot.playerIndex;
    }

    _rewardTrends.ObserveEpisode(snapshot.episodeIndex);

    std::unordered_map<std::string, const ForgeDebug::RewardDebugData*> live;
    for (const auto& rd : snapshot.rewards)
        live[rd.name] = &rd;

    std::vector<RewardDisplayRow> contRows, evtRows;
    for (const auto& entry : snapshot.roster) {
        RewardDisplayRow row;
        row.index          = entry.index;
        row.name           = entry.name;
        row.desiredEnabled = entry.desiredEnabled;
        auto it = live.find(entry.name);
        if (it != live.end()) {
            const auto& rd = *it->second;
            row.enabled         = true;
            row.displayNorm     = rd.normalizedValue;
            row.displayWeighted = rd.weightedValue;
            row.decayAlpha      = 1.0f;
            row.absSum          = rd.episodeAbsWeightedSum;

            if (rd.category == RLGC::RewardCategory::EVENT) {
                if (rd.eventDecayValue > 0.f) {
                    if (rd.eventFiredThisFrame)
                        _heldEventValues[rd.name] = { rd.normalizedValue, rd.weightedValue };
                    auto held = _heldEventValues.find(rd.name);
                    if (held != _heldEventValues.end()) {
                        row.displayNorm     = held->second.norm;
                        row.displayWeighted = held->second.weighted;
                    }
                    row.decayAlpha = rd.eventDecayValue;
                } else {
                    _heldEventValues.erase(rd.name);
                    row.displayNorm = 0.f; row.displayWeighted = 0.f; row.decayAlpha = 0.f;
                }
            }

            _rewardTrends.Record(rd.name, snapshot.gameTimeSeconds, row.displayWeighted);
            const ForgeGUI::TrendReading trend = _rewardTrends.Get(rd.name);
            row.trendDelta     = trend.delta;
            row.trendDirection = trend.direction;
        } else if (entry.category == RLGC::RewardCategory::EVENT) {
            _heldEventValues.erase(entry.name);
        }

        if (entry.category == RLGC::RewardCategory::EVENT)
            evtRows.push_back(row);
        else
            contRows.push_back(row);
    }

    ScaleTrendIntensities(contRows);
    ScaleTrendIntensities(evtRows);
    SortRewardDisplayRows(contRows, _sortKey, _sortDescending);
    SortRewardDisplayRows(evtRows, _sortKey, _sortDescending);

    const auto countEnabled = [](const std::vector<RewardDisplayRow>& rows) {
        int n = 0;
        for (const auto& r : rows) if (r.enabled) ++n;
        return n;
    };
    _lastContMetrics = { countEnabled(contRows), (int)contRows.size() };
    _lastEvtMetrics  = { countEnabled(evtRows),  (int)evtRows.size() };

    _continuousBox->SetRows(contRows);
    _eventBox->SetRows(evtRows);

    _footerLabel->setText(
        QString("Continuous: %1   Events: %2   Total: %3")
            .arg(FormatValue(snapshot.totalContinuousWeighted))
            .arg(FormatValue(snapshot.totalEventWeighted))
            .arg(FormatValue(snapshot.totalWeighted)));

    UpdateCritic(snapshot);
    Relayout();
}

void RewardVisualizerWidget::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    Relayout();
}

void RewardVisualizerWidget::Relayout() {
    const bool criticStripShown = !_criticStrip->isHidden();
    const bool criticHeadsShown = !_criticHeadsBox->isHidden();

    std::vector<CategoryRewardTable*> boxes = { _continuousBox, _eventBox };
    std::vector<RewardBoxMetrics> metrics = { _lastContMetrics, _lastEvtMetrics };
    if (criticHeadsShown) {
        boxes.push_back(_criticHeadsBox);
        metrics.push_back(_lastCriticMetrics);
    }

    // Everything but the tables is sized from hints and margins, never from current geometry:
    // when the boxes don't fit, Qt squeezes them, and measuring that would feed back into the next allocation.
    int reserved = _footerLabel->sizeHint().height();
    int shownWidgets = 1;
    if (criticStripShown) {
        reserved += _criticStrip->sizeHint().height();
        shownWidgets++;
    }

    for (const CategoryRewardTable* box : boxes)
        reserved += box->ChromeHeight();
    shownWidgets += (int)boxes.size();

    reserved += layout()->spacing() * (shownWidgets - 1);

    const int available = std::max(0, height() - reserved);

    // Use the tables' *measured* row/header pixel sizes rather than assuming the
    // 20/24 defaults. On Windows/Qt6 rows render a few pixels taller than the
    // nominal section size, so a height computed from the defaults would clip the
    // last enabled rows below the fold and force scrolling. minRows defaults are
    // fine; extraVisibleRows=2 keeps a small disabled buffer visible.
    RewardBoxLayoutConfig cfg;
    cfg.rowHeight    = 1;
    cfg.headerHeight = 1;
    for (const CategoryRewardTable* box : boxes) {
        cfg.rowHeight    = std::max(cfg.rowHeight,    box->RowPixelHeight());
        cfg.headerHeight = std::max(cfg.headerHeight, box->HeaderPixelHeight());
    }
    cfg.extraVisibleRows = 2;

    const std::vector<int> heights = AllocateBoxHeights(available, metrics, cfg);
    for (size_t i = 0; i < boxes.size(); i++)
        boxes[i]->SetAllocatedHeight(heights[i]);
}
