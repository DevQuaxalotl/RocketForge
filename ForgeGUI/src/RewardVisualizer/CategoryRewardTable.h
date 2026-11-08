#pragma once

#include <QGroupBox>
#include <unordered_map>
#include <string>
#include <vector>

#include "RewardDisplayRow.h"

class QTableWidget;

/**
 * CategoryRewardTable
 *
 * A titled box (QGroupBox) wrapping one QTableWidget of checkbox rows
 * Columns: On | Name | Weighted | Norm | Abs Sum | Δ (trend) | Bar.
 * Owns its optimistic toggle-override map and its row-layout cache;
 * the cache means clearContents() runs only when the layout changes, so checkbox clicks survive between frames.
 *
 * SetRows() expects rows pre-sorted (enabled first, then disabled).
 * SetAllocatedHeight() sets the inner table's fixed height; disabled rows fall
 * below the fold and are reached by scrolling.
 */
class CategoryRewardTable : public QGroupBox {
    Q_OBJECT
public:
    explicit CategoryRewardTable(const QString& title, QWidget* parent = nullptr);

    void SetRows(const std::vector<ForgeGUI::RewardDisplayRow>& rows);
    void SetAllocatedHeight(int px);
    /** Height the box adds around its table (title, frame, margins), independent of the box's current geometry. */
    int  ChromeHeight() const;

    /** Actual rendered height of one table row, in pixels. Falls back to the
     *  default section size when the table has no rows yet. **/
    int  RowPixelHeight() const;
    /** Actual rendered height of the table's horizontal header, in pixels. **/
    int  HeaderPixelHeight() const;

    bool HasPendingOverrides() const { return !_localDesiredOverride.empty(); }

signals:
    /** Keyed by name because a reward's index is only stable until the debug env's reward list is replaced. */
    void RewardToggled(const QString& name, bool enabled);

private slots:
    void OnCellChanged(int row, int column);

private:
    void WriteRosterRow(int row, const ForgeGUI::RewardDisplayRow& r);
    void WriteRowValues(int row, const ForgeGUI::RewardDisplayRow& r);
    void SetCell(int row, int col, const QString& text);

    /** The table frame's top + bottom border, which the table's fixed height must include on top of its rows. */
    int  TableFrameHeight() const;

    QTableWidget* _table = nullptr;

    std::vector<std::string> _rowRewardName;      // row -> RewardDisplayRow::name
    std::vector<std::string> _currentRowLayout;   // RewardDisplayRow::name per row, last rebuild
    std::unordered_map<std::string, bool> _localDesiredOverride;

    enum Column {
        ColEnabled, ColName, ColWeighted, ColNorm, ColAbsSum, ColTrend, ColBar, ColCount
    };
};
