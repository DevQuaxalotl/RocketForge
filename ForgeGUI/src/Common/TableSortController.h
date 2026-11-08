#pragma once

#include <QString>
#include <QTableWidget>
#include <Qt>

#include <functional>
#include <string>
#include <vector>

/**
 * Reusable hierarchical, priority-based table sort.
 *
 * Columns are identified by an integer priority index (0 = highest priority). The host
 * widget keeps its own row storage and supplies a per-column comparator over row indices;
 * the controller owns the sort state, the fall-through tiebreak algorithm, header-click
 * toggling, and the UI-column <-> priority-column mapping.
 *
 * Sorting semantics (preserved from the original VersionManagementDialog implementation):
 * compare by the selected column (or defaultPriorityColumn when none is selected); on a tie,
 * walk priority columns 0..columnCount-1; if the selected column tied and was not already
 * priority 0, restart the tiebreak from priority 0; finally tiebreak on rowKey. Descending
 * negates the result.
 *
 * NOTE: This class is deliberately not QObject-derived, so it follows the CodeStandards
 * default leading-underscore prefix for members rather than the Qt m_ exception.
 */
class TableSortController
{
public:
    /** Ascending -1/0/1 comparison of two rows for a single priority column. */
    using ColumnComparator = std::function<int(int lhsRow, int rhsRow, int priorityColumn)>;

    /** Stable final tiebreak key so equal rows keep a deterministic order. */
    using RowKeyFn = std::function<std::string(int row)>;

    struct Config
    {
        int columnCount = 0;
        std::vector<int> columnsInUiOrder;
        ColumnComparator compareColumn;
        RowKeyFn rowKey;
        int defaultPriorityColumn = 0;
    };

    void Configure(const Config& config);

    /**
     * Wire header clicks: same column toggles order; a new column selects it ascending.
     * Updates the sort indicator, then calls onResort so the widget can re-sort + repopulate.
     */
    void AttachHeader(QTableWidget* table, std::function<void()> onResort);

    /** Hierarchical compare used by the widget's std::sort predicate over row indices. */
    int CompareRows(int lhsRow, int rhsRow) const;

    int PriorityColumnFromUiIndex(int uiIndex) const;
    int UiIndexFromPriorityColumn(int priorityColumn) const;

    int SortColumn() const { return _sortColumn; }
    Qt::SortOrder SortOrder() const { return _sortOrder; }
    void SetSort(int priorityColumn, Qt::SortOrder order);

    /** Shared comparison primitives (were copy-pasted in both widgets). */
    static int CompareStrings(const QString& lhs, const QString& rhs);

    template <class T>
    static int CompareValues(const T& lhs, const T& rhs)
    {
        if (lhs < rhs)
            return -1;
        if (rhs < lhs)
            return 1;
        return 0;
    }

private:
    void OnHeaderClicked(int uiColumn);

    Config _config;
    QTableWidget* _table = nullptr;
    std::function<void()> _onResort;

    int _sortColumn = -1;   // -1 == unselected
    Qt::SortOrder _sortOrder = Qt::AscendingOrder;
};
