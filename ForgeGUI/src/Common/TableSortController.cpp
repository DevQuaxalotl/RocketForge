#include "TableSortController.h"

#include <QHeaderView>

void TableSortController::Configure(const Config& config)
{
    _config = config;
}

void TableSortController::AttachHeader(QTableWidget* table, std::function<void()> onResort)
{
    _table = table;
    _onResort = std::move(onResort);

    QObject::connect(
        table->horizontalHeader(),
        &QHeaderView::sectionClicked,
        table,
        [this](int uiColumn) { OnHeaderClicked(uiColumn); });
}

void TableSortController::OnHeaderClicked(int uiColumn)
{
    const int priorityColumn = PriorityColumnFromUiIndex(uiColumn);
    if (priorityColumn < 0)
        return;

    if (_sortColumn == priorityColumn) {
        _sortOrder = _sortOrder == Qt::AscendingOrder ? Qt::DescendingOrder : Qt::AscendingOrder;
    } else {
        _sortColumn = priorityColumn;
        _sortOrder = Qt::AscendingOrder;
    }

    if (_table && _table->horizontalHeader()) {
        _table->horizontalHeader()->setSortIndicatorShown(true);
        _table->horizontalHeader()->setSortIndicator(UiIndexFromPriorityColumn(_sortColumn), _sortOrder);
    }

    if (_onResort)
        _onResort();
}

void TableSortController::SetSort(int priorityColumn, Qt::SortOrder order)
{
    _sortColumn = priorityColumn;
    _sortOrder = order;

    if (_table && _table->horizontalHeader() && priorityColumn >= 0) {
        _table->horizontalHeader()->setSortIndicatorShown(true);
        _table->horizontalHeader()->setSortIndicator(UiIndexFromPriorityColumn(priorityColumn), order);
    }
}

int TableSortController::PriorityColumnFromUiIndex(int uiIndex) const
{
    if (uiIndex < 0 || uiIndex >= (int)_config.columnsInUiOrder.size())
        return -1;
    return _config.columnsInUiOrder[uiIndex];
}

int TableSortController::UiIndexFromPriorityColumn(int priorityColumn) const
{
    for (int index = 0; index < (int)_config.columnsInUiOrder.size(); index++)
        if (_config.columnsInUiOrder[index] == priorityColumn)
            return index;
    return -1;
}

int TableSortController::CompareStrings(const QString& lhs, const QString& rhs)
{
    const int result = QString::localeAwareCompare(lhs, rhs);
    if (result < 0)
        return -1;
    if (result > 0)
        return 1;
    return 0;
}

int TableSortController::CompareRows(int lhsRow, int rhsRow) const
{
    int result = 0;
    int sortColumn = _sortColumn;

    // When no column is selected, sort by the configured default priority column.
    if (sortColumn < 0)
        sortColumn = _config.defaultPriorityColumn;

    // Termination guard: mark each priority column visited and never revisit it. The
    // "restart tiebreak from priority 0" rule can otherwise cycle forever when every
    // priority column in [0..selectedColumn] ties for two distinct rows. Skipping
    // already-visited columns bounds the walk to at most columnCount steps while leaving
    // the visited order (and therefore the observable sort) unchanged for every case that
    // previously terminated -- those exit at the first non-tie before any column repeats.
    std::vector<bool> visited(_config.columnCount > 0 ? _config.columnCount : 0, false);

    while (result == 0 && sortColumn >= 0 && sortColumn < _config.columnCount) {
        result = _config.compareColumn(lhsRow, rhsRow, sortColumn);
        visited[sortColumn] = true;

        if (result != 0)
            break;

        // Selected column tied: restart the tiebreak from the highest-priority column
        // (unless it was already priority 0). Otherwise advance to the next priority.
        int nextColumn = (sortColumn == _sortColumn && _sortColumn != 0) ? 0 : sortColumn + 1;

        // Skip columns already compared so the walk visits each priority at most once.
        while (nextColumn < _config.columnCount && visited[nextColumn])
            nextColumn++;

        sortColumn = nextColumn;
    }

    if (result == 0 && _config.rowKey)
        result = CompareStrings(
            QString::fromStdString(_config.rowKey(lhsRow)),
            QString::fromStdString(_config.rowKey(rhsRow)));

    return _sortOrder == Qt::AscendingOrder ? result : -result;
}
