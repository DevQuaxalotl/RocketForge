#include "CategoryRewardTable.h"
#include "ValueDisplay.h"
#include "ThemeUtils.h"

#include <QVBoxLayout>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QHeaderView>
#include <QPainter>
#include <QStyledItemDelegate>
#include <QStyleOptionViewItem>
#include <QSignalBlocker>

#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------
// BarDelegate — bipolar [-1, 1] bar with decay-alpha for events.
//   UserRole   = float normalizedValue (bar extent, -1..1)
//   UserRole+1 = float decayAlpha      (1.0 opaque .. 0.0 invisible)
// ---------------------------------------------------------------------------
class BarDelegate : public QStyledItemDelegate {
public:
    using QStyledItemDelegate::QStyledItemDelegate;

    void paint(QPainter* painter, const QStyleOptionViewItem& option,
               const QModelIndex& index) const override {
        const float value = static_cast<float>(index.data(Qt::UserRole).toDouble());
        const float alpha = static_cast<float>(index.data(Qt::UserRole + 1).toDouble());
        PaintBipolarBar(painter, option.rect, option.palette, value, alpha);
    }
};

// ---------------------------------------------------------------------------
// CategoryRewardTable
// ---------------------------------------------------------------------------
CategoryRewardTable::CategoryRewardTable(const QString& title, QWidget* parent)
    : QGroupBox(title, parent)
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(4, 2, 4, 4);
    layout->setSpacing(0);

    _table = new QTableWidget(0, ColCount, this);
    _table->setHorizontalHeaderLabels({ "On", "Name", "Weighted", "Norm", "Abs Sum", QString(QChar(0x0394)), "Bar" });
    _table->horizontalHeader()->setSectionResizeMode(ColEnabled,  QHeaderView::ResizeToContents);
    _table->horizontalHeader()->setSectionResizeMode(ColName,     QHeaderView::Stretch);
    _table->horizontalHeader()->setSectionResizeMode(ColBar,      QHeaderView::Stretch);

    const auto fixColumnWidth = [this](int column, const QString& widestValue) {
        const QString header = _table->horizontalHeaderItem(column)->text();
        const int width = std::max(
            FixedTextWidth(_table->fontMetrics(), widestValue),
            FixedTextWidth(_table->horizontalHeader()->fontMetrics(), header));

        _table->horizontalHeader()->setSectionResizeMode(column, QHeaderView::Fixed);
        _table->horizontalHeader()->resizeSection(column, width);
    };

    const int rising = 1;
    fixColumnWidth(ColWeighted, FormatValue(-888.888f));
    fixColumnWidth(ColNorm,     FormatValue(-1.f));
    fixColumnWidth(ColAbsSum,   FormatValue(8888.888f));
    fixColumnWidth(ColTrend,    FormatTrendText(rising, 888.888f));
    _table->verticalHeader()->setVisible(false);
    _table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _table->setSelectionMode(QAbstractItemView::NoSelection);
    _table->setFocusPolicy(Qt::NoFocus);

    // A horizontal bar would steal table height whenever the vertical bar narrows the columns, resizing the rows shown.
    _table->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    _table->setItemDelegateForColumn(ColBar, new BarDelegate(_table));
    _table->setShowGrid(false);
    _table->setAlternatingRowColors(false);
    _table->verticalHeader()->setDefaultSectionSize(20);

    connect(_table, &QTableWidget::cellChanged, this, &CategoryRewardTable::OnCellChanged);

    layout->addWidget(_table);
}

void CategoryRewardTable::SetAllocatedHeight(int px) {
    _table->setFixedHeight(std::max(px, 0) + TableFrameHeight());
}

int CategoryRewardTable::ChromeHeight() const {
    const QMargins boxMargins    = contentsMargins();
    const QMargins layoutMargins = layout()->contentsMargins();
    return boxMargins.top() + boxMargins.bottom() + layoutMargins.top() + layoutMargins.bottom() + TableFrameHeight();
}

int CategoryRewardTable::TableFrameHeight() const {
    return 2 * _table->frameWidth();
}

int CategoryRewardTable::RowPixelHeight() const {
    if (_table->rowCount() > 0) {
        const int h = _table->rowHeight(0);
        if (h > 0)
            return h;
    }
    return _table->verticalHeader()->defaultSectionSize();
}

int CategoryRewardTable::HeaderPixelHeight() const {
    const int h = _table->horizontalHeader()->height();
    return h > 0 ? h : _table->horizontalHeader()->sizeHint().height();
}

void CategoryRewardTable::SetCell(int row, int col, const QString& text) {
    QTableWidgetItem* item = _table->item(row, col);
    if (!item) {
        item = new QTableWidgetItem;
        if (col == ColName)
            item->setTextAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        else if (col == ColTrend)
            item->setTextAlignment(Qt::AlignCenter);
        else
            item->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
        _table->setItem(row, col, item);
    }
    item->setText(text);
}

void CategoryRewardTable::WriteRosterRow(int row, const ForgeGUI::RewardDisplayRow& r) {
    _rowRewardName[row] = r.name;

    QTableWidgetItem* chk = _table->item(row, ColEnabled);
    if (!chk) {
        chk = new QTableWidgetItem;
        chk->setTextAlignment(Qt::AlignCenter);
        _table->setItem(row, ColEnabled, chk);
    }
    chk->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);

    bool desiredState = r.desiredEnabled;
    auto overrideIt = _localDesiredOverride.find(r.name);
    if (overrideIt != _localDesiredOverride.end()) {
        if (overrideIt->second == r.desiredEnabled)
            _localDesiredOverride.erase(overrideIt);
        else
            desiredState = overrideIt->second;
    }
    chk->setCheckState(desiredState ? Qt::Checked : Qt::Unchecked);

    SetCell(row, ColName, QString::fromStdString(r.name));
    WriteRowValues(row, r);
}

void CategoryRewardTable::WriteRowValues(int row, const ForgeGUI::RewardDisplayRow& r) {
    const QString unavailable(QChar(0x2014));

    if (!r.enabled) {
        SetCell(row, ColWeighted, "");
        SetCell(row, ColNorm,     "");
        SetCell(row, ColAbsSum,   "");
        SetCell(row, ColTrend,    "");
    } else if (!r.hasValue) {
        SetCell(row, ColWeighted, unavailable);
        SetCell(row, ColNorm,     unavailable);
        SetCell(row, ColAbsSum,   unavailable);
        SetCell(row, ColTrend,    unavailable);
    } else {
        SetCell(row, ColWeighted, FormatValue(r.displayWeighted));
        SetCell(row, ColNorm,     FormatValue(r.displayNorm));
        SetCell(row, ColAbsSum,   FormatValue(r.absSum));
        SetCell(row, ColTrend,    FormatTrendText(r.trendDirection, r.trendDelta));
    }

    const bool showBar = r.enabled && r.hasValue;
    QTableWidgetItem* barItem = _table->item(row, ColBar);
    if (!barItem) { barItem = new QTableWidgetItem; _table->setItem(row, ColBar, barItem); }
    barItem->setData(Qt::UserRole,     (double)(showBar ? r.displayNorm : 0.f));
    barItem->setData(Qt::UserRole + 1, (double)(showBar ? r.decayAlpha  : 0.f));

    const QColor fg = r.enabled ? ForgeGUI::Theme::TEXT : ForgeGUI::Theme::TEXT_DISABLED;
    for (int c = ColName; c <= ColTrend; ++c)
        if (QTableWidgetItem* it = _table->item(row, c)) it->setForeground(fg);

    if (showBar)
        if (QTableWidgetItem* it = _table->item(row, ColTrend)) it->setForeground(TrendColor(r.trendDirection, r.trendIntensity));
}

void CategoryRewardTable::SetRows(const std::vector<ForgeGUI::RewardDisplayRow>& rows) {
    const int rowCount = static_cast<int>(rows.size());

    std::vector<std::string> newLayout;
    newLayout.reserve(rowCount);
    for (const ForgeGUI::RewardDisplayRow& r : rows) newLayout.push_back(r.name);

    if (newLayout != _currentRowLayout) {
        // A row that left the table can never confirm its override, which would otherwise stay pending forever.
        std::erase_if(_localDesiredOverride, [&](const auto& entry) {
            return std::find(newLayout.begin(), newLayout.end(), entry.first) == newLayout.end();
        });

        // Layout changed: full rebuild. clearContents() only here keeps checkbox
        // items (and their pending clicks) alive between frames otherwise.
        // Block table signals during the rebuild so the programmatic checkbox
        // writes below aren't mistaken for user toggles (RAII = exception-safe).
        const QSignalBlocker blocker(_table);
        _table->clearContents();
        _table->setRowCount(rowCount);
        _rowRewardName.assign(rowCount, {});
        _currentRowLayout = newLayout;
        for (int i = 0; i < rowCount; ++i)
            WriteRosterRow(i, rows[i]);
    } else {
        for (int i = 0; i < rowCount; ++i)
            WriteRowValues(i, rows[i]);
    }

    _table->viewport()->update();
}

void CategoryRewardTable::OnCellChanged(int row, int column) {
    if (column != ColEnabled)
        return;
    if (row < 0 || row >= (int)_rowRewardName.size())
        return;
    const std::string& rewardName = _rowRewardName[row];
    if (rewardName.empty())
        return;
    QTableWidgetItem* item = _table->item(row, ColEnabled);
    if (!item)
        return;
    const bool enabled = item->checkState() == Qt::Checked;
    _localDesiredOverride[rewardName] = enabled;
    emit RewardToggled(QString::fromStdString(rewardName), enabled);
}
