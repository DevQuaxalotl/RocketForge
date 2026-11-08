#include "RewardComparisonWidget.h"
#include "ThemeUtils.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSplitter>
#include <QLabel>
#include <QComboBox>
#include <QCheckBox>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QHeaderView>
#include <QPainter>
#include <QStyledItemDelegate>
#include <QStyleOptionViewItem>
#include <QColor>
#include <QFont>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <iomanip>

using namespace RocketForge;
using namespace ForgeGUI;

// ---------------------------------------------------------------------------
// EffMaxBarDelegate — bidirectional bar: green right (positive), red left (negative)
// ---------------------------------------------------------------------------
// Qt::UserRole     = float posFrac (0..1)  — green bar extends right of centre
// Qt::UserRole + 1 = float negFrac (0..1)  — red bar extends left of centre
class EffMaxBarDelegate : public QStyledItemDelegate
{
public:
    using QStyledItemDelegate::QStyledItemDelegate;

    void paint(QPainter* painter, const QStyleOptionViewItem& option,
               const QModelIndex& index) const override
    {
        const float pos = std::clamp(
            static_cast<float>(index.data(Qt::UserRole).toDouble()), 0.f, 1.f);
        const float neg = std::clamp(
            static_cast<float>(index.data(Qt::UserRole + 1).toDouble()), 0.f, 1.f);

        painter->fillRect(option.rect, option.palette.base());

        const int top    = option.rect.top()  + 2;
        const int height = option.rect.height() - 4;
        const int cx     = option.rect.left() + option.rect.width() / 2;

        if (height > 0) {
            // Green bar right (positive side)
            if (pos > 0.001f) {
                int barW = static_cast<int>(pos * option.rect.width() / 2);
                if (barW > 0)
                    painter->fillRect(QRect(cx, top, barW, height), Theme::POSITIVE);
            }
            // Red bar left (negative side)
            if (neg > 0.001f) {
                int barW = static_cast<int>(neg * option.rect.width() / 2);
                if (barW > 0)
                    painter->fillRect(QRect(cx - barW, top, barW, height), Theme::NEGATIVE);
            }
        }

        painter->setPen(Theme::AXIS);
        painter->drawLine(cx, option.rect.top(), cx, option.rect.bottom());
    }
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string TruncStr(const std::string& s, int maxLen)
{
    return (int)s.size() <= maxLen ? s : s.substr(0, maxLen - 3) + "...";
}

static QString FormatRatio(float r)
{
    if (std::isnan(r))
        return "N/A";
    const float absR = std::abs(r);
    if (absR >= 1000.f)
        return QString::number(static_cast<double>(r), 'g', 3);
    if (absR >= 100.f)
        return QString::number(static_cast<double>(r), 'f', 0);
    return QString::number(static_cast<double>(r), 'f', 2);
}

static QTableWidgetItem* MakeItem(const QString& text = {})
{
    auto* item = new QTableWidgetItem(text);
    item->setTextAlignment(Qt::AlignCenter);
    item->setFlags(Qt::ItemIsEnabled);
    return item;
}

static bool IsEvent(const RewardTableEntry& e)
{
    return e.category == RLGC::RewardCategory::EVENT;
}

// ---------------------------------------------------------------------------
// RewardComparisonWidget
// ---------------------------------------------------------------------------

RewardComparisonWidget::RewardComparisonWidget(const RewardComparisonTable& table,
                                               QWidget* parent)
    : QWidget(parent, parent ? Qt::Widget : Qt::Window)
    , _table(table)
{
    setMinimumSize(600, 500);
    resize(900, 680);

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(6, 6, 6, 6);
    root->setSpacing(4);

    auto* controlRow = new QWidget(this);
    BuildControls(controlRow);
    root->addWidget(controlRow);

    auto* splitter = new QSplitter(Qt::Vertical, this);
    splitter->setChildrenCollapsible(false);

    auto* topWidget = new QWidget(splitter);
    BuildEffectiveMaxPanel(topWidget);

    auto* bottomWidget = new QWidget(splitter);
    auto* botLayout = new QVBoxLayout(bottomWidget);
    botLayout->setContentsMargins(0, 0, 0, 0);
    botLayout->setSpacing(2);

    auto* ratioLabel = new QLabel(
        "Ratio Matrix   table[row][col] = EffMax(row) / EffMax(col)"
        "   —   Read: \"1 [row] = X [col]\"",
        bottomWidget);
    ratioLabel->setProperty(ForgeGUI::TextRoleProperty, "caption");
    botLayout->addWidget(ratioLabel);

    BuildRatioTable();
    botLayout->addWidget(_ratioTable);

    splitter->addWidget(topWidget);
    splitter->addWidget(bottomWidget);
    splitter->setStretchFactor(0, 2);
    splitter->setStretchFactor(1, 3);

    root->addWidget(splitter);

    OnSortChanged(0);
}

void RewardComparisonWidget::BuildControls(QWidget* container)
{
    auto* row = new QHBoxLayout(container);
    row->setContentsMargins(0, 0, 0, 0);
    row->setSpacing(10);

    row->addWidget(new QLabel("Sort:", container));
    _sortCombo = new QComboBox(container);
    _sortCombo->addItem("Default");
    _sortCombo->addItem("Effective Max (descending)");
    _sortCombo->addItem("Category (Event first)");
    connect(_sortCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &RewardComparisonWidget::OnSortChanged);
    row->addWidget(_sortCombo);

    _splitCatCheck = new QCheckBox("Split bar by category", container);
    _splitCatCheck->setToolTip(
        "Scale event and continuous rewards against their own group maximum in the bar chart, "
        "with a visual separator between the two groups.");
    connect(_splitCatCheck, &QCheckBox::checkStateChanged,
            this, [this](int) { OnSplitChanged(); });
    row->addWidget(_splitCatCheck);

    row->addStretch();
}

void RewardComparisonWidget::BuildEffectiveMaxPanel(QWidget* container)
{
    auto* layout = new QVBoxLayout(container);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(2);

    auto* titleLabel = new QLabel(
        "Effective Max per Reward   green = positive output   red = negative output",
        container);
    titleLabel->setProperty(ForgeGUI::TextRoleProperty, "caption");
    layout->addWidget(titleLabel);

    const int n = static_cast<int>(_table.rewards.size());
    _effMaxTable = new QTableWidget(n, 5, container);
    _effMaxTable->setHorizontalHeaderLabels({ "Name", "Weight", "Range", "Eff Max", "Bar" });
    _effMaxTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    _effMaxTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    _effMaxTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    _effMaxTable->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    _effMaxTable->horizontalHeader()->setSectionResizeMode(4, QHeaderView::Stretch);
    _effMaxTable->verticalHeader()->setVisible(false);
    _effMaxTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _effMaxTable->setSelectionMode(QAbstractItemView::NoSelection);
    _effMaxTable->setFocusPolicy(Qt::NoFocus);
    _effMaxTable->setShowGrid(false);
    _effMaxTable->setAlternatingRowColors(false);
    _effMaxTable->verticalHeader()->setDefaultSectionSize(22);
    _effMaxTable->setItemDelegateForColumn(4, new EffMaxBarDelegate(_effMaxTable));

    layout->addWidget(_effMaxTable);
}

void RewardComparisonWidget::BuildRatioTable()
{
    const int n = static_cast<int>(_table.rewards.size());
    _ratioTable = new QTableWidget(n, n, nullptr);
    _ratioTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _ratioTable->setSelectionMode(QAbstractItemView::NoSelection);
    _ratioTable->setFocusPolicy(Qt::NoFocus);
    _ratioTable->setShowGrid(true);
    _ratioTable->setAlternatingRowColors(false);
    _ratioTable->verticalHeader()->setDefaultSectionSize(24);
    _ratioTable->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    _ratioTable->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    _ratioTable->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
}

// Writes a spanning header row across all 5 columns of _effMaxTable.
void RewardComparisonWidget::SetBarSectionHeader(int row, const QString& text)
{
    _effMaxTable->setSpan(row, 0, 1, 5);
    auto* item = new QTableWidgetItem(text);
    item->setFlags(Qt::NoItemFlags);
    item->setBackground(Theme::SURFACE_RAISED);
    item->setForeground(Theme::TEXT);
    QFont f = item->font();
    f.setBold(true);
    item->setFont(f);
    _effMaxTable->setItem(row, 0, item);
}

void RewardComparisonWidget::PopulateTables(const std::vector<int>& order)
{
    const int n         = static_cast<int>(order.size());
    const bool splitCat = _splitCatCheck && _splitCatCheck->isChecked();

    // --- Bar fractions: positive side = maxValue*weight clamped ≥0,
    //                    negative side = |minValue*weight| clamped ≥0
    // Scale denominator is the max of max(|pos|,|neg|) across the relevant group.
    float maxAbsAll = 0.f, maxAbsEvt = 0.f, maxAbsCont = 0.f;
    for (const auto& e : _table.rewards) {
        const float posAbs = std::max(e.range.maxValue * e.weight, 0.f);
        const float negAbs = std::abs(std::min(e.range.minValue * e.weight, 0.f));
        const float m      = std::max(posAbs, negAbs);
        maxAbsAll = std::max(maxAbsAll, m);
        if (IsEvent(e)) maxAbsEvt  = std::max(maxAbsEvt,  m);
        else             maxAbsCont = std::max(maxAbsCont, m);
    }
    if (maxAbsAll  == 0.f) maxAbsAll  = 1.f;
    if (maxAbsEvt  == 0.f) maxAbsEvt  = 1.f;
    if (maxAbsCont == 0.f) maxAbsCont = 1.f;

    // --- Build bar display list: separators (rewardIdx=-1) + data rows ---
    struct BarRow { int rewardIdx; QString sepLabel; };
    std::vector<BarRow> barRows;

    if (splitCat) {
        std::vector<int> evtIdxs, contIdxs;
        for (int idx : order) {
            if (IsEvent(_table.rewards[idx])) evtIdxs.push_back(idx);
            else                               contIdxs.push_back(idx);
        }
        if (!evtIdxs.empty()) {
            barRows.push_back({ -1, "── Event ──" });
            for (int idx : evtIdxs)  barRows.push_back({ idx, {} });
        }
        if (!contIdxs.empty()) {
            barRows.push_back({ -1, "── Continuous ──" });
            for (int idx : contIdxs) barRows.push_back({ idx, {} });
        }
    } else {
        for (int idx : order) barRows.push_back({ idx, {} });
    }

    // --- Populate _effMaxTable ---
    _effMaxTable->clearSpans();
    _effMaxTable->setRowCount(static_cast<int>(barRows.size()));

    for (int row = 0; row < (int)barRows.size(); row++) {
        const auto& br = barRows[row];

        if (br.rewardIdx < 0) {
            SetBarSectionHeader(row, br.sepLabel);
            continue;
        }

        const auto& e = _table.rewards[br.rewardIdx];

        auto* nameItem = MakeItem(QString::fromStdString(e.name));
        nameItem->setTextAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        nameItem->setToolTip(QString::fromStdString(e.name));
        _effMaxTable->setItem(row, 0, nameItem);

        _effMaxTable->setItem(row, 1,
            MakeItem(QString::number(static_cast<double>(e.weight), 'f', 2)));

        std::ostringstream rs;
        rs << std::fixed << std::setprecision(2)
           << "[" << e.range.minValue << ", " << e.range.maxValue << "]";
        _effMaxTable->setItem(row, 2, MakeItem(QString::fromStdString(rs.str())));

        auto* effItem = MakeItem(
            QString::number(static_cast<double>(e.effectiveMax), 'f', 2));
        if (e.effectiveMax == 0.f)
            effItem->setForeground(Theme::TEXT_DISABLED);
        _effMaxTable->setItem(row, 3, effItem);

        const float denom  = splitCat ? (IsEvent(e) ? maxAbsEvt : maxAbsCont) : maxAbsAll;
        const float posFrac = std::max(e.range.maxValue * e.weight, 0.f) / denom;
        const float negFrac = std::abs(std::min(e.range.minValue * e.weight, 0.f)) / denom;
        auto* barItem = new QTableWidgetItem;
        barItem->setFlags(Qt::ItemIsEnabled);
        barItem->setData(Qt::UserRole,     static_cast<double>(posFrac));
        barItem->setData(Qt::UserRole + 1, static_cast<double>(negFrac));
        _effMaxTable->setItem(row, 4, barItem);
    }

    // --- Populate _ratioTable (always follows `order`, no separators) ---
    float absMax = 1.f;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            const float r = _table.ratios[order[i]][order[j]];
            if (!std::isnan(r) && i != j)
                absMax = std::max(absMax, std::abs(r));
        }

    _ratioTable->setRowCount(n);
    _ratioTable->setColumnCount(n);

    QStringList colHeaders, rowHeaders;
    for (int idx : order) {
        colHeaders << QString::fromStdString(TruncStr(_table.rewards[idx].name, 12));
        rowHeaders << QString::fromStdString(TruncStr(_table.rewards[idx].name, 22));
    }
    _ratioTable->setHorizontalHeaderLabels(colHeaders);
    _ratioTable->setVerticalHeaderLabels(rowHeaders);

    for (int i = 0; i < n; i++) {
        _ratioTable->verticalHeaderItem(i)->setToolTip(
            QString::fromStdString(_table.rewards[order[i]].name));
        _ratioTable->horizontalHeaderItem(i)->setToolTip(
            QString::fromStdString(_table.rewards[order[i]].name));
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            const float r = _table.ratios[order[i]][order[j]];
            auto* item = MakeItem(FormatRatio(r));
            item->setBackground(RatioColor(r, absMax));

            if (std::isnan(r)) {
                item->setToolTip("Comparison undefined (zero effective max in column)");
            } else if (i == j) {
                item->setToolTip(QString::fromStdString(_table.rewards[order[i]].name));
            } else {
                const auto& rowName = _table.rewards[order[i]].name;
                const auto& colName = _table.rewards[order[j]].name;
                // Tooltip shows the inverse: "1 [col] = (EffMax(col)/EffMax(row)) [row]"
                const float absR   = std::abs(r);
                const QString invStr = (absR > 0.f) ? FormatRatio(1.f / absR) : QString("\u221e");

                if (r < 0.f) {
                    item->setToolTip(
                        QString("Opposite-sign rewards \u2014 1 %1 = %2 %3")
                            .arg(QString::fromStdString(colName))
                            .arg(invStr)
                            .arg(QString::fromStdString(rowName)));
                } else {
                    item->setToolTip(
                        QString("1 %1 = %2 %3")
                            .arg(QString::fromStdString(colName))
                            .arg(invStr)
                            .arg(QString::fromStdString(rowName)));
                }
            }

            _ratioTable->setItem(i, j, item);
        }
    }
}

void RewardComparisonWidget::OnSortChanged(int index)
{
    const int n = static_cast<int>(_table.rewards.size());
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);

    if (index == 1) {
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return std::abs(_table.rewards[a].effectiveMax)
                 > std::abs(_table.rewards[b].effectiveMax);
        });
    } else if (index == 2) {
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return IsEvent(_table.rewards[a]) > IsEvent(_table.rewards[b]);
        });
    }

    PopulateTables(order);
}

void RewardComparisonWidget::OnSplitChanged()
{
    OnSortChanged(_sortCombo ? _sortCombo->currentIndex() : 0);
}

QColor RewardComparisonWidget::RatioColor(float ratio, float absMax) const
{
    if (std::isnan(ratio))
        return Theme::SURFACE_SUNKEN;

    if (ratio == 1.0f)
        return Theme::SURFACE_RAISED;

    // Capped below full strength so the cell text stays readable on the strongest tint.
    const float minTint = 0.2f;
    const float tintRange = 0.55f;
    const float absR = std::abs(ratio);

    if (absR > 1.f) {
        // Red: row is worth more than column (by absolute magnitude)
        const float t = std::clamp(absR / absMax, 0.f, 1.f);
        return BlendColor(Theme::SURFACE_RAISED, Theme::NEGATIVE, minTint + t * tintRange);
    } else {
        // Blue: row is worth less than column (by absolute magnitude)
        const float inv = (absR > 0.f) ? (1.f / absR) : absMax;
        const float t   = std::clamp(inv / absMax, 0.f, 1.f);
        return BlendColor(Theme::SURFACE_RAISED, Theme::COOL, minTint + t * tintRange);
    }
}
