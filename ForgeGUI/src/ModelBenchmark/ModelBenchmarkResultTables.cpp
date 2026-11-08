#include "ModelBenchmarkResultTables.h"

#include "ThemeUtils.h"

#include <QAbstractItemModel>
#include <QAbstractTextDocumentLayout>
#include <QApplication>
#include <QEvent>
#include <QHeaderView>
#include <QIcon>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QStyle>
#include <QStyledItemDelegate>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QTextDocument>
#include <QTextOption>

#include <algorithm>
#include <array>
#include <limits>
#include <map>

namespace ForgeGUI::ModelBenchmarkTables
{
    namespace
    {
        constexpr std::array<const char*, ModelColorCount> ModelColorHex = {
            "#b78bd9",
            "#d98bbd",
            "#e47777",
            "#e2a84b",
            "#b9cf45",
            "#73c88f",
            "#62c6cc",
            "#5aa9c4"
        };

        bool PrepareResultTable(QTableWidget* table, const QStringList& headers, int rows)
        {
            bool schemaChanged = table->columnCount() != headers.size();
            if (!schemaChanged) {
                for (int column = 0; column < headers.size(); ++column) {
                    const QTableWidgetItem* header = table->horizontalHeaderItem(column);
                    if (!header || header->text() != headers[column]) {
                        schemaChanged = true;
                        break;
                    }
                }
            }

            if (schemaChanged) {
                table->clear();
                table->setColumnCount(headers.size());
                table->setHorizontalHeaderLabels(headers);
            }

            table->setRowCount(rows);

            return schemaChanged;
        }

        void SetOrReuseItem(QTableWidget* table, int row, int column, QTableWidgetItem* replacement)
        {
            QTableWidgetItem* existing = table->item(row, column);
            if (!existing) {
                table->setItem(row, column, replacement);
                return;
            }

            *existing = *replacement;
            delete replacement;

            // Item assignment copies values without notifying the table model,
            // so reused cells would stay visibly stale until an unrelated repaint.
            const QModelIndex index = table->model()->index(row, column);
            emit table->model()->dataChanged(index, index);
        }

        QColor ModelColor(int colorIndex)
        {
            const int normalizedIndex = ((colorIndex % ModelColorCount) + ModelColorCount) % ModelColorCount;
            return ApplyColorPreference(
                QColor(ModelColorHex[normalizedIndex]),
                ApplicationColorsEnabled());
        }

        QColor FadeToward(const QColor& color, const QColor& background)
        {
            constexpr double ColorAmount = 0.42;
            constexpr double BackgroundAmount = 1.0 - ColorAmount;
            return QColor(
                (int)(color.red() * ColorAmount + background.red() * BackgroundAmount),
                (int)(color.green() * ColorAmount + background.green() * BackgroundAmount),
                (int)(color.blue() * ColorAmount + background.blue() * BackgroundAmount));
        }

        class ModelColorTextDelegate : public QStyledItemDelegate
        {
        public:
            explicit ModelColorTextDelegate(QObject* parent = nullptr)
                : QStyledItemDelegate(parent)
            {
            }

            void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override
            {
                QStyleOptionViewItem coloredOption(option);
                initStyleOption(&coloredOption, index);

                const QVariant mentions = index.data(ModelMentionsRole);
                if (mentions.isValid()) {
                    PaintColoredMentions(painter, coloredOption, mentions.toList());
                    return;
                }

                const QVariant colorIndex = index.data(ModelColorRole);
                if (colorIndex.isValid()) {
                    QColor color = ModelColor(colorIndex.toInt());
                    if (index.data(ModelFadedRole).toBool())
                        color = FadeToward(color, coloredOption.palette.color(QPalette::Base));
                    coloredOption.palette.setColor(QPalette::Text, color);
                    coloredOption.palette.setColor(QPalette::HighlightedText, color);
                }

                const QWidget* widget = option.widget;
                QStyle* style = widget ? widget->style() : QApplication::style();
                style->drawControl(QStyle::CE_ItemViewItem, &coloredOption, painter, widget);
            }

        private:
            static QString BuildColoredMentionsHtml(
                const QString& text,
                const QVariantList& mentions,
                const QColor& defaultColor)
            {
                QString content;
                int offset = 0;
                while (offset < text.size()) {
                    int nextPosition = -1;
                    QString nextName;
                    int nextColorIndex = -1;

                    for (const QVariant& mentionValue : mentions) {
                        const QVariantMap mention = mentionValue.toMap();
                        const QString name = mention.value("name").toString();
                        if (name.isEmpty())
                            continue;

                        const int position = text.indexOf(name, offset, Qt::CaseSensitive);
                        if (position < 0)
                            continue;
                        if (nextPosition < 0
                            || position < nextPosition
                            || (position == nextPosition && name.size() > nextName.size())) {
                            nextPosition = position;
                            nextName = name;
                            nextColorIndex = mention.value("colorIndex").toInt();
                        }
                    }

                    if (nextPosition < 0) {
                        content += text.mid(offset).toHtmlEscaped();
                        break;
                    }

                    content += text.mid(offset, nextPosition - offset).toHtmlEscaped();
                    content += QString("<span style=\"color:%1\">%2</span>")
                        .arg(ModelColor(nextColorIndex).name(QColor::HexRgb), nextName.toHtmlEscaped());
                    offset = nextPosition + nextName.size();
                }

                return QString("<span style=\"color:%1\">%2</span>")
                    .arg(defaultColor.name(QColor::HexRgb), content);
            }

            static void PaintColoredMentions(
                QPainter* painter,
                const QStyleOptionViewItem& option,
                const QVariantList& mentions)
            {
                const QWidget* widget = option.widget;
                QStyle* style = widget ? widget->style() : QApplication::style();

                QStyleOptionViewItem backgroundOption(option);
                const QString text = backgroundOption.text;
                backgroundOption.text.clear();
                style->drawControl(QStyle::CE_ItemViewItem, &backgroundOption, painter, widget);

                const QPalette::ColorRole textRole = option.state & QStyle::State_Selected
                    ? QPalette::HighlightedText
                    : QPalette::Text;

                QTextDocument document;
                document.setDocumentMargin(0.0);
                document.setDefaultFont(option.font);
                QTextOption textOption = document.defaultTextOption();
                textOption.setAlignment(option.displayAlignment);
                textOption.setWrapMode(QTextOption::NoWrap);
                document.setDefaultTextOption(textOption);
                document.setHtml(BuildColoredMentionsHtml(
                    text,
                    mentions,
                    option.palette.color(textRole)));
                document.setTextWidth(option.rect.width());

                painter->save();
                painter->setClipRect(option.rect);
                painter->translate(
                    option.rect.left(),
                    option.rect.top() + (option.rect.height() - document.size().height()) / 2.0);

                QAbstractTextDocumentLayout::PaintContext context;
                context.palette = option.palette;
                context.clip = QRectF(0.0, 0.0, option.rect.width(), option.rect.height());
                document.documentLayout()->draw(painter, context);
                painter->restore();
            }
        };

        class CenteredIconDelegate : public QStyledItemDelegate
        {
        public:
            using QStyledItemDelegate::QStyledItemDelegate;

            void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override
            {
                QStyleOptionViewItem cell(option);
                initStyleOption(&cell, index);
                const QIcon icon = cell.icon;
                const QIcon::Mode mode = !(cell.state & QStyle::State_Enabled) ? QIcon::Disabled
                    : (cell.state & QStyle::State_Selected) ? QIcon::Selected : QIcon::Normal;
                const QSize iconSize = icon.actualSize(cell.decorationSize, mode);

                // Let the style paint selection, focus and alternating rows without a left-side icon.
                cell.icon = QIcon();
                cell.features &= ~QStyleOptionViewItem::HasDecoration;
                const QStyle* style = cell.widget ? cell.widget->style() : QApplication::style();
                style->drawControl(QStyle::CE_ItemViewItem, &cell, painter, cell.widget);
                icon.paint(painter, QStyle::alignedRect(cell.direction, Qt::AlignCenter, iconSize, cell.rect), Qt::AlignCenter, mode);
            }
        };

        class CenteredCheckBoxDelegate : public QStyledItemDelegate
        {
        public:
            explicit CenteredCheckBoxDelegate(QObject* parent = nullptr)
                : QStyledItemDelegate(parent)
            {
            }

            void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override
            {
                const QVariant checkState = index.data(Qt::CheckStateRole);
                if (!checkState.isValid()) {
                    QStyledItemDelegate::paint(painter, option, index);
                    return;
                }

                QStyleOptionViewItem backgroundOption(option);
                initStyleOption(&backgroundOption, index);
                backgroundOption.features &= ~QStyleOptionViewItem::HasCheckIndicator;
                backgroundOption.text.clear();
                backgroundOption.icon = QIcon();

                const QWidget* widget = option.widget;
                QStyle* style = widget ? widget->style() : QApplication::style();
                style->drawPrimitive(QStyle::PE_PanelItemViewItem, &backgroundOption, painter, widget);

                const bool checked = checkState.toInt() == Qt::Checked;
                const bool enabled = (option.state & QStyle::State_Enabled)
                    && (index.flags() & Qt::ItemIsEnabled);
                const bool visuallyEnabled = enabled && !index.data(ModelFadedRole).toBool();
                DrawCheckBox(painter, CheckBoxRect(option), checked, visuallyEnabled);
            }

            bool editorEvent(
                QEvent* event,
                QAbstractItemModel* model,
                const QStyleOptionViewItem& option,
                const QModelIndex& index) override
            {
                if (!model || !(index.flags() & Qt::ItemIsUserCheckable) || !(index.flags() & Qt::ItemIsEnabled))
                    return false;
                if (!index.data(Qt::CheckStateRole).isValid())
                    return false;

                if (event->type() == QEvent::MouseButtonRelease) {
                    auto* mouseEvent = static_cast<QMouseEvent*>(event);
                    if (mouseEvent->button() != Qt::LeftButton || !CheckBoxRect(option).contains(mouseEvent->pos()))
                        return false;
                } else if (event->type() == QEvent::KeyPress) {
                    auto* keyEvent = static_cast<QKeyEvent*>(event);
                    if (keyEvent->key() != Qt::Key_Space && keyEvent->key() != Qt::Key_Select)
                        return false;
                } else {
                    return false;
                }

                const Qt::CheckState nextState = index.data(Qt::CheckStateRole).toInt() == Qt::Checked
                    ? Qt::Unchecked
                    : Qt::Checked;
                return model->setData(index, nextState, Qt::CheckStateRole);
            }

        private:
            static QRect CheckBoxRect(const QStyleOptionViewItem& option)
            {
                constexpr int boxSize = 16;
                return QStyle::alignedRect(
                    option.direction,
                    Qt::AlignCenter,
                    QSize(boxSize, boxSize),
                    option.rect);
            }

            static void DrawCheckBox(QPainter* painter, const QRect& rect, bool checked, bool enabled)
            {
                const QColor baseFill = enabled
                    ? (checked ? QColor("#2b747a") : QColor("#202526"))
                    : QColor("#202526");
                const QColor baseBorder = enabled
                    ? (checked ? QColor("#62c6cc") : QColor("#464f50"))
                    : QColor("#333a3b");
                const QColor fill = ApplyColorPreference(baseFill, ApplicationColorsEnabled());
                const QColor border = ApplyColorPreference(baseBorder, ApplicationColorsEnabled());

                painter->save();
                painter->setRenderHint(QPainter::Antialiasing, true);
                painter->setPen(QPen(border, 1));
                painter->setBrush(fill);
                painter->drawRoundedRect(rect.adjusted(0, 0, -1, -1), 4, 4);

                if (checked) {
                    QPen checkPen(enabled ? QColor("#d8d0bf") : QColor("#6f746f"), 2);
                    checkPen.setCapStyle(Qt::RoundCap);
                    checkPen.setJoinStyle(Qt::RoundJoin);
                    painter->setPen(checkPen);
                    QPainterPath checkPath;
                    checkPath.moveTo(rect.left() + 4, rect.center().y());
                    checkPath.lineTo(rect.left() + 7, rect.bottom() - 4);
                    checkPath.lineTo(rect.right() - 4, rect.top() + 4);
                    painter->drawPath(checkPath);
                }

                painter->restore();
            }
        };

        int ModeSortKey(const std::string& modeName)
        {
            if (modeName.empty() || modeName[0] < '0' || modeName[0] > '9')
                return std::numeric_limits<int>::max();
            return modeName[0] - '0';
        }

        void FitRankingTableColumnsImpl(QTableWidget* table, int betterThanColumn)
        {
            if (!table || !table->horizontalHeader())
                return;

            auto* header = table->horizontalHeader();
            table->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
            table->setWordWrap(false);
            header->setStretchLastSection(false);
            header->setSectionResizeMode(QHeaderView::Interactive);
            table->resizeColumnsToContents();

            int fixedWidth = 0;
            for (int column = 0; column < table->columnCount(); column++) {
                if (column != betterThanColumn)
                    fixedWidth += table->columnWidth(column);
            }

            const int availableWidth = std::max(0, table->viewport()->width());
            const int betterWidth = std::max(80, availableWidth - fixedWidth);
            header->resizeSection(betterThanColumn, betterWidth);

            // Follow viewport resizing without refitting every live result update.
            header->setSectionResizeMode(betterThanColumn, QHeaderView::Stretch);
        }
    }

    void InstallModelColorDelegate(QTableWidget* table)
    {
        if (table)
            table->setItemDelegate(new ModelColorTextDelegate(table));
    }

    void InstallCenteredIconDelegate(QTableWidget* table, int column)
    {
        if (table)
            table->setItemDelegateForColumn(column, new CenteredIconDelegate(table));
    }

    void InstallCenteredCheckBoxDelegate(QTableWidget* table, int column)
    {
        if (table)
            table->setItemDelegateForColumn(column, new CenteredCheckBoxDelegate(table));
    }

    int ResultRowColorIndex(int row)
    {
        return std::max(0, row) % ModelColorCount;
    }

    QTableWidgetItem* ReadOnlyItem(const QString& text)
    {
        auto* item = new QTableWidgetItem(text);
        item->setFlags(item->flags() & ~Qt::ItemIsEditable);
        item->setTextAlignment(Qt::AlignCenter);
        item->setToolTip(text);
        return item;
    }

    QTableWidgetItem* ReadOnlyModelItem(const QString& text, int colorIndex)
    {
        QTableWidgetItem* item = ReadOnlyItem(text);
        if (colorIndex >= 0)
            item->setData(ModelColorRole, colorIndex);
        return item;
    }

    QString FormatRecord(const RocketForgeTraining::ModelBenchmarkRecord& record)
    {
        const int scored = record.wins + record.losses;
        const double pct = scored > 0 ? (100.0 * record.wins / scored) : 0.0;
        return QString("%1-%2 (%3%)")
            .arg(record.wins)
            .arg(record.losses)
            .arg(pct, 0, 'f', 1);
    }

    QString FormatRecordWinPct(const RocketForgeTraining::ModelBenchmarkRecord& record)
    {
        const int scored = record.wins + record.losses;
        const double pct = scored > 0 ? (100.0 * record.wins / scored) : 0.0;
        return QString::number(pct, 'f', 1) + "%";
    }

    QString FormatSigned(double value, int precision)
    {
        return QString("%1%2").arg(value >= 0.0 ? "+" : "").arg(value, 0, 'f', precision);
    }

    QString FormatOptionalElo(bool available, double value)
    {
        return available ? FormatSigned(value) : "N/A";
    }

    QString FormatEloEstimate(bool available, double elo, double rangeLow, double rangeHigh)
    {
        if (!available)
            return "N/A";
        return QString("%1 [%2, %3]")
            .arg(FormatSigned(elo), FormatSigned(rangeLow), FormatSigned(rangeHigh));
    }

    void FitTableColumnsToContentsAndFillWidth(QTableWidget* table)
    {
        if (!table || !table->horizontalHeader())
            return;

        auto* header = table->horizontalHeader();
        header->setStretchLastSection(false);
        header->setSectionResizeMode(QHeaderView::Interactive);
        table->resizeColumnsToContents();

        const int columnCount = table->columnCount();
        int contentWidth = 0;
        for (int column = 0; column < columnCount; column++)
            contentWidth += table->columnWidth(column);

        const int availableWidth = std::max(0, table->viewport()->width());
        if (availableWidth <= contentWidth || columnCount <= 0)
            return;

        const int extraWidth = availableWidth - contentWidth;
        const int extraPerColumn = extraWidth / columnCount;
        const int remainder = extraWidth % columnCount;
        for (int column = 0; column < columnCount; column++) {
            const int growth = extraPerColumn + (column < remainder ? 1 : 0);
            header->resizeSection(column, table->columnWidth(column) + growth);
        }
    }

    void FitRankingTableColumns(QTableWidget* table, int betterThanColumn)
    {
        FitRankingTableColumnsImpl(table, betterThanColumn);
    }

    void PopulateRankingTable(QTableWidget* table, const RocketForgeTraining::ModelBenchmarkResult& result, bool refreshLayout)
    {
        const bool firstRows = table->rowCount() == 0 && !result.rankings.empty();
        const bool schemaChanged = PrepareResultTable(table, {
            "Rank", "Model", "Benchmark Elo", "Elo Range", "1v1 Elo", "2v2 Elo", "3v3 Elo", "Record", "Better Than"
        }, (int)result.rankings.size());

        table->horizontalHeaderItem(8)->setToolTip(
            "<qt>Estimated probability this model is stronger than each listed lower-rated model. "
            "Probabilities are shown for all lower-rated models.</qt>");
        table->horizontalHeaderItem(0)->setToolTip("<qt>Position in this comparison, ordered from highest to lowest overall Elo. N/A means the saved games do not connect all models into one comparison.</qt>");
        table->horizontalHeaderItem(1)->setToolTip("The model version represented by this row.");
        table->horizontalHeaderItem(2)->setToolTip(
            "<qt>Estimated relative strength from the included wins and losses. Higher is stronger; the lowest overall model is set to zero. "
            "Each matchup is balanced so frequently tested pairs do not dominate. Shared opponents can connect models that never played directly. "
            "These ratings belong to this comparison, not an external leaderboard.</qt>");
        table->horizontalHeaderItem(3)->setToolTip("<qt>Uncertainty interval for the overall Elo estimate, at 95% confidence. A narrower interval means a more precise estimate; it is not a minimum and maximum observed score.</qt>");
        for (int column = 4; column <= 6; ++column)
            table->horizontalHeaderItem(column)->setToolTip(QString("Relative Elo using only %1v%1 games. N/A means those games do not provide a connected comparison for all models.").arg(column - 3));
        table->horizontalHeaderItem(7)->setToolTip("<qt>Wins and losses, followed by win percentage, across the included games. Draws are omitted here and do not affect Elo or win percentage; see Pairwise Summary for draw counts.</qt>");

        QVariantList modelMentions;
        for (int row = 0; row < (int)result.rankings.size(); row++) {
            QVariantMap mention;
            mention.insert("name", QString::fromStdString(result.rankings[row].modelName));
            mention.insert("colorIndex", ResultRowColorIndex(row));
            modelMentions.push_back(mention);
        }

        for (int row = 0; row < (int)result.rankings.size(); row++) {
            const auto& value = result.rankings[row];
            const int colorIndex = ResultRowColorIndex(row);
            SetOrReuseItem(table, row, 0, ReadOnlyItem(result.overallEloAvailable ? QString::number(value.rank) : "N/A"));
            SetOrReuseItem(table, row, 1, ReadOnlyModelItem(QString::fromStdString(value.modelName), colorIndex));
            SetOrReuseItem(table, row, 2, ReadOnlyItem(FormatOptionalElo(result.overallEloAvailable, value.relativeElo)));
            SetOrReuseItem(table, row, 3, ReadOnlyItem(result.overallEloAvailable
                ? QString("[%1, %2]").arg(FormatSigned(value.rangeLow), FormatSigned(value.rangeHigh))
                : "N/A"));
            SetOrReuseItem(table, row, 4, ReadOnlyItem(FormatOptionalElo(value.elo1v1Available, value.elo1v1)));
            SetOrReuseItem(table, row, 5, ReadOnlyItem(FormatOptionalElo(value.elo2v2Available, value.elo2v2)));
            SetOrReuseItem(table, row, 6, ReadOnlyItem(FormatOptionalElo(value.elo3v3Available, value.elo3v3)));
            SetOrReuseItem(table, row, 7, ReadOnlyItem(FormatRecord(value.totalRecord)));
            QTableWidgetItem* betterThanItem = ReadOnlyItem(QString::fromStdString(value.betterThanSummary));
            betterThanItem->setData(ModelMentionsRole, modelMentions);
            SetOrReuseItem(table, row, 8, betterThanItem);

            for (int column = 0; column < table->columnCount(); column++) {
                if (column != 8)
                    table->item(row, column)->setData(ModelColorRole, colorIndex);
            }
        }

        if (refreshLayout || schemaChanged || firstRows)
            FitRankingTableColumns(table, 8);
    }

    void PopulateBreakdownTable(QTableWidget* table, const RocketForgeTraining::ModelBenchmarkResult& result, bool refreshLayout)
    {
        std::map<int, int> rankByModel;
        for (const auto& ranking : result.rankings)
            rankByModel[ranking.modelIndex] = ranking.rank;

        std::vector<RocketForgeTraining::ModelBenchmarkBreakdownRow> rows = result.breakdown;
        std::sort(rows.begin(), rows.end(), [&](const auto& a, const auto& b) {
            const int rankA = rankByModel.count(a.modelIndex) ? rankByModel[a.modelIndex] : std::numeric_limits<int>::max();
            const int rankB = rankByModel.count(b.modelIndex) ? rankByModel[b.modelIndex] : std::numeric_limits<int>::max();
            return rankA != rankB ? rankA < rankB : a.modelName < b.modelName;
        });

        const QString rangeLabel = QString("%1% Range")
            .arg(result.eloRangeConfidence * 100.0f, 0, 'f', 0);
        const bool schemaChanged = PrepareResultTable(table, {
            "Model",
            "Overall Elo",
            QString("Kickoff Elo (%1)").arg(rangeLabel),
            "Kickoff Games",
            QString("Post-Kickoff Elo (%1)").arg(rangeLabel),
            "Post-Kickoff Games",
            "Kickoff Delta"
        }, (int)rows.size());
        const std::array<QString, 7> breakdownTooltips = {
            "The model version represented by this row.",
            "Relative strength from all included games with a goal, regardless of when the goal occurred.",
            QString("<qt>Elo from games decided within the kickoff goal window (%1 seconds), with its uncertainty range. Uses the overall ranking's zero point, so values can be negative. N/A means these games do not connect all models.</qt>").arg(result.kickoffGoalWindowSeconds),
            "Games with a goal during the kickoff window that contribute to this model's kickoff estimate.",
            QString("<qt>Elo from games decided after the kickoff goal window (%1 seconds), with its uncertainty range. Uses the overall ranking's zero point. N/A means these games do not connect all models.</qt>").arg(result.kickoffGoalWindowSeconds),
            "Games with a goal after the kickoff window that contribute to this model's post-kickoff estimate.",
            "Kickoff Elo minus post-kickoff Elo. A positive value means the model performs relatively better in early-decided games; a negative value means it performs better in later-decided games."
        };
        for (int column = 0; column < (int)breakdownTooltips.size(); ++column)
            table->horizontalHeaderItem(column)->setToolTip(breakdownTooltips[column]);

        for (int row = 0; row < (int)rows.size(); row++) {
            const auto& value = rows[row];
            const int colorIndex = ResultRowColorIndex(row);
            SetOrReuseItem(table, row, 0, ReadOnlyModelItem(QString::fromStdString(value.modelName), colorIndex));
            SetOrReuseItem(table, row, 1, ReadOnlyItem(FormatOptionalElo(result.overallEloAvailable, value.combinedElo)));
            SetOrReuseItem(table, row, 2, ReadOnlyItem(FormatEloEstimate(
                value.kickoffEloAvailable,
                value.kickoffElo,
                value.kickoffRangeLow,
                value.kickoffRangeHigh)));
            SetOrReuseItem(table, row, 3, ReadOnlyItem(QString::number(value.kickoffGames)));
            SetOrReuseItem(table, row, 4, ReadOnlyItem(FormatEloEstimate(
                value.postKickoffEloAvailable,
                value.postKickoffElo,
                value.postKickoffRangeLow,
                value.postKickoffRangeHigh)));
            SetOrReuseItem(table, row, 5, ReadOnlyItem(QString::number(value.postKickoffGames)));
            SetOrReuseItem(table, row, 6, ReadOnlyItem(
                value.kickoffEloAvailable && value.postKickoffEloAvailable
                    ? FormatSigned(value.kickoffDelta)
                    : "N/A"));
            for (int column = 0; column < table->columnCount(); column++)
                table->item(row, column)->setData(ModelColorRole, colorIndex);
        }

        if (refreshLayout || schemaChanged) {
            // Empty-result views stretch their last column; clear that cached width before stretching every column,
            // and scroll per pixel at fit boundaries.
            table->horizontalHeader()->setStretchLastSection(false);
            table->setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);
            table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
        }
    }

    void PopulatePairwiseTable(QTableWidget* table, const RocketForgeTraining::ModelBenchmarkResult& result, bool refreshLayout)
    {
        std::map<int, int> rankByModel;
        std::map<int, int> colorByModel;
        for (int row = 0; row < (int)result.rankings.size(); row++) {
            rankByModel[result.rankings[row].modelIndex] = result.rankings[row].rank;
            colorByModel[result.rankings[row].modelIndex] = ResultRowColorIndex(row);
        }

        std::vector<RocketForgeTraining::ModelBenchmarkPairwiseRow> rows = result.pairwise;
        std::sort(rows.begin(), rows.end(), [&](const auto& a, const auto& b) {
            const int rankA = rankByModel.count(a.modelIndex) ? rankByModel[a.modelIndex] : std::numeric_limits<int>::max();
            const int rankB = rankByModel.count(b.modelIndex) ? rankByModel[b.modelIndex] : std::numeric_limits<int>::max();
            if (rankA != rankB)
                return rankA < rankB;
            const int opponentRankA = rankByModel.count(a.opponentIndex) ? rankByModel[a.opponentIndex] : std::numeric_limits<int>::max();
            const int opponentRankB = rankByModel.count(b.opponentIndex) ? rankByModel[b.opponentIndex] : std::numeric_limits<int>::max();
            if (opponentRankA != opponentRankB)
                return opponentRankA < opponentRankB;
            const int modeA = ModeSortKey(a.modeName);
            const int modeB = ModeSortKey(b.modeName);
            if (modeA != modeB)
                return modeA < modeB;
            if (a.modelName != b.modelName)
                return a.modelName < b.modelName;
            return a.opponentName < b.opponentName;
        });

        const bool schemaChanged = PrepareResultTable(
            table,
            { "Model", "Opponent", "Mode", "Overall", "Kickoff", "Post-Kickoff" },
            (int)rows.size());
        table->horizontalHeaderItem(0)->setToolTip("The model whose wins, losses, and draws are shown on this row.");
        table->horizontalHeaderItem(1)->setToolTip("<qt>The other model in these games. This heading describes a matchup, not the model's Next run role.</qt>");
        table->horizontalHeaderItem(2)->setToolTip("The team size for these games: 1v1, 2v2, or 3v3.");
        for (int column = 3; column <= 5; ++column)
            table->horizontalHeaderItem(column)->setToolTip(
                "<qt>Wins/losses/draws (win percentage). Draws do not affect Elo or win percentage. "
                "Games without a goal are counted as draws in Overall only.</qt>");

        const auto formatRecord = [](const RocketForgeTraining::ModelBenchmarkRecord& record) {
            return QString("%1/%2/%3 (%4)")
                .arg(record.wins)
                .arg(record.losses)
                .arg(record.noResults)
                .arg(FormatRecordWinPct(record));
        };

        for (int row = 0; row < (int)rows.size(); row++) {
            const auto& value = rows[row];
            const int modelColorIndex = colorByModel.count(value.modelIndex)
                ? colorByModel[value.modelIndex]
                : std::max(0, value.modelIndex) % ModelColorCount;
            const int opponentColorIndex = colorByModel.count(value.opponentIndex)
                ? colorByModel[value.opponentIndex]
                : std::max(0, value.opponentIndex) % ModelColorCount;
            SetOrReuseItem(table, row, 0, ReadOnlyModelItem(QString::fromStdString(value.modelName), modelColorIndex));
            SetOrReuseItem(table, row, 1, ReadOnlyModelItem(QString::fromStdString(value.opponentName), opponentColorIndex));
            SetOrReuseItem(table, row, 2, ReadOnlyItem(QString("%1 (%2)")
                .arg(QString::fromStdString(value.modeName), FormatSigned(value.modeElo))));
            SetOrReuseItem(table, row, 3, ReadOnlyItem(formatRecord(value.combined)));
            SetOrReuseItem(table, row, 4, ReadOnlyItem(formatRecord(value.kickoff)));
            SetOrReuseItem(table, row, 5, ReadOnlyItem(formatRecord(value.postKickoff)));
        }

        if (refreshLayout || schemaChanged) {
            // Empty-result views stretch their last column; clear that cached width before stretching every column,
            // and scroll per pixel at fit boundaries.
            table->horizontalHeader()->setStretchLastSection(false);
            table->setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);
            table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
        }
    }

    void PopulateStatsTable(
        QTableWidget* table,
        const std::vector<RocketForgeTraining::ModelBenchmarkRankingRow>& rankings,
        const std::vector<RocketForgeTraining::ModelBenchmarkStatRow>& stats,
        bool refreshLayout)
    {
        const QStringList headers = {
            "Model", "5m\nGames", "Goals", "Shots", "Shot\nConversion", "Assists", "Assisted\nGoals",
            "Saves", "Save\n%", "Passes", "Ball\nTouches", "Touch\nShare", "Bumps", "Demos",
            "Boost\nSteals", "Big\nBoost", "Small\nBoost", "Boost/\nmin", "Boost", "Low Boost\nTime", "Off.\nThird", "Neutral\nThird",
            "Def.\nThird", "Speed", "Low\nAir", "High\nAir", "Supersonic", "Flip\nResets",
            "Touch\nHeight", "Speed\nTo Ball", "Dist.\nTo Ball"
        };
        const std::array<const char*, 31> statTooltips = {
            "Model and benchmark version represented by this row.",
            "Total simulated game time for this model, expressed as five-minute games.",
            "Goals scored by the model's team.",
            "Shots credited to the model's team.",
            "Percentage of the model team's shots that resulted in goals.",
            "Assists credited to the model's team.",
            "Percentage of the model team's goals that were credited with an assist.",
            "Saves credited to the model's team.",
            "Saves as a percentage of saves plus goals conceded.",
            "Completed passes per eligible 2v2/3v3 game: the next uncontested touch is by a different teammate within 2 seconds and at least 500 uu away.",
            "Distinct ball contacts started by players on the model's team.",
            "Percentage of all distinct ball-contact starts made by the model's team.",
            "Opponent cars bumped by the model's team.",
            "Opponent cars demolished by the model's team.",
            "Large boost pads collected by the model's team on the opponent's half of the field.",
            "Large boost pads collected by the model's team.",
            "Small boost pads collected by the model's team.",
            "Boost points depleted per player-minute.",
            "Average player boost across samples, on a 0-100 scale.",
            "Percentage of player time spent at 0-20 boost.",
            "Percentage of sampled game steps with the ball in the model team's offensive third.",
            "Percentage of sampled game steps with the ball in the neutral third.",
            "Percentage of sampled game steps with the ball in the model team's defensive third.",
            "Average player speed across samples, in uu/s.",
            "Percentage of player time airborne below crossbar height.",
            "Percentage of player time airborne at or above crossbar height.",
            "Percentage of player time spent supersonic.",
            "Flip resets acquired from a ball touch at least 200 uu high and within 130 uu of the ball.",
            "Average ball height during the model team's ball-touch steps, in uu.",
            "Average player speed toward the ball across samples, in uu/s.",
            "Average player distance from the ball across samples, in uu."
        };
        std::map<int, int> rankByModel;
        for (const auto& ranking : rankings)
            rankByModel[ranking.modelIndex] = ranking.rank;
        std::vector<RocketForgeTraining::ModelBenchmarkStatRow> rows = stats;
        std::sort(rows.begin(), rows.end(), [&](const auto& a, const auto& b) {
            const int rankA = rankByModel.count(a.modelIndex) ? rankByModel[a.modelIndex] : std::numeric_limits<int>::max();
            const int rankB = rankByModel.count(b.modelIndex) ? rankByModel[b.modelIndex] : std::numeric_limits<int>::max();
            return rankA != rankB ? rankA < rankB : a.modelName < b.modelName;
        });

        const bool schemaChanged = PrepareResultTable(table, headers, (int)rows.size());
        for (int column = 0; column < (int)statTooltips.size(); column++)
            table->horizontalHeaderItem(column)->setToolTip(statTooltips[column]);

        for (int row = 0; row < (int)rows.size(); row++) {
            const auto& value = rows[row];
            const auto& stat = value.stats;
            const int colorRow = rankByModel.count(value.modelIndex)
                ? rankByModel[value.modelIndex] - 1
                : row;
            const int colorIndex = ResultRowColorIndex(colorRow);
            const QString unavailable = QString::fromUtf8("\xE2\x80\x94");
            const auto expanded = [&](double number, int decimals, const QString& suffix = QString()) {
                return stat.expandedStatsAvailable
                    ? QString::number(number, 'f', decimals) + suffix
                    : unavailable;
            };
            const auto efficiency = [&](double number, int decimals, const QString& suffix = QString()) {
                return stat.efficiencyStatsAvailable
                    ? QString::number(number, 'f', decimals) + suffix
                    : unavailable;
            };
            const QStringList values = {
                QString::fromStdString(value.modelName),
                QString::number(stat.equivalentGames, 'f', 2),
                QString::number(stat.goals, 'f', 2),
                QString::number(stat.shots, 'f', 2),
                QString::number(stat.shotConversionPercent, 'f', 1) + "%",
                QString::number(stat.assists, 'f', 2),
                QString::number(stat.assistedGoalPercent, 'f', 1) + "%",
                QString::number(stat.saves, 'f', 2),
                efficiency(stat.savePercent, 1, "%"),
                stat.passesAvailable ? QString::number(stat.passes, 'f', 2) : unavailable,
                QString::number(stat.ballTouches, 'f', 2),
                efficiency(stat.touchSharePercent, 1, "%"),
                expanded(stat.bumps, 2),
                QString::number(stat.demos, 'f', 2),
                expanded(stat.boostSteals, 2),
                QString::number(stat.bigBoostPads, 'f', 2),
                QString::number(stat.smallBoostPads, 'f', 2),
                efficiency(stat.boostConsumptionPerMinute, 1),
                QString::number(stat.averageBoost, 'f', 1),
                QString::number(stat.lowBoostTimePercent, 'f', 1) + "%",
                expanded(stat.offensiveThirdPercent, 1, "%"),
                expanded(stat.neutralThirdPercent, 1, "%"),
                expanded(stat.defensiveThirdPercent, 1, "%"),
                QString::number(stat.averageSpeed, 'f', 0),
                expanded(stat.lowAirTimePercent, 1, "%"),
                expanded(stat.highAirTimePercent, 1, "%"),
                QString::number(stat.averageSupersonicTime, 'f', 1) + "%",
                QString::number(stat.flipResets, 'f', 2),
                QString::number(stat.averageTouchHeight, 'f', 0),
                QString::number(stat.averageSpeedTowardsBall, 'f', 0),
                QString::number(stat.averageDistanceToBall, 'f', 0)
            };
            for (int column = 0; column < values.size(); column++)
                SetOrReuseItem(table, row, column, ReadOnlyModelItem(values[column], colorIndex));
        }

        if (refreshLayout || schemaChanged) {
            QHeaderView* header = table->horizontalHeader();
            header->setDefaultAlignment(Qt::AlignCenter);
            header->setStyleSheet("QHeaderView::section { padding: 4px 3px; }");
            header->setSectionResizeMode(QHeaderView::ResizeToContents);
        }
    }
}
