#pragma once

#include <QWidget>

#include <TrainingSetup/RewardComparison.h>

class QTableWidget;
class QSplitter;
class QComboBox;
class QCheckBox;

/**
 * RewardComparisonWidget
 *
 * Standalone floating window showing a static reward comparison table for a
 * training stage. The top panel is a bar chart of each reward's effective max
 * (linear scale, optionally split by category so event and continuous rewards
 * are each scaled to their own maximum). The bottom panel is an N×N ratio
 * matrix where table[row][col] = EffMax(row)/EffMax(col), read as
 * "1 [row] = X [col]".
 *
 * Both panels share the same sort order, selectable via a combo box.
 */
class RewardComparisonWidget : public QWidget
{
    Q_OBJECT

public:
    explicit RewardComparisonWidget(const RocketForge::RewardComparisonTable& table,
                                    QWidget* parent = nullptr);

private slots:
    void OnSortChanged(int index);
    void OnSplitChanged();

private:
    void BuildControls(QWidget* container);
    void BuildEffectiveMaxPanel(QWidget* container);
    void BuildRatioTable();
    void SetBarSectionHeader(int row, const QString& text);
    void PopulateTables(const std::vector<int>& order);

    QColor RatioColor(float ratio, float absMax) const;

    RocketForge::RewardComparisonTable _table;

    QComboBox*    _sortCombo       = nullptr;
    QCheckBox*    _splitCatCheck   = nullptr;
    QTableWidget* _effMaxTable     = nullptr;
    QTableWidget* _ratioTable      = nullptr;
};
