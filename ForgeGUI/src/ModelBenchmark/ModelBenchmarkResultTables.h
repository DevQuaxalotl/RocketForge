#pragma once

#include <ModelBenchmark/ModelBenchmark.h>

#include <QString>
#include <QVariant>

#include <vector>

class QTableWidget;
class QTableWidgetItem;

namespace ForgeGUI::ModelBenchmarkTables
{
    inline constexpr int ModelColorRole = Qt::UserRole + 100;
    inline constexpr int ModelMentionsRole = Qt::UserRole + 101;
    inline constexpr int ModelFadedRole = Qt::UserRole + 102;
    inline constexpr int ModelColorCount = 8;

    /** Installs the delegate that colors model names and model mentions. **/
    void InstallModelColorDelegate(QTableWidget* table);

    /** Centers icon-only cells while preserving row backgrounds and selection. **/
    void InstallCenteredIconDelegate(QTableWidget* table, int column);

    /** Installs the themed centered checkbox delegate on one table column. **/
    void InstallCenteredCheckBoxDelegate(QTableWidget* table, int column);

    /** Returns the stable display color assigned to a ranked row. **/
    int ResultRowColorIndex(int row);

    /** Creates a centered, read-only table item. **/
    QTableWidgetItem* ReadOnlyItem(const QString& text);

    /** Creates a centered, read-only item rendered with a model color. **/
    QTableWidgetItem* ReadOnlyModelItem(const QString& text, int colorIndex);

    /** Formats a scored-game record with its win percentage. **/
    QString FormatRecord(const RocketForgeTraining::ModelBenchmarkRecord& record);
    QString FormatRecordWinPct(const RocketForgeTraining::ModelBenchmarkRecord& record);
    QString FormatSigned(double value, int precision = 0);
    QString FormatOptionalElo(bool available, double value);
    QString FormatEloEstimate(bool available, double elo, double rangeLow, double rangeHigh);

    /** Fits content columns and distributes unused viewport width evenly. **/
    void FitTableColumnsToContentsAndFillWidth(QTableWidget* table);
    void FitRankingTableColumns(QTableWidget* table, int betterThanColumn);

    /** Populates result tables identically in live and saved benchmark views. **/
    void PopulateRankingTable(QTableWidget* table, const RocketForgeTraining::ModelBenchmarkResult& result, bool refreshLayout = true);
    void PopulateBreakdownTable(QTableWidget* table, const RocketForgeTraining::ModelBenchmarkResult& result, bool refreshLayout = true);
    void PopulatePairwiseTable(QTableWidget* table, const RocketForgeTraining::ModelBenchmarkResult& result, bool refreshLayout = true);
    void PopulateStatsTable(
        QTableWidget* table,
        const std::vector<RocketForgeTraining::ModelBenchmarkRankingRow>& rankings,
        const std::vector<RocketForgeTraining::ModelBenchmarkStatRow>& stats,
        bool refreshLayout = true);
}
