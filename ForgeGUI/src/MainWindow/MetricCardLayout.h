#pragma once

class QGridLayout;
class QGroupBox;
class QLabel;
class QString;
class QWidget;

/** Layout helpers shared by the Training tab's fixed-height metric cards. */
namespace ForgeGUI
{
    /** Creates a value label that starts as "N/A". */
    QLabel* MakeMetricValueLabel(const char* objectName, QWidget* parent);

    /** Fixes a card's height so the three card rows stay aligned. */
    void ConfigureMetricGroup(QGroupBox* group);

    /** Keeps a card's rows at the top when the card is taller than its content. */
    void PinGridRowsToTop(QGridLayout* layout);

    /** Adds a "name: value" row to a card's grid. */
    void AddMetricRow(QGridLayout* layout, int row, const QString& name, QLabel* value);
}
