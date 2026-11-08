#include "MainWindow/MetricCardLayout.h"

#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QSizePolicy>

#include <algorithm>

namespace
{
    constexpr int MetricGroupHeight = 140;
}

namespace ForgeGUI
{
    QLabel* MakeMetricValueLabel(const char* objectName, QWidget* parent)
    {
        auto* label = new QLabel("N/A", parent);
        label->setObjectName(objectName);
        label->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);

        return label;
    }

    void ConfigureMetricGroup(QGroupBox* group)
    {
        if (!group)
            return;

        group->setMinimumHeight(MetricGroupHeight);
        group->setMaximumHeight(MetricGroupHeight);
        group->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    }

    void PinGridRowsToTop(QGridLayout* layout)
    {
        if (!layout)
            return;

        int lastContentRow = -1;

        for (int i = 0; i < layout->count(); ++i) {
            int row = 0;
            int column = 0;
            int rowSpan = 0;
            int columnSpan = 0;
            layout->getItemPosition(i, &row, &column, &rowSpan, &columnSpan);
            lastContentRow = std::max(lastContentRow, row + rowSpan - 1);
        }

        for (int row = 0; row <= lastContentRow; ++row)
            layout->setRowStretch(row, 0);
        layout->setRowStretch(lastContentRow + 1, 1);
        layout->setAlignment(Qt::AlignTop);
    }

    void AddMetricRow(QGridLayout* layout, int row, const QString& name, QLabel* value)
    {
        if (!layout || !value)
            return;

        auto* nameLabel = new QLabel(name, value->parentWidget());
        layout->addWidget(nameLabel, row, 0);
        layout->addWidget(value, row, 1);
    }
}
