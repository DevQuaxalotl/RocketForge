#include "CriticTotalStrip.h"
#include "ValueDisplay.h"

#include <QHBoxLayout>
#include <QLabel>

CriticTotalStrip::CriticTotalStrip(QWidget* parent)
    : QGroupBox("Critic", parent)
{
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(6, 2, 6, 4);
    layout->setSpacing(8);

    auto* nameLabel = new QLabel("V(s):", this);
    nameLabel->setToolTip("Discounted future reward the critic expects from this state under the current policy");

    _valueLabel = new QLabel(this);
    _valueLabel->setFixedWidth(FixedTextWidth(_valueLabel->fontMetrics(), FormatValue(-888.888f)));
    _valueLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);

    _trendLabel = new QLabel(this);
    const int rising = 1;
    _trendLabel->setFixedWidth(FixedTextWidth(_trendLabel->fontMetrics(), FormatTrendText(rising, 888.888f)));
    _trendLabel->setToolTip("Change over the last 0.5 s of game time");

    _bar = new BipolarBarWidget(this);
    _bar->setToolTip("Scaled by the largest |V(s)| seen this session");

    layout->addWidget(nameLabel);
    layout->addWidget(_valueLabel);
    layout->addWidget(_trendLabel);
    layout->addWidget(_bar, /*stretch=*/1);
}

void CriticTotalStrip::SetTotal(float value, float norm, const ForgeGUI::TrendReading& trend) {
    const float fullIntensity = 1.f;

    _valueLabel->setText(FormatValue(value));
    _trendLabel->setText(FormatTrendText(trend.direction, trend.delta));
    _trendLabel->setStyleSheet(QString("color: %1;").arg(TrendColor(trend.direction, fullIntensity).name()));
    _bar->SetValue(norm);
}
