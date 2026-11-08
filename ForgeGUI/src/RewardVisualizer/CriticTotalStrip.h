#pragma once

#include <QGroupBox>

#include "ValueTrendTracker.h"

class QLabel;
class BipolarBarWidget;

/** The "Critic" box: the critic's total predicted value V(s), its trend, and a bar. */
class CriticTotalStrip : public QGroupBox {
    Q_OBJECT
public:
    explicit CriticTotalStrip(QWidget* parent = nullptr);

    /** norm is the bar extent in [-1, 1]. */
    void SetTotal(float value, float norm, const ForgeGUI::TrendReading& trend);

private:
    QLabel*           _valueLabel = nullptr;
    QLabel*           _trendLabel = nullptr;
    BipolarBarWidget* _bar        = nullptr;
};
