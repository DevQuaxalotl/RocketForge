#include "ValueDisplay.h"
#include "ThemeUtils.h"

#include <QFontMetrics>
#include <QPainter>

#include <algorithm>
#include <cmath>

using namespace ForgeGUI;

void PaintBipolarBar(QPainter* painter, const QRect& rect, const QPalette& palette, float value, float alpha) {
    value = std::clamp(value, -1.f, 1.f);
    alpha = std::clamp(alpha, 0.f, 1.f);

    painter->fillRect(rect, palette.base());

    const int cx     = rect.left() + rect.width() / 2;
    const int top    = rect.top() + 2;
    const int height = rect.height() - 4;

    if (height > 0 && alpha > 0.01f) {
        QColor barColor = value >= 0.f ? Theme::POSITIVE : Theme::NEGATIVE;
        barColor.setAlphaF(static_cast<qreal>(alpha));
        const int barWidth = static_cast<int>(std::abs(value) * rect.width() / 2);
        if (barWidth > 0) {
            if (value >= 0.f)
                painter->fillRect(QRect(cx, top, barWidth, height), barColor);
            else
                painter->fillRect(QRect(cx - barWidth, top, barWidth, height), barColor);
        }
    }

    painter->setPen(Theme::AXIS);
    painter->drawLine(cx, rect.top(), cx, rect.bottom());
}

QString FormatValue(float value) {
    return QString::number((double)value, 'f', 3);
}

QString FormatTrendText(int direction, float delta) {
    if (direction == 0)
        return QString(QChar(0x2013));

    const QChar arrow = direction > 0 ? QChar(0x25B2) : QChar(0x25BC);
    const QString sign = delta >= 0.f ? "+" : "";
    return QString("%1 %2%3").arg(arrow).arg(sign).arg(FormatValue(delta));
}

int FixedTextWidth(const QFontMetrics& metrics, const QString& widestText) {
    const int cellPadding = 12;
    return metrics.horizontalAdvance(widestText) + cellPadding;
}

QColor TrendColor(int direction, float intensity) {
    if (direction == 0)
        return Theme::TEXT_MUTED;

    // Even the weakest visible trend keeps some color so its direction stays readable.
    const float blend = 0.35f + 0.65f * std::clamp(intensity, 0.f, 1.f);
    const QColor target = direction > 0 ? Theme::POSITIVE : Theme::NEGATIVE;

    return BlendColor(Theme::TEXT_MUTED, target, blend);
}

BipolarBarWidget::BipolarBarWidget(QWidget* parent)
    : QWidget(parent)
{
    setMinimumHeight(16);
}

void BipolarBarWidget::SetValue(float value) {
    _value = value;
    update();
}

void BipolarBarWidget::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    const float opaque = 1.f;
    PaintBipolarBar(&painter, rect(), palette(), _value, opaque);
}
