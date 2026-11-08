#pragma once

#include <QColor>
#include <QString>
#include <QWidget>

class QFontMetrics;
class QPainter;
class QPalette;
class QRect;

/** Paints a bar from the rect's centre: rightwards and green for positive values, leftwards and red for negative ones. */
void PaintBipolarBar(QPainter* painter, const QRect& rect, const QPalette& palette, float value, float alpha);

/** A value the way every reward debugger column shows it, to 0.001. */
QString FormatValue(float value);

/** "▲ +0.123", "▼ -0.123", or "–" when flat. */
QString FormatTrendText(int direction, float delta);

/**
 * Fixed width for a column or label whose widest expected text is widestText,
 * so it doesn't resize as values change length; longer text is elided.
 */
int FixedTextWidth(const QFontMetrics& metrics, const QString& widestText);

/** Grey when flat; otherwise green (rising) or red (falling), stronger as intensity goes 0..1. */
QColor TrendColor(int direction, float intensity);

/** One bipolar bar for a value in [-1, 1]. */
class BipolarBarWidget : public QWidget {
    Q_OBJECT
public:
    explicit BipolarBarWidget(QWidget* parent = nullptr);

    void SetValue(float value);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    float _value = 0.f;
};
