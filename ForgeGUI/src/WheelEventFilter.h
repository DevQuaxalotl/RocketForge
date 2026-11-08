#pragma once

#include <QEvent>
#include <QObject>
#include <QWidget>

/**
 * WheelEventFilter
 *
 * Installs as an event filter on QSpinBox / QDoubleSpinBox / QComboBox widgets
 * that live inside a scroll area.  Without this filter the widgets consume every
 * wheel event that passes over them — even when the user is just trying to scroll
 * the containing list — silently changing values.
 *
 * Behaviour:
 *   - Widget does NOT have keyboard focus  → wheel event is ignored (swallowed),
 *     so the scroll area scrolls normally.
 *   - Widget HAS keyboard focus (user clicked into it) → wheel passes through,
 *     allowing normal spin/combo wheel behaviour.
 *
 * Usage:
 *   auto* filter = new WheelEventFilter(parent);
 *   spinBox->installEventFilter(filter);
 *   spinBox->setFocusPolicy(Qt::StrongFocus);   // must click to gain focus
 */
class WheelEventFilter : public QObject
{
    Q_OBJECT
public:
    explicit WheelEventFilter(QObject* parent = nullptr) : QObject(parent) {}

protected:
    bool eventFilter(QObject* watched, QEvent* event) override
    {
        if (event->type() == QEvent::Wheel) {
            auto* widget = qobject_cast<QWidget*>(watched);
            if (widget && !widget->hasFocus()) {
                // Swallow the wheel event — the scroll area will handle it instead.
                event->ignore();
                return true;
            }
        }
        return QObject::eventFilter(watched, event);
    }
};
