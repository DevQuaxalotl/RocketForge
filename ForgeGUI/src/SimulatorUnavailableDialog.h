#pragma once

#include <QString>

class QWidget;

/**
 * Returns the first double-quoted command in an engine's unavailable reason, without its quotes.
 * Returns an empty string when the reason quotes none.
 */
inline QString QuotedCommandFromReason(const QString& reason)
{
    const qsizetype open = reason.indexOf('"');
    if (open < 0)
        return {};

    const qsizetype close = reason.indexOf('"', open + 1);
    if (close < 0)
        return {};

    return reason.mid(open + 1, close - open - 1);
}

/** Shows the "Simulator not available" warning, with a "Copy command" button when the reason quotes a command. */
void ShowSimulatorUnavailable(QWidget* parent, const QString& reason);
