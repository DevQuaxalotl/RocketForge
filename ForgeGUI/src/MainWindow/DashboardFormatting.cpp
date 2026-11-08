#include "MainWindow/DashboardFormatting.h"

#include <QLocale>
#include <QRegularExpression>

#include <cmath>
#include <limits>
#include <string>

namespace ForgeGUI
{
    std::optional<double> ParseMetricNumber(QString text)
    {
        text = text.trimmed();
        text.remove(',');
        bool ok = false;
        const double value = text.toDouble(&ok);

        if (!ok || !std::isfinite(value))
            return std::nullopt;
        return value;
    }

    bool ParseMetricTimesteps(QString text, quint64& timesteps)
    {
        text = text.trimmed();
        text.remove(',');
        bool ok = false;
        timesteps = text.toULongLong(&ok);
        if (ok) return true;

        const std::optional<double> value = ParseMetricNumber(text);
        if (!value || *value < 0.0 || *value >= std::ldexp(1.0, 64)) return false;

        timesteps = static_cast<quint64>(std::round(*value));
        return true;
    }

    bool ParseCompactTimesteps(QString text, quint64& timesteps)
    {
        text = text.trimmed();
        text.remove(',');

        static const QRegularExpression rxCompact(
            R"(^([0-9]+(?:\.[0-9]+)?)\s*([KMB]?)$)",
            QRegularExpression::CaseInsensitiveOption);

        const QRegularExpressionMatch match = rxCompact.match(text);

        if (!match.hasMatch())
            return false;

        bool ok = false;
        const double value = match.captured(1).toDouble(&ok);

        if (!ok || value < 0.0)
            return false;

        double multiplier = 1.0;
        const QString unit = match.captured(2).toUpper();
        if (unit == "K")
            multiplier = 1'000.0;
        else if (unit == "M")
            multiplier = 1'000'000.0;
        else if (unit == "B")
            multiplier = 1'000'000'000.0;

        const double scaled = value * multiplier;

        if (scaled > (double)std::numeric_limits<quint64>::max())
            return false;

        timesteps = (quint64)std::llround(scaled);
        return true;
    }

    QString NormalizeReportKey(QString key)
    {
        key = key.trimmed();

        while (key.startsWith('-')) {
            key.remove(0, 1);
            key = key.trimmed();
        }

        return key;
    }

    QString FormatTrainingMetric(const QString& key, const QString& raw)
    {
        const std::optional<double> value = ParseMetricNumber(raw);
        if (!value) return "N/A";

        const QLocale locale(QLocale::English);

        if (key == "Total Timesteps" || key == "Total Iterations" || key == "Collected Timesteps") {
            quint64 count = 0;
            return ParseMetricTimesteps(raw, count) ? locale.toString(count) : QString("N/A");
        }

        if (*value != 0.0 && (std::abs(*value) < 0.0001 || std::abs(*value) >= 1e12))
            return QString::number(*value, 'e', 2);

        const bool throughput = key.startsWith("Steps Per Second/") || key == "Overall Steps/Second";
        QString text = locale.toString(*value, 'f', throughput ? 1 : 4);

        if (text.contains('.')) {
            while (text.endsWith('0')) text.chop(1);
            if (text.endsWith('.')) text.chop(1);
        }

        return text == "-0" ? QString("0") : text;
    }

    QString FormatSkillNumber(double value)
    {
        const double rounded = std::round(value);

        if (std::abs(value - rounded) < 0.05)
            return QString::number(static_cast<qint64>(rounded));
        return QString::number(value, 'f', 1);
    }

    QString FormatSkillDelta(double delta)
    {
        if (std::abs(delta) < 0.05)
            delta = 0.0;

        const QString sign = delta >= 0.0 ? QString("+") : QString();
        return sign + FormatSkillNumber(delta);
    }

    QString FormatSkillTestTimesteps(quint64 timesteps)
    {
        if (timesteps >= 1'000'000'000ull)
            return QString("%1B").arg((double)timesteps / 1'000'000'000.0, 0, 'f', 3);
        if (timesteps >= 1'000'000ull)
            return QString("%1M").arg((double)timesteps / 1'000'000.0, 0, 'f', 1);
        if (timesteps >= 1'000ull)
            return QString("%1K").arg((double)timesteps / 1'000.0, 0, 'f', 1);
        return QString::number(static_cast<qulonglong>(timesteps));
    }

    QString FormatEta(double seconds)
    {
        if (seconds <= 0.0)
            return QString("ETA: done");
        if (seconds < 60.0)
            return QString("ETA: ~%1s").arg((int)seconds);
        if (seconds < 3600.0)
            return QString("ETA: ~%1m %2s").arg((int)(seconds / 60)).arg((int)seconds % 60);
        int hours   = (int)(seconds / 3600);
        int minutes = ((int)seconds % 3600) / 60;

        return QString("ETA: ~%1h %2m").arg(hours).arg(minutes);
    }

    QString EtaValueText(QString etaText)
    {
        etaText = etaText.trimmed();
        if (etaText.startsWith("ETA:", Qt::CaseInsensitive))
            etaText = etaText.mid(4).trimmed();
        return etaText.isEmpty() ? QString("N/A") : etaText;
    }

    QString FormatElapsedTrainingDuration(double seconds)
    {
        if (seconds <= 0.0)
            return QString("0s");

        const int totalSeconds = (int)std::llround(seconds);
        const int hours = totalSeconds / 3600;
        const int minutes = (totalSeconds % 3600) / 60;
        const int secs = totalSeconds % 60;

        if (hours > 0)
            return QString("%1h %2m").arg(hours).arg(minutes);
        if (minutes > 0)
            return QString("%1m %2s").arg(minutes).arg(secs);
        return QString("%1s").arg(secs);
    }

    std::optional<int64_t> CheckpointFolderTimesteps(const std::filesystem::path& folder)
    {
        const std::string name = folder.filename().string();

        if (name.empty())
            return std::nullopt;

        try {
            size_t parsedChars = 0;
            const int64_t value = std::stoll(name, &parsedChars);

            if (parsedChars == name.size())
                return value;
        } catch (...) {
        }

        return std::nullopt;
    }
}
