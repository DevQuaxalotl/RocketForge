#pragma once

#include <QString>

#include <cstdint>
#include <filesystem>
#include <optional>

/** Text formatting and parsing for the Training tab's dashboard cards and training log. */
namespace ForgeGUI
{
    /** Parses a report value such as "12,345.6", rejecting non-finite numbers. */
    std::optional<double> ParseMetricNumber(QString text);

    /** Parses a report timestep count, accepting both integer and floating-point text. */
    bool ParseMetricTimesteps(QString text, quint64& timesteps);

    /** Parses a log timestep count with an optional K/M/B suffix, such as "1.5M". */
    bool ParseCompactTimesteps(QString text, quint64& timesteps);

    /** Strips the leading dashes the report uses to indent nested keys. */
    QString NormalizeReportKey(QString key);

    /** Formats one report value for its dashboard label, or "N/A" when it isn't a finite number. */
    QString FormatTrainingMetric(const QString& key, const QString& raw);

    QString FormatSkillNumber(double value);
    QString FormatSkillDelta(double delta);
    QString FormatSkillTestTimesteps(quint64 timesteps);

    /** Formats a remaining duration as "ETA: ~1h 5m". */
    QString FormatEta(double seconds);

    /** Strips the "ETA:" prefix for the Progress card, which already labels the row. */
    QString EtaValueText(QString etaText);

    QString FormatElapsedTrainingDuration(double seconds);

    /** Returns the timesteps a numbered checkpoint folder is named after, or nullopt for any other folder name. */
    std::optional<int64_t> CheckpointFolderTimesteps(const std::filesystem::path& folder);
}
