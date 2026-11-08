#pragma once

#include <QObject>
#include <QString>
#include <QMap>

/**
 * TrainingLogEvent
 *
 * Represents a single parsed line from the learner's stdout output.
 * ProcessLine() classifies each line into one of the Type values and
 * populates the `fields` map with extracted key-value data. The original
 * raw text is always preserved in `rawLine` for fallback display.
 */
struct TrainingLogEvent
{
    enum class Type {
        ReportStarted,  ///< Beginning of a complete iteration report
        ReportFinished, ///< End of an iteration report; every stat in it has been delivered
        Raw,         ///< Unrecognized line — display as-is in the console log
        TrainingStat, ///< Per-iteration numeric stat (e.g. "Total Timesteps: 1,000") → Training Stats panel
        RunInfo,      ///< One-time startup field (e.g. checkpoint path, device) → Run Info panel
        SkillRating,  ///< Skill match result line (e.g. " >  > 1v1 = 50.3") → Skill Ratings panel
        Error         ///< Line containing "error", "exception", etc. → displayed in red
    };

    Type    type = Type::Raw;
    QString rawLine;                   ///< The unmodified original line
    QMap<QString, QString> fields;     ///< Structured fields extracted by the parser
};

Q_DECLARE_METATYPE(TrainingLogEvent)

/**
 * TrainingLogParser
 *
 * Classifies raw stdout lines emitted by the ForgeTraining learner into typed
 * TrainingLogEvent values. Consumers connect to EventParsed and route each
 * event to the appropriate UI panel (stats labels, run-info labels, log widget).
 *
 * This is the natural extension point for richer parsing: swap the regex
 * patterns for JSON parsing if the learner is updated to emit structured logs.
 */
class TrainingLogParser : public QObject
{
    Q_OBJECT

public:
    explicit TrainingLogParser(QObject* parent = nullptr);

    /**
     * Classifies a single line and emits EventParsed.
     * Blank lines and RG_DIVIDER lines (40 '=' characters) are silently dropped.
     */
    void ProcessLine(const QString& line);

signals:
    /** Emitted once for every line that is not blank or a divider. */
    void EventParsed(TrainingLogEvent event);
};
