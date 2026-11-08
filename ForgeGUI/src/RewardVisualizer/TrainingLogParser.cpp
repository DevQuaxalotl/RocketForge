#include "TrainingLogParser.h"

#include <QRegularExpression>

TrainingLogParser::TrainingLogParser(QObject* parent) : QObject(parent) {}

void TrainingLogParser::ProcessLine(const QString& line)
{
    // Skip divider lines (RG_DIVIDER = 40 '=' chars) and blank lines
    const QString trimmed = line.trimmed();
    if (trimmed.isEmpty())
        return;
    if (trimmed.count('=') == trimmed.length())
        return;

    TrainingLogEvent event;
    event.rawLine = line;

    // -----------------------------------------------------------------------
    // RunInfo — specific startup/config lines that populate the Run Info panel.
    // Checked before TrainingStat to win on overlapping patterns (e.g. "Obs size: N").
    // -----------------------------------------------------------------------

    // "\tCheckpoint Save/Load Dir: path"
    static const QRegularExpression rxCheckpoint(R"(^\tCheckpoint Save/Load Dir:\s*(.+)$)");
    // "\tCheckpoint Resume Dir: path"
    static const QRegularExpression rxCheckpointResume(R"(^\tCheckpoint Resume Dir:\s*(.+)$)");
    // "\tUsing CUDA GPU device..." or "\tUsing CPU device..."
    static const QRegularExpression rxDevice(R"(^\tUsing (CUDA GPU|CPU) device)");
    // "\tObs size: 107"
    static const QRegularExpression rxObsSize(R"(^\tObs size:\s*(\d+)$)");
    // "\tAction amount: 90"
    static const QRegularExpression rxActions(R"(^\tAction amount:\s*(\d+)$)");
    // "\t[Total]: 1,234,567"
    static const QRegularExpression rxTotalParams(R"(^\t\[Total\]:\s*([\d,]+)$)");
    // "\tRun ID: abc123"
    static const QRegularExpression rxRunId(R"(^\tRun ID:\s*(.+)$)");
    // "PPOLearner: ...Set learning rate to [1.5e-04, 1.5e-04]"
    static const QRegularExpression rxLR(R"(Set learning rate to \[([\d.e+\-]+),\s*([\d.e+\-]+)\])");

    // -----------------------------------------------------------------------
    // SkillRating — " >  > 1v1 = 50.3956 (-0.947723)"
    // RG_LOG(" > " << " > " << mode << " = " << rating << optionalDelta)
    // -----------------------------------------------------------------------
    static const QRegularExpression rxSkill(
        R"(^\s*>\s+>\s*(.+?)\s*=\s*([+\-]?[\d,]+(?:\.\d+)?)\s*(?:\(([+\-][\d,]+(?:\.\d+)?)\))?\s*$)"
    );
    // Skill-progress lines stay Raw. They contain tentative mid-test estimates,
    // while final SkillTest/Rating rows are emitted through Report::Display().

    // -----------------------------------------------------------------------
    // TrainingStat — "Label: <number>" (may use commas, sign, exponent). MSVC prints
    // non-finite values as "nan(ind)" / "-nan(ind)" / "inf", which must stay stats:
    // metric labels such as "Value Error" would otherwise match rxError below.
    // -----------------------------------------------------------------------
    static const QRegularExpression rxStat(
        R"(^([^:]+):\s*([-+]?(?:[\d,]+(?:\.\d+)?(?:[eE][+-]?\d+)?|nan(?:\([^)]*\))?|inf(?:inity)?))\s*$)",
        QRegularExpression::CaseInsensitiveOption
    );

    // -----------------------------------------------------------------------
    // Error
    // -----------------------------------------------------------------------
    static const QRegularExpression rxError(
        R"(error|exception|failed|fatal|critical)",
        QRegularExpression::CaseInsensitiveOption
    );

    QRegularExpressionMatch m;

    if (trimmed == "[Training metrics]") {
        event.type = TrainingLogEvent::Type::ReportStarted;
    }
    else if (trimmed == "[Training metrics end]") {
        event.type = TrainingLogEvent::Type::ReportFinished;
    }
    else if ((m = rxSkill.match(line)).hasMatch()) {
        event.type              = TrainingLogEvent::Type::SkillRating;
        event.fields["mode"]    = m.captured(1).trimmed();
        event.fields["rating"]  = m.captured(2).trimmed();
        event.fields["delta"]   = m.captured(3).trimmed(); // empty if no delta
    }
    else if ((m = rxCheckpoint.match(line)).hasMatch()) {
        event.type              = TrainingLogEvent::Type::RunInfo;
        event.fields["key"]     = "checkpoint";
        event.fields["value"]   = m.captured(1).trimmed();
    }
    else if ((m = rxCheckpointResume.match(line)).hasMatch()) {
        event.type              = TrainingLogEvent::Type::RunInfo;
        event.fields["key"]     = "checkpoint_resume";
        event.fields["value"]   = m.captured(1).trimmed();
    }
    else if ((m = rxDevice.match(line)).hasMatch()) {
        event.type              = TrainingLogEvent::Type::RunInfo;
        event.fields["key"]     = "device";
        event.fields["value"]   = m.captured(1);
    }
    else if ((m = rxObsSize.match(line)).hasMatch()) {
        event.type              = TrainingLogEvent::Type::RunInfo;
        event.fields["key"]     = "obs_size";
        event.fields["value"]   = m.captured(1);
    }
    else if ((m = rxActions.match(line)).hasMatch()) {
        event.type              = TrainingLogEvent::Type::RunInfo;
        event.fields["key"]     = "action_count";
        event.fields["value"]   = m.captured(1);
    }
    else if ((m = rxTotalParams.match(line)).hasMatch()) {
        event.type              = TrainingLogEvent::Type::RunInfo;
        event.fields["key"]     = "model_params";
        event.fields["value"]   = m.captured(1);
    }
    else if ((m = rxRunId.match(line)).hasMatch()) {
        event.type              = TrainingLogEvent::Type::RunInfo;
        event.fields["key"]     = "run_id";
        event.fields["value"]   = m.captured(1).trimmed();
    }
    else if ((m = rxLR.match(line)).hasMatch()) {
        event.type              = TrainingLogEvent::Type::RunInfo;
        event.fields["key"]     = "learning_rate";
        event.fields["value"]   = m.captured(1) + " / " + m.captured(2);
    }
    else if ((m = rxStat.match(line)).hasMatch()) {
        event.type              = TrainingLogEvent::Type::TrainingStat;
        event.fields["key"]     = m.captured(1).trimmed();
        event.fields["value"]   = m.captured(2).trimmed();
    }
    else if (rxError.match(line).hasMatch()) {
        event.type = TrainingLogEvent::Type::Error;
    }
    else {
        event.type = TrainingLogEvent::Type::Raw;
    }

    emit EventParsed(event);
}
