#include "MainWindow/SkillRatingsPanel.h"

#include "MainWindow/DashboardFormatting.h"
#include "MainWindow/MetricCardLayout.h"
#include "MainWindow/TrainingStartPreflight.h"

#include <AgentModelPaths.h>
#include <Storage/BuildStorage.h>

#include <QFile>
#include <QGridLayout>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QRegularExpression>
#include <QSizePolicy>
#include <QStringList>

#include <cmath>
#include <filesystem>
#include <optional>

using namespace RocketForgeTraining;
using namespace ForgeGUI;

namespace
{
    constexpr const char* OverallSkillMode = "Overall";
    constexpr const char* NextSkillTestRow = "Next Test";
    constexpr const char* LastSkillTestRow = "Prev Time";
    constexpr const char* RunningStatsFileName = "RUNNING_STATS.json";

    bool IsHiddenSkillRatingMode(const QString& mode)
    {
        static const QRegularExpression rxTeamMode(
            R"(^\s*([0-9]+)v([0-9]+)\s*$)",
            QRegularExpression::CaseInsensitiveOption);
        const QRegularExpressionMatch match = rxTeamMode.match(mode);

        return match.hasMatch() && match.captured(1) != match.captured(2);
    }

    int SkillModeSortRank(const QString& mode)
    {
        if (mode == OverallSkillMode)
            return 0;
        if (mode == "1v1")
            return 1;
        if (mode == "2v2")
            return 2;
        if (mode == "3v3")
            return 3;
        if (mode == NextSkillTestRow)
            return 4;
        if (mode == LastSkillTestRow)
            return 5;
        return 100;
    }

    bool SkillModeComesBefore(const QString& lhs, const QString& rhs)
    {
        const int lhsRank = SkillModeSortRank(lhs);
        const int rhsRank = SkillModeSortRank(rhs);

        if (lhsRank != rhsRank)
            return lhsRank < rhsRank;
        return QString::localeAwareCompare(lhs, rhs) < 0;
    }

    std::optional<std::filesystem::path> PreferredCheckpointRootForSegment(
        const BuildStorage& storage,
        const TrainingSegmentMetadata& segment)
    {
        const std::optional<ResolvedCheckpoint> latest = storage.LatestSegmentCheckpoint(segment.folderName);

        if (!latest.has_value())
            return std::nullopt;

        return latest->root;
    }

    std::optional<std::filesystem::path> LatestCheckpointFolderIn(const QString& rootText)
    {
        QString pathText = rootText.trimmed();
        if (pathText.startsWith('"') && pathText.endsWith('"'))
            pathText = pathText.mid(1, pathText.size() - 2);
        const std::filesystem::path root = pathText.toStdString();
        std::error_code fileError;

        if (root.empty() || !std::filesystem::exists(root, fileError))
            return std::nullopt;

        if (std::filesystem::exists(root / RunningStatsFileName, fileError))
            return root;

        int64_t highestTimesteps = -1;
        std::filesystem::path latest;

        for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(root, fileError)) {
            if (fileError)
                break;

            std::error_code entryError;

            if (!entry.is_directory(entryError))
                continue;

            const std::optional<int64_t> timesteps = CheckpointFolderTimesteps(entry.path());

            if (timesteps.has_value() && timesteps.value() > highestTimesteps) {
                highestTimesteps = timesteps.value();
                latest = entry.path();
            }
        }

        if (highestTimesteps < 0)
            return std::nullopt;
        return latest;
    }

    QMap<QString, double> ReadSkillRatingBaselines(const std::filesystem::path& checkpointFolder)
    {
        QMap<QString, double> baselines;
        QFile file(QString::fromStdString((checkpointFolder / RunningStatsFileName).string()));

        if (!file.open(QIODevice::ReadOnly))
            return baselines;

        const QJsonDocument doc = QJsonDocument::fromJson(file.readAll());

        if (!doc.isObject())
            return baselines;

        const QJsonObject ratings = doc.object().value("skill_ratings").toObject();

        for (auto it = ratings.begin(); it != ratings.end(); ++it) {
            const QJsonObject estimate = it.value().toObject();
            const QJsonValue mu = estimate.value("mu");
            if (mu.isDouble() && std::isfinite(mu.toDouble()))
                baselines.insert(it.key(), mu.toDouble());
        }

        return baselines;
    }
}

SkillRatingsPanel::SkillRatingsPanel(QWidget* container)
    : m_container(container)
{
    m_layout = new QGridLayout(container);
    m_layout->setVerticalSpacing(2);
    m_layout->setContentsMargins(0, 0, 0, 0);
    m_layout->setAlignment(Qt::AlignTop);

    Reset();
}

void SkillRatingsPanel::Reset()
{
    m_startRatings.clear();
    m_currentRatings.clear();

    for (QLabel* label : m_nameLabels) {
        m_layout->removeWidget(label);
        delete label;
    }

    for (QLabel* label : m_valueLabels) {
        m_layout->removeWidget(label);
        delete label;
    }

    m_nameLabels.clear();
    m_valueLabels.clear();

    for (const QString& mode : QStringList{ OverallSkillMode, "1v1", "2v2", "3v3", NextSkillTestRow, LastSkillTestRow })
        EnsureValueLabel(mode)->setText("N/A");
}

void SkillRatingsPanel::SeedFromSegment(const BuildStorage& storage, const TrainingSegmentMetadata& segment)
{
    m_startRatings.clear();

    for (const auto& [mode, rating] : storage.ResolveSegmentSkillStartRatings(segment)) {
        if (!IsHiddenSkillRatingMode(QString::fromStdString(mode)))
            m_startRatings.insert(QString::fromStdString(mode), rating);
    }

    const bool updateDisplayedRatings = true;
    const std::optional<std::filesystem::path> root = PreferredCheckpointRootForSegment(storage, segment);

    if (root) {
        SeedFromCheckpointRoot(QString::fromStdString(root->string()), updateDisplayedRatings);
    } else if (!segment.sourceCheckpointPath.empty()) {
        const QString sourceRoot = QString::fromStdString(storage.ResolveBuildPath(segment.sourceCheckpointPath).string());
        SeedFromCheckpointRoot(sourceRoot, updateDisplayedRatings);
    }
}

void SkillRatingsPanel::SeedFromTrainingStart(const ForgeSettings& settings)
{
    if (settings.agentName.empty() || settings.buildName.empty() || settings.stages.empty())
        return;

    BuildStorage storage(ResolveAgentModelsFolder(settings.agentName), settings.buildName);
    ForgeSettings resolvedSettings = settings;
    PrepareSettingsForQueuePreflight(resolvedSettings, storage);
    const SegmentInspector inspector = MakePreflightInspector(storage);

    for (size_t i = 0; i < resolvedSettings.stages.size(); ++i) {
        const StageAction action = DecideAction(resolvedSettings, i, inspector);

        if (action.kind == ActionKind::Skip)
            continue;

        std::string sourceSegmentId;
        if (action.kind == ActionKind::Continue)
            sourceSegmentId = action.continueSegmentId;
        else if (!action.startNewSourceSegmentId.empty())
            sourceSegmentId = action.startNewSourceSegmentId;
        else
            return;

        std::optional<TrainingSegmentMetadata> segment = storage.LoadSegmentById(sourceSegmentId);

        if (!segment.has_value())
            return;

        if (action.kind == ActionKind::Continue) {
            SeedFromSegment(storage, *segment);
            return;
        }

        const std::optional<std::filesystem::path> checkpointRoot = PreferredCheckpointRootForSegment(storage, segment.value());

        if (!checkpointRoot.has_value())
            return;

        const bool updateDisplayedRatings = true;
        SeedFromCheckpointRoot(QString::fromStdString(checkpointRoot->string()), updateDisplayedRatings);
        return;
    }
}

void SkillRatingsPanel::SeedFromCheckpointRoot(const QString& checkpointRootText, bool updateDisplayedRatings)
{
    const std::optional<std::filesystem::path> checkpointFolder = LatestCheckpointFolderIn(checkpointRootText);

    if (!checkpointFolder.has_value())
        return;

    const QMap<QString, double> baselines = ReadSkillRatingBaselines(checkpointFolder.value());

    if (baselines.isEmpty())
        return;

    for (auto it = baselines.cbegin(); it != baselines.cend(); ++it) {
        if (IsHiddenSkillRatingMode(it.key()))
            continue;

        if (updateDisplayedRatings && !m_currentRatings.contains(it.key()))
            UpdateRating(it.key(), QString::number(it.value(), 'f', 6));
    }
}

void SkillRatingsPanel::UpdateRating(const QString& mode, const QString& ratingText)
{
    const QString normalizedMode = mode.trimmed();

    if (normalizedMode.isEmpty() || normalizedMode == OverallSkillMode || IsHiddenSkillRatingMode(normalizedMode))
        return;

    const QString trimmedRating = ratingText.trimmed();
    const std::optional<double> rating = ParseMetricNumber(trimmedRating);
    QString valueText;

    if (rating.has_value()) {
        const QString delta = m_startRatings.contains(normalizedMode)
            ? FormatSkillDelta(*rating - m_startRatings.value(normalizedMode)) : QString::fromUtf8("—");
        valueText = QString("%1 (%2)").arg(FormatSkillNumber(*rating), delta);
        m_currentRatings.insert(normalizedMode, rating.value());
    } else {
        valueText = "N/A";
        m_currentRatings.remove(normalizedMode);
    }

    QLabel* valueLabel = EnsureValueLabel(normalizedMode);
    valueLabel->setText(valueText);
    valueLabel->setToolTip(m_startRatings.contains(normalizedMode)
        ? QString("Segment start: %1. Delta includes earlier training sessions.")
            .arg(FormatSkillNumber(m_startRatings.value(normalizedMode)))
        : QString("The segment-start rating is unavailable. A later checkpoint is not used as its baseline."));
    valueLabel->show();
    UpdateOverallRow();
}

void SkillRatingsPanel::SetTestTimingRows(const QString& nextTestText, const QString& previousTestText)
{
    QLabel* nextValueLabel = EnsureValueLabel(NextSkillTestRow);
    nextValueLabel->setText(nextTestText);
    nextValueLabel->show();

    QLabel* previousValueLabel = EnsureValueLabel(LastSkillTestRow);
    previousValueLabel->setText(previousTestText);
    previousValueLabel->show();
}

QLabel* SkillRatingsPanel::EnsureValueLabel(const QString& mode)
{
    const QString normalizedMode = mode.trimmed();
    QLabel* valueLabel = m_valueLabels.value(normalizedMode, nullptr);

    if (valueLabel)
        return valueLabel;

    QLabel* nameLabel = new QLabel(QString("%1:").arg(normalizedMode), m_container);
    nameLabel->setObjectName("skillRatingNameLabel");
    nameLabel->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);

    valueLabel = MakeMetricValueLabel("skillRatingValueLabel", m_container);
    m_nameLabels.insert(normalizedMode, nameLabel);
    m_valueLabels.insert(normalizedMode, valueLabel);

    int insertIndex = 0;

    for (auto it = m_valueLabels.cbegin(); it != m_valueLabels.cend(); ++it) {
        if (it.key() == normalizedMode)
            continue;
        if (SkillModeComesBefore(it.key(), normalizedMode))
            insertIndex++;
    }

    m_layout->addWidget(nameLabel, insertIndex, 0);
    m_layout->addWidget(valueLabel, insertIndex, 1);
    PinGridRowsToTop(m_layout);

    return valueLabel;
}

void SkillRatingsPanel::UpdateOverallRow()
{
    double ratingSum = 0.0;
    double deltaSum = 0.0;
    int modeCount = 0;
    bool hasAllBaselines = true;

    for (auto it = m_currentRatings.cbegin(); it != m_currentRatings.cend(); ++it) {
        ratingSum += it.value();
        if (m_startRatings.contains(it.key()))
            deltaSum += it.value() - m_startRatings.value(it.key());
        else hasAllBaselines = false;
        modeCount++;
    }

    const QString valueText = modeCount > 0
        ? QString("%1 (%2)")
            .arg(FormatSkillNumber(ratingSum / modeCount), hasAllBaselines
                ? FormatSkillDelta(deltaSum / modeCount) : QString::fromUtf8("—"))
        : QString("N/A");

    QLabel* valueLabel = EnsureValueLabel(OverallSkillMode);
    valueLabel->setText(valueText);
    valueLabel->show();
}
