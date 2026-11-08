#pragma once

#include <QMap>
#include <QString>

class QGridLayout;
class QLabel;
class QWidget;

namespace RocketForgeTraining
{
    class BuildStorage;
    struct ForgeSettings;
    struct TrainingSegmentMetadata;
}

/**
 * The Training tab's Skill card.
 * Shows the latest rating for each mode with its change since the segment began, an overall average,
 * and the next/previous skill test rows, whose text the owner supplies.
 */
class SkillRatingsPanel
{
public:
    /** Builds the card's grid inside the container, which owns the rows' labels. */
    explicit SkillRatingsPanel(QWidget* container);

    /** Clears every rating and baseline, leaving the default rows at "N/A". */
    void Reset();

    /** Loads the segment's start ratings, then shows the ratings saved with its latest checkpoint. */
    void SeedFromSegment(const RocketForgeTraining::BuildStorage& storage, const RocketForgeTraining::TrainingSegmentMetadata& segment);

    /** Seeds from the segment the queue's first trained stage continues or starts from. */
    void SeedFromTrainingStart(const RocketForgeTraining::ForgeSettings& settings);

    /**
     * Reads the ratings saved in the newest checkpoint under a checkpoint root.
     * With updateDisplayedRatings, shows them for each mode that has no live rating yet.
     */
    void SeedFromCheckpointRoot(const QString& checkpointRootText, bool updateDisplayedRatings);

    /** Shows a mode's latest rating. Unparsable text clears it to "N/A". */
    void UpdateRating(const QString& mode, const QString& ratingText);

    void SetTestTimingRows(const QString& nextTestText, const QString& previousTestText);

    bool HasCurrentRatings() const { return !m_currentRatings.isEmpty(); }

private:
    friend class TrainingDashboardTests;

    QLabel* EnsureValueLabel(const QString& mode);
    void UpdateOverallRow();

    QWidget*     m_container = nullptr;
    QGridLayout* m_layout    = nullptr;

    QMap<QString, QLabel*> m_nameLabels;
    QMap<QString, QLabel*> m_valueLabels;

    /** Each mode's rating when the segment began, which the displayed delta is measured from. */
    QMap<QString, double> m_startRatings;

    QMap<QString, double> m_currentRatings;
};
