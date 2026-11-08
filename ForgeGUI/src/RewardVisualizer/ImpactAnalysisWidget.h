#pragma once

#include <QWidget>
#include <QString>

class AgentComboBox;
class BuildComboBox;
class SegmentComboBox;
class RewardImpactWidget;
class QComboBox;
class QLabel;

/**
 * Dedicated "Impact Analysis" tab. Lets the user pick an agent, build, and segment,
 * then independently pick which registered training stage to analyse (defaulting to
 * the segment's trained-with stage). Resolves the segment's checkpoint/version paths
 * and drives an embedded RewardImpactWidget via SetContext().
 */
class ImpactAnalysisWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ImpactAnalysisWidget(QWidget* parent = nullptr);

    /** Repopulate selectors (e.g. when the tab becomes visible or builds change). */
    void Refresh();

    /** Switch to the training agent. */
    void FollowTrainingAgent(const QString& agentName);

private slots:
    void OnAgentChanged();
    void OnBuildChanged();
    void OnSegmentChanged();
    void OnStageChanged();

private:
    /** Re-resolve paths for the current selection and push them into the results widget. */
    void PushContext();

    /** (Re)populate the stage combo from StageRegistry, defaulting to the segment's stage. */
    void RefreshStageCombo();

    /** Show statusText and clear the embedded results widget (no target selected/resolvable). */
    void ClearResults(const QString& statusText);

    AgentComboBox*      m_agentCombo   = nullptr;
    BuildComboBox*      m_buildCombo   = nullptr;
    SegmentComboBox*    m_segmentCombo = nullptr;
    QComboBox*          m_stageCombo   = nullptr;
    QLabel*             m_trainedHint  = nullptr;
    QLabel*             m_statusLabel  = nullptr;
    RewardImpactWidget* m_results      = nullptr;
};
