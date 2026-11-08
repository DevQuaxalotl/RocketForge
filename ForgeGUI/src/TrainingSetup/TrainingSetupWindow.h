#pragma once

#include <QWidget>
#include <QString>

#include <TrainingSetup/TrainingSetupSnapshot.h>
#include <GigaLearnCPP/SegmentVersioning.h>

#include <vector>

class QComboBox;
class QLabel;
class QStackedWidget;
class QTableWidget;
class RewardComparisonWidget;

/**
 * Read-only window that presents a saved training setup snapshot.
 * It switches between the reward table view and the serialized
 * hyperparameter list for the selected segment.
 **/
class TrainingSetupWindow : public QWidget
{
public:
    explicit TrainingSetupWindow(const RocketForge::TrainingSetupSnapshot& snapshot,
                                 const QString& versionId,
                                 const QString& checkpointPath,
                                 const QString& versionPath,
                                 const QString& buildName,
                                 QWidget* parent = nullptr);

private:
    void OnModeChanged(int index);
    void OnAgentChanged(const QString& agentName);
    void OnBuildChanged(const QString& buildName);
    void OnSegmentChanged(int index);
    void PopulateAgentSelector();
    void RefreshBuildSelector(const QString& preferredBuildName = {});
    void RefreshSegmentSelector(const QString& preferredSegmentId = {});
    bool LoadSelectedSnapshot();
    bool LoadCurrentStageSnapshot();
    void RebuildPages();
    void UpdateTitleLabels();
    void BuildHyperparameterTable();

    RocketForge::TrainingSetupSnapshot m_snapshot;
    QString m_agentName;
    QString m_buildName;
    QString m_versionId;
    QString m_checkpointPath;
    QString m_versionPath;
    bool m_updatingSelectors = false;
    std::vector<GGL::SegmentVersionSummary> m_segments;

    QLabel* m_titleLabel = nullptr;
    QLabel* m_savedAtLabel = nullptr;
    QLabel* m_statusLabel = nullptr;
    QComboBox* m_agentCombo = nullptr;
    QComboBox* m_buildCombo = nullptr;
    QComboBox* m_segmentCombo = nullptr;
    QComboBox* m_modeCombo = nullptr;
    QStackedWidget* m_stack = nullptr;
    RewardComparisonWidget* m_rewardWidget = nullptr;
    QTableWidget* m_hyperTable = nullptr;
};
