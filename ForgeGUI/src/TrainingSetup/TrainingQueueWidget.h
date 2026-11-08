#pragma once

#include <QWidget>
#include <QVector>

#include "Settings/ForgeSettings.h"

class TrainingStageRowWidget;
class QPushButton;
class QVBoxLayout;
class QLabel;

class TrainingQueueWidget : public QWidget
{
	Q_OBJECT
public:
	explicit TrainingQueueWidget(QWidget* parent = nullptr);

	void SetAgentName(const QString& agentName);
	void SetBuildName(const QString& buildName);
	void SetStages(const std::vector<RocketForgeTraining::StageSettings>& stages);
	std::vector<RocketForgeTraining::StageSettings> Stages() const;
	void ClearCreatedIds();

	void SetReadOnly(bool readOnly);
	void SetValidationStylesSuppressed(bool suppressed);
	void UpdateLiveTotalTimesteps(int stageIndex, quint64 totalTimesteps);
	void UpdateLiveEta(int stageIndex, const QString& etaText);
	void MarkStageCompleted(int stageIndex, const QString& versionId);
	void SetActiveStage(int stageIndex);
	void RefreshColorMode();

	QStringList ValidationErrors() const;

signals:
	void changed();

private slots:
	void OnAddStageClicked();
	void OnRowChanged();
	void OnRowDelete();
	void OnRowMoveUp();
	void OnRowMoveDown();

private:
	void RebuildEmptyState();
	void RenumberRows();
	void RefreshValidationStyles();
	void RefreshAllVersionLabels();

	QString                              m_agentName;
	QString                              m_buildName;
	QVBoxLayout*                         m_rowsLayout = nullptr;
	QVector<TrainingStageRowWidget*>     m_rows;
	QPushButton*                         m_addButton = nullptr;
	QLabel*                              m_emptyLabel = nullptr;
	bool                                 m_validationStylesSuppressed = false;
};
