#pragma once

#include "Common/WorkerThreadHost.h"
#include "Common/ModelCatalog.h"
#include "Widgets/RunProgressPanel.h"

#include <ParamTesting/ParamTesting.h>

#include <QWidget>

#include <optional>
#include <vector>

class AgentComboBox;
class TimestepSpinBox;
class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QScrollArea;
class QSpinBox;
class QVBoxLayout;
class QWidget;
class ParamTestingWorker;

class ParamTestingWidget : public QWidget
{
	Q_OBJECT
public:
	explicit ParamTestingWidget(QWidget* parent = nullptr);
	~ParamTestingWidget() override;

	bool IsRunning() const;
	void SetExternalRunActive(bool active);

	/** Switch to the training agent unless a run holds the selection. */
	void FollowTrainingAgent(const QString& agentName);
	void SetColorsEnabled(bool enabled);

signals:
	void RunningChanged(bool running);
	void BenchmarkSetupReady();
	void TrainingPreviewRequested(RocketForgeTraining::ParamTestingConfig config);
	void TrainingPreviewStageStarted(int index, QString stageName, QString versionId);
	void TrainingPreviewStageCompleted(int index, QString versionId);
	void TrainingPreviewFinished();
	void BenchmarkRunRequested();

private slots:
	void OnAgentChanged(const QString& agentName);
	void OnBuildFilterChanged();
	void OnStageFilterChanged();
	void OnAddSegmentClicked();
	void OnRunClicked();
	void OnCancelClicked();
	void OnProgressUpdated(const RocketForgeTraining::ParamTestingProgress& progress);
	void OnCompleted(const RocketForgeTraining::ParamTestingResult& result);
	void OnCancelled();
	void OnFailed(const QString& message, bool simulatorUnavailable);
	void UpdateInputState();

private:
	friend class TrainingDashboardTests;

	std::shared_ptr<const ModelCatalogSnapshot> m_catalog;

	struct SegmentRow {
		QString stageName;
		RocketForgeTraining::VersionIncrement versionIncrement = RocketForgeTraining::VersionIncrement::Minor;
		std::optional<GGL::SegmentVersionNumber> versionOverride;
		QString versionNotes;
		bool detailsExpanded = false;
		std::vector<RocketForgeTraining::StageParameterOverride> overrides;
	};

	QString CurrentAgentName() const;
	QString CurrentBuildFilterName() const;
	QString CurrentStageFilterName() const;
	QString CurrentBaselineBuildName() const;
	QString CurrentBaselineStageName() const;
	QString CurrentBaselineSegmentId() const;
	void RefreshBuildSelector();
	void RefreshBaselineStageSelector();
	void RefreshBaselineVersionSelector();
	void RefreshSegmentList();
	void SetRunning(bool running);
	void AppendLog(const QString& text);
	RocketForgeTraining::ParamTestingConfig BuildConfig() const;
	QStringList ValidationErrors() const;
	QString ModsSummary(const SegmentRow& row) const;
	QString PlannedVersionText(int segmentIndex) const;
	QStringList CompatibleStageNames() const;

	AgentComboBox* m_agentCombo = nullptr;
	QComboBox* m_buildCombo = nullptr;
	QComboBox* m_baselineStageCombo = nullptr;
	QComboBox* m_baselineVersionCombo = nullptr;
	TimestepSpinBox* m_segmentTimestepsSpin = nullptr;
	QLineEdit* m_branchEdit = nullptr;
	QCheckBox* m_metricsCheck = nullptr;
	QCheckBox* m_skillTestingCheck = nullptr;
	QCheckBox* m_runBenchmarkCheck = nullptr;
	QSpinBox* m_benchmarkGamesSpin = nullptr;
	QScrollArea* m_segmentScrollArea = nullptr;
	QWidget* m_segmentListWidget = nullptr;
	QVBoxLayout* m_segmentListLayout = nullptr;
	QPushButton* m_addSegmentButton = nullptr;
	QPushButton* m_runButton = nullptr;
	QPushButton* m_cancelButton = nullptr;
	RunProgressPanel* m_progressPanel = nullptr;

	WorkerThreadHost m_workerHost;
	ParamTestingWorker* m_worker = nullptr;
	bool m_running = false;
	bool m_externalRunActive = false;
	bool m_runBenchmarkAfterTraining = false;
	bool m_refreshingSegments = false;
	std::vector<SegmentRow> m_segments;
};
