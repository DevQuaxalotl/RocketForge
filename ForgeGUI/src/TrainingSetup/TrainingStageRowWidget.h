#pragma once

#include <QWidget>
#include <QStringList>

#include "Settings/ForgeSettings.h"

#include <optional>

class QCheckBox;
class QComboBox;
class TimestepSpinBox;
class QFormLayout;
class QFrame;
class QGroupBox;
class QHBoxLayout;
class QLabel;
class QLineEdit;
class QPushButton;
class QRadioButton;
class QSpinBox;
class QTextEdit;
class QTimer;
class QToolButton;
class QVBoxLayout;

namespace RocketForge { class TrainingStage; }

/**
 * One row in the training queue. Contains a collapsible body with one segment's settings.
 * The bootstrap section is shown only when m_isFirstStage == true.
 *
 * NOTE: Target total timesteps uses TimestepSpinBox, which stores values in millions internally
 * and auto-scales the display/input to K/M/B/T (e.g. "1.5B", "500M").
 * When converting to/from StageSettings, the value is multiplied/divided by 1,000,000.
 */
class TrainingStageRowWidget : public QWidget
{
	Q_OBJECT
public:
	explicit TrainingStageRowWidget(QWidget* parent = nullptr);

	void SetAgentName(const QString& agentName);
	void SetBuildName(const QString& buildName);
	void SetIsFirstStage(bool first);
	void SetStageNumberPrefixVisible(bool visible);
	void SetStageNumber(int number);

	RocketForgeTraining::StageSettings ToStageSettings() const;
	void FromStageSettings(const RocketForgeTraining::StageSettings& s);

	void SetReadOnly(bool readOnly);
	void UpdateLiveTotalTimesteps(quint64 currentTotal);
	void UpdateEtaText(const QString& etaText);
	void MarkCompleted(const QString& versionId);
	void SetActive(bool active);
	void RefreshColorMode();
	void UpdateVersionLabel();
	void ClearCreatedIds();
	void SetLatestSegmentRevision(std::optional<GGL::SegmentVersionNumber> latestRevision);
	void SetMinorSegmentRevisionBase(std::optional<GGL::SegmentVersionNumber> baseRevision);
	void SetCompletionPreviewText(const QString& text);
	void SetDefaultBranchName(const QString& branchName);
	void SetAvailableBranchNames(const QStringList& branchNames);

	/**
	 * Called when an earlier queue entry will train or switch output segments.
	 * Queue-lineage preparation will clear this row's created ID at training
	 * start, so its label should preview a fresh bootstrap instead of a resume.
	 */
	void SetFreshQueueMode(bool fresh);

signals:
	void changed();
	void deleteRequested();
	void moveUpRequested();
	void moveDownRequested();

private slots:
	void OnExpandToggled();
	void OnAnyChanged();

private:
	void BuildHeader(QHBoxLayout* layout);
	void BuildBody();
	void RefreshAvailableStages();
	void RefreshLatestSegment();
	void UpdateBootstrapVisibility();
	void UpdateVersionGroupVisibility();
	void UpdateBranchControls();
	void RefreshVersionsForStage(QComboBox* targetCombo, const QString& stageName);
	void SetVersionPreviewText(const QString& text);
	QString CurrentComboId(const QComboBox* combo) const;
	QString CurrentComboStage(const QComboBox* combo) const;
	QString CurrentComboBranch(const QComboBox* combo) const;
	QString CurrentContinueBranchName() const;
	QString CurrentSelectedBranchName() const;
	QString ComputeNextSegmentString() const;

	QString       m_agentName;
	QString       m_buildName;
	bool          m_isFirstStage = true;
	int           m_segmentNumber = 1;
	std::optional<GGL::SegmentVersionNumber> m_latestSegmentRevision;
	std::optional<GGL::SegmentVersionNumber> m_minorSegmentRevisionBase;

	// Header widgets:
	QLabel*          m_stagePrefixLabel = nullptr;
	QComboBox*       m_stageNameCombo = nullptr;
	TimestepSpinBox* m_targetSpin = nullptr;           // stores millions; multiply by 1e6 for uint64
	QLabel*          m_currentTimestepsLabel = nullptr;
	QLabel*          m_etaLabel = nullptr;
	QLabel*          m_versionLabel = nullptr;
	QToolButton*     m_expandButton = nullptr;
	QPushButton*     m_upButton = nullptr;
	QPushButton*     m_downButton = nullptr;
	QPushButton*     m_deleteButton = nullptr;

	// Body widgets:
	QGroupBox*    m_bodyBox = nullptr;
	QGroupBox*    m_bootstrapGroup = nullptr;
	QRadioButton* m_bootstrapNewRadio = nullptr;
	QRadioButton* m_bootstrapLatestRadio = nullptr;
	QRadioButton* m_bootstrapContinueRadio = nullptr;
	QRadioButton* m_bootstrapLoadRadio = nullptr;
	QLabel*       m_latestSegmentLabel = nullptr;
	QComboBox*    m_existingVersionCombo = nullptr;
	QComboBox*    m_sourceStageCombo = nullptr;
	QComboBox*    m_sourceVersionCombo = nullptr;
	QLabel*       m_sourceStageLabel = nullptr;
	QLabel*       m_sourceVersionLabel = nullptr;
	QWidget*      m_branchOptionsWidget = nullptr;
	QRadioButton* m_branchContinueRadio = nullptr;
	QRadioButton* m_branchSelectedRadio = nullptr;
	QComboBox*    m_branchSelectedCombo = nullptr;
	QRadioButton* m_branchCreateRadio = nullptr;
	QLineEdit*    m_newBranchEdit = nullptr;
	QLabel*       m_subsequentInfoLabel = nullptr;
	QGroupBox*    m_versionGroup = nullptr;

	QRadioButton* m_versionNoneRadio = nullptr;
	QRadioButton* m_versionMinorRadio = nullptr;
	QRadioButton* m_versionMajorRadio = nullptr;
	QCheckBox*    m_overrideVersionCheck = nullptr;
	QSpinBox*     m_overrideMajorSpin = nullptr;
	QSpinBox*     m_overrideMinorSpin = nullptr;
	QTextEdit*    m_versionNotesEdit = nullptr;

	QCheckBox*    m_warmStartCheck = nullptr;
	QSpinBox*     m_warmStartIterSpin = nullptr;
	QCheckBox*    m_keepRunIdCheck = nullptr;
	QCheckBox*    m_staticOpponentsCheck = nullptr;

	QTimer*       m_notesDebounceTimer = nullptr;
	QString       m_createdSegmentId;
	QString       m_completionPreviewText;
	QString       m_latestSegmentId;
	QString       m_latestSegmentBranchName = "main";
	QString       m_continueBranchName = "main";
	bool          m_emittingChange = false;
	bool          m_freshQueueMode = false;  // an earlier row invalidates this row's saved segment
	bool          m_active = false;
};
