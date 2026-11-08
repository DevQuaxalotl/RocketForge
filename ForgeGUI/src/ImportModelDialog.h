#pragma once

#include <Storage/ImportedModelStorage.h>

#include <QDialog>
#include <QString>

#include <optional>

class QCheckBox;
class QComboBox;
class QKeyEvent;
class QLabel;
class QLineEdit;
class QPlainTextEdit;
class QPushButton;
class QToolButton;
class QWidget;

/**
 * Guided dialog for copying a trained model into an agent's Imported Models folder.
 * Shows what the picked folder contains, where it will be saved, what it can be used for,
 * and whether it loads into the agent's current networks, before anything is copied.
 * A model that doesn't fit only warns: the agent may be changed to match it later.
 */
class ImportModelDialog : public QDialog
{
	Q_OBJECT

public:
	explicit ImportModelDialog(const QString& defaultAgentName, QWidget* parent = nullptr);

	/** The import the user confirmed; valid after the dialog is accepted. */
	RocketForgeTraining::ImportedModelImportConfig ImportConfig() const;

	/** Whether the model loads into the agent's current networks; nullopt when the check hasn't finished or didn't run. */
	std::optional<bool> ModelFitsAgent() const;

protected:
	void keyPressEvent(QKeyEvent* event) override;

private:
	void BrowseForSource();
	void OnSourceChanged();
	void FillSourceFields();
	void UpdateDestination();
	void UpdateSummary();
	void StartFitCheck();
	void SetDetailsVisible(bool visible);
	void AcceptIfValid();

	/** Sizes the window to its content at the current width, growing or shrinking. */
	void FitHeightToContent();

	QString AgentName() const;
	QString FolderName() const;
	std::filesystem::path DestinationFolder() const;

	QComboBox* m_agentCombo = nullptr;
	QLineEdit* m_sourceEdit = nullptr;
	QLineEdit* m_nameEdit = nullptr;
	QLabel* m_destinationLabel = nullptr;
	QLabel* m_summaryLabel = nullptr;
	QCheckBox* m_replaceCheck = nullptr;
	QPushButton* m_importButton = nullptr;

	QToolButton* m_detailsToggle = nullptr;
	QWidget* m_detailsWidget = nullptr;
	QLineEdit* m_sourceBuildEdit = nullptr;
	QLineEdit* m_sourceStageEdit = nullptr;
	QLineEdit* m_sourceVersionEdit = nullptr;
	QPlainTextEdit* m_notesEdit = nullptr;

	/** The source path m_inspection describes; the text field may hold an unconfirmed edit. */
	QString m_inspectedSource;
	RocketForgeTraining::ImportSourceInspection m_inspection;

	/**************** Agent fit check (runs off the UI thread) ****************/
	enum class FitState { NotChecked, Checking, Fits, DoesNotFit };
	FitState m_fitState = FitState::NotChecked;
	QString m_fitProblem;

	/** Incremented per check so a result for an older folder or agent is ignored. */
	int m_fitRequestId = 0;
};
