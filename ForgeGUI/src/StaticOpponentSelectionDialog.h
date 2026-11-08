#pragma once

#include <QDialog>
#include <QString>

#include <vector>

#include "StaticOpponents/SkillReferenceSelection.h"
#include "StaticOpponents/StaticOpponentSelection.h"

class QLabel;
class QPushButton;
class QTreeWidget;
class QTreeWidgetItem;

/** What the dialog's selection is used for; SkillReferences adds an editable MMR column per mode. */
enum class SelectionPurpose
{
	StaticOpponents,
	SkillReferences
};

/**
 * Dialog for selecting the exact saved versions or imported models a target
 * agent may train against as static opponents, or measure its skill against as references.
 **/
class StaticOpponentSelectionDialog : public QDialog
{
	Q_OBJECT

public:
	StaticOpponentSelectionDialog(
		const QString& targetAgentName,
		const QString& targetBuildName,
		bool trainingActive,
		QWidget* parent = nullptr,
		SelectionPurpose purpose = SelectionPurpose::StaticOpponents);

private slots:
	void Refresh();
	void ApplySelection();
	void OnItemChanged(QTreeWidgetItem* item, int column);

private:
	void UpdateSummary();
	void SaveSelection();
	int NotesColumn() const;
	QString PurposeTitle() const;
	std::vector<const QTreeWidgetItem*> GatherCheckedItems() const;
	std::vector<RocketForgeTraining::StaticOpponentSelectionEntry> GatherCheckedEntries() const;
	std::vector<RocketForgeTraining::SkillReferenceEdit> GatherSkillReferenceEdits() const;

	QString m_targetAgentName;
	QString m_targetBuildName;
	bool m_trainingActive = false;
	SelectionPurpose m_purpose = SelectionPurpose::StaticOpponents;
	bool m_syncingTree = false;
	QTreeWidget* m_tree = nullptr;
	QLabel* m_summaryLabel = nullptr;
	QLabel* m_readOnlyLabel = nullptr;
	QPushButton* m_applyBtn = nullptr;
	std::vector<RocketForgeTraining::StaticOpponentCandidate> m_candidates;
};
