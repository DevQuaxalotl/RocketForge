#include "StaticOpponentSelectionDialog.h"
#include "FormatUtils.h"
#include "ThemeUtils.h"

#include <GigaLearnCPP/SegmentVersioning.h>

#include <QDialogButtonBox>
#include <QDoubleValidator>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QLocale>
#include <QMessageBox>
#include <QPushButton>
#include <QStyledItemDelegate>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QVBoxLayout>

#include <exception>
#include <iomanip>
#include <map>
#include <optional>
#include <sstream>

namespace
{
	constexpr int SourceAgentRole = Qt::UserRole;
	constexpr int SourceBuildRole = Qt::UserRole + 1;
	constexpr int SourceStageRole = Qt::UserRole + 2;
	constexpr int SourceVersionRole = Qt::UserRole + 3;
	constexpr int DisplayNameRole = Qt::UserRole + 4;
	constexpr int MmrLoadedTextRole = Qt::UserRole + 5;

	constexpr int FIRST_MMR_COLUMN = 3;
	constexpr double MMR_LIMIT = 1000000.0;
	constexpr int MMR_DECIMALS = 1;

	// Mode i is edited in column FIRST_MMR_COLUMN + i.
	const std::vector<std::string> MMR_MODES = { "1v1", "2v2", "3v3" };

	bool ParseMmr(const QString& text, float& value)
	{
		bool ok = false;
		value = QLocale::c().toFloat(text.trimmed(), &ok);
		return ok;
	}

	QString FormatMmr(float value)
	{
		return QString::number(value, 'f', MMR_DECIMALS);
	}

	class ReadOnlyDelegate : public QStyledItemDelegate
	{
	public:
		using QStyledItemDelegate::QStyledItemDelegate;

		QWidget* createEditor(QWidget*, const QStyleOptionViewItem&, const QModelIndex&) const override { return nullptr; }
	};

	// Edits one MMR cell: numbers only (negatives allowed), and only on a checked row.
	class MmrDelegate : public QStyledItemDelegate
	{
	public:
		using QStyledItemDelegate::QStyledItemDelegate;

		QWidget* createEditor(QWidget* parent, const QStyleOptionViewItem&, const QModelIndex& index) const override
		{
			if (index.sibling(index.row(), 0).data(Qt::CheckStateRole).toInt() != Qt::Checked)
				return nullptr;

			auto* editor = new QLineEdit(parent);
			auto* validator = new QDoubleValidator(-MMR_LIMIT, MMR_LIMIT, MMR_DECIMALS, editor);
			validator->setNotation(QDoubleValidator::StandardNotation);
			validator->setLocale(QLocale::c());
			editor->setValidator(validator);
			editor->setPlaceholderText("pending");

			return editor;
		}

		void setModelData(QWidget* editor, QAbstractItemModel* model, const QModelIndex& index) const override
		{
			const QString text = static_cast<QLineEdit*>(editor)->text().trimmed();

			// Losing focus commits whatever is typed, so an unfinished entry such as "-" is dropped rather than stored.
			float value = 0.f;
			if (!text.isEmpty() && !ParseMmr(text, value))
				return;

			model->setData(index, text.isEmpty() ? QString() : FormatMmr(value));
		}
	};

	QString FormatSkillRating(double skillRating, bool hasSkillRating)
	{
		if (!hasSkillRating)
			return {};

		std::ostringstream stream;
		stream << std::fixed << std::setprecision(1) << skillRating;
		return QString::fromStdString(stream.str());
	}

	QString FormatSourceGroupName(const QString& stageName)
	{
		return stageName == "__imported_models__"
			? "Imported Models"
			: stageName;
	}
}

StaticOpponentSelectionDialog::StaticOpponentSelectionDialog(
	const QString& targetAgentName,
	const QString& targetBuildName,
	bool trainingActive,
	QWidget* parent,
	SelectionPurpose purpose) :
	QDialog(parent),
	m_targetAgentName(targetAgentName),
	m_targetBuildName(targetBuildName.isEmpty() ? QString(GGL::DEFAULT_BUILD_NAME) : targetBuildName),
	m_trainingActive(trainingActive),
	m_purpose(purpose)
{
	setWindowTitle(PurposeTitle());
	resize(980, 560);

	auto* layout = new QVBoxLayout(this);
	layout->addWidget(new QLabel("Target: " + targetAgentName + " / " + m_targetBuildName, this));

	m_summaryLabel = new QLabel(this);
	layout->addWidget(m_summaryLabel);

	m_tree = new QTreeWidget(this);
	QStringList headerLabels = { "Source", "Latest TS", "Skill Rating" };
	if (m_purpose == SelectionPurpose::SkillReferences) {
		for (const std::string& mode : MMR_MODES)
			headerLabels << QString::fromStdString(mode) + " MMR";
	}
	headerLabels << "Notes";

	m_tree->setColumnCount(static_cast<int>(headerLabels.size()));
	m_tree->setHeaderLabels(headerLabels);
	m_tree->setRootIsDecorated(true);
	m_tree->setAlternatingRowColors(true);
	m_tree->setSelectionMode(QAbstractItemView::NoSelection);
	m_tree->header()->setSectionResizeMode(0, QHeaderView::Stretch);
	m_tree->header()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
	m_tree->header()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
	for (int column = FIRST_MMR_COLUMN; column < NotesColumn(); column++)
		m_tree->header()->setSectionResizeMode(column, QHeaderView::ResizeToContents);
	m_tree->header()->setSectionResizeMode(NotesColumn(), QHeaderView::Stretch);

	m_tree->setItemDelegate(new ReadOnlyDelegate(m_tree));
	for (int column = FIRST_MMR_COLUMN; column < NotesColumn(); column++)
		m_tree->setItemDelegateForColumn(column, new MmrDelegate(m_tree));
	layout->addWidget(m_tree, 1);

	auto* buttonRow = new QDialogButtonBox(QDialogButtonBox::Close, this);
	auto* refreshBtn = buttonRow->addButton("Refresh", QDialogButtonBox::ActionRole);
	m_applyBtn = buttonRow->addButton("Apply", QDialogButtonBox::AcceptRole);
	m_readOnlyLabel = new QLabel("Read only during training", this);
	m_readOnlyLabel->setProperty(ForgeGUI::TextRoleProperty, "muted");
	layout->addWidget(m_readOnlyLabel);
	layout->addWidget(buttonRow);

	m_readOnlyLabel->setVisible(m_trainingActive);
	m_applyBtn->setEnabled(!m_trainingActive);
	m_tree->setEnabled(!m_trainingActive);

	connect(refreshBtn, &QPushButton::clicked, this, &StaticOpponentSelectionDialog::Refresh);
	connect(m_applyBtn, &QPushButton::clicked, this, &StaticOpponentSelectionDialog::ApplySelection);
	connect(buttonRow, &QDialogButtonBox::rejected, this, &QDialog::reject);
	connect(m_tree, &QTreeWidget::itemChanged, this, &StaticOpponentSelectionDialog::OnItemChanged);

	Refresh();
}

void StaticOpponentSelectionDialog::Refresh()
{
	m_syncingTree = true;
	m_tree->clear();

	const QString failureText = "Failed to load " + PurposeTitle().toLower() + ".";
	std::map<std::string, std::map<std::string, float>> ratingsByKey = {};
	try {
		m_candidates = RocketForgeTraining::ListStaticOpponentCandidates(
			m_targetAgentName.toStdString(),
			m_targetBuildName.toStdString());

		if (m_purpose == SelectionPurpose::SkillReferences) {
			// The candidates mark the static-opponent selection, so the checks and ratings come from the reference file instead.
			const std::vector<RocketForgeTraining::SkillReferenceEntry> references = RocketForgeTraining::LoadSkillReferenceSelection(
				m_targetAgentName.toStdString(),
				m_targetBuildName.toStdString());
			for (const RocketForgeTraining::SkillReferenceEntry& reference : references)
				ratingsByKey[RocketForgeTraining::MakeStaticOpponentSelectionKey(reference.selection)] = reference.skillRatings;

			for (RocketForgeTraining::StaticOpponentCandidate& candidate : m_candidates)
				candidate.selected = ratingsByKey.contains(RocketForgeTraining::MakeStaticOpponentSelectionKey(candidate.selection));
		}
	} catch (const std::exception& e) {
		m_candidates.clear();
		m_syncingTree = false;
		m_summaryLabel->setText(failureText);
		QMessageBox::critical(this, PurposeTitle(), QString::fromStdString(e.what()));
		return;
	} catch (...) {
		m_candidates.clear();
		m_syncingTree = false;
		m_summaryLabel->setText(failureText);
		QMessageBox::critical(this, PurposeTitle(), failureText);
		return;
	}

	QTreeWidgetItem* currentAgentItem = nullptr;
	QTreeWidgetItem* currentBuildItem = nullptr;
	QTreeWidgetItem* currentStageItem = nullptr;
	QString currentAgentName;
	QString currentBuildName;
	QString currentStageName;

	for (const auto& candidate : m_candidates) {
		const QString agentName = QString::fromStdString(candidate.selection.sourceAgentName);
		const QString buildName = QString::fromStdString(candidate.selection.sourceBuildName.empty()
			? std::string(GGL::DEFAULT_BUILD_NAME)
			: candidate.selection.sourceBuildName);
		const QString rawStageName = QString::fromStdString(candidate.selection.sourceStageName);
		const QString stageName = FormatSourceGroupName(rawStageName);
		const QString versionId = QString::fromStdString(candidate.selection.sourceVersionId);
		const QString notes = QString::fromStdString(candidate.notes).simplified();

		if (!currentAgentItem || currentAgentName != agentName) {
			currentAgentName = agentName;
			currentBuildName.clear();
			currentStageName.clear();
			currentAgentItem = new QTreeWidgetItem(m_tree, { agentName });
			currentAgentItem->setFirstColumnSpanned(true);
			currentAgentItem->setFlags(currentAgentItem->flags() | Qt::ItemIsAutoTristate);
		}

		if (!currentBuildItem || currentBuildName != buildName) {
			currentBuildName = buildName;
			currentStageName.clear();
			currentBuildItem = new QTreeWidgetItem(currentAgentItem, { buildName });
			currentBuildItem->setFirstColumnSpanned(true);
			currentBuildItem->setFlags(currentBuildItem->flags() | Qt::ItemIsAutoTristate);
		}

		if (!currentStageItem || currentStageName != stageName) {
			currentStageName = stageName;
			currentStageItem = new QTreeWidgetItem(currentBuildItem, { stageName });
			currentStageItem->setFirstColumnSpanned(true);
			currentStageItem->setFlags(currentStageItem->flags() | Qt::ItemIsAutoTristate);
		}

		auto* versionItem = new QTreeWidgetItem(currentStageItem);
		versionItem->setText(0, versionId);
		versionItem->setText(1, ForgeGUI::FormatTimesteps(candidate.latestTimesteps));
		versionItem->setText(2, FormatSkillRating(candidate.skillRating, candidate.hasSkillRating));
		versionItem->setText(NotesColumn(), notes);
		versionItem->setToolTip(0, QString::fromStdString(candidate.selection.displayName));
		versionItem->setToolTip(NotesColumn(), QString::fromStdString(candidate.notes));

		if (m_purpose == SelectionPurpose::SkillReferences) {
			const auto storedRatings = ratingsByKey.find(RocketForgeTraining::MakeStaticOpponentSelectionKey(candidate.selection));
			for (size_t mode = 0; mode < MMR_MODES.size(); mode++) {
				const int column = FIRST_MMR_COLUMN + static_cast<int>(mode);
				if (storedRatings != ratingsByKey.end() && storedRatings->second.contains(MMR_MODES[mode]))
					versionItem->setText(column, FormatMmr(storedRatings->second.at(MMR_MODES[mode])));

				versionItem->setData(column, MmrLoadedTextRole, versionItem->text(column));
				versionItem->setToolTip(column, "Blank means pending calibration; clearing a value recalibrates it.");
			}

			versionItem->setFlags(versionItem->flags() | Qt::ItemIsEditable);
		}

		versionItem->setData(0, SourceAgentRole, agentName);
		versionItem->setData(0, SourceBuildRole, buildName);
		versionItem->setData(0, SourceStageRole, rawStageName);
		versionItem->setData(0, SourceVersionRole, versionId);
		versionItem->setData(0, DisplayNameRole, QString::fromStdString(candidate.selection.displayName));
		versionItem->setFlags(versionItem->flags() | Qt::ItemIsUserCheckable);
		versionItem->setCheckState(0, candidate.selected ? Qt::Checked : Qt::Unchecked);
	}

	m_tree->expandAll();
	m_syncingTree = false;
	UpdateSummary();
}

void StaticOpponentSelectionDialog::ApplySelection()
{
	try {
		SaveSelection();
		accept();
	} catch (const std::exception& e) {
		QMessageBox::critical(this, PurposeTitle(), QString::fromStdString(e.what()));
	} catch (...) {
		QMessageBox::critical(this, PurposeTitle(), "Failed to save " + PurposeTitle().toLower() + ".");
	}
}

void StaticOpponentSelectionDialog::SaveSelection()
{
	const std::string agentName = m_targetAgentName.toStdString();
	const std::string buildName = m_targetBuildName.toStdString();

	if (m_purpose == SelectionPurpose::StaticOpponents) {
		RocketForgeTraining::SaveStaticOpponentSelection(agentName, buildName, GatherCheckedEntries());
		return;
	}

	// Merge against the file as it is now (see MergeSkillReferenceEdits).
	const std::vector<RocketForgeTraining::SkillReferenceEntry> onDisk =
		RocketForgeTraining::LoadSkillReferenceSelection(agentName, buildName);
	RocketForgeTraining::SaveSkillReferenceSelection(
		agentName,
		buildName,
		RocketForgeTraining::MergeSkillReferenceEdits(onDisk, GatherSkillReferenceEdits()));
}

void StaticOpponentSelectionDialog::OnItemChanged(QTreeWidgetItem* /*item*/, int column)
{
	if (m_syncingTree)
		return;

	if (column >= FIRST_MMR_COLUMN && column < NotesColumn())
		return;

	UpdateSummary();
}

int StaticOpponentSelectionDialog::NotesColumn() const
{
	return m_purpose == SelectionPurpose::SkillReferences ? FIRST_MMR_COLUMN + static_cast<int>(MMR_MODES.size()) : FIRST_MMR_COLUMN;
}

QString StaticOpponentSelectionDialog::PurposeTitle() const
{
	return m_purpose == SelectionPurpose::SkillReferences ? "Skill References" : "Static Opponents";
}

std::vector<const QTreeWidgetItem*> StaticOpponentSelectionDialog::GatherCheckedItems() const
{
	std::vector<const QTreeWidgetItem*> items = {};

	for (int i = 0; i < m_tree->topLevelItemCount(); i++) {
		const QTreeWidgetItem* agentItem = m_tree->topLevelItem(i);
		for (int j = 0; j < agentItem->childCount(); j++) {
			const QTreeWidgetItem* buildItem = agentItem->child(j);
			for (int k = 0; k < buildItem->childCount(); k++) {
				const QTreeWidgetItem* stageItem = buildItem->child(k);
				for (int m = 0; m < stageItem->childCount(); m++) {
					const QTreeWidgetItem* versionItem = stageItem->child(m);
					if (versionItem->checkState(0) == Qt::Checked)
						items.push_back(versionItem);
				}
			}
		}
	}

	return items;
}

std::vector<RocketForgeTraining::StaticOpponentSelectionEntry> StaticOpponentSelectionDialog::GatherCheckedEntries() const
{
	std::vector<RocketForgeTraining::StaticOpponentSelectionEntry> entries = {};

	for (const QTreeWidgetItem* versionItem : GatherCheckedItems()) {
		RocketForgeTraining::StaticOpponentSelectionEntry entry = {};
		entry.sourceAgentName = versionItem->data(0, SourceAgentRole).toString().toStdString();
		entry.sourceBuildName = versionItem->data(0, SourceBuildRole).toString().toStdString();
		entry.sourceStageName = versionItem->data(0, SourceStageRole).toString().toStdString();
		entry.sourceVersionId = versionItem->data(0, SourceVersionRole).toString().toStdString();
		entry.displayName = versionItem->data(0, DisplayNameRole).toString().toStdString();
		entries.push_back(entry);
	}

	return entries;
}

std::vector<RocketForgeTraining::SkillReferenceEdit> StaticOpponentSelectionDialog::GatherSkillReferenceEdits() const
{
	const std::vector<RocketForgeTraining::StaticOpponentSelectionEntry> entries = GatherCheckedEntries();
	const std::vector<const QTreeWidgetItem*> items = GatherCheckedItems();

	std::vector<RocketForgeTraining::SkillReferenceEdit> edits = {};
	for (size_t i = 0; i < entries.size(); i++) {
		RocketForgeTraining::SkillReferenceEdit edit = {};
		edit.selection = entries[i];

		for (size_t mode = 0; mode < MMR_MODES.size(); mode++) {
			const int column = FIRST_MMR_COLUMN + static_cast<int>(mode);
			// Only a cell whose text differs from what was loaded is an edit; the rest keep the file's current rating.
			if (items[i]->text(column) == items[i]->data(column, MmrLoadedTextRole).toString())
				continue;

			float rating = 0.f;
			const bool hasValue = ParseMmr(items[i]->text(column), rating);
			edit.editedRatings[MMR_MODES[mode]] = hasValue ? std::optional<float>(rating) : std::nullopt;
		}

		edits.push_back(std::move(edit));
	}

	return edits;
}

void StaticOpponentSelectionDialog::UpdateSummary()
{
	const size_t selectedCount = GatherCheckedItems().size();
	if (m_candidates.empty()) {
		m_summaryLabel->setText("No saved models are available.");
		return;
	}

	const QString noun = m_purpose == SelectionPurpose::SkillReferences ? "skill reference" : "static opponent";
	m_summaryLabel->setText(QString("Selected %1 %2%3").arg(selectedCount).arg(noun).arg(selectedCount == 1 ? "" : "s"));
}
