#include "TrainingQueueWidget.h"
#include "TrainingStageRowWidget.h"
#include "ThemeUtils.h"

#include <Storage/BuildStorage.h>
#include <TrainingSetup/VersionManagement.h>

#include <QHash>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QStringList>
#include <QVBoxLayout>

#include <optional>

using namespace RocketForgeTraining;

namespace
{
	constexpr const char* InvalidRowStyle = "QWidget { border: 1px solid #ff7d7d; }";

	bool RowWillCreateSegmentForPreview(const StageSettings& settings, int index, bool freshQueueMode)
	{
		if (index == 0) {
			if (settings.bootstrap.mode == BootstrapConfig::Mode::New)
				return true;

			if (settings.bootstrap.mode == BootstrapConfig::Mode::ContinueExisting ||
				settings.bootstrap.mode == BootstrapConfig::Mode::ContinueLatest) {
				return settings.bootstrap.selectedExistingVersionId.empty() ||
					settings.bootstrap.branchMode == BootstrapConfig::BranchMode::ContinueSelected ||
					settings.bootstrap.branchMode == BootstrapConfig::BranchMode::CreateNew ||
					settings.versionOverride.has_value() ||
					settings.versionIncrement != VersionIncrement::None;
			}

			if (settings.bootstrap.mode == BootstrapConfig::Mode::LoadFromOtherStage) {
				return settings.bootstrap.sourceVersionId.empty() ||
					settings.bootstrap.branchMode == BootstrapConfig::BranchMode::ContinueSelected ||
					settings.bootstrap.branchMode == BootstrapConfig::BranchMode::CreateNew ||
					settings.versionOverride.has_value() ||
					settings.versionIncrement != VersionIncrement::None;
			}

			return false;
		}

		return settings.createdSegmentId.empty() ||
			freshQueueMode ||
			settings.bootstrap.branchMode == BootstrapConfig::BranchMode::ContinueSelected ||
			settings.bootstrap.branchMode == BootstrapConfig::BranchMode::CreateNew ||
			settings.versionOverride.has_value() ||
			settings.versionIncrement != VersionIncrement::None;
	}

	bool IsVersionAfterOrEqual(const GGL::SegmentVersionNumber& a, const GGL::SegmentVersionNumber& b)
	{
		return a.major > b.major || (a.major == b.major && a.minor >= b.minor);
	}

	struct SegmentRevisionPreviewState
	{
		std::optional<GGL::SegmentVersionNumber> latest;
		QHash<int, GGL::SegmentVersionNumber> latestByMajor;
	};

	void RecordSegmentRevision(
		SegmentRevisionPreviewState& state,
		const GGL::SegmentVersionNumber& revision)
	{
		if (!state.latest.has_value() || IsVersionAfterOrEqual(revision, state.latest.value()))
			state.latest = revision;

		if (!state.latestByMajor.contains(revision.major) ||
			IsVersionAfterOrEqual(revision, state.latestByMajor.value(revision.major))) {
			state.latestByMajor.insert(revision.major, revision);
		}
	}

	SegmentRevisionPreviewState LoadSegmentRevisionPreviewState(
		const QString& agentName,
		const QString& buildName)
	{
		if (agentName.trimmed().isEmpty() || buildName.trimmed().isEmpty())
			return {};

		SegmentRevisionPreviewState state;
		auto segments = RocketForgeTraining::ListBuildVersions(
			agentName.toStdString(),
			buildName.toStdString(),
			{},
			true);

		for (const GGL::SegmentVersionSummary& segment : segments) {
			std::optional<GGL::SegmentVersionNumber> parsed = BuildStorage::TryParseSegmentRevision(segment.id);
			if (!parsed.has_value())
				continue;

			RecordSegmentRevision(state, parsed.value());
		}

		return state;
	}

	QHash<QString, quint64> LatestTimestepsBySegmentForBuild(
		const QString& agentName,
		const QString& buildName)
	{
		QHash<QString, quint64> result;
		if (agentName.trimmed().isEmpty() || buildName.trimmed().isEmpty())
			return result;

		auto segments = RocketForgeTraining::ListBuildVersions(
			agentName.toStdString(),
			buildName.toStdString(),
			{},
			true);

		for (const GGL::SegmentVersionSummary& segment : segments)
			result.insert(QString::fromStdString(segment.id), static_cast<quint64>(segment.latestTimesteps));

		return result;
	}

	std::optional<GGL::SegmentVersionNumber> MinorBaseRevision(
		const SegmentRevisionPreviewState& state,
		const QString& parentSegmentId)
	{
		std::optional<GGL::SegmentVersionNumber> parent =
			BuildStorage::TryParseSegmentRevision(parentSegmentId.toStdString());
		if (!parent.has_value())
			return state.latest;

		if (!state.latestByMajor.contains(parent->major))
			return parent.value();

		const GGL::SegmentVersionNumber latestInParentMajor = state.latestByMajor.value(parent->major);
		return IsVersionAfterOrEqual(latestInParentMajor, parent.value()) ? latestInParentMajor : parent.value();
	}

	QString ParentSegmentIdForPreview(
		const StageSettings& settings,
		int index,
		const QString& previousResolvedSegmentId)
	{
		if (index > 0)
			return previousResolvedSegmentId;

		if (settings.bootstrap.mode == BootstrapConfig::Mode::ContinueExisting ||
			settings.bootstrap.mode == BootstrapConfig::Mode::ContinueLatest) {
			return QString::fromStdString(settings.bootstrap.selectedExistingVersionId);
		}

		if (settings.bootstrap.mode == BootstrapConfig::Mode::LoadFromOtherStage)
			return QString::fromStdString(settings.bootstrap.sourceVersionId);

		return {};
	}

	GGL::SegmentVersionNumber ResolvePreviewSegmentRevision(
		const StageSettings& settings,
		const SegmentRevisionPreviewState& state,
		const QString& parentSegmentId)
	{
		if (settings.versionOverride.has_value())
			return settings.versionOverride.value();

		if (settings.versionIncrement == VersionIncrement::Major) {
			GGL::SegmentVersionNumber next = state.latest.value_or(GGL::SegmentVersionNumber{ 1, -1 });
			next.major += 1;
			next.minor = 0;
			return next;
		}

		GGL::SegmentVersionNumber next =
			MinorBaseRevision(state, parentSegmentId).value_or(GGL::SegmentVersionNumber{ 1, -1 });
		next.minor += 1;
		if (next.minor < 0)
			next.minor = 0;

		return next;
	}

	QString NormalizeBranchName(const QString& branchName)
	{
		const QString trimmed = branchName.trimmed();
		return trimmed.isEmpty() ? QString("main") : trimmed;
	}

	void AddUniqueBranchName(QStringList& branchNames, const QString& branchName)
	{
		const QString normalized = NormalizeBranchName(branchName);
		if (!branchNames.contains(normalized, Qt::CaseSensitive))
			branchNames.append(normalized);
	}

	QString FirstStageResumeSegmentId(const StageSettings& settings)
	{
		if (settings.bootstrap.mode == BootstrapConfig::Mode::ContinueExisting ||
			settings.bootstrap.mode == BootstrapConfig::Mode::ContinueLatest) {
			return QString::fromStdString(settings.bootstrap.selectedExistingVersionId);
		}

		if (settings.bootstrap.mode == BootstrapConfig::Mode::LoadFromOtherStage)
			return QString::fromStdString(settings.bootstrap.sourceVersionId);

		return {};
	}

	QString SegmentRevisionId(const GGL::SegmentVersionNumber& revision)
	{
		return QString::fromStdString(GGL::MakeSegmentVersionId(revision.major, revision.minor));
	}
}

// ─────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────

TrainingQueueWidget::TrainingQueueWidget(QWidget* parent)
	: QWidget(parent)
{
	setObjectName("trainingQueueWidget");

	auto* rootLayout = new QVBoxLayout(this);
	rootLayout->setContentsMargins(4, 4, 4, 4);
	rootLayout->setSpacing(4);

	// Empty state label
	m_emptyLabel = new QLabel("No segments queued. Click '+ Add Training Segment' to begin.", this);
	m_emptyLabel->setAlignment(Qt::AlignCenter);
	m_emptyLabel->setProperty(ForgeGUI::TextRoleProperty, "muted");
	m_emptyLabel->setStyleSheet("font-style: italic;");
	rootLayout->addWidget(m_emptyLabel);

	// Stage rows layout
	m_rowsLayout = new QVBoxLayout();
	m_rowsLayout->setContentsMargins(0, 0, 0, 0);
	m_rowsLayout->setSpacing(2);
	rootLayout->addLayout(m_rowsLayout);

	m_addButton = new QPushButton("+ Add Training Segment", this);
	connect(m_addButton, &QPushButton::clicked, this, &TrainingQueueWidget::OnAddStageClicked);
	rootLayout->addWidget(m_addButton, 0, Qt::AlignLeft);

	rootLayout->addStretch();

	RebuildEmptyState();
}

// ─────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────

void TrainingQueueWidget::SetAgentName(const QString& agentName)
{
	m_agentName = agentName;
	for (auto* row : m_rows)
		row->SetAgentName(agentName);
}

void TrainingQueueWidget::SetBuildName(const QString& buildName)
{
	m_buildName = buildName.trimmed();
	for (auto* row : m_rows)
		row->SetBuildName(m_buildName);
	RefreshAllVersionLabels();
}

void TrainingQueueWidget::SetStages(const std::vector<StageSettings>& stages)
{
	// Clear existing rows
	for (auto* row : m_rows) {
		m_rowsLayout->removeWidget(row);
		row->deleteLater();
	}
	m_rows.clear();

	// Populate from provided stages
	for (const auto& stage : stages) {
		auto* row = new TrainingStageRowWidget(this);
		row->SetAgentName(m_agentName);
		row->SetBuildName(m_buildName);
		row->FromStageSettings(stage);

		connect(row, &TrainingStageRowWidget::changed,          this, &TrainingQueueWidget::OnRowChanged);
		connect(row, &TrainingStageRowWidget::deleteRequested,  this, &TrainingQueueWidget::OnRowDelete);
		connect(row, &TrainingStageRowWidget::moveUpRequested,  this, &TrainingQueueWidget::OnRowMoveUp);
		connect(row, &TrainingStageRowWidget::moveDownRequested,this, &TrainingQueueWidget::OnRowMoveDown);

		m_rows.append(row);
		m_rowsLayout->addWidget(row);
	}

	RenumberRows();
	RebuildEmptyState();
	RefreshValidationStyles();
	// No changed() emit — this is a load, not a user action
}

std::vector<StageSettings> TrainingQueueWidget::Stages() const
{
	std::vector<StageSettings> result;
	result.reserve(static_cast<size_t>(m_rows.size()));
	for (const auto* row : m_rows)
		result.push_back(row->ToStageSettings());
	return result;
}

void TrainingQueueWidget::SetReadOnly(bool readOnly)
{
	for (auto* row : m_rows)
		row->SetReadOnly(readOnly);
	m_addButton->setEnabled(!readOnly);
}

void TrainingQueueWidget::SetValidationStylesSuppressed(bool suppressed)
{
	if (m_validationStylesSuppressed == suppressed)
		return;

	m_validationStylesSuppressed = suppressed;
	RefreshValidationStyles();
}

void TrainingQueueWidget::ClearCreatedIds()
{
	for (auto* row : m_rows)
		row->ClearCreatedIds();
	RefreshAllVersionLabels();
}

void TrainingQueueWidget::UpdateLiveTotalTimesteps(int stageIndex, quint64 totalTimesteps)
{
	if (stageIndex < 0 || stageIndex >= m_rows.size())
		return;
	m_rows[stageIndex]->UpdateLiveTotalTimesteps(totalTimesteps);
}

void TrainingQueueWidget::UpdateLiveEta(int stageIndex, const QString& etaText)
{
	if (stageIndex < 0 || stageIndex >= m_rows.size())
		return;
	m_rows[stageIndex]->UpdateEtaText(etaText);
}

void TrainingQueueWidget::MarkStageCompleted(int stageIndex, const QString& versionId)
{
	if (stageIndex < 0 || stageIndex >= m_rows.size())
		return;
	m_rows[stageIndex]->MarkCompleted(versionId);
}

void TrainingQueueWidget::SetActiveStage(int stageIndex)
{
	for (int i = 0; i < m_rows.size(); ++i)
		m_rows[i]->SetActive(i == stageIndex);
}

void TrainingQueueWidget::RefreshColorMode()
{
	for (TrainingStageRowWidget* row : m_rows)
		row->RefreshColorMode();
}

QStringList TrainingQueueWidget::ValidationErrors() const
{
	QStringList errors;

	quint64 prevTarget = 0;
	for (int i = 0; i < m_rows.size(); ++i) {
		const StageSettings s = m_rows[i]->ToStageSettings();
		const QString name    = QString::fromStdString(s.stageName);
		const int     segmentN  = i + 1;

		// Empty name
		if (name.trimmed().isEmpty())
			errors.append(QString("Segment %1: empty training stage name.").arg(segmentN));

		if ((s.bootstrap.branchMode == BootstrapConfig::BranchMode::CreateNew ||
			 s.bootstrap.branchMode == BootstrapConfig::BranchMode::ContinueSelected) &&
			QString::fromStdString(s.branchName).trimmed().isEmpty()) {
			errors.append(QString("Segment %1: branch name is required.").arg(segmentN));
		}

		// Non-increasing target
		if (i > 0 && s.targetTotalTimesteps <= prevTarget)
			errors.append(QString("Segment %1: target must be greater than previous segment.").arg(segmentN));

		prevTarget = s.targetTotalTimesteps;
	}

	return errors;
}

// ─────────────────────────────────────────────
// Private slots
// ─────────────────────────────────────────────

void TrainingQueueWidget::OnAddStageClicked()
{
	// Determine default target: previous row's target + 100M, or 100M if first
	quint64 defaultTarget = 100'000'000ULL;
	if (!m_rows.isEmpty()) {
		const StageSettings prev = m_rows.last()->ToStageSettings();
		defaultTarget = prev.targetTotalTimesteps + 100'000'000ULL;
	}

	StageSettings defaults;
	defaults.targetTotalTimesteps = defaultTarget;

	auto* row = new TrainingStageRowWidget(this);
	row->SetAgentName(m_agentName);
	row->SetBuildName(m_buildName);
	row->FromStageSettings(defaults);

	connect(row, &TrainingStageRowWidget::changed,          this, &TrainingQueueWidget::OnRowChanged);
	connect(row, &TrainingStageRowWidget::deleteRequested,  this, &TrainingQueueWidget::OnRowDelete);
	connect(row, &TrainingStageRowWidget::moveUpRequested,  this, &TrainingQueueWidget::OnRowMoveUp);
	connect(row, &TrainingStageRowWidget::moveDownRequested,this, &TrainingQueueWidget::OnRowMoveDown);

	m_rows.append(row);
	m_rowsLayout->addWidget(row);

	RenumberRows();
	RebuildEmptyState();
	RefreshValidationStyles();
	emit changed();
}

void TrainingQueueWidget::OnRowChanged()
{
	RefreshValidationStyles();
	RefreshAllVersionLabels();
	emit changed();
}

void TrainingQueueWidget::OnRowDelete()
{
	auto* row = qobject_cast<TrainingStageRowWidget*>(sender());
	if (!row)
		return;

	int idx = m_rows.indexOf(row);
	if (idx < 0)
		return;

	m_rows.removeAt(idx);
	m_rowsLayout->removeWidget(row);
	row->deleteLater();

	RenumberRows();
	RebuildEmptyState();
	RefreshValidationStyles();
	emit changed();
}

void TrainingQueueWidget::OnRowMoveUp()
{
	auto* row = qobject_cast<TrainingStageRowWidget*>(sender());
	if (!row)
		return;

	int idx = m_rows.indexOf(row);
	if (idx <= 0)
		return;

	int newIdx = idx - 1;
	m_rows.swapItemsAt(idx, newIdx);
	m_rowsLayout->removeWidget(row);
	m_rowsLayout->insertWidget(newIdx, row);

	RenumberRows();
	RefreshValidationStyles();
	emit changed();
}

void TrainingQueueWidget::OnRowMoveDown()
{
	auto* row = qobject_cast<TrainingStageRowWidget*>(sender());
	if (!row)
		return;

	int idx = m_rows.indexOf(row);
	if (idx < 0 || idx >= m_rows.size() - 1)
		return;

	int newIdx = idx + 1;
	m_rows.swapItemsAt(idx, newIdx);
	m_rowsLayout->removeWidget(row);
	m_rowsLayout->insertWidget(newIdx, row);

	RenumberRows();
	RefreshValidationStyles();
	emit changed();
}

// ─────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────

void TrainingQueueWidget::RebuildEmptyState()
{
	m_emptyLabel->setVisible(m_rows.isEmpty());
}

void TrainingQueueWidget::RenumberRows()
{
	const bool showPrefix = !m_rows.isEmpty();
	for (int i = 0; i < m_rows.size(); ++i) {
		m_rows[i]->SetIsFirstStage(i == 0);
		m_rows[i]->SetStageNumber(i + 1);
		m_rows[i]->SetStageNumberPrefixVisible(showPrefix);
	}
	RefreshAllVersionLabels();
}

void TrainingQueueWidget::RefreshAllVersionLabels()
{
	if (m_rows.isEmpty())
		return;

	// Mirror queue-lineage preparation: after the first row that will train or switch
	// outputs, every later row must show a fresh bootstrap instead of its persisted ID.
	SegmentRevisionPreviewState revisionState =
		LoadSegmentRevisionPreviewState(m_agentName, m_buildName);
	const QHash<QString, quint64> latestTimestepsBySegment =
		LatestTimestepsBySegmentForBuild(m_agentName, m_buildName);
	QStringList knownBranches;
	AddUniqueBranchName(knownBranches, "main");
	if (!m_agentName.trimmed().isEmpty() && !m_buildName.trimmed().isEmpty()) {
		for (const std::string& branchName : RocketForgeTraining::ListBuildBranches(
				 m_agentName.toStdString(),
				 m_buildName.toStdString(),
				 true)) {
			AddUniqueBranchName(knownBranches, QString::fromStdString(branchName));
		}
	}

	QString previousBranchName = "main";
	QString previousResolvedSegmentId;
	quint64 previousResolvedTimesteps = 0;
	bool invalidateFollowing = false;
	for (int i = 0; i < m_rows.size(); ++i) {
		m_rows[i]->SetDefaultBranchName(i == 0 ? QString("main") : previousBranchName);
		m_rows[i]->SetAvailableBranchNames(knownBranches);
		const StageSettings settings = m_rows[i]->ToStageSettings();
		const bool rowFreshQueueMode = i > 0 && invalidateFollowing;
		const bool rowWillCreate = RowWillCreateSegmentForPreview(settings, i, rowFreshQueueMode);
		const QString parentSegmentId = rowWillCreate
			? ParentSegmentIdForPreview(settings, i, previousResolvedSegmentId)
			: QString();
		QString completedSegmentId;
		QString resolvedSegmentId;
		quint64 resolvedTimesteps = 0;

		if (i == 0 && !rowWillCreate) {
			const QString selectedSegmentId = FirstStageResumeSegmentId(settings);
			resolvedSegmentId = selectedSegmentId.isEmpty()
				? QString::fromStdString(settings.createdSegmentId)
				: selectedSegmentId;
			resolvedTimesteps = latestTimestepsBySegment.value(resolvedSegmentId, 0);
			if (!resolvedSegmentId.isEmpty() && resolvedTimesteps >= settings.targetTotalTimesteps)
				completedSegmentId = resolvedSegmentId;
		} else if (i > 0 && !rowWillCreate) {
			resolvedSegmentId = QString::fromStdString(settings.createdSegmentId);
			resolvedTimesteps = latestTimestepsBySegment.value(resolvedSegmentId, 0);
			if (!resolvedSegmentId.isEmpty() && resolvedTimesteps >= settings.targetTotalTimesteps)
				completedSegmentId = resolvedSegmentId;
		} else if (i > 0 &&
				   !previousResolvedSegmentId.isEmpty() &&
				   previousResolvedTimesteps >= settings.targetTotalTimesteps) {
			completedSegmentId = previousResolvedSegmentId;
			resolvedSegmentId = previousResolvedSegmentId;
			resolvedTimesteps = previousResolvedTimesteps;
		}

		m_rows[i]->SetLatestSegmentRevision(revisionState.latest);
		m_rows[i]->SetMinorSegmentRevisionBase(MinorBaseRevision(revisionState, parentSegmentId));
		m_rows[i]->SetFreshQueueMode(rowFreshQueueMode);
		m_rows[i]->SetCompletionPreviewText(completedSegmentId.isEmpty()
			? QString()
			: QString("%1 Completed").arg(completedSegmentId));
		m_rows[i]->UpdateVersionLabel();

		const bool rowCreatesSegment = rowWillCreate && completedSegmentId.isEmpty();
		if (rowCreatesSegment) {
			const GGL::SegmentVersionNumber createdRevision =
				ResolvePreviewSegmentRevision(settings, revisionState, parentSegmentId);
			if (resolvedSegmentId.isEmpty())
				resolvedSegmentId = SegmentRevisionId(createdRevision);
			RecordSegmentRevision(revisionState, createdRevision);
		}

		if (!resolvedSegmentId.isEmpty()) {
			previousResolvedSegmentId = resolvedSegmentId;
			previousResolvedTimesteps = resolvedTimesteps;
		}

		if (!invalidateFollowing) {
			const QString persistedSegmentId = QString::fromStdString(settings.createdSegmentId);
			const bool outputIdentityChanged = resolvedSegmentId != persistedSegmentId;
			const bool rowWillTrain = completedSegmentId.isEmpty();
			if (rowWillCreate || outputIdentityChanged || rowWillTrain)
				invalidateFollowing = true;
		}

		const QString rowBranchName = NormalizeBranchName(QString::fromStdString(settings.branchName));
		AddUniqueBranchName(knownBranches, rowBranchName);
		previousBranchName = rowBranchName;
	}
}

void TrainingQueueWidget::RefreshValidationStyles()
{
	if (m_validationStylesSuppressed) {
		for (auto* row : m_rows) {
			if (row->styleSheet() == InvalidRowStyle)
				row->setStyleSheet("");
		}
		return;
	}

	quint64 prevTarget = 0;
	for (int i = 0; i < m_rows.size(); ++i) {
		const StageSettings s = m_rows[i]->ToStageSettings();
		const QString name    = QString::fromStdString(s.stageName);

		bool isInvalid = false;

		// Empty name
		if (name.trimmed().isEmpty())
			isInvalid = true;

		// Non-increasing target (only for segments after the first)
		if (!isInvalid && i > 0 && s.targetTotalTimesteps <= prevTarget)
			isInvalid = true;

		if (isInvalid)
			m_rows[i]->setStyleSheet(InvalidRowStyle);
		else
			m_rows[i]->setStyleSheet("");

		prevTarget = s.targetTotalTimesteps;
	}
}
