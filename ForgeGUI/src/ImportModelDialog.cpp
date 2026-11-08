#include "ImportModelDialog.h"

#include "ThemeUtils.h"

#include <GigaLearnCPP/TrainingStorage.h>
#include <GigaLearnInfer/AgentRegistry.h>
#include <Inference/InferenceModelSource.h>

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDir>
#include <QFileDialog>
#include <QFormLayout>
#include <QFrame>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QStringList>
#include <QThread>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>

#include <algorithm>
#include <memory>
#include <mutex>

using namespace RocketForgeTraining;

namespace
{
	constexpr const char* DIALOG_TITLE = "Import Model";
	constexpr const char* UNKNOWN_SOURCE_VERSION = "1.0";

	// UTF-8 byte escapes, so the symbols survive any source code page the compiler assumes.
	const QString CHECK_MARK = QString::fromUtf8("\xE2\x9C\x93");
	const QString WARNING_SIGN = QString::fromUtf8("\xE2\x9A\xA0");

	QString ToQString(const std::filesystem::path& path)
	{
		return QString::fromStdWString(path.wstring());
	}

	QString FormatMegabytes(uint64_t bytes)
	{
		return QString("%1 MB").arg(bytes / (1024.0 * 1024.0), 0, 'f', 1);
	}

	QString Colored(const QString& text, const QColor& color)
	{
		return QString("<span style='color:%1'>%2</span>").arg(color.name(), text);
	}

	// A segment checkpoint is named after its build and segment;
	// anything else after its folder, skipping timestep and generic folders.
	QString DefaultModelName(const ImportSourceInspection& inspection)
	{
		if (inspection.segment.has_value())
			return QString::fromStdString(inspection.segment->buildName + " " + inspection.segment->segmentId);

		std::filesystem::path namedFolder = inspection.modelFolder;
		if (GGL::TrainingStorage::IsAllDigits(namedFolder.filename().string()))
			namedFolder = namedFolder.parent_path();

		// Bot releases keep their model in a generically named folder; the folder above names the bot.
		const QStringList genericFolderNames = { "rlbot", "src", "bot", "model", "models", "checkpoints" };
		if (genericFolderNames.contains(ToQString(namedFolder.filename()), Qt::CaseInsensitive) && namedFolder.has_parent_path())
			namedFolder = namedFolder.parent_path();

		return ToQString(namedFolder.filename());
	}

	QString TrainingStateSummary(const ImportedModelTrainingFileStatus& files)
	{
		const std::vector<std::string> missing = files.MissingTrainingFiles();
		if (files.hasCritic && missing.empty())
			return "Also includes the critic, optimizers and running stats.";

		if (!files.hasCritic && !files.hasRunningStats && !files.hasPolicyOptim)
			return "Policy only: no critic or optimizer files.";

		QStringList missingNames;
		for (const std::string& name : missing)
			missingNames << QString::fromStdString(name);

		return "Includes part of the training state (missing " + missingNames.join(", ") + ").";
	}
}

ImportModelDialog::ImportModelDialog(const QString& defaultAgentName, QWidget* parent)
	: QDialog(parent)
{
	setWindowTitle(DIALOG_TITLE);
	resize(680, 0);

	auto* layout = new QVBoxLayout(this);

	auto* introLabel = new QLabel(
		"Copies a trained model into an agent, so it can be used as a static opponent, "
		"in Model Benchmark and in Debug render. "
		"The agent's observation builder, action parser and network sizes must match the model.",
		this);
	introLabel->setWordWrap(true);
	introLabel->setProperty(ForgeGUI::TextRoleProperty, "muted");
	layout->addWidget(introLabel);

	// Agent, source and name
	auto* form = new QFormLayout;

	m_agentCombo = new QComboBox(this);
	for (const std::string& name : AgentRegistry::GetAvailableAgents())
		m_agentCombo->addItem(QString::fromStdString(name));

	const int agentIndex = m_agentCombo->findText(defaultAgentName);
	if (agentIndex >= 0)
		m_agentCombo->setCurrentIndex(agentIndex);

	m_agentCombo->setToolTip(
		"<qt>The agent that will use this model. "
		"Its observation builder, action parser and network sizes must match the ones the model was trained with.</qt>");

	m_sourceEdit = new QLineEdit(this);
	m_sourceEdit->setPlaceholderText("Browse, or paste the path of the folder that holds the model files");
	auto* browseButton = new QPushButton("Browse...", this);
	browseButton->setAutoDefault(false);

	auto* sourceRow = new QHBoxLayout;
	sourceRow->addWidget(m_sourceEdit, 1);
	sourceRow->addWidget(browseButton);

	const QString sourceTooltip =
		"<qt>The folder that holds the model files: <b>POLICY.lt</b> and <b>SHARED_HEAD.lt</b>, "
		"or <b>INFERENCE_POLICY.json</b> for a scripted policy.<br><br>"
		"You can also pick a folder of numbered checkpoints, "
		"or a RocketForge segment folder (<i>Models/&lt;build&gt;/segments/&lt;id&gt;_&lt;stage&gt;</i>); "
		"the newest checkpoint is used.<br><br>"
		"Only the model files are copied.</qt>";
	m_sourceEdit->setToolTip(sourceTooltip);
	browseButton->setToolTip(sourceTooltip);

	m_nameEdit = new QLineEdit(this);
	m_nameEdit->setToolTip("<qt>Shown in model lists. The model's folder name is made from it.</qt>");

	m_destinationLabel = new QLabel(this);
	m_destinationLabel->setWordWrap(true);
	m_destinationLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
	m_destinationLabel->setProperty(ForgeGUI::TextRoleProperty, "muted");

	form->addRow("Agent:", m_agentCombo);
	form->addRow("Model folder:", sourceRow);
	form->addRow("Name:", m_nameEdit);
	form->addRow("Saves to:", m_destinationLabel);
	layout->addLayout(form);

	// What the folder contains
	auto* summaryFrame = new QFrame(this);
	summaryFrame->setFrameShape(QFrame::StyledPanel);

	m_summaryLabel = new QLabel(summaryFrame);
	m_summaryLabel->setWordWrap(true);
	m_summaryLabel->setTextFormat(Qt::RichText);
	m_summaryLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);

	auto* summaryLayout = new QVBoxLayout(summaryFrame);
	summaryLayout->addWidget(m_summaryLabel);
	layout->addWidget(summaryFrame);

	// Optional provenance, collapsed by default
	m_detailsToggle = new QToolButton(this);
	m_detailsToggle->setText("Where it came from (optional)");
	m_detailsToggle->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
	m_detailsToggle->setAutoRaise(true);
	m_detailsToggle->setCheckable(true);
	layout->addWidget(m_detailsToggle);

	m_detailsWidget = new QWidget(this);
	m_sourceBuildEdit = new QLineEdit(m_detailsWidget);
	m_sourceStageEdit = new QLineEdit(m_detailsWidget);
	m_sourceVersionEdit = new QLineEdit(m_detailsWidget);
	m_notesEdit = new QPlainTextEdit(m_detailsWidget);

	m_sourceBuildEdit->setPlaceholderText("e.g. a build name or a bot pack");
	m_sourceVersionEdit->setPlaceholderText("Segment or version, e.g. v1.2 or 1.0");
	m_sourceStageEdit->setPlaceholderText("Stage it was trained on, if known");
	m_notesEdit->setPlaceholderText("Notes");
	m_notesEdit->setFixedHeight(72);

	const QString provenanceTooltip =
		"<qt>For your reference only: shown in the Build, Made From and Stage columns of File Management, and in Model Benchmark. "
		"These labels don't change how the model is used. "
		"They're filled in automatically for a RocketForge segment checkpoint.</qt>";
	for (QLineEdit* field : { m_sourceBuildEdit, m_sourceStageEdit, m_sourceVersionEdit })
		field->setToolTip(provenanceTooltip);

	auto* detailsForm = new QFormLayout(m_detailsWidget);
	detailsForm->setContentsMargins(16, 0, 0, 0);
	detailsForm->addRow("Build:", m_sourceBuildEdit);
	detailsForm->addRow("Made from:", m_sourceVersionEdit);
	detailsForm->addRow("Stage:", m_sourceStageEdit);
	detailsForm->addRow("Notes:", m_notesEdit);
	layout->addWidget(m_detailsWidget);
	SetDetailsVisible(false);

	// Confirm
	m_replaceCheck = new QCheckBox("Replace the existing model with this name", this);
	layout->addWidget(m_replaceCheck);

	auto* buttons = new QDialogButtonBox(QDialogButtonBox::Cancel, this);
	m_importButton = buttons->addButton("Import", QDialogButtonBox::AcceptRole);
	layout->addWidget(buttons);

	connect(browseButton, &QPushButton::clicked, this, &ImportModelDialog::BrowseForSource);
	connect(m_sourceEdit, &QLineEdit::editingFinished, this, &ImportModelDialog::OnSourceChanged);
	connect(m_agentCombo, &QComboBox::currentTextChanged, this, [this]() {
		UpdateDestination();
		StartFitCheck();
	});
	connect(m_nameEdit, &QLineEdit::textChanged, this, &ImportModelDialog::UpdateDestination);
	connect(m_detailsToggle, &QToolButton::toggled, this, &ImportModelDialog::SetDetailsVisible);
	connect(m_importButton, &QPushButton::clicked, this, &ImportModelDialog::AcceptIfValid);
	connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

	UpdateDestination();
	UpdateSummary();
}

ImportedModelImportConfig ImportModelDialog::ImportConfig() const
{
	ImportedModelImportConfig config = {};
	config.targetAgentName = AgentName().toStdString();
	config.sourceFolder = std::filesystem::path(m_inspectedSource.toStdWString());
	config.folderName = FolderName().toStdString();
	config.displayName = m_nameEdit->text().trimmed().toStdString();
	config.sourceType = m_inspection.segment.has_value() ? "rocketforge-segment" : "external";
	config.sourceBuildName = m_sourceBuildEdit->text().trimmed().toStdString();
	config.sourceStageName = m_sourceStageEdit->text().trimmed().toStdString();
	config.sourceVersionId = m_sourceVersionEdit->text().trimmed().toStdString();
	config.notes = m_notesEdit->toPlainText().trimmed().toStdString();
	config.overwrite = m_replaceCheck->isChecked();

	return config;
}

void ImportModelDialog::BrowseForSource()
{
	const QString selected = QFileDialog::getExistingDirectory(this, "Select Model Folder", m_sourceEdit->text());
	if (selected.isEmpty())
		return;

	m_sourceEdit->setText(QDir::toNativeSeparators(selected));
	OnSourceChanged();
}

void ImportModelDialog::OnSourceChanged()
{
	QString sourceText = m_sourceEdit->text().trimmed();

	// Explorer's "Copy as path" wraps the path in quotes.
	if (sourceText.size() >= 2 && sourceText.startsWith('"') && sourceText.endsWith('"'))
		sourceText = sourceText.mid(1, sourceText.size() - 2);

	sourceText = QDir::toNativeSeparators(sourceText);

	if (sourceText == m_inspectedSource)
		return;

	m_sourceEdit->setText(sourceText);
	m_inspectedSource = sourceText;
	m_inspection = InspectImportSource(std::filesystem::path(sourceText.toStdWString()));

	FillSourceFields();
	UpdateDestination();
	StartFitCheck();
}

void ImportModelDialog::FillSourceFields()
{
	m_sourceBuildEdit->clear();
	m_sourceStageEdit->clear();
	m_sourceVersionEdit->clear();

	if (!m_inspection.IsLoadable())
		return;

	m_nameEdit->setText(DefaultModelName(m_inspection));

	// Only a recognised RocketForge segment fills these; an unknown layout isn't guessed from its path.
	if (m_inspection.segment.has_value()) {
		m_sourceBuildEdit->setText(QString::fromStdString(m_inspection.segment->buildName));
		m_sourceStageEdit->setText(QString::fromStdString(m_inspection.segment->stageName));
		m_sourceVersionEdit->setText(QString::fromStdString(m_inspection.segment->segmentId));
	} else {
		m_sourceVersionEdit->setText(UNKNOWN_SOURCE_VERSION);
	}
}

void ImportModelDialog::UpdateDestination()
{
	const std::filesystem::path destination = DestinationFolder();
	if (destination.empty()) {
		m_destinationLabel->setText(AgentName().isEmpty() ? "Pick an agent." : "Enter a name.");
		m_replaceCheck->setEnabled(false);
	} else {
		const bool exists = std::filesystem::exists(destination);

		QString text = ToQString(destination).toHtmlEscaped();
		if (exists)
			text += "<br>" + Colored("A model with this name already exists.", ForgeGUI::Theme::NEGATIVE);
		m_destinationLabel->setText(text);

		m_replaceCheck->setEnabled(exists);
		if (!exists)
			m_replaceCheck->setChecked(false);
	}

	m_importButton->setEnabled(m_inspection.IsLoadable() && !destination.empty());
}

void ImportModelDialog::UpdateSummary()
{
	if (m_inspectedSource.isEmpty()) {
		m_summaryLabel->setText("Pick a model folder to see what it contains.");
		return;
	}

	if (!m_inspection.IsLoadable()) {
		m_summaryLabel->setText(Colored(
			"No model found here. Pick the folder that contains POLICY.lt (or INFERENCE_POLICY.json), "
			"a folder of numbered checkpoints, or a RocketForge segment folder.",
			ForgeGUI::Theme::NEGATIVE));
		return;
	}

	QStringList lines;

	QStringList fileNames;
	for (const std::string& name : m_inspection.filesToCopy)
		fileNames << QString::fromStdString(name);
	lines << QString("<b>Files to copy:</b> %1 (%2)").arg(fileNames.join(", "), FormatMegabytes(m_inspection.bytesToCopy));

	const std::filesystem::path source(m_inspectedSource.toStdWString());
	if (m_inspection.modelFolder != source)
		lines << "Uses the newest checkpoint: " + ToQString(m_inspection.modelFolder.filename()).toHtmlEscaped();

	if (m_inspection.segment.has_value()) {
		const TrainingSegmentMetadata& segment = *m_inspection.segment;
		lines << QString("RocketForge segment %1 from build %2, stage %3.")
			.arg(QString::fromStdString(segment.segmentId).toHtmlEscaped())
			.arg(QString::fromStdString(segment.buildName).toHtmlEscaped())
			.arg(QString::fromStdString(segment.stageName).toHtmlEscaped());
	}

	const QString agentName = AgentName().toHtmlEscaped();
	if (m_inspection.files.hasInferencePolicy && m_fitState == FitState::DoesNotFit) {
		lines << QString("<b>Usable as:</b> a static opponent, in Model Benchmark and in Debug render, once %1 matches it.").arg(agentName);
	} else if (m_inspection.files.hasInferencePolicy) {
		lines << "<b>Usable as:</b> a static opponent, in Model Benchmark and in Debug render.";
	} else {
		lines << Colored("No POLICY.lt or INFERENCE_POLICY.json, so the model can't play.", ForgeGUI::Theme::NEGATIVE);
	}

	lines << TrainingStateSummary(m_inspection.files).toHtmlEscaped();

	switch (m_fitState) {
	case FitState::NotChecked:
		break;
	case FitState::Checking:
		lines << QString("Checking that it loads into %1...").arg(agentName);
		break;
	case FitState::Fits:
		lines << Colored(QString("%1 Loads into %2's networks.").arg(CHECK_MARK, agentName), ForgeGUI::Theme::POSITIVE);
		break;
	case FitState::DoesNotFit:
		lines << Colored(QString(WARNING_SIGN + " Doesn't load into %1 as it is now. You can still import it, "
			"for example if you'll change the agent to match.").arg(agentName), ForgeGUI::Theme::NEGATIVE);
		lines << m_fitProblem.toHtmlEscaped().replace("\n", "<br>");
		break;
	}

	m_summaryLabel->setText(lines.join("<br>"));
	FitHeightToContent();
}

void ImportModelDialog::StartFitCheck()
{
	const int requestId = ++m_fitRequestId;

	if (!m_inspection.IsLoadable() || !m_inspection.files.hasInferencePolicy || AgentName().isEmpty()) {
		m_fitState = FitState::NotChecked;
		UpdateSummary();
		return;
	}

	m_fitState = FitState::Checking;
	UpdateSummary();

	auto result = std::make_shared<RocketForge::ModelFitCheck>();
	const std::string agentName = AgentName().toStdString();
	const std::filesystem::path modelFolder = m_inspection.modelFolder;

	QThread* thread = QThread::create([result, agentName, modelFolder]() {
		// The first check initializes RocketSim to measure the observation size; never let two checks race that.
		static std::mutex checkMutex;
		const std::lock_guard<std::mutex> lock(checkMutex);
		*result = RocketForge::CheckModelFitsAgent(agentName, modelFolder);
	});

	connect(thread, &QThread::finished, thread, &QObject::deleteLater);
	connect(thread, &QThread::finished, this, [this, requestId, result]() {
		if (requestId != m_fitRequestId)
			return;

		m_fitState = result->fits ? FitState::Fits : FitState::DoesNotFit;
		m_fitProblem = QString::fromStdString(result->problem);
		UpdateSummary();
	});

	thread->start();
}

void ImportModelDialog::SetDetailsVisible(bool visible)
{
	m_detailsWidget->setVisible(visible);
	m_detailsToggle->setArrowType(visible ? Qt::DownArrow : Qt::RightArrow);
	FitHeightToContent();
}

void ImportModelDialog::AcceptIfValid()
{
	OnSourceChanged();
	if (DestinationFolder().empty() || !m_inspection.IsLoadable())
		return;

	if (std::filesystem::exists(DestinationFolder()) && !m_replaceCheck->isChecked()) {
		QMessageBox::information(
			this,
			DIALOG_TITLE,
			QString("%1 already has a model named \"%2\".\n\nTick \"Replace the existing model\" or choose another name.")
				.arg(AgentName(), FolderName()));
		return;
	}

	accept();
}

void ImportModelDialog::FitHeightToContent()
{
	// A word-wrapped label only reports its new height after the layout has processed the text change,
	// so measure on the next event-loop pass.
	QTimer::singleShot(0, this, [this]() {
		layout()->activate();

		const int contentHeight = std::max(layout()->totalHeightForWidth(width()), layout()->totalMinimumSize().height());
		if (contentHeight > 0)
			resize(width(), contentHeight);
	});
}

void ImportModelDialog::keyPressEvent(QKeyEvent* event)
{
	// Enter in the folder field confirms the typed path; QDialog would otherwise press Import before the summary is read.
	const bool isEnter = event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter;
	if (isEnter && m_sourceEdit->hasFocus()) {
		OnSourceChanged();
		return;
	}

	QDialog::keyPressEvent(event);
}

std::optional<bool> ImportModelDialog::ModelFitsAgent() const
{
	if (m_fitState == FitState::Fits)
		return true;
	if (m_fitState == FitState::DoesNotFit)
		return false;

	return std::nullopt;
}

QString ImportModelDialog::AgentName() const
{
	return m_agentCombo->currentText().trimmed();
}

QString ImportModelDialog::FolderName() const
{
	const QString name = m_nameEdit->text().trimmed();
	if (name.isEmpty())
		return {};

	return QString::fromStdString(SanitizeImportedModelFolderName(name.toStdString()));
}

std::filesystem::path ImportModelDialog::DestinationFolder() const
{
	if (AgentName().isEmpty() || FolderName().isEmpty())
		return {};

	const std::filesystem::path importedModelsFolder = GetAgentImportedModelsFolder(AgentName().toStdString());
	if (importedModelsFolder.empty())
		return {};

	return importedModelsFolder / FolderName().toStdString();
}
