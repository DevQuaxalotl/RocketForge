#include "ParamTestingWidget.h"

#include "ParamTestingWorker.h"
#include "../FormatUtils.h"
#include "../SimulatorUnavailableDialog.h"
#include "../ThemeUtils.h"
#include "../TimestepSpinBox.h"
#include "../Widgets/AgentComboBox.h"

#include <AgentModelPaths.h>
#include <ForgeTraining.h>
#include <StageRegistry.h>
#include <Storage/BuildStorage.h>
#include <TrainingSetup/TrainingSetupSnapshot.h>
#include <TrainingSetup/VersionManagement.h>

#include <GigaLearnInfer/AgentRegistry.h>

#include <QCheckBox>
#include <QColor>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QIcon>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPainter>
#include <QPen>
#include <QPushButton>
#include <QResizeEvent>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSize>
#include <QSpinBox>
#include <QSplitter>
#include <QStyle>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QTextEdit>
#include <QToolButton>
#include <QVariant>
#include <QVBoxLayout>

#include <algorithm>
#include <filesystem>
#include <functional>
#include <memory>
#include <set>
#include <utility>

using RocketForgeTraining::ParamTestingConfig;
using RocketForgeTraining::ParamTestingProgress;
using RocketForgeTraining::ParamTestingResult;
using RocketForgeTraining::StageParameterOverride;
using RocketForgeTraining::StageParameterOverrideKind;

namespace
{
	constexpr const char* AllBuildsLabel = "All Builds";
	constexpr const char* AllStagesLabel = "All Stages";
	constexpr int RowActionButtonSize = 24;
	constexpr int RowActionButtonSpacing = 6;
	constexpr int OverrideValueEditWidth = 150;
	constexpr const char* NeutralBorderColor = "#d8c990";
	constexpr const char* BaselineModelColor = "#73c88f";
	constexpr const char* GlobalSettingsColor = "#b9cf45";
	constexpr const char* TestSegmentsColor = "#5aa9c4";
	constexpr const char* ParamTestingBorderColorProperty = "paramTestingBorderColor";
	constexpr const char* ParamTestingAccentColorProperty = "paramTestingAccentColor";

	const QStringList SegmentGroupColors = {
		"#e2a84b",
		"#e47777",
		"#d98bbd",
		"#b78bd9",
		"#73c88f",
		"#b9cf45",
		"#62c6cc",
		"#5aa9c4"
	};

	struct ParameterChoice {
		StageParameterOverrideKind kind = StageParameterOverrideKind::Hyperparameter;
		QString group;
		QString name;
		QString currentValue;
		QString displayName;
	};

	QString OverrideKindLabel(StageParameterOverrideKind kind)
	{
		return kind == StageParameterOverrideKind::RewardWeight ? "Reward Weight" : "Hyperparameter";
	}

	QString OverrideDisplayName(const StageParameterOverride& override)
	{
		if (override.kind == StageParameterOverrideKind::RewardWeight)
			return QString::fromStdString(override.name);

		return QString::fromStdString(override.group + " / " + override.name);
	}

	QLabel* SettingsLabel(const QString& text, QWidget* parent)
	{
		auto* label = new QLabel(text, parent);
		label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
		return label;
	}

	bool SameOverride(const StageParameterOverride& a, const StageParameterOverride& b)
	{
		return a.kind == b.kind && a.group == b.group && a.name == b.name;
	}

	QTableWidgetItem* ReadOnlyItem(const QString& text)
	{
		auto* item = new QTableWidgetItem(text);
		item->setFlags(item->flags() & ~Qt::ItemIsEditable);
		return item;
	}

	QIcon MakeChevronIcon(bool up)
	{
		constexpr int iconSize = 16;
		QPixmap pixmap(iconSize, iconSize);
		pixmap.fill(Qt::transparent);

		QPainter painter(&pixmap);
		painter.setRenderHint(QPainter::Antialiasing, true);
		QPen pen(QColor("#d8d0bf"), 2.0, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
		painter.setPen(pen);

		const QPointF left(4.0, up ? 10.0 : 6.0);
		const QPointF center(8.0, up ? 6.0 : 10.0);
		const QPointF right(12.0, up ? 10.0 : 6.0);
		painter.drawLine(left, center);
		painter.drawLine(center, right);

		return QIcon(pixmap);
	}

	void AddVersionIncrementItems(QComboBox* combo)
	{
		combo->addItem("No Increment", static_cast<int>(RocketForgeTraining::VersionIncrement::None));
		combo->addItem("Next Minor", static_cast<int>(RocketForgeTraining::VersionIncrement::Minor));
		combo->addItem("Next Major", static_cast<int>(RocketForgeTraining::VersionIncrement::Major));
	}

	int VersionIncrementIndex(QComboBox* combo, RocketForgeTraining::VersionIncrement increment)
	{
		return combo->findData(static_cast<int>(increment));
	}

	bool RevisionAfterOrEqual(const GGL::SegmentVersionNumber& a, const GGL::SegmentVersionNumber& b)
	{
		return a.major > b.major || (a.major == b.major && a.minor >= b.minor);
	}

	QString SegmentRevisionLabel(const GGL::SegmentVersionNumber& version)
	{
		return QString::fromStdString(RocketForgeTraining::BuildStorage::MakeSegmentId(version.major, version.minor));
	}

	std::optional<GGL::SegmentVersionNumber> LatestRevision(const std::vector<GGL::SegmentVersionNumber>& revisions)
	{
		std::optional<GGL::SegmentVersionNumber> latest;
		for (const GGL::SegmentVersionNumber& revision : revisions) {
			if (!latest.has_value() || RevisionAfterOrEqual(revision, latest.value()))
				latest = revision;
		}
		return latest;
	}

	std::optional<GGL::SegmentVersionNumber> LatestRevisionInMajor(
		const std::vector<GGL::SegmentVersionNumber>& revisions,
		int major)
	{
		std::optional<GGL::SegmentVersionNumber> latest;
		for (const GGL::SegmentVersionNumber& revision : revisions) {
			if (revision.major != major)
				continue;
			if (!latest.has_value() || RevisionAfterOrEqual(revision, latest.value()))
				latest = revision;
		}
		return latest;
	}

	class InlineStepSpinBox : public QSpinBox
	{
	public:
		explicit InlineStepSpinBox(QWidget* parent = nullptr)
			: QSpinBox(parent)
		{
			setButtonSymbols(QAbstractSpinBox::NoButtons);
			lineEdit()->setTextMargins(0, 0, (StepButtonSize * 2) + StepButtonGap + StepButtonTextMargin, 0);

			m_stepUpButton = new QToolButton(this);
			m_stepUpButton->setObjectName("targetTimestepsStepUpButton");
			m_stepUpButton->setFixedSize(StepButtonSize, StepButtonSize);
			m_stepUpButton->setIcon(MakeChevronIcon(true));
			m_stepUpButton->setIconSize(QSize(15, 15));
			m_stepUpButton->setAutoRepeat(true);
			m_stepUpButton->setFocusPolicy(Qt::NoFocus);
			m_stepUpButton->setToolTip("Increase value");
			connect(m_stepUpButton, &QToolButton::clicked, this, [this]() { stepBy(1); });

			m_stepDownButton = new QToolButton(this);
			m_stepDownButton->setObjectName("targetTimestepsStepDownButton");
			m_stepDownButton->setFixedSize(StepButtonSize, StepButtonSize);
			m_stepDownButton->setIcon(MakeChevronIcon(false));
			m_stepDownButton->setIconSize(QSize(15, 15));
			m_stepDownButton->setAutoRepeat(true);
			m_stepDownButton->setFocusPolicy(Qt::NoFocus);
			m_stepDownButton->setToolTip("Decrease value");
			connect(m_stepDownButton, &QToolButton::clicked, this, [this]() { stepBy(-1); });
		}

	protected:
		void resizeEvent(QResizeEvent* event) override
		{
			QSpinBox::resizeEvent(event);

			const int y = (height() - StepButtonSize) / 2;
			const int downX = width() - StepButtonRightMargin - StepButtonSize;
			const int upX = downX - StepButtonGap - StepButtonSize;

			if (m_stepUpButton)
				m_stepUpButton->setGeometry(upX, y, StepButtonSize, StepButtonSize);
			if (m_stepDownButton)
				m_stepDownButton->setGeometry(downX, y, StepButtonSize, StepButtonSize);
		}

	private:
		static constexpr int StepButtonSize = 22;
		static constexpr int StepButtonGap = 2;
		static constexpr int StepButtonRightMargin = 3;
		static constexpr int StepButtonTextMargin = 18;

		QToolButton* m_stepUpButton = nullptr;
		QToolButton* m_stepDownButton = nullptr;
	};

	QString ParamTestingGroupStyle(const QString& borderColor, const QString& accentColor)
	{
		return QString(
			"QGroupBox { border: 1px solid %1; border-radius: 4px; margin-top: 8px; }"
			"QGroupBox::title { color: %2; subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
			"QGroupBox QComboBox, QGroupBox QComboBox:disabled, QGroupBox QLineEdit, QGroupBox QLineEdit:disabled, QGroupBox QTextEdit, QGroupBox QTextEdit:disabled, QGroupBox QAbstractSpinBox, QGroupBox QAbstractSpinBox:disabled, QGroupBox QTableWidget { color: %2; font-weight: 600; }"
			"QGroupBox QComboBox QAbstractItemView { selection-color: %2; }"
			"QGroupBox QTableWidget::item:selected { color: %2; }"
			"QGroupBox QCheckBox:checked, QGroupBox QCheckBox:checked:disabled { color: %2; font-weight: 600; }"
			"QGroupBox QCheckBox::indicator:checked, QGroupBox QCheckBox::indicator:checked:disabled { background-color: %2; border-color: %2; image: url(:/icons/check-dark.svg); }"
			"QGroupBox QPushButton:hover, QGroupBox QToolButton:hover, QGroupBox QComboBox:hover, QGroupBox QLineEdit:hover, QGroupBox QAbstractSpinBox:hover { border-color: %2; }"
			"QGroupBox QCheckBox::indicator:hover:!checked { border-color: %2; }")
			.arg(borderColor, accentColor);
	}

	void SetParamTestingGroupStyle(
		QGroupBox* group,
		const QString& borderColor,
		const QString& accentColor)
	{
		group->setProperty(ParamTestingBorderColorProperty, borderColor);
		group->setProperty(ParamTestingAccentColorProperty, accentColor);
		group->setStyleSheet(ForgeGUI::ApplyColorPreference(
			ParamTestingGroupStyle(borderColor, accentColor),
			ForgeGUI::ApplicationColorsEnabled()));
	}

	bool IsSupportedHyperparameter(const QString& group, const QString& name)
	{
		if (group == "Environment")
			return name == "Games" || name == "Tick Skip" || name == "Action Delay";

		if (group == "Training") {
			static const std::set<QString> supported = {
				"Timesteps / Iteration",
				"Batch Size",
				"Micro Batch Size",
				"Epochs",
				"Random Seed",
				"Learning Rate",
				"Policy LR",
				"Critic LR",
				"Entropy Scale",
				"GAE Gamma",
				"GAE Lambda",
				"Clip Range",
				"Normalize Advantages",
				"Policy Grad Clip Norm",
				"Critic Head Loss Weighting",
				"Reward Clip Range",
				"Policy Temperature",
				"Min Action Probability",
				"Mask Entropy",
				"Deterministic",
				"Use Half Precision",
				"Overbatching",
				"Max Episode Duration",
			};
			return supported.contains(name);
		}

		if (group == "Schedules") {
			return name == "PPO/Entropy Scale Target" ||
				name == "PPO/Entropy Scale Amplitude" ||
				name == "PPO/Entropy Scale Period Timesteps" ||
				name == "PPO/Entropy Scale Start Timestep";
		}

		if (group == "PPG") {
			static const std::set<QString> supported = {
				"Policy Epochs",
				"Critic Epochs",
				"Aux Phase Interval",
				"Aux Epochs",
				"Aux Batch Size",
				"Aux Policy LR",
				"Aux Critic LR",
				"Aux Value Loss Scale",
				"Aux Policy Clone Loss Scale",
				"Aux True Critic Loss Scale",
				"Aux Grad Clip",
			};
			return supported.contains(name);
		}

		if (group == "Saving") {
			static const std::set<QString> supported = {
				"Timesteps / Save",
				"Timesteps / Version",
				"Checkpoints To Keep",
				"Save Policy Versions",
				"Max Recent Versions",
				"Train Against Recent Versions",
				"Train Against Recent Percent",
				"Recent Opponent Cohort Size",
				"Recent Opponent Rotation Timesteps",
				"Train Against Static Opponents",
				"Train Against Static Percent",
				"Static Opponent Cohort Size",
				"Static Opponent Rotation Timesteps",
				"Warm Start Policy",
				"Warm Start Iterations",
				"Warm Start Policy LR Multiplier",
			};
			return supported.contains(name);
		}

		if (group == "Normalization") {
			static const std::set<QString> supported = {
				"Standardize Obs",
				"Standardize Returns",
				"Min Obs STD",
				"Max Obs Mean Range",
				"Max Obs Samples",
				"Max Return Samples",
			};
			return supported.contains(name);
		}

		return false;
	}

	StageParameterOverride MakeOverride(const ParameterChoice& choice, const QString& value)
	{
		StageParameterOverride override = {};
		override.kind = choice.kind;
		override.group = choice.group.toStdString();
		override.name = choice.name.toStdString();
		override.value = value.trimmed().toStdString();
		return override;
	}

	std::vector<ParameterChoice> BuildParameterChoices(const QString& agentName, const QString& stageName)
	{
		RocketForgeTraining::StageSettings settings = {};
		settings.stageName = stageName.toStdString();

		std::unique_ptr<RocketForge::TrainingStage> stage =
			RocketForgeTraining::CreateTrainingStage(agentName.toStdString(), stageName.toStdString(), {}, settings);
		if (!stage)
			return {};

		RocketForge::TrainingSetupSnapshot snapshot = RocketForge::MakeTrainingSetupSnapshot(stage.get());

		std::vector<ParameterChoice> choices;
		for (const RocketForge::RewardTableEntry& reward : snapshot.rewards) {
			ParameterChoice choice = {};
			choice.kind = StageParameterOverrideKind::RewardWeight;
			choice.name = QString::fromStdString(reward.name);
			choice.currentValue = QString::number(reward.weight, 'g', 7);
			choice.displayName = QString("%1 (current %2)").arg(choice.name, choice.currentValue);
			choices.push_back(choice);
		}

		for (const RocketForge::TrainingSetupParameter& parameter : snapshot.hyperparameters) {
			const QString group = QString::fromStdString(parameter.group);
			const QString name = QString::fromStdString(parameter.name);
			if (!RocketForge::TrainingSetupParameterAffectsIdentity(parameter))
				continue;
			if (!IsSupportedHyperparameter(group, name))
				continue;

			ParameterChoice choice = {};
			choice.kind = StageParameterOverrideKind::Hyperparameter;
			choice.group = group;
			choice.name = name;
			choice.currentValue = QString::fromStdString(parameter.value);
			choice.displayName = QString("%1 / %2 (current %3)").arg(group, name, choice.currentValue);
			choices.push_back(choice);
		}

		std::sort(choices.begin(), choices.end(), [](const ParameterChoice& a, const ParameterChoice& b) {
			if (a.kind != b.kind)
				return a.kind == StageParameterOverrideKind::RewardWeight;
			return a.displayName < b.displayName;
		});
		return choices;
	}
}

ParamTestingWidget::ParamTestingWidget(QWidget* parent)
	: QWidget(parent)
{
	qRegisterMetaType<ParamTestingConfig>("RocketForgeTraining::ParamTestingConfig");
	qRegisterMetaType<ParamTestingProgress>("RocketForgeTraining::ParamTestingProgress");
	qRegisterMetaType<ParamTestingResult>("RocketForgeTraining::ParamTestingResult");

	auto* root = new QVBoxLayout(this);
	auto* splitter = new QSplitter(Qt::Horizontal, this);
	root->addWidget(splitter, 1);

	auto* leftPane = new QWidget(splitter);
	auto* leftLayout = new QVBoxLayout(leftPane);
	leftLayout->setContentsMargins(0, 0, 6, 0);

	auto* rightPane = new QWidget(splitter);
	auto* rightLayout = new QVBoxLayout(rightPane);
	rightLayout->setContentsMargins(6, 0, 0, 0);

	splitter->addWidget(leftPane);
	splitter->addWidget(rightPane);
	splitter->setStretchFactor(0, 1);
	splitter->setStretchFactor(1, 1);
	splitter->setSizes({ 600, 700 });

	auto* baselineGroup = new QGroupBox("Baseline Model", this);
	SetParamTestingGroupStyle(baselineGroup, NeutralBorderColor, BaselineModelColor);
	auto* baselineLayout = new QGridLayout(baselineGroup);

	m_agentCombo = new AgentComboBox(baselineGroup);
	m_buildCombo = new QComboBox(baselineGroup);
	m_baselineStageCombo = new QComboBox(baselineGroup);
	m_baselineVersionCombo = new QComboBox(baselineGroup);

	baselineLayout->addWidget(new QLabel("Agent Filter:", baselineGroup), 0, 0);
	baselineLayout->addWidget(m_agentCombo, 0, 1);
	baselineLayout->addWidget(new QLabel("Build Filter:", baselineGroup), 0, 2);
	baselineLayout->addWidget(m_buildCombo, 0, 3);
	baselineLayout->addWidget(new QLabel("Stage Filter:", baselineGroup), 0, 4);
	baselineLayout->addWidget(m_baselineStageCombo, 0, 5);
	baselineLayout->addWidget(new QLabel("Baseline Model:", baselineGroup), 1, 0);
	baselineLayout->addWidget(m_baselineVersionCombo, 1, 1, 1, 5);
	baselineLayout->setColumnStretch(1, 1);
	baselineLayout->setColumnStretch(3, 1);
	baselineLayout->setColumnStretch(5, 1);
	leftLayout->addWidget(baselineGroup);

	auto* settingsGroup = new QGroupBox("Global Settings", this);
	SetParamTestingGroupStyle(settingsGroup, NeutralBorderColor, GlobalSettingsColor);
	auto* settingsLayout = new QGridLayout(settingsGroup);

	m_segmentTimestepsSpin = new TimestepSpinBox(settingsGroup);
	m_segmentTimestepsSpin->setDecimals(3);
	m_segmentTimestepsSpin->setRange(0.001, 1'000'000.0);
	m_segmentTimestepsSpin->setValue(250.0);
	m_segmentTimestepsSpin->setMinimumWidth(180);

	m_branchEdit = new QLineEdit("param-test", settingsGroup);

	m_metricsCheck = new QCheckBox("Metrics", settingsGroup);
	m_metricsCheck->setChecked(true);

	m_skillTestingCheck = new QCheckBox("Enable Skill Testing during Training", settingsGroup);
	m_skillTestingCheck->setChecked(false);

	m_runBenchmarkCheck = new QCheckBox("Run Model Benchmark After All Segments", settingsGroup);
	m_runBenchmarkCheck->setChecked(true);
	m_benchmarkGamesSpin = new InlineStepSpinBox(settingsGroup);
	m_benchmarkGamesSpin->setRange(1, 100'000);
	m_benchmarkGamesSpin->setValue(1000);
	m_benchmarkGamesSpin->setMinimumWidth(180);

	auto* trainingOptions = new QWidget(settingsGroup);
	auto* trainingOptionsLayout = new QHBoxLayout(trainingOptions);
	trainingOptionsLayout->setContentsMargins(0, 0, 0, 0);
	trainingOptionsLayout->setSpacing(12);
	trainingOptionsLayout->addWidget(m_metricsCheck);
	trainingOptionsLayout->addWidget(m_skillTestingCheck);
	trainingOptionsLayout->addStretch();

	auto* benchmarkOptions = new QWidget(settingsGroup);
	auto* benchmarkOptionsLayout = new QHBoxLayout(benchmarkOptions);
	benchmarkOptionsLayout->setContentsMargins(0, 0, 0, 0);
	benchmarkOptionsLayout->setSpacing(10);
	benchmarkOptionsLayout->addWidget(m_runBenchmarkCheck);
	benchmarkOptionsLayout->addSpacing(12);
	benchmarkOptionsLayout->addWidget(new QLabel("Benchmark Games per Mode:", settingsGroup));
	benchmarkOptionsLayout->addWidget(m_benchmarkGamesSpin);
	benchmarkOptionsLayout->addStretch();

	settingsLayout->setHorizontalSpacing(12);
	settingsLayout->setVerticalSpacing(8);
	settingsLayout->addWidget(SettingsLabel("Branch Name:", settingsGroup), 0, 0);
	settingsLayout->addWidget(m_branchEdit, 0, 1);
	settingsLayout->addWidget(SettingsLabel("Training Options:", settingsGroup), 1, 0);
	settingsLayout->addWidget(trainingOptions, 1, 1);
	settingsLayout->addWidget(SettingsLabel("Timesteps per Segment:", settingsGroup), 2, 0);
	settingsLayout->addWidget(m_segmentTimestepsSpin, 2, 1);
	settingsLayout->addWidget(SettingsLabel("Benchmarking:", settingsGroup), 3, 0);
	settingsLayout->addWidget(benchmarkOptions, 3, 1);
	settingsLayout->setColumnStretch(1, 1);
	leftLayout->addWidget(settingsGroup);

	auto* segmentsGroup = new QGroupBox("Test Segments", this);
	SetParamTestingGroupStyle(segmentsGroup, NeutralBorderColor, TestSegmentsColor);
	auto* segmentsLayout = new QVBoxLayout(segmentsGroup);

	m_addSegmentButton = new QPushButton("Add Segment", segmentsGroup);

	auto* segmentButtons = new QHBoxLayout();
	segmentButtons->addWidget(m_addSegmentButton);
	segmentButtons->addStretch();
	segmentsLayout->addLayout(segmentButtons);

	m_segmentScrollArea = new QScrollArea(segmentsGroup);
	m_segmentScrollArea->setWidgetResizable(true);

	m_segmentListWidget = new QWidget(m_segmentScrollArea);
	m_segmentListLayout = new QVBoxLayout(m_segmentListWidget);
	m_segmentListLayout->setContentsMargins(0, 0, 0, 0);
	m_segmentListLayout->setSpacing(8);
	m_segmentScrollArea->setWidget(m_segmentListWidget);
	segmentsLayout->addWidget(m_segmentScrollArea, 1);
	rightLayout->addWidget(segmentsGroup, 1);

	auto* runLayout = new QHBoxLayout();
	m_runButton = new QPushButton("Run Param Test", this);
	m_cancelButton = new QPushButton("Cancel", this);
	m_cancelButton->setEnabled(false);
	runLayout->addWidget(m_runButton);
	runLayout->addWidget(m_cancelButton);
	runLayout->addStretch();
	leftLayout->addLayout(runLayout);

	m_progressPanel = new RunProgressPanel(this);
	m_progressPanel->SetConsoleVisible(true);
	leftLayout->addWidget(m_progressPanel, 1);

	connect(m_agentCombo, &AgentComboBox::AgentChanged, this, &ParamTestingWidget::OnAgentChanged);
	connect(m_buildCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, &ParamTestingWidget::OnBuildFilterChanged);
	connect(m_baselineStageCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, &ParamTestingWidget::OnStageFilterChanged);
	connect(m_baselineVersionCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, &ParamTestingWidget::UpdateInputState);
	connect(m_segmentTimestepsSpin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &ParamTestingWidget::UpdateInputState);
	connect(m_branchEdit, &QLineEdit::textChanged, this, &ParamTestingWidget::UpdateInputState);
	connect(m_metricsCheck, &QCheckBox::toggled, this, &ParamTestingWidget::UpdateInputState);
	connect(m_skillTestingCheck, &QCheckBox::toggled, this, &ParamTestingWidget::UpdateInputState);
	connect(m_benchmarkGamesSpin, qOverload<int>(&QSpinBox::valueChanged), this, &ParamTestingWidget::UpdateInputState);
	connect(m_addSegmentButton, &QPushButton::clicked, this, &ParamTestingWidget::OnAddSegmentClicked);
	connect(m_runButton, &QPushButton::clicked, this, &ParamTestingWidget::OnRunClicked);
	connect(m_cancelButton, &QPushButton::clicked, this, &ParamTestingWidget::OnCancelClicked);
	connect(m_runBenchmarkCheck, &QCheckBox::toggled, this, &ParamTestingWidget::UpdateInputState);

	connect(&ModelCatalog::Instance(), &ModelCatalog::Changed, this, [this](const QString& agent) {
		if (agent == CurrentAgentName() && !m_running) {
			RefreshBuildSelector();
			RefreshSegmentList();
		}
	});
	connect(&ModelCatalog::Instance(), &ModelCatalog::Failed, this, [this](const QString& agent, const QString& error) {
		if (agent == CurrentAgentName()) m_baselineVersionCombo->setToolTip("Model refresh failed: " + error);
	});

	RefreshBuildSelector();
	RefreshSegmentList();
	UpdateInputState();
}

ParamTestingWidget::~ParamTestingWidget()
{
	// Cancel cooperatively before joining: RunParamTesting() blocks, so a bare quit()/wait()
	// would stall closing the tab mid-run until the job finished on its own.
	if (m_worker)
		m_worker->Cancel();
	m_workerHost.Stop();
}

bool ParamTestingWidget::IsRunning() const
{
	return m_running;
}

void ParamTestingWidget::SetExternalRunActive(bool active)
{
	m_externalRunActive = active;
	UpdateInputState();
}

void ParamTestingWidget::SetColorsEnabled(bool enabled)
{
	const QList<QGroupBox*> groups = findChildren<QGroupBox*>();
	for (QGroupBox* group : groups) {
		const QVariant borderColor = group->property(ParamTestingBorderColorProperty);
		const QVariant accentColor = group->property(ParamTestingAccentColorProperty);
		if (!borderColor.isValid() || !accentColor.isValid())
			continue;

		group->setStyleSheet(ForgeGUI::ApplyColorPreference(
			ParamTestingGroupStyle(borderColor.toString(), accentColor.toString()),
			enabled));
	}
}

QString ParamTestingWidget::CurrentAgentName() const
{
	return m_agentCombo ? m_agentCombo->CurrentAgentName() : QString();
}

QString ParamTestingWidget::CurrentBuildFilterName() const
{
	if (!m_buildCombo)
		return {};

	const QString buildName = m_buildCombo->currentText().trimmed();
	return buildName == AllBuildsLabel ? QString() : buildName;
}

QString ParamTestingWidget::CurrentStageFilterName() const
{
	if (!m_baselineStageCombo)
		return {};

	const QString stageName = m_baselineStageCombo->currentText().trimmed();
	return stageName == AllStagesLabel ? QString() : stageName;
}

QString ParamTestingWidget::CurrentBaselineBuildName() const
{
	if (!m_baselineVersionCombo)
		return {};

	const QStringList data = m_baselineVersionCombo->currentData().toStringList();
	return data.size() >= 1 ? data[0].trimmed() : QString();
}

QString ParamTestingWidget::CurrentBaselineStageName() const
{
	if (!m_baselineVersionCombo)
		return {};

	const QStringList data = m_baselineVersionCombo->currentData().toStringList();
	return data.size() >= 2 ? data[1].trimmed() : QString();
}

QString ParamTestingWidget::CurrentBaselineSegmentId() const
{
	if (!m_baselineVersionCombo)
		return {};

	const QStringList data = m_baselineVersionCombo->currentData().toStringList();
	return data.size() >= 3 ? data[2].trimmed() : QString();
}

void ParamTestingWidget::FollowTrainingAgent(const QString& agentName)
{
	m_agentCombo->FollowAgent(agentName);
}

void ParamTestingWidget::OnAgentChanged(const QString&)
{
	// Selections belong to an agent, even when another agent has identical labels.
	{ QSignalBlocker blocker(m_buildCombo); m_buildCombo->clear(); }
	{ QSignalBlocker blocker(m_baselineStageCombo); m_baselineStageCombo->clear(); }
	{ QSignalBlocker blocker(m_baselineVersionCombo); m_baselineVersionCombo->clear(); }
	RefreshBuildSelector();
}

void ParamTestingWidget::OnBuildFilterChanged()
{
	RefreshBaselineStageSelector();
}

void ParamTestingWidget::OnStageFilterChanged()
{
	RefreshBaselineVersionSelector();
}

void ParamTestingWidget::RefreshBuildSelector()
{
	if (!m_buildCombo)
		return;

	m_catalog = ModelCatalog::Instance().Get(CurrentAgentName());
	m_baselineVersionCombo->setToolTip({});
	const QString previous = m_buildCombo->currentText().trimmed();
	QSignalBlocker blocker(m_buildCombo);
	m_buildCombo->clear();
	m_buildCombo->addItem(AllBuildsLabel);

	for (const RocketForgeTraining::BuildSummary& build : m_catalog->storage.builds) {
		const std::string buildName = build.buildName.empty() ? build.folderName : build.buildName;
		if (!buildName.empty())
			m_buildCombo->addItem(QString::fromStdString(buildName));
	}

	const int previousIndex = m_buildCombo->findText(previous);
	m_buildCombo->setCurrentIndex(previousIndex >= 0 ? previousIndex : 0);
	RefreshBaselineStageSelector();
}

void ParamTestingWidget::RefreshBaselineStageSelector()
{
	if (!m_baselineStageCombo)
		return;

	const QString previous = m_baselineStageCombo->currentText().trimmed();
	QSignalBlocker blocker(m_baselineStageCombo);
	m_baselineStageCombo->clear();
	m_baselineStageCombo->addItem(AllStagesLabel);

	const std::vector<std::string> stages = m_catalog->TrainingStages(CurrentBuildFilterName().toStdString());
	for (const std::string& stage : stages)
		m_baselineStageCombo->addItem(QString::fromStdString(stage));

	const int previousIndex = m_baselineStageCombo->findText(previous);
	if (previousIndex >= 0)
		m_baselineStageCombo->setCurrentIndex(previousIndex);

	RefreshBaselineVersionSelector();
}

void ParamTestingWidget::RefreshBaselineVersionSelector()
{
	if (!m_baselineVersionCombo)
		return;

	const QStringList previous = m_baselineVersionCombo->currentData().toStringList();
	QSignalBlocker blocker(m_baselineVersionCombo);
	m_baselineVersionCombo->clear();

	const bool includeInactive = false;
	std::vector<GGL::SegmentVersionSummary> versions = m_catalog->Versions(
		CurrentBuildFilterName().toStdString(), CurrentStageFilterName().toStdString(), includeInactive);

	std::sort(versions.begin(), versions.end(), [](const GGL::SegmentVersionSummary& a, const GGL::SegmentVersionSummary& b) {
		if (a.updatedAt != b.updatedAt)
			return a.updatedAt > b.updatedAt;
		if (a.latestTimesteps != b.latestTimesteps)
			return a.latestTimesteps > b.latestTimesteps;
		return a.id > b.id;
	});

	for (const GGL::SegmentVersionSummary& version : versions) {
		if (version.modelCount <= 0)
			continue;

		QString buildName = QString::fromStdString(version.buildName);
		if (buildName.isEmpty())
			buildName = CurrentBuildFilterName();
		if (buildName.isEmpty()) {
			const std::filesystem::path buildRoot = std::filesystem::path(version.rootPath).parent_path().parent_path();
			if (!buildRoot.empty())
				buildName = QString::fromStdString(buildRoot.filename().string());
		}
		if (buildName.isEmpty())
			continue;

		const QString stageName = QString::fromStdString(version.stageName);
		const QString versionId = QString::fromStdString(version.id);
		const QString label = QString("%1 / %2 / %3 (%4)")
			.arg(buildName, stageName, versionId, ForgeGUI::FormatTimesteps(version.latestTimesteps));

		m_baselineVersionCombo->addItem(label, QStringList{ buildName, stageName, versionId });
	}

	const int previousIndex = m_baselineVersionCombo->findData(previous);
	if (previousIndex >= 0)
		m_baselineVersionCombo->setCurrentIndex(previousIndex);

	UpdateInputState();
}

QStringList ParamTestingWidget::CompatibleStageNames() const
{
	std::unique_ptr<Agent> agent = AgentRegistry::Create(CurrentAgentName().toStdString());
	std::vector<std::string> stages = agent
		? RocketForge::StageRegistry::GetCompatibleStages(*agent)
		: RocketForge::StageRegistry::GetAvailableStages();

	QStringList result;
	for (const std::string& stage : stages)
		result << QString::fromStdString(stage);
	return result;
}

void ParamTestingWidget::OnAddSegmentClicked()
{
	SegmentRow row = {};

	const QStringList stages = CompatibleStageNames();
	row.stageName = stages.contains(CurrentBaselineStageName()) ? CurrentBaselineStageName() : (stages.isEmpty() ? QString() : stages.front());

	m_segments.push_back(std::move(row));
	RefreshSegmentList();
	UpdateInputState();
}

QString ParamTestingWidget::ModsSummary(const SegmentRow& row) const
{
	if (row.overrides.empty())
		return "0 Mods";

	return QString("%1 Mod%2")
		.arg(static_cast<int>(row.overrides.size()))
		.arg(row.overrides.size() == 1 ? "" : "s");
}

QString ParamTestingWidget::PlannedVersionText(int segmentIndex) const
{
	if (segmentIndex < 0 || segmentIndex >= static_cast<int>(m_segments.size()))
		return "Version: ?";

	std::vector<GGL::SegmentVersionNumber> revisions;
	if (m_catalog) {
		const auto found = m_catalog->storage.segmentsByBuild.find(CurrentBaselineBuildName().toStdString());
		if (found != m_catalog->storage.segmentsByBuild.end()) {
			for (const RocketForgeTraining::TrainingSegmentMetadata& segment : found->second) {
				const std::optional<GGL::SegmentVersionNumber> parsed =
					RocketForgeTraining::BuildStorage::TryParseSegmentRevision(segment.segmentId);
				if (parsed) revisions.push_back(*parsed);
			}
		}
	}

	std::optional<GGL::SegmentVersionNumber> parent =
		RocketForgeTraining::BuildStorage::TryParseSegmentRevision(CurrentBaselineSegmentId().toStdString());

	for (int i = 0; i <= segmentIndex; i++) {
		const SegmentRow& segment = m_segments[i];
		GGL::SegmentVersionNumber planned = {};

		if (segment.versionOverride.has_value()) {
			planned = segment.versionOverride.value();
		} else if (segment.versionIncrement == RocketForgeTraining::VersionIncrement::Major) {
			std::optional<GGL::SegmentVersionNumber> latest = LatestRevision(revisions);
			planned = latest.has_value()
				? GGL::SegmentVersionNumber{ latest->major + 1, 0 }
				: GGL::SegmentVersionNumber{ 1, 0 };
		} else {
			std::optional<GGL::SegmentVersionNumber> base;
			if (parent.has_value()) {
				std::optional<GGL::SegmentVersionNumber> latestInParentMajor =
					LatestRevisionInMajor(revisions, parent->major);
				if (latestInParentMajor.has_value() &&
					RevisionAfterOrEqual(latestInParentMajor.value(), parent.value())) {
					base = latestInParentMajor;
				} else {
					base = parent;
				}
			} else {
				base = LatestRevision(revisions);
			}

			planned = base.has_value()
				? GGL::SegmentVersionNumber{ base->major, base->minor + 1 }
				: GGL::SegmentVersionNumber{ 1, 0 };
		}

		if (i == segmentIndex)
			return QString("Will create: %1").arg(SegmentRevisionLabel(planned));

		revisions.push_back(planned);
	}

	return "Version: ?";
}

void ParamTestingWidget::RefreshSegmentList()
{
	if (!m_segmentListLayout)
		return;

	m_refreshingSegments = true;

	while (QLayoutItem* item = m_segmentListLayout->takeAt(0)) {
		if (QWidget* widget = item->widget())
			widget->deleteLater();
		delete item;
	}

	const QStringList stages = CompatibleStageNames();
	for (int row = 0; row < static_cast<int>(m_segments.size()); row++) {
		SegmentRow& segment = m_segments[row];
		if (segment.stageName.isEmpty() && !stages.isEmpty())
			segment.stageName = stages.front();

		auto* segmentGroup = new QGroupBox(QString("Segment %1").arg(row + 1), m_segmentListWidget);
		const QString segmentColor = SegmentGroupColors[row % SegmentGroupColors.size()];
		SetParamTestingGroupStyle(segmentGroup, segmentColor, segmentColor);
		auto* segmentLayout = new QVBoxLayout(segmentGroup);
		auto* rowLayout = new QGridLayout();

		auto* stageCombo = new QComboBox(segmentGroup);
		stageCombo->addItems(stages);

		const int stageIndex = stageCombo->findText(segment.stageName);
		if (stageIndex >= 0)
			stageCombo->setCurrentIndex(stageIndex);

		auto* versionPreviewLabel = new QLabel(PlannedVersionText(row), segmentGroup);
		versionPreviewLabel->setToolTip(versionPreviewLabel->text());
		auto* modsSummaryLabel = new QLabel(ModsSummary(segment), segmentGroup);
		modsSummaryLabel->setMinimumWidth(modsSummaryLabel->fontMetrics().horizontalAdvance("999 Mods"));
		modsSummaryLabel->setToolTip("Number of parameter overrides applied to this segment");
		auto* detailsButton = new QToolButton(segmentGroup);
		auto* moveUpButton = new QPushButton(segmentGroup);
		auto* moveDownButton = new QPushButton(segmentGroup);
		auto* removeButton = new QPushButton("X", segmentGroup);

		detailsButton->setObjectName("rowSettingsButton");
		detailsButton->setIcon(style()->standardIcon(QStyle::SP_FileDialogDetailedView));
		detailsButton->setIconSize(QSize(14, 14));
		detailsButton->setFixedSize(RowActionButtonSize, RowActionButtonSize);
		detailsButton->setCheckable(true);
		detailsButton->setChecked(segment.detailsExpanded);
		detailsButton->setToolTip(segment.detailsExpanded ? "Hide segment details" : "Show segment details");

		moveUpButton->setObjectName("rowMoveUpButton");
		moveUpButton->setIcon(MakeChevronIcon(true));
		moveUpButton->setIconSize(QSize(16, 16));
		moveUpButton->setFixedSize(RowActionButtonSize, RowActionButtonSize);
		moveUpButton->setToolTip("Move up");

		moveDownButton->setObjectName("rowMoveDownButton");
		moveDownButton->setIcon(MakeChevronIcon(false));
		moveDownButton->setIconSize(QSize(16, 16));
		moveDownButton->setFixedSize(RowActionButtonSize, RowActionButtonSize);
		moveDownButton->setToolTip("Move down");

		removeButton->setObjectName("rowDeleteButton");
		removeButton->setFixedSize(RowActionButtonSize, RowActionButtonSize);
		removeButton->setToolTip("Remove segment");

		moveUpButton->setEnabled(row > 0);
		moveDownButton->setEnabled(row + 1 < static_cast<int>(m_segments.size()));

		auto* actionButtonsWidget = new QWidget(segmentGroup);
		auto* actionButtonsLayout = new QHBoxLayout(actionButtonsWidget);
		actionButtonsLayout->setContentsMargins(0, 0, 0, 0);
		actionButtonsLayout->setSpacing(RowActionButtonSpacing);
		actionButtonsLayout->addWidget(detailsButton);
		actionButtonsLayout->addWidget(moveUpButton);
		actionButtonsLayout->addWidget(moveDownButton);
		actionButtonsLayout->addWidget(removeButton);
		actionButtonsWidget->setFixedWidth(RowActionButtonSize * 4 + RowActionButtonSpacing * 3);

		auto* modsLabel = new QLabel("Mods:", segmentGroup);
		modsLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);

		rowLayout->addWidget(new QLabel("Training Stage:", segmentGroup), 0, 0);
		rowLayout->addWidget(stageCombo, 0, 1);
		rowLayout->addWidget(actionButtonsWidget, 0, 2, 1, 2, Qt::AlignRight | Qt::AlignVCenter);
		rowLayout->addWidget(new QLabel("Version:", segmentGroup), 1, 0);
		rowLayout->addWidget(versionPreviewLabel, 1, 1);
		rowLayout->addWidget(modsLabel, 1, 2);
		rowLayout->addWidget(modsSummaryLabel, 1, 3);
		rowLayout->setColumnStretch(1, 1);
		segmentLayout->addLayout(rowLayout);

		QComboBox* versionCombo = nullptr;
		QCheckBox* overrideVersionCheck = nullptr;
		QSpinBox* overrideMajorSpin = nullptr;
		QSpinBox* overrideMinorSpin = nullptr;
		QTextEdit* notesEdit = nullptr;
		QPushButton* removeModButton = nullptr;
		QTableWidget* modsTable = nullptr;

		if (segment.detailsExpanded) {
			auto* detailsWidget = new QWidget(segmentGroup);
			auto* detailsLayout = new QVBoxLayout(detailsWidget);
			detailsLayout->setContentsMargins(14, 4, 0, 0);

			auto* versionGroup = new QGroupBox("Segment Revision and Notes", detailsWidget);
			auto* versionLayout = new QGridLayout(versionGroup);
			versionCombo = new QComboBox(versionGroup);
			overrideVersionCheck = new QCheckBox("Override", versionGroup);
			overrideMajorSpin = new QSpinBox(versionGroup);
			overrideMinorSpin = new QSpinBox(versionGroup);
			notesEdit = new QTextEdit(versionGroup);

			AddVersionIncrementItems(versionCombo);
			const int incrementIndex = VersionIncrementIndex(versionCombo, segment.versionIncrement);
			versionCombo->setCurrentIndex(incrementIndex >= 0 ? incrementIndex : 1);

			overrideMajorSpin->setRange(0, 999);
			overrideMinorSpin->setRange(0, 99);
			if (segment.versionOverride.has_value()) {
				overrideVersionCheck->setChecked(true);
				overrideMajorSpin->setValue(segment.versionOverride->major);
				overrideMinorSpin->setValue(segment.versionOverride->minor);
			} else {
				overrideMajorSpin->setValue(1);
				overrideMinorSpin->setValue(0);
			}
			overrideMajorSpin->setEnabled(overrideVersionCheck->isChecked());
			overrideMinorSpin->setEnabled(overrideVersionCheck->isChecked());

			notesEdit->setFixedHeight(56);
			notesEdit->setPlaceholderText("Optional notes for this segment...");
			notesEdit->setPlainText(segment.versionNotes);

			auto* overrideLayout = new QHBoxLayout();
			overrideLayout->setContentsMargins(0, 0, 0, 0);
			overrideLayout->addWidget(overrideVersionCheck);
			overrideLayout->addWidget(new QLabel("Major:", versionGroup));
			overrideLayout->addWidget(overrideMajorSpin);
			overrideLayout->addWidget(new QLabel("Minor:", versionGroup));
			overrideLayout->addWidget(overrideMinorSpin);

			versionLayout->addWidget(new QLabel("Revision:", versionGroup), 0, 0);
			versionLayout->addWidget(versionCombo, 0, 1);
			versionLayout->addWidget(new QLabel("Override Revision:", versionGroup), 0, 2);
			versionLayout->addLayout(overrideLayout, 0, 3);
			versionLayout->addWidget(new QLabel("Notes:", versionGroup), 1, 0);
			versionLayout->addWidget(notesEdit, 1, 1, 1, 3);
			versionLayout->setColumnStretch(1, 1);
			versionLayout->setColumnStretch(3, 1);
			detailsLayout->addWidget(versionGroup);

			auto* overridesGroup = new QGroupBox("Parameter Overrides", detailsWidget);
			auto* overridesLayout = new QVBoxLayout(overridesGroup);

			auto choices = std::make_shared<std::vector<ParameterChoice>>(BuildParameterChoices(CurrentAgentName(), segment.stageName));
			auto* editLayout = new QGridLayout();
			auto* typeCombo = new QComboBox(overridesGroup);
			auto* parameterCombo = new QComboBox(overridesGroup);
			auto* valueEdit = new QLineEdit(overridesGroup);
			auto* addButton = new QPushButton("Add / Update", overridesGroup);
			removeModButton = new QPushButton("Remove Selected", overridesGroup);
			valueEdit->setFixedWidth(OverrideValueEditWidth);

			typeCombo->addItem("Reward Weight", static_cast<int>(StageParameterOverrideKind::RewardWeight));
			typeCombo->addItem("Hyperparameter", static_cast<int>(StageParameterOverrideKind::Hyperparameter));

			editLayout->addWidget(new QLabel("Type:", overridesGroup), 0, 0);
			editLayout->addWidget(typeCombo, 0, 1);
			editLayout->addWidget(new QLabel("Parameter:", overridesGroup), 0, 2);
			editLayout->addWidget(parameterCombo, 0, 3);
			editLayout->addWidget(new QLabel("Override Value:", overridesGroup), 1, 0);
			editLayout->addWidget(valueEdit, 1, 1);
			editLayout->addWidget(addButton, 1, 2);
			editLayout->addWidget(removeModButton, 1, 3);
			editLayout->setColumnStretch(3, 1);
			overridesLayout->addLayout(editLayout);

			modsTable = new QTableWidget(overridesGroup);
			modsTable->setColumnCount(3);
			modsTable->setHorizontalHeaderLabels({ "Type", "Parameter", "Override Value" });
			modsTable->horizontalHeader()->setStretchLastSection(true);
			modsTable->verticalHeader()->setVisible(false);
			modsTable->setSelectionBehavior(QAbstractItemView::SelectRows);
			modsTable->setSelectionMode(QAbstractItemView::SingleSelection);
			overridesLayout->addWidget(modsTable);

			std::function<void()> refreshParameterCombo = [choices, typeCombo, parameterCombo, valueEdit, addButton]() {
				const StageParameterOverrideKind kind =
					static_cast<StageParameterOverrideKind>(typeCombo->currentData().toInt());

				QSignalBlocker blocker(parameterCombo);
				parameterCombo->clear();

				for (int i = 0; i < static_cast<int>(choices->size()); i++) {
					if ((*choices)[i].kind == kind)
						parameterCombo->addItem((*choices)[i].displayName, i);
				}

				const int choiceIndex = parameterCombo->currentData().toInt();
				const bool hasChoice = choiceIndex >= 0 && choiceIndex < static_cast<int>(choices->size());
				parameterCombo->setEnabled(hasChoice);
				valueEdit->setEnabled(hasChoice);
				addButton->setEnabled(hasChoice);
				valueEdit->setText(hasChoice ? (*choices)[choiceIndex].currentValue : QString());
			};

			std::function<void()> refreshModTable = [this, row, modsTable, modsSummaryLabel, removeModButton]() {
				if (row < 0 || row >= static_cast<int>(m_segments.size()))
					return;

				const std::vector<StageParameterOverride>& overrides = m_segments[row].overrides;
				modsTable->setRowCount(static_cast<int>(overrides.size()));

				for (int modRow = 0; modRow < static_cast<int>(overrides.size()); modRow++) {
					const StageParameterOverride& override = overrides[modRow];
					modsTable->setItem(modRow, 0, ReadOnlyItem(OverrideKindLabel(override.kind)));
					modsTable->setItem(modRow, 1, ReadOnlyItem(OverrideDisplayName(override)));
					modsTable->setItem(modRow, 2, ReadOnlyItem(QString::fromStdString(override.value)));
				}

				modsSummaryLabel->setText(ModsSummary(m_segments[row]));
				removeModButton->setEnabled(modsTable->currentRow() >= 0);
				modsTable->resizeColumnsToContents();
			};

			connect(typeCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, refreshParameterCombo);
			connect(parameterCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [choices, parameterCombo, valueEdit]() {
				const int choiceIndex = parameterCombo->currentData().toInt();
				if (choiceIndex < 0 || choiceIndex >= static_cast<int>(choices->size())) {
					valueEdit->clear();
					return;
				}

				valueEdit->setText((*choices)[choiceIndex].currentValue);
			});
			connect(modsTable, &QTableWidget::itemSelectionChanged, this, [this, row, choices, typeCombo, parameterCombo, valueEdit, removeModButton]() {
				if (row < 0 || row >= static_cast<int>(m_segments.size()))
					return;

				QTableWidget* table = qobject_cast<QTableWidget*>(sender());
				if (!table)
					return;

				const int modRow = table->currentRow();
				removeModButton->setEnabled(modRow >= 0);
				if (modRow < 0 || modRow >= static_cast<int>(m_segments[row].overrides.size()))
					return;

				const StageParameterOverride& override = m_segments[row].overrides[modRow];
				typeCombo->setCurrentIndex(override.kind == StageParameterOverrideKind::RewardWeight ? 0 : 1);

				for (int i = 0; i < parameterCombo->count(); i++) {
					const int choiceIndex = parameterCombo->itemData(i).toInt();
					if (choiceIndex < 0 || choiceIndex >= static_cast<int>(choices->size()))
						continue;

					const ParameterChoice& choice = (*choices)[choiceIndex];
					if (choice.group.toStdString() == override.group && choice.name.toStdString() == override.name) {
						parameterCombo->setCurrentIndex(i);
						break;
					}
				}

				valueEdit->setText(QString::fromStdString(override.value));
			});
			connect(addButton, &QPushButton::clicked, this, [this, row, choices, parameterCombo, valueEdit, modsTable, refreshModTable]() {
				if (row < 0 || row >= static_cast<int>(m_segments.size()))
					return;

				const int choiceIndex = parameterCombo->currentData().toInt();
				if (choiceIndex < 0 || choiceIndex >= static_cast<int>(choices->size()))
					return;

				const QString value = valueEdit->text().trimmed();
				if (value.isEmpty()) {
					QMessageBox::warning(this, "Param Testing Mods", "Enter a new value.");
					return;
				}

				StageParameterOverride override = MakeOverride((*choices)[choiceIndex], value);
				std::vector<StageParameterOverride>& overrides = m_segments[row].overrides;
				auto existing = std::find_if(overrides.begin(), overrides.end(), [&](const StageParameterOverride& current) {
					return SameOverride(current, override);
				});

				const int selectedRow = modsTable->currentRow();
				if (existing != overrides.end()) {
					*existing = std::move(override);
				} else if (selectedRow >= 0 && selectedRow < static_cast<int>(overrides.size())) {
					overrides[selectedRow] = std::move(override);
				} else {
					overrides.push_back(std::move(override));
				}

				refreshModTable();
				UpdateInputState();
			});
			connect(removeModButton, &QPushButton::clicked, this, [this, row, modsTable, refreshModTable]() {
				if (row < 0 || row >= static_cast<int>(m_segments.size()))
					return;

				const int modRow = modsTable->currentRow();
				if (modRow < 0 || modRow >= static_cast<int>(m_segments[row].overrides.size()))
					return;

				m_segments[row].overrides.erase(m_segments[row].overrides.begin() + modRow);
				refreshModTable();
				UpdateInputState();
			});

			refreshParameterCombo();
			refreshModTable();
			detailsLayout->addWidget(overridesGroup);

			connect(versionCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this, row, versionCombo](int) {
				if (m_refreshingSegments || row < 0 || row >= static_cast<int>(m_segments.size()))
					return;

				m_segments[row].versionIncrement =
					static_cast<RocketForgeTraining::VersionIncrement>(versionCombo->currentData().toInt());
				RefreshSegmentList();
				UpdateInputState();
			});

			auto updateOverrideVersion = [this, row, overrideVersionCheck, overrideMajorSpin, overrideMinorSpin]() {
				if (row < 0 || row >= static_cast<int>(m_segments.size()))
					return;

				overrideMajorSpin->setEnabled(overrideVersionCheck->isChecked());
				overrideMinorSpin->setEnabled(overrideVersionCheck->isChecked());
				if (overrideVersionCheck->isChecked()) {
					GGL::SegmentVersionNumber version = {};
					version.major = overrideMajorSpin->value();
					version.minor = overrideMinorSpin->value();
					m_segments[row].versionOverride = version;
				} else {
					m_segments[row].versionOverride.reset();
				}
				RefreshSegmentList();
				UpdateInputState();
			};
			connect(overrideVersionCheck, &QCheckBox::toggled, this, updateOverrideVersion);
			connect(overrideMajorSpin, qOverload<int>(&QSpinBox::valueChanged), this, updateOverrideVersion);
			connect(overrideMinorSpin, qOverload<int>(&QSpinBox::valueChanged), this, updateOverrideVersion);
			connect(notesEdit, &QTextEdit::textChanged, this, [this, row, notesEdit]() {
				if (m_refreshingSegments || row < 0 || row >= static_cast<int>(m_segments.size()))
					return;

				m_segments[row].versionNotes = notesEdit->toPlainText().trimmed();
				UpdateInputState();
			});

			segmentLayout->addWidget(detailsWidget);
		}

		connect(stageCombo, &QComboBox::currentTextChanged, this, [this, row](const QString& stageName) {
			if (m_refreshingSegments || row < 0 || row >= static_cast<int>(m_segments.size()))
				return;

			if (m_segments[row].stageName == stageName)
				return;

			m_segments[row].stageName = stageName;
			m_segments[row].overrides.clear();
			RefreshSegmentList();
			UpdateInputState();
		});
		connect(detailsButton, &QToolButton::clicked, this, [this, row](bool checked) {
			if (row < 0 || row >= static_cast<int>(m_segments.size()))
				return;

			m_segments[row].detailsExpanded = checked;
			RefreshSegmentList();
		});
		connect(moveUpButton, &QPushButton::clicked, this, [this, row]() {
			if (row <= 0 || row >= static_cast<int>(m_segments.size()))
				return;

			std::swap(m_segments[row - 1], m_segments[row]);
			RefreshSegmentList();
			UpdateInputState();
		});
		connect(moveDownButton, &QPushButton::clicked, this, [this, row]() {
			if (row < 0 || row + 1 >= static_cast<int>(m_segments.size()))
				return;

			std::swap(m_segments[row], m_segments[row + 1]);
			RefreshSegmentList();
			UpdateInputState();
		});
		connect(removeButton, &QPushButton::clicked, this, [this, row]() {
			if (row < 0 || row >= static_cast<int>(m_segments.size()))
				return;

			m_segments.erase(m_segments.begin() + row);
			RefreshSegmentList();
			UpdateInputState();
		});

		m_segmentListLayout->addWidget(segmentGroup);
	}

	m_segmentListLayout->addStretch(1);
	m_refreshingSegments = false;
}

QStringList ParamTestingWidget::ValidationErrors() const
{
	QStringList errors;
	if (CurrentAgentName().isEmpty())
		errors << "Select an agent.";
	if (CurrentBaselineBuildName().isEmpty())
		errors << "Select a baseline model build.";
	if (CurrentBaselineStageName().isEmpty())
		errors << "Select a baseline model stage.";
	if (CurrentBaselineSegmentId().isEmpty())
		errors << "Select a baseline model with checkpoints.";
	if (!m_branchEdit || m_branchEdit->text().trimmed().isEmpty())
		errors << "Enter a branch name.";
	if (m_segmentTimestepsSpin->value() <= 0.0)
		errors << "Enter Timesteps per Segment.";
	if (m_segments.empty())
		errors << "Add at least one test segment.";

	for (const SegmentRow& row : m_segments) {
		if (row.stageName.trimmed().isEmpty())
			errors << "Every segment needs a training stage.";
	}

	return errors;
}

ParamTestingConfig ParamTestingWidget::BuildConfig() const
{
	ParamTestingConfig config = {};
	config.agentName = CurrentAgentName().toStdString();
	config.buildName = CurrentBaselineBuildName().toStdString();
	config.baselineStageName = CurrentBaselineStageName().toStdString();
	config.baselineSegmentId = CurrentBaselineSegmentId().toStdString();
	config.branchName = m_branchEdit ? m_branchEdit->text().trimmed().toStdString() : std::string();
	config.segmentTimesteps = static_cast<uint64_t>(m_segmentTimestepsSpin->value() * 1'000'000.0 + 0.5);
	config.metricsEnabled = m_metricsCheck->isChecked();
	config.skillTestingEnabled = m_skillTestingCheck->isChecked();
	config.runBenchmark = m_runBenchmarkCheck->isChecked();
	config.benchmarkGamesPerMode = m_benchmarkGamesSpin->value();

	for (int i = 0; i < static_cast<int>(m_segments.size()); i++) {
		const SegmentRow& row = m_segments[i];
		RocketForgeTraining::ParamTestingSegmentConfig segment = {};
		segment.label = QString("Segment %1").arg(i + 1).toStdString();
		segment.stageName = row.stageName.trimmed().toStdString();
		segment.versionIncrement = row.versionIncrement;
		segment.versionOverride = row.versionOverride;
		segment.versionNotes = row.versionNotes.trimmed().toStdString();
		segment.parameterOverrides = row.overrides;
		config.segments.push_back(std::move(segment));
	}

	return config;
}

void ParamTestingWidget::OnRunClicked()
{
	const QStringList errors = ValidationErrors();
	if (!errors.empty()) {
		QMessageBox::warning(this, "Cannot Start Param Testing", errors.join("\n"));
		return;
	}

	ParamTestingConfig config = BuildConfig();
	try {
		RocketForgeTraining::BuildParamTestingPreviewStages(config);
	} catch (const std::exception& e) {
		QMessageBox::critical(this, "Cannot Start Param Testing", QString::fromStdString(e.what()));
		return;
	}

	m_runBenchmarkAfterTraining = config.runBenchmark;
	m_progressPanel->Reset();
	m_progressPanel->BeginRun("Starting Param Testing.");
	AppendLog("Starting Param Testing.");
	emit TrainingPreviewRequested(config);

	m_worker = new ParamTestingWorker;

	connect(m_worker, &ParamTestingWorker::ProgressUpdated, this, &ParamTestingWidget::OnProgressUpdated);
	connect(m_worker, &ParamTestingWorker::Completed, this, &ParamTestingWidget::OnCompleted);
	connect(m_worker, &ParamTestingWorker::Cancelled, this, &ParamTestingWidget::OnCancelled);
	connect(m_worker, &ParamTestingWorker::Failed, this, &ParamTestingWidget::OnFailed);
	connect(m_worker, &QObject::destroyed, this, [this]() { m_worker = nullptr; });

	m_workerHost.Start<ParamTestingWorker>(m_worker, [config](ParamTestingWorker* worker) { worker->RunParamTesting(config); });

	SetRunning(true);
}

void ParamTestingWidget::OnCancelClicked()
{
	if (!m_worker)
		return;

	m_progressPanel->SetStatus("Cancelling...");
	QMetaObject::invokeMethod(m_worker, "Cancel", Qt::DirectConnection);
}

void ParamTestingWidget::OnProgressUpdated(const ParamTestingProgress& progress)
{
	m_progressPanel->SetStatus(QString::fromStdString(progress.label));
	if (!progress.details.empty())
		AppendLog(QString::fromStdString(progress.label + ": " + progress.details));
	else
		AppendLog(QString::fromStdString(progress.label));

	if (progress.stageStarted) {
		emit TrainingPreviewStageStarted(
			progress.segmentIndex,
			QString::fromStdString(progress.stageName),
			QString::fromStdString(progress.versionId));
	}
	if (progress.stageCompleted) {
		emit TrainingPreviewStageCompleted(progress.segmentIndex, QString::fromStdString(progress.versionId));
	}

	if (progress.benchmark) {
		const int maxGames = std::max(1, progress.benchmarkMaxGames);
		m_progressPanel->SetProgress(progress.benchmarkFinishedGames, maxGames);
		return;
	}

	const int total = std::max(1, progress.segmentCount);
	const int current = std::clamp(progress.segmentIndex + 1, 0, total);
	m_progressPanel->SetProgress(current, total);
}

void ParamTestingWidget::OnCompleted(const ParamTestingResult& result)
{
	AppendLog(QString("Model Benchmark setup saved: %1").arg(QString::fromStdString(result.benchmarkSettingsPath.string())));
	if (result.benchmarkRan)
		AppendLog(QString("Benchmark complete: %1 scored games.").arg(result.benchmarkResult.scoredGames));

	emit BenchmarkSetupReady();
	emit TrainingPreviewFinished();
	SetRunning(false);
	m_progressPanel->EndRun("Complete");
	if (m_runBenchmarkAfterTraining)
		emit BenchmarkRunRequested();
}

void ParamTestingWidget::OnCancelled()
{
	AppendLog("Param Testing cancelled.");
	emit TrainingPreviewFinished();
	SetRunning(false);
	m_progressPanel->EndRun("Cancelled", /*markComplete*/ false);
}

void ParamTestingWidget::OnFailed(const QString& message, bool simulatorUnavailable)
{
	AppendLog("Param Testing failed: " + message);
	if (simulatorUnavailable)
		ShowSimulatorUnavailable(this, message);
	else
		QMessageBox::critical(this, "Param Testing Failed", message);
	emit TrainingPreviewFinished();
	SetRunning(false);
	m_progressPanel->EndRun("Failed", /*markComplete*/ false);
}

void ParamTestingWidget::SetRunning(bool running)
{
	m_running = running;
	if (!running) {
		RefreshBuildSelector();
		ModelCatalog::Instance().Refresh(CurrentAgentName());
		m_workerHost.Stop();
		m_worker = nullptr;
	}

	emit RunningChanged(running);
	UpdateInputState();
}

void ParamTestingWidget::UpdateInputState()
{
	const bool busy = m_running || m_externalRunActive;
	const QStringList errors = ValidationErrors();

	m_agentCombo->setEnabled(!busy);
	m_buildCombo->setEnabled(!busy);
	m_baselineStageCombo->setEnabled(!busy);
	m_baselineVersionCombo->setEnabled(!busy);
	m_segmentTimestepsSpin->setEnabled(!busy);
	m_branchEdit->setEnabled(!busy);
	m_metricsCheck->setEnabled(!busy);
	m_skillTestingCheck->setEnabled(!busy);
	m_runBenchmarkCheck->setEnabled(!busy);
	m_benchmarkGamesSpin->setEnabled(!busy && m_runBenchmarkCheck->isChecked());
	m_segmentScrollArea->setEnabled(!busy);
	m_addSegmentButton->setEnabled(!busy);
	m_runButton->setEnabled(!busy && errors.empty());
	m_runButton->setToolTip(busy ? "Another run is in progress." : errors.join('\n'));
	m_cancelButton->setEnabled(m_running);
}

void ParamTestingWidget::AppendLog(const QString& text)
{
	m_progressPanel->AppendLog(text);
}
