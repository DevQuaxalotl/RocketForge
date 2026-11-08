#include "TrainingStageRowWidget.h"
#include "TimestepSpinBox.h"
#include "FormatUtils.h"
#include "ThemeUtils.h"
#include "WheelEventFilter.h"

#include <StageRegistry.h>
#include <GigaLearnInfer/AgentRegistry.h>
#include <ForgeTraining.h>
#include <TrainingStage.h>
#include <TrainingSetup/VersionManagement.h>
#include <GigaLearnCPP/SegmentVersioning.h>

#include <QButtonGroup>
#include <QCheckBox>
#include <QColor>
#include <QComboBox>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QIcon>
#include <QLabel>
#include <QLineEdit>
#include <QPainter>
#include <QPen>
#include <QPushButton>
#include <QRadioButton>
#include <QSignalBlocker>
#include <QSize>
#include <QSpinBox>
#include <QStyle>
#include <QTextEdit>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>

#include <algorithm>
#include <optional>

using namespace RocketForgeTraining;

namespace
{
	constexpr int SegmentStageRole = Qt::UserRole + 1;
	constexpr int SegmentBranchRole = Qt::UserRole + 2;
	constexpr const char* AllStagesLabel = "All Stages";
	constexpr int RowActionButtonSize = 24;
	constexpr int TargetStepButtonSize = 24;
	constexpr int TargetStepButtonGap = 2;
	constexpr int TargetSpinExtraWidth = 50;
	constexpr int StageNameComboMinWidth = 190;
	constexpr const char* CurrentTimestepsWidthProbe = "Cur: 999.9M";
	constexpr const char* EtaWidthProbe = "ETA: ~99m 59s";
	constexpr const char* VersionWidthProbe = "Will continue: v999.99";
	constexpr int LiveProgressColumnPad = 22;
	constexpr int VersionColumnPad = 38;

	QString SegmentDisplayLabel(const GGL::SegmentVersionSummary& segment)
	{
		QString label = QString::fromStdString(segment.id);
		if (segment.latestTimesteps > 0)
			label += QString(" - %1").arg(ForgeGUI::FormatTimesteps(segment.latestTimesteps));
		if (!segment.stageName.empty())
			label += QString(" - %1").arg(QString::fromStdString(segment.stageName));
		return label;
	}

	QString MakeSegmentRevisionLabel(const GGL::SegmentVersionNumber& version)
	{
		return QString::fromStdString(GGL::MakeSegmentVersionId(version.major, version.minor));
	}

	QString NormalizeBranchName(const QString& branchName)
	{
		const QString trimmed = branchName.trimmed();
		return trimmed.isEmpty() ? QString("main") : trimmed;
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

	void AddUniqueBranchName(QStringList& branchNames, const QString& branchName)
	{
		const QString normalized = NormalizeBranchName(branchName);
		if (!branchNames.contains(normalized, Qt::CaseSensitive))
			branchNames.append(normalized);
	}
}

// ─────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────

TrainingStageRowWidget::TrainingStageRowWidget(QWidget* parent)
	: QWidget(parent)
{
	auto* rootLayout = new QVBoxLayout(this);
	rootLayout->setContentsMargins(0, 0, 0, 0);
	rootLayout->setSpacing(0);

	// ── Header ──────────────────────────────
	auto* headerWidget = new QWidget(this);
	auto* headerLayout = new QHBoxLayout(headerWidget);
	headerLayout->setContentsMargins(4, 4, 4, 4);
	headerLayout->setSpacing(6);

	BuildHeader(headerLayout);
	rootLayout->addWidget(headerWidget);

	// ── Body (collapsed by default) ─────────
	BuildBody();
	m_bodyBox->setVisible(false);
	rootLayout->addWidget(m_bodyBox);

	// ── Connect all change signals ───────────
	// Header widgets
	connect(m_stageNameCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
	        this, &TrainingStageRowWidget::OnAnyChanged);
	connect(m_targetSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
	        this, &TrainingStageRowWidget::OnAnyChanged);

	// Bootstrap
	connect(m_bootstrapNewRadio,      &QRadioButton::toggled, this, &TrainingStageRowWidget::OnAnyChanged);
	connect(m_bootstrapLatestRadio,   &QRadioButton::toggled, this, &TrainingStageRowWidget::OnAnyChanged);
	connect(m_bootstrapLoadRadio,     &QRadioButton::toggled, this, &TrainingStageRowWidget::OnAnyChanged);
	connect(m_sourceStageCombo,  QOverload<int>::of(&QComboBox::currentIndexChanged),
	        this, &TrainingStageRowWidget::OnAnyChanged);
	connect(m_sourceVersionCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
	        this, &TrainingStageRowWidget::OnAnyChanged);
	connect(m_branchContinueRadio, &QRadioButton::toggled, this, &TrainingStageRowWidget::OnAnyChanged);
	connect(m_branchSelectedRadio, &QRadioButton::toggled, this, &TrainingStageRowWidget::OnAnyChanged);
	connect(m_branchSelectedCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
	        this, &TrainingStageRowWidget::OnAnyChanged);
	connect(m_branchCreateRadio, &QRadioButton::toggled, this, &TrainingStageRowWidget::OnAnyChanged);
	connect(m_newBranchEdit, &QLineEdit::textChanged, this, &TrainingStageRowWidget::OnAnyChanged);

	// Version
	connect(m_versionNoneRadio,    &QRadioButton::toggled, this, &TrainingStageRowWidget::OnAnyChanged);
	connect(m_versionMinorRadio,   &QRadioButton::toggled, this, &TrainingStageRowWidget::OnAnyChanged);
	connect(m_versionMajorRadio,   &QRadioButton::toggled, this, &TrainingStageRowWidget::OnAnyChanged);
	connect(m_overrideVersionCheck, &QCheckBox::toggled,   this, &TrainingStageRowWidget::OnAnyChanged);
	connect(m_overrideMajorSpin, QOverload<int>::of(&QSpinBox::valueChanged),
	        this, &TrainingStageRowWidget::OnAnyChanged);
	connect(m_overrideMinorSpin, QOverload<int>::of(&QSpinBox::valueChanged),
	        this, &TrainingStageRowWidget::OnAnyChanged);
	m_notesDebounceTimer = new QTimer(this);
	m_notesDebounceTimer->setSingleShot(true);
	m_notesDebounceTimer->setInterval(400);
	connect(m_notesDebounceTimer, &QTimer::timeout, this, &TrainingStageRowWidget::OnAnyChanged);
	connect(m_versionNotesEdit, &QTextEdit::textChanged,
	        this, [this]() { m_notesDebounceTimer->start(); });

	// Misc
	connect(m_warmStartCheck,      &QCheckBox::toggled, this, &TrainingStageRowWidget::OnAnyChanged);
	connect(m_warmStartIterSpin, QOverload<int>::of(&QSpinBox::valueChanged),
	        this, &TrainingStageRowWidget::OnAnyChanged);
	connect(m_keepRunIdCheck,        &QCheckBox::toggled, this, &TrainingStageRowWidget::OnAnyChanged);
	connect(m_staticOpponentsCheck,  &QCheckBox::toggled, this, &TrainingStageRowWidget::OnAnyChanged);

	// Bootstrap radio buttons also control sub-widget visibility, and refresh the version
	// dropdowns when their owning mode is selected (so the list stays in sync with disk).
	auto updateBootstrapSubWidgets = [this]() {
		bool loadMode = m_bootstrapLoadRadio->isChecked();
		bool latestMode = m_bootstrapLatestRadio->isChecked();
		m_sourceStageCombo->setVisible(loadMode);
		m_sourceVersionCombo->setVisible(loadMode);
		m_sourceStageLabel->setVisible(loadMode);
		m_sourceVersionLabel->setVisible(loadMode);
		m_latestSegmentLabel->setVisible(latestMode);
		UpdateVersionGroupVisibility();
		if (latestMode)
			RefreshLatestSegment();
		if (loadMode) {
			RefreshVersionsForStage(m_sourceVersionCombo, m_sourceStageCombo->currentText());
		}
		UpdateBranchControls();
	};
	connect(m_bootstrapNewRadio,      &QRadioButton::toggled, this, updateBootstrapSubWidgets);
	connect(m_bootstrapLatestRadio,   &QRadioButton::toggled, this, updateBootstrapSubWidgets);
	connect(m_bootstrapLoadRadio,     &QRadioButton::toggled, this, updateBootstrapSubWidgets);

	// Segment combo refresh: source stage combo is only a filter for LoadFromExisting mode.
	connect(m_sourceStageCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
	        this, [this](int) {
		if (m_bootstrapLoadRadio->isChecked())
			RefreshVersionsForStage(m_sourceVersionCombo, m_sourceStageCombo->currentText());
	});
	connect(m_sourceVersionCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
	        this, [this](int) { UpdateBranchControls(); });
	connect(m_branchCreateRadio, &QRadioButton::toggled, this, [this](bool checked) {
		m_newBranchEdit->setVisible(checked);
		UpdateBranchControls();
		UpdateVersionLabel();
	});
	connect(m_branchContinueRadio, &QRadioButton::toggled, this, [this](bool) { UpdateVersionLabel(); });
	connect(m_branchSelectedRadio, &QRadioButton::toggled, this, [this](bool) {
		UpdateBranchControls();
		UpdateVersionLabel();
	});
	connect(m_branchSelectedCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
	        this, [this](int) { UpdateVersionLabel(); });
	connect(m_newBranchEdit, &QLineEdit::textChanged, this, [this](const QString&) { UpdateVersionLabel(); });

	// Override version controls visibility; also grey out the increment radios when override is active
	connect(m_overrideVersionCheck, &QCheckBox::toggled, this, [this](bool checked) {
		m_overrideMajorSpin->setVisible(checked);
		m_overrideMinorSpin->setVisible(checked);
		m_versionNoneRadio->setEnabled(!checked);
		m_versionMinorRadio->setEnabled(!checked);
		m_versionMajorRadio->setEnabled(!checked);
	});
	m_overrideMajorSpin->setVisible(false);
	m_overrideMinorSpin->setVisible(false);

	// Warm start iter visibility
	connect(m_warmStartCheck, &QCheckBox::toggled, this, [this](bool checked) {
		m_warmStartIterSpin->setVisible(checked);
	});
	m_warmStartIterSpin->setVisible(m_warmStartCheck->isChecked());

	// Version label: update whenever anything that affects the displayed version changes
	auto refreshVersionLabel = [this]() { UpdateVersionLabel(); };
	connect(m_bootstrapNewRadio,      &QRadioButton::toggled, this, refreshVersionLabel);
	connect(m_bootstrapLatestRadio,   &QRadioButton::toggled, this, refreshVersionLabel);
	connect(m_bootstrapLoadRadio,     &QRadioButton::toggled, this, refreshVersionLabel);
	connect(m_sourceStageCombo,       QOverload<int>::of(&QComboBox::currentIndexChanged),
	        this, refreshVersionLabel);
	connect(m_sourceVersionCombo,     QOverload<int>::of(&QComboBox::currentIndexChanged),
	        this, refreshVersionLabel);
	connect(m_overrideVersionCheck,   &QCheckBox::toggled,    this, refreshVersionLabel);
	connect(m_overrideMajorSpin,      QOverload<int>::of(&QSpinBox::valueChanged),
	        this, refreshVersionLabel);
	connect(m_overrideMinorSpin,      QOverload<int>::of(&QSpinBox::valueChanged),
	        this, refreshVersionLabel);
	connect(m_versionNoneRadio,       &QRadioButton::toggled, this, refreshVersionLabel);
	connect(m_versionMinorRadio,      &QRadioButton::toggled, this, refreshVersionLabel);
	connect(m_versionMajorRadio,      &QRadioButton::toggled, this, refreshVersionLabel);
	connect(m_stageNameCombo,         QOverload<int>::of(&QComboBox::currentIndexChanged),
	        this, refreshVersionLabel);

	// Populate stage list
	RefreshAvailableStages();

	// Apply initial visibility for bootstrap sub-widgets
	updateBootstrapSubWidgets();
	UpdateBootstrapVisibility();

	// Set initial version label text
	UpdateVersionLabel();

	// ── Bug fix: prevent accidental scroll-wheel edits ────────────────────
	// Install a filter that swallows wheel events unless the widget has focus
	// (i.e. the user explicitly clicked into it).  Also set StrongFocus so
	// the wheel itself cannot grant focus — the user must click first.
	auto* wheelFilter = new WheelEventFilter(this);
	auto installWheelFilter = [&](QWidget* w) {
		w->installEventFilter(wheelFilter);
		w->setFocusPolicy(Qt::StrongFocus);
	};
	installWheelFilter(m_stageNameCombo);
	installWheelFilter(m_targetSpin);
	installWheelFilter(m_existingVersionCombo);
	installWheelFilter(m_sourceStageCombo);
	installWheelFilter(m_sourceVersionCombo);
	installWheelFilter(m_branchSelectedCombo);
	installWheelFilter(m_overrideMajorSpin);
	installWheelFilter(m_overrideMinorSpin);
	installWheelFilter(m_warmStartIterSpin);
}

// ─────────────────────────────────────────────
// BuildHeader
// ─────────────────────────────────────────────

void TrainingStageRowWidget::BuildHeader(QHBoxLayout* layout)
{
	m_stagePrefixLabel = new QLabel("Segment 1:", this);
	layout->addWidget(m_stagePrefixLabel);

	m_stageNameCombo = new QComboBox(this);
	m_stageNameCombo->setMinimumWidth(StageNameComboMinWidth);
	m_stageNameCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
	layout->addWidget(m_stageNameCombo);

	layout->addWidget(new QLabel("→", this));

	m_targetSpin = new TimestepSpinBox(this);
	m_targetSpin->setObjectName("targetTimestepsSpin");
	m_targetSpin->setRange(1.0, 1e7);        // 1M to 10 trillion timesteps (stored in millions)
	m_targetSpin->setDecimals(2);
	// Size for the widest expected text ("999.99M") + spin arrows + frame
	m_targetSpin->setMinimumWidth(m_targetSpin->fontMetrics().horizontalAdvance("999.99M") + TargetStepButtonSize * 2 + TargetStepButtonGap + TargetSpinExtraWidth);
	m_targetSpin->setSingleStep(1.0);
	m_targetSpin->setValue(1.0);
	m_targetSpin->setToolTip("Target total timesteps (e.g. 500M, 1.5B)");
	layout->addWidget(m_targetSpin);

	m_currentTimestepsLabel = new QLabel("Cur: \xe2\x80\x94", this);  // "Cur: —"
	m_currentTimestepsLabel->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
	m_currentTimestepsLabel->setMinimumWidth(fontMetrics().horizontalAdvance(CurrentTimestepsWidthProbe) + LiveProgressColumnPad);
	m_currentTimestepsLabel->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	layout->addWidget(m_currentTimestepsLabel);

	m_etaLabel = new QLabel("", this);
	m_etaLabel->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
	m_etaLabel->setMinimumWidth(fontMetrics().horizontalAdvance(EtaWidthProbe) + LiveProgressColumnPad);
	m_etaLabel->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	layout->addWidget(m_etaLabel);

	m_versionLabel = new QLabel("", this);
	m_versionLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
	m_versionLabel->setToolTip("Segment that will be created or used when this queue entry runs");
	m_versionLabel->setFixedWidth(fontMetrics().horizontalAdvance(VersionWidthProbe) + VersionColumnPad);
	layout->addWidget(m_versionLabel);

	m_expandButton = new QToolButton(this);
	m_expandButton->setObjectName("rowSettingsButton");
	m_expandButton->setIcon(style()->standardIcon(QStyle::SP_FileDialogDetailedView));
	m_expandButton->setFixedSize(RowActionButtonSize, RowActionButtonSize);
	m_expandButton->setIconSize(QSize(14, 14));
	m_expandButton->setCheckable(true);
	m_expandButton->setChecked(false);
	m_expandButton->setToolTip("Show / hide segment settings");
	connect(m_expandButton, &QToolButton::clicked, this, &TrainingStageRowWidget::OnExpandToggled);
	layout->addWidget(m_expandButton);

	m_upButton = new QPushButton(this);
	m_upButton->setObjectName("rowMoveUpButton");
	m_upButton->setIcon(MakeChevronIcon(true));
	m_upButton->setIconSize(QSize(16, 16));
	m_upButton->setFixedSize(RowActionButtonSize, RowActionButtonSize);
	m_upButton->setToolTip("Move up");
	connect(m_upButton, &QPushButton::clicked, this, &TrainingStageRowWidget::moveUpRequested);
	layout->addWidget(m_upButton);

	m_downButton = new QPushButton(this);
	m_downButton->setObjectName("rowMoveDownButton");
	m_downButton->setIcon(MakeChevronIcon(false));
	m_downButton->setIconSize(QSize(16, 16));
	m_downButton->setFixedSize(RowActionButtonSize, RowActionButtonSize);
	m_downButton->setToolTip("Move down");
	connect(m_downButton, &QPushButton::clicked, this, &TrainingStageRowWidget::moveDownRequested);
	layout->addWidget(m_downButton);

	m_deleteButton = new QPushButton("X", this);
	m_deleteButton->setObjectName("rowDeleteButton");
	m_deleteButton->setFixedSize(RowActionButtonSize, RowActionButtonSize);
	m_deleteButton->setToolTip("Remove segment");
	connect(m_deleteButton, &QPushButton::clicked, this, &TrainingStageRowWidget::deleteRequested);
	layout->addWidget(m_deleteButton);
}

// ─────────────────────────────────────────────
// BuildBody
// ─────────────────────────────────────────────

void TrainingStageRowWidget::BuildBody()
{
	m_bodyBox = new QGroupBox(this);
	m_bodyBox->setObjectName("stageOptionsGroup");
	auto* bodyLayout = new QVBoxLayout(m_bodyBox);

	// ── Bootstrap group (first stage only) ──
	m_bootstrapGroup = new QGroupBox("Bootstrap Segment From...", m_bodyBox);
	m_bootstrapGroup->setObjectName("stageBootstrapGroup");
	auto* bootstrapLayout = new QVBoxLayout(m_bootstrapGroup);

	m_bootstrapNewRadio      = new QRadioButton("New (start from scratch)", m_bootstrapGroup);
	m_bootstrapLatestRadio   = new QRadioButton("Continue Latest Segment", m_bootstrapGroup);
	m_bootstrapContinueRadio = new QRadioButton(m_bootstrapGroup);
	m_bootstrapContinueRadio->setVisible(false);
	m_bootstrapLoadRadio     = new QRadioButton("Load From Existing Segment", m_bootstrapGroup);
	m_bootstrapNewRadio->setChecked(true);

	bootstrapLayout->addWidget(m_bootstrapNewRadio);
	bootstrapLayout->addWidget(m_bootstrapLatestRadio);
	m_latestSegmentLabel = new QLabel("Latest: none", m_bootstrapGroup);
	m_latestSegmentLabel->setProperty(ForgeGUI::TextRoleProperty, "muted");
	bootstrapLayout->addWidget(m_latestSegmentLabel);

	m_existingVersionCombo = new QComboBox(m_bootstrapGroup);
	m_existingVersionCombo->setVisible(false);

	bootstrapLayout->addWidget(m_bootstrapLoadRadio);

	auto* sourceRow = new QGridLayout();
	sourceRow->setContentsMargins(0, 0, 0, 0);
	sourceRow->setHorizontalSpacing(8);
	m_sourceStageCombo   = new QComboBox(m_bootstrapGroup);
	m_sourceVersionCombo = new QComboBox(m_bootstrapGroup);
	m_sourceStageCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
	m_sourceVersionCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
	m_sourceStageCombo->setVisible(false);
	m_sourceVersionCombo->setVisible(false);
	m_sourceStageLabel   = new QLabel("Training Stage:", m_bootstrapGroup);
	m_sourceVersionLabel = new QLabel("Segment:", m_bootstrapGroup);
	m_sourceStageLabel->setVisible(false);
	m_sourceVersionLabel->setVisible(false);
	sourceRow->addWidget(m_sourceVersionLabel, 0, 0);
	sourceRow->addWidget(m_sourceVersionCombo, 0, 1);
	sourceRow->addWidget(m_sourceStageLabel, 0, 2);
	sourceRow->addWidget(m_sourceStageCombo, 0, 3);
	sourceRow->setColumnStretch(1, 1);
	sourceRow->setColumnStretch(3, 1);
	bootstrapLayout->addLayout(sourceRow);

	bodyLayout->addWidget(m_bootstrapGroup);

	// ── Subsequent stage info label ──────────
	m_subsequentInfoLabel = new QLabel(m_bodyBox);
	m_subsequentInfoLabel->setText("<i>Bootstraps from the previous segment.</i>");
	m_subsequentInfoLabel->setTextFormat(Qt::RichText);
	bodyLayout->addWidget(m_subsequentInfoLabel);

	// ── Segment revision section ─────────────
	m_versionGroup = new QGroupBox("Segment Branch and Revision", m_bodyBox);
	m_versionGroup->setObjectName("stageVersionGroup");
	auto* versionGroup = m_versionGroup;
	auto* versionLayout = new QFormLayout(versionGroup);

	m_branchOptionsWidget = new QWidget(versionGroup);
	auto* branchLayout = new QHBoxLayout(m_branchOptionsWidget);
	branchLayout->setContentsMargins(0, 0, 0, 0);
	m_branchContinueRadio = new QRadioButton("Continue branch (main)", m_branchOptionsWidget);
	m_branchSelectedRadio = new QRadioButton("Continue existing branch", m_branchOptionsWidget);
	m_branchSelectedCombo = new QComboBox(m_branchOptionsWidget);
	m_branchSelectedCombo->addItem("main");
	m_branchCreateRadio = new QRadioButton("Create new branch", m_branchOptionsWidget);
	m_newBranchEdit = new QLineEdit(m_branchOptionsWidget);
	m_newBranchEdit->setPlaceholderText("Branch name");
	auto* branchButtonGroup = new QButtonGroup(m_branchOptionsWidget);
	branchButtonGroup->setExclusive(true);
	branchButtonGroup->addButton(m_branchContinueRadio);
	branchButtonGroup->addButton(m_branchSelectedRadio);
	branchButtonGroup->addButton(m_branchCreateRadio);
	branchLayout->addWidget(m_branchContinueRadio);
	branchLayout->addWidget(m_branchSelectedRadio);
	branchLayout->addWidget(m_branchSelectedCombo);
	branchLayout->addWidget(m_branchCreateRadio);
	branchLayout->addWidget(m_newBranchEdit);
	branchLayout->addStretch();
	m_branchContinueRadio->setChecked(true);
	m_branchSelectedCombo->setVisible(false);
	m_newBranchEdit->setVisible(false);
	versionLayout->addRow("Branch:", m_branchOptionsWidget);

	auto* versionIncrRow = new QHBoxLayout();
	m_versionNoneRadio = new QRadioButton("No Increment", versionGroup);
	m_versionMinorRadio = new QRadioButton("Next Minor", versionGroup);
	m_versionMajorRadio = new QRadioButton("Next Major", versionGroup);
	auto* revisionButtonGroup = new QButtonGroup(versionGroup);
	revisionButtonGroup->setExclusive(true);
	revisionButtonGroup->addButton(m_versionNoneRadio);
	revisionButtonGroup->addButton(m_versionMinorRadio);
	revisionButtonGroup->addButton(m_versionMajorRadio);
	m_versionMinorRadio->setChecked(true);
	versionIncrRow->addWidget(m_versionNoneRadio);
	versionIncrRow->addWidget(m_versionMinorRadio);
	versionIncrRow->addWidget(m_versionMajorRadio);
	versionIncrRow->addStretch();
	versionLayout->addRow("Revision:", versionIncrRow);

	auto* overrideRow = new QHBoxLayout();
	m_overrideVersionCheck = new QCheckBox("Override", versionGroup);
	m_overrideMajorSpin = new QSpinBox(versionGroup);
	m_overrideMajorSpin->setRange(1, 99999);
	m_overrideMajorSpin->setPrefix("Major: ");
	m_overrideMinorSpin = new QSpinBox(versionGroup);
	m_overrideMinorSpin->setRange(0, 99999);
	m_overrideMinorSpin->setPrefix("Minor: ");
	overrideRow->addWidget(m_overrideVersionCheck);
	overrideRow->addWidget(m_overrideMajorSpin);
	overrideRow->addWidget(m_overrideMinorSpin);
	overrideRow->addStretch();
	versionLayout->addRow("Override Revision:", overrideRow);

	m_versionNotesEdit = new QTextEdit(versionGroup);
	m_versionNotesEdit->setFixedHeight(60);
	m_versionNotesEdit->setPlaceholderText("Optional notes for this segment...");
	versionLayout->addRow("Notes:", m_versionNotesEdit);

	bodyLayout->addWidget(versionGroup);

	// ── Warm start group ─────────────────────
	auto* warmGroup = new QGroupBox("Warm Start", m_bodyBox);
	warmGroup->setObjectName("stageWarmStartGroup");
	auto* warmLayout = new QHBoxLayout(warmGroup);

	m_warmStartCheck = new QCheckBox("Enable warm start", warmGroup);
	m_warmStartCheck->setChecked(true);
	m_warmStartIterSpin = new QSpinBox(warmGroup);
	m_warmStartIterSpin->setRange(1, 10000);
	m_warmStartIterSpin->setValue(20);
	m_warmStartIterSpin->setPrefix("Iterations: ");

	warmLayout->addWidget(m_warmStartCheck);
	warmLayout->addWidget(m_warmStartIterSpin);
	warmLayout->addStretch();

	bodyLayout->addWidget(warmGroup);

	// ── Misc options ─────────────────────────
	auto* miscGroup = new QGroupBox("Run Options", m_bodyBox);
	miscGroup->setObjectName("stageRunOptionsGroup");
	auto* miscLayout = new QHBoxLayout(miscGroup);
	m_keepRunIdCheck       = new QCheckBox("Keep Run ID", miscGroup);
	m_staticOpponentsCheck = new QCheckBox("Static Opponents", miscGroup);
	miscLayout->addWidget(m_keepRunIdCheck);
	miscLayout->addWidget(m_staticOpponentsCheck);
	miscLayout->addStretch();
	bodyLayout->addWidget(miscGroup);
}

// ─────────────────────────────────────────────
// Slots
// ─────────────────────────────────────────────

void TrainingStageRowWidget::OnExpandToggled()
{
	bool expanded = m_expandButton->isChecked();
	m_bodyBox->setVisible(expanded);
}

void TrainingStageRowWidget::OnAnyChanged()
{
	if (m_emittingChange)
		return;
	emit changed();
}

// ─────────────────────────────────────────────
// Public setters
// ─────────────────────────────────────────────

void TrainingStageRowWidget::SetAgentName(const QString& agentName)
{
	m_agentName = agentName;
	RefreshAvailableStages();
	RefreshLatestSegment();
	// Refresh version combos now that we know which agent's registry to query.
	if (m_isFirstStage) {
		if (m_bootstrapLoadRadio->isChecked())
			RefreshVersionsForStage(m_sourceVersionCombo, m_sourceStageCombo->currentText());
	}
	UpdateVersionLabel();
}

void TrainingStageRowWidget::SetBuildName(const QString& buildName)
{
	m_buildName = buildName.trimmed();
	RefreshLatestSegment();
	if (m_isFirstStage) {
		if (m_bootstrapLoadRadio && m_bootstrapLoadRadio->isChecked())
			RefreshVersionsForStage(m_sourceVersionCombo, m_sourceStageCombo->currentText());
	}
	UpdateVersionLabel();
}

void TrainingStageRowWidget::SetLatestSegmentRevision(std::optional<GGL::SegmentVersionNumber> latestRevision)
{
	m_latestSegmentRevision = latestRevision;
	UpdateVersionLabel();
}

void TrainingStageRowWidget::SetMinorSegmentRevisionBase(std::optional<GGL::SegmentVersionNumber> baseRevision)
{
	m_minorSegmentRevisionBase = baseRevision;
	UpdateVersionLabel();
}

void TrainingStageRowWidget::SetCompletionPreviewText(const QString& text)
{
	if (m_completionPreviewText == text)
		return;

	m_completionPreviewText = text;
	UpdateVersionLabel();
}

void TrainingStageRowWidget::SetDefaultBranchName(const QString& branchName)
{
	const QString normalized = NormalizeBranchName(branchName);
	if (m_continueBranchName == normalized)
		return;

	m_continueBranchName = normalized;
	UpdateBranchControls();
	UpdateVersionLabel();
}

void TrainingStageRowWidget::SetAvailableBranchNames(const QStringList& branchNames)
{
	if (!m_branchSelectedCombo)
		return;

	QStringList normalized;
	AddUniqueBranchName(normalized, "main");
	AddUniqueBranchName(normalized, m_continueBranchName);
	AddUniqueBranchName(normalized, m_latestSegmentBranchName);
	AddUniqueBranchName(normalized, m_newBranchEdit ? m_newBranchEdit->text() : QString());
	for (const QString& branchName : branchNames)
		AddUniqueBranchName(normalized, branchName);

	normalized.sort(Qt::CaseInsensitive);

	const QString previous = CurrentSelectedBranchName();
	QSignalBlocker blocker(m_branchSelectedCombo);
	m_branchSelectedCombo->clear();
	for (const QString& branchName : normalized)
		m_branchSelectedCombo->addItem(branchName);

	int index = m_branchSelectedCombo->findText(previous);
	if (index < 0)
		index = m_branchSelectedCombo->findText(m_continueBranchName);
	if (index < 0)
		index = m_branchSelectedCombo->findText("main");
	m_branchSelectedCombo->setCurrentIndex(std::max(0, index));
	UpdateBranchControls();
}

void TrainingStageRowWidget::SetIsFirstStage(bool first)
{
	m_isFirstStage = first;
	UpdateBootstrapVisibility();

	// Re-apply version group / keepRunId visibility now that m_isFirstStage is correct.
	// (FromStageSettings runs before SetIsFirstStage, so its visibility call was stale.)
	UpdateVersionGroupVisibility();

	if (first) {
		if (m_bootstrapLoadRadio->isChecked())
			RefreshVersionsForStage(m_sourceVersionCombo, m_sourceStageCombo->currentText());
	}
	UpdateVersionLabel();
}

void TrainingStageRowWidget::SetStageNumberPrefixVisible(bool visible)
{
	m_stagePrefixLabel->setVisible(visible);
}

void TrainingStageRowWidget::SetStageNumber(int number)
{
	m_segmentNumber = std::max(1, number);
	m_stagePrefixLabel->setText(QString("Segment %1:").arg(m_segmentNumber));
}

void TrainingStageRowWidget::SetReadOnly(bool readOnly)
{
	m_stageNameCombo->setEnabled(!readOnly);
	m_targetSpin->setEnabled(!readOnly);

	m_bootstrapNewRadio->setEnabled(!readOnly);
	m_bootstrapLatestRadio->setEnabled(!readOnly);
	m_bootstrapContinueRadio->setEnabled(!readOnly);
	m_bootstrapLoadRadio->setEnabled(!readOnly);
	m_existingVersionCombo->setEnabled(!readOnly);
	m_sourceStageCombo->setEnabled(!readOnly);
	m_sourceVersionCombo->setEnabled(!readOnly);
	m_branchContinueRadio->setEnabled(!readOnly);
	m_branchSelectedRadio->setEnabled(!readOnly);
	m_branchSelectedCombo->setEnabled(!readOnly);
	m_branchCreateRadio->setEnabled(!readOnly);
	m_newBranchEdit->setEnabled(!readOnly);

	const bool overrideChecked = m_overrideVersionCheck && m_overrideVersionCheck->isChecked();
	m_versionNoneRadio->setEnabled(!readOnly && !overrideChecked);
	m_versionMinorRadio->setEnabled(!readOnly && !overrideChecked);
	m_versionMajorRadio->setEnabled(!readOnly && !overrideChecked);
	m_overrideVersionCheck->setEnabled(!readOnly);
	m_overrideMajorSpin->setEnabled(!readOnly);
	m_overrideMinorSpin->setEnabled(!readOnly);
	m_versionNotesEdit->setEnabled(!readOnly);

	m_warmStartCheck->setEnabled(!readOnly);
	m_warmStartIterSpin->setEnabled(!readOnly);
	m_keepRunIdCheck->setEnabled(!readOnly);
	m_staticOpponentsCheck->setEnabled(!readOnly);

	m_upButton->setEnabled(!readOnly);
	m_downButton->setEnabled(!readOnly);
	m_deleteButton->setEnabled(!readOnly);
}

void TrainingStageRowWidget::SetActive(bool active)
{
	m_active = active;
	if (active) {
		m_completionPreviewText.clear();
		m_currentTimestepsLabel->setText("Cur: -");
		m_etaLabel->clear();
		UpdateVersionLabel();
	}
	RefreshColorMode();
}

void TrainingStageRowWidget::RefreshColorMode()
{
	if (!m_active) {
		setStyleSheet("");
		return;
	}

	const QString activeStyle =
		"TrainingStageRowWidget { background-color: #1d2223; border: 1px solid #62c6cc; border-radius: 4px; }";
	setStyleSheet(ForgeGUI::ApplyColorPreference(activeStyle, ForgeGUI::ApplicationColorsEnabled()));
}

void TrainingStageRowWidget::SetFreshQueueMode(bool fresh)
{
	m_freshQueueMode = fresh;
	UpdateVersionGroupVisibility();
}

void TrainingStageRowWidget::ClearCreatedIds()
{
	m_createdSegmentId.clear();
	UpdateVersionGroupVisibility();
	UpdateVersionLabel();
}

void TrainingStageRowWidget::UpdateLiveTotalTimesteps(quint64 currentTotal)
{
	m_currentTimestepsLabel->setText(QString("Cur: %1").arg(ForgeGUI::FormatTimesteps(currentTotal)));
}

void TrainingStageRowWidget::UpdateEtaText(const QString& etaText)
{
	m_etaLabel->setText(etaText);
}

void TrainingStageRowWidget::MarkCompleted(const QString& versionId)
{
	const QString completedVersion = versionId.trimmed();
	m_currentTimestepsLabel->setText("Cur: -");
	m_etaLabel->clear();
	m_completionPreviewText = completedVersion.isEmpty()
		? QString("Completed")
		: QString("%1 Completed").arg(completedVersion);
	UpdateVersionLabel();
	SetActive(false);
}

// ─────────────────────────────────────────────
// Serialization
// ─────────────────────────────────────────────

StageSettings TrainingStageRowWidget::ToStageSettings() const
{
	StageSettings s;

	s.stageName             = m_stageNameCombo->currentText().toStdString();
	s.targetTotalTimesteps  = static_cast<uint64_t>(m_targetSpin->value() * 1'000'000.0);

	// Bootstrap
	BootstrapConfig bc;
	if (m_bootstrapContinueRadio->isChecked()) {
		bc.mode = BootstrapConfig::Mode::ContinueExisting;
		bc.selectedExistingVersionId = CurrentComboId(m_existingVersionCombo).toStdString();
	} else if (m_bootstrapLatestRadio->isChecked()) {
		bc.mode = BootstrapConfig::Mode::ContinueLatest;
		bc.selectedExistingVersionId = m_latestSegmentId.toStdString();
	} else if (m_bootstrapLoadRadio->isChecked()) {
		bc.mode = BootstrapConfig::Mode::LoadFromOtherStage;
		bc.sourceStage     = CurrentComboStage(m_sourceVersionCombo).toStdString();
		bc.sourceVersionId = CurrentComboId(m_sourceVersionCombo).toStdString();
	} else {
		bc.mode = BootstrapConfig::Mode::New;
	}

	if (m_branchCreateRadio->isChecked()) {
		bc.branchMode = BootstrapConfig::BranchMode::CreateNew;
		s.branchName = m_newBranchEdit->text().trimmed().toStdString();
	} else if (m_branchSelectedRadio->isChecked()) {
		bc.branchMode = BootstrapConfig::BranchMode::ContinueSelected;
		s.branchName = CurrentSelectedBranchName().toStdString();
	} else {
		bc.branchMode = BootstrapConfig::BranchMode::ContinueExisting;
		s.branchName = CurrentContinueBranchName().toStdString();
	}
	s.bootstrap = bc;

	// Version increment
	if (m_versionMajorRadio->isChecked())
		s.versionIncrement = VersionIncrement::Major;
	else if (m_versionMinorRadio->isChecked())
		s.versionIncrement = VersionIncrement::Minor;
	else
		s.versionIncrement = VersionIncrement::None;

	// Version override
	if (m_overrideVersionCheck->isChecked()) {
		GGL::SegmentVersionNumber vn;
		vn.major = m_overrideMajorSpin->value();
		vn.minor = m_overrideMinorSpin->value();
		s.versionOverride = vn;
	}

	s.versionNotes = m_versionNotesEdit->toPlainText().toStdString();

	// Warm start
	s.warmStartPolicy     = m_warmStartCheck->isChecked();
	s.warmStartIterations = m_warmStartIterSpin->value();

	s.keepRunId              = m_keepRunIdCheck->isChecked();
	s.staticOpponentsEnabled = m_staticOpponentsCheck->isChecked();

	s.createdSegmentId = m_createdSegmentId.toStdString();

	return s;
}

void TrainingStageRowWidget::FromStageSettings(const StageSettings& s)
{
	m_emittingChange = true;

	// Stage name
	{
		int idx = m_stageNameCombo->findText(QString::fromStdString(s.stageName));
		if (idx >= 0)
			m_stageNameCombo->setCurrentIndex(idx);
		else if (!s.stageName.empty())
			m_stageNameCombo->setCurrentText(QString::fromStdString(s.stageName));
	}

	// Target timesteps (convert from raw count to millions)
	m_targetSpin->setValue(static_cast<double>(s.targetTotalTimesteps) / 1'000'000.0);

	// Bootstrap
	switch (s.bootstrap.mode) {
	case BootstrapConfig::Mode::ContinueLatest:
		m_bootstrapLatestRadio->setChecked(true);
		RefreshLatestSegment();
		break;
	case BootstrapConfig::Mode::ContinueExisting:
		{
			m_bootstrapLoadRadio->setChecked(true);
			const int allStagesIndex = m_sourceStageCombo->findText(AllStagesLabel);
			if (allStagesIndex >= 0)
				m_sourceStageCombo->setCurrentIndex(allStagesIndex);
			RefreshVersionsForStage(m_sourceVersionCombo, m_sourceStageCombo->currentText());
			QString vid = QString::fromStdString(s.bootstrap.selectedExistingVersionId);
			int idx = m_sourceVersionCombo->findData(vid);
			if (idx >= 0)
				m_sourceVersionCombo->setCurrentIndex(idx);
			else
				m_sourceVersionCombo->setCurrentText(vid);
		}
		break;
	case BootstrapConfig::Mode::LoadFromOtherStage:
		m_bootstrapLoadRadio->setChecked(true);
		{
			QString stage = QString::fromStdString(s.bootstrap.sourceStage.empty()
				? std::string(AllStagesLabel)
				: s.bootstrap.sourceStage);
			int idx = m_sourceStageCombo->findText(stage);
			if (idx >= 0)
				m_sourceStageCombo->setCurrentIndex(idx);
			else {
				const int allStagesIndex = m_sourceStageCombo->findText(AllStagesLabel);
				if (allStagesIndex >= 0)
					m_sourceStageCombo->setCurrentIndex(allStagesIndex);
			}

			RefreshVersionsForStage(m_sourceVersionCombo, m_sourceStageCombo->currentText());
			QString vid = QString::fromStdString(s.bootstrap.sourceVersionId);
			idx = m_sourceVersionCombo->findData(vid);
			if (idx >= 0)
				m_sourceVersionCombo->setCurrentIndex(idx);
			else
				m_sourceVersionCombo->setCurrentText(vid);
		}
		break;
	case BootstrapConfig::Mode::New:
	default:
		m_bootstrapNewRadio->setChecked(true);
		break;
	}

	if (s.bootstrap.branchMode == BootstrapConfig::BranchMode::CreateNew) {
		m_branchCreateRadio->setChecked(true);
		m_newBranchEdit->setText(QString::fromStdString(s.branchName.empty() ? std::string("main") : s.branchName));
		m_newBranchEdit->setVisible(true);
	} else if (s.bootstrap.branchMode == BootstrapConfig::BranchMode::ContinueSelected) {
		m_branchSelectedRadio->setChecked(true);
		SetAvailableBranchNames(QStringList{ QString::fromStdString(s.branchName.empty() ? std::string("main") : s.branchName) });
		const int index = m_branchSelectedCombo->findText(QString::fromStdString(s.branchName.empty() ? std::string("main") : s.branchName));
		if (index >= 0)
			m_branchSelectedCombo->setCurrentIndex(index);
		m_branchSelectedCombo->setVisible(true);
		m_newBranchEdit->setVisible(false);
	} else {
		m_continueBranchName = QString::fromStdString(s.branchName.empty() ? std::string("main") : s.branchName);
		m_branchContinueRadio->setChecked(true);
		m_branchSelectedCombo->setVisible(false);
		m_newBranchEdit->setVisible(false);
	}
	UpdateBranchControls();

	// Segment revision increment. A new/uncreated segment allocates the next minor revision
	// even when older settings JSON has the previous "none" default.
	VersionIncrement displayedIncrement = s.versionIncrement;
	const bool hasCreatedSegment = !s.createdSegmentId.empty();
	const bool continuesExisting =
		((s.bootstrap.mode == BootstrapConfig::Mode::ContinueExisting ||
		  s.bootstrap.mode == BootstrapConfig::Mode::ContinueLatest) &&
		 (!s.bootstrap.selectedExistingVersionId.empty() || hasCreatedSegment)) ||
		(s.bootstrap.mode == BootstrapConfig::Mode::LoadFromOtherStage &&
		 !s.bootstrap.sourceVersionId.empty());
	if (displayedIncrement == VersionIncrement::None && !hasCreatedSegment && !continuesExisting)
		displayedIncrement = VersionIncrement::Minor;

	if (displayedIncrement == VersionIncrement::Major)
		m_versionMajorRadio->setChecked(true);
	else if (displayedIncrement == VersionIncrement::Minor)
		m_versionMinorRadio->setChecked(true);
	else
		m_versionNoneRadio->setChecked(true);

	// Version override
	if (s.versionOverride.has_value()) {
		m_overrideVersionCheck->setChecked(true);
		m_overrideMajorSpin->setValue(s.versionOverride->major);
		m_overrideMinorSpin->setValue(s.versionOverride->minor);
		m_overrideMajorSpin->setVisible(true);
		m_overrideMinorSpin->setVisible(true);
		m_versionNoneRadio->setEnabled(false);
		m_versionMinorRadio->setEnabled(false);
		m_versionMajorRadio->setEnabled(false);
	} else {
		m_overrideVersionCheck->setChecked(false);
		m_overrideMajorSpin->setVisible(false);
		m_overrideMinorSpin->setVisible(false);
		m_versionNoneRadio->setEnabled(true);
		m_versionMinorRadio->setEnabled(true);
		m_versionMajorRadio->setEnabled(true);
	}

	m_versionNotesEdit->setPlainText(QString::fromStdString(s.versionNotes));

	// Warm start
	m_warmStartCheck->setChecked(s.warmStartPolicy);
	m_warmStartIterSpin->setValue(s.warmStartIterations);
	m_warmStartIterSpin->setVisible(s.warmStartPolicy);

	m_keepRunIdCheck->setChecked(s.keepRunId);
	m_staticOpponentsCheck->setChecked(s.staticOpponentsEnabled);

	m_createdSegmentId = QString::fromStdString(s.createdSegmentId);

	m_emittingChange = false;

	// Sync sub-widget visibility for bootstrap mode
	{
		bool loadMode     = m_bootstrapLoadRadio->isChecked();
		bool latestMode   = m_bootstrapLatestRadio->isChecked();
		m_sourceStageCombo->setVisible(loadMode);
		m_sourceVersionCombo->setVisible(loadMode);
		m_sourceStageLabel->setVisible(loadMode);
		m_sourceVersionLabel->setVisible(loadMode);
		m_latestSegmentLabel->setVisible(latestMode);
		UpdateVersionGroupVisibility();
		UpdateBranchControls();
	}

	UpdateVersionLabel();
}

// ─────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────

void TrainingStageRowWidget::UpdateVersionGroupVisibility()
{
	// Hide version settings and Keep Run ID only when this row is locked to a previously
	// created queue segment. First-stage ContinueExisting keeps the controls visible so
	// Override can create a new segment revision from the selected source segment.
	//
	// "Will continue" is true when:
	//   - Non-first segment with a recorded created segment that won't be cleared
	//     by an earlier row's pending output change.
	bool willContinue = false;
	if (!m_isFirstStage) {
		const bool requestsNewRevision =
			(m_branchCreateRadio && m_branchCreateRadio->isChecked()) ||
			(m_branchSelectedRadio && m_branchSelectedRadio->isChecked()) ||
			(m_overrideVersionCheck && m_overrideVersionCheck->isChecked()) ||
			(m_versionMinorRadio && m_versionMinorRadio->isChecked()) ||
			(m_versionMajorRadio && m_versionMajorRadio->isChecked());
		willContinue = !m_createdSegmentId.isEmpty() && !m_freshQueueMode && !requestsNewRevision;
	}

	if (m_versionGroup)   m_versionGroup->setVisible(!willContinue);
	if (m_keepRunIdCheck) m_keepRunIdCheck->setVisible(!willContinue);
}

void TrainingStageRowWidget::RefreshAvailableStages()
{
	// Repopulate stage combo from registry
	std::vector<std::string> stages;
	if (!m_agentName.isEmpty()) {
		auto agent = AgentRegistry::Create(m_agentName.toStdString());
		stages = RocketForge::StageRegistry::GetCompatibleStages(*agent);
	} else {
		stages = RocketForge::StageRegistry::GetAvailableStages();
	}
	{
		QSignalBlocker blocker(m_stageNameCombo);
		QString current = m_stageNameCombo->currentText();
		m_stageNameCombo->clear();
		for (const auto& name : stages)
			m_stageNameCombo->addItem(QString::fromStdString(name));
		if (!current.isEmpty()) {
			int idx = m_stageNameCombo->findText(current);
			if (idx >= 0)
				m_stageNameCombo->setCurrentIndex(idx);
			else if (m_stageNameCombo->count() > 0)
				m_stageNameCombo->setCurrentIndex(0);
		}
	}

	// Repopulate source stage combo (for bootstrap LoadFromOtherStage)
	{
		QSignalBlocker blocker(m_sourceStageCombo);
		QString current = m_sourceStageCombo->currentText();
		m_sourceStageCombo->clear();
		m_sourceStageCombo->addItem(AllStagesLabel);
		for (const auto& name : stages)
			m_sourceStageCombo->addItem(QString::fromStdString(name));
		if (!current.isEmpty()) {
			int idx = m_sourceStageCombo->findText(current);
			if (idx >= 0)
				m_sourceStageCombo->setCurrentIndex(idx);
			else if (m_sourceStageCombo->count() > 0)
				m_sourceStageCombo->setCurrentIndex(0);
		} else if (m_sourceStageCombo->count() > 0) {
			m_sourceStageCombo->setCurrentIndex(0);
		}
	}
}

void TrainingStageRowWidget::RefreshLatestSegment()
{
	m_latestSegmentId.clear();
	m_latestSegmentBranchName = "main";

	if (!m_latestSegmentLabel)
		return;

	if (m_agentName.isEmpty() || m_buildName.isEmpty()) {
		if (m_bootstrapLatestRadio)
			m_bootstrapLatestRadio->setText("Continue Latest Segment (none)");
		m_latestSegmentLabel->setText("Latest: none");
		UpdateBranchControls();
		return;
	}

	std::optional<GGL::SegmentVersionSummary> latest =
		RocketForgeTraining::LatestBuildVersion(m_agentName.toStdString(), m_buildName.toStdString());
	if (!latest.has_value()) {
		if (m_bootstrapLatestRadio)
			m_bootstrapLatestRadio->setText("Continue Latest Segment (none)");
		m_latestSegmentLabel->setText("Latest: none");
		UpdateBranchControls();
		return;
	}

	m_latestSegmentId = QString::fromStdString(latest->id);
	m_latestSegmentBranchName = QString::fromStdString(latest->branchName.empty() ? std::string("main") : latest->branchName);
	if (m_bootstrapLatestRadio)
		m_bootstrapLatestRadio->setText(QString("Continue Latest Segment (%1)").arg(m_latestSegmentId));
	m_latestSegmentLabel->setText(
		QString("Latest: %1 - Branch: %2")
			.arg(SegmentDisplayLabel(latest.value()))
			.arg(m_latestSegmentBranchName));
	UpdateBranchControls();
}

void TrainingStageRowWidget::UpdateBootstrapVisibility()
{
	if (m_bootstrapGroup && m_subsequentInfoLabel) {
		m_bootstrapGroup->setVisible(m_isFirstStage);
		m_subsequentInfoLabel->setVisible(!m_isFirstStage);
	}
}

void TrainingStageRowWidget::UpdateBranchControls()
{
	if (!m_branchContinueRadio || !m_branchSelectedCombo || !m_newBranchEdit)
		return;

	const QString branchName = CurrentContinueBranchName();
	m_branchContinueRadio->setText(QString("Continue branch (%1)").arg(branchName));
	m_branchSelectedCombo->setVisible(m_branchSelectedRadio && m_branchSelectedRadio->isChecked());
	m_newBranchEdit->setVisible(m_branchCreateRadio && m_branchCreateRadio->isChecked());
	if (m_newBranchEdit->text().trimmed().isEmpty())
		m_newBranchEdit->setPlaceholderText(branchName == "main" ? "new-branch" : branchName + "-2");
}

void TrainingStageRowWidget::RefreshVersionsForStage(QComboBox* targetCombo, const QString& stageName)
{
	QSignalBlocker blocker(targetCombo);
	QString prevSelection = CurrentComboId(targetCombo);
	targetCombo->clear();
	if (m_agentName.isEmpty() || m_buildName.isEmpty())
		return;

	const std::string stageFilter = stageName == AllStagesLabel
		? std::string()
		: stageName.toStdString();
	auto versions = RocketForgeTraining::ListBuildVersions(
		m_agentName.toStdString(),
		m_buildName.toStdString(),
		stageFilter,
		false);
	std::sort(versions.begin(), versions.end(), [](const GGL::SegmentVersionSummary& a, const GGL::SegmentVersionSummary& b) {
		if (a.major != b.major)
			return a.major > b.major;
		if (a.minor != b.minor)
			return a.minor > b.minor;
		return a.id > b.id;
	});
	for (const auto& v : versions) {
		targetCombo->addItem(SegmentDisplayLabel(v), QString::fromStdString(v.id));
		targetCombo->setItemData(targetCombo->count() - 1, QString::fromStdString(v.stageName), SegmentStageRole);
		targetCombo->setItemData(targetCombo->count() - 1, QString::fromStdString(v.branchName.empty() ? std::string("main") : v.branchName), SegmentBranchRole);
	}
	if (!prevSelection.isEmpty()) {
		int idx = targetCombo->findData(prevSelection);
		if (idx >= 0)
			targetCombo->setCurrentIndex(idx);
	}

	UpdateBranchControls();
}

QString TrainingStageRowWidget::CurrentComboId(const QComboBox* combo) const
{
	if (!combo)
		return {};

	const QString id = combo->currentData().toString().trimmed();
	return id.isEmpty() ? combo->currentText().trimmed() : id;
}

QString TrainingStageRowWidget::CurrentComboStage(const QComboBox* combo) const
{
	if (!combo)
		return {};

	return combo->currentData(SegmentStageRole).toString().trimmed();
}

QString TrainingStageRowWidget::CurrentComboBranch(const QComboBox* combo) const
{
	if (!combo)
		return "main";

	const QString branch = combo->currentData(SegmentBranchRole).toString().trimmed();
	return branch.isEmpty() ? QString("main") : branch;
}

QString TrainingStageRowWidget::CurrentContinueBranchName() const
{
	if (m_isFirstStage) {
		if (m_bootstrapLatestRadio && m_bootstrapLatestRadio->isChecked()) {
			const QString branch = m_latestSegmentBranchName.trimmed();
			return branch.isEmpty() ? QString("main") : branch;
		}

		if (m_bootstrapLoadRadio && m_bootstrapLoadRadio->isChecked())
			return CurrentComboBranch(m_sourceVersionCombo);

		if (m_bootstrapContinueRadio && m_bootstrapContinueRadio->isChecked())
			return CurrentComboBranch(m_existingVersionCombo);
	}

	const QString branch = m_continueBranchName.trimmed();
	return branch.isEmpty() ? QString("main") : branch;
}

QString TrainingStageRowWidget::CurrentSelectedBranchName() const
{
	if (!m_branchSelectedCombo)
		return "main";

	return NormalizeBranchName(m_branchSelectedCombo->currentText());
}

void TrainingStageRowWidget::UpdateVersionLabel()
{
	if (!m_versionLabel)
		return;

	if (m_buildName.trimmed().isEmpty()) {
		SetVersionPreviewText("Set build first");
		return;
	}

	if (!m_completionPreviewText.isEmpty()) {
		SetVersionPreviewText(m_completionPreviewText);
		return;
	}

	if (m_isFirstStage && m_bootstrapLatestRadio && m_bootstrapLatestRadio->isChecked()) {
		const bool createsRevision =
			(m_branchCreateRadio && m_branchCreateRadio->isChecked()) ||
			(m_branchSelectedRadio && m_branchSelectedRadio->isChecked()) ||
			(m_overrideVersionCheck && m_overrideVersionCheck->isChecked()) ||
			(m_versionMinorRadio && m_versionMinorRadio->isChecked()) ||
			(m_versionMajorRadio && m_versionMajorRadio->isChecked());
		if (m_latestSegmentId.isEmpty())
			SetVersionPreviewText(createsRevision ? QString("From latest: ? -> %1").arg(ComputeNextSegmentString()) : "Will continue latest: none");
		else if (createsRevision)
			SetVersionPreviewText(QString("From latest: %1 -> %2").arg(m_latestSegmentId, ComputeNextSegmentString()));
		else
			SetVersionPreviewText(QString("Will continue latest: %1").arg(m_latestSegmentId));
		return;
	}

	// LoadFromExistingSegment: No Increment resumes the selected segment; increment
	// or override creates a new segment revision from that selected checkpoint.
	if (m_isFirstStage && m_bootstrapLoadRadio && m_bootstrapLoadRadio->isChecked()) {
		QString srcVer = CurrentComboId(m_sourceVersionCombo);
		const bool createsRevision =
			(m_branchCreateRadio && m_branchCreateRadio->isChecked()) ||
			(m_branchSelectedRadio && m_branchSelectedRadio->isChecked()) ||
			(m_overrideVersionCheck && m_overrideVersionCheck->isChecked()) ||
			(m_versionMinorRadio && m_versionMinorRadio->isChecked()) ||
			(m_versionMajorRadio && m_versionMajorRadio->isChecked());
		if (srcVer.isEmpty())
			SetVersionPreviewText(createsRevision ? QString("From: ? -> %1").arg(ComputeNextSegmentString()) : "Will continue: (none selected)");
		else if (createsRevision)
			SetVersionPreviewText(QString("From: %1 -> %2").arg(srcVer, ComputeNextSegmentString()));
		else
			SetVersionPreviewText(QString("Will continue: %1").arg(srcVer));
		return;
	}

	// A non-first segment with a previously-created revision is resumed unless an earlier
	// queue entry changes output, in which case lineage prep clears the created ID first.
	const bool requestsNewRevision =
		(m_branchCreateRadio && m_branchCreateRadio->isChecked()) ||
		(m_branchSelectedRadio && m_branchSelectedRadio->isChecked()) ||
		(m_overrideVersionCheck && m_overrideVersionCheck->isChecked()) ||
		(m_versionMinorRadio && m_versionMinorRadio->isChecked()) ||
		(m_versionMajorRadio && m_versionMajorRadio->isChecked());
	if (!m_isFirstStage && !m_createdSegmentId.isEmpty() && !m_freshQueueMode && !requestsNewRevision) {
		SetVersionPreviewText(QString("Will continue: %1").arg(m_createdSegmentId));
		return;
	}

	// New (or non-first segment without a prior segment): show revision that will be created.
	QString newVer = ComputeNextSegmentString();
	SetVersionPreviewText(QString("Will create: %1").arg(newVer));
}

void TrainingStageRowWidget::SetVersionPreviewText(const QString& text)
{
	m_versionLabel->setText(text);
	m_versionLabel->setToolTip(text);
}

QString TrainingStageRowWidget::ComputeNextSegmentString() const
{
	const bool firstStageContinueCreatingRevision =
		m_isFirstStage &&
		((m_bootstrapContinueRadio && m_bootstrapContinueRadio->isChecked()) ||
		 (m_bootstrapLatestRadio && m_bootstrapLatestRadio->isChecked()) ||
		 (m_bootstrapLoadRadio && m_bootstrapLoadRadio->isChecked())) &&
		((m_branchCreateRadio && m_branchCreateRadio->isChecked()) ||
		 (m_branchSelectedRadio && m_branchSelectedRadio->isChecked()) ||
		 (m_overrideVersionCheck && m_overrideVersionCheck->isChecked()) ||
		 (m_versionMinorRadio && m_versionMinorRadio->isChecked()) ||
		 (m_versionMajorRadio && m_versionMajorRadio->isChecked()));
	const bool firstStageExplicitNew =
		m_isFirstStage &&
		m_bootstrapNewRadio &&
		m_bootstrapNewRadio->isChecked();
	const bool branchStartsNewPath =
		(m_branchCreateRadio && m_branchCreateRadio->isChecked()) ||
		(m_branchSelectedRadio && m_branchSelectedRadio->isChecked());

	if (!firstStageExplicitNew &&
		!firstStageContinueCreatingRevision &&
		!branchStartsNewPath &&
		!m_createdSegmentId.isEmpty() &&
		!m_freshQueueMode) {
		return m_createdSegmentId;
	}

	if (m_overrideVersionCheck && m_overrideVersionCheck->isChecked()) {
		GGL::SegmentVersionNumber overrideVersion = {};
		overrideVersion.major = m_overrideMajorSpin->value();
		overrideVersion.minor = m_overrideMinorSpin->value();
		return MakeSegmentRevisionLabel(overrideVersion);
	}

	if (m_versionMajorRadio && m_versionMajorRadio->isChecked()) {
		GGL::SegmentVersionNumber next = m_latestSegmentRevision.value_or(GGL::SegmentVersionNumber{ 1, -1 });
		next.major += 1;
		next.minor = 0;
		return MakeSegmentRevisionLabel(next);
	}

	GGL::SegmentVersionNumber next =
		m_minorSegmentRevisionBase.value_or(m_latestSegmentRevision.value_or(GGL::SegmentVersionNumber{ 1, -1 }));
	next.minor += 1;
	if (next.minor < 0)
		next.minor = 0;

	return MakeSegmentRevisionLabel(next);
}
