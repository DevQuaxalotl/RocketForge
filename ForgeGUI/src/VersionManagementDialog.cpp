#include "VersionManagementDialog.h"
#include "ImportModelDialog.h"
#include <QApplication>
#include <QTimer>
#include "FormatUtils.h"
#include "ThemeUtils.h"

#include "TrainingSetup/TrainingSetupWindow.h"
#include "Widgets/AgentComboBox.h"

#include <ForgeTraining.h>
#include <Storage/BuildStorage.h>
#include <Storage/ImportedModelStorage.h>
#include <TrainingSetup/SubmittalExport/SubmittalExport.h>
#include <TrainingStage.h>
#include <GigaLearnInfer/AgentRegistry.h>
#include <StageRegistry.h>

#include <QApplication>
#include <QPainter>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDateTime>
#include <QDir>
#include <QBrush>
#include <QColor>
#include <QFontMetrics>
#include <QGroupBox>
#include <QHeaderView>
#include <QItemSelection>
#include <QItemSelectionModel>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QCheckBox>
#include <QComboBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QRadioButton>
#include <QSettings>
#include <QSizePolicy>
#include <QSignalBlocker>
#include <QStandardPaths>
#include <QStyle>
#include <QStringList>
#include <QStyledItemDelegate>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QTextOption>
#include <QVBoxLayout>
#include <QHBoxLayout>

#include <functional>
#include <algorithm>
#include <filesystem>
#include <memory>
#include <optional>
#include <set>
#include <sstream>

namespace
{
    constexpr int PreserveSelectedForegroundRole = Qt::UserRole + 1;
    constexpr int VersionNotesColumnWidth = 280;

    constexpr const char* AllBuildsLabel = "All Builds";
    constexpr const char* AllStagesLabel = "All Training Stages";
    constexpr const char* AllBranchesLabel = "All Branches";
    constexpr const char* SortColumnSettingsKey = "fileManagement/sortColumn";
    constexpr const char* SortOrderSettingsKey = "fileManagement/sortOrder";

    class PreserveSelectedForegroundDelegate : public QStyledItemDelegate
    {
    public:
        using QStyledItemDelegate::QStyledItemDelegate;

    protected:
        void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override
        {
            QStyleOptionViewItem opt(option);
            initStyleOption(&opt, index);
            if (!index.data(PreserveSelectedForegroundRole).toBool())
            {
                QStyledItemDelegate::paint(painter, opt, index);
                return;
            }

            const QVariant foreground = index.data(Qt::ForegroundRole);
            if (!(opt.state & QStyle::State_Selected) || !foreground.canConvert<QBrush>())
            {
                QStyledItemDelegate::paint(painter, opt, index);
                return;
            }

            const QBrush textBrush = qvariant_cast<QBrush>(foreground);
            const QWidget* widget = opt.widget;
            QStyle* style = widget ? widget->style() : QApplication::style();

            painter->save();
            style->drawPrimitive(QStyle::PE_PanelItemViewItem, &opt, painter, widget);

            QRect textRect = style->subElementRect(QStyle::SE_ItemViewItemText, &opt, widget);
            textRect.adjust(4, 0, -4, 0);
            painter->setPen(textBrush.color());
            painter->drawText(textRect, opt.displayAlignment | Qt::TextSingleLine, opt.text);
            painter->restore();
        }
    };

    QString FormatTimestamp(const std::string& rawTimestamp)
    {
        if (rawTimestamp.empty())
            return {};

        const QString raw = QString::fromStdString(rawTimestamp);
        const QDateTime parsed = QDateTime::fromString(raw, "yyyyMMdd-HHmmss");
        if (!parsed.isValid())
            return raw;
        return parsed.toString("MM/dd/yy hh:mm");
    }

    QString FormatNotesPreview(const QString& notes, const QWidget* widget)
    {
        const QString singleLineNotes = notes.simplified();
        if (singleLineNotes.isEmpty() || !widget)
            return singleLineNotes;

        return widget->fontMetrics().elidedText(
            singleLineNotes,
            Qt::ElideRight,
            VersionNotesColumnWidth - 12);
    }

    std::optional<QString> PromptForVersionNotes(QWidget* parent, const QString& versionId, const QString& existingNotes)
    {
        QDialog dialog(parent);
        dialog.setWindowTitle("Edit Version Notes");
        dialog.resize(520, 260);

        auto* layout = new QVBoxLayout(&dialog);
        layout->addWidget(new QLabel("Notes for " + versionId + ":", &dialog));

        auto* notesEdit = new QPlainTextEdit(existingNotes, &dialog);
        notesEdit->setLineWrapMode(QPlainTextEdit::WidgetWidth);
        notesEdit->setWordWrapMode(QTextOption::WrapAtWordBoundaryOrAnywhere);
        layout->addWidget(notesEdit);

        auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
        QObject::connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
        QObject::connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
        layout->addWidget(buttons);

        if (dialog.exec() != QDialog::Accepted)
            return std::nullopt;
        return notesEdit->toPlainText();
    }

    std::optional<QString> PromptForBranchMove(QWidget* parent, const std::vector<std::string>& branches)
    {
        QDialog dialog(parent);
        dialog.setWindowTitle("Move To Branch");
        dialog.resize(420, 160);

        auto* layout = new QVBoxLayout(&dialog);
        auto* existingRadio = new QRadioButton("Move to existing branch", &dialog);
        auto* existingCombo = new QComboBox(&dialog);
        auto* newRadio = new QRadioButton("Create new branch", &dialog);
        auto* newEdit = new QLineEdit(&dialog);
        newEdit->setPlaceholderText("Branch name");

        for (const std::string& branch : branches)
            existingCombo->addItem(QString::fromStdString(branch.empty() ? std::string("main") : branch));
        if (existingCombo->count() == 0)
            existingCombo->addItem("main");

        existingRadio->setChecked(true);
        newEdit->setEnabled(false);

        QObject::connect(existingRadio, &QRadioButton::toggled, existingCombo, &QComboBox::setEnabled);
        QObject::connect(newRadio, &QRadioButton::toggled, newEdit, &QLineEdit::setEnabled);

        auto* existingRow = new QHBoxLayout;
        existingRow->addWidget(existingRadio);
        existingRow->addWidget(existingCombo, 1);
        layout->addLayout(existingRow);

        auto* newRow = new QHBoxLayout;
        newRow->addWidget(newRadio);
        newRow->addWidget(newEdit, 1);
        layout->addLayout(newRow);

        auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
        QObject::connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
        QObject::connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
        layout->addWidget(buttons);

        if (dialog.exec() != QDialog::Accepted)
            return std::nullopt;

        const QString branchName = newRadio->isChecked()
            ? newEdit->text().trimmed()
            : existingCombo->currentText().trimmed();
        if (branchName.isEmpty()) {
            QMessageBox::information(parent, "Move To Branch", "Branch name is required.");
            return std::nullopt;
        }

        return branchName;
    }

    QString JoinSubmittalTags(const std::vector<std::string>& tags)
    {
        QStringList values;
        for (const std::string& tag : tags)
            values.append(QString::fromStdString(tag));
        return values.join(", ");
    }

    std::vector<std::string> ParseSubmittalTags(const QString& text)
    {
        std::vector<std::string> tags;
        for (const QString& value : text.split(',', Qt::SkipEmptyParts)) {
            const QString tag = value.trimmed();
            if (!tag.isEmpty())
                tags.push_back(tag.toStdString());
        }
        return tags;
    }

    struct SubmittalExportDialogResult
    {
        RocketForgeTraining::SubmittalBotMetadata metadata;
        std::filesystem::path outputParent;
    };

    std::optional<SubmittalExportDialogResult> PromptForSubmittalExport(
        QWidget* parent,
        const RocketForgeTraining::SubmittalBotMetadata& defaults)
    {
        QDialog dialog(parent);
        dialog.setWindowTitle("Submittal Bot Details");
        dialog.resize(620, 500);

        auto* layout = new QVBoxLayout(&dialog);
        auto* explanation = new QLabel(
            "Choose the identity and metadata written to rlbot/bot.toml. "
            "The export folder and Bob project will use the bot name.",
            &dialog);
        explanation->setWordWrap(true);
        layout->addWidget(explanation);

        QString selectedParent = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
        if (selectedParent.isEmpty())
            selectedParent = QDir::homePath();

        auto* form = new QFormLayout;
        auto* nameEdit = new QLineEdit(QString::fromStdString(defaults.name), &dialog);
        auto* folderRow = new QHBoxLayout;
        auto* folderEdit = new QLineEdit(&dialog);
        auto* browseFolderBtn = new QPushButton("Browse...", &dialog);
        folderRow->addWidget(folderEdit, 1);
        folderRow->addWidget(browseFolderBtn);
        auto* agentIdEdit = new QLineEdit(QString::fromStdString(defaults.agentId), &dialog);
        auto* developerEdit = new QLineEdit(QString::fromStdString(defaults.developer), &dialog);
        auto* languageEdit = new QLineEdit(QString::fromStdString(defaults.language), &dialog);
        auto* funFactEdit = new QLineEdit(QString::fromStdString(defaults.funFact), &dialog);
        auto* sourceLinkEdit = new QLineEdit(QString::fromStdString(defaults.sourceLink), &dialog);
        auto* tagsEdit = new QLineEdit(JoinSubmittalTags(defaults.tags), &dialog);
        auto* descriptionEdit = new QPlainTextEdit(QString::fromStdString(defaults.description), &dialog);
        auto* hivemindCheck = new QCheckBox("Team batching (hive mind)", &dialog);
        hivemindCheck->setChecked(defaults.hivemind);
        hivemindCheck->setToolTip(
            "Run matching teammates together to share one model and batch their decisions. "
            "Each car keeps its own controls. Different models must use different Agent IDs.");
        auto* gpuCheck = new QCheckBox("GPU inference", &dialog);
        gpuCheck->setChecked(defaults.useGPU);
        gpuCheck->setToolTip(
            "Windows with an NVIDIA GPU and shared CUDA LibTorch. "
            "Leave unchecked for the RLBot bot pack, which only builds CPU packages (Windows and Linux).");

        folderEdit->setReadOnly(true);
        folderEdit->setToolTip("The complete package path. Use Browse to choose its parent folder.");
        agentIdEdit->setToolTip("A unique RLBot identifier, usually Developer/BotName.");
        tagsEdit->setPlaceholderText("Comma-separated tags, for example: 1v1, teamplay");
        descriptionEdit->setFixedHeight(72);
        descriptionEdit->setLineWrapMode(QPlainTextEdit::WidgetWidth);

        form->addRow("Bot Name:", nameEdit);
        form->addRow("Export To:", folderRow);
        form->addRow("Agent ID:", agentIdEdit);
        form->addRow("Developer:", developerEdit);
        form->addRow("Description:", descriptionEdit);
        form->addRow("Fun Fact:", funFactEdit);
        form->addRow("Source Link:", sourceLinkEdit);
        form->addRow("Language:", languageEdit);
        form->addRow("Tags:", tagsEdit);
        form->addRow("Team Inference:", hivemindCheck);
        form->addRow("Device:", gpuCheck);
        layout->addLayout(form);

        auto* managedFields = new QLabel(
            "Run commands and loadout/logo paths are managed automatically so the package stays portable.",
            &dialog);
        managedFields->setWordWrap(true);
        layout->addWidget(managedFields);

        auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
        layout->addWidget(buttons);

        bool agentIdEdited = false;
        auto updateExportPath = [&]() {
            const QString folderName = QString::fromStdString(
                RocketForgeTraining::MakeSubmittalExportFolderName(nameEdit->text().trimmed().toStdString()));
            folderEdit->setText(QDir::toNativeSeparators(QDir(selectedParent).filePath(folderName)));
        };
        updateExportPath();

        QObject::connect(agentIdEdit, &QLineEdit::textEdited, [&](const QString&) {
            agentIdEdited = true;
        });
        QObject::connect(nameEdit, &QLineEdit::textEdited, [&](const QString& text) {
            const std::string botName = text.trimmed().toStdString();
            updateExportPath();
            if (!agentIdEdited) {
                agentIdEdit->setText(QString::fromStdString(
                    RocketForgeTraining::MakeSubmittalAgentId(botName)));
            }
        });
        QObject::connect(browseFolderBtn, &QPushButton::clicked, [&]() {
            const QString selected = QFileDialog::getExistingDirectory(
                &dialog,
                "Select Parent Folder for Submittal Export",
                selectedParent,
                QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
            if (selected.isEmpty())
                return;
            selectedParent = QDir::cleanPath(selected);
            updateExportPath();
        });
        QObject::connect(buttons->button(QDialogButtonBox::Ok), &QPushButton::clicked, [&]() {
            if (nameEdit->text().trimmed().isEmpty()) {
                QMessageBox::information(&dialog, "Submittal Bot Details", "Bot name is required.");
                return;
            }
            if (agentIdEdit->text().trimmed().isEmpty()) {
                QMessageBox::information(&dialog, "Submittal Bot Details", "Agent ID is required.");
                return;
            }
            dialog.accept();
        });
        QObject::connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);

        if (dialog.exec() != QDialog::Accepted)
            return std::nullopt;

        RocketForgeTraining::SubmittalBotMetadata metadata;
        metadata.name = nameEdit->text().trimmed().toStdString();
        metadata.agentId = agentIdEdit->text().trimmed().toStdString();
        metadata.developer = developerEdit->text().trimmed().toStdString();
        metadata.description = descriptionEdit->toPlainText().trimmed().toStdString();
        metadata.funFact = funFactEdit->text().trimmed().toStdString();
        metadata.sourceLink = sourceLinkEdit->text().trimmed().toStdString();
        metadata.language = languageEdit->text().trimmed().toStdString();
        metadata.tags = ParseSubmittalTags(tagsEdit->text());
        metadata.hivemind = hivemindCheck->isChecked();
        metadata.useGPU = gpuCheck->isChecked();

        return SubmittalExportDialogResult{
            std::move(metadata),
            std::filesystem::path(selectedParent.toStdWString())
        };
    }

    bool HasTrainingSetupSnapshotFile(const GGL::SegmentVersionSummary& version)
    {
        if (version.rootPath.empty())
            return false;

        return std::filesystem::exists(std::filesystem::path(version.rootPath) / "training_setup.json");
    }

    std::string BranchNameOrDefault(const std::string& branchName)
    {
        return branchName.empty() ? "main" : branchName;
    }

    QString FormatVersionStatus(bool importedModelFolderRow, const GGL::SegmentVersionSummary& version)
    {
        if (importedModelFolderRow)
            return "Imported";
        return version.available ? "Active" : "Inactive";
    }

    bool IsInternalImportedModelFolderName(const std::string& folderName)
    {
        return folderName.empty() || folderName.find("__") != std::string::npos;
    }

    std::string ImportedModelStageName(const std::optional<RocketForgeTraining::ImportedModelMetadata>& metadata)
    {
        if (metadata.has_value() && !metadata->sourceStageName.empty())
            return metadata->sourceStageName;
        return "Imported Models";
    }

    uint64_t ImportedModelTimesteps(
        const std::optional<RocketForgeTraining::ImportedModelMetadata>& metadata,
        const std::filesystem::path& modelFolder)
    {
        if (metadata.has_value() && metadata->latestTimesteps > 0)
            return metadata->latestTimesteps;
        if (!modelFolder.empty() && GGL::TrainingStorage::IsAllDigits(modelFolder.filename().string()))
            return static_cast<uint64_t>(std::stoull(modelFolder.filename().string()));
        return 0;
    }

    bool ImportedModelCanResume(const RocketForgeTraining::ImportedModelTrainingFileStatus& files)
    {
        return files.hasPolicy
            && files.hasCritic
            && files.hasRunningStats
            && files.hasPolicyOptim
            && files.hasCriticOptim
            && (!files.hasSharedHead || files.hasSharedHeadOptim)
            && (!files.hasAuxCritic || files.hasAuxCriticOptim);
    }

    void RefreshImportedModelMetadataFromFolder(
        RocketForgeTraining::ImportedModelMetadata& metadata,
        const std::filesystem::path& importedModelRoot)
    {
        const std::filesystem::path modelFolder = GGL::TrainingStorage::ResolveModelFolder(importedModelRoot);
        if (modelFolder.empty())
            return;

        metadata.files = RocketForgeTraining::InspectImportedModelTrainingFiles(modelFolder);
        metadata.canUseForInference = metadata.files.hasPolicy;
        metadata.canUseForTrainingWarmStart = metadata.files.hasPolicy;
        metadata.canUseForTrainingResume = ImportedModelCanResume(metadata.files);
        if (metadata.latestTimesteps == 0 && GGL::TrainingStorage::IsAllDigits(modelFolder.filename().string()))
            metadata.latestTimesteps = static_cast<uint64_t>(std::stoull(modelFolder.filename().string()));
    }

    QGroupBox* CreateActionGroup(
        const QString& title,
        const std::vector<QWidget*>& widgets,
        QWidget* parent)
    {
        auto* group = new QGroupBox(title, parent);
        auto* layout = new QVBoxLayout(group);
        for (QWidget* widget : widgets) {
            if (!widget)
                continue;
            widget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
            layout->addWidget(widget);
        }
        return group;
    }
}

VersionManagementDialog::VersionManagementDialog(
    const QString& initialAgentName,
    const QString& initialBuildName,
    const QString& initialStageName,
    bool trainingActive,
    const QString& activeAgentName,
    const QString& activeBuildName,
    const QString& activeStageName,
    const QString& activeVersionId,
    QWidget* parent)
    : QWidget(parent)
    , m_activeAgentName(activeAgentName)
    , m_activeBuildName(activeBuildName)
    , m_activeStageName(activeStageName)
    , m_trainingActive(trainingActive)
    , m_activeVersionId(activeVersionId)
{
    (void)initialStageName;

    setWindowTitle("File Management");
    resize(760, 480);

    auto* layout = new QVBoxLayout(this);

    auto* filtersGroup = new QGroupBox("Filters", this);
    auto* filtersGroupLayout = new QVBoxLayout(filtersGroup);

    auto* selectionLayout = new QHBoxLayout;
    selectionLayout->addWidget(new QLabel("Agent:", filtersGroup));
    m_agentCombo = new AgentComboBox(this);
    selectionLayout->addWidget(m_agentCombo);
    selectionLayout->addWidget(new QLabel("Build:", filtersGroup));
    m_buildCombo = new QComboBox(this);
    selectionLayout->addWidget(m_buildCombo);
    selectionLayout->addWidget(new QLabel("Training Stage:", filtersGroup));
    m_stageCombo = new QComboBox(this);
    selectionLayout->addWidget(m_stageCombo);
    selectionLayout->addWidget(new QLabel("Branch:", filtersGroup));
    m_branchCombo = new QComboBox(this);
    m_branchCombo->addItem(AllBranchesLabel);
    selectionLayout->addWidget(m_branchCombo);
    selectionLayout->addStretch();
    filtersGroupLayout->addLayout(selectionLayout);

    auto* filterLayout = new QHBoxLayout;
    m_hideInactiveCheck = new QCheckBox("Hide Inactive", this);
    m_searchEdit = new QLineEdit(this);
    m_searchEdit->setPlaceholderText("Search");
    filterLayout->addWidget(m_hideInactiveCheck);
    filterLayout->addSpacing(12);
    filterLayout->addWidget(new QLabel("Search:", filtersGroup));
    filterLayout->addWidget(m_searchEdit, 1);
    filtersGroupLayout->addLayout(filterLayout);
    layout->addWidget(filtersGroup);

    const int initialAgentIndex = m_agentCombo->findText(initialAgentName);
    if (initialAgentIndex >= 0)
        m_agentCombo->setCurrentIndex(initialAgentIndex);

    m_catalog = ModelCatalog::Instance().Get(CurrentAgentName());
    m_pendingBuild = initialBuildName;
    RefreshBuilds();
    const QString requestedBuildName = initialBuildName.isEmpty() ? QString(AllBuildsLabel) : initialBuildName;
    const int initialBuildIndex = m_buildCombo->findText(requestedBuildName);
    if (initialBuildIndex >= 0)
        m_buildCombo->setCurrentIndex(initialBuildIndex);
    RefreshStages();
    const int initialStageIndex = m_stageCombo->findText(AllStagesLabel);
    if (initialStageIndex >= 0)
        m_stageCombo->setCurrentIndex(initialStageIndex);

    m_table = new QTableWidget(this);
    m_table->setColumnCount(13);
    m_table->setHorizontalHeaderLabels({ "Build", "Segment", "Made From", "Stage", "Latest TS", "Branch", "Status", "Last Trained", "Skill Rating", "Model Count", "RLBot", "Training Setup", "Notes" });
    m_table->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_table->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_table->setFocusPolicy(Qt::NoFocus);
    m_table->setItemDelegate(new PreserveSelectedForegroundDelegate(m_table));
    m_table->setStyleSheet(
        "QTableWidget {"
        "  outline: 0;"
        "}"
        "QTableWidget::item {"
        "  border: none;"
        "}"
        "QTableWidget::item:focus {"
        "  outline: none;"
        "  border: none;"
        "}"
        "QTableWidget::item:selected,"
        "QTableWidget::item:selected:active,"
        "QTableWidget::item:selected:!active {"
        "  background-color: #2b747a;"
        "  border: none;"
        "  color: #ffffff;"
        "}");
    auto* header = m_table->horizontalHeader();
    header->setSectionsClickable(true);
    header->setSortIndicatorShown(false);
    header->setStretchLastSection(false);
    header->setDefaultAlignment(Qt::AlignCenter);

    TableSortController::Config sortConfig;
    sortConfig.columnCount = static_cast<int>(VersionColumn::Count);
    sortConfig.columnsInUiOrder.assign(ColumnsInUiOrder.begin(), ColumnsInUiOrder.end());
    sortConfig.defaultPriorityColumn = Segment;
    sortConfig.rowKey = [this](int row) { return RowKey(row); };
    sortConfig.compareColumn = [this](int lhsRow, int rhsRow, int priorityColumn) {
        return CompareVersionColumn(lhsRow, rhsRow, static_cast<VersionColumn>(priorityColumn));
    };
    m_sortController.Configure(sortConfig);
    m_sortController.AttachHeader(m_table, [this]() {
        QSettings settings("RocketForge", "ForgeGUI");
        settings.setValue(SortColumnSettingsKey, m_sortController.SortColumn());
        settings.setValue(SortOrderSettingsKey, static_cast<int>(m_sortController.SortOrder()));

        const std::vector<int> widths = CaptureColumnWidths();
        RefreshFromCatalog();
        RestoreColumnWidths(widths);
    });

    QSettings settings("RocketForge", "ForgeGUI");
    bool hasValidSavedColumn = false;
    const int savedSortColumn = settings.value(SortColumnSettingsKey, Segment).toInt(&hasValidSavedColumn);
    if (!hasValidSavedColumn || savedSortColumn < 0 || savedSortColumn >= Count)
        hasValidSavedColumn = false;

    bool hasValidSavedOrder = false;
    const int savedSortOrder = settings.value(
        SortOrderSettingsKey,
        static_cast<int>(Qt::DescendingOrder)).toInt(&hasValidSavedOrder);
    if (savedSortOrder != Qt::AscendingOrder && savedSortOrder != Qt::DescendingOrder)
        hasValidSavedOrder = false;

    m_sortController.SetSort(
        hasValidSavedColumn ? savedSortColumn : Segment,
        hasValidSavedOrder ? static_cast<Qt::SortOrder>(savedSortOrder) : Qt::DescendingOrder);

    const int notesColumn = m_sortController.UiIndexFromPriorityColumn(Notes);
    for (int col = 0; col < notesColumn; col++)
        header->setSectionResizeMode(col, QHeaderView::Interactive);
    header->setSectionResizeMode(notesColumn, QHeaderView::Fixed);
    header->resizeSection(notesColumn, VersionNotesColumnWidth);

    auto* contentLayout = new QHBoxLayout;
    contentLayout->addWidget(m_table, 1);

    auto* actionsWidget = new QWidget(this);
    actionsWidget->setFixedWidth(210);
    auto* actionsLayout = new QVBoxLayout(actionsWidget);
    actionsLayout->setContentsMargins(0, 0, 0, 0);
    auto* refreshBtn = new QPushButton("Refresh Files", this);
    m_deleteBtn = new QPushButton("Delete", this);
    m_pruneBtn = new QPushButton("Prune Old Models", this);
    m_importExternalModelBtn = new QPushButton("Import Model...", this);
    m_importExternalModelBtn->setToolTip(
        "<qt>Copy a trained model, such as an older checkpoint or a ported bot's policy, into an agent "
        "so it can be used as a static opponent, in Model Benchmark and in Debug render.</qt>");
    m_editNotesBtn = new QPushButton("Edit Notes...", this);
    m_moveBranchBtn = new QPushButton("Move Branch...", this);
    m_activeBtn = new QPushButton("Deactivate Segment", this);
    m_readOnlyLabel = new QLabel("Active training segment is protected", this);
    m_readOnlyLabel->setWordWrap(true);
    m_rlbotBtn = new QPushButton("Export to RLBot", this);
    m_submittalBtn = new QPushButton("Export for Submittal...", this);
    m_submittalBtn->setToolTip(
        "Choose the bot name and metadata, then package the selected model and minimum source needed "
        "for Bob the Bot Builder to build a standalone RLBot bot on another computer.");
    actionsLayout->addWidget(CreateActionGroup("Storage", { refreshBtn, m_importExternalModelBtn, m_deleteBtn, m_pruneBtn }, actionsWidget));
    actionsLayout->addWidget(CreateActionGroup("Metadata", { m_editNotesBtn, m_moveBranchBtn }, actionsWidget));
    actionsLayout->addWidget(CreateActionGroup("Availability", { m_activeBtn, m_rlbotBtn }, actionsWidget));
    actionsLayout->addWidget(CreateActionGroup("Export", { m_submittalBtn }, actionsWidget));
    actionsLayout->addWidget(m_readOnlyLabel);
    actionsLayout->addStretch();
    contentLayout->addWidget(actionsWidget);
    layout->addLayout(contentLayout, 1);
    m_readOnlyLabel->setVisible(false);
    m_readOnlyLabel->setProperty(ForgeGUI::TextRoleProperty, "muted");

    connect(refreshBtn, &QPushButton::clicked, this, [this]() {
        const bool repair = true;
        ModelCatalog::Instance().Refresh(CurrentAgentName(), repair);
    });
    connect(m_importExternalModelBtn, &QPushButton::clicked, this, &VersionManagementDialog::ImportExternalModel);
    connect(m_deleteBtn, &QPushButton::clicked, this, &VersionManagementDialog::DeleteVersion);
    connect(m_pruneBtn, &QPushButton::clicked, this, &VersionManagementDialog::PruneOldModels);
    connect(m_editNotesBtn, &QPushButton::clicked, this, &VersionManagementDialog::EditNotes);
    connect(m_moveBranchBtn, &QPushButton::clicked, this, &VersionManagementDialog::MoveBranch);
    connect(m_activeBtn, &QPushButton::clicked, this, &VersionManagementDialog::ToggleActive);
    connect(m_rlbotBtn, &QPushButton::clicked, this, &VersionManagementDialog::ExportRLBot);
    connect(m_submittalBtn, &QPushButton::clicked, this, &VersionManagementDialog::ExportForSubmittal);
    connect(m_table, &QTableWidget::itemSelectionChanged, this, &VersionManagementDialog::OnSelectionChanged);
    connect(m_hideInactiveCheck, &QCheckBox::toggled, this, [this](bool) { ApplyRowFilters(); });
    connect(m_searchEdit, &QLineEdit::textChanged, this, [this](const QString&) { ApplyRowFilters(); });
    connect(m_branchCombo, &QComboBox::currentTextChanged, this, [this](const QString&) { ApplyRowFilters(); });
    connect(m_agentCombo, &QComboBox::currentTextChanged, this, &VersionManagementDialog::OnContextChanged);
    connect(m_buildCombo, &QComboBox::currentTextChanged, this, &VersionManagementDialog::OnContextChanged);
    connect(m_stageCombo, &QComboBox::currentTextChanged, this, &VersionManagementDialog::OnContextChanged);

    connect(&ModelCatalog::Instance(), &ModelCatalog::Changed, this, [this](const QString& agent) {
        if (agent == CurrentAgentName()) ApplyCatalogUpdate();
    });
    connect(&ModelCatalog::Instance(), &ModelCatalog::Failed, this, [this](const QString& agent, const QString& error) {
        if (agent == CurrentAgentName()) m_table->setToolTip("Model refresh failed: " + error);
    });

    UpdateContextLabel();
    Refresh();
}

void VersionManagementDialog::SetContext(
    const QString& agentName,
    const QString& buildName,
    const QString& stageName)
{
    (void)stageName;

    QSignalBlocker agentBlocker(m_agentCombo);
    QSignalBlocker buildBlocker(m_buildCombo);
    QSignalBlocker stageBlocker(m_stageCombo);
    QSignalBlocker branchBlocker(m_branchCombo);

    const int agentIndex = m_agentCombo->findText(agentName);
    if (agentIndex >= 0)
        m_agentCombo->setCurrentIndex(agentIndex);

    m_catalog = ModelCatalog::Instance().Get(CurrentAgentName());
    m_pendingBuild = buildName;
    RefreshBuilds();
    const QString requestedBuildName = buildName.isEmpty() ? QString(AllBuildsLabel) : buildName;
    const int buildIndex = m_buildCombo->findText(requestedBuildName);
    if (buildIndex >= 0)
        m_buildCombo->setCurrentIndex(buildIndex);

    RefreshStages();
    const int stageIndex = m_stageCombo->findText(AllStagesLabel);
    if (stageIndex >= 0)
        m_stageCombo->setCurrentIndex(stageIndex);
    if (m_branchCombo && m_branchCombo->count() > 0)
        m_branchCombo->setCurrentIndex(0);

    UpdateContextLabel();
    RefreshPreservingColumnWidths();
}

void VersionManagementDialog::SetTrainingContext(
    bool trainingActive,
    const QString& activeAgentName,
    const QString& activeBuildName,
    const QString& activeStageName,
    const QString& activeVersionId)
{
    m_trainingActive = trainingActive;
    m_activeAgentName = activeAgentName;
    m_activeBuildName = activeBuildName;
    m_activeStageName = activeStageName;
    m_activeVersionId = activeVersionId;

    UpdateContextLabel();
    UpdateActionButtons();
}

void VersionManagementDialog::Refresh()
{
    ModelCatalog::Instance().Refresh(CurrentAgentName());
    ApplyCatalogUpdate();
}

void VersionManagementDialog::ApplyCatalogUpdate()
{
    // Keep row identities stable while a confirmation dialog is using a selection.
    if (QApplication::activeModalWidget()) {
        if (!m_catalogUpdateDeferred) {
            m_catalogUpdateDeferred = true;
            QTimer::singleShot(100, this, [this]() {
                m_catalogUpdateDeferred = false;
                ApplyCatalogUpdate();
            });
        }
        return;
    }

    // Let the first populated table size its columns; preserve user widths thereafter.
    const std::vector<int> widths = m_versions.empty() ? std::vector<int>{} : CaptureColumnWidths();
    m_catalog = ModelCatalog::Instance().Get(CurrentAgentName());
    m_table->setToolTip({});
    RefreshBuilds();
    RefreshStages();
    RefreshFromCatalog();
    RestoreColumnWidths(widths);
}

void VersionManagementDialog::RefreshFromCatalog()
{
    std::set<std::string> previousSelectedKeys;
    for (int row : SelectedRows())
        previousSelectedKeys.insert(RowKey(row));
    m_versions.clear();
    m_versionBuildNames.clear();
    m_versionStageNames.clear();
    m_versionIsImportedModelFolder.clear();
    m_importedModelRootPaths.clear();

    const std::string agentName = CurrentAgentName().toStdString();
    const std::string buildFilter = CurrentBuildName() == AllBuildsLabel ? std::string() : CurrentBuildName().toStdString();
    const std::string stageFilter = CurrentStageName() == AllStagesLabel ? std::string() : CurrentStageName().toStdString();

    const bool includeInactive = true;
    std::vector<GGL::SegmentVersionSummary> versions = m_catalog->Versions(buildFilter, stageFilter, includeInactive);

    for (GGL::SegmentVersionSummary& version : versions) {
        std::filesystem::path rootPath = version.rootPath;
        std::string buildName = version.buildName.empty() ? buildFilter : version.buildName;
        if (buildName.empty()) {
            std::filesystem::path buildRoot = rootPath.parent_path().parent_path();
            if (!buildRoot.empty())
                buildName = buildRoot.filename().string();
        }
        const std::string stageName = version.stageName.empty() ? stageFilter : version.stageName;
        m_versionBuildNames.push_back(buildName.empty() ? "N/A" : buildName);
        m_versionStageNames.push_back(stageName.empty() ? "N/A" : stageName);
        m_versionIsImportedModelFolder.push_back(false);
        m_importedModelRootPaths.push_back({});
        m_versions.push_back(std::move(version));
    }
    AddImportedModelRows(agentName, buildFilter, stageFilter);

    std::vector<int> order(m_versions.size());
    for (int i = 0; i < (int)order.size(); i++)
        order[i] = i;
    std::sort(order.begin(), order.end(), [&](int ia, int ib) {
        return m_sortController.CompareRows(ia, ib) < 0;
    });

    std::vector<GGL::SegmentVersionSummary> sortedVersions;
    std::vector<std::string> sortedBuilds;
    std::vector<std::string> sortedStages;
    std::vector<bool> sortedImportedModelRows;
    std::vector<std::filesystem::path> sortedImportedModelRoots;
    sortedVersions.reserve(m_versions.size());
    sortedBuilds.reserve(m_versionBuildNames.size());
    sortedStages.reserve(m_versionStageNames.size());
    sortedImportedModelRows.reserve(m_versionIsImportedModelFolder.size());
    sortedImportedModelRoots.reserve(m_importedModelRootPaths.size());
    for (int index : order) {
        sortedVersions.push_back(std::move(m_versions[index]));
        sortedBuilds.push_back(std::move(m_versionBuildNames[index]));
        sortedStages.push_back(std::move(m_versionStageNames[index]));
        sortedImportedModelRows.push_back(m_versionIsImportedModelFolder[index]);
        sortedImportedModelRoots.push_back(std::move(m_importedModelRootPaths[index]));
    }
    m_versions = std::move(sortedVersions);
    m_versionBuildNames = std::move(sortedBuilds);
    m_versionStageNames = std::move(sortedStages);
    m_versionIsImportedModelFolder = std::move(sortedImportedModelRows);
    m_importedModelRootPaths = std::move(sortedImportedModelRoots);
    RefreshBranches();

    const int notesColumn = m_sortController.UiIndexFromPriorityColumn(Notes);
    const int lastTrainedColumn = m_sortController.UiIndexFromPriorityColumn(LastTrained);
    const int trainingSetupColumn = m_sortController.UiIndexFromPriorityColumn(TrainingSetup);

    m_table->setRowCount((int)m_versions.size());
    for (int row = 0; row < (int)m_versions.size(); row++) {
        const auto& v = m_versions[row];
        const bool importedModelFolderRow = IsImportedModelFolderRow(row);
        const bool activeTrainingRow = IsActiveTrainingRow(row);
        std::ostringstream ratingStream;
        if (v.hasSkillRating)
            ratingStream << std::fixed << std::setprecision(1) << v.skillRating;
        const QString notesText = QString::fromStdString(v.notes);
        QStringList values = {
            QString::fromStdString(m_versionBuildNames[row]),
            QString::fromStdString(v.id),
            v.sourceVersionId.empty() ? "N/A" : QString::fromStdString(v.sourceVersionId),
            QString::fromStdString(m_versionStageNames[row]),
            ForgeGUI::FormatTimesteps(v.latestTimesteps) + (v.latestIsAutosave ? " (autosave)" : ""),
            QString::fromStdString(BranchNameOrDefault(v.branchName)),
            FormatVersionStatus(importedModelFolderRow, v),
            activeTrainingRow ? "Training" : FormatTimestamp(v.lastTrainedAt),
            v.hasSkillRating ? QString::fromStdString(ratingStream.str()) : QString(),
            QString::number(static_cast<qulonglong>(v.modelCount)),
            v.eligibleRLBot ? "Yes" : "No",
            QString(),
            FormatNotesPreview(notesText, m_table)
        };

        for (int col = 0; col < values.size(); col++) {
            auto* item = new QTableWidgetItem(values[col]);
            if (col == notesColumn) {
                item->setTextAlignment(Qt::AlignLeft | Qt::AlignVCenter);
                item->setToolTip(notesText);
            } else {
                item->setTextAlignment(Qt::AlignCenter);
            }
            if (!v.available) {
                item->setBackground(ForgeGUI::Theme::SURFACE_SUNKEN);
                item->setForeground(ForgeGUI::Theme::TEXT_DISABLED);
                item->setData(PreserveSelectedForegroundRole, true);
            }
            if (col == lastTrainedColumn && activeTrainingRow) {
                item->setBackground(ForgeGUI::BlendColor(ForgeGUI::Theme::SURFACE_PANEL, ForgeGUI::Theme::NEGATIVE, 0.25));
                item->setForeground(ForgeGUI::Theme::NEGATIVE_BRIGHT);
                item->setData(PreserveSelectedForegroundRole, true);
            }
            m_table->setItem(row, col, item);
        }

        auto* setupBtn = new QPushButton("Training Setup", m_table);
        setupBtn->setProperty("versionId", QString::fromStdString(v.id));
        setupBtn->setProperty("row", row);
        setupBtn->setProperty("buildName", QString::fromStdString(m_versionBuildNames[row]));
        setupBtn->setProperty("stageName", QString::fromStdString(m_versionStageNames[row]));
        setupBtn->setEnabled(!importedModelFolderRow && HasTrainingSetupSnapshotFile(v));
        if (!v.available)
            setupBtn->setStyleSheet(QString("color: %1;").arg(ForgeGUI::Theme::TEXT_DISABLED.name()));
        if (!setupBtn->isEnabled()) {
            setupBtn->setToolTip(importedModelFolderRow
                ? "Training setup snapshots are not available for imported models."
                : "Training setup snapshot not available for this version.");
        }
        connect(setupBtn, &QPushButton::clicked, this, &VersionManagementDialog::OpenTrainingSetup);
        m_table->setCellWidget(row, trainingSetupColumn, setupBtn);
    }

    ApplyRowFilters();

    for (int col = 0; col < m_table->columnCount(); col++) {
        if (col == notesColumn)
            continue;
        m_table->resizeColumnToContents(col);
    }
    m_table->setColumnWidth(notesColumn, VersionNotesColumnWidth);

    m_table->clearSelection();
    bool restoredSelection = false;
    if (!previousSelectedKeys.empty() && m_table->selectionModel()) {
        QItemSelection selection;
        for (int row = 0; row < (int)m_versions.size(); row++) {
            if (m_table->isRowHidden(row) || !previousSelectedKeys.contains(RowKey(row)))
                continue;

            selection.select(
                m_table->model()->index(row, 0),
                m_table->model()->index(row, m_table->columnCount() - 1));
            restoredSelection = true;
        }
        if (restoredSelection) {
            m_table->selectionModel()->select(
                selection,
                QItemSelectionModel::Select | QItemSelectionModel::Rows);
        }
    }

    int selectedRow = restoredSelection ? SelectedRow() : -1;
    if (selectedRow < 0 && !m_versions.empty())
        selectedRow = 0;
    if (selectedRow >= 0 && m_table->isRowHidden(selectedRow))
        selectedRow = -1;
    if (selectedRow < 0) {
        for (int row = 0; row < m_table->rowCount(); row++) {
            if (!m_table->isRowHidden(row)) {
                selectedRow = row;
                break;
            }
        }
    }
    if (selectedRow >= 0)
        m_table->selectRow(selectedRow);
    else
        m_table->clearSelection();
    UpdateDialogWidth();
    UpdateActionButtons();
}

void VersionManagementDialog::DeleteVersion()
{
    const std::vector<int> rows = SelectedRows();
    if (rows.empty()) {
        QMessageBox::information(this, "File Management", "Select a version first.");
        return;
    }
    for (int row : rows) {
        if (!CanMutateRow(row))
            return;
    }

    const bool hasImportedModelRows = std::any_of(rows.begin(), rows.end(), [this](int row) {
        return IsImportedModelFolderRow(row);
    });
    const bool allImportedModelRows = std::all_of(rows.begin(), rows.end(), [this](int row) {
        return IsImportedModelFolderRow(row);
    });

    QString prompt;
    if (rows.size() == 1 && allImportedModelRows) {
        prompt = "Delete imported model " + QString::fromStdString(m_versions[rows.front()].id)
            + "?\n\nThis permanently deletes that imported model folder.";
    } else if (rows.size() == 1) {
        prompt = "Delete segment " + QString::fromStdString(m_versions[rows.front()].id)
            + "?\n\nThis permanently deletes that segment folder.";
    } else if (hasImportedModelRows) {
        prompt = QString("Delete %1 selected items?\n\nThis permanently deletes those segment and imported model folders.")
            .arg(rows.size());
    } else {
        prompt = QString("Delete %1 selected segments?\n\nThis permanently deletes those segment folders.")
            .arg(rows.size());
    }

    if (QMessageBox::warning(
            this,
            "Delete Version",
            prompt,
            QMessageBox::Yes | QMessageBox::Cancel,
            QMessageBox::Cancel) != QMessageBox::Yes)
        return;

    for (int row : rows) {
        if (IsImportedModelFolderRow(row)) {
            const std::filesystem::path importedModelRoot = m_importedModelRootPaths[row];
            try {
                if (!importedModelRoot.empty() && std::filesystem::exists(importedModelRoot))
                    std::filesystem::remove_all(importedModelRoot);
            } catch (const std::exception& e) {
                QMessageBox::critical(this, "Delete Version", QString::fromStdString(e.what()));
                return;
            } catch (...) {
                QMessageBox::critical(this, "Delete Version", "Failed to delete imported model folder.");
                return;
            }
            continue;
        }

        const std::string versionId = m_versions[row].id;
        WithStageForRow(row, [&](RocketForge::TrainingStage* stage) {
            RocketForgeTraining::DeleteVersion(stage, versionId);
        });
    }
    RefreshPreservingColumnWidths();
}

void VersionManagementDialog::PruneOldModels()
{
    std::vector<int> rows = SelectedRows();
    if (rows.empty()) {
        QMessageBox::information(this, "File Management", "Select a version first.");
        return;
    }
    rows.erase(
        std::remove_if(rows.begin(), rows.end(), [this](int row) {
            return row < 0 || row >= (int)m_versions.size() || m_versions[row].modelCount <= 1;
        }),
        rows.end());
    if (rows.empty())
        return;
    for (int row : rows) {
        if (!CanMutateRow(row))
            return;
    }

    const QString prompt = rows.size() == 1
        ? "Delete all but the latest saved model for " + QString::fromStdString(m_versions[rows.front()].id) + "?"
        : QString("Delete all but the latest saved model for %1 selected segments?").arg(rows.size());

    if (QMessageBox::warning(
            this,
            "Prune Old Models",
            prompt,
            QMessageBox::Yes | QMessageBox::Cancel,
            QMessageBox::Cancel) != QMessageBox::Yes)
        return;

    for (int row : rows) {
        const std::string versionId = m_versions[row].id;
        WithStageForRow(row, [&](RocketForge::TrainingStage* stage) {
            RocketForgeTraining::PruneVersionModels(stage, versionId);
        });
    }
    RefreshPreservingColumnWidths();
}

void VersionManagementDialog::ImportExternalModel()
{
    ImportModelDialog dialog(CurrentAgentName(), this);
    if (dialog.exec() != QDialog::Accepted)
        return;

    const RocketForgeTraining::ImportedModelImportConfig config = dialog.ImportConfig();
    const QString agentName = QString::fromStdString(config.targetAgentName);

    try {
        const RocketForgeTraining::ImportedModelImportResult result = RocketForgeTraining::ImportExternalModel(config);

        const std::optional<bool> fitsAgent = dialog.ModelFitsAgent();
        QString usage = "It's listed here as an imported model, and can be picked as a static opponent, "
                        "in Model Benchmark and in Debug render.";
        if (fitsAgent == false) {
            usage = QString("It doesn't load into %1 yet. Once the agent matches the model, "
                            "it can be picked as a static opponent, in Model Benchmark and in Debug render.").arg(agentName);
        } else if (!fitsAgent.has_value()) {
            usage += QString("\n\nIt wasn't confirmed that it loads into %1; Debug render is a quick way to check.").arg(agentName);
        }
        const QString displayName = QString::fromStdString(result.metadata.displayName);
        const QString importedModelRoot = QString::fromStdWString(result.importedModelRoot.wstring());
        const QString message = QString("Imported \"%1\" for %2.\n\n%3\n\n%4").arg(displayName, agentName, usage, importedModelRoot);
        QMessageBox::information(this, "Import Model", message);

        const int agentIndex = m_agentCombo->findText(agentName);
        if (agentIndex >= 0)
            m_agentCombo->setCurrentIndex(agentIndex);
        RefreshPreservingColumnWidths();
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Import Model", QString::fromStdString(e.what()));
    } catch (...) {
        QMessageBox::critical(this, "Import Model", "Unknown model import error.");
    }
}

void VersionManagementDialog::ExportRLBot()
{
    const std::vector<int> rows = SelectedRows();
    if (rows.empty()) {
        QMessageBox::information(this, "Export to RLBot", "Select a version first.");
        return;
    }
    for (int row : rows) {
        if (!CanMutateRow(row))
            return;
    }
    if (std::any_of(rows.begin(), rows.end(), [this](int row) { return IsImportedModelFolderRow(row); })) {
        QMessageBox::information(this, "Export to RLBot", "RLBot export is only available for training segments.");
        return;
    }

    const bool exportToRLBot = std::any_of(rows.begin(), rows.end(), [this](int row) {
        return row >= 0 && row < (int)m_versions.size() && !m_versions[row].eligibleRLBot;
    });
    try {
        const std::string agentName = CurrentAgentName().toStdString();
        for (int row : rows) {
            if (m_versions[row].eligibleRLBot == exportToRLBot)
                continue;

            const std::string versionId = m_versions[row].id;
            const std::string buildName = m_versionBuildNames[row];
            if (exportToRLBot)
                RocketForgeTraining::ExportVersionToRLBot(agentName, buildName, versionId);
            else
                RocketForgeTraining::RemoveVersionFromRLBot(agentName, buildName, versionId);
        }
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Export to RLBot", QString::fromStdString(e.what()));
        return;
    } catch (...) {
        QMessageBox::critical(this, "Export to RLBot", "Unknown RLBot export error.");
        return;
    }
    RefreshPreservingColumnWidths();
}

void VersionManagementDialog::ExportForSubmittal()
{
    const std::vector<int> rows = SelectedRows();
    if (rows.size() != 1) {
        QMessageBox::information(this, "Export for Submittal", "Select one training segment first.");
        return;
    }

    const int row = rows.front();
    if (row < 0 || row >= (int)m_versions.size() || row >= (int)m_versionBuildNames.size())
        return;
    if (IsImportedModelFolderRow(row)) {
        QMessageBox::information(
            this,
            "Export for Submittal",
            "Submittal export is only available for saved training segments.");
        return;
    }
    if (m_versions[row].modelCount == 0) {
        QMessageBox::information(
            this,
            "Export for Submittal",
            "The selected segment does not have a full model checkpoint to package.");
        return;
    }

    const std::string agentName = CurrentAgentName().toStdString();
    const std::string buildName = m_versionBuildNames[row];
    const std::string versionId = m_versions[row].id;
    RocketForgeTraining::SubmittalBotMetadata defaultMetadata;
    try {
        defaultMetadata = RocketForgeTraining::GetDefaultSubmittalBotMetadata(
            agentName,
            buildName,
            versionId);
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Export for Submittal", QString::fromStdString(e.what()));
        return;
    }

    const std::optional<SubmittalExportDialogResult> exportOptions =
        PromptForSubmittalExport(this, defaultMetadata);
    if (!exportOptions.has_value())
        return;

    const std::filesystem::path& outputParent = exportOptions->outputParent;
    const std::filesystem::path packageRoot = outputParent /
        RocketForgeTraining::MakeSubmittalExportFolderName(exportOptions->metadata.name);

    bool overwrite = false;
    if (std::filesystem::exists(packageRoot)) {
        const QMessageBox::StandardButton answer = QMessageBox::question(
            this,
            "Replace Existing Submittal Export?",
            "This package already exists:\n\n" + QString::fromStdWString(packageRoot.wstring()) +
                "\n\nReplace it with a fresh export?",
            QMessageBox::Yes | QMessageBox::No,
            QMessageBox::No);
        if (answer != QMessageBox::Yes)
            return;
        overwrite = true;
    }

    QApplication::setOverrideCursor(Qt::WaitCursor);
    try {
        const RocketForgeTraining::SubmittalExportResult result =
            RocketForgeTraining::ExportVersionForSubmittal(
                agentName,
                buildName,
                versionId,
                outputParent,
                exportOptions->metadata,
                overwrite);
        QApplication::restoreOverrideCursor();
        QMessageBox::information(
            this,
            "Export for Submittal",
            "Created a Bob-compatible submittal package:\n\n" +
                QString::fromStdWString(result.packageRoot.wstring()) +
                "\n\nRun `python scripts/build_bob.py` from that folder. The generated bot does not require RocketForge.");
    } catch (const std::exception& e) {
        QApplication::restoreOverrideCursor();
        QMessageBox::critical(this, "Export for Submittal", QString::fromStdString(e.what()));
    } catch (...) {
        QApplication::restoreOverrideCursor();
        QMessageBox::critical(this, "Export for Submittal", "Unknown submittal export error.");
    }
}

void VersionManagementDialog::EditNotes()
{
    const std::vector<int> rows = SelectedRows();
    if (rows.size() != 1) {
        QMessageBox::information(this, "Edit Version Notes", "Select one version first.");
        return;
    }
    const int row = rows.front();
    if (!CanMutateRow(row))
        return;

    const auto& version = m_versions[row];
    std::optional<QString> notes = PromptForVersionNotes(
        this,
        QString::fromStdString(version.id),
        QString::fromStdString(version.notes));
    if (!notes.has_value())
        return;

    if (IsImportedModelFolderRow(row)) {
        try {
            SaveImportedModelNotes(row, notes.value());
        } catch (const std::exception& e) {
            QMessageBox::critical(this, "Edit Version Notes", QString::fromStdString(e.what()));
            return;
        } catch (...) {
            QMessageBox::critical(this, "Edit Version Notes", "Failed to update imported model notes.");
            return;
        }
        RefreshPreservingColumnWidths();
        return;
    }

    WithStageForRow(row, [&](RocketForge::TrainingStage* stage) {
        RocketForgeTraining::SetVersionNotes(stage, version.id, notes->toStdString());
    });
    RefreshPreservingColumnWidths();
}

void VersionManagementDialog::MoveBranch()
{
    const std::vector<int> rows = SelectedRows();
    if (rows.empty()) {
        QMessageBox::information(this, "Move To Branch", "Select a version first.");
        return;
    }
    for (int row : rows) {
        if (!CanMutateRow(row))
            return;
    }
    if (std::any_of(rows.begin(), rows.end(), [this](int row) { return IsImportedModelFolderRow(row); })) {
        QMessageBox::information(this, "Move To Branch", "Branches are only available for training segments.");
        return;
    }

    std::set<std::string> branchSet;
    for (const GGL::SegmentVersionSummary& version : m_versions)
        branchSet.insert(version.branchName.empty() ? std::string("main") : version.branchName);
    std::vector<std::string> branches(branchSet.begin(), branchSet.end());

    std::optional<QString> branchName = PromptForBranchMove(this, branches);
    if (!branchName.has_value())
        return;

    const std::string targetBranch = branchName->trimmed().toStdString();
    for (int row : rows) {
        if (row < 0 || row >= (int)m_versions.size())
            continue;

        const std::string currentBranch = m_versions[row].branchName.empty() ? std::string("main") : m_versions[row].branchName;
        if (currentBranch == targetBranch)
            continue;

        const std::string versionId = m_versions[row].id;
        WithStageForRow(row, [&](RocketForge::TrainingStage* stage) {
            RocketForgeTraining::SetVersionBranch(stage, versionId, targetBranch);
        });
    }
    RefreshPreservingColumnWidths();
}

void VersionManagementDialog::ToggleActive()
{
    const std::vector<int> rows = SelectedRows();
    if (rows.empty()) {
        QMessageBox::information(this, "File Management", "Select a version first.");
        return;
    }
    for (int row : rows) {
        if (!CanMutateRow(row))
            return;
    }
    if (std::any_of(rows.begin(), rows.end(), [this](int row) { return IsImportedModelFolderRow(row); })) {
        QMessageBox::information(this, "File Management", "Active status is only available for training segments.");
        return;
    }

    const bool makeActive = std::any_of(rows.begin(), rows.end(), [this](int row) {
        return row >= 0 && row < (int)m_versions.size() && !m_versions[row].available;
    });
    for (int row : rows) {
        if (m_versions[row].available == makeActive)
            continue;

        const std::string versionId = m_versions[row].id;
        WithStageForRow(row, [&](RocketForge::TrainingStage* stage) {
            RocketForgeTraining::SetVersionActive(stage, versionId, makeActive);
        });
    }
    RefreshPreservingColumnWidths();
}

void VersionManagementDialog::OpenTrainingSetup()
{
    auto* setupBtn = qobject_cast<QPushButton*>(sender());
    const std::string versionId = setupBtn
        ? setupBtn->property("versionId").toString().toStdString()
        : SelectedVersionId();
    if (versionId.empty()) {
        QMessageBox::information(this, "File Management", "Select a version first.");
        return;
    }

    const std::string buttonBuildName = setupBtn
        ? setupBtn->property("buildName").toString().toStdString()
        : std::string();
    const std::string buttonStageName = setupBtn
        ? setupBtn->property("stageName").toString().toStdString()
        : std::string();
    bool rowPropertyOk = false;
    int row = setupBtn ? setupBtn->property("row").toInt(&rowPropertyOk) : SelectedRow();

    auto rowMatches = [&](int candidate) {
        if (candidate < 0 || candidate >= (int)m_versions.size())
            return false;
        if (m_versions[candidate].id != versionId)
            return false;
        if (!buttonBuildName.empty() && m_versionBuildNames[candidate] != buttonBuildName)
            return false;
        if (!buttonStageName.empty() && m_versionStageNames[candidate] != buttonStageName)
            return false;
        return true;
    };

    if (!rowPropertyOk && setupBtn)
        row = -1;
    if (!rowMatches(row)) {
        for (int i = 0; i < (int)m_versions.size(); i++) {
            if (rowMatches(i)) {
                row = i;
                break;
            }
        }
    }

    if (!rowMatches(row)) {
        QMessageBox::information(this, "File Management", "Select a version first.");
        return;
    }
    if (IsImportedModelFolderRow(row)) {
        QMessageBox::information(this, "Training Setup", "Training setup snapshots are not available for imported models.");
        return;
    }

    WithStageForRow(row, [&](RocketForge::TrainingStage* stage) {
        std::optional<RocketForge::TrainingSetupSnapshot> snapshot =
            RocketForgeTraining::LoadTrainingSetupSnapshot(stage, versionId);
        if (!snapshot.has_value()) {
            QMessageBox::information(this, "Training Setup", "No saved training setup was found for that version.");
            return;
        }

        // Close any existing window. WA_DeleteOnClose will destroy it and the
        // destroyed() connection below will null m_trainingSetupWindow.
        if (m_trainingSetupWindow)
            m_trainingSetupWindow->close();

        // Resolve the checkpoint subfolder and the version root for saving/loading cached results.
        const std::filesystem::path versionRoot = std::filesystem::path(m_versions[row].rootPath);
        const std::filesystem::path checkpointRoot = versionRoot / "checkpoints";
        const std::filesystem::path checkpointPath = GGL::TrainingStorage::GetMostRecentCheckpointIn(checkpointRoot);
        const QString checkpointPathStr = QString::fromStdString(checkpointPath.string());
        const QString versionPathStr    = QString::fromStdString(versionRoot.string());

        m_trainingSetupWindow = new TrainingSetupWindow(
            snapshot.value(),
            QString::fromStdString(versionId),
            checkpointPathStr,
            versionPathStr,
            row >= 0 && row < (int)m_versionBuildNames.size()
                ? QString::fromStdString(m_versionBuildNames[row])
                : CurrentBuildName(),
            this);
        // Free the window when the user closes it via the OS close button.
        m_trainingSetupWindow->setAttribute(Qt::WA_DeleteOnClose);
        // Null our pointer when Qt destroys the object (via WA_DeleteOnClose or parent deletion).
        connect(m_trainingSetupWindow, &QObject::destroyed, this, [this]() {
            m_trainingSetupWindow = nullptr;
        });
        m_trainingSetupWindow->setWindowTitle(
            QString("Training Setup - %1 / %2 / %3")
                .arg(
                    CurrentAgentName(),
                    row >= 0 && row < (int)m_versionStageNames.size()
                        ? QString::fromStdString(m_versionStageNames[row])
                        : CurrentStageName(),
                    QString::fromStdString(versionId)));
        m_trainingSetupWindow->show();
        m_trainingSetupWindow->raise();
    });
}

void VersionManagementDialog::OnSelectionChanged()
{
    UpdateActionButtons();
}

void VersionManagementDialog::OnContextChanged()
{
    if (sender() == m_agentCombo) {
        m_pendingBuild.clear();
        m_catalog = ModelCatalog::Instance().Get(CurrentAgentName());
        RefreshBuilds();
        RefreshStages();
    } else if (sender() == m_buildCombo) {
        m_pendingBuild.clear();
        RefreshStages();
    }

    UpdateContextLabel();

    const std::vector<int> widths = CaptureColumnWidths();
    RefreshFromCatalog();
    RestoreColumnWidths(widths);
}

void VersionManagementDialog::ApplyRowFilters()
{
    if (!m_table)
        return;

    const bool hideInactive = m_hideInactiveCheck && m_hideInactiveCheck->isChecked();
    const QString searchText = m_searchEdit ? m_searchEdit->text().trimmed() : QString();
    const QString branchFilter = CurrentBranchName();
    const bool hasSearchText = !searchText.isEmpty();
    const bool hasBranchFilter = !branchFilter.isEmpty() && branchFilter != AllBranchesLabel;
    int selectedRow = SelectedRow();
    bool selectedStillVisible = false;

    auto textMatchesSearch = [&](const QString& text) {
        return text.contains(searchText, Qt::CaseInsensitive);
    };

    for (int row = 0; row < m_table->rowCount(); row++) {
        bool hideRow = false;
        if (hideInactive && row >= 0 && row < (int)m_versions.size() && !m_versions[row].available)
            hideRow = true;

        if (!hideRow && hasBranchFilter && row >= 0 && row < (int)m_versions.size()) {
            const QString rowBranch = QString::fromStdString(BranchNameOrDefault(m_versions[row].branchName));
            hideRow = rowBranch != branchFilter;
        }

        if (!hideRow && hasSearchText) {
            bool rowMatches = false;
            for (int col = 0; col < m_table->columnCount() && !rowMatches; col++) {
                if (QTableWidgetItem* item = m_table->item(row, col)) {
                    rowMatches = textMatchesSearch(item->text())
                        || textMatchesSearch(item->toolTip());
                }

                if (!rowMatches) {
                    if (QPushButton* button = qobject_cast<QPushButton*>(m_table->cellWidget(row, col)))
                        rowMatches = textMatchesSearch(button->text()) || textMatchesSearch(button->toolTip());
                }
            }
            hideRow = !rowMatches;
        }

        m_table->setRowHidden(row, hideRow);
        if (row == selectedRow && !hideRow)
            selectedStillVisible = true;
    }

    if (selectedRow >= 0 && !selectedStillVisible) {
        m_table->clearSelection();
        selectedRow = -1;
    }

    if (selectedRow < 0) {
        for (int row = 0; row < m_table->rowCount(); row++) {
            if (!m_table->isRowHidden(row)) {
                m_table->selectRow(row);
                break;
            }
        }
    }

    UpdateActionButtons();
}

void VersionManagementDialog::UpdateDialogWidth()
{
    if (!m_table)
        return;

    QHeaderView* header = m_table->horizontalHeader();
    const int notesColumn = m_sortController.UiIndexFromPriorityColumn(Notes);
    for (int col = 0; col < m_table->columnCount(); ++col) {
        if (col == notesColumn)
            continue;
        m_table->resizeColumnToContents(col);
    }
    m_table->setColumnWidth(notesColumn, VersionNotesColumnWidth);

    if (header)
        header->resizeSection(notesColumn, VersionNotesColumnWidth);
}

std::vector<int> VersionManagementDialog::CaptureColumnWidths() const
{
    std::vector<int> columnWidths;
    if (!m_table)
        return columnWidths;

    columnWidths.reserve(m_table->columnCount());
    for (int col = 0; col < m_table->columnCount(); col++)
        columnWidths.push_back(m_table->columnWidth(col));
    return columnWidths;
}

void VersionManagementDialog::RestoreColumnWidths(const std::vector<int>& columnWidths)
{
    if (!m_table || (int)columnWidths.size() != m_table->columnCount())
        return;

    for (int col = 0; col < m_table->columnCount(); col++)
        m_table->setColumnWidth(col, columnWidths[col]);
}

void VersionManagementDialog::RefreshPreservingColumnWidths()
{
    const std::vector<int> columnWidths = CaptureColumnWidths();
    Refresh();
    RestoreColumnWidths(columnWidths);
}

void VersionManagementDialog::AddImportedModelRows(
    const std::string& agentName,
    const std::string& buildFilter,
    const std::string& stageFilter)
{
    try {
        std::set<std::string> seenFolderNames = {};
        for (const std::filesystem::path& importedRoot : RocketForgeTraining::GetAgentImportedModelSearchFolders(agentName)) {
            if (importedRoot.empty() || !std::filesystem::exists(importedRoot))
                continue;

            for (const auto& entry : std::filesystem::directory_iterator(importedRoot)) {
                if (!entry.is_directory())
                    continue;

                const std::string folderName = entry.path().filename().string();
                if (IsInternalImportedModelFolderName(folderName) || seenFolderNames.contains(folderName))
                    continue;
                seenFolderNames.insert(folderName);

                const std::optional<RocketForgeTraining::ImportedModelMetadata> metadata =
                    RocketForgeTraining::LoadImportedModelMetadata(entry.path());
                const std::string sourceBuildName = metadata ? metadata->sourceBuildName : std::string();
                if (!buildFilter.empty() && sourceBuildName != buildFilter)
                    continue;

                const std::string sourceStageName = ImportedModelStageName(metadata);
                if (!stageFilter.empty() && sourceStageName != stageFilter)
                    continue;

                const std::filesystem::path modelFolder = GGL::TrainingStorage::ResolveModelFolder(entry.path());

                GGL::SegmentVersionSummary summary = {};
                summary.id = metadata && !metadata->displayName.empty()
                    ? metadata->displayName
                    : folderName;
                summary.type = "imported_model";
                summary.notes = metadata ? metadata->notes : std::string("Imported model folder");
                summary.createdAt = metadata ? metadata->importedAt : std::string();
                summary.updatedAt = metadata ? metadata->importedAt : std::string();
                summary.lastTrainedAt = metadata ? metadata->importedAt : std::string();
                summary.rootPath = entry.path().string();
                summary.buildName = sourceBuildName;
                summary.sourceVersionId = metadata && !metadata->sourceVersionId.empty()
                    ? metadata->sourceVersionId
                    : std::string("Imported");
                summary.branchName = "N/A";
                summary.stageName = sourceStageName;
                summary.available = !modelFolder.empty();
                summary.latestTimesteps = ImportedModelTimesteps(metadata, modelFolder);
                summary.modelCount = modelFolder.empty() ? 0 : 1;

                m_versionBuildNames.push_back(sourceBuildName.empty() ? "Imported" : sourceBuildName);
                m_versionStageNames.push_back(sourceStageName);
                m_versionIsImportedModelFolder.push_back(true);
                m_importedModelRootPaths.push_back(entry.path());
                m_versions.push_back(std::move(summary));
            }
        }
    } catch (const std::exception&) {
        return;
    }
}

void VersionManagementDialog::SaveImportedModelNotes(int row, const QString& notes)
{
    if (!IsImportedModelFolderRow(row))
        return;

    const std::filesystem::path importedModelRoot = m_importedModelRootPaths[row];
    const std::string folderName = importedModelRoot.filename().string();
    RocketForgeTraining::ImportedModelMetadata metadata =
        RocketForgeTraining::LoadImportedModelMetadata(importedModelRoot).value_or(RocketForgeTraining::ImportedModelMetadata{});

    if (metadata.folderName.empty())
        metadata.folderName = folderName;
    if (metadata.displayName.empty())
        metadata.displayName = m_versions[row].id.empty() ? folderName : m_versions[row].id;
    if (metadata.targetAgentName.empty())
        metadata.targetAgentName = CurrentAgentName().toStdString();
    if (metadata.sourceType.empty())
        metadata.sourceType = "external";
    if (metadata.sourceVersionId.empty())
        metadata.sourceVersionId = folderName;
    if (metadata.importedAt.empty())
        metadata.importedAt = RocketForgeTraining::BuildStorage::MakeTimestamp();

    metadata.notes = notes.toStdString();
    RefreshImportedModelMetadataFromFolder(metadata, importedModelRoot);
    RocketForgeTraining::SaveImportedModelMetadata(importedModelRoot, metadata);
}

std::string VersionManagementDialog::SelectedVersionId() const
{
    const int row = SelectedRow();
    if (row < 0)
        return {};
    QTableWidgetItem* item = m_table->item(row, m_sortController.UiIndexFromPriorityColumn(Segment));
    return item ? item->text().toStdString() : std::string();
}

std::vector<int> VersionManagementDialog::SelectedRows() const
{
    std::vector<int> rows;
    if (!m_table || !m_table->selectionModel())
        return rows;

    const QModelIndexList selected = m_table->selectionModel()->selectedRows();
    rows.reserve(selected.size());
    for (const QModelIndex& index : selected) {
        const int row = index.row();
        if (row < 0 || row >= m_table->rowCount() || m_table->isRowHidden(row))
            continue;
        if (std::find(rows.begin(), rows.end(), row) == rows.end())
            rows.push_back(row);
    }

    std::sort(rows.begin(), rows.end());
    return rows;
}

int VersionManagementDialog::SelectedRow() const
{
    const std::vector<int> rows = SelectedRows();
    if (rows.empty())
        return -1;
    return rows.front();
}

QString VersionManagementDialog::CurrentAgentName() const
{
    return m_agentCombo ? m_agentCombo->currentText() : QString();
}

QString VersionManagementDialog::CurrentBuildName() const
{
    return m_buildCombo ? m_buildCombo->currentText() : QString(AllBuildsLabel);
}

QString VersionManagementDialog::CurrentStageName() const
{
    return m_stageCombo ? m_stageCombo->currentText() : QString();
}

QString VersionManagementDialog::CurrentBranchName() const
{
    return m_branchCombo ? m_branchCombo->currentText() : QString(AllBranchesLabel);
}

bool VersionManagementDialog::IsActiveTrainingRow(int row) const
{
    return m_trainingActive
        && !IsImportedModelFolderRow(row)
        && row >= 0
        && row < (int)m_versions.size()
        && !m_activeAgentName.isEmpty()
        && !m_activeBuildName.isEmpty()
        && !m_activeStageName.isEmpty()
        && !m_activeVersionId.isEmpty()
        && CurrentAgentName() == m_activeAgentName
        && QString::fromStdString(m_versionBuildNames[row]) == m_activeBuildName
        && QString::fromStdString(m_versionStageNames[row]) == m_activeStageName
        && QString::fromStdString(m_versions[row].id) == m_activeVersionId;
}

bool VersionManagementDialog::IsImportedModelFolderRow(int row) const
{
    return row >= 0
        && row < (int)m_versionIsImportedModelFolder.size()
        && m_versionIsImportedModelFolder[row];
}

bool VersionManagementDialog::CanMutateRow(int row)
{
    if (!IsActiveTrainingRow(row))
        return true;

    QMessageBox::information(
        this,
        "File Management",
        "The active training segment is protected while training is running.");
    return false;
}

int VersionManagementDialog::CompareVersionColumn(int lhsRow, int rhsRow, VersionColumn column) const
{
    const auto& a = m_versions[lhsRow];
    const auto& b = m_versions[rhsRow];

    switch (column) {
    case Build:
        return TableSortController::CompareStrings(
            QString::fromStdString(m_versionBuildNames[lhsRow]),
            QString::fromStdString(m_versionBuildNames[rhsRow]));
    case Segment: {
        int result = TableSortController::CompareValues(a.major, b.major);
        if (result == 0)
            result = TableSortController::CompareValues(a.minor, b.minor);
        if (result == 0)
            result = TableSortController::CompareStrings(
                QString::fromStdString(a.id), QString::fromStdString(b.id));
        return result;
    }
    case MadeFrom:
        return TableSortController::CompareStrings(
            QString::fromStdString(a.sourceVersionId), QString::fromStdString(b.sourceVersionId));
    case Stage:
        return TableSortController::CompareStrings(
            QString::fromStdString(m_versionStageNames[lhsRow]),
            QString::fromStdString(m_versionStageNames[rhsRow]));
    case LatestTimesteps:
        return TableSortController::CompareValues(a.latestTimesteps, b.latestTimesteps);
    case Branch:
        return TableSortController::CompareStrings(
            QString::fromStdString(BranchNameOrDefault(a.branchName)),
            QString::fromStdString(BranchNameOrDefault(b.branchName)));
    case Status:
        return TableSortController::CompareStrings(
            FormatVersionStatus(IsImportedModelFolderRow(lhsRow), a),
            FormatVersionStatus(IsImportedModelFolderRow(rhsRow), b));
    case LastTrained:
        return TableSortController::CompareStrings(
            IsActiveTrainingRow(lhsRow) ? "Training" : QString::fromStdString(a.lastTrainedAt),
            IsActiveTrainingRow(rhsRow) ? "Training" : QString::fromStdString(b.lastTrainedAt));
    case SkillRating: {
        int result = TableSortController::CompareValues(a.hasSkillRating ? 1 : 0, b.hasSkillRating ? 1 : 0);
        if (result == 0 && a.hasSkillRating && b.hasSkillRating)
            result = TableSortController::CompareValues(a.skillRating, b.skillRating);
        return result;
    }
    case ModelCount:
        return TableSortController::CompareValues(a.modelCount, b.modelCount);
    case RLBot:
        return TableSortController::CompareStrings(a.eligibleRLBot ? "Yes" : "No", b.eligibleRLBot ? "Yes" : "No");
    case TrainingSetup:
        return TableSortController::CompareValues(
            HasTrainingSetupSnapshotFile(a) ? 1 : 0, HasTrainingSetupSnapshotFile(b) ? 1 : 0);
    case Notes:
        return TableSortController::CompareStrings(
            QString::fromStdString(a.notes), QString::fromStdString(b.notes));
    case Unselected:
    case Count:
        break;
    }

    return 0;
}

std::string VersionManagementDialog::RowKey(int row) const
{
    if (row < 0 || row >= (int)m_versions.size())
        return {};

    if (IsImportedModelFolderRow(row)) {
        const std::string rootPath = row < (int)m_importedModelRootPaths.size()
            ? m_importedModelRootPaths[row].string()
            : m_versions[row].rootPath;
        return CurrentAgentName().toStdString()
            + "|static|"
            + rootPath;
    }

    return CurrentAgentName().toStdString()
        + "|" + m_versionBuildNames[row]
        + "|" + m_versionStageNames[row]
        + "|" + m_versions[row].id;
}

bool VersionManagementDialog::IsViewingActiveTrainingStage() const
{
    return m_trainingActive
        && CurrentAgentName() == m_activeAgentName
        && CurrentBuildName() == m_activeBuildName
        && CurrentStageName() == m_activeStageName;
}

void VersionManagementDialog::UpdateContextLabel()
{
}

void VersionManagementDialog::RefreshBuilds()
{
    if (!m_buildCombo)
        return;

    const QString previous = m_buildCombo->currentText();
    QSignalBlocker blocker(m_buildCombo);
    m_buildCombo->clear();
    m_buildCombo->addItem(AllBuildsLabel);

    for (const RocketForgeTraining::BuildSummary& build : m_catalog->storage.builds)
        m_buildCombo->addItem(QString::fromStdString(build.buildName));

    const int pendingIndex = m_buildCombo->findText(m_pendingBuild);
    const int previousIndex = pendingIndex >= 0 ? pendingIndex : m_buildCombo->findText(previous);
    if (pendingIndex >= 0) m_pendingBuild.clear();
    if (previousIndex >= 0)
        m_buildCombo->setCurrentIndex(previousIndex);
    else
        m_buildCombo->setCurrentIndex(0);
}

void VersionManagementDialog::RefreshStages()
{
    if (!m_stageCombo)
        return;

    const QString previous = m_stageCombo->currentText();
    const std::string buildFilter = CurrentBuildName() == AllBuildsLabel ? std::string() : CurrentBuildName().toStdString();
    QSignalBlocker blocker(m_stageCombo);
    m_stageCombo->clear();
    m_stageCombo->addItem(AllStagesLabel);

    for (const std::string& stageName : m_catalog->TrainingStages(buildFilter))
        m_stageCombo->addItem(QString::fromStdString(stageName));

    const int previousIndex = m_stageCombo->findText(previous);
    if (previousIndex >= 0)
        m_stageCombo->setCurrentIndex(previousIndex);
}

void VersionManagementDialog::RefreshBranches()
{
    if (!m_branchCombo)
        return;

    const QString previous = m_branchCombo->currentText();
    std::set<std::string> branches;
    for (const GGL::SegmentVersionSummary& version : m_versions)
        branches.insert(BranchNameOrDefault(version.branchName));

    QSignalBlocker blocker(m_branchCombo);
    m_branchCombo->clear();
    m_branchCombo->addItem(AllBranchesLabel);
    for (const std::string& branch : branches)
        m_branchCombo->addItem(QString::fromStdString(branch));

    const int previousIndex = m_branchCombo->findText(previous);
    m_branchCombo->setCurrentIndex(previousIndex >= 0 ? previousIndex : 0);
}

void VersionManagementDialog::UpdateActionButtons()
{
    const std::vector<int> rows = SelectedRows();
    const bool hasSelection = !rows.empty();
    const bool singleSelection = rows.size() == 1;
    bool containsActiveTrainingRow = false;
    bool containsImportedModelFolderRow = false;
    bool canPrune = false;
    bool allActive = hasSelection;
    bool allRLBot = hasSelection;
    for (int row : rows) {
        containsActiveTrainingRow = containsActiveTrainingRow || IsActiveTrainingRow(row);
        containsImportedModelFolderRow = containsImportedModelFolderRow || IsImportedModelFolderRow(row);
        if (row < 0 || row >= (int)m_versions.size())
            continue;

        canPrune = canPrune || (!IsImportedModelFolderRow(row) && m_versions[row].modelCount > 1);
        allActive = allActive && m_versions[row].available;
        allRLBot = allRLBot && m_versions[row].eligibleRLBot;
    }

    const bool allowMutations = hasSelection && !containsActiveTrainingRow;
    const bool allowSingleRowMutation = allowMutations && singleSelection;
    const bool allowSegmentMutations = allowMutations && !containsImportedModelFolderRow;
    const bool allowSubmittalExport = singleSelection &&
        !containsImportedModelFolderRow &&
        rows.front() >= 0 &&
        rows.front() < (int)m_versions.size() &&
        m_versions[rows.front()].modelCount > 0;

    if (m_importExternalModelBtn)
        m_importExternalModelBtn->setEnabled(!m_trainingActive);
    m_deleteBtn->setText(hasSelection && !singleSelection ? "Delete Selected" : "Delete");
    m_deleteBtn->setEnabled(allowMutations);
    m_pruneBtn->setEnabled(allowSegmentMutations && canPrune);
    m_editNotesBtn->setEnabled(allowSingleRowMutation);
    m_moveBranchBtn->setEnabled(allowSegmentMutations);
    m_activeBtn->setEnabled(allowSegmentMutations);
    m_rlbotBtn->setEnabled(allowSegmentMutations);
    m_submittalBtn->setEnabled(allowSubmittalExport);
    m_readOnlyLabel->setVisible(containsActiveTrainingRow);

    if (!hasSelection) {
        m_activeBtn->setText("Deactivate Segment");
        m_moveBranchBtn->setText("Move Branch...");
        m_rlbotBtn->setText("Export to RLBot");
        return;
    }

    m_moveBranchBtn->setText(singleSelection ? "Move Branch..." : "Move Selected Branches...");
    m_activeBtn->setText(allActive ? "Deactivate Segment" : "Activate Segment");
    m_rlbotBtn->setText(allRLBot ? "Remove from RLBot" : "Export to RLBot");
}

void VersionManagementDialog::WithStage(const std::function<void(RocketForge::TrainingStage*)>& fn)
{
    WithStageForRow(SelectedRow(), fn);
}

void VersionManagementDialog::WithStageForRow(int row, const std::function<void(RocketForge::TrainingStage*)>& fn)
{
    const std::string stageName = row >= 0 && row < (int)m_versionStageNames.size()
        ? m_versionStageNames[row]
        : (CurrentStageName() == AllStagesLabel ? std::string() : CurrentStageName().toStdString());
    if (stageName.empty()) {
        QMessageBox::information(this, "File Management", "Select a training stage first.");
        return;
    }

    auto stage = RocketForgeTraining::CreateTrainingStage(
        CurrentAgentName().toStdString(),
        stageName,
        RocketForgeTraining::GlobalSettings{},
        RocketForgeTraining::StageSettings{});
    if (!stage) {
        QMessageBox::critical(this, "File Management", "Failed to create training stage.");
        return;
    }
    if (row >= 0 && row < (int)m_versionBuildNames.size())
        stage->learnerConfig.runtime.buildName = m_versionBuildNames[row];

    try {
        fn(stage.get());
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "File Management", QString::fromStdString(e.what()));
    } catch (...) {
        QMessageBox::critical(this, "File Management", "Unknown version operation error.");
    }
}
