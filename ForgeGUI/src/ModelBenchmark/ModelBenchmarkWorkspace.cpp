#include "ModelBenchmarkWidget.h"
#include "ModelBenchmarkSavedResultsWidget.h"

#include <QCheckBox>
#include <QComboBox>
#include <QCryptographicHash>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFileInfo>
#include <QHeaderView>
#include <QInputDialog>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QScreen>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QTableWidget>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

using namespace RocketForgeTraining;

namespace {
    QString PresetContent(ModelBenchmarkSettings settings)
    {
        // Browsing another agent or filtering the picker does not edit a benchmark preset.
        settings.stageFilter.clear();
        settings.hideImportedModels = false;
        settings.presetName.clear();
        if (settings.reuseSelectedEngine) settings.reuseEngineVersions.clear();

        // Retired controls do not make a legacy preset appear modified on load.
        settings.matchSavedConditions = true;
        settings.runUntilMaxGames = true;
        settings.stopWhenRankingKnown = false;
        settings.betterConfidence = 0.95f;
        settings.eloRangeConfidence = 0.95f;
        settings.maxEloUncertaintyRange = 50.0f;
        settings.collectOptionalStats = true;

        for (ModelBenchmarkSelectedModelSettings& model : settings.selectedModels) {
            model.libraryModelId = -1;
            model.colorIndex = -1;
        }

        return QString::fromStdString(SerializeModelBenchmarkSettings(settings));
    }

    bool HasCheckpoint(const std::filesystem::path& path)
    {
        std::error_code error;
        return !path.empty() && std::filesystem::is_directory(path, error) && !error;
    }
}

void ModelBenchmarkWidget::ShowTestSetup()
{
    if (m_running) return;

    const ModelBenchmarkSettings previous = BuildPersistedSetup();
    m_editingSetup = true;
    ++m_workspaceGeneration;
    m_workspaceRefreshTimer->stop();
    m_setupSaveTimer->stop();
    UpdateInputState();

    const bool accepted = m_testSetupDialog->exec() == QDialog::Accepted;
    if (!accepted) {
        m_loadingPersistedSetup = true;
        ApplyPersistedSetup(previous);
        m_loadingPersistedSetup = false;
    }

    m_editingSetup = false;
    OnSetupChanged();
    UpdateInputState();
}

void ModelBenchmarkWidget::ShowModelPicker()
{
    if (m_running) return;

    RefreshWorkspace();

    m_savedModelPicker->setRowCount((int)m_libraryModels.size());
    for (int row = 0; row < (int)m_libraryModels.size(); ++row) {
        const ModelBenchmarkLibraryModel& model = m_libraryModels[row];
        m_savedModelPicker->setItem(row, 0, new QTableWidgetItem(model.displayName));
        m_savedModelPicker->setItem(row, 1, new QTableWidgetItem(model.versionId));
        m_savedModelPicker->setItem(row, 2, new QTableWidgetItem(model.availableLocally ? "Available locally" : "Results only · checkpoint unavailable"));
    }
    m_savedModelPicker->resizeColumnsToContents();

    EnsureVersionCatalog();
    m_modelPickerDialog->exec();
    RefreshWorkspace();
}

void ModelBenchmarkWidget::UpdateCompatibilityControls()
{
    const bool followEngine = m_reuseSelectedEngineCheck->isChecked();
    const int selectedEngine = m_engineCombo->currentData().toInt();

    for (const auto& [engine, check] : m_reuseEngineChecks) {
        if (followEngine) {
            const QSignalBlocker blocker(check);
            check->setChecked(engine == selectedEngine);
        }

        check->setEnabled(!followEngine);

        QPushButton* button = m_reuseEnvironmentButtons.at(engine);
        button->setEnabled(!followEngine && check->isChecked());

        const auto found = m_reuseEnvironmentKeys.find(engine);
        button->setText(found == m_reuseEnvironmentKeys.end() ? "All environments..."
            : QString("%1 environments selected...").arg(found->second.size()));
    }
}

QString ModelBenchmarkWidget::CompatibilitySummary() const
{
    if (m_reuseSelectedEngineCheck->isChecked())
        return "Saved games and results: " + m_engineCombo->currentText() + " (all environments)";

    QStringList selections;
    for (const auto& [engine, check] : m_reuseEngineChecks) {
        if (!check->isChecked()) continue;

        const auto found = m_reuseEnvironmentKeys.find(engine);
        selections << check->text() + (found == m_reuseEnvironmentKeys.end() ? " (all environments)"
            : QString(" (%1 selected environments)").arg(found->second.size()));
    }

    return "Saved games and results: " + (selections.isEmpty() ? QString("none") : selections.join("; "));
}

void ModelBenchmarkWidget::ShowCompatibilityEnvironments(int engine)
{
    QString error;
    const std::map<std::string, ModelBenchmarkLibraryEnvironment> catalog = m_savedResultsWidget->Library().ListEnvironments(&error);
    if (!error.isEmpty()) {
        QMessageBox::warning(m_testSetupDialog, "Unable to list environments", error);
        return;
    }

    QDialog dialog(m_testSetupDialog);
    dialog.setObjectName("benchmarkCompatibilityDialog");
    dialog.setWindowTitle("Compatibility: " + m_reuseEngineChecks.at(engine)->text());
    auto* layout = new QVBoxLayout(&dialog);

    auto* help = new QLabel("Choose which environments count toward the game target and appear in results. New games still use the simulation engine selected in Test Setup.", &dialog);
    help->setWordWrap(true);
    layout->addWidget(help);

    auto* all = new QCheckBox("Include all environments (including future ones)", &dialog);
    all->setObjectName("benchmarkAllEnvironments");
    const auto selection = m_reuseEnvironmentKeys.find(engine);
    all->setChecked(selection == m_reuseEnvironmentKeys.end());
    layout->addWidget(all);

    auto* search = new QLineEdit(&dialog);
    search->setPlaceholderText("Filter environments by build, physics, kickoff, or recorded settings...");
    layout->addWidget(search);

    auto* table = new QTableWidget(0, 2, &dialog);
    table->setObjectName("benchmarkCompatibilityEnvironments");
    table->setHorizontalHeaderLabels({ "Use", "Recorded environment" });
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    table->setSelectionMode(QAbstractItemView::SingleSelection);
    table->setAlternatingRowColors(true);
    table->verticalHeader()->hide();
    table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    table->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    table->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Expanding);
    layout->addWidget(table, 2);

    auto* details = new QPlainTextEdit(&dialog);
    details->setReadOnly(true);
    details->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Expanding);
    layout->addWidget(details, 1);

    std::set<std::string> listed;
    auto addEnvironment = [&](const std::string& key, const QString& label, const QString& description) {
        listed.insert(key);

        const int row = table->rowCount();
        table->insertRow(row);

        auto* use = new QTableWidgetItem;
        use->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable | Qt::ItemIsUserCheckable);
        const bool checked = selection == m_reuseEnvironmentKeys.end()
            || std::find(selection->second.begin(), selection->second.end(), key) != selection->second.end();
        use->setCheckState(checked ? Qt::Checked : Qt::Unchecked);
        use->setData(Qt::UserRole, QString::fromStdString(key));
        table->setItem(row, 0, use);

        auto* name = new QTableWidgetItem(label);
        name->setData(Qt::UserRole, description);
        name->setToolTip("<qt>" + label.toHtmlEscaped() + "</qt>");
        table->setItem(row, 1, name);
    };

    for (const auto& [key, environment] : catalog) {
        const ModelBenchmarkResult& source = environment.provenance;
        if (!source.rocketSimVersions.contains(engine)) continue;

        QString label = ModelBenchmarkEnvironmentLabel(environment);
        label += source.kickoffInferenceSettingsMixed ? " / mixed kickoff settings"
            : source.stochasticKickoff ? QString(" / stochastic kickoff (%1)").arg(source.kickoffTemperature)
            : " / deterministic kickoff";

        const QJsonDocument identity = QJsonDocument::fromJson(QByteArray::fromStdString(source.evaluationIdentity));
        const QString integration = source.integrationRevision.empty() ? QString()
            : "\nIntegration revision: " + QString::fromStdString(source.integrationRevision);
        const QString description = label + "\n\nPhysics revision: " + QString::fromStdString(source.engineRevision)
            + integration
            + "\n\nRecorded settings:\n" + (source.evaluationIdentity.empty() ? QString("Unrecorded")
                : identity.isNull() ? QString::fromStdString(source.evaluationIdentity)
                : QString::fromUtf8(identity.toJson(QJsonDocument::Indented)));
        addEnvironment(key, label, description);
    }

    // Preserve explicit choices when opening a different library or after historical data is removed.
    if (selection != m_reuseEnvironmentKeys.end()) {
        for (const std::string& key : selection->second) {
            if (!listed.contains(key)) {
                const QString digest = QString::fromLatin1(QCryptographicHash::hash(QByteArray::fromStdString(key), QCryptographicHash::Sha256).toHex().left(12));
                addEnvironment(key, "Saved selection unavailable in this library / " + digest, QString::fromStdString(key));
            }
        }
    }

    auto* selectionButtons = new QHBoxLayout;
    auto* selectAll = new QPushButton("Select All", &dialog);
    auto* selectNone = new QPushButton("Select None", &dialog);
    selectionButtons->addWidget(selectAll);
    selectionButtons->addWidget(selectNone);
    selectionButtons->addStretch();
    layout->addLayout(selectionButtons);

    auto setSelection = [&](Qt::CheckState state) {
        all->setChecked(false);
        for (int row = 0; row < table->rowCount(); ++row) table->item(row, 0)->setCheckState(state);
    };
    connect(selectAll, &QPushButton::clicked, &dialog, [&]() { setSelection(Qt::Checked); });
    connect(selectNone, &QPushButton::clicked, &dialog, [&]() { setSelection(Qt::Unchecked); });

    auto updateAll = [&]() {
        for (int row = 0; row < table->rowCount(); ++row) {
            QTableWidgetItem* use = table->item(row, 0);
            use->setFlags(all->isChecked() ? Qt::ItemIsSelectable
                : Qt::ItemIsSelectable | Qt::ItemIsEnabled | Qt::ItemIsUserCheckable);
            if (all->isChecked()) use->setCheckState(Qt::Checked);
        }
    };
    connect(all, &QCheckBox::toggled, &dialog, updateAll);
    updateAll();

    connect(table, &QTableWidget::currentCellChanged, &dialog, [&](int row, int, int, int) {
        details->setPlainText(row < 0 ? QString() : table->item(row, 1)->data(Qt::UserRole).toString());
    });
    connect(search, &QLineEdit::textChanged, &dialog, [&](const QString& text) {
        for (int row = 0; row < table->rowCount(); ++row)
            table->setRowHidden(row, !table->item(row, 1)->data(Qt::UserRole).toString().contains(text, Qt::CaseInsensitive));
    });

    if (table->rowCount()) table->setCurrentCell(0, 1);
    else details->setPlainText("No saved environments for this simulator yet. Include all environments will accept future saved games.");

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    layout->addWidget(buttons);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);

    dialog.resize(QSize(900, 650).boundedTo(screen()->availableGeometry().size() * 0.9));
    if (dialog.exec() != QDialog::Accepted) return;

    if (all->isChecked()) {
        m_reuseEnvironmentKeys.erase(engine);
    } else {
        std::vector<std::string>& keys = m_reuseEnvironmentKeys[engine];
        keys.clear();
        for (int row = 0; row < table->rowCount(); ++row)
            if (table->item(row, 0)->checkState() == Qt::Checked)
                keys.push_back(table->item(row, 0)->data(Qt::UserRole).toString().toStdString());
    }

    UpdateCompatibilityControls();
    OnSetupChanged();
}

ModelBenchmarkLibraryFilter ModelBenchmarkWidget::WorkspaceFilter() const
{
    ModelBenchmarkLibraryFilter filter;
    filter.restrictEnvironments = true;
    if (m_reuseSelectedEngineCheck->isChecked()) {
        filter.engineVersions.insert(m_engineCombo->currentData().toInt());
    } else {
        for (const auto& [engine, check] : m_reuseEngineChecks)
            if (check->isChecked()) filter.engineVersions.insert(engine);
        filter.environmentKeys = m_reuseEnvironmentKeys;
    }

    filter.modeMask = (m_mode1v1Check->isChecked() ? 1 : 0) | (m_mode2v2Check->isChecked() ? 2 : 0) | (m_mode3v3Check->isChecked() ? 4 : 0);

    return filter;
}

void ModelBenchmarkWidget::UpdateWorkspaceSummary()
{
    const std::filesystem::path libraryPath = m_savedResultsWidget->LibraryPath();
    if (m_savedResultsWidget->Library().IsOpen()) {
        const QString path = QString::fromStdString(libraryPath.string());
        const QString name = QFileInfo(path).fileName();
        m_activeLibraryLabel->setText(m_activeLibraryLabel->fontMetrics().elidedText(
            "Library: " + name, Qt::ElideMiddle, m_activeLibraryLabel->maximumWidth()));
        m_activeLibraryLabel->setToolTip("<qt>Active benchmark library:<br>"
            + QFileInfo(path).absoluteFilePath().toHtmlEscaped() + "</qt>");
    } else {
        m_activeLibraryLabel->setText("No library open");
        m_activeLibraryLabel->setToolTip("Open or create a benchmark library using the Library menu.");
    }

    QStringList modes;
    if (m_mode1v1Check->isChecked()) modes << "1v1";
    if (m_mode2v2Check->isChecked()) modes << "2v2";
    if (m_mode3v3Check->isChecked()) modes << "3v3";

    m_setupSummaryLabel->setText(QString("%1 · %2 · %3 %4 games per matchup per mode · %5")
        .arg(m_engineCombo->currentText(), modes.isEmpty() ? "No modes selected" : modes.join(" / "),
            m_scheduleModeCombo->currentIndex() == 0 ? "Fill to" : "Add",
            QString::number(m_maxGamesSpin->value()),
            "Deterministic opening"));
    m_setupSummaryLabel->setToolTip("<qt>Current test setup: physics engine, enabled team sizes, games per matchup and mode, and kickoff behavior. Change these with Test Setup.<br><br>Active library: "
        + QString::fromStdString(m_savedResultsWidget->LibraryPath().string()).toHtmlEscaped() + "</qt>");

    const bool modified = !m_savedPresetJson.isEmpty() && PresetContent(BuildPersistedSetup()) != m_savedPresetJson;
    m_presetStatusLabel->setText(m_presetName.isEmpty() ? "Working configuration" : modified ? "Unsaved changes" : "Preset saved");
}

void ModelBenchmarkWidget::RefreshPresetList()
{
    if (!m_presetCombo) return;

    const QSignalBlocker blocker(m_presetCombo);
    m_presetCombo->clear();
    m_presetCombo->addItem("Working configuration", QString());

    QString error;
    for (const ModelBenchmarkWorkspaceRecord& preset : m_savedResultsWidget->Library().ListPresets(&error)) {
        m_presetCombo->addItem(preset.label, preset.id);
        if (m_savedPresetJson.isEmpty() && preset.id == m_presetName) {
            ModelBenchmarkSettingsLoadResult saved = DeserializeModelBenchmarkSettings(preset.settingsJson.toStdString());
            if (saved.ok) {
                saved.value.libraryPath = m_savedResultsWidget->LibraryPath();
                m_savedPresetJson = PresetContent(saved.value);
            }
        }
    }

    const int current = m_presetCombo->findData(m_presetName);
    m_presetCombo->setCurrentIndex(std::max(0, current));
    m_renamePresetButton->setEnabled(current > 0);
    if (!error.isEmpty()) AppendConsoleLine(error);
}

void ModelBenchmarkWidget::ApplyWorkspaceSettings(const ModelBenchmarkSettings& settings)
{
    m_workspaceRefreshTimer->stop();
    m_setupSaveTimer->stop();

    m_loadingPersistedSetup = true;
    ApplyPersistedSetup(settings);
    m_loadingPersistedSetup = false;

    SortSelectedVersions();
    PopulateSelectedTable();
    UpdateAvailableSelection();
    m_coverageDirty = true;
    RefreshPresetList();
    RefreshWorkspace();
    UpdateInputState();
    OnSetupChanged();
}

void ModelBenchmarkWidget::LoadWorkspacePreset()
{
    if (m_running) return;

    const QString selected = m_presetCombo->currentData().toString();

    // Keep the active preset selected until saving and loading both succeed.
    // Saving refreshes the dropdown, so retain the requested destination separately.
    {
        const QSignalBlocker blocker(m_presetCombo);
        m_presetCombo->setCurrentIndex(std::max(0, m_presetCombo->findData(m_presetName)));
    }

    if (selected.isEmpty() || selected == m_presetName) return;

    for (const ModelBenchmarkWorkspaceRecord& preset : m_savedResultsWidget->Library().ListPresets()) {
        if (preset.id != selected) continue;

        ModelBenchmarkSettingsLoadResult decoded = DeserializeModelBenchmarkSettings(preset.settingsJson.toStdString());
        if (!decoded.ok) {
            QStringList errors;
            for (const std::string& error : decoded.errors) errors << QString::fromStdString(error);
            QMessageBox::warning(this, "Unable to load preset",
                QString("The saved configuration could not be read.\n\n") + errors.join("\n"));
            return;
        }

        if (!m_savedPresetJson.isEmpty() && PresetContent(BuildPersistedSetup()) != m_savedPresetJson) {
            const QMessageBox::StandardButton choice = QMessageBox::question(this, "Unsaved benchmark preset",
                QString("Save changes to \"%1\" before switching presets?").arg(m_presetName),
                QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel, QMessageBox::Save);
            if (choice == QMessageBox::Cancel) return;

            const bool saveAs = false;
            if (choice == QMessageBox::Save && !SaveWorkspacePreset(saveAs)) return;
        }

        decoded.value.libraryPath = m_savedResultsWidget->LibraryPath();
        decoded.value.presetName = preset.id.toStdString();
        ApplyWorkspaceSettings(decoded.value);
        m_savedPresetJson = PresetContent(BuildPersistedSetup());
        UpdateWorkspaceSummary();
        return;
    }
}

void ModelBenchmarkWidget::NewWorkspacePreset()
{
    if (m_running) return;

    bool accepted = false;
    const QString name = QInputDialog::getText(this, "New benchmark preset", "Preset name:",
        QLineEdit::Normal, QString(), &accepted).trimmed();
    if (!accepted || name.isEmpty()) return;

    QString error;
    const std::vector<ModelBenchmarkWorkspaceRecord> presets = m_savedResultsWidget->Library().ListPresets(&error);
    if (!error.isEmpty()) {
        QMessageBox::warning(this, "Unable to create preset", error);
        return;
    }

    for (const ModelBenchmarkWorkspaceRecord& preset : presets) {
        if (preset.id.compare(name, Qt::CaseInsensitive) == 0) {
            QMessageBox::information(this, "Preset name already used", "Choose another name for the new preset.");
            return;
        }
    }

    const bool unsaved = m_savedPresetJson.isEmpty() ? !m_selectedVersions.empty()
        : PresetContent(BuildPersistedSetup()) != m_savedPresetJson;
    if (unsaved) {
        const QMessageBox::StandardButton choice = QMessageBox::question(this, "Unsaved benchmark preset",
            "Save your current configuration before creating a new preset?",
            QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel, QMessageBox::Save);
        if (choice == QMessageBox::Cancel) return;

        const bool saveAs = false;
        if (choice == QMessageBox::Save && !SaveWorkspacePreset(saveAs)) return;
    }

    ModelBenchmarkSettings settings;
    settings.libraryPath = m_savedResultsWidget->LibraryPath();
    settings.presetName = name.toStdString();

    const QString settingsJson = QString::fromStdString(SerializeModelBenchmarkSettings(settings));
    if (!m_savedResultsWidget->Library().CreatePreset(name, settingsJson, &error)) {
        QMessageBox::warning(this, "Unable to create preset", error);
        return;
    }

    m_savedPresetJson.clear();
    ApplyWorkspaceSettings(settings);
    m_savedPresetJson = PresetContent(BuildPersistedSetup());
    UpdateWorkspaceSummary();
}

void ModelBenchmarkWidget::RenameWorkspacePreset()
{
    if (m_running || m_presetName.isEmpty() || m_presetCombo->currentData().toString() != m_presetName) return;

    bool accepted = false;
    const QString name = QInputDialog::getText(this, "Rename benchmark preset", "Preset name:",
        QLineEdit::Normal, m_presetName, &accepted).trimmed();
    if (!accepted || name.isEmpty() || name == m_presetName) return;

    QString error;
    if (!m_savedResultsWidget->Library().RenamePreset(m_presetName, name, &error)) {
        QMessageBox::warning(this, "Unable to rename preset", error);
        return;
    }

    m_presetName = name;
    RefreshPresetList();
    OnSetupChanged();
}

bool ModelBenchmarkWidget::SaveWorkspacePreset(bool saveAs)
{
    if (m_running) return false;

    QString name = m_presetName;
    if (saveAs || name.isEmpty()) {
        bool accepted = false;
        name = QInputDialog::getText(this, "Save benchmark preset", "Preset name:", QLineEdit::Normal,
            saveAs && !name.isEmpty() ? name + " copy" : name, &accepted).trimmed();
        if (!accepted || name.isEmpty()) return false;

        for (const ModelBenchmarkWorkspaceRecord& preset : m_savedResultsWidget->Library().ListPresets()) {
            if (preset.id.compare(name, Qt::CaseInsensitive) == 0) {
                QMessageBox::information(this, "Preset name already used", "Choose another name, or load the existing preset and use Save.");
                return false;
            }
        }
    }

    RefreshWorkspace();

    ModelBenchmarkSettings settings = BuildPersistedSetup();
    settings.presetName = name.toStdString();

    QString error;
    const QString settingsJson = QString::fromStdString(SerializeModelBenchmarkSettings(settings));
    if (!m_savedResultsWidget->Library().SavePreset(name, settingsJson, &error)) {
        QMessageBox::warning(this, "Unable to save preset", error);
        return false;
    }

    m_presetName = name;
    m_savedPresetJson = PresetContent(settings);
    RefreshPresetList();
    OnSetupChanged();

    return true;
}

void ModelBenchmarkWidget::LoadPreviousRun()
{
    const std::vector<ModelBenchmarkWorkspaceRecord> runs = m_savedResultsWidget->Library().ListRuns();
    if (runs.empty()) {
        QMessageBox::information(this, "Use previous run", "This library has no saved benchmark runs.");
        return;
    }

    QDialog dialog(this);
    dialog.setObjectName("previousBenchmarkRunDialog");
    dialog.setWindowTitle("Use previous run");
    auto* layout = new QVBoxLayout(&dialog);
    layout->addWidget(new QLabel("Select a run to restore its models and test setup:", &dialog));

    auto* search = new QLineEdit(&dialog);
    search->setPlaceholderText("Filter by date or model...");
    layout->addWidget(search);

    auto* list = new QListWidget(&dialog);
    list->setObjectName("previousBenchmarkRunList");
    list->setTextElideMode(Qt::ElideRight);
    list->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    list->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Expanding);
    for (const ModelBenchmarkWorkspaceRecord& run : runs)
        list->addItem(run.label + (run.settingsJson.isEmpty() ? " [setup not recorded]" : ""));
    layout->addWidget(list, 2);
    layout->addWidget(new QLabel("Selected run", &dialog));

    auto* details = new QPlainTextEdit(&dialog);
    details->setObjectName("previousBenchmarkRunDetails");
    details->setReadOnly(true);
    details->setLineWrapMode(QPlainTextEdit::WidgetWidth);
    details->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Expanding);
    layout->addWidget(details, 1);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    buttons->button(QDialogButtonBox::Ok)->setText("Use Run");
    layout->addWidget(buttons);

    connect(list, &QListWidget::currentRowChanged, &dialog, [&](int row) {
        buttons->button(QDialogButtonBox::Ok)->setEnabled(row >= 0 && !list->item(row)->isHidden());
        details->setPlainText(row < 0 ? QString() : list->item(row)->text());
    });
    connect(search, &QLineEdit::textChanged, &dialog, [&](const QString& text) {
        int firstVisible = -1;
        for (int row = 0; row < list->count(); ++row) {
            const bool matches = list->item(row)->text().contains(text, Qt::CaseInsensitive);
            list->item(row)->setHidden(!matches);
            if (matches && firstVisible < 0) firstVisible = row;
        }

        if (list->currentRow() < 0 || list->currentItem()->isHidden()) list->setCurrentRow(firstVisible);
    });
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    connect(list, &QListWidget::itemDoubleClicked, &dialog, [&]() { dialog.accept(); });
    list->setCurrentRow(0);

    // Long rosters must never determine the window width; full text wraps below.
    const QSize available = screen()->availableGeometry().size();
    dialog.resize(QSize(900, 600).boundedTo(available * 0.9));
    if (dialog.exec() != QDialog::Accepted || list->currentRow() < 0) return;

    const ModelBenchmarkWorkspaceRecord& run = runs[list->currentRow()];
    if (run.settingsJson.isEmpty()) {
        QMessageBox::information(this, "Setup not recorded", "This older run does not contain a complete test setup. Use Add Models to select models from the saved-results library, then check Test Setup before running.");
        return;
    }

    ModelBenchmarkSettingsLoadResult decoded = DeserializeModelBenchmarkSettings(run.settingsJson.toStdString());
    if (!decoded.ok) {
        QStringList errors;
        for (const std::string& error : decoded.errors) errors << QString::fromStdString(error);
        QMessageBox::warning(this, "Unable to load run", QString("The run configuration could not be read.\n\n") + errors.join("\n"));
        return;
    }

    decoded.value.libraryPath = m_savedResultsWidget->LibraryPath();
    decoded.value.presetName.clear();
    m_savedPresetJson.clear();
    ApplyWorkspaceSettings(decoded.value);
}

void ModelBenchmarkWidget::LoadLegacyComparison()
{
    const std::vector<ModelBenchmarkComparisonSet> comparisons = m_savedResultsWidget->Library().ListComparisonSets();
    QStringList names;
    for (const ModelBenchmarkComparisonSet& comparison : comparisons) names << comparison.name;
    if (names.isEmpty()) {
        QMessageBox::information(this, "Older comparison sets", "This library has no older comparison sets.");
        return;
    }

    bool accepted = false;
    const QString name = QInputDialog::getItem(this, "Load comparison", "Comparison set (keeps current test settings):", names, 0, false, &accepted);
    if (!accepted) return;

    const ModelBenchmarkComparisonSet& comparison = comparisons[names.indexOf(name)];
    m_selectedVersions.clear();
    for (const ModelBenchmarkComparisonMember& member : comparison.members) {
        const auto found = std::find_if(m_libraryModels.begin(), m_libraryModels.end(), [&](const ModelBenchmarkLibraryModel& model) {
            return model.id == member.modelId;
        });
        if (found == m_libraryModels.end()) continue;

        AddSavedModel(*found);
        m_selectedVersions.back().included = member.included;
        m_selectedVersions.back().primary = member.primary;
    }

    m_presetName.clear();
    m_savedPresetJson.clear();
    SortSelectedVersions();
    PopulateSelectedTable();
    OnSetupChanged();
}

void ModelBenchmarkWidget::AddSavedModel(const ModelBenchmarkLibraryModel& model, bool resultsOnly)
{
    for (const BenchmarkVersionEntry& entry : m_selectedVersions) {
        if (entry.libraryModelId == model.id) return;
        if (entry.agentName == model.agentName.toStdString() && entry.buildName == model.buildName.toStdString()
            && entry.version.id == model.versionId.toStdString() && entry.fingerprint == model.fingerprint) return;
    }

    BenchmarkVersionEntry entry;
    entry.agentName = model.agentName.toStdString();
    entry.buildName = model.buildName.toStdString();
    entry.stageName = model.stageName.toStdString();
    entry.sourceLabel = model.sourceLabel.toStdString();
    entry.version.id = model.versionId.toStdString();
    entry.version.branchName = model.branchName.toStdString();
    entry.version.latestTimesteps = model.latestTimesteps;
    entry.version.lastTrainedAt = model.lastTrainedAt.toStdString();
    entry.version.modelCount = model.availableLocally ? 1 : 0;
    entry.checkpointPath = model.checkpointPath.toStdString();
    entry.trainingSetupPath = model.trainingSetupPath.toStdString();
    entry.libraryModelId = model.id;
    entry.fingerprint = model.fingerprint;
    entry.primary = false;
    entry.resultsOnly = resultsOnly || !model.availableLocally;

    RefreshAgentSetupStatus(entry);
    RefreshTrainingMetadata(entry);
    m_selectedVersions.push_back(std::move(entry));
}

void ModelBenchmarkWidget::AddOpponents()
{
    if (m_running) return;

    m_opponentsAfterWorkspaceRefresh = true;
    RefreshWorkspace();
}

void ModelBenchmarkWidget::ShowOpponentsPicker()
{
    if (m_libraryModels.empty()) return;

    QStringList labels;
    int initial = 0;
    const int selectedRow = m_selectedTable->currentRow();
    for (int i = 0; i < (int)m_libraryModels.size(); ++i) {
        const ModelBenchmarkLibraryModel& model = m_libraryModels[i];
        labels << model.displayName + " · " + model.fingerprint.left(8);
        if (selectedRow >= 0 && selectedRow < (int)m_selectedVersions.size()
            && m_selectedVersions[selectedRow].libraryModelId == model.id) initial = i;
    }

    bool accepted = false;
    const QString selected = QInputDialog::getItem(this, "Add opponents", "Add opponents with saved games against:", labels, initial, false, &accepted);
    if (!accepted) return;

    const ModelBenchmarkLibraryModel& anchor = m_libraryModels[labels.indexOf(selected)];
    std::set<qint64> opponents;
    for (const ModelBenchmarkLibraryCoverage& pair : m_discoveryCoverage) {
        if (pair.modelA == anchor.id) opponents.insert(pair.modelB);
        if (pair.modelB == anchor.id) opponents.insert(pair.modelA);
    }

    const bool anchorResultsOnly = true;
    AddSavedModel(anchor, anchorResultsOnly);

    const size_t before = m_selectedVersions.size();
    for (const ModelBenchmarkLibraryModel& model : m_libraryModels) {
        if (opponents.contains(model.id)) AddSavedModel(model);
    }

    const bool preserveColumnWidths = true;
    SortSelectedVersions();
    PopulateSelectedTable(preserveColumnWidths);
    OnSetupChanged();
    m_progressPanel->SetStatus(QString("Added %1 opponents of %2. Existing roles and exclusions were kept.")
        .arg(m_selectedVersions.size() - before).arg(anchor.displayName));
}

void ModelBenchmarkWidget::RefreshWorkspace()
{
    if (m_loadingPersistedSetup || m_editingSetup || m_running) return;

    m_workspaceRefreshTimer->stop();
    ++m_workspaceGeneration;
    m_refreshingWorkspace = true;
    m_runButton->setEnabled(false);
    m_scheduleSummaryLabel->setText("Updating saved-game coverage…");
    if (m_workspaceScan.valid()) {
        m_workspaceRefreshPending = true;
        return;
    }

    m_workspaceRefreshPending = false;

    WorkspaceData input;
    input.selected = m_selectedVersions;
    input.filter = WorkspaceFilter();
    input.coverageFilter = {};
    const bool reload = m_coverageDirty || input.coverageFilter != m_coverageFilter || input.filter != m_loadedWorkspaceFilter;
    input.coverage = m_coverage;
    input.discoveryCoverage = m_discoveryCoverage;

    const std::filesystem::path path = m_savedResultsWidget->LibraryPath();
    m_scanningWorkspaceGeneration = m_workspaceGeneration;
    try {
        m_workspaceScan = std::async(std::launch::async, [input = std::move(input), path, reload]() mutable {
            try {
                ModelBenchmarkLibrary library;
                if (path.empty() || !library.OpenReadOnly(path, &input.error)) return input;

                input.models = library.ListModels(&input.error);
                if (!input.error.isEmpty()) return input;

                for (BenchmarkVersionEntry& entry : input.selected) {
                    // Public policies may have no .lt files.
                    // Match the same fallback identity used when saving them, including existing history.
                    if (entry.fingerprint.isEmpty() && HasCheckpoint(entry.checkpointPath)) {
                        const bool allowCached = true;
                        entry.fingerprint = BuildLibraryModels({entry}, allowCached).front().fingerprint;
                    }

                    entry.libraryModelId = -1;
                    for (const ModelBenchmarkLibraryModel& model : input.models) {
                        if (entry.agentName == model.agentName.toStdString() && entry.buildName == model.buildName.toStdString()
                            && entry.stageName == model.stageName.toStdString() && entry.sourceLabel == model.sourceLabel.toStdString()
                            && entry.version.id == model.versionId.toStdString()
                            && entry.fingerprint == model.fingerprint) {
                            entry.libraryModelId = model.id;
                            break;
                        }
                    }
                }

                if (reload) {
                    ModelBenchmarkLibrary::EnsureCoverageIndex(path);
                    input.coverage = library.LoadCoverage(input.coverageFilter, &input.error);
                    if (input.error.isEmpty())
                        input.discoveryCoverage = input.filter == input.coverageFilter ? input.coverage
                            : library.LoadCoverage(input.filter, &input.error);
                }
            } catch (const std::exception& error) {
                input.error = QString::fromUtf8(error.what());
            }

            return input;
        });
        m_workspacePollTimer->start();
    } catch (const std::exception& error) {
        m_refreshingWorkspace = false;
        m_coverageDirty = true;
        m_coverageError = QString::fromUtf8(error.what());
        m_scheduleSummaryLabel->setText(m_coverageError);
    }
}

void ModelBenchmarkWidget::FinishWorkspaceRefresh()
{
    if (!m_workspaceScan.valid() || m_workspaceScan.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
        return;

    m_workspacePollTimer->stop();
    WorkspaceData result = m_workspaceScan.get();
    m_refreshingWorkspace = false;
    if (m_workspaceRefreshPending || m_scanningWorkspaceGeneration != m_workspaceGeneration) {
        RefreshWorkspace();
        return;
    }

    m_coverageError = result.error;
    m_coverageDirty = !result.error.isEmpty();
    if (m_coverageDirty) {
        m_scheduleSummaryLabel->setText(result.error);
        m_runAfterWorkspaceRefresh = m_previewAfterWorkspaceRefresh = m_opponentsAfterWorkspaceRefresh = false;
        return;
    }

    // Only worker-owned fields are published.
    // Catalog display updates may have arrived meanwhile without changing the user's schedule or selection.
    for (size_t i = 0; i < m_selectedVersions.size(); ++i) {
        if (i >= result.selected.size() || MakeEntryKey(m_selectedVersions[i]) != MakeEntryKey(result.selected[i])) {
            RefreshWorkspace();
            return;
        }

        m_selectedVersions[i].fingerprint = result.selected[i].fingerprint;
        m_selectedVersions[i].libraryModelId = result.selected[i].libraryModelId;
    }

    m_libraryModels = std::move(result.models);
    if (m_modelPickerDialog->isVisible()) {
        m_savedModelPicker->setRowCount((int)m_libraryModels.size());
        for (int row = 0; row < (int)m_libraryModels.size(); ++row) {
            const ModelBenchmarkLibraryModel& model = m_libraryModels[row];
            m_savedModelPicker->setItem(row, 0, new QTableWidgetItem(model.displayName));
            m_savedModelPicker->setItem(row, 1, new QTableWidgetItem(model.versionId));
            m_savedModelPicker->setItem(row, 2, new QTableWidgetItem(model.availableLocally ? "Available locally" : "Results only · checkpoint unavailable"));
        }
    }

    m_coverage = std::move(result.coverage);
    m_discoveryCoverage = std::move(result.discoveryCoverage);
    m_coverageFilter = result.coverageFilter;
    m_loadedWorkspaceFilter = result.filter;

    std::vector<ModelBenchmarkLibrarySelection> selection;
    for (const BenchmarkVersionEntry& entry : m_selectedVersions)
        if (entry.included && entry.libraryModelId >= 0)
            selection.push_back({ entry.libraryModelId, true });
    m_savedResultsWidget->SetWorkspaceSelection(selection, result.filter);
    UpdateWorkspaceSummary();

    try {
        const bool preview = std::exchange(m_previewAfterWorkspaceRefresh, false);
        PrepareSchedule(preview);
        if (std::exchange(m_opponentsAfterWorkspaceRefresh, false)) ShowOpponentsPicker();
        if (std::exchange(m_runAfterWorkspaceRefresh, false)) {
            m_verifiedWorkspaceStart = true;
            OnRunClicked();
        }
    } catch (const std::exception& error) {
        m_runAfterWorkspaceRefresh = false;
        m_scheduleSummaryLabel->setText(QString::fromUtf8(error.what()));
        m_runButton->setEnabled(false);
    }
}

ModelBenchmarkConfig ModelBenchmarkWidget::PrepareSchedule(bool showPreview, bool requireRunnable)
{
    if (!m_coverageError.isEmpty()) throw std::runtime_error(m_coverageError.toStdString());

    ModelBenchmarkConfig config = BuildConfig();
    config.usePairBudgets = true;
    const ModelBenchmarkLibraryFilter filter = WorkspaceFilter();

    // Freeze the same setup for eligibility and execution.
    int modelIndex = 0;
    for (const BenchmarkVersionEntry& entry : m_selectedVersions) {
        if (entry.resultsOnly) continue;
        if (HasCheckpoint(entry.checkpointPath)) config.models[modelIndex].agentSetupJson = CaptureModelSetup(entry);
        ++modelIndex;
    }

    // User-selected simulators and environments control reuse; build and legacy compatibility markers do not.
    // It must also work for covered models whose checkpoints are unavailable.
    qint64 incompatibleSavedGames = 0;
    std::vector<int> rosterIndices;
    std::set<qint64> selectedIds;
    for (int i = 0; i < (int)m_selectedVersions.size(); ++i) {
        if (m_selectedVersions[i].resultsOnly) continue;

        rosterIndices.push_back(i);
        if (m_selectedVersions[i].libraryModelId >= 0)
            selectedIds.insert(m_selectedVersions[i].libraryModelId);
    }

    using CoverageKey = std::tuple<qint64, qint64, int>;
    std::map<CoverageKey, qint64> coverage, incompatibleCoverage;
    std::map<CoverageKey, uint64_t> lastSnapshots;
    for (const ModelBenchmarkLibraryCoverage& pair : m_coverage) {
        if (!selectedIds.contains(pair.modelA) || !selectedIds.contains(pair.modelB)) continue;

        const CoverageKey key{ pair.modelA, pair.modelB, pair.mode };
        const bool reuse = filter.MatchesEnvironmentSelection(pair.engineVersion, pair.environmentKey);
        if (reuse || pair.engineVersion == m_engineCombo->currentData().toInt())
            lastSnapshots[key] = std::max(lastSnapshots[key], pair.lastSnapshotId);

        if (reuse) {
            coverage[key] += pair.games;
        } else {
            incompatibleCoverage[key] += pair.games;
        }
    }

    auto savedGames = [&](int a, int b, int mode) {
        const qint64 first = m_selectedVersions[rosterIndices[a]].libraryModelId;
        const qint64 second = m_selectedVersions[rosterIndices[b]].libraryModelId;
        return first < 0 || second < 0 ? qint64(0) : coverage[{ std::min(first, second), std::max(first, second), mode }];
    };

    std::set<int> needsCheckpoint;
    for (const auto& [a, b] : BuildModelBenchmarkMatchups(config.models)) {
        for (int mode : { 1, 2, 3 }) {
            if ((filter.modeMask & (1 << (mode - 1)))
                && (m_scheduleModeCombo->currentIndex() != 0 || savedGames(a, b, mode) < config.maxGamesPerMode)) {
                needsCheckpoint.insert(a);
                needsCheckpoint.insert(b);
            }
        }
    }

    ModelBenchmarkConfig probe = config;
    probe.models.clear();
    std::vector<int> probeIndices;
    for (int i = 0; i < (int)config.models.size(); ++i) {
        if (!needsCheckpoint.contains(i)) continue;

        ModelBenchmarkModel model = config.models[i];
        model.primary = true;
        probe.models.push_back(model);
        probeIndices.push_back(i);
    }

    std::set<std::tuple<int, int, int>> eligible;
    if (probe.models.size() >= 2 && (config.run1v1 || config.run2v2 || config.run3v3)) {
        const ModelBenchmarkEligibilityAnalysis analysis = AnalyzeModelBenchmarkEligibility(probe);
        for (const ModelBenchmarkPairBudget& pair : analysis.eligiblePairModes)
            eligible.insert({ probeIndices[pair.modelA], probeIndices[pair.modelB], pair.mode });
    }

    m_scheduleTable->setRowCount(0);
    std::vector<QStringList> displayRows;
    qint64 newGames = 0, coveredGames = 0, blockedGames = 0;
    int covered = 0, scheduled = 0, blocked = 0;
    std::set<int> participating;
    for (const auto& [a, b] : BuildModelBenchmarkMatchups(config.models)) {
        const BenchmarkVersionEntry& first = m_selectedVersions[rosterIndices[a]];
        const BenchmarkVersionEntry& second = m_selectedVersions[rosterIndices[b]];

        struct ScheduleGroup {
            QString status;
            QStringList modes;
            qint64 saved = 0;
            qint64 needed = 0;
        };

        std::vector<ScheduleGroup> groups;
        for (int mode : { 1, 2, 3 }) {
            if (!(filter.modeMask & (1 << (mode - 1)))) continue;

            const bool locallyAvailable = HasCheckpoint(first.checkpointPath) && HasCheckpoint(second.checkpointPath);
            const qint64 saved = savedGames(a, b, mode);
            const int needed = m_scheduleModeCombo->currentIndex() == 0
                ? (int)std::max<qint64>(0, config.maxGamesPerMode - saved) : config.maxGamesPerMode;
            if (needed > 0 && !eligible.contains({ a, b, mode })) continue;

            const CoverageKey key{ std::min(first.libraryModelId, second.libraryModelId),
                std::max(first.libraryModelId, second.libraryModelId), mode };
            if (first.libraryModelId >= 0 && second.libraryModelId >= 0) {
                incompatibleSavedGames += incompatibleCoverage[key];
                if (config.seed != 0) config.snapshotOffset = std::max(config.snapshotOffset, lastSnapshots[key]);
            }

            QString status;
            if (!needed) {
                status = "Covered";
                ++covered;
            } else if (!locallyAvailable) {
                status = "Checkpoint unavailable";
                ++blocked;
                blockedGames += needed;
            } else {
                status = "Scheduled";
                ++scheduled;
                newGames += needed;
                config.pairBudgets.push_back({ a, b, mode, needed });
                participating.insert(a);
                participating.insert(b);
            }

            coveredGames += saved;

            // Combine display rows only; execution budgets remain per pair and mode.
            auto group = std::find_if(groups.begin(), groups.end(), [&](const ScheduleGroup& item) { return item.status == status; });
            if (group == groups.end()) {
                groups.push_back({ status, {}, 0, 0 });
                group = std::prev(groups.end());
            }

            group->modes.push_back(QString("%1v%1").arg(mode));
            group->saved += saved;
            group->needed += needed;
        }

        QString modelName = QString::fromStdString(MakeDisplayName(first));
        QString opponentName = QString::fromStdString(MakeDisplayName(second));

        // Orient display labels only; canonical pairs, coverage and execution budgets stay unchanged.
        if ((!config.models[a].primary && config.models[b].primary)
            || (config.models[a].primary == config.models[b].primary
                && QString::compare(modelName, opponentName, Qt::CaseInsensitive) > 0))
            std::swap(modelName, opponentName);

        for (const ScheduleGroup& group : groups)
            displayRows.push_back({ modelName, opponentName,
                group.modes.join(", "), QString::number(qint64(config.maxGamesPerMode) * group.modes.size()), QString::number(group.saved), QString::number(group.needed), group.status });
    }

    // Sort cached labels in memory, without extra catalog reads or eligibility work.
    std::stable_sort(displayRows.begin(), displayRows.end(), [](const QStringList& lhs, const QStringList& rhs) {
        const int modelOrder = QString::compare(lhs[0], rhs[0], Qt::CaseInsensitive);
        return modelOrder != 0 ? modelOrder < 0 : QString::compare(lhs[1], rhs[1], Qt::CaseInsensitive) < 0;
    });

    m_scheduleTable->setRowCount(static_cast<int>(displayRows.size()));
    for (int row = 0; row < static_cast<int>(displayRows.size()); ++row) {
        const QStringList& cells = displayRows[row];
        for (int column = 0; column < cells.size(); ++column) {
            auto* item = new QTableWidgetItem(cells[column]);
            item->setTextAlignment(Qt::AlignCenter);
            m_scheduleTable->setItem(row, column, item);
        }
    }

    m_scheduleSummaryLabel->setText(QString("%1 new games · %2 covered games · %3 new matchups to run · %4 covered matchups%5")
        .arg(newGames).arg(coveredGames).arg(scheduled).arg(covered).arg(blocked ? QString(" · %1 matchups blocked (%2 games)").arg(blocked).arg(blockedGames) : QString()));
    m_scheduleSummaryLabel->setToolTip("<qt>New games is the sum of the Scheduled rows in Review Schedule; blocked games are listed separately. Covered games counts compatible saved games across the displayed modes, including partial coverage and games above the target. A matchup is one pair of models in one team size; covered matchups already meet the game target. Review Schedule combines modes with the same status for each pair and sums their game counts. Blocked means required checkpoint files are missing. Saved games can still contribute to Elo when no new games are needed.</qt>");
    m_scheduleSummaryLabel->setToolTip(m_scheduleSummaryLabel->toolTip().chopped(5)
        + "<br><br>" + CompatibilitySummary().toHtmlEscaped()
        + "<br>Saved games must match the model checkpoints and mode. Historical compatibility markers do not restrict reuse. Results use the same Compatibility selection.</qt>");
    if (incompatibleSavedGames > 0) {
        m_scheduleSummaryLabel->setText(m_scheduleSummaryLabel->text()
            + QString(" | %1 saved games excluded by Compatibility").arg(incompatibleSavedGames));
    }

    if (!filter.modeMask) m_scheduleSummaryLabel->setText("Choose a game mode in Test Setup");
    else if (rosterIndices.size() < 2) m_scheduleSummaryLabel->setText("Choose at least two participating models");
    else if (std::none_of(config.models.begin(), config.models.end(), [](const ModelBenchmarkModel& model) { return model.primary; }))
        m_scheduleSummaryLabel->setText("Choose at least one Target, or use Make all Target");

    m_runButton->setEnabled(!m_running && !m_trainingActive && scheduled > 0 && blocked == 0);

    if (showPreview) {
        QDialog dialog(this);
        dialog.setObjectName("benchmarkScheduleDialog");
        dialog.setWindowTitle("Benchmark schedule");
        dialog.resize(980, 560);
        auto* layout = new QVBoxLayout(&dialog);

        auto* summary = new QLabel(m_scheduleSummaryLabel->text(), &dialog);
        summary->setWordWrap(true);
        layout->addWidget(summary);

        auto* hideCovered = new QCheckBox("Hide Covered Matchups", &dialog);
        hideCovered->setObjectName("hideCoveredMatchupsCheck");
        hideCovered->setToolTip("<qt>Hide covered rows from this preview. Schedule totals and games to run stay unchanged.</qt>");
        layout->addWidget(hideCovered);
        connect(hideCovered, &QCheckBox::toggled, &dialog, [this](bool hide) {
            for (int row = 0; row < m_scheduleTable->rowCount(); ++row)
                m_scheduleTable->setRowHidden(row, hide && m_scheduleTable->item(row, 6)->text() == "Covered");
        });
        hideCovered->setChecked(true);

        layout->addWidget(m_scheduleTable);
        m_scheduleTable->show();
        m_scheduleTable->resizeColumnsToContents();
        m_scheduleTable->horizontalHeader()->setStretchLastSection(true);

        auto* buttons = new QDialogButtonBox(QDialogButtonBox::Close, &dialog);
        layout->addWidget(buttons);
        connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);

        dialog.exec();
        m_scheduleTable->setParent(this);
        m_scheduleTable->hide();
    }

    if (blocked && requireRunnable)
        throw std::runtime_error("Some scheduled matchups have unavailable checkpoints. Review Schedule, restore those checkpoints, or mark those models Results only.");

    std::map<int, int> remap;
    std::vector<ModelBenchmarkModel> models;
    m_plannedModelIndices.clear();
    for (int index : participating) {
        remap[index] = (int)models.size();
        models.push_back(config.models[index]);
        m_plannedModelIndices.push_back(rosterIndices[index]);
    }

    for (ModelBenchmarkPairBudget& pair : config.pairBudgets) {
        pair.modelA = remap.at(pair.modelA);
        pair.modelB = remap.at(pair.modelB);
    }

    config.models = std::move(models);
    return config;
}
