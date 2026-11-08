#pragma once

#include <QString>
#include <QWidget>

#include "Common/TableSortController.h"
#include "Common/ModelCatalog.h"

#include <GigaLearnCPP/SegmentVersioning.h>

#include <array>
#include <filesystem>
#include <functional>
#include <vector>

class QTableWidget;
class QPushButton;
class QLabel;
class QComboBox;
class AgentComboBox;
class QCheckBox;
class QGroupBox;
class QLineEdit;
class TrainingSetupWindow;
namespace RocketForge { class TrainingStage; }

/**
 * Widget for reviewing and managing saved training segments.
 * It provides destructive actions, RLBot and Bob submittal exports,
 * and access to the saved training-setup snapshot for each version.
 **/
class VersionManagementDialog : public QWidget
{
    Q_OBJECT

public:
    VersionManagementDialog(
        const QString& initialAgentName,
        const QString& initialBuildName,
        const QString& initialStageName,
        bool trainingActive,
        const QString& activeAgentName,
        const QString& activeBuildName,
        const QString& activeStageName,
        const QString& activeVersionId,
        QWidget* parent = nullptr);
    void SetContext(
        const QString& agentName,
        const QString& buildName,
        const QString& stageName);
    void SetTrainingContext(
        bool trainingActive,
        const QString& activeAgentName,
        const QString& activeBuildName,
        const QString& activeStageName,
        const QString& activeVersionId);

public slots:
    void Refresh();

private slots:
    void DeleteVersion();
    void PruneOldModels();
    void ImportExternalModel();
    void ExportRLBot();
    void ExportForSubmittal();
    void EditNotes();
    void MoveBranch();
    void ToggleActive();
    void OpenTrainingSetup();
    void OnSelectionChanged();
    void OnContextChanged();

private:
    friend class TrainingDashboardTests;

    void RefreshFromCatalog();
    void ApplyCatalogUpdate();
    QString CurrentAgentName() const;
    QString CurrentBuildName() const;
    QString CurrentStageName() const;
    QString CurrentBranchName() const;
    bool IsViewingActiveTrainingStage() const;
    bool IsActiveTrainingRow(int row) const;
    bool IsImportedModelFolderRow(int row) const;
    bool CanMutateRow(int row);
    std::string SelectedVersionId() const;
    std::vector<int> SelectedRows() const;
    int SelectedRow() const;
    void ApplyRowFilters();
    std::vector<int> CaptureColumnWidths() const;
    void RestoreColumnWidths(const std::vector<int>& columnWidths);
    void RefreshPreservingColumnWidths();
    void AddImportedModelRows(
        const std::string& agentName,
        const std::string& buildFilter,
        const std::string& stageFilter);
    void SaveImportedModelNotes(int row, const QString& notes);
    std::string RowKey(int row) const;
    void UpdateContextLabel();
    void RefreshBuilds();
    void RefreshStages();
    void RefreshBranches();
    void UpdateActionButtons();
    void UpdateDialogWidth();
    void WithStage(const std::function<void(RocketForge::TrainingStage*)>& fn);
    void WithStageForRow(int row, const std::function<void(RocketForge::TrainingStage*)>& fn);
	
	/** Each column a version can have in order of sort priority */
	enum VersionColumn
	{
		Unselected = -1,
		Build = 0,
		Segment,
		Branch,
		Stage,
		MadeFrom,
		LatestTimesteps,
		SkillRating,
		Status,
		LastTrained,
		ModelCount,
		RLBot,
		TrainingSetup,
		Notes,
		Count
	};

	/** 
	 * The columns in UI (left-to-right) order. Single source of truth for the mapping between a
	 * UI column index and its VersionColumn: index -> column is a lookup, column -> index a search.
	 * Reorder this to rearrange the table columns; the enum values fix sort priority independently. 
	 */
	static constexpr std::array<VersionColumn, static_cast<std::size_t>(Count)> ColumnsInUiOrder = {
		Build,          
		Segment,        
		MadeFrom,       
		Stage,          
		LatestTimesteps,
		Branch,         
		Status,         
		LastTrained,    
		SkillRating,    
		ModelCount,     
		RLBot,          
		TrainingSetup,  
		Notes,          
	};

	/** Ascending -1/0/1 comparison of two rows for a single VersionColumn; drives the sort controller. */
	int CompareVersionColumn(int lhsRow, int rhsRow, VersionColumn column) const;

    QString m_activeAgentName;
    QString m_activeBuildName;
    QString m_activeStageName;
    bool m_trainingActive = false;
    QString m_activeVersionId;
    AgentComboBox* m_agentCombo = nullptr;
    QComboBox* m_buildCombo = nullptr;
    QComboBox* m_stageCombo = nullptr;
    QComboBox* m_branchCombo = nullptr;
    QCheckBox* m_hideInactiveCheck = nullptr;
    QLineEdit* m_searchEdit = nullptr;
    QTableWidget* m_table = nullptr;
    QPushButton* m_deleteBtn = nullptr;
    QPushButton* m_pruneBtn = nullptr;
    QPushButton* m_importExternalModelBtn = nullptr;
    QPushButton* m_editNotesBtn = nullptr;
    QPushButton* m_moveBranchBtn = nullptr;
    QPushButton* m_activeBtn = nullptr;
    QLabel* m_readOnlyLabel = nullptr;
    QPushButton* m_rlbotBtn = nullptr;
    QPushButton* m_submittalBtn = nullptr;
    TrainingSetupWindow* m_trainingSetupWindow = nullptr;
    TableSortController m_sortController;
    std::vector<GGL::SegmentVersionSummary> m_versions;
    std::vector<std::string> m_versionBuildNames;
    std::vector<std::string> m_versionStageNames;
    std::vector<bool> m_versionIsImportedModelFolder;
    std::vector<std::filesystem::path> m_importedModelRootPaths;
    std::shared_ptr<const ModelCatalogSnapshot> m_catalog;
    QString m_pendingBuild;
    bool m_catalogUpdateDeferred = false;
};
