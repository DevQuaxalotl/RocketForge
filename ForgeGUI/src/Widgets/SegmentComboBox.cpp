#include "SegmentComboBox.h"

#include <Storage/BuildStorage.h>
#include <Storage/ImportedModelStorage.h>
#include <AgentModelPaths.h>

#include <GigaLearnCPP/TrainingStorage.h>

#include <QSignalBlocker>
#include <QStandardItemModel>

#include <algorithm>
#include <filesystem>
#include <optional>
#include <set>

using namespace RocketForgeTraining;

namespace {
    constexpr int kSegmentRole        = Qt::UserRole;        // segment id
    constexpr int kStageRole          = Qt::UserRole + 1;    // stage name
    constexpr int kCheckpointPathRole = Qt::UserRole + 2;    // imported: resolved checkpoint model folder
    constexpr int kVersionPathRole    = Qt::UserRole + 3;    // imported: cache/version folder
    constexpr int kIsImportedRole     = Qt::UserRole + 4;    // imported: true for every imported item, runnable or not
}

SegmentComboBox::SegmentComboBox(QWidget* parent) : QComboBox(parent)
{
    connect(this, qOverload<int>(&QComboBox::currentIndexChanged),
            this, [this](int) { emit SegmentChanged(QString::fromStdString(CurrentSegmentId())); });
}

void SegmentComboBox::SetAgentBuild(const QString& agentName, const QString& buildName)
{
    m_agentName = agentName.trimmed();
    m_buildName = buildName.trimmed();
    Refresh();
}

void SegmentComboBox::SetIncludeImportedModels(bool include)
{
    m_includeImportedModels = include;
    Refresh();
}

void SegmentComboBox::Refresh(const QString& preferredSegmentId)
{
    const QString requested = preferredSegmentId.isEmpty()
        ? QString::fromStdString(CurrentSegmentId())
        : preferredSegmentId;

    {
        QSignalBlocker blocker(this);
        clear();

        if (!m_agentName.isEmpty() && !m_buildName.isEmpty()) {
            std::vector<TrainingSegmentMetadata> segments;
            std::vector<std::optional<ResolvedCheckpoint>> latestCheckpoints;
            try {
                BuildStorage storage(ResolveAgentModelsFolder(m_agentName.toStdString()),
                                     m_buildName.toStdString());
                segments = storage.ListSegments();

                // segment.json lists only durable checkpoints, so resolve from disk to include temp_checkpoints/ autosaves.
                for (const TrainingSegmentMetadata& seg : segments)
                    latestCheckpoints.push_back(storage.LatestSegmentCheckpoint(seg.folderName));
            } catch (...) {
                segments.clear();
                latestCheckpoints.clear();
            }

            // ListSegments() returns oldest-first (ascending revision). Show newest-first
            // so the latest segment is at the top of the dropdown; this also makes the
            // auto-selected default (firstRenderable below) the latest renderable segment.
            std::reverse(segments.begin(), segments.end());
            std::reverse(latestCheckpoints.begin(), latestCheckpoints.end());

            int preferredIdx = -1;
            int firstRenderable = -1;
            for (size_t segmentIndex = 0; segmentIndex < segments.size(); ++segmentIndex) {
                const TrainingSegmentMetadata& seg = segments[segmentIndex];
                const std::optional<ResolvedCheckpoint>& latestCheckpoint = latestCheckpoints[segmentIndex];
                const uint64_t latest = latestCheckpoint ? latestCheckpoint->timesteps : 0;
                const bool renderable = latestCheckpoint.has_value();
                QString text = QString("%1 — %2 — %3")
                    .arg(QString::fromStdString(seg.segmentId))
                    .arg(latest)
                    .arg(QString::fromStdString(seg.stageName));
                if (latestCheckpoint && latestCheckpoint->isAutosave)
                    text += " (autosave)";
                addItem(text);
                const int row = count() - 1;
                setItemData(row, QString::fromStdString(seg.segmentId), kSegmentRole);
                setItemData(row, QString::fromStdString(seg.stageName), kStageRole);
                if (!renderable) {
                    if (auto* model = qobject_cast<QStandardItemModel*>(this->model()))
                        if (QStandardItem* item = model->item(row))
                            item->setFlags(item->flags() & ~Qt::ItemIsEnabled);
                } else if (firstRenderable < 0) {
                    firstRenderable = row;
                }
                if (QString::fromStdString(seg.segmentId) == requested && renderable)
                    preferredIdx = row;
            }

            if (count() > 0) {
                const int target = preferredIdx >= 0 ? preferredIdx
                                 : firstRenderable >= 0 ? firstRenderable
                                 : 0;

                setCurrentIndex(target);
            }
        }
        else if (m_includeImportedModels && !m_agentName.isEmpty() && m_buildName.isEmpty()) {
            // No build selected: list the agent's imported models as pseudo-segments via the
            // shared discovery helper. The combo carries resolved checkpoint + cache paths as
            // item data for the caller to read back.
            int preferredIdx = -1;
            int firstRenderable = -1;

            for (const DiscoveredImportedModel& model : EnumerateImportedModels(m_agentName.toStdString())) {
                const QString folderName  = QString::fromStdString(model.folderName);
                const QString sourceStage = QString::fromStdString(model.metadata.sourceStageName);

                QString text = QString("%1 — %2 — imported")
                    .arg(folderName)
                    .arg(model.metadata.latestTimesteps);
                if (!sourceStage.isEmpty())
                    text += QString(" (%1)").arg(sourceStage);
                if (!model.canUseForInference)
                    text += " (no policy)";

                addItem(text);
                const int row = count() - 1;
                setItemData(row, folderName, kSegmentRole);
                setItemData(row, sourceStage, kStageRole);
                setItemData(row, true, kIsImportedRole);

                if (model.canUseForInference) {
                    setItemData(row, QString::fromStdString(model.resolvedModelFolder.string()), kCheckpointPathRole);
                    setItemData(row, QString::fromStdString(model.importedRoot.string()), kVersionPathRole);

                    if (firstRenderable < 0)
                        firstRenderable = row;
                } else {
                    if (auto* itemModel = qobject_cast<QStandardItemModel*>(this->model()))
                        if (QStandardItem* item = itemModel->item(row)) {
                            item->setFlags(item->flags() & ~Qt::ItemIsEnabled);
                            item->setToolTip("This imported model has no usable inference policy "
                                              "(no POLICY.lt or INFERENCE_POLICY.json in its model "
                                              "folder), so reward analysis can't run on it.");
                        }
                }

                if (folderName == requested && model.canUseForInference)
                    preferredIdx = row;
            }

            if (count() > 0) {
                const int target = preferredIdx >= 0 ? preferredIdx
                                 : firstRenderable >= 0 ? firstRenderable
                                 : 0;

                setCurrentIndex(target);
            }
        }
    } // blocker destroyed here — signals unblocked

    // Always re-announce the resolved selection (even if the index didn't change) so
    // dependents like team-size refresh update on every Refresh().
    emit SegmentChanged(QString::fromStdString(CurrentSegmentId()));
}

std::string SegmentComboBox::CurrentSegmentId() const
{
    if (currentIndex() < 0)
        return {};

    return currentData(kSegmentRole).toString().toStdString();
}

QString SegmentComboBox::CurrentStageName() const
{
    if (currentIndex() < 0)
        return {};

    return currentData(kStageRole).toString();
}

void SegmentComboBox::SetCurrentSegment(const std::string& segmentId)
{
    for (int i = 0; i < count(); ++i) {
        if (itemData(i, kSegmentRole).toString().toStdString() == segmentId) {
            QSignalBlocker blocker(this);
            setCurrentIndex(i);
            return;
        }
    }
}

bool SegmentComboBox::HasRenderableSelection() const
{
    const int idx = currentIndex();

    if (idx < 0)
        return false;

    if (const auto* model = qobject_cast<const QStandardItemModel*>(this->model()))
        if (const QStandardItem* item = model->item(idx))
            return (item->flags() & Qt::ItemIsEnabled) && !CurrentSegmentId().empty();

    return !CurrentSegmentId().empty();
}

bool SegmentComboBox::CurrentIsImported() const
{
    if (currentIndex() < 0)
        return false;

    return currentData(kIsImportedRole).toBool();
}

QString SegmentComboBox::CurrentCheckpointPath() const
{
    if (currentIndex() < 0)
        return {};

    return currentData(kCheckpointPathRole).toString();
}

QString SegmentComboBox::CurrentVersionPath() const
{
    if (currentIndex() < 0)
        return {};

    return currentData(kVersionPathRole).toString();
}
