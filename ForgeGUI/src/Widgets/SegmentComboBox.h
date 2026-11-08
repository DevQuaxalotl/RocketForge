#pragma once

#include <QComboBox>
#include <QString>
#include <string>

/**
 * QComboBox listing the Training Segments of an agent+build.
 * Item text: "<segmentId> — <latest timesteps> — <stageName>"; item data: segmentId (QString).
 * Segments with no checkpoints are listed but disabled (nothing to render).
 *
 * Optionally (via SetIncludeImportedModels) also lists an agent's imported models as
 * pseudo-segments when no build is selected. Imported models with no usable inference
 * policy are listed but disabled.
 */
class SegmentComboBox : public QComboBox
{
    Q_OBJECT
public:
    explicit SegmentComboBox(QWidget* parent = nullptr);

    void SetAgentBuild(const QString& agentName, const QString& buildName);

    /**
     * Opt-in: when set with an agent and an EMPTY build, also list the agent's imported models
     * as pseudo-segments. Off by default, so the shared Debug-tab usage is unaffected.
     */
    void SetIncludeImportedModels(bool include);

    /**
     * Repopulate the combo, then ALWAYS emit SegmentChanged for the resolved
     * selection — even when the index didn't change.  Unlike a plain
     * currentIndexChanged, callers (e.g. team-size refresh) can rely on this
     * signal firing on every Refresh() to stay in sync.
     */
    void Refresh(const QString& preferredSegmentId = QString());

    /** Selected segment id, or empty when none/placeholder. */
    std::string CurrentSegmentId() const;

    /** The stage name of the currently-selected segment (empty if none). */
    QString CurrentStageName() const;
    void SetCurrentSegment(const std::string& segmentId);

    /** True when the current selection is a real, checkpoint-bearing segment. */
    bool HasRenderableSelection() const;

    /**
     * True when the current item is an imported model, runnable or not. Non-runnable imported
     * items (no usable inference policy) are still marked so callers can show a specific
     * explanation instead of falling through to the trained-bot path.
     */
    bool CurrentIsImported() const;

    /** Resolved checkpoint model folder for an imported item (empty for normal segments). */
    QString CurrentCheckpointPath() const;

    /** Cache/version folder for an imported item (empty for normal segments). */
    QString CurrentVersionPath() const;

signals:
    void SegmentChanged(const QString& segmentId);

private:
    QString m_agentName;
    QString m_buildName;
    bool m_includeImportedModels = false;
};
