#pragma once

#include <QComboBox>
#include <QString>

/**
 * QComboBox preloaded with the available agents (AgentRegistry::GetAvailableAgents()).
 * Starts on the training agent saved in forge_app_state.json.
 * Emits AgentChanged when the selection changes.
 *
 * Note: the constructor calls Populate() immediately — agents are context-free
 * so the combo is ready to use as soon as it is constructed.  This differs from
 * BuildComboBox and SegmentComboBox, which wait for SetAgent / SetAgentBuild
 * before they can populate.
 */
class AgentComboBox : public QComboBox
{
    Q_OBJECT
public:
    explicit AgentComboBox(QWidget* parent = nullptr);

    /** (Re)populate from AgentRegistry. Preserves current selection if still present. */
    void Populate();

    QString CurrentAgentName() const;
    void SetCurrentAgent(const QString& agentName);   // no-op if not present; blocks signals

    /** Switches to agentName and emits AgentChanged so dependents refresh. Ignored while the combo is disabled. */
    void FollowAgent(const QString& agentName);

signals:
    void AgentChanged(const QString& agentName);
};
