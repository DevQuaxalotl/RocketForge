#pragma once

#include <QComboBox>
#include <QString>

/**
 * QComboBox listing the builds for a given agent (RocketForgeTraining::ListAgentBuilds).
 * Shows a placeholder ("N/A") when the agent has no builds. Emits BuildChanged on user change.
 */
class BuildComboBox : public QComboBox
{
    Q_OBJECT
public:
    explicit BuildComboBox(QWidget* parent = nullptr);

    static constexpr const char* NoBuildLabel = "N/A";

    /** Set the agent whose builds are listed, then repopulate. */
    void SetAgent(const QString& agentName);

    /** Repopulate for the current agent, preferring `preferred` (or current) selection. */
    void Refresh(const QString& preferred = QString());

    /** Empty string when no real build is selected (i.e. placeholder showing). */
    QString CurrentBuildName() const;
    void SetCurrentBuild(const QString& buildName);   // blocks signals

signals:
    void BuildChanged(const QString& buildName);

private:
    QString m_agentName;
};
