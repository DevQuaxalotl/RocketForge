#include "AgentComboBox.h"

#include <GigaLearnInfer/AgentRegistry.h>
#include <Settings/ForgeSettings.h>

#include <QSignalBlocker>

AgentComboBox::AgentComboBox(QWidget* parent) : QComboBox(parent)
{
    connect(this, qOverload<int>(&QComboBox::currentIndexChanged),
            this, [this](int) { emit AgentChanged(CurrentAgentName()); });
    Populate();
    SetCurrentAgent(QString::fromStdString(RocketForgeTraining::LoadLastSelectedAgent()));
}

void AgentComboBox::Populate()
{
    const QString previous = CurrentAgentName();
    QSignalBlocker blocker(this);
    clear();
    for (const auto& name : AgentRegistry::GetAvailableAgents())
        addItem(QString::fromStdString(name));
    const int idx = findText(previous);
    if (idx >= 0)
        setCurrentIndex(idx);
}

QString AgentComboBox::CurrentAgentName() const
{
    return currentText().trimmed();
}

void AgentComboBox::SetCurrentAgent(const QString& agentName)
{
    const int idx = findText(agentName);
    if (idx < 0)
        return;
    QSignalBlocker blocker(this);
    setCurrentIndex(idx);
}

void AgentComboBox::FollowAgent(const QString& agentName)
{
    if (!isEnabled())
        return;

    const int idx = findText(agentName);
    if (idx >= 0)
        setCurrentIndex(idx);
}
