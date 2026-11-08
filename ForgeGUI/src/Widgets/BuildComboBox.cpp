#include "BuildComboBox.h"

#include <ForgeTraining.h>
#include <Storage/BuildStorage.h>

#include <QSignalBlocker>

BuildComboBox::BuildComboBox(QWidget* parent) : QComboBox(parent)
{
    connect(this, qOverload<int>(&QComboBox::currentIndexChanged),
            this, [this](int) { emit BuildChanged(CurrentBuildName()); });
}

void BuildComboBox::SetAgent(const QString& agentName)
{
    m_agentName = agentName.trimmed();
    Refresh();
}

void BuildComboBox::Refresh(const QString& preferred)
{
    const QString requested = preferred.trimmed().isEmpty()
        ? CurrentBuildName()
        : preferred.trimmed();

    QSignalBlocker blocker(this);
    clear();
    for (const RocketForgeTraining::BuildSummary& build :
            RocketForgeTraining::ListAgentBuilds(m_agentName.toStdString()))
        addItem(QString::fromStdString(build.buildName));

    if (count() == 0)
        addItem(NoBuildLabel);

    const int idx = findText(requested);
    setCurrentIndex(idx >= 0 ? idx : 0);
}

QString BuildComboBox::CurrentBuildName() const
{
    const QString name = currentText().trimmed();
    return (name.isEmpty() || name == NoBuildLabel) ? QString() : name;
}

void BuildComboBox::SetCurrentBuild(const QString& buildName)
{
    const int idx = findText(buildName);
    if (idx < 0)
        return;
    QSignalBlocker blocker(this);
    setCurrentIndex(idx);
}
