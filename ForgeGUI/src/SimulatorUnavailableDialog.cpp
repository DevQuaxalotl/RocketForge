#include "SimulatorUnavailableDialog.h"

#include <QClipboard>
#include <QGuiApplication>
#include <QMessageBox>
#include <QPushButton>

void ShowSimulatorUnavailable(QWidget* parent, const QString& reason)
{
    QMessageBox box(QMessageBox::Warning, "Simulator not available", reason, QMessageBox::Ok, parent);

    const QString command = QuotedCommandFromReason(reason);
    QPushButton* copyButton = command.isEmpty() ? nullptr : box.addButton("Copy command", QMessageBox::ActionRole);

    box.exec();

    if (copyButton && box.clickedButton() == copyButton)
        QGuiApplication::clipboard()->setText(command);
}
