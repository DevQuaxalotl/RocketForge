#include "RemoteSync/RemoteSyncPanel.h"

#include "RemoteSync/PushImageDialog.h"
#include "RemoteSync/RemoteSyncRunner.h"
#include "ThemeUtils.h"

#include <QCheckBox>
#include <QDir>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>

namespace ForgeGUI
{
	namespace
	{
		constexpr int CANCEL_WAIT_MS = 5000;
		const char* const POWERSHELL_PROGRAM = "powershell.exe";
		const char* const STEP_PREFIX = "==> ";
	}

	RemoteSyncPanel::RemoteSyncPanel(const QString& repoRoot, QWidget* parent)
		: QGroupBox("Remote Sync", parent), m_repoRoot(repoRoot)
	{
		m_runner = new RemoteSyncRunner(this);

		m_pullButton = new QPushButton("Pull Models", this);
		m_pushButton = new QPushButton("Push Models", this);
		m_allModelsCheck = new QCheckBox("All models", this);
		m_tempCheckpointsCheck = new QCheckBox("Include temp checkpoints", this);
		m_allTempCheckpointsCheck = new QCheckBox("All timesteps", this);
		m_pushImageButton = new QPushButton("Push Training Image…", this);
		m_statusLabel = new QLabel(this);
		m_cancelButton = new QPushButton("Cancel", this);

		m_allModelsCheck->setToolTip("Pull the agent's whole Models tree instead of only what the stage queue needs.");
		m_tempCheckpointsCheck->setToolTip("Also pull each queue segment's newest remote autosave (temp_checkpoints/). Not available with All models.");
		m_allTempCheckpointsCheck->setToolTip("Pull every remote autosave of each queue segment instead of only the newest, to keep earlier timesteps to roll back to.");
		m_statusLabel->setProperty(TextRoleProperty, "muted");
		m_cancelButton->setVisible(false);

		auto* syncRow = new QHBoxLayout();
		syncRow->addWidget(m_pullButton);
		syncRow->addWidget(m_pushButton);
		syncRow->addSpacing(12);
		syncRow->addWidget(m_allModelsCheck);
		syncRow->addWidget(m_tempCheckpointsCheck);
		syncRow->addWidget(m_allTempCheckpointsCheck);
		syncRow->addStretch();

		auto* imageRow = new QHBoxLayout();
		imageRow->addWidget(m_pushImageButton);
		imageRow->addStretch();

		auto* statusRow = new QHBoxLayout();
		statusRow->addWidget(m_statusLabel, 1);
		statusRow->addWidget(m_cancelButton);

		auto* layout = new QVBoxLayout(this);
		layout->addLayout(syncRow);
		layout->addLayout(imageRow);
		layout->addLayout(statusRow);

		// clang-format off
		connect(m_pullButton,            &QPushButton::clicked,           this,     &RemoteSyncPanel::OnPullClicked);
		connect(m_pushButton,            &QPushButton::clicked,           this,     &RemoteSyncPanel::OnPushClicked);
		connect(m_pushImageButton,       &QPushButton::clicked,           this,     &RemoteSyncPanel::OnPushImageClicked);
		connect(m_cancelButton,          &QPushButton::clicked,           m_runner, &RemoteSyncRunner::Cancel);
		connect(m_allModelsCheck,        &QCheckBox::toggled,             this,     &RemoteSyncPanel::UpdateControls);
		connect(m_tempCheckpointsCheck,  &QCheckBox::toggled,             this,     &RemoteSyncPanel::UpdateControls);
		connect(m_runner,                &RemoteSyncRunner::LineReceived, this,     &RemoteSyncPanel::OnRunnerLine);
		connect(m_runner,                &RemoteSyncRunner::Finished,     this,     &RemoteSyncPanel::OnRunnerFinished);
		// clang-format on

		UpdateControls();
	}

	void RemoteSyncPanel::SetTarget(const QString& agent, const QString& build)
	{
		m_agent = agent;
		m_build = build;
		UpdateControls();
	}

	void RemoteSyncPanel::SetAvailability(bool canPush, bool canPull)
	{
		m_canPush = canPush;
		m_canPull = canPull;
		UpdateControls();
	}

	bool RemoteSyncPanel::IsRunning() const
	{
		return m_runner->IsRunning();
	}

	void RemoteSyncPanel::CancelAndWait()
	{
		m_runner->CancelAndWait(CANCEL_WAIT_MS);
	}

	void RemoteSyncPanel::OnPullClicked()
	{
		PullOptions options;
		options.agent = m_agent;
		options.build = m_build;
		options.trainingOnly = !m_allModelsCheck->isChecked();
		options.tempCheckpoints = m_tempCheckpointsCheck->isChecked();
		options.allTempCheckpoints = m_allTempCheckpointsCheck->isChecked();

		emit FlushSettingsRequested();
		StartSync(RemoteSyncKind::Pull, BuildPowerShellArguments(ScriptPath(RemoteSyncKind::Pull), options));
	}

	void RemoteSyncPanel::OnPushClicked()
	{
		emit FlushSettingsRequested();
		StartSync(RemoteSyncKind::Push, BuildPowerShellArguments(ScriptPath(RemoteSyncKind::Push), PushOptions{ m_agent, m_build }));
	}

	void RemoteSyncPanel::OnPushImageClicked()
	{
		// The dialog's build-info call reads forge_settings.json, so flush before it opens.
		emit FlushSettingsRequested();

		PushImageDialog dialog(m_repoRoot, m_agent, m_build, this);
		if (dialog.exec() != QDialog::Accepted)
			return;

		StartSync(RemoteSyncKind::PushImage, BuildPowerShellArguments(ScriptPath(RemoteSyncKind::PushImage), dialog.Options()));
	}

	void RemoteSyncPanel::OnRunnerLine(const QString& line)
	{
		if (line.startsWith(STEP_PREFIX))
			SetStatus(line.mid(QString(STEP_PREFIX).size()), Theme::TEXT_MUTED);

		emit LineReceived(line);
	}

	void RemoteSyncPanel::OnRunnerFinished(RemoteSyncKind kind, RemoteSyncOutcome outcome, int exitCode)
	{
		switch (outcome) {
		case RemoteSyncOutcome::Succeeded:
			SetStatus("Done", Theme::POSITIVE);
			break;
		case RemoteSyncOutcome::Failed:
			SetStatus(QString("Failed (exit %1)").arg(exitCode), Theme::NEGATIVE);
			break;
		case RemoteSyncOutcome::Cancelled:
			SetStatus("Cancelled", Theme::TEXT_MUTED);
			break;
		}

		UpdateControls();

		emit RunningChanged(false, kind);
		emit Finished(kind, outcome, exitCode);
	}

	void RemoteSyncPanel::StartSync(RemoteSyncKind kind, const QStringList& arguments)
	{
		if (arguments.isEmpty())
			return;

		SetStatus("Starting…", Theme::TEXT_MUTED);
		emit RunningChanged(true, kind);

		// A program that fails to start finishes synchronously in here, so RunningChanged(true) must already be out.
		m_runner->Start(kind, POWERSHELL_PROGRAM, arguments, m_repoRoot);
		UpdateControls();
	}

	QString RemoteSyncPanel::ScriptPath(RemoteSyncKind kind) const
	{
		return QDir::toNativeSeparators(QDir(m_repoRoot).filePath(RemoteSyncScriptRelativePath(kind)));
	}

	void RemoteSyncPanel::SetStatus(const QString& text, const QColor& color)
	{
		m_statusLabel->setText(text);
		m_statusLabel->setStyleSheet(QString("color: %1").arg(color.name()));
	}

	void RemoteSyncPanel::UpdateControls()
	{
		const bool running = m_runner->IsRunning();
		const bool canStart = !running && !m_agent.trimmed().isEmpty();

		if (m_allModelsCheck->isChecked())
			m_tempCheckpointsCheck->setChecked(false);
		if (!m_tempCheckpointsCheck->isChecked())
			m_allTempCheckpointsCheck->setChecked(false);

		m_pullButton->setEnabled(canStart && m_canPull);
		m_pushButton->setEnabled(canStart && m_canPush);
		m_pushImageButton->setEnabled(canStart && m_canPush);
		m_allModelsCheck->setEnabled(!running);
		m_tempCheckpointsCheck->setEnabled(!running && !m_allModelsCheck->isChecked());
		m_allTempCheckpointsCheck->setEnabled(!running && m_tempCheckpointsCheck->isChecked());
		m_cancelButton->setVisible(running);
	}
}
