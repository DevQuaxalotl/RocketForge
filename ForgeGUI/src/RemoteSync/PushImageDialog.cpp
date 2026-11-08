#include "RemoteSync/PushImageDialog.h"

#include "ThemeUtils.h"

#include <QCheckBox>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QGuiApplication>
#include <QLabel>
#include <QLineEdit>
#include <QProcess>
#include <QPushButton>
#include <QScopeGuard>
#include <QVBoxLayout>

namespace ForgeGUI
{
	namespace
	{
		constexpr int BUILD_INFO_TIMEOUT_MS = 15000;

		QString ResolveBuildInfoTag(const QString& repoRoot, const QString& agent, const QString& build)
		{
			const QString consoleTrainer = FindNewestConsoleTrainer(repoRoot);
			if (consoleTrainer.isEmpty())
				return {};

			// build-info blocks the GUI thread for up to the timeout, so show a busy cursor.
			QGuiApplication::setOverrideCursor(Qt::WaitCursor);
			const auto restoreCursor = qScopeGuard([] { QGuiApplication::restoreOverrideCursor(); });

			QProcess process;
			process.setWorkingDirectory(repoRoot);
			QStringList arguments = { "build-info", "--agent", agent };
			if (!build.isEmpty())
				arguments << "--build" << build;

			process.start(consoleTrainer, arguments);
			if (!process.waitForFinished(BUILD_INFO_TIMEOUT_MS) || process.exitCode() != 0)
				return {};

			return ParseBuildInfoTag(QString::fromLocal8Bit(process.readAllStandardOutput()));
		}
	}

	PushImageDialog::PushImageDialog(const QString& repoRoot, const QString& agent, const QString& build, QWidget* parent)
		: QDialog(parent), m_agent(agent), m_build(build), m_dockerUser(qEnvironmentVariable("DOCKERHUB_USER"))
	{
		setWindowTitle("Push Training Image");

		m_tagEdit = new QLineEdit(ResolveBuildInfoTag(repoRoot, agent, build), this);
		m_tagLatestCheck = new QCheckBox("Also tag :latest", this);
		m_pushModelsCheck = new QCheckBox("Also push models to R2", this);
		m_previewLabel = new QLabel(this);
		m_buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);

		m_tagEdit->setPlaceholderText("timestamp (yyyyMMdd-HHmm)");
		m_tagLatestCheck->setChecked(true);
		m_pushModelsCheck->setChecked(true);
		m_previewLabel->setProperty(TextRoleProperty, "muted");
		m_previewLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
		m_buttons->button(QDialogButtonBox::Ok)->setText("Build && Push");

		auto* form = new QFormLayout();
		form->addRow("Tag:", m_tagEdit);
		form->addRow(QString(), m_tagLatestCheck);
		form->addRow(QString(), m_pushModelsCheck);

		auto* layout = new QVBoxLayout(this);
		layout->addLayout(form);
		layout->addWidget(m_previewLabel);

		if (m_dockerUser.isEmpty()) {
			auto* userError = new QLabel(
				"DOCKERHUB_USER is not set. Set it to your Docker Hub username, then restart the GUI "
				"(environment variables are read when the GUI starts).",
				this);
			userError->setWordWrap(true);
			userError->setStyleSheet(QString("color: %1").arg(Theme::NEGATIVE.name()));
			layout->addWidget(userError);
			m_buttons->button(QDialogButtonBox::Ok)->setEnabled(false);
		}

		layout->addWidget(m_buttons);

		connect(m_tagEdit, &QLineEdit::textChanged, this, &PushImageDialog::UpdatePreview);
		connect(m_buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
		connect(m_buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

		UpdatePreview();
	}

	PushImageOptions PushImageDialog::Options() const
	{
		PushImageOptions options;
		options.agent = m_agent;
		options.build = m_build;
		options.tag = m_tagEdit->text().trimmed();
		options.tagLatest = m_tagLatestCheck->isChecked();
		options.pushModels = m_pushModelsCheck->isChecked();
		return options;
	}

	void PushImageDialog::UpdatePreview()
	{
		const QString imageRepo = "rocketforge";
		const QString user = m_dockerUser.isEmpty() ? QString("<DOCKERHUB_USER>") : m_dockerUser;
		const QString tag = m_tagEdit->text().trimmed().isEmpty() ? QString("<timestamp>") : m_tagEdit->text().trimmed();

		m_previewLabel->setText(QString("docker.io/%1/%2:%3").arg(user, imageRepo, tag));
	}
}
