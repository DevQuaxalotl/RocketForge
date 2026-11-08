#pragma once

#include "RemoteSync/RemoteSyncCommand.h"

#include <QDialog>

class QCheckBox;
class QDialogButtonBox;
class QLabel;
class QLineEdit;

namespace ForgeGUI
{
	/** Options for push_training_image.ps1, with the tag pre-filled from `ForgeConsoleTrainer build-info`. */
	class PushImageDialog : public QDialog
	{
		Q_OBJECT

	public:
		PushImageDialog(const QString& repoRoot, const QString& agent, const QString& build, QWidget* parent = nullptr);

		PushImageOptions Options() const;

	private:
		void UpdatePreview();

		QString m_agent;
		QString m_build;
		QString m_dockerUser;

		QLineEdit* m_tagEdit = nullptr;
		QCheckBox* m_tagLatestCheck = nullptr;
		QCheckBox* m_pushModelsCheck = nullptr;
		QLabel* m_previewLabel = nullptr;
		QDialogButtonBox* m_buttons = nullptr;
	};
}
