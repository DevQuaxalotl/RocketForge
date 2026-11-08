#pragma once

#include "RemoteSync/RemoteSyncCommand.h"

#include <QGroupBox>

class QCheckBox;
class QLabel;
class QPushButton;

namespace ForgeGUI
{
	class RemoteSyncRunner;

	/**
	 * Training-tab controls for the remote-training sync scripts (pull, push, training image).
	 * Emits RunningChanged(true) before a process starts and RunningChanged(false) before Finished,
	 * so a sync that fails to start still unlocks whatever RunningChanged(true) locked.
	 */
	class RemoteSyncPanel : public QGroupBox
	{
		Q_OBJECT

	public:
		explicit RemoteSyncPanel(const QString& repoRoot, QWidget* parent = nullptr);

		void SetTarget(const QString& agent, const QString& build);

		/** Which syncs may start; the owner derives these from the other runs in progress. */
		void SetAvailability(bool canPush, bool canPull);

		bool IsRunning() const;
		void CancelAndWait();

	signals:
		/** Emitted before anything reads forge_settings.json from disk, so pending UI edits can be saved first. */
		void FlushSettingsRequested();

		void RunningChanged(bool running, ForgeGUI::RemoteSyncKind kind);
		void LineReceived(const QString& line);
		void Finished(ForgeGUI::RemoteSyncKind kind, ForgeGUI::RemoteSyncOutcome outcome, int exitCode);

	private:
		void OnPullClicked();
		void OnPushClicked();
		void OnPushImageClicked();
		void OnRunnerLine(const QString& line);
		void OnRunnerFinished(RemoteSyncKind kind, RemoteSyncOutcome outcome, int exitCode);
		void StartSync(RemoteSyncKind kind, const QStringList& arguments);
		QString ScriptPath(RemoteSyncKind kind) const;
		void SetStatus(const QString& text, const QColor& color);
		void UpdateControls();

		QString m_repoRoot;
		QString m_agent;
		QString m_build;
		bool m_canPush = false;
		bool m_canPull = false;

		RemoteSyncRunner* m_runner = nullptr;
		QPushButton* m_pullButton = nullptr;
		QPushButton* m_pushButton = nullptr;
		QCheckBox* m_allModelsCheck = nullptr;
		QCheckBox* m_tempCheckpointsCheck = nullptr;
		QCheckBox* m_allTempCheckpointsCheck = nullptr;
		QPushButton* m_pushImageButton = nullptr;
		QLabel* m_statusLabel = nullptr;
		QPushButton* m_cancelButton = nullptr;
	};
}
