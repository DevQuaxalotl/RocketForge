#pragma once

#include "RemoteSync/RemoteSyncCommand.h"

#include <QObject>
#include <QProcess>
#include <QStringDecoder>

namespace ForgeGUI
{
	/**
	 * Runs one sync process at a time and streams its merged stdout/stderr as lines.
	 * On Windows the process is started suspended and placed in a job object before it runs,
	 * so Cancel (and destruction) kill every process it spawns, not just the direct child.
	 */
	class RemoteSyncRunner : public QObject
	{
		Q_OBJECT

	public:
		explicit RemoteSyncRunner(QObject* parent = nullptr);
		~RemoteSyncRunner() override;

		/** Must not be called while IsRunning(). Finished is always emitted exactly once per Start. */
		void Start(RemoteSyncKind kind, const QString& program, const QStringList& arguments, const QString& workingDirectory);

		void Cancel();

		/** Cancels, then blocks until the process has exited or timeoutMs elapses. */
		void CancelAndWait(int timeoutMs);

		bool IsRunning() const;
		RemoteSyncKind Kind() const;

	signals:
		void LineReceived(const QString& line);
		void Finished(ForgeGUI::RemoteSyncKind kind, ForgeGUI::RemoteSyncOutcome outcome, int exitCode);

	private:
		void OnStarted();
		void OnReadyRead();
		void OnProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
		void OnErrorOccurred(QProcess::ProcessError error);
		void Complete(RemoteSyncOutcome outcome, int exitCode);
		void CloseJob();

		QProcess* m_process = nullptr;
		QStringDecoder m_decoder{ QStringConverter::System };
		RemoteSyncLineSplitter m_splitter;
		RemoteSyncKind m_kind = RemoteSyncKind::Pull;
		bool m_running = false;
		bool m_cancelRequested = false;

		// NOTE: a Win32 HANDLE kept as void* so this header does not pull in <windows.h>.
		void* m_job = nullptr;
	};
}
