#include "RemoteSync/RemoteSyncRunner.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <tlhelp32.h>
#endif

namespace ForgeGUI
{
	namespace
	{
		constexpr int CANCELLED_EXIT_CODE = 1;

#ifdef _WIN32
		HANDLE CreateKillOnCloseJob()
		{
			HANDLE job = CreateJobObjectW(nullptr, nullptr);
			if (!job)
				return nullptr;

			JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
			limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
			if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits))) {
				CloseHandle(job);
				return nullptr;
			}

			return job;
		}

		bool AssignToJob(HANDLE job, DWORD processId)
		{
			HANDLE process = OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, FALSE, processId);
			if (!process)
				return false;

			const bool assigned = AssignProcessToJobObject(job, process) != FALSE;
			CloseHandle(process);
			return assigned;
		}

		// QProcess does not expose the primary thread handle, so find the suspended thread by owner pid.
		// Returns whether at least one thread was resumed.
		bool ResumeProcessThreads(DWORD processId)
		{
			HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
			if (snapshot == INVALID_HANDLE_VALUE)
				return false;

			bool resumed = false;

			THREADENTRY32 entry{};
			entry.dwSize = sizeof(entry);

			for (BOOL found = Thread32First(snapshot, &entry); found; found = Thread32Next(snapshot, &entry)) {
				if (entry.th32OwnerProcessID != processId)
					continue;

				HANDLE thread = OpenThread(THREAD_SUSPEND_RESUME, FALSE, entry.th32ThreadID);
				if (thread) {
					resumed |= ResumeThread(thread) != static_cast<DWORD>(-1);
					CloseHandle(thread);
				}
			}

			CloseHandle(snapshot);
			return resumed;
		}
#endif
	}

	RemoteSyncRunner::RemoteSyncRunner(QObject* parent)
		: QObject(parent)
	{
		m_process = new QProcess(this);
		m_process->setProcessChannelMode(QProcess::MergedChannels);
		m_process->setStandardInputFile(QProcess::nullDevice());

#ifdef _WIN32
		// Suspended until OnStarted has put it in the job, so nothing it spawns can escape Cancel.
		m_process->setCreateProcessArgumentsModifier([](QProcess::CreateProcessArguments* arguments) { arguments->flags |= CREATE_SUSPENDED; });
#endif

		connect(m_process, &QProcess::started,                 this, &RemoteSyncRunner::OnStarted);
		connect(m_process, &QProcess::readyReadStandardOutput, this, &RemoteSyncRunner::OnReadyRead);
		connect(m_process, &QProcess::finished,                this, &RemoteSyncRunner::OnProcessFinished);
		connect(m_process, &QProcess::errorOccurred,           this, &RemoteSyncRunner::OnErrorOccurred);
	}

	RemoteSyncRunner::~RemoteSyncRunner()
	{
		m_process->disconnect(this);
		CloseJob();
	}

	void RemoteSyncRunner::Start(RemoteSyncKind kind, const QString& program, const QStringList& arguments, const QString& workingDirectory)
	{
		Q_ASSERT(!m_running);

		m_kind = kind;
		m_running = true;
		m_cancelRequested = false;
		m_decoder.resetState();
		m_splitter.Flush();

		m_process->setWorkingDirectory(workingDirectory);
		m_process->start(program, arguments);
	}

	void RemoteSyncRunner::Cancel()
	{
		if (!m_running)
			return;

		m_cancelRequested = true;

#ifdef _WIN32
		if (m_job) {
			TerminateJobObject(static_cast<HANDLE>(m_job), CANCELLED_EXIT_CODE);
			return;
		}
#endif

		m_process->kill();
	}

	void RemoteSyncRunner::CancelAndWait(int timeoutMs)
	{
		Cancel();

		if (m_running)
			m_process->waitForFinished(timeoutMs);
	}

	bool RemoteSyncRunner::IsRunning() const
	{
		return m_running;
	}

	RemoteSyncKind RemoteSyncRunner::Kind() const
	{
		return m_kind;
	}

	void RemoteSyncRunner::OnStarted()
	{
#ifdef _WIN32
		const DWORD processId = static_cast<DWORD>(m_process->processId());

		HANDLE job = CreateKillOnCloseJob();
		if (job && AssignToJob(job, processId)) {
			m_job = job;
		} else {
			if (job)
				CloseHandle(job);

			emit LineReceived("Could not attach the sync to a job object; Cancel will stop PowerShell but not its child processes.");
		}

		if (!ResumeProcessThreads(processId)) {
			emit LineReceived("Could not resume the sync process, so it cannot run.");
			Cancel();
		}
#endif
	}

	void RemoteSyncRunner::OnReadyRead()
	{
		const QString text = m_decoder.decode(m_process->readAllStandardOutput());

		for (const QString& line : m_splitter.Append(text))
			emit LineReceived(line);
	}

	void RemoteSyncRunner::OnProcessFinished(int exitCode, QProcess::ExitStatus exitStatus)
	{
		OnReadyRead();

		if (m_cancelRequested)
			Complete(RemoteSyncOutcome::Cancelled, exitCode);
		else if (exitStatus == QProcess::NormalExit && exitCode == 0)
			Complete(RemoteSyncOutcome::Succeeded, exitCode);
		else
			Complete(RemoteSyncOutcome::Failed, exitCode);
	}

	void RemoteSyncRunner::OnErrorOccurred(QProcess::ProcessError error)
	{
		// Only FailedToStart ends a run without a finished() signal.
		if (error != QProcess::FailedToStart || !m_running)
			return;

		emit LineReceived("Could not start " + m_process->program() + ": " + m_process->errorString());
		Complete(RemoteSyncOutcome::Failed, -1);
	}

	void RemoteSyncRunner::Complete(RemoteSyncOutcome outcome, int exitCode)
	{
		const QString remainder = m_splitter.Flush();
		if (!remainder.isEmpty())
			emit LineReceived(remainder);

		CloseJob();
		m_running = false;

		emit Finished(m_kind, outcome, exitCode);
	}

	void RemoteSyncRunner::CloseJob()
	{
#ifdef _WIN32
		// Kill-on-close: closing the last handle also ends anything the script left running.
		if (m_job)
			CloseHandle(static_cast<HANDLE>(m_job));
#endif

		m_job = nullptr;
	}
}
