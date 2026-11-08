#include <gtest/gtest.h>

#include "TestSupport.h"

#if defined(ROCKETFORGE_TESTS_HAVE_QT) && defined(_WIN32)

#include "RemoteSync/RemoteSyncRunner.h"

#include <QDir>
#include <QEventLoop>
#include <QTimer>

#include <functional>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

using namespace ForgeGUI;

namespace
{
	constexpr int RUN_TIMEOUT_MS = 30000;
	constexpr DWORD EXIT_WAIT_MS = 5000;

	struct SyncRun
	{
		QStringList lines;
		RemoteSyncKind kind = RemoteSyncKind::Pull;
		RemoteSyncOutcome outcome = RemoteSyncOutcome::Failed;
		int exitCode = 0;
		bool finished = false;
	};

	// -EncodedCommand sidesteps quoting the script through QProcess's Windows command-line escaping.
	// With redirected output it also serializes the progress stream as CLIXML, so progress is silenced.
	QStringList PowerShellScript(const QString& script)
	{
		const QString quietScript = "$ProgressPreference = 'SilentlyContinue'; " + script;
		const QByteArray utf16(reinterpret_cast<const char*>(quietScript.utf16()), quietScript.size() * 2);
		return { "-NoProfile", "-NonInteractive", "-EncodedCommand", QString::fromLatin1(utf16.toBase64()) };
	}

	SyncRun RunUntilFinished(
		RemoteSyncRunner& runner,
		const QString& program,
		const QStringList& arguments,
		const std::function<void(const QString&)>& onLine = {})
	{
		SyncRun run;
		QEventLoop loop;

		QObject::connect(&runner, &RemoteSyncRunner::LineReceived, &loop, [&run, &onLine](const QString& line) {
			run.lines.append(line);
			if (onLine)
				onLine(line);
		});
		QObject::connect(&runner, &RemoteSyncRunner::Finished, &loop, [&run, &loop](RemoteSyncKind kind, RemoteSyncOutcome outcome, int exitCode) {
			run.kind = kind;
			run.outcome = outcome;
			run.exitCode = exitCode;
			run.finished = true;
			loop.quit();
		});
		QTimer::singleShot(RUN_TIMEOUT_MS, &loop, &QEventLoop::quit);

		runner.Start(RemoteSyncKind::Push, program, arguments, QDir::currentPath());
		if (!run.finished)
			loop.exec();

		return run;
	}

	bool ProcessExitsWithin(DWORD processId, DWORD timeoutMs)
	{
		HANDLE process = OpenProcess(SYNCHRONIZE | PROCESS_TERMINATE, FALSE, processId);
		if (!process)
			return true;

		const bool exited = WaitForSingleObject(process, timeoutMs) == WAIT_OBJECT_0;
		if (!exited)
			TerminateProcess(process, 1);

		CloseHandle(process);
		return exited;
	}
}

TEST(RemoteSyncRunner, StreamsLinesInOrderAndReportsSuccess)
{
	RemoteSyncRunner runner;

	const SyncRun run = RunUntilFinished(runner, "powershell.exe", PowerShellScript("Write-Output 'first'; Write-Output 'second'; exit 0"));

	ASSERT_TRUE(run.finished);
	EXPECT_EQ(run.kind, RemoteSyncKind::Push);
	EXPECT_EQ(run.outcome, RemoteSyncOutcome::Succeeded);
	EXPECT_EQ(run.exitCode, 0);
	const QStringList expectedLines{ "first", "second" };
	EXPECT_EQ(run.lines, expectedLines);
	EXPECT_FALSE(runner.IsRunning());
}

TEST(RemoteSyncRunner, NonZeroExitIsAFailure)
{
	RemoteSyncRunner runner;

	const SyncRun run = RunUntilFinished(runner, "powershell.exe", PowerShellScript("Write-Output 'boom'; exit 3"));

	ASSERT_TRUE(run.finished);
	EXPECT_EQ(run.outcome, RemoteSyncOutcome::Failed);
	EXPECT_EQ(run.exitCode, 3);
}

TEST(RemoteSyncRunner, MissingProgramFailsWithoutHanging)
{
	RemoteSyncRunner runner;

	const SyncRun run = RunUntilFinished(runner, "rocketforge-no-such-program.exe", {});

	ASSERT_TRUE(run.finished);
	EXPECT_EQ(run.outcome, RemoteSyncOutcome::Failed);
	EXPECT_FALSE(run.lines.isEmpty());
	EXPECT_FALSE(runner.IsRunning());
}

TEST(RemoteSyncRunner, CancelKillsTheWholeProcessTree)
{
	RemoteSyncRunner runner;
	DWORD grandchildId = 0;

	const QString script =
		"$info = New-Object System.Diagnostics.ProcessStartInfo 'ping.exe', '-t 127.0.0.1'; "
		"$info.UseShellExecute = $false; "
		"$info.RedirectStandardOutput = $true; "
		"$child = [System.Diagnostics.Process]::Start($info); "
		"Write-Output \"PID=$($child.Id)\"; "
		"$child.WaitForExit()";

	const SyncRun run = RunUntilFinished(runner, "powershell.exe", PowerShellScript(script), [&runner, &grandchildId](const QString& line) {
		if (line.startsWith("PID=")) {
			grandchildId = line.mid(4).toULong();
			runner.Cancel();
		}
	});

	ASSERT_TRUE(run.finished);
	ASSERT_NE(grandchildId, 0u);
	EXPECT_EQ(run.outcome, RemoteSyncOutcome::Cancelled);
	EXPECT_TRUE(ProcessExitsWithin(grandchildId, EXIT_WAIT_MS));
}

#else

TEST(RemoteSyncRunner, RequiresQtOnWindows)
{
	GTEST_SKIP() << "RemoteSyncRunner tests need Qt 6 and Windows job objects";
}

#endif
