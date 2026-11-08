#pragma once

#include <QString>
#include <QStringList>

namespace ForgeGUI
{
	enum class RemoteSyncKind { Pull, Push, PushImage };

	enum class RemoteSyncOutcome { Succeeded, Failed, Cancelled };

	struct PullOptions
	{
		QString agent;
		QString build;
		bool trainingOnly = false;

		/** Only honored with trainingOnly; the script rejects it otherwise. */
		bool tempCheckpoints = false;

		/** Pull every remote autosave instead of only the newest. Only honored with tempCheckpoints. */
		bool allTempCheckpoints = false;
	};

	struct PushOptions
	{
		QString agent;
		QString build;
	};

	struct PushImageOptions
	{
		QString agent;
		QString build;

		/** Empty lets the script choose its own tag. */
		QString tag;

		bool tagLatest = true;
		bool pushModels = true;
	};

	/** The script that performs a sync, relative to the repo root (e.g. "tools/remote/pull_models.ps1"). */
	QString RemoteSyncScriptRelativePath(RemoteSyncKind kind);

	/**
	 * powershell.exe arguments that run scriptPath for the given options.
	 * Returns an empty list when the agent is blank:
	 * the scripts treat a missing -Agent as "the GUI-selected agent" or "the whole Bots tree", so an empty agent never runs a script.
	 */
	QStringList BuildPowerShellArguments(const QString& scriptPath, const PullOptions& options);
	QStringList BuildPowerShellArguments(const QString& scriptPath, const PushOptions& options);
	QStringList BuildPowerShellArguments(const QString& scriptPath, const PushImageOptions& options);

	/** The TAG= value from `ForgeConsoleTrainer build-info` output, or empty when absent. */
	QString ParseBuildInfoTag(const QString& buildInfoOutput);

	/** Newest ForgeConsoleTrainer.exe under <repoRoot>/out/build, matching the sync scripts' lookup. Empty when none is built. */
	QString FindNewestConsoleTrainer(const QString& repoRoot);

	/**
	 * Turns a stream of process output chunks into complete lines.
	 * "\n" and "\r\n" end a line; a bare "\r" means the tool redrew the line in place, so the text before it is dropped.
	 */
	class RemoteSyncLineSplitter
	{
	public:
		QStringList Append(const QString& text);

		/** Returns the unterminated remainder (empty if none) and resets the splitter. */
		QString Flush();

	private:
		QString _pending;
		bool _carriageReturnPending = false;
	};
}
