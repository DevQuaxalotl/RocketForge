#include "RemoteSync/RemoteSyncCommand.h"

#include <QDateTime>
#include <QDir>
#include <QDirIterator>
#include <QFileInfo>

namespace ForgeGUI
{
	namespace
	{
		QStringList BaseArguments(const QString& scriptPath, const QString& agent, const QString& build)
		{
			if (agent.trimmed().isEmpty())
				return {};

			QStringList arguments = { "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", scriptPath, "-Agent", agent };

			if (!build.isEmpty())
				arguments << "-Build" << build;

			arguments << "-PlainProgress";
			return arguments;
		}
	}

	QString RemoteSyncScriptRelativePath(RemoteSyncKind kind)
	{
		switch (kind) {
		case RemoteSyncKind::Pull:
			return "tools/remote/pull_models.ps1";
		case RemoteSyncKind::Push:
			return "tools/remote/push_models.ps1";
		case RemoteSyncKind::PushImage:
			return "tools/remote/push_training_image.ps1";
		}

		return {};
	}

	QStringList BuildPowerShellArguments(const QString& scriptPath, const PullOptions& options)
	{
		QStringList arguments = BaseArguments(scriptPath, options.agent, options.build);
		if (arguments.isEmpty())
			return arguments;

		if (options.trainingOnly) {
			arguments << "-TrainingOnly";
			if (options.tempCheckpoints) {
				arguments << "-TempCheckpoints";
				if (options.allTempCheckpoints)
					arguments << "-AllTempCheckpoints";
			}
		}

		return arguments;
	}

	QStringList BuildPowerShellArguments(const QString& scriptPath, const PushOptions& options)
	{
		return BaseArguments(scriptPath, options.agent, options.build);
	}

	QStringList BuildPowerShellArguments(const QString& scriptPath, const PushImageOptions& options)
	{
		QStringList arguments = BaseArguments(scriptPath, options.agent, options.build);
		if (arguments.isEmpty())
			return arguments;

		if (!options.tag.isEmpty())
			arguments << "-Tag" << options.tag;
		if (!options.tagLatest)
			arguments << "-SkipLatest";
		if (!options.pushModels)
			arguments << "-SkipModels";

		return arguments;
	}

	QString ParseBuildInfoTag(const QString& buildInfoOutput)
	{
		const QString tagPrefix = "TAG=";

		for (const QString& line : buildInfoOutput.split('\n')) {
			const QString trimmed = line.trimmed();
			if (trimmed.startsWith(tagPrefix))
				return trimmed.mid(tagPrefix.size());
		}

		return {};
	}

	QString FindNewestConsoleTrainer(const QString& repoRoot)
	{
		QString newestPath;
		QDateTime newestTime;

		QDirIterator iterator(QDir(repoRoot).filePath("out/build"), { "ForgeConsoleTrainer.exe" }, QDir::Files, QDirIterator::Subdirectories);
		while (iterator.hasNext()) {
			const QFileInfo info(iterator.next());
			if (newestPath.isEmpty() || info.lastModified() > newestTime) {
				newestPath = info.absoluteFilePath();
				newestTime = info.lastModified();
			}
		}

		return newestPath;
	}

	QStringList RemoteSyncLineSplitter::Append(const QString& text)
	{
		QStringList lines;

		for (const QChar character : text) {
			if (_carriageReturnPending) {
				_carriageReturnPending = false;

				if (character == u'\n') {
					lines.append(_pending);
					_pending.clear();
					continue;
				}

				_pending.clear();
			}

			if (character == u'\r') {
				_carriageReturnPending = true;
			} else if (character == u'\n') {
				lines.append(_pending);
				_pending.clear();
			} else {
				_pending.append(character);
			}
		}

		return lines;
	}

	QString RemoteSyncLineSplitter::Flush()
	{
		QString remainder = _pending;
		_pending.clear();
		_carriageReturnPending = false;
		return remainder;
	}
}
