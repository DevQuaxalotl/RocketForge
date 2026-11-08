#include "AgentModelPaths.h"

#include <GigaLearnInfer/AgentRegistry.h>

#include <cctype>
#include <memory>

namespace fs = std::filesystem;

namespace
{
	fs::path MakePreferredCopy(fs::path path)
	{
		path.make_preferred();
		return path;
	}

	std::string MakeAgentPathKey(const std::string& text)
	{
		std::string key = {};
		for (unsigned char c : text) {
			if (std::isalnum(c))
				key.push_back((char)std::tolower(c));
		}
		return key;
	}

	fs::path FindModelsRootFromPath(
		const fs::path& path,
		const fs::path& botsRoot)
	{
		if (path.empty())
			return {};

		fs::path current = path;
		if (!fs::is_directory(current))
			current = current.parent_path();

		while (!current.empty()) {
			if (current.filename() == "Models")
				return MakePreferredCopy(current);
			if (current == botsRoot)
				break;

			// A filesystem root is its own parent, so it would never empty `current`.
			fs::path parent = current.parent_path();
			if (parent == current)
				break;
			current = std::move(parent);
		}

		return {};
	}

	fs::path ResolveAgentModelsFolderImpl(const std::string& agentName, const fs::path& modelPath)
	{
		const fs::path botsRoot = fs::path(PROJECT_ROOT) / "Bots";
		const fs::path agentsRoot = botsRoot / "Agents";
		const std::string agentKey = MakeAgentPathKey(agentName);

		// Training artifacts should live under the source agent folder even when
		// the runtime modelPath points somewhere under the build tree.
		if (!agentKey.empty() && fs::exists(agentsRoot)) {
			for (const auto& entry : fs::directory_iterator(agentsRoot)) {
				if (!entry.is_directory())
					continue;
				if (MakeAgentPathKey(entry.path().filename().string()) == agentKey)
					return MakePreferredCopy(entry.path() / "Models");
			}
		}

		// Prefer the registered agent's own modelPath over broad folder discovery.
		// Older sibling folders can contain historical settings for renamed variants.
		if (!modelPath.empty()) {
			const fs::path modelsRoot = FindModelsRootFromPath(modelPath, botsRoot);
			if (!modelsRoot.empty())
				return modelsRoot;
			return MakePreferredCopy(modelPath);
		}

		if (!agentKey.empty() && fs::exists(botsRoot)) {
			for (const auto& entry : fs::recursive_directory_iterator(botsRoot)) {
				if (!entry.is_directory() || entry.path().filename() != "Models")
					continue;

				for (fs::path current = entry.path().parent_path();
					!current.empty() && current != botsRoot.parent_path();
					current = current.parent_path()) {
					if (MakeAgentPathKey(current.filename().string()) == agentKey)
						return MakePreferredCopy(entry.path());
					if (current == botsRoot)
						break;
				}
			}
		}

		return MakePreferredCopy(agentsRoot / (agentName.empty() ? "unknown-agent" : agentName) / "Models");
	}
}

fs::path RocketForgeTraining::ResolveAgentModelsFolder(const std::string& agentName)
{
	std::unique_ptr<Agent> agent = AgentRegistry::Create(agentName);
	return ResolveAgentModelsFolderImpl(agentName, agent ? agent->modelPath : fs::path());
}

fs::path RocketForgeTraining::ResolveAgentModelsFolder(const Agent* agent)
{
	return ResolveAgentModelsFolderImpl(
		agent ? agent->agentName : std::string(),
		agent ? agent->modelPath : fs::path());
}

std::string RocketForgeTraining::ResolveAgentModelsRelativePath(const std::string& agentName)
{
	// Resolve through the agent's real (compile-time) modelPath, not directory-name
	// matching alone. An agent can live in a folder whose name matches neither the
	// agent nor a sibling agent that shares the folder, so only the agent's
	// own modelPath maps it to the correct on-disk Models folder.
	const fs::path absolutePath = ResolveAgentModelsFolder(agentName);
	const fs::path projectRoot = fs::path(PROJECT_ROOT);
	fs::path relativePath = absolutePath.lexically_relative(projectRoot);
	if (relativePath.empty())
		relativePath = absolutePath;
	return relativePath.generic_string();   // forward slashes, for rclone
}
