#pragma once

#include <GigaLearnInfer/Agent.h>

#include <filesystem>
#include <string>

namespace RocketForgeTraining
{
	/** Resolves the stable Models root for a named agent. **/
	std::filesystem::path ResolveAgentModelsFolder(const std::string& agentName);
	/** Resolves the stable Models root for agent runtime metadata. **/
	std::filesystem::path ResolveAgentModelsFolder(const Agent* agent);
	/**
	 * Returns the agent's canonical Models directory as a path relative to the repo root (PROJECT_ROOT),
	 * using forward slashes — suitable for rclone remote paths on the training pod.
	 * Resolves through the agent's real modelPath, so the agent's registration must be linked in.
	 **/
	std::string ResolveAgentModelsRelativePath(const std::string& agentName);
}
