#include "SubmittalExportInternal.h"

#include <cctype>
#include <deque>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

namespace RocketForgeTraining::SubmittalExportInternal
{
	namespace
	{
		bool IsTrainingPlanDirectory(const std::string& name)
		{
			std::string normalized;
			for (unsigned char c : name) {
				if (std::isalnum(c))
					normalized.push_back(static_cast<char>(std::tolower(c)));
			}

			return normalized.ends_with("trainingplan") || normalized.ends_with("trainingplans");
		}

		bool ShouldSkipSourceDirectory(const fs::path& path, bool skipRLBotMetadata, bool skipTrainingPlans, bool skipRLBotDevelopment)
		{
			const std::string name = ToLower(path.filename().string());
			return (skipRLBotMetadata && name == "rlbot") ||
				name == "rlbot_exports" ||
				(skipTrainingPlans && IsTrainingPlanDirectory(name)) ||
				(skipRLBotDevelopment && (name == "benchmark" || name == "cmake" || name == "examples")) ||
				name == ".git" || name == ".vs" || name == ".idea" || name == ".agents" ||
				name == ".claude" || name == "__pycache__" || name == "bob_build" ||
				name == "build" || name == "cmakefiles" || name == "debug" || name == "docs" ||
				name == "documentation" || name == "models" || name == "model benchmarks" ||
				name == "out" || name == "release" || name == "wandb";
		}

		bool ShouldSkipRegistrationScanDirectory(const fs::path& path)
		{
			const bool skipRLBotMetadata = true;
			const bool skipTrainingPlans = true;
			const bool skipRLBotDevelopment = false;
			return ShouldSkipSourceDirectory(path, skipRLBotMetadata, skipTrainingPlans, skipRLBotDevelopment);
		}

		std::optional<fs::path> ParseIncludePath(const std::string& line)
		{
			const std::string includeDirective = "#include";
			const size_t includeStart = line.find(includeDirective);
			if (includeStart == std::string::npos)
				return std::nullopt;

			const size_t pathStart = line.find_first_of("\"<", includeStart + includeDirective.size());
			if (pathStart == std::string::npos)
				return std::nullopt;

			const char pathEndCharacter = line[pathStart] == '"' ? '"' : '>';
			const size_t pathEnd = line.find(pathEndCharacter, pathStart + 1);
			if (pathEnd == std::string::npos)
				return std::nullopt;

			return fs::path(line.substr(pathStart + 1, pathEnd - pathStart - 1));
		}

		// True when `text` calls `macroName(quotedName, ...)`;
		// a comment or string that names the macro and the component separately doesn't count.
		bool InvokesRegistration(const std::string& text, const std::string& macroName, const std::string& quotedName)
		{
			for (size_t at = text.find(macroName); at != std::string::npos; at = text.find(macroName, at + 1)) {
				size_t cursor = at + macroName.size();
				while (cursor < text.size() && std::isspace(static_cast<unsigned char>(text[cursor])))
					++cursor;
				if (cursor >= text.size() || text[cursor] != '(')
					continue;

				++cursor;
				while (cursor < text.size() && std::isspace(static_cast<unsigned char>(text[cursor])))
					++cursor;
				if (text.compare(cursor, quotedName.size(), quotedName) == 0)
					return true;
			}

			return false;
		}

		std::optional<fs::path> FindRegistrationSource(
			const fs::path& sourceRoot,
			const std::string& macroName,
			const std::string& registeredName)
		{
			const std::string quotedName = "\"" + registeredName + "\"";
			std::optional<fs::path> match;

			for (fs::recursive_directory_iterator it(sourceRoot, fs::directory_options::skip_permission_denied), end; it != end; ++it) {
				std::error_code error;
				if (it->is_directory(error)) {
					if (ShouldSkipRegistrationScanDirectory(it->path()))
						it.disable_recursion_pending();
					continue;
				}

				if (error || !it->is_regular_file(error) || !IsCppSourceFile(it->path()))
					continue;

				if (!InvokesRegistration(ReadTextFile(it->path()), macroName, quotedName))
					continue;

				if (match.has_value())
					throw std::runtime_error(
						"More than one source file registers " + registeredName + " with " + macroName +
						". The submittal exporter requires one unambiguous registration unit.");

				match = fs::relative(it->path(), sourceRoot).lexically_normal();
			}

			return match;
		}

		std::set<fs::path> CollectLocalSourceClosure(const fs::path& sourceRoot, const std::vector<fs::path>& seeds)
		{
			std::set<fs::path> result;
			std::deque<fs::path> pending(seeds.begin(), seeds.end());
			const std::vector<std::string> counterpartExtensions = { ".cpp", ".cc", ".cxx", ".h", ".hpp" };

			while (!pending.empty()) {
				const fs::path relative = pending.front().lexically_normal();
				pending.pop_front();
				if (relative.empty() || relative.is_absolute() || *relative.begin() == ".." || result.count(relative))
					continue;

				const fs::path source = sourceRoot / relative;
				if (!fs::exists(source) || !fs::is_regular_file(source) || !IsCppFile(source))
					continue;

				result.insert(relative);

				for (const std::string& extension : counterpartExtensions) {
					fs::path counterpart = relative;
					counterpart.replace_extension(extension);
					if (fs::exists(sourceRoot / counterpart) && !result.count(counterpart))
						pending.push_back(counterpart);
				}

				std::istringstream input(ReadTextFile(source));
				std::string line;
				while (std::getline(input, line)) {
					const std::optional<fs::path> includePath = ParseIncludePath(line);
					if (!includePath.has_value())
						continue;

					fs::path candidate = (relative.parent_path() / includePath.value()).lexically_normal();
					if (!fs::exists(sourceRoot / candidate))
						candidate = includePath->lexically_normal();

					if (!candidate.empty() && !candidate.is_absolute() && *candidate.begin() != ".." &&
						fs::exists(sourceRoot / candidate) && !result.count(candidate))
						pending.push_back(candidate);
				}
			}

			return result;
		}

		std::set<fs::path> CollectIncludedRLGymFiles(const fs::path& botSourceRoot, const std::set<fs::path>& botFiles)
		{
			const fs::path rlgymRoot = fs::path(PROJECT_ROOT) / "ForgeTraining" / "RLGym";
			const fs::path includePrefix = "RLGymCPP";
			const std::vector<std::string> counterpartExtensions = { ".cpp", ".cc", ".cxx", ".h", ".hpp" };
			std::set<fs::path> result;

			for (const fs::path& botFile : botFiles) {
				std::istringstream input(ReadTextFile(botSourceRoot / botFile));
				std::string line;
				while (std::getline(input, line)) {
					const std::optional<fs::path> includePath = ParseIncludePath(line);
					if (!includePath.has_value() || includePath->empty() || *includePath->begin() != includePrefix)
						continue;

					const fs::path relative = (fs::path("src") / includePath.value()).lexically_normal();
					if (!fs::exists(rlgymRoot / relative) || !fs::is_regular_file(rlgymRoot / relative))
						continue;

					result.insert(relative);

					for (const std::string& extension : counterpartExtensions) {
						fs::path counterpart = relative;
						counterpart.replace_extension(extension);
						if (fs::exists(rlgymRoot / counterpart) && fs::is_regular_file(rlgymRoot / counterpart))
							result.insert(counterpart);
					}
				}
			}

			return result;
		}

		bool SourceTreeRegistersAgent(const fs::path& sourceRoot, const std::string& agentName)
		{
			const std::string quotedName = "\"" + agentName + "\"";

			for (fs::recursive_directory_iterator it(sourceRoot, fs::directory_options::skip_permission_denied), end; it != end; ++it) {
				std::error_code error;
				if (it->is_directory(error)) {
					if (ShouldSkipRegistrationScanDirectory(it->path()))
						it.disable_recursion_pending();
					continue;
				}

				if (error || !it->is_regular_file(error))
					continue;

				const std::string extension = ToLower(it->path().extension().string());
				if (extension != ".cpp" && extension != ".cc" && extension != ".cxx" && extension != ".h" && extension != ".hpp")
					continue;

				std::ifstream input(it->path());
				if (!input.good())
					continue;

				std::ostringstream contents;
				contents << input.rdbuf();
				const std::string text = contents.str();
				if (text.find("REGISTER_AGENT") != std::string::npos && text.find(quotedName) != std::string::npos)
					return true;
			}

			return false;
		}

		struct BuiltInComponentExport
		{
			std::set<fs::path> rlgymFiles;
			std::string registrationSource;
		};

		std::optional<BuiltInComponentExport> GetBuiltInObsExport(const std::string& name)
		{
			if (name == "DefaultObsPadded") {
				return BuiltInComponentExport{
					{
						"src/RLGymCPP/ObsBuilders/DefaultObsPadded.cpp",
						"src/RLGymCPP/ObsBuilders/DefaultObsPadded.h",
					},
					"#include <RLGymCPP/ObsBuilders/DefaultObsPadded.h>\n"
					"REGISTER_OBS_BUILDER(\"DefaultObsPadded\", [](const Agent& agent) {\n"
					"    return std::make_unique<RLGC::DefaultObsPadded>(agent.maxTeamSize);\n"
					"})\n"
				};
			}

			if (name == "DefaultObs") {
				return BuiltInComponentExport{
					{
						"src/RLGymCPP/ObsBuilders/DefaultObs.cpp",
						"src/RLGymCPP/ObsBuilders/DefaultObs.h",
					},
					"#include <RLGymCPP/ObsBuilders/DefaultObs.h>\n"
					"REGISTER_OBS_BUILDER(\"DefaultObs\", [](const Agent&) {\n"
					"    return std::make_unique<RLGC::DefaultObs>();\n"
					"})\n"
				};
			}

			if (name == "AdvancedObs") {
				return BuiltInComponentExport{
					{
						"src/RLGymCPP/ObsBuilders/AdvancedObs.cpp",
						"src/RLGymCPP/ObsBuilders/AdvancedObs.h",
					},
					"#include <RLGymCPP/ObsBuilders/AdvancedObs.h>\n"
					"REGISTER_OBS_BUILDER(\"AdvancedObs\", [](const Agent&) {\n"
					"    return std::make_unique<RLGC::AdvancedObs>();\n"
					"})\n"
				};
			}

			if (name == "AdvancedObsPadded") {
				return BuiltInComponentExport{
					{
						"src/RLGymCPP/ObsBuilders/AdvancedObs.cpp",
						"src/RLGymCPP/ObsBuilders/AdvancedObs.h",
						"src/RLGymCPP/ObsBuilders/AdvancedObsPadded.cpp",
						"src/RLGymCPP/ObsBuilders/AdvancedObsPadded.h",
					},
					"#include <RLGymCPP/ObsBuilders/AdvancedObsPadded.h>\n"
					"REGISTER_OBS_BUILDER(\"AdvancedObsPadded\", [](const Agent& agent) {\n"
					"    return std::make_unique<RLGC::AdvancedObsPadded>(agent.maxTeamSize);\n"
					"})\n"
				};
			}

			return std::nullopt;
		}

		std::optional<BuiltInComponentExport> GetBuiltInActionExport(const std::string& name)
		{
			if (name == "DefaultAction") {
				return BuiltInComponentExport{
					{
						"src/RLGymCPP/ActionParsers/DefaultAction.cpp",
						"src/RLGymCPP/ActionParsers/DefaultAction.h",
					},
					"#include <RLGymCPP/ActionParsers/DefaultAction.h>\n"
					"REGISTER_ACTION_PARSER(\"DefaultAction\", [](const Agent&) {\n"
					"    return std::make_unique<RLGC::DefaultAction>();\n"
					"})\n"
				};
			}

			return std::nullopt;
		}
	}

	fs::path ResolveBotSourceRoot(const std::string& agentName, const fs::path& modelsRoot)
	{
		const fs::path botsRoot = fs::path(PROJECT_ROOT) / "Bots";
		const fs::path relativeModels = modelsRoot.lexically_normal().lexically_relative(botsRoot.lexically_normal());
		if (!relativeModels.empty() && *relativeModels.begin() != "..") {
			const fs::path candidate = botsRoot / *relativeModels.begin();
			if (fs::exists(candidate / "CMakeLists.txt"))
				return candidate;
		}

		const fs::path adjacentProject = modelsRoot.parent_path();
		if (fs::exists(adjacentProject / "CMakeLists.txt") && SourceTreeRegistersAgent(adjacentProject, agentName))
			return adjacentProject;

		if (fs::exists(botsRoot)) {
			for (const fs::directory_entry& entry : fs::directory_iterator(botsRoot)) {
				if (!entry.is_directory() || !fs::exists(entry.path() / "CMakeLists.txt"))
					continue;
				if (SourceTreeRegistersAgent(entry.path(), agentName))
					return entry.path();
			}
		}

		throw std::runtime_error(
			"Unable to find the source project that registers agent '" + agentName +
			"'. Keep the agent source under RocketForge/Bots before exporting.");
	}

	BotSourceExport ResolveBotSourceExport(
		const fs::path& botSourceRoot,
		const std::string& agentName,
		const std::string& obsBuilderName,
		const std::string& actionParserName)
	{
		const std::optional<fs::path> agentSource = FindRegistrationSource(botSourceRoot, "REGISTER_AGENT", agentName);
		if (!agentSource.has_value())
			throw std::runtime_error("Unable to find the source file that registers agent '" + agentName + "'.");

		std::vector<fs::path> seeds;
		BotSourceExport result = {};

		const std::string selectedObs = obsBuilderName.empty() ? "DefaultObsPadded" : obsBuilderName;
		const std::optional<fs::path> obsSource = FindRegistrationSource(botSourceRoot, "REGISTER_OBS_BUILDER", selectedObs);
		if (obsSource.has_value()) {
			if (obsSource.value() == agentSource.value())
				throw std::runtime_error(
					"Observation builder '" + selectedObs +
					"' is registered in the agent source. Move that registration into the "
					"observation builder source so the runtime export can omit the agent class.");
			seeds.push_back(obsSource.value());
		} else {
			const std::optional<BuiltInComponentExport> builtIn = GetBuiltInObsExport(selectedObs);
			if (!builtIn.has_value())
				throw std::runtime_error(
					"Unable to export observation builder '" + selectedObs +
					"'. Put its REGISTER_OBS_BUILDER call in its own source file so the exporter "
					"can include only that implementation.");
			result.rlgymFiles.insert(builtIn->rlgymFiles.begin(), builtIn->rlgymFiles.end());
			result.registrationSource += builtIn->registrationSource + "\n";
		}

		const std::string selectedAction = actionParserName.empty() ? "DefaultAction" : actionParserName;
		const std::optional<fs::path> actionSource = FindRegistrationSource(botSourceRoot, "REGISTER_ACTION_PARSER", selectedAction);
		if (actionSource.has_value()) {
			if (actionSource.value() == agentSource.value())
				throw std::runtime_error(
					"Action parser '" + selectedAction +
					"' is registered in the agent source. Move that registration into the "
					"action parser source so the runtime export can omit the agent class.");
			seeds.push_back(actionSource.value());
		} else {
			const std::optional<BuiltInComponentExport> builtIn = GetBuiltInActionExport(selectedAction);
			if (!builtIn.has_value())
				throw std::runtime_error(
					"Unable to export action parser '" + selectedAction +
					"'. Put its REGISTER_ACTION_PARSER call in its own source file so the exporter "
					"can include only that implementation.");
			result.rlgymFiles.insert(builtIn->rlgymFiles.begin(), builtIn->rlgymFiles.end());
			result.registrationSource += builtIn->registrationSource + "\n";
		}

		result.botFiles = CollectLocalSourceClosure(botSourceRoot, seeds);
		if (result.botFiles.count(agentSource.value()))
			throw std::runtime_error(
				"The selected runtime component closure depends on the agent implementation. "
				"Move shared runtime code out of the agent source/header before exporting.");

		const std::set<fs::path> includedRLGymFiles = CollectIncludedRLGymFiles(botSourceRoot, result.botFiles);
		result.rlgymFiles.insert(includedRLGymFiles.begin(), includedRLGymFiles.end());

		result.registrationSource =
			"#include <GigaLearnInfer/Agent.h>\n"
			"#include <GigaLearnInfer/AgentRegistry.h>\n"
			"#include <RLGymCPP/ActionParsers/ActionParser.h>\n"
			"#include <RLGymCPP/ObsBuilders/ObsBuilder.h>\n"
			"#include <GigaLearnInfer/AgentComponentRegistry.h>\n\n"
			"REGISTER_AGENT(\"" + EscapeTomlString(agentName) + "\", Agent)\n\n" +
			result.registrationSource;

		return result;
	}

	std::string MakeBotsCMake(const fs::path& botFolderName, const std::set<fs::path>& botFiles)
	{
		std::set<fs::path> includeDirectories;
		std::ostringstream cmake;
		cmake <<
			"cmake_minimum_required(VERSION 3.8)\n\n"
			"project(ExportedBot)\n\n"
			"add_library(Agents INTERFACE)\n"
			"add_library(ExportedAgent OBJECT\n"
			"    \"ExportedRuntimeRegistration.cpp\"\n";

		for (const fs::path& relative : botFiles) {
			if (!IsCppSourceFile(relative))
				continue;

			const fs::path packageRelative = botFolderName / relative;
			cmake << "    \"" << EscapeCMakeString(packageRelative.generic_string()) << "\"\n";
			includeDirectories.insert(packageRelative.parent_path());
		}

		cmake <<
			")\n\n"
			"target_include_directories(ExportedAgent PRIVATE\n";
		for (const fs::path& includeDirectory : includeDirectories)
			cmake << "    \"" << EscapeCMakeString(includeDirectory.generic_string()) << "\"\n";

		cmake <<
			")\n"
			"target_compile_features(ExportedAgent PRIVATE cxx_std_20)\n"
			"target_link_libraries(ExportedAgent PUBLIC GigaLearnInfer)\n"
			"target_sources(Agents INTERFACE $<TARGET_OBJECTS:ExportedAgent>)\n";

		return cmake.str();
	}
}
