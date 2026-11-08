#include "SubmittalExportInternal.h"

#include <cctype>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace RocketForgeTraining::SubmittalExportInternal
{
	namespace
	{
		std::string SanitizeIdPart(const std::string& text)
		{
			std::string result;
			bool lastWasSeparator = false;
			for (unsigned char c : text) {
				if (std::isalnum(c)) {
					result.push_back(static_cast<char>(std::tolower(c)));
					lastWasSeparator = false;
				} else if (!lastWasSeparator) {
					result.push_back('-');
					lastWasSeparator = true;
				}
			}

			while (!result.empty() && result.front() == '-')
				result.erase(result.begin());
			while (!result.empty() && result.back() == '-')
				result.pop_back();

			return result.empty() ? "bot" : result;
		}

		std::string MakeCMakeProjectName(const std::string& text)
		{
			std::string result = "RocketForge_";
			for (unsigned char c : text)
				result.push_back(std::isalnum(c) ? static_cast<char>(c) : '_');

			return result;
		}

		std::string FormatTimesteps(uint64_t timesteps)
		{
			constexpr uint64_t MILLION = 1'000'000;
			constexpr uint64_t BILLION = 1'000'000'000;
			const uint64_t unit = timesteps < BILLION ? MILLION : BILLION;
			const char suffix = timesteps < BILLION ? 'M' : 'B';
			const uint64_t rounded = (timesteps + unit / 2) / unit;
			return std::to_string(rounded) + suffix;
		}

		bool IsRLBotMetadataAsset(const fs::path& path)
		{
			const std::string extension = ToLower(path.extension().string());
			return extension == ".toml" || extension == ".png" || extension == ".jpg" ||
				extension == ".jpeg" || extension == ".webp" || extension == ".gif" || extension == ".svg";
		}

		std::string BotTomlKey(const std::string& line)
		{
			const size_t equals = line.find('=');
			return equals == std::string::npos ? std::string() : Trim(line.substr(0, equals));
		}

		bool IsTomlSectionHeader(const std::string& trimmedLine)
		{
			return trimmedLine.size() >= 2 && trimmedLine.front() == '[' && trimmedLine.back() == ']';
		}

		// The launcher supplies Torch's platform-specific loader path at runtime, so a source bot.toml's own entry is dropped.
		bool IsTorchLoaderPathEntry(const std::string& key, const std::string& line)
		{
			return (key == "Path" || key == "PATH" || key == "LD_LIBRARY_PATH") && ToLower(line).find("torch") != std::string::npos;
		}

		bool HasPortableEnvironmentEntry(const std::string& tomlContents)
		{
			std::istringstream input(tomlContents);
			std::string line;
			bool inEnvironment = false;
			while (std::getline(input, line)) {
				const std::string trimmed = Trim(line);
				if (IsTomlSectionHeader(trimmed)) {
					inEnvironment = trimmed == "[settings.environment]";
					continue;
				}

				if (!inEnvironment)
					continue;

				const std::string key = BotTomlKey(line);
				if (!key.empty() && !IsTorchLoaderPathEntry(key, line))
					return true;
			}

			return false;
		}

		std::optional<std::string> ReadTomlQuotedString(const std::string& text, size_t quoteStart, size_t& nextPosition)
		{
			std::string value;
			bool escaped = false;
			for (size_t index = quoteStart + 1; index < text.size(); ++index) {
				const char c = text[index];
				if (escaped) {
					switch (c) {
					case 'n': value.push_back('\n'); break;
					case 'r': value.push_back('\r'); break;
					case 't': value.push_back('\t'); break;
					default: value.push_back(c); break;
					}
					escaped = false;
				} else if (c == '\\') {
					escaped = true;
				} else if (c == '"') {
					nextPosition = index + 1;
					return value;
				} else {
					value.push_back(c);
				}
			}

			return std::nullopt;
		}

		std::optional<std::string> ReadTomlStringValue(const std::string& line)
		{
			const size_t equals = line.find('=');
			const size_t quoteStart = equals == std::string::npos ? std::string::npos : line.find('"', equals + 1);
			if (quoteStart == std::string::npos)
				return std::nullopt;

			size_t nextPosition = quoteStart + 1;
			return ReadTomlQuotedString(line, quoteStart, nextPosition);
		}

		std::vector<std::string> ReadTomlStringArray(const std::string& line)
		{
			std::vector<std::string> result;
			const size_t equals = line.find('=');
			const size_t arrayStart = equals == std::string::npos ? std::string::npos : line.find('[', equals + 1);
			const size_t arrayEnd = arrayStart == std::string::npos ? std::string::npos : line.find(']', arrayStart + 1);
			if (arrayStart == std::string::npos || arrayEnd == std::string::npos)
				return result;

			size_t position = arrayStart + 1;
			while (position < arrayEnd) {
				const size_t quoteStart = line.find('"', position);
				if (quoteStart == std::string::npos || quoteStart >= arrayEnd)
					break;

				const std::optional<std::string> value = ReadTomlQuotedString(line, quoteStart, position);
				if (!value.has_value())
					break;

				result.push_back(value.value());
			}

			return result;
		}

		std::string FormatTomlStringArray(const std::vector<std::string>& values)
		{
			std::ostringstream result;
			result << '[';
			for (size_t index = 0; index < values.size(); ++index) {
				if (index > 0)
					result << ", ";
				result << '"' << EscapeTomlString(values[index]) << '"';
			}

			result << ']';
			return result.str();
		}
	}

	std::string MakeProjectName(
		const std::string& agentName,
		const std::string& buildName,
		const std::string& versionId,
		uint64_t timesteps)
	{
		std::string result = agentName;
		if (!buildName.empty())
			result += " " + buildName;
		if (!versionId.empty())
			result += " " + versionId;

		return result + " " + FormatTimesteps(timesteps);
	}

	std::string MakeGeneratedAgentId(const std::string& botName)
	{
		return "RocketForge/" + SanitizeIdPart(botName);
	}

	SubmittalBotMetadata ReadSubmittalBotMetadata(const fs::path& sourcePath, const std::string& defaultProjectName)
	{
		SubmittalBotMetadata metadata;
		metadata.name = defaultProjectName;
		metadata.agentId = MakeGeneratedAgentId(defaultProjectName);

		if (!fs::exists(sourcePath)) {
			metadata.description = "Exported from RocketForge for Bob the Bot Builder.";
			return metadata;
		}

		std::istringstream input(ReadTextFile(sourcePath));
		std::string line;
		bool inDetails = false;
		bool inSettings = false;
		while (std::getline(input, line)) {
			const std::string trimmed = Trim(line);
			if (IsTomlSectionHeader(trimmed)) {
				inDetails = trimmed == "[details]";
				inSettings = trimmed == "[settings]";
				continue;
			}

			if (inSettings && BotTomlKey(line) == "hivemind") {
				const size_t equals = line.find('=');
				const std::string value = Trim(line.substr(equals + 1));
				metadata.hivemind = Trim(value.substr(0, value.find('#'))) == "true";
			}

			if (!inDetails)
				continue;

			const std::string key = BotTomlKey(line);
			const std::optional<std::string> value = ReadTomlStringValue(line);
			if (key == "description" && value.has_value())
				metadata.description = value.value();
			else if (key == "fun_fact" && value.has_value())
				metadata.funFact = value.value();
			else if (key == "source_link" && value.has_value())
				metadata.sourceLink = value.value();
			else if (key == "developer" && value.has_value())
				metadata.developer = value.value();
			else if (key == "language" && value.has_value())
				metadata.language = value.value();
			else if (key == "tags")
				metadata.tags = ReadTomlStringArray(line);
		}

		return metadata;
	}

	std::string MakePortableBotToml(const fs::path& sourcePath, const SubmittalBotMetadata& metadata, bool hasLoadout)
	{
		const char* hiveValue = metadata.hivemind ? "true" : "false";
		const char* hiveOption = metadata.hivemind ? "on" : "off";

		if (!fs::exists(sourcePath)) {
			std::ostringstream generated;
			generated <<
				"#:schema https://rlbot.org/schemas/agent.json\n"
				"[settings]\n"
				"agent_id = \"" << EscapeTomlString(metadata.agentId) << "\"\n"
				"name = \"" << EscapeTomlString(metadata.name) << "\"\n";
			if (hasLoadout)
				generated << "loadout_file = \"loadout.toml\"\n";
			generated << "hivemind = " << hiveValue << "\n";
			generated <<
				"run_command = \"RLBot.exe --hivemind " << hiveOption << "\"\n"
				"run_command_linux = \"./RLBot --hivemind " << hiveOption << "\"\n\n"
				"[details]\n"
				"description = \"" << EscapeTomlString(metadata.description) << "\"\n"
				"fun_fact = \"" << EscapeTomlString(metadata.funFact) << "\"\n"
				"source_link = \"" << EscapeTomlString(metadata.sourceLink) << "\"\n"
				"developer = \"" << EscapeTomlString(metadata.developer) << "\"\n"
				"language = \"" << EscapeTomlString(metadata.language) << "\"\n"
				"tags = " << FormatTomlStringArray(metadata.tags) << "\n";
			return generated.str();
		}

		const std::string sourceContents = ReadTextFile(sourcePath);
		const bool keepEnvironmentSection = HasPortableEnvironmentEntry(sourceContents);

		std::istringstream input(sourceContents);
		std::ostringstream output;
		std::string line;
		bool inSettings = false;
		bool inEnvironment = false;
		bool inDetails = false;
		bool foundSettings = false;
		bool foundDetails = false;
		bool wroteAgentId = false;
		bool wroteName = false;
		bool wroteHivemind = false;
		bool wroteRunCommand = false;
		bool wroteLinuxRunCommand = false;
		bool wroteDescription = false;
		bool wroteFunFact = false;
		bool wroteSourceLink = false;
		bool wroteDeveloper = false;
		bool wroteLanguage = false;
		bool wroteTags = false;

		auto writeMissingSettings = [&]() {
			if (!wroteHivemind)
				output << "hivemind = " << hiveValue << "\n";
			if (!wroteAgentId)
				output << "agent_id = \"" << EscapeTomlString(metadata.agentId) << "\"\n";
			if (!wroteName)
				output << "name = \"" << EscapeTomlString(metadata.name) << "\"\n";
			if (!wroteRunCommand)
				output << "run_command = \"RLBot.exe --hivemind " << hiveOption << "\"\n";
			if (!wroteLinuxRunCommand)
				output << "run_command_linux = \"./RLBot --hivemind " << hiveOption << "\"\n";
		};

		auto writeMissingDetails = [&]() {
			if (!wroteDescription)
				output << "description = \"" << EscapeTomlString(metadata.description) << "\"\n";
			if (!wroteFunFact)
				output << "fun_fact = \"" << EscapeTomlString(metadata.funFact) << "\"\n";
			if (!wroteSourceLink)
				output << "source_link = \"" << EscapeTomlString(metadata.sourceLink) << "\"\n";
			if (!wroteDeveloper)
				output << "developer = \"" << EscapeTomlString(metadata.developer) << "\"\n";
			if (!wroteLanguage)
				output << "language = \"" << EscapeTomlString(metadata.language) << "\"\n";
			if (!wroteTags)
				output << "tags = " << FormatTomlStringArray(metadata.tags) << "\n";
		};

		while (std::getline(input, line)) {
			const std::string trimmed = Trim(line);
			if (IsTomlSectionHeader(trimmed)) {
				if (inSettings)
					writeMissingSettings();
				if (inDetails)
					writeMissingDetails();

				inSettings = trimmed == "[settings]";
				inEnvironment = trimmed == "[settings.environment]";
				inDetails = trimmed == "[details]";
				foundSettings = foundSettings || inSettings;
				foundDetails = foundDetails || inDetails;
				if (inEnvironment && !keepEnvironmentSection)
					continue;

				output << line << '\n';
				continue;
			}

			if (inEnvironment && !keepEnvironmentSection)
				continue;

			if (!inSettings && !inEnvironment && !inDetails) {
				output << line << '\n';
				continue;
			}

			const std::string key = BotTomlKey(line);
			if (inSettings && key == "agent_id") {
				output << "agent_id = \"" << EscapeTomlString(metadata.agentId) << "\"\n";
				wroteAgentId = true;
			} else if (inSettings && key == "name") {
				output << "name = \"" << EscapeTomlString(metadata.name) << "\"\n";
				wroteName = true;
			} else if (inSettings && key == "hivemind") {
				output << "hivemind = " << hiveValue << "\n";
				wroteHivemind = true;
			} else if (inSettings && key == "run_command") {
				output << "run_command = \"RLBot.exe --hivemind " << hiveOption << "\"\n";
				wroteRunCommand = true;
			} else if (inSettings && key == "run_command_linux") {
				output << "run_command_linux = \"./RLBot --hivemind " << hiveOption << "\"\n";
				wroteLinuxRunCommand = true;
			} else if (inEnvironment && IsTorchLoaderPathEntry(key, line)) {
				continue;
			} else if (inDetails && key == "description") {
				output << "description = \"" << EscapeTomlString(metadata.description) << "\"\n";
				wroteDescription = true;
			} else if (inDetails && key == "fun_fact") {
				output << "fun_fact = \"" << EscapeTomlString(metadata.funFact) << "\"\n";
				wroteFunFact = true;
			} else if (inDetails && key == "source_link") {
				output << "source_link = \"" << EscapeTomlString(metadata.sourceLink) << "\"\n";
				wroteSourceLink = true;
			} else if (inDetails && key == "developer") {
				output << "developer = \"" << EscapeTomlString(metadata.developer) << "\"\n";
				wroteDeveloper = true;
			} else if (inDetails && key == "language") {
				output << "language = \"" << EscapeTomlString(metadata.language) << "\"\n";
				wroteLanguage = true;
			} else if (inDetails && key == "tags") {
				output << "tags = " << FormatTomlStringArray(metadata.tags) << "\n";
				wroteTags = true;
			} else {
				output << line << '\n';
			}
		}

		if (!foundSettings)
			return MakePortableBotToml({}, metadata, hasLoadout);

		if (inSettings)
			writeMissingSettings();
		if (inDetails)
			writeMissingDetails();

		if (!foundDetails) {
			output << "\n[details]\n";
			writeMissingDetails();
		}

		return output.str();
	}

	bool CopySelectedRLBotMetadata(const fs::path& sourceRoot, const fs::path& destinationRoot)
	{
		const fs::path botToml = sourceRoot / "bot.toml";
		if (!fs::exists(botToml))
			return false;

		bool copiedLoadout = false;
		std::istringstream input(ReadTextFile(botToml));
		std::string line;
		while (std::getline(input, line)) {
			const std::string key = BotTomlKey(line);
			if (key != "loadout_file" && key != "logo_file")
				continue;

			const std::optional<std::string> value = ReadTomlStringValue(line);
			if (!value.has_value() || value->empty())
				continue;

			const fs::path relative = fs::path(value.value()).lexically_normal();
			const fs::path source = sourceRoot / relative;
			if (relative.is_absolute() || relative.empty() || *relative.begin() == ".." || !IsPathInside(source, sourceRoot))
				throw std::runtime_error("RLBot metadata references a path outside its folder: " + value.value());
			if (!IsRLBotMetadataAsset(source))
				throw std::runtime_error("Unsupported RLBot metadata asset: " + source.string());

			CopyFile(source, destinationRoot / relative);
			if (key == "loadout_file")
				copiedLoadout = true;
		}

		return copiedLoadout;
	}

	nlohmann::json MakeRuntimeAgentSetupJson(const RocketForge::AgentSetupSnapshot& snapshot)
	{
		nlohmann::json result = RocketForge::AgentSetupSnapshotToJson(snapshot);
		result.erase("add_layer_norm");
		result.erase("optim_type");
		result.erase("activation_type");
		result.erase("critic");
		for (const char* modelName : { "shared_head", "policy" }) {
			if (result.contains(modelName) && result[modelName].is_object())
				result[modelName].erase("optim_type");
		}

		return result;
	}

	std::string MakeRootCMake(const std::string& agentName, const std::string& projectName, bool enableV3)
	{
		std::ostringstream cmake;
		cmake <<
			"cmake_minimum_required(VERSION 3.22)\n\n"
			"project(" << MakeCMakeProjectName(projectName) << ")\n\n"
			"set(DEVICE_FILE \"${CMAKE_SOURCE_DIR}/rlbot/device.txt\")\n"
			"include(cmake/ReadDevice.cmake)\n"
			"set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS \"${DEVICE_FILE}\")\n"
			"if(WIN32)\n"
			"    set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)\n"
			"    set(CMAKE_CONFIGURATION_TYPES \"Release;RelWithDebInfo\" CACHE STRING \"Supported configurations\" FORCE)\n"
			"endif()\n\n"
			"set(LIBRARY_OUTPUT_PATH \"${CMAKE_BINARY_DIR}\")\n"
			"set(EXECUTABLE_OUTPUT_PATH \"${CMAKE_BINARY_DIR}\")\n"
			"set(RLBOT ON CACHE BOOL \"Build the portable RLBot runtime\" FORCE)\n\n"
			"set(RLBOT_CPP_BUILD_BENCHMARK OFF CACHE BOOL \"Exclude RLBotCPP benchmark\" FORCE)\n\n"
			// A cache entry, so v3's detection can't fall back to AUTO and quietly build a v3 bot without its engine.
			"set(ROCKETFORGE_ENABLE_V3 " << (enableV3 ? "ON" : "OFF") <<
			" CACHE STRING \"Build RocketSim v3 for a bot that trains or predicts with it\" FORCE)\n\n"
			"# RLBot's shared PyTorch archive uses the pre-C++11 libstdc++ string ABI.\n"
			"if(NOT WIN32)\n"
			"    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)\n"
			"endif()\n\n"
			"add_subdirectory(ForgeTraining/RLGym)\n"
			"add_subdirectory(ForgeTraining/GigaLearn/GigaLearnInfer)\n"
			"add_subdirectory(Bots)\n"
			"add_subdirectory(RocketForgeRLBot)\n\n"
			"target_compile_definitions(RLBotCore PRIVATE\n"
			"    RLBOT_DEFAULT_AGENT=\"" << EscapeCMakeString(agentName) << "\"\n"
			")\n";

		return cmake.str();
	}

	std::string MakeBobToml(const std::string& projectName)
	{
		std::ostringstream bob;
		bob <<
			"[[config]]\n"
			"project_name = \"" << EscapeTomlString(projectName) << "\"\n"
			"bot_configs = [\"rlbot/bot.toml\"]\n\n"
			"[config.builder_config]\n"
			"builder_type = \"custom\"\n"
			"dockerfile = \"cpp.Dockerfile\"\n";

		return bob.str();
	}
}
