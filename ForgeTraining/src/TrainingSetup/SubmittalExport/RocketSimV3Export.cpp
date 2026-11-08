#include "SubmittalExportInternal.h"

#include <GigaLearnInfer/Agent.h>
#include <GigaLearnInfer/AgentSetupSnapshot.h>
#include <RocketSim/Engine.h>

#include <algorithm>
#include <cstdio>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
// windows.h defines CopyFile as a macro, which would hide the package helper of the same name.
#undef CopyFile
#else
#include <sys/wait.h>
#endif

#ifndef ROCKETFORGE_CARGO_EXECUTABLE
#define ROCKETFORGE_CARGO_EXECUTABLE ""
#endif

namespace fs = std::filesystem;

namespace RocketForgeTraining
{
	bool PackageNeedsV3(const Agent& agent)
	{
		const std::vector<RocketSim::EngineId> engines = RocketForge::RequiredEngines(RocketForge::CaptureAgentSetupSnapshot(agent));
		return std::find(engines.begin(), engines.end(), RocketSim::EngineId::V3) != engines.end();
	}
}

namespace RocketForgeTraining::SubmittalExportInternal
{
	namespace
	{
		constexpr const char* CARGO_VENDOR_CONFIG =
			"[source.crates-io]\n"
			"replace-with = \"vendored-sources\"\n\n"
			"[source.vendored-sources]\n"
			"directory = \"vendor\"\n";

		constexpr size_t CARGO_ERROR_OUTPUT_LIMIT = 4000;

		// True for `[build-dependencies]`, `[build-dependencies.x]` and their `[target.<cfg>.build-dependencies]` forms.
		bool IsBuildDependenciesHeader(const std::string& line)
		{
			const std::string trimmed = Trim(line);
			if (trimmed.empty() || trimmed.front() != '[')
				return false;

			// Dotted keys may quote a cfg() expression, so split on dots outside quotes; the first unquoted ']' ends the header.
			std::vector<std::string> keys(1);
			char quote = 0;

			for (size_t index = trimmed.find_first_not_of('['); index < trimmed.size(); ++index) {
				const char c = trimmed[index];
				if (quote) {
					if (c == quote)
						quote = 0;
					keys.back().push_back(c);
				} else if (c == '\'' || c == '"') {
					quote = c;
					keys.back().push_back(c);
				} else if (c == ']') {
					break;
				} else if (c == '.') {
					keys.emplace_back();
				} else if (c != ' ' && c != '\t') {
					keys.back().push_back(c);
				}
			}

			const auto isBuildDependencies = [](const std::string& key) {
				return key == "build-dependencies" || key == "build_dependencies";
			};
			return isBuildDependencies(keys[0]) || (keys[0] == "target" && keys.size() >= 3 && isBuildDependencies(keys[2]));
		}

		fs::path CargoExecutable()
		{
			return fs::path(ROCKETFORGE_CARGO_EXECUTABLE);
		}

#ifdef _WIN32
		// Quotes one argument so CommandLineToArgvW and the MSVC runtime read it back unchanged.
		std::wstring QuoteWindowsArgument(const std::wstring& argument)
		{
			std::wstring quoted = L"\"";
			size_t backslashes = 0;
			for (wchar_t c : argument) {
				if (c == L'\\') {
					++backslashes;
					continue;
				}

				quoted.append(c == L'"' ? backslashes * 2 + 1 : backslashes, L'\\');
				backslashes = 0;
				quoted.push_back(c);
			}
			quoted.append(backslashes * 2, L'\\');
			quoted.push_back(L'"');

			return quoted;
		}

		// Runs cargo without a console window, so a GUI export doesn't flash one. Returns the exit code.
		int RunProcess(
			const fs::path& executable,
			const std::vector<std::string>& arguments,
			const fs::path& workingDirectory,
			std::string& output)
		{
			std::wstring commandLine = QuoteWindowsArgument(executable.wstring());
			for (const std::string& argument : arguments)
				commandLine += L" " + QuoteWindowsArgument(fs::path(argument).wstring());

			SECURITY_ATTRIBUTES pipeAttributes{ sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE };
			HANDLE readPipe = nullptr;
			HANDLE writePipe = nullptr;
			if (!CreatePipe(&readPipe, &writePipe, &pipeAttributes, 0))
				throw std::runtime_error("Unable to create a pipe for cargo's output.");
			SetHandleInformation(readPipe, HANDLE_FLAG_INHERIT, 0);

			STARTUPINFOW startupInfo{};
			startupInfo.cb = sizeof(startupInfo);
			startupInfo.dwFlags = STARTF_USESTDHANDLES;
			startupInfo.hStdInput = nullptr;
			startupInfo.hStdOutput = writePipe;
			startupInfo.hStdError = writePipe;

			const BOOL inheritHandles = TRUE;
			PROCESS_INFORMATION processInfo{};
			const BOOL started = CreateProcessW(
				nullptr,
				commandLine.data(),
				nullptr,
				nullptr,
				inheritHandles,
				CREATE_NO_WINDOW,
				nullptr,
				workingDirectory.wstring().c_str(),
				&startupInfo,
				&processInfo);
			CloseHandle(writePipe);
			if (!started) {
				CloseHandle(readPipe);
				throw std::runtime_error("Unable to start cargo at " + executable.string() + ".");
			}

			char buffer[4096];
			DWORD bytesRead = 0;
			while (ReadFile(readPipe, buffer, sizeof(buffer), &bytesRead, nullptr) && bytesRead > 0)
				output.append(buffer, bytesRead);
			CloseHandle(readPipe);

			WaitForSingleObject(processInfo.hProcess, INFINITE);
			DWORD exitCode = 1;
			GetExitCodeProcess(processInfo.hProcess, &exitCode);
			CloseHandle(processInfo.hThread);
			CloseHandle(processInfo.hProcess);

			return static_cast<int>(exitCode);
		}
#else
		std::string QuotePosixArgument(const std::string& argument)
		{
			std::string quoted = "'";
			for (char c : argument) {
				if (c == '\'')
					quoted += "'\\''";
				else
					quoted.push_back(c);
			}

			return quoted + "'";
		}

		int RunProcess(
			const fs::path& executable,
			const std::vector<std::string>& arguments,
			const fs::path& workingDirectory,
			std::string& output)
		{
			std::string command = "cd " + QuotePosixArgument(workingDirectory.string()) + " && " + QuotePosixArgument(executable.string());
			for (const std::string& argument : arguments)
				command += " " + QuotePosixArgument(argument);
			command += " 2>&1";

			FILE* pipe = popen(command.c_str(), "r");
			if (!pipe)
				throw std::runtime_error("Unable to start cargo at " + executable.string() + ".");

			char buffer[4096];
			for (size_t bytesRead; (bytesRead = fread(buffer, 1, sizeof(buffer), pipe)) > 0;)
				output.append(buffer, bytesRead);

			const int status = pclose(pipe);
			return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
		}
#endif

		CargoRunner& ActiveCargoRunner()
		{
			static CargoRunner runner = RunConfiguredCargo;
			return runner;
		}
	}

	void RunConfiguredCargo(const std::vector<std::string>& arguments, const fs::path& workingDirectory, const std::string& purpose)
	{
		std::string output;
		const int exitCode = RunProcess(CargoExecutable(), arguments, workingDirectory, output);
		if (exitCode == 0)
			return;

		if (output.size() > CARGO_ERROR_OUTPUT_LIMIT)
			output = "..." + output.substr(output.size() - CARGO_ERROR_OUTPUT_LIMIT);

		throw std::runtime_error("cargo failed to " + purpose + " (exit code " + std::to_string(exitCode) + "):\n" + output);
	}

	Testing::ScopedCargoRunner::ScopedCargoRunner(CargoRunner runner) : _previous(std::move(ActiveCargoRunner()))
	{
		ActiveCargoRunner() = std::move(runner);
	}

	Testing::ScopedCargoRunner::~ScopedCargoRunner()
	{
		ActiveCargoRunner() = std::move(_previous);
	}

	std::string RemoveBuildDependencies(const std::string& manifest)
	{
		std::istringstream input(NormalizeLineEndingsToLf(manifest));
		std::ostringstream output;
		bool inBuildDependencies = false;
		for (std::string line; std::getline(input, line);) {
			const std::string trimmed = Trim(line);
			if (!trimmed.empty() && trimmed.front() == '[')
				inBuildDependencies = IsBuildDependenciesHeader(trimmed);

			if (!inBuildDependencies)
				output << line << "\n";
		}

		return output.str();
	}

	void RequireV3PackageTools()
	{
		RocketSim::RequireEngine(RocketSim::EngineId::V3);

		const fs::path cargo = CargoExecutable();
		std::error_code error;
		if (cargo.empty() || !fs::is_regular_file(cargo, error)) {
			throw std::runtime_error(
				"Exporting a RocketSim v3 bot vendors its Rust crates with cargo, "
				"but the cargo this build was configured with (" + cargo.string() + ") is missing. "
				"Reinstall Rust (tools/setup/install_rust.ps1 or tools/setup/install_rust.sh) and re-run CMake.");
		}
	}

	void CopyRocketSimV3(const fs::path& projectRoot, const fs::path& packageRoot)
	{
		const fs::path sourceRoot = projectRoot / "ForgeTraining" / "RLGym" / "RocketSimV3";
		const fs::path destinationRoot = packageRoot / "ForgeTraining" / "RLGym" / "RocketSimV3";
		CopyFile(sourceRoot / "CMakeLists.txt", destinationRoot / "CMakeLists.txt");
		CopyFile(projectRoot / "cmake" / "DetectRocketSimV3.cmake", packageRoot / "cmake" / "DetectRocketSimV3.cmake");

		// Upstream's workspace manifest supplies the crate's inherited package fields; its other members aren't needed.
		const fs::path upstreamSource = sourceRoot / "upstream";
		const fs::path upstreamDestination = destinationRoot / "upstream";
		for (const char* file : { "Cargo.toml", "LICENSE", "README.md" })
			CopyFile(upstreamSource / file, upstreamDestination / file);
		CopyFile(upstreamSource / "rocketsim" / "Cargo.toml", upstreamDestination / "rocketsim" / "Cargo.toml");
		CopyRuntimeAssetTree(upstreamSource / "rocketsim" / "src", upstreamDestination / "rocketsim" / "src");

		const fs::path bindingsSource = sourceRoot / "bindings";
		const fs::path bindingsDestination = destinationRoot / "bindings";
		for (const char* file : { "Cargo.lock", "rust-toolchain.toml", "README.md" })
			CopyFile(bindingsSource / file, bindingsDestination / file);

		CopyRuntimeAssetTree(bindingsSource / "include", bindingsDestination / "include");
		CopyRuntimeAssetTree(bindingsSource / "src", bindingsDestination / "src");

		// The build script only checks the checked-in C header against cbindgen's output, and packages ship that header.
		// Dropping it keeps cbindgen's crates out of the vendored sources.
		WriteTextFile(bindingsDestination / "Cargo.toml", RemoveBuildDependencies(ReadTextFile(bindingsSource / "Cargo.toml")));

		// Resolving the trimmed manifest prunes the lock file without moving any remaining crate off its locked version.
		const CargoRunner& runCargo = ActiveCargoRunner();
		runCargo(
			{ "metadata", "--format-version", "1", "--manifest-path", "Cargo.toml" },
			bindingsDestination,
			"prune the RocketSim v3 lock file");
		runCargo({ "vendor", "--locked", "--manifest-path", "Cargo.toml", "vendor" }, bindingsDestination, "vendor the RocketSim v3 crates");

		WriteTextFile(bindingsDestination / ".cargo" / "config.toml", CARGO_VENDOR_CONFIG);
	}
}
