#include "SubmittalExportInternal.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace fs = std::filesystem;

namespace RocketForgeTraining::SubmittalExportInternal
{
	std::string ToLower(std::string text)
	{
		std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
			return static_cast<char>(std::tolower(c));
		});
		return text;
	}

	std::string Trim(std::string text)
	{
		const size_t first = text.find_first_not_of(" \t\r\n");
		if (first == std::string::npos)
			return {};
		const size_t last = text.find_last_not_of(" \t\r\n");
		return text.substr(first, last - first + 1);
	}

	std::string EscapeTomlString(const std::string& text)
	{
		std::string escaped;
		for (char c : text) {
			switch (c) {
			case '\\': escaped += "\\\\"; break;
			case '"': escaped += "\\\""; break;
			case '\n': escaped += "\\n"; break;
			case '\r': escaped += "\\r"; break;
			case '\t': escaped += "\\t"; break;
			default: escaped.push_back(c); break;
			}
		}
		return escaped;
	}

	std::string EscapeCMakeString(const std::string& text)
	{
		std::string escaped;
		for (char c : text) {
			if (c == '\\' || c == '"' || c == ';')
				escaped.push_back('\\');
			escaped.push_back(c);
		}
		return escaped;
	}

	std::string NormalizeLineEndingsToLf(const std::string& text)
	{
		std::string normalized;
		normalized.reserve(text.size());
		for (size_t index = 0; index < text.size(); ++index) {
			if (text[index] != '\r') {
				normalized.push_back(text[index]);
				continue;
			}

			normalized.push_back('\n');
			if (index + 1 < text.size() && text[index + 1] == '\n')
				++index;
		}

		return normalized;
	}

	std::string ReadTextFile(const fs::path& path)
	{
		std::ifstream input(path, std::ios::binary);
		if (!input.good())
			throw std::runtime_error("Unable to read " + path.string());

		std::ostringstream contents;
		contents << input.rdbuf();
		return contents.str();
	}

	void WriteTextFile(const fs::path& path, const std::string& contents)
	{
		fs::create_directories(path.parent_path());
		std::ofstream output(path, std::ios::binary);
		if (!output.good())
			throw std::runtime_error("Unable to write " + path.string());

		output << contents;
		if (!output.good())
			throw std::runtime_error("Unable to finish writing " + path.string());
	}

	void CopyFile(const fs::path& source, const fs::path& destination)
	{
		if (!fs::exists(source) || !fs::is_regular_file(source))
			throw std::runtime_error("Required submittal source file is missing: " + source.string());

		fs::create_directories(destination.parent_path());
		std::error_code error;
		fs::copy_file(source, destination, fs::copy_options::overwrite_existing, error);
		if (error)
			throw std::runtime_error("Unable to copy " + source.string() + " to " + destination.string() + ": " + error.message());
	}

	namespace
	{
		std::string NormalizePathForComparison(const fs::path& path)
		{
			std::error_code error;
			fs::path normalized = fs::absolute(path, error).lexically_normal();
			std::string result = normalized.generic_string();
#ifdef _WIN32
			result = ToLower(result);
#endif
			while (result.size() > 1 && result.back() == '/')
				result.pop_back();

			return result;
		}
	}

	bool IsPathInside(const fs::path& path, const fs::path& parent)
	{
		const std::string normalizedPath = NormalizePathForComparison(path);
		const std::string normalizedParent = NormalizePathForComparison(parent);
		return normalizedPath == normalizedParent ||
			(normalizedPath.size() > normalizedParent.size() &&
				normalizedPath.compare(0, normalizedParent.size(), normalizedParent) == 0 &&
				normalizedPath[normalizedParent.size()] == '/');
	}

	void CopyRuntimeAssetTree(const fs::path& sourceRoot, const fs::path& destinationRoot)
	{
		if (!fs::exists(sourceRoot) || !fs::is_directory(sourceRoot))
			throw std::runtime_error("Required runtime asset directory is missing: " + sourceRoot.string());

		for (const fs::directory_entry& entry : fs::recursive_directory_iterator(sourceRoot)) {
			if (!entry.is_regular_file())
				continue;
			CopyFile(entry.path(), destinationRoot / fs::relative(entry.path(), sourceRoot));
		}
	}

	bool IsCppSourceFile(const fs::path& path)
	{
		const std::string extension = ToLower(path.extension().string());
		return extension == ".cpp" || extension == ".cc" || extension == ".cxx";
	}

	bool IsCppFile(const fs::path& path)
	{
		const std::string extension = ToLower(path.extension().string());
		return IsCppSourceFile(path) || extension == ".h" || extension == ".hpp";
	}

	void CopyExplicitFiles(const fs::path& sourceRoot, const fs::path& destinationRoot, const std::set<fs::path>& relativePaths)
	{
		for (const fs::path& relative : relativePaths)
			CopyFile(sourceRoot / relative, destinationRoot / relative);
	}

	void CopyCodeTree(const fs::path& sourceRoot, const fs::path& destinationRoot, const std::set<fs::path>& excludedRelativeRoots)
	{
		if (!fs::exists(sourceRoot) || !fs::is_directory(sourceRoot))
			throw std::runtime_error("Required submittal source directory is missing: " + sourceRoot.string());

		for (fs::recursive_directory_iterator it(sourceRoot), end; it != end; ++it) {
			const fs::path source = it->path();
			const fs::path relative = fs::relative(source, sourceRoot).lexically_normal();
			const bool excluded = std::any_of(excludedRelativeRoots.begin(), excludedRelativeRoots.end(), [&](const fs::path& root) {
				return IsPathInside(relative, root);
			});

			if (it->is_directory()) {
				if (excluded)
					it.disable_recursion_pending();
				continue;
			}

			if (!excluded && it->is_regular_file() && IsCppFile(source))
				CopyFile(source, destinationRoot / relative);
		}
	}

	fs::path MakeScratchRoot(const fs::path& outputParent, const std::string& prefix)
	{
		const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
		for (int attempt = 0; attempt < 100; ++attempt) {
			const fs::path candidate = outputParent / (prefix + std::to_string(nonce) + "-" + std::to_string(attempt));
			if (!fs::exists(candidate))
				return candidate;
		}

		throw std::runtime_error("Unable to reserve a temporary submittal export folder.");
	}

	void CommitPackage(const fs::path& temporaryRoot, const fs::path& packageRoot, bool overwrite)
	{
		if (!fs::exists(packageRoot)) {
			fs::rename(temporaryRoot, packageRoot);
			return;
		}

		if (!overwrite)
			throw std::runtime_error("Submittal export already exists: " + packageRoot.string());

		const fs::path backupRoot = MakeScratchRoot(packageRoot.parent_path(), ".rfb-");
		fs::rename(packageRoot, backupRoot);
		try {
			fs::rename(temporaryRoot, packageRoot);
		} catch (...) {
			std::error_code restoreError;
			fs::rename(backupRoot, packageRoot, restoreError);
			throw;
		}

		std::error_code error;
		fs::remove_all(backupRoot, error);
		if (error)
			throw std::runtime_error("The export succeeded, but the previous package backup could not be removed: " + backupRoot.string());
	}
}
