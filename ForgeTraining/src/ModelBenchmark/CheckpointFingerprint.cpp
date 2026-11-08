#include "CheckpointFingerprint.h"
#include "CheckpointFingerprintInternal.h"

#include <Utils/FileRevision.h>
#include <Utils/Sha256.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <mutex>
#include <random>
#include <vector>

using namespace RocketForge;
namespace fs = std::filesystem;

namespace
{
	std::string Utf8(const fs::path& path)
	{
		const std::u8string text = path.u8string();
		return std::string(text.begin(), text.end());
	}

	// Qt's Unicode toUpper differs only for non-ASCII letters that fold to OPTIM or CRITIC, so ASCII is deliberately enough.
	std::string AsciiUpper(std::string text)
	{
		std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
		return text;
	}

	// POLICY.lt and SHARED_HEAD.lt, else every .lt that is not optimizer or critic state.
	// False when the folder can't be listed.
	bool CollectWeightFiles(const fs::path& checkpointPath, std::vector<fs::path>& files)
	{
		std::error_code filesystemError;
		for (const char* name : { "POLICY.lt", "SHARED_HEAD.lt" }) {
			const fs::path candidate = checkpointPath / name;
			filesystemError.clear();
			if (fs::is_regular_file(candidate, filesystemError) && !filesystemError)
				files.push_back(candidate);
		}

		if (!files.empty())
			return true;

		filesystemError.clear();
		fs::directory_iterator iterator(checkpointPath, filesystemError);
		const fs::directory_iterator end;
		for (; iterator != end && !filesystemError; iterator.increment(filesystemError)) {
			const fs::directory_entry& entry = *iterator;
			std::error_code entryError;
			if (!entry.is_regular_file(entryError) || entryError || entry.path().extension() != ".lt")
				continue;

			const std::string name = AsciiUpper(entry.path().filename().string());
			if (name.find("OPTIM") == std::string::npos && name.find("CRITIC") == std::string::npos)
				files.push_back(entry.path());
		}

		return !filesystemError;
	}

	// Imported inference policies can use TorchScript weights instead of POLICY.lt,
	// so the manifest and the artifact it declares are hashed too; provenance text alone is not a weight identity.
	// False when the manifest exists but is unreadable or declares an artifact outside the checkpoint folder.
	bool AddInferenceManifest(const fs::path& checkpointPath, std::vector<fs::path>& files)
	{
		const fs::path manifestPath = checkpointPath / "INFERENCE_POLICY.json";
		if (!fs::exists(manifestPath))
			return true;

		std::ifstream manifest(manifestPath, std::ios::binary);
		if (!manifest)
			return false;

		const bool allowExceptions = false;
		const nlohmann::json document = nlohmann::json::parse(manifest, nullptr, allowExceptions);
		if (!document.is_object())
			return false;

		const auto artifactField = document.find("artifact");
		if (artifactField == document.end() || !artifactField->is_string())
			return false;

		const std::string artifactText = artifactField->get<std::string>();
		const fs::path artifactName(std::u8string(artifactText.begin(), artifactText.end()));
		if (artifactName.empty() || artifactName.is_absolute())
			return false;

		const fs::path artifact = fs::weakly_canonical(checkpointPath / artifactName);
		const fs::path relative = artifact.lexically_relative(fs::canonical(checkpointPath));
		if (relative.empty() || *relative.begin() == ".." || !fs::is_regular_file(artifact))
			return false;

		files.push_back(manifestPath);
		files.push_back(artifact);
		return true;
	}

	// Revisions of everything the digest depends on; a cache entry is valid only while this is unchanged.
	nlohmann::json RevisionStamp(const fs::path& checkpointPath, const fs::path& trainingSetupPath, const std::vector<fs::path>& files)
	{
		nlohmann::json stamp = nlohmann::json::object();
		stamp["checkpoint-directory"] = FileRevision(checkpointPath);
		stamp["setup-path"] = Utf8(trainingSetupPath);
		stamp["setup-revision"] = trainingSetupPath.empty() ? std::string() : FileRevision(trainingSetupPath);
		for (const fs::path& path : files)
			stamp[Utf8(fs::absolute(path).lexically_normal())] = FileRevision(path);

		return stamp;
	}

	// <Models>/.rocketforge/fingerprints/<sha256 of the identity>.json, or empty outside a Models tree.
	fs::path CachePath(const fs::path& checkpointPath, const fs::path& trainingSetupPath)
	{
		const fs::path absoluteCheckpoint = fs::absolute(checkpointPath);
		for (fs::path parent = absoluteCheckpoint; !parent.empty() && parent != parent.parent_path(); parent = parent.parent_path()) {
			if (parent.filename() != "Models")
				continue;

			Sha256 identity;
			identity.Update(absoluteCheckpoint.lexically_normal().string() + "\n" + trainingSetupPath.lexically_normal().string());
			return parent / ".rocketforge" / "fingerprints" / (Sha256::ToHex(identity.Finish()) + ".json");
		}

		return {};
	}

	bool IsSha256Hex(const std::string& digest)
	{
		return digest.size() == 64 && std::all_of(digest.begin(), digest.end(), [](unsigned char c) { return std::isxdigit(c) != 0; });
	}

	// The cached digest when the entry's dependencies match, else empty.
	std::string ReadCachedDigest(const fs::path& cachePath, const nlohmann::json& revisions)
	{
		std::ifstream file(cachePath, std::ios::binary);
		if (!file)
			return {};

		const bool allowExceptions = false;
		const nlohmann::json cached = nlohmann::json::parse(file, nullptr, allowExceptions);
		if (!cached.is_object())
			return {};

		const auto version = cached.find("version");
		const auto dependencies = cached.find("dependencies");
		const auto digest = cached.find("sha256");
		if (version == cached.end() || !version->is_number() || version->get<double>() != 1.0)
			return {};
		if (dependencies == cached.end() || *dependencies != revisions)
			return {};
		if (digest == cached.end() || !digest->is_string() || !IsSha256Hex(digest->get<std::string>()))
			return {};

		return digest->get<std::string>();
	}

	// Each file's name, then its contents. False on any read failure.
	bool HashFiles(const std::vector<fs::path>& files, Sha256& hash)
	{
		for (const fs::path& path : files) {
			hash.Update(path.filename().generic_string());

			std::error_code sizeError;
			const uintmax_t size = fs::file_size(path, sizeError);
			std::ifstream file(path, std::ios::binary);
			if (sizeError || !file || !RocketForgeTraining::HashStreamContents(file, size, hash))
				return false;
		}

		return true;
	}
}

bool RocketForgeTraining::HashStreamContents(std::istream& stream, uintmax_t expectedBytes, Sha256& hash)
{
	std::vector<char> buffer(1024 * 1024);
	uintmax_t hashedBytes = 0;
	while (stream) {
		stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
		hash.Update(buffer.data(), static_cast<size_t>(stream.gcount()));
		hashedBytes += static_cast<uintmax_t>(stream.gcount());
	}

	return !stream.bad() && hashedBytes == expectedBytes;
}

namespace
{
	// Best-effort atomic write: a failed cache write only costs a rehash next time.
	void WriteCache(const fs::path& cachePath, const nlohmann::json& revisions, const std::string& digest)
	{
		std::error_code filesystemError;
		fs::create_directories(cachePath.parent_path(), filesystemError);

		const nlohmann::json entry = { { "version", 1 }, { "dependencies", revisions }, { "sha256", digest } };
		const std::string bytes = entry.dump();

		// Several processes may write the same entry at once, so each writer stages its own file.
		fs::path stagingPath = cachePath;
		stagingPath += "." + std::to_string(std::random_device{}()) + ".tmp";
		{
			std::ofstream file(stagingPath, std::ios::binary | std::ios::trunc);
			file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
			if (!file.flush()) {
				file.close();
				fs::remove(stagingPath, filesystemError);
				return;
			}
		}

		fs::rename(stagingPath, cachePath, filesystemError);
		if (filesystemError)
			fs::remove(stagingPath, filesystemError);
	}
}

std::string RocketForgeTraining::FingerprintCheckpoint(const fs::path& checkpointPath, const fs::path& trainingSetupPath, bool allowCached)
{
	static std::mutex fingerprintMutex;
	const std::lock_guard<std::mutex> lock(fingerprintMutex);

	try {
		std::error_code filesystemError;
		if (checkpointPath.empty() || !fs::is_directory(checkpointPath, filesystemError) || filesystemError)
			return {};

		std::vector<fs::path> files;
		if (!CollectWeightFiles(checkpointPath, files) || !AddInferenceManifest(checkpointPath, files))
			return {};

		filesystemError.clear();
		if (!trainingSetupPath.empty() && fs::is_regular_file(trainingSetupPath, filesystemError) && !filesystemError)
			files.push_back(trainingSetupPath);
		if (files.empty())
			return {};

		std::sort(files.begin(), files.end());
		files.erase(std::unique(files.begin(), files.end()), files.end());

		// Stamping before and after hashing catches a concurrent writer, which the strict path must never fingerprint.
		const nlohmann::json revisions = RevisionStamp(checkpointPath, trainingSetupPath, files);
		const fs::path cachePath = CachePath(checkpointPath, trainingSetupPath);

		if (allowCached && !cachePath.empty()) {
			const std::string cached = ReadCachedDigest(cachePath, revisions);
			if (!cached.empty() && RevisionStamp(checkpointPath, trainingSetupPath, files) == revisions)
				return cached;
		}

		Sha256 hash;
		if (!HashFiles(files, hash) || RevisionStamp(checkpointPath, trainingSetupPath, files) != revisions)
			return {};

		const std::string digest = Sha256::ToHex(hash.Finish());
		if (!cachePath.empty())
			WriteCache(cachePath, revisions, digest);

		return digest;
	} catch (const std::exception&) {
		return {};
	}
}
