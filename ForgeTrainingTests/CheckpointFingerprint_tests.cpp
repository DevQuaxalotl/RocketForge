#include "TestSupport.h"

#include <ModelBenchmark/CheckpointFingerprintInternal.h>

#include <sstream>

#ifdef ROCKETFORGE_TESTS_HAVE_QT
#include <ModelBenchmark/CheckpointFingerprint.h>
#include <Utils/FileRevision.h>

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSaveFile>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{
// The Qt implementation from ForgeGUI/src/ModelBenchmark/ModelBenchmarkLibrary.cpp at commit 08bfd6f,
// kept verbatim apart from its name and FileRevision's namespace as the oracle that proves the port is byte-identical.
QString LegacyQtFingerprint(
    const std::filesystem::path& checkpointPath,
    const std::filesystem::path& trainingSetupPath, bool allowCached)
{
    // Only background display may reuse a digest.
    // Actions use the default strict path, including the before/after revision check for a concurrent writer.
    static std::mutex fingerprintMutex;
    const std::lock_guard<std::mutex> lock(fingerprintMutex);

    try {
    std::error_code filesystemError;
    if (checkpointPath.empty()
        || !std::filesystem::is_directory(checkpointPath, filesystemError)
        || filesystemError)
        return {};
    std::vector<std::filesystem::path> files;
    for (const char* name : { "POLICY.lt", "SHARED_HEAD.lt" }) {
        const std::filesystem::path candidate = checkpointPath / name;
        filesystemError.clear();
        if (std::filesystem::is_regular_file(candidate, filesystemError) && !filesystemError)
            files.push_back(candidate);
    }
    if (files.empty()) {
        std::filesystem::directory_iterator iterator(checkpointPath, filesystemError);
        const std::filesystem::directory_iterator end;
        for (; iterator != end && !filesystemError; iterator.increment(filesystemError)) {
            const std::filesystem::directory_entry& entry = *iterator;
            std::error_code entryError;
            if (!entry.is_regular_file(entryError) || entryError || entry.path().extension() != ".lt")
                continue;
            const QString name = QString::fromStdString(entry.path().filename().string()).toUpper();
            if (!name.contains("OPTIM") && !name.contains("CRITIC"))
                files.push_back(entry.path());
        }
        if (filesystemError)
            return {};
    }

    // Imported inference policies can use TorchScript weights instead of POLICY.lt.
    // Hash the actual declared artifact and manifest; provenance text alone is not a weight identity.
    const std::filesystem::path manifestPath = checkpointPath / "INFERENCE_POLICY.json";
    if (std::filesystem::exists(manifestPath)) {
        QFile manifest(QString::fromStdWString(manifestPath.wstring()));
        if (!manifest.open(QIODevice::ReadOnly)) return {};

        QJsonParseError parseError;
        const QJsonDocument document = QJsonDocument::fromJson(manifest.readAll(), &parseError);
        if (parseError.error != QJsonParseError::NoError || !document.isObject()) return {};

        const std::filesystem::path artifactName(document.object()["artifact"].toString().toStdWString());
        if (artifactName.empty() || artifactName.is_absolute()) return {};

        const std::filesystem::path artifact = std::filesystem::weakly_canonical(checkpointPath / artifactName);
        const std::filesystem::path relative = artifact.lexically_relative(std::filesystem::canonical(checkpointPath));
        if (relative.empty() || *relative.begin() == ".." || !std::filesystem::is_regular_file(artifact)) return {};

        files.push_back(manifestPath);
        files.push_back(artifact);
    }

    filesystemError.clear();
    if (!trainingSetupPath.empty()
        && std::filesystem::is_regular_file(trainingSetupPath, filesystemError)
        && !filesystemError)
        files.push_back(trainingSetupPath);
    if (files.empty())
        return {};
    std::sort(files.begin(), files.end());
    files.erase(std::unique(files.begin(), files.end()), files.end());

    QJsonObject revisions;
    auto stamp = [&]() {
        QJsonObject result;
        result["checkpoint-directory"] = QString::fromStdString(RocketForge::FileRevision(checkpointPath));
        result["setup-path"] = QString::fromStdWString(trainingSetupPath.wstring());
        result["setup-revision"] = trainingSetupPath.empty() ? QString() : QString::fromStdString(RocketForge::FileRevision(trainingSetupPath));
        for (const std::filesystem::path& path : files)
            result[QString::fromStdWString(std::filesystem::absolute(path).lexically_normal().wstring())]
                = QString::fromStdString(RocketForge::FileRevision(path));

        return result;
    };
    revisions = stamp();

    std::filesystem::path cachePath;
    for (auto parent = std::filesystem::absolute(checkpointPath); !parent.empty() && parent != parent.parent_path(); parent = parent.parent_path()) {
        if (parent.filename() != "Models") continue;

        const QByteArray identity = QByteArray::fromStdString(std::filesystem::absolute(checkpointPath).lexically_normal().string()
            + "\n" + trainingSetupPath.lexically_normal().string());
        const QByteArray key = QCryptographicHash::hash(identity, QCryptographicHash::Sha256).toHex();
        cachePath = parent / ".rocketforge" / "fingerprints" / (key.toStdString() + ".json");
        break;
    }

    if (allowCached && !cachePath.empty()) {
        QFile file(QString::fromStdWString(cachePath.wstring()));
        if (file.open(QIODevice::ReadOnly)) {
            const QJsonObject cached = QJsonDocument::fromJson(file.readAll()).object();
            const QString digest = cached["sha256"].toString();
            if (cached["version"].toInt() == 1 && cached["dependencies"].toObject() == revisions
                && digest.size() == 64 && QByteArray::fromHex(digest.toLatin1()).size() == 32
                && stamp() == revisions) return digest;
        }
    }

    QCryptographicHash hash(QCryptographicHash::Sha256);
    for (const std::filesystem::path& path : files) {
        hash.addData(QByteArray::fromStdString(path.filename().generic_string()));
        QFile file(QString::fromStdString(path.string()));
        if (!file.open(QIODevice::ReadOnly))
            return {};
        if (path == trainingSetupPath) {
            hash.addData(file.readAll());
        } else {
            while (!file.atEnd()) {
                const QByteArray bytes = file.read(1024 * 1024);
                if (bytes.isEmpty() && !file.atEnd()) return {};
                hash.addData(bytes);
            }
        }

        if (file.error() != QFileDevice::NoError) return {};
    }

    if (stamp() != revisions) return {};

    const QString digest = QString::fromLatin1(hash.result().toHex());
    if (!cachePath.empty()) {
        QDir().mkpath(QString::fromStdWString(cachePath.parent_path().wstring()));
        QSaveFile file(QString::fromStdWString(cachePath.wstring()));
        const QJsonObject cacheEntry{{"version", 1}, {"dependencies", revisions}, {"sha256", digest}};
        const QByteArray bytes = QJsonDocument(cacheEntry).toJson(QJsonDocument::Compact);
        if (file.open(QIODevice::WriteOnly) && file.write(bytes) == bytes.size()) file.commit();
    }

    return digest;
    } catch (const std::exception&) { return {}; }
}

	void WriteTestFile(const fs::path& path, const std::string& contents)
	{
		fs::create_directories(path.parent_path());
		std::ofstream file(path, std::ios::binary);
		file << contents;
	}

	/** The one cache entry under <models>/.rocketforge/fingerprints, or empty when there is not exactly one. */
	fs::path OnlyCacheFile(const fs::path& modelsRoot)
	{
		const fs::path folder = modelsRoot / ".rocketforge" / "fingerprints";
		std::vector<fs::path> entries;
		if (fs::is_directory(folder))
			for (const fs::directory_entry& entry : fs::directory_iterator(folder))
				entries.push_back(entry.path());

		return entries.size() == 1 ? entries.front() : fs::path();
	}

	// Edits the raw text so the reader sees exactly the bytes the writer produced, apart from the digest.
	bool ReplaceCachedDigest(const fs::path& cacheFile, const std::string& digest)
	{
		std::ifstream input(cacheFile, std::ios::binary);
		std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
		input.close();

		const std::string field = "\"sha256\":\"";
		const size_t start = text.find(field);
		if (start == std::string::npos || start + field.size() + digest.size() > text.size())
			return false;

		text.replace(start + field.size(), digest.size(), digest);
		std::ofstream(cacheFile, std::ios::binary | std::ios::trunc) << text;
		return true;
	}
}

TEST(CheckpointFingerprint, MatchesQtImplementation)
{
	const fs::path root = RocketForgeTests::TestTempDir("fingerprint");
	const fs::path models = root / "Models" / "Build" / "segments" / "v1.0_stage" / "checkpoints";
	const fs::path setup = root / "Models" / "Build" / "segments" / "v1.0_stage" / "training_setup.json";
	WriteTestFile(setup, R"({"stage":"example","rewards":[1,2,3]})");

	const fs::path trained = models / "1000";
	WriteTestFile(trained / "POLICY.lt", std::string(3 * 1024 * 1024 + 17, 'p'));
	WriteTestFile(trained / "SHARED_HEAD.lt", "shared head weights");
	WriteTestFile(trained / "POLICY_OPTIM.lt", "optimizer state");
	WriteTestFile(trained / "CRITIC.lt", "critic weights");

	const fs::path fallback = models / "2000";
	WriteTestFile(fallback / "ACTOR.lt", "actor weights");
	WriteTestFile(fallback / "extra_head.lt", "extra head weights");
	WriteTestFile(fallback / "actor_optim.lt", "optimizer state");
	WriteTestFile(fallback / "Critic_Head.lt", "critic weights");
	WriteTestFile(fallback / "notes.txt", "not weights");

	const fs::path imported = root / "Imported" / "Nexto";
	WriteTestFile(imported / "INFERENCE_POLICY.json", R"({"artifact":"weights/nexto.pt","format":"torchscript"})");
	WriteTestFile(imported / "weights" / "nexto.pt", "torchscript weights");

	const fs::path escaping = root / "Imported" / "Escaping";
	WriteTestFile(escaping / "INFERENCE_POLICY.json", R"({"artifact":"../outside.pt"})");
	WriteTestFile(root / "Imported" / "outside.pt", "outside weights");

	const fs::path malformed = root / "Imported" / "Malformed";
	WriteTestFile(malformed / "INFERENCE_POLICY.json", "{ not json");

	const fs::path empty = models / "3000";
	fs::create_directories(empty);

	struct Case {
		const char* label;
		fs::path checkpoint;
		fs::path setup;
		bool expectDigest;
	};
	const Case cases[] = {
		{ "trained", trained, {}, true },
		{ "trained with setup", trained, setup, true },
		{ "lt fallback", fallback, {}, true },
		{ "lt fallback with setup", fallback, setup, true },
		{ "imported", imported, {}, true },
		{ "imported with setup", imported, setup, true },
		{ "artifact outside the folder", escaping, {}, false },
		{ "malformed manifest", malformed, {}, false },
		{ "no weights", empty, {}, false },
		{ "no weights with setup", empty, setup, true },
		{ "missing folder", models / "missing", setup, false },
		{ "missing setup", trained, root / "missing_setup.json", true },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);
		for (const bool allowCached : { false, true }) {
			SCOPED_TRACE(allowCached ? "cached" : "strict");
			const std::string expected = LegacyQtFingerprint(testCase.checkpoint, testCase.setup, allowCached).toStdString();
			EXPECT_EQ(RocketForgeTraining::FingerprintCheckpoint(testCase.checkpoint, testCase.setup, allowCached), expected);
			EXPECT_EQ(expected.empty(), !testCase.expectDigest);
		}
	}
}

TEST(CheckpointFingerprint, ReusesCacheWrittenByEitherImplementation)
{
	const fs::path modelsRoot = RocketForgeTests::TestTempDir("fingerprint-cache") / "Models";
	const fs::path checkpoint = modelsRoot / "Build" / "segments" / "v1.0_stage" / "checkpoints" / "1000";
	const fs::path setup = modelsRoot / "Build" / "segments" / "v1.0_stage" / "training_setup.json";
	WriteTestFile(checkpoint / "POLICY.lt", "policy weights");
	WriteTestFile(checkpoint / "SHARED_HEAD.lt", "shared head weights");
	WriteTestFile(setup, R"({"stage":"example"})");

	const std::string sentinel(64, 'a');
	const bool allowCached = true;
	const bool strict = false;

	// A cache entry is trusted whenever its dependency stamps match,
	// so a planted digest is returned only if the other implementation's entry was found and accepted.
	const std::string qtDigest = LegacyQtFingerprint(checkpoint, setup, allowCached).toStdString();
	ASSERT_EQ(qtDigest.size(), 64u);
	const fs::path qtCache = OnlyCacheFile(modelsRoot);
	ASSERT_FALSE(qtCache.empty());
	ASSERT_TRUE(ReplaceCachedDigest(qtCache, sentinel));

	EXPECT_EQ(RocketForgeTraining::FingerprintCheckpoint(checkpoint, setup, allowCached), sentinel);
	EXPECT_EQ(RocketForgeTraining::FingerprintCheckpoint(checkpoint, setup, strict), qtDigest);

	fs::remove(qtCache);
	EXPECT_EQ(RocketForgeTraining::FingerprintCheckpoint(checkpoint, setup, allowCached), qtDigest);
	const fs::path portedCache = OnlyCacheFile(modelsRoot);
	ASSERT_EQ(portedCache, qtCache);
	ASSERT_TRUE(ReplaceCachedDigest(portedCache, sentinel));

	EXPECT_EQ(LegacyQtFingerprint(checkpoint, setup, allowCached).toStdString(), sentinel);
}

TEST(CheckpointFingerprint, IgnoresCacheWhenWeightsChange)
{
	const fs::path modelsRoot = RocketForgeTests::TestTempDir("fingerprint-stale") / "Models";
	const fs::path checkpoint = modelsRoot / "Build" / "checkpoints" / "1000";
	WriteTestFile(checkpoint / "POLICY.lt", "policy weights");

	const bool allowCached = true;
	const std::string before = RocketForgeTraining::FingerprintCheckpoint(checkpoint, {}, allowCached);
	WriteTestFile(checkpoint / "POLICY.lt", "retrained policy weights");
	const std::string after = RocketForgeTraining::FingerprintCheckpoint(checkpoint, {}, allowCached);

	EXPECT_EQ(before.size(), 64u);
	EXPECT_NE(after, before);
	EXPECT_EQ(after, LegacyQtFingerprint(checkpoint, {}, false).toStdString());
}

#else
TEST(CheckpointFingerprint, RequiresQt) { GTEST_SKIP() << "Qt is unavailable"; }
#endif

TEST(CheckpointFingerprint, RejectsReadsThatDisagreeWithTheFileSize)
{
	struct Case {
		const char* label;
		uintmax_t expectedBytes;
		bool expectSuccess;
	};
	const Case cases[] = {
		{ "complete", 14, true },
		{ "truncated", 20, false },
		{ "longer than expected", 10, false },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);
		std::istringstream stream("policy weights");
		RocketForge::Sha256 hash;
		EXPECT_EQ(RocketForgeTraining::HashStreamContents(stream, testCase.expectedBytes, hash), testCase.expectSuccess);
	}
}
