#include <gtest/gtest.h>

#include "TestSupport.h"

#include <GigaLearnCPP/SkillBenchmarkRegistry.h>

namespace fs = std::filesystem;

namespace {
	fs::path TempRegistryRoot(const std::string& name)
	{
		return RocketForgeTests::TestTempDir(name);
	}
}

TEST(SkillBenchmarkRegistry, UpsertThenListRoundTripsFields)
{
	fs::path root = TempRegistryRoot("upsert_list");
	fs::path registryPath = root / GGL::SkillBenchmarkRegistry::REGISTRY_FILE_NAME;

	GGL::SkillBenchmarkRegistry registry(registryPath, root, "example-agent");

	GGL::SkillBenchmarkSummary benchmark = {};
	benchmark.id = "fixed_anchor";
	benchmark.sourceStageName = "starter";
	benchmark.sourceVersionId = "v1.00";
	benchmark.createdAt = "20260101-010101";
	benchmark.timesteps = 1000;
	benchmark.path = (root / "skill_benchmarks" / "fixed_anchor").string();

	registry.UpsertBenchmark(benchmark);

	std::vector<GGL::SkillBenchmarkSummary> listed = registry.ListBenchmarks();
	ASSERT_EQ(listed.size(), size_t(1));
	ASSERT_EQ(listed[0].id, std::string("fixed_anchor"));
	ASSERT_EQ(listed[0].sourceStageName, std::string("starter"));
	ASSERT_EQ(listed[0].sourceVersionId, std::string("v1.00"));
	ASSERT_EQ(listed[0].createdAt, std::string("20260101-010101"));
	ASSERT_EQ(listed[0].timesteps, uint64_t(1000));

	fs::remove_all(root);
}

TEST(SkillBenchmarkRegistry, UpsertStoresPathRelativeToRootFolder)
{
	fs::path root = TempRegistryRoot("relative_path");
	fs::path registryPath = root / GGL::SkillBenchmarkRegistry::REGISTRY_FILE_NAME;

	GGL::SkillBenchmarkRegistry registry(registryPath, root, "example-agent");

	GGL::SkillBenchmarkSummary benchmark = {};
	benchmark.id = "anchor";
	benchmark.sourceStageName = "starter";
	benchmark.sourceVersionId = "v1.00";
	benchmark.createdAt = "20260101-010101";
	benchmark.timesteps = 500;
	// Absolute path under rootFolder -- MakeRelativePath should strip the root prefix on save.
	benchmark.path = (root / "skill_benchmarks" / "anchor").string();

	registry.UpsertBenchmark(benchmark);

	// Reload from disk via a fresh registry instance to exercise the actual on-disk shape,
	// not just the in-memory JSON the first instance already holds.
	GGL::SkillBenchmarkRegistry reloaded(registryPath, root, "example-agent");
	std::vector<GGL::SkillBenchmarkSummary> listed = reloaded.ListBenchmarks();

	ASSERT_EQ(listed.size(), size_t(1));
	ASSERT_EQ(listed[0].path, std::string("skill_benchmarks/anchor"));

	fs::remove_all(root);
}

TEST(SkillBenchmarkRegistry, UpsertOnExistingIdReplacesRatherThanDuplicates)
{
	fs::path root = TempRegistryRoot("upsert_replace");
	fs::path registryPath = root / GGL::SkillBenchmarkRegistry::REGISTRY_FILE_NAME;

	GGL::SkillBenchmarkRegistry registry(registryPath, root, "example-agent");

	GGL::SkillBenchmarkSummary original = {};
	original.id = "anchor";
	original.sourceStageName = "starter";
	original.sourceVersionId = "v1.00";
	original.timesteps = 100;
	original.path = (root / "skill_benchmarks" / "anchor").string();
	registry.UpsertBenchmark(original);

	GGL::SkillBenchmarkSummary updated = original;
	updated.timesteps = 200;
	updated.sourceVersionId = "v1.01";
	registry.UpsertBenchmark(updated);

	std::vector<GGL::SkillBenchmarkSummary> listed = registry.ListBenchmarks();
	ASSERT_EQ(listed.size(), size_t(1));
	ASSERT_EQ(listed[0].timesteps, uint64_t(200));
	ASSERT_EQ(listed[0].sourceVersionId, std::string("v1.01"));

	fs::remove_all(root);
}

TEST(SkillBenchmarkRegistry, AnchorBenchmarkIdRoundTrips)
{
	fs::path root = TempRegistryRoot("anchor_id");
	fs::path registryPath = root / GGL::SkillBenchmarkRegistry::REGISTRY_FILE_NAME;

	GGL::SkillBenchmarkRegistry registry(registryPath, root, "example-agent");
	ASSERT_EQ(registry.GetAnchorBenchmarkId(), std::string());

	registry.SetAnchorBenchmarkId("anchor_benchmark");

	GGL::SkillBenchmarkRegistry reloaded(registryPath, root, "example-agent");
	ASSERT_EQ(reloaded.GetAnchorBenchmarkId(), std::string("anchor_benchmark"));

	fs::remove_all(root);
}
