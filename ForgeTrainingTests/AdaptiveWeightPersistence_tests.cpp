#include <gtest/gtest.h>

#include <GigaLearnCPP/AdaptiveWeightPersistence.h>

#include <nlohmann/json.hpp>

namespace {
	// GGL::JsonSave/JsonLoad require a member Serialize; the entry uses the free SerializeFields overload.
	nlohmann::json SaveEntry(RLGC::AdaptivePersistEntry entry) {
		nlohmann::json j = nlohmann::json::object();
		GGL::JsonArchive ar = GGL::JsonArchive::MakeSaver(j);
		RLGC::SerializeFields(ar, entry);
		return j;
	}

	RLGC::AdaptivePersistEntry LoadEntry(const nlohmann::json& j) {
		RLGC::AdaptivePersistEntry entry;
		GGL::JsonArchive ar = GGL::JsonArchive::MakeLoader(j);
		RLGC::SerializeFields(ar, entry);
		return entry;
	}
}

TEST(AdaptiveWeightPersistence, RoundTripsPhaseAndTimer) {
	RLGC::AdaptivePersistEntry entry = { "Flip", 4, -12.5f, 0.75f };
	entry.phase = RLGC::AdaptivePhase::Mastered;
	entry.pendingExitSteps = 12345.0;

	const nlohmann::json j = SaveEntry(entry);
	ASSERT_EQ(j.value("phase", std::string()), std::string("Mastered"));

	const RLGC::AdaptivePersistEntry loaded = LoadEntry(j);
	ASSERT_EQ(loaded.name, std::string("Flip"));
	ASSERT_EQ(loaded.playerCount, 4);
	ASSERT_EQ(loaded.weight, -12.5f);
	ASSERT_EQ(loaded.rawRateEma, 0.75f);
	ASSERT_TRUE(loaded.phase.has_value());
	ASSERT_TRUE(*loaded.phase == RLGC::AdaptivePhase::Mastered);
	ASSERT_EQ(loaded.pendingExitSteps, 12345.0);
}

TEST(AdaptiveWeightPersistence, PrePhaseEntryLoadsWithoutPhase) {
	nlohmann::json j = nlohmann::json::parse(R"({"name":"Flip","player_count":2,"weight":30.0,"raw_rate_ema":0.4})");

	const RLGC::AdaptivePersistEntry loaded = LoadEntry(j);

	ASSERT_EQ(loaded.weight, 30.f);
	ASSERT_FALSE(loaded.phase.has_value());
	ASSERT_EQ(loaded.pendingExitSteps, 0.0);
}

TEST(AdaptiveWeightPersistence, PhaselessEntrySavesNoPhaseKey) {
	RLGC::AdaptivePersistEntry entry = { "Flip", 2, 3.f, 1.f };

	ASSERT_FALSE(SaveEntry(entry).contains("phase"));
}

TEST(AdaptiveWeightPersistence, UnknownPhaseIsReportedAndLeftEmpty) {
	nlohmann::json j = nlohmann::json::parse(R"({"name":"Flip","player_count":2,"weight":3.0,"raw_rate_ema":1.0,"phase":"Pinned"})");

	RLGC::AdaptivePersistEntry loaded;
	GGL::JsonArchive ar = GGL::JsonArchive::MakeLoader(j);
	RLGC::SerializeFields(ar, loaded);

	ASSERT_FALSE(loaded.phase.has_value());
	ASSERT_FALSE(ar.GetErrors().empty());
}
