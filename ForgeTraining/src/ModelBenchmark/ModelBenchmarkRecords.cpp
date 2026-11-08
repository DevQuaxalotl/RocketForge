#include "ModelBenchmarkInternal.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace RocketForgeTraining::ModelBenchmarkInternal
{
	namespace
	{
		int RecordTypeIndex(BenchmarkRecordType recordType) {
			switch (recordType) {
			case BenchmarkRecordType::Kickoff: return 1;
			case BenchmarkRecordType::PostKickoff: return 2;
			default: return 0;
			}
		}

		BenchmarkRecordType RecordTypeForGoalTiming(ModelBenchmarkGoalTiming goalTiming) {
			switch (goalTiming) {
			case ModelBenchmarkGoalTiming::Kickoff: return BenchmarkRecordType::Kickoff;
			case ModelBenchmarkGoalTiming::PostKickoff: return BenchmarkRecordType::PostKickoff;
			default: throw std::logic_error("ModelBenchmark: scored result has no goal timing.");
			}
		}
	}

	ModelBenchmarkRecord& GetRecord(
		BenchmarkRecords& records,
		int model,
		int opponent,
		int mode,
		BenchmarkRecordType recordType) {
		PairKey key{ std::min(model, opponent), std::max(model, opponent), mode };
		int modelSlot = (model == key.a) ? 0 : 1;
		return records[key][RecordTypeIndex(recordType)][modelSlot];
	}

	void AddResultToRecords(
		BenchmarkRecords& records,
		int winner,
		int loser,
		int mode,
		ModelBenchmarkGoalTiming goalTiming) {
		for (BenchmarkRecordType recordType : {
			BenchmarkRecordType::Overall,
			RecordTypeForGoalTiming(goalTiming)
		}) {
			GetRecord(records, winner, loser, mode, recordType).wins++;
			GetRecord(records, loser, winner, mode, recordType).losses++;
		}
	}

	void AddNoResultToRecords(
		BenchmarkRecords& records,
		int modelA,
		int modelB,
		int mode) {
		GetRecord(records, modelA, modelB, mode, BenchmarkRecordType::Overall).noResults++;
		GetRecord(records, modelB, modelA, mode, BenchmarkRecordType::Overall).noResults++;
	}

	double WinPct(const ModelBenchmarkRecord& record) {
		int scored = record.wins + record.losses;
		return scored > 0 ? (double)record.wins / (double)scored : 0.5;
	}

	std::string FormatRecordSummary(
		const BenchmarkRecords& records,
		const std::vector<std::string>& names,
		const std::vector<int>& rankOrder) {
		if (names.empty())
			return {};

		std::vector<ModelBenchmarkRecord> totals(names.size());
		for (const auto& recordPair : records) {
			const PairKey& key = recordPair.first;
			if (key.a < 0 || key.b < 0 || key.a >= (int)names.size() || key.b >= (int)names.size())
				continue;
			const auto& overall = recordPair.second[RecordTypeIndex(BenchmarkRecordType::Overall)];
			totals[key.a].wins += overall[0].wins;
			totals[key.a].losses += overall[0].losses;
			totals[key.a].noResults += overall[0].noResults;
			totals[key.b].wins += overall[1].wins;
			totals[key.b].losses += overall[1].losses;
			totals[key.b].noResults += overall[1].noResults;
		}

		std::ostringstream out;
		out << "Records:";
		if (rankOrder.empty()) {
			for (int i = 0; i < (int)names.size(); i++)
				out << "\n  " << names[i] << " " << totals[i].wins << "-" << totals[i].losses;
			return out.str();
		}

		for (int modelIdx : rankOrder) {
			if (modelIdx < 0 || modelIdx >= (int)names.size())
				continue;
			out << "\n  " << names[modelIdx] << " " << totals[modelIdx].wins << "-" << totals[modelIdx].losses;
		}
		return out.str();
	}

	std::vector<BenchmarkOutcome> FilterOutcomes(
		const std::vector<BenchmarkOutcome>& outcomes,
		ModelBenchmarkGoalTiming goalTiming) {
		std::vector<BenchmarkOutcome> result;
		for (BenchmarkOutcome outcome : outcomes) {
			if (outcome.goalTiming != goalTiming)
				continue;
			result.push_back(outcome);
		}
		return result;
	}

	std::vector<BenchmarkOutcome> FilterOutcomesByMode(
		const std::vector<BenchmarkOutcome>& outcomes,
		int mode) {
		std::vector<BenchmarkOutcome> result;
		for (BenchmarkOutcome outcome : outcomes) {
			if (outcome.mode != mode)
				continue;
			result.push_back(outcome);
		}
		return result;
	}
}
