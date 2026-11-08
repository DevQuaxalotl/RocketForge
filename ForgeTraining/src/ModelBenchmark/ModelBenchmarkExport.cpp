#include "ModelBenchmarkExport.h"

#include "CheckpointFingerprint.h"

#include <Storage/BuildStorage.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <system_error>

using namespace std;
using namespace RocketForgeTraining;
namespace fs = std::filesystem;

namespace
{
	using Cells = vector<string>;

	const bool HEADER_ROW = true;

	// Matches QString::number's output so workbooks stay byte-comparable with the GUI's.
	string Fixed(double value, int decimals)
	{
		char buffer[512];
		snprintf(buffer, sizeof(buffer), "%.*f", decimals, value);
		return buffer;
	}

	string Precise(double value)
	{
		char buffer[64];
		snprintf(buffer, sizeof(buffer), "%.17g", value);
		return buffer;
	}

	string Signed(double value, int decimals = 0)
	{
		return (value >= 0.0 ? "+" : "") + Fixed(value, decimals);
	}

	string OptionalElo(bool available, double value)
	{
		return available ? Signed(value) : "N/A";
	}

	string YesNo(bool value)
	{
		return value ? "Yes" : "No";
	}

	string WinPercent(const ModelBenchmarkRecord& record)
	{
		const int scored = record.wins + record.losses;
		const double percent = scored > 0 ? (100.0 * record.wins / scored) : 0.0;
		return Fixed(percent, 1) + "%";
	}

	string GoalTimingLabel(ModelBenchmarkGoalTiming goalTiming)
	{
		switch (goalTiming) {
		case ModelBenchmarkGoalTiming::Kickoff: return "Kickoff";
		case ModelBenchmarkGoalTiming::PostKickoff: return "Post-Kickoff";
		default: return "No Goal";
		}
	}

	int ModeSortKey(const string& modeName)
	{
		if (modeName.empty() || modeName[0] < '0' || modeName[0] > '9')
			return numeric_limits<int>::max();

		return modeName[0] - '0';
	}

	string ReplaceAll(string text, const string& from, const string& to)
	{
		for (size_t position = text.find(from); position != string::npos; position = text.find(from, position + to.size()))
			text.replace(position, from.size(), to);

		return text;
	}

	string PathText(const fs::path& path)
	{
		const u8string text = path.u8string();
		return string(text.begin(), text.end());
	}

	class WorkbookWriter
	{
	public:
		WorkbookWriter(SpreadsheetXmlWriter& writer, const ModelBenchmarkExportInput& input)
			: _writer(writer), _input(input), _result(*input.result)
		{
			for (const ModelBenchmarkRankingRow& ranking : _result.rankings)
				_rankByModel[ranking.modelIndex] = ranking.rank;
		}

		void Write()
		{
			WriteSummary();
			_writer.Lines("Console", _input.consoleLines);

			if (_input.extraWorksheets)
				_input.extraWorksheets(_writer);

			WriteModels();
			WriteOverallRanking();
			WriteGoalTimingBreakdown();
			WritePairwiseSummary();
			WriteGameResults();

			static const vector<ModelBenchmarkStatRow> emptyStats;
			WriteStats("Stats - All Modes", _result.optionalStats);
			for (int mode : { 1, 2, 3 }) {
				const auto modeStats = _result.optionalStatsByMode.find(mode);
				const string modeName = to_string(mode) + "v" + to_string(mode);
				WriteStats("Stats - " + modeName, modeStats != _result.optionalStatsByMode.end() ? modeStats->second : emptyStats);
			}

			WriteRawStatistics();
		}

	private:
		int Rank(int modelIndex) const
		{
			const auto rank = _rankByModel.find(modelIndex);
			return rank != _rankByModel.end() ? rank->second : numeric_limits<int>::max();
		}

		string ModelName(int modelIndex) const
		{
			if (modelIndex < 0 || modelIndex >= (int)_result.modelNames.size())
				return "--";

			return _result.modelNames[modelIndex];
		}

		void WriteSummary()
		{
			const ModelBenchmarkResult& result = _result;
			const size_t eloRatedGames = (size_t)count_if(result.games.begin(), result.games.end(), [](const auto& game) {
				return game.includedInRating;
			});

			string modelNames;
			for (size_t i = 0; i < result.modelNames.size(); ++i)
				modelNames += (i ? " | " : "") + result.modelNames[i];

			_writer.OpenWorksheet("Summary");
			_writer.Row({ "Field", "Value" }, HEADER_ROW);
			_writer.Row({ "Library Format Version", "1" });
			_writer.Row({ "Run ID", _input.runId });
			_writer.Row({ "RocketSim Versions", ModelBenchmarkRocketSimVersionsText(result.rocketSimVersions) });
			if (!_input.environmentComparisonNote.empty())
				_writer.Row({ "Environment Comparison", _input.environmentComparisonNote });

			_writer.Row({ "Engine Family", result.engineFamily });
			_writer.Row({ "Engine Revision", result.engineRevision });
			_writer.Row({ "Engine Car Order", result.engineCarOrder });
			if (!result.integrationRevision.empty())
				_writer.Row({ "Integration Source Revision", result.integrationRevision });

			_writer.Row({ "Prediction Configuration", result.predictionConfigurationJson });
			_writer.Row({ "Evaluation Identity", result.evaluationIdentity });
			_writer.Row({ "Raw Statistics Complete", YesNo(result.statTotalsComplete) });
			_writer.Row({ "Run Seed", to_string(result.seed) });
			_writer.Row({ "Maximum Games", to_string(result.maxGames) });
			_writer.Row({ "Scored Games", to_string(result.scoredGames) });
			_writer.Row({ "No-Result Games", to_string(result.noResultGames) });
			_writer.Row({ "Elo-Rated Games", to_string(eloRatedGames) });
			_writer.Row({ "Total Elapsed Seconds", Fixed(result.elapsedSeconds, 3) });
			_writer.Row({ "Simulation Seconds", Fixed(result.simulationSeconds, 3) });
			_writer.Row({ "Simulation Games / Second", Fixed(result.gamesPerSecond, 2) });
			_writer.Row({ "Benchmark Strategy", result.benchmarkStrategy });
			_writer.Row({ "Arena Creates", to_string(result.arenaCreates) });
			_writer.Row({ "Arena Rebuilds", to_string(result.arenaRebuilds) });
			_writer.Row({ "Arena Reuses", to_string(result.arenaReuses) });
			_writer.Row({ "Timing Summary", ReplaceAll(result.timingSummary, "\n", " | ") });
			_writer.Row({ "Stop Reason", result.stopReason });
			_writer.Row({ "Kickoff Goal Window Seconds", Fixed(result.kickoffGoalWindowSeconds, 3) });
			_writer.Row({ "Stochastic Kickoff", result.kickoffInferenceSettingsMixed ? "Mixed" : YesNo(result.stochasticKickoff) });
			_writer.Row({ "Kickoff Temperature", result.kickoffInferenceSettingsMixed ? "N/A" : Fixed(result.kickoffTemperature, 3) });
			_writer.Row({ "Kickoff Goal Count", to_string(result.kickoffGoalCount) });
			_writer.Row({ "Post-Kickoff Goal Count", to_string(result.postKickoffGoalCount) });

			// Multiply in float: promoting to double first can change the last rounded digit.
			const float confidencePercent = result.eloRangeConfidence * 100.0f;
			_writer.Row({ "Elo Range Confidence", Fixed(confidencePercent, 2) + "%" });
			_writer.Row({ "Overall Elo Available", YesNo(result.overallEloAvailable) });
			_writer.Row({ "Kickoff Elo Available", YesNo(result.kickoffEloAvailable) });
			_writer.Row({ "Post-Kickoff Elo Available", YesNo(result.postKickoffEloAvailable) });
			_writer.Row({
				"Pairing Note",
				"Overall Elo connects models through every eligible matchup, including indirect comparisons through shared opponents. "
				"Each model pair has equal total rating weight, divided across the selected modes both models support. "
				"Timing Elo splits those same outcomes by goal time; "
				"all three Elo columns use the worst overall model's overall rating as zero, so timing Elo may be negative. "
				"N/A means that timing subset does not connect every selected model. Records also include no-results from completed groups. "
				"Optional statistics cover all physically completed simulation steps, including an unmatched half if a run is cancelled. "
				"Event statistics are normalized to one game per five minutes of simulated game time. "
				"Statistics are exported for all modes combined and separately for 1v1, 2v2, and 3v3."
			});
			_writer.Row({ "Models", modelNames });
			_writer.CloseWorksheet();
		}

		void WriteModels()
		{
			_writer.OpenWorksheet("Models");
			_writer.Row({
				"Model Index", "Display Name", "Primary", "Agent", "Build", "Stage", "Branch", "Source", "Version ID",
				"Latest Timesteps", "Last Trained", "Checkpoint Path", "Checkpoint Fingerprint"
			}, HEADER_ROW);

			for (int modelIndex = 0; modelIndex < (int)_result.modelNames.size(); ++modelIndex) {
				const ModelBenchmarkExportModel* model = modelIndex < (int)_input.models.size() ? &_input.models[modelIndex] : nullptr;
				const string versionId = modelIndex < (int)_result.modelVersionIds.size() ? _result.modelVersionIds[modelIndex] : string();
				const fs::path checkpointPath = modelIndex < (int)_result.modelCheckpointPaths.size()
					? _result.modelCheckpointPaths[modelIndex]
					: fs::path();

				_writer.Row({
					to_string(modelIndex),
					_result.modelNames[modelIndex],
					YesNo(!model || model->primary),
					model ? model->agentName : string(),
					model ? model->buildName : string(),
					model ? model->stageName : string(),
					model ? model->branchName : string(),
					model ? model->sourceLabel : string(),
					versionId,
					model ? to_string(model->latestTimesteps) : string(),
					model ? model->lastTrainedAt : string(),
					PathText(checkpointPath),
					model ? model->fingerprint : FingerprintCheckpoint(checkpointPath)
				});
			}

			_writer.CloseWorksheet();
		}

		void WriteOverallRanking()
		{
			_writer.OpenWorksheet("Overall Ranking");
			_writer.Row({
				"Rank", "Model", "Benchmark Elo", "Elo Range Low", "Elo Range High",
				"1v1 Elo", "2v2 Elo", "3v3 Elo", "Better Than",
				"Record Wins", "Record Losses", "Record Win %", "Record No Results"
			}, HEADER_ROW);

			const bool overallAvailable = _result.overallEloAvailable;
			for (const ModelBenchmarkRankingRow& row : _result.rankings) {
				_writer.Row({
					overallAvailable ? to_string(row.rank) : "N/A",
					row.modelName,
					OptionalElo(overallAvailable, row.relativeElo),
					OptionalElo(overallAvailable, row.rangeLow),
					OptionalElo(overallAvailable, row.rangeHigh),
					OptionalElo(row.elo1v1Available, row.elo1v1),
					OptionalElo(row.elo2v2Available, row.elo2v2),
					OptionalElo(row.elo3v3Available, row.elo3v3),
					row.betterThanSummary,
					to_string(row.totalRecord.wins),
					to_string(row.totalRecord.losses),
					WinPercent(row.totalRecord),
					to_string(row.totalRecord.noResults)
				});
			}

			_writer.CloseWorksheet();
		}

		void WriteGoalTimingBreakdown()
		{
			vector<ModelBenchmarkBreakdownRow> sortedBreakdown = _result.breakdown;
			sort(sortedBreakdown.begin(), sortedBreakdown.end(), [&](const auto& a, const auto& b) {
				if (Rank(a.modelIndex) != Rank(b.modelIndex))
					return Rank(a.modelIndex) < Rank(b.modelIndex);

				return a.modelName < b.modelName;
			});

			_writer.OpenWorksheet("Goal-Timing Breakdown");
			_writer.Row({
				"Model", "Overall Elo",
				"Kickoff Elo", "Kickoff Elo Range Low", "Kickoff Elo Range High", "Kickoff Games", "Kickoff Elo Available",
				"Post-Kickoff Elo", "Post-Kickoff Elo Range Low", "Post-Kickoff Elo Range High",
				"Post-Kickoff Games", "Post-Kickoff Elo Available",
				"Kickoff Delta"
			}, HEADER_ROW);

			for (const ModelBenchmarkBreakdownRow& row : sortedBreakdown) {
				_writer.Row({
					row.modelName,
					OptionalElo(_result.overallEloAvailable, row.combinedElo),
					OptionalElo(row.kickoffEloAvailable, row.kickoffElo),
					OptionalElo(row.kickoffEloAvailable, row.kickoffRangeLow),
					OptionalElo(row.kickoffEloAvailable, row.kickoffRangeHigh),
					to_string(row.kickoffGames),
					YesNo(row.kickoffEloAvailable),
					OptionalElo(row.postKickoffEloAvailable, row.postKickoffElo),
					OptionalElo(row.postKickoffEloAvailable, row.postKickoffRangeLow),
					OptionalElo(row.postKickoffEloAvailable, row.postKickoffRangeHigh),
					to_string(row.postKickoffGames),
					YesNo(row.postKickoffEloAvailable),
					OptionalElo(row.kickoffEloAvailable && row.postKickoffEloAvailable, row.kickoffDelta)
				});
			}

			_writer.CloseWorksheet();
		}

		void WritePairwiseSummary()
		{
			vector<ModelBenchmarkPairwiseRow> sortedPairwise = _result.pairwise;
			sort(sortedPairwise.begin(), sortedPairwise.end(), [&](const auto& a, const auto& b) {
				if (Rank(a.modelIndex) != Rank(b.modelIndex))
					return Rank(a.modelIndex) < Rank(b.modelIndex);

				if (Rank(a.opponentIndex) != Rank(b.opponentIndex))
					return Rank(a.opponentIndex) < Rank(b.opponentIndex);

				if (ModeSortKey(a.modeName) != ModeSortKey(b.modeName))
					return ModeSortKey(a.modeName) < ModeSortKey(b.modeName);

				if (a.modelName != b.modelName)
					return a.modelName < b.modelName;

				return a.opponentName < b.opponentName;
			});

			_writer.OpenWorksheet("Pairwise Summary");
			_writer.Row({
				"Model", "Opponent", "Mode", "Mode Elo",
				"Combined Wins", "Combined Losses", "Combined Win %", "Combined No Results",
				"Kickoff Wins", "Kickoff Losses", "Kickoff Win %",
				"Post-Kickoff Wins", "Post-Kickoff Losses", "Post-Kickoff Win %",
				"Kickoff vs Post-Kickoff Delta", "Timing Delta Available"
			}, HEADER_ROW);

			for (const ModelBenchmarkPairwiseRow& row : sortedPairwise) {
				_writer.Row({
					row.modelName,
					row.opponentName,
					row.modeName,
					Signed(row.modeElo),
					to_string(row.combined.wins),
					to_string(row.combined.losses),
					WinPercent(row.combined),
					to_string(row.combined.noResults),
					to_string(row.kickoff.wins),
					to_string(row.kickoff.losses),
					WinPercent(row.kickoff),
					to_string(row.postKickoff.wins),
					to_string(row.postKickoff.losses),
					WinPercent(row.postKickoff),
					row.timingDeltaAvailable ? Fixed(row.kickoffVsPostKickoffDeltaPercent, 1) + "%" : "N/A",
					YesNo(row.timingDeltaAvailable)
				});
			}

			_writer.CloseWorksheet();
		}

		void WriteGameResults()
		{
			const Cells headers = {
				"Game ID", "Snapshot ID",
				"Model A Index", "Model A", "Model B Index", "Model B",
				"Blue Index", "Blue Model", "Orange Index", "Orange Model",
				"Mode", "Goal Time Seconds", "Goal Timing", "Goal Scored", "Winner Index", "Winner",
				"Included In Rating", "Mirror Completed"
			};

			const size_t maxRows = SpreadsheetXmlWriter::MAX_DATA_ROWS;
			const vector<ModelBenchmarkGameMetadata>& games = _result.games;
			const size_t sheetCount = std::max<size_t>(1, (games.size() + maxRows - 1) / maxRows);

			for (size_t sheetIndex = 0; sheetIndex < sheetCount; ++sheetIndex) {
				_writer.OpenWorksheet(sheetCount == 1 ? "Game Results" : "Game Results " + to_string(sheetIndex + 1));
				_writer.Row(headers, HEADER_ROW);

				const size_t begin = sheetIndex * maxRows;
				const size_t end = std::min(begin + maxRows, games.size());
				for (size_t gameIndex = begin; gameIndex < end; ++gameIndex) {
					const ModelBenchmarkGameMetadata& game = games[gameIndex];
					_writer.Row({
						to_string(game.gameId),
						to_string(game.snapshotId),
						to_string(game.modelA),
						ModelName(game.modelA),
						to_string(game.modelB),
						ModelName(game.modelB),
						to_string(game.blueModel),
						ModelName(game.blueModel),
						to_string(game.orangeModel),
						ModelName(game.orangeModel),
						to_string(game.mode) + "v" + to_string(game.mode),
						Fixed(game.goalTimeSeconds, 3),
						GoalTimingLabel(game.goalTiming),
						YesNo(game.goalScored),
						game.winner >= 0 ? to_string(game.winner) : "--",
						ModelName(game.winner),
						YesNo(game.includedInRating),
						YesNo(game.mirrorCompleted)
					});
				}

				_writer.CloseWorksheet();
			}
		}

		void WriteStats(const string& sheetName, const vector<ModelBenchmarkStatRow>& stats)
		{
			vector<ModelBenchmarkStatRow> sortedStats = stats;
			sort(sortedStats.begin(), sortedStats.end(), [&](const auto& a, const auto& b) {
				if (Rank(a.modelIndex) != Rank(b.modelIndex))
					return Rank(a.modelIndex) < Rank(b.modelIndex);

				return a.modelName < b.modelName;
			});

			_writer.OpenWorksheet(sheetName);
			_writer.Row({
				"Model", "5m Games", "Goals", "Shots", "Shot Conversion", "Assists", "Assisted Goals",
				"Saves", "Save %", "Passes", "Ball Touches", "Touch Share", "Bumps", "Demos",
				"Boost Steals", "Big Boost", "Small Boost", "Boost/min", "Boost", "Low Boost Time", "Off. Third", "Neutral Third",
				"Def. Third", "Speed", "Low Air", "High Air", "Supersonic", "Flip Resets",
				"Touch Height", "Speed To Ball", "Dist. To Ball"
			}, HEADER_ROW);

			for (const ModelBenchmarkStatRow& row : sortedStats) {
				const ModelBenchmarkOptionalStats& s = row.stats;
				auto expanded = [&](double value, int decimals, const string& suffix = string()) {
					return s.expandedStatsAvailable ? Fixed(value, decimals) + suffix : string("N/A");
				};
				auto efficiency = [&](double value, int decimals, const string& suffix = string()) {
					return s.efficiencyStatsAvailable ? Fixed(value, decimals) + suffix : string("N/A");
				};

				_writer.Row({
					row.modelName,
					Fixed(s.equivalentGames, 2),
					Fixed(s.goals, 2),
					Fixed(s.shots, 2),
					Fixed(s.shotConversionPercent, 1) + "%",
					Fixed(s.assists, 2),
					Fixed(s.assistedGoalPercent, 1) + "%",
					Fixed(s.saves, 2),
					efficiency(s.savePercent, 1, "%"),
					s.passesAvailable ? Fixed(s.passes, 2) : string("N/A"),
					Fixed(s.ballTouches, 2),
					efficiency(s.touchSharePercent, 1, "%"),
					expanded(s.bumps, 2),
					Fixed(s.demos, 2),
					expanded(s.boostSteals, 2),
					Fixed(s.bigBoostPads, 2),
					Fixed(s.smallBoostPads, 2),
					efficiency(s.boostConsumptionPerMinute, 1),
					Fixed(s.averageBoost, 1),
					Fixed(s.lowBoostTimePercent, 1) + "%",
					expanded(s.offensiveThirdPercent, 1, "%"),
					expanded(s.neutralThirdPercent, 1, "%"),
					expanded(s.defensiveThirdPercent, 1, "%"),
					Fixed(s.averageSpeed, 0),
					expanded(s.lowAirTimePercent, 1, "%"),
					expanded(s.highAirTimePercent, 1, "%"),
					Fixed(s.averageSupersonicTime, 1) + "%",
					Fixed(s.flipResets, 2),
					Fixed(s.averageTouchHeight, 0),
					Fixed(s.averageSpeedTowardsBall, 0),
					Fixed(s.averageDistanceToBall, 0)
				});
			}

			_writer.CloseWorksheet();
		}

		void WriteRawStatistics()
		{
			_writer.OpenWorksheet("Raw Statistics");
			_writer.Row({
				"Model Index", "Opponent Index", "Mode", "Definition Version", "Goals", "Shots",
				"Assists", "Saves", "Goals Conceded", "Passes", "Pass Game Time Sum", "Ball Touches",
				"Opponent Ball Touches", "Bumps", "Demos",
				"Flip Resets", "Boost Steals", "Big Boost Pads", "Small Boost Pads", "Game Time Sum",
				"Boost Consumed",
				"Boost Sum", "Speed Sum", "Aerial Time Sum", "Low Aerial Time Sum", "High Aerial Time Sum",
				"Low Boost Time Sum", "Supersonic Time Sum", "Player Time Sum", "Opponent Half Ball Steps",
				"Offensive Third Ball Steps", "Neutral Third Ball Steps", "Defensive Third Ball Steps",
				"Touch Height Sum", "Touch Count", "Speed Towards Ball Sum", "Distance To Ball Sum",
				"Step Samples", "Player Samples"
			}, HEADER_ROW);

			for (const ModelBenchmarkStatTotalsRow& row : _result.statTotals) {
				const ModelBenchmarkStatTotals& s = row.totals;
				_writer.Row({
					to_string(row.modelIndex), to_string(row.opponentIndex), to_string(row.mode),
					to_string(s.definitionVersion), Precise(s.goals), Precise(s.shots), Precise(s.assists),
					Precise(s.saves), Precise(s.goalsConceded), Precise(s.passes), Precise(s.passGameTimeSum), Precise(s.ballTouches),
					Precise(s.opponentBallTouches),
					Precise(s.bumps), Precise(s.demos), Precise(s.flipResets), Precise(s.boostSteals),
					Precise(s.bigBoostPads), Precise(s.smallBoostPads), Precise(s.gameTimeSum), Precise(s.boostConsumed),
					Precise(s.boostSum),
					Precise(s.speedSum), Precise(s.aerialTimeSum), Precise(s.lowAerialTimeSum),
					Precise(s.highAerialTimeSum), Precise(s.lowBoostTimeSum), Precise(s.supersonicTimeSum),
					Precise(s.playerTimeSum), Precise(s.opponentHalfBallSteps), Precise(s.offensiveThirdBallSteps),
					Precise(s.neutralThirdBallSteps), Precise(s.defensiveThirdBallSteps), Precise(s.touchHeightSum),
					Precise(s.touchCount), Precise(s.speedTowardsBallSum), Precise(s.distanceToBallSum),
					to_string(s.stepSamples), to_string(s.playerSamples)
				});
			}

			_writer.CloseWorksheet();
		}

		SpreadsheetXmlWriter& _writer;
		const ModelBenchmarkExportInput& _input;
		const ModelBenchmarkResult& _result;
		map<int, int> _rankByModel;
	};
}

bool RocketForgeTraining::WriteModelBenchmarkExport(const fs::path& path, const ModelBenchmarkExportInput& input, string* error)
{
	if (error)
		error->clear();

	if (!input.result) {
		if (error)
			*error = "No benchmark result to export.";
		return false;
	}

	// A unique staging name keeps concurrent exports to one path from sharing a temp file.
	fs::path stagingPath = path;
	stagingPath += "." + to_string(random_device{}()) + ".tmp";

	bool written = false;
	try {
		// Binary mode keeps newlines inside cells as written on every platform.
		ofstream file(stagingPath, ios::binary | ios::trunc);
		if (file) {
			SpreadsheetXmlWriter writer(file);
			WorkbookWriter(writer, input).Write();
			writer.Finish();
			written = static_cast<bool>(file.flush());
		}
	} catch (const exception& failure) {
		if (error)
			*error = failure.what();
	}

	error_code filesystemError;
	if (!written) {
		fs::remove(stagingPath, filesystemError);
		if (error && error->empty())
			*error = "Unable to write the export file: " + PathText(stagingPath);
		return false;
	}

	fs::rename(stagingPath, path, filesystemError);
	if (filesystemError) {
		const string message = filesystemError.message();
		fs::remove(stagingPath, filesystemError);
		if (error)
			*error = "Unable to finalize the export file " + PathText(path) + ": " + message;
		return false;
	}

	return true;
}

string RocketForgeTraining::DefaultModelBenchmarkExportName()
{
	return "model-benchmark-" + BuildStorage::MakeTimestamp() + ".xml";
}

fs::path RocketForgeTraining::DefaultModelBenchmarkExportFolder()
{
	return fs::path(TRAINING_ROOT).parent_path() / "Model Benchmarks";
}

fs::path RocketForgeTraining::DefaultModelBenchmarkLibraryPath()
{
	return DefaultModelBenchmarkExportFolder() / "model-benchmarks.sqlite";
}
