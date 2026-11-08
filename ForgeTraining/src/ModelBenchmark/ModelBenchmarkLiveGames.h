#pragma once

#include "ModelBenchmark.h"
#include <stdexcept>

namespace RocketForgeTraining {
	/** Emits immutable completed mirror groups; unfinished games can change in place. **/
	class ModelBenchmarkLiveGameDeltas {
	public:
		int64_t Append(const std::vector<ModelBenchmarkGameMetadata>& games, ModelBenchmarkResult& snapshot) {
			const int64_t offset = (int64_t)_sent;
			for (; _scanned < games.size(); ++_scanned)
				_pending.push_back(_scanned);

			size_t retained = 0;
			for (size_t index : _pending) {
				if (games[index].mirrorCompleted)
					snapshot.games.push_back(games[index]);
				else
					_pending[retained++] = index;
			}

			_pending.resize(retained);
			_sent += snapshot.games.size();

			return offset;
		}

	private:
		size_t _scanned = 0;
		size_t _sent = 0;
		std::vector<size_t> _pending;
	};

	/** Reconstructs negotiated deltas before GUI progress updates can be coalesced. **/
	class ModelBenchmarkLiveGames {
	public:
		void Reset() { _games.clear(); }

		void Apply(ModelBenchmarkProgress& progress) {
			if (!progress.liveResult) {
				if (progress.liveGamesOffset >= 0)
					throw std::runtime_error("Benchmark game delta omitted its live result.");
				return;
			}

			if (progress.liveGamesOffset < 0) {
				_games = progress.liveResult->games;
				return;
			}

			if ((uint64_t)progress.liveGamesOffset != _games.size())
				throw std::runtime_error("Benchmark live game delta is missing or out of order.");

			std::shared_ptr<ModelBenchmarkResult> expanded = std::make_shared<ModelBenchmarkResult>(*progress.liveResult);
			_games.insert(_games.end(), expanded->games.begin(), expanded->games.end());
			expanded->games = _games;
			progress.liveResult = std::move(expanded);
			progress.liveGamesOffset = -1;
		}

	private:
		std::vector<ModelBenchmarkGameMetadata> _games;
	};
}
