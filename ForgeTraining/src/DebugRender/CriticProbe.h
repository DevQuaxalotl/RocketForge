#pragma once

#include <ForgeDebugger.h>
#include <RLGymCPP/Rewards/Reward.h>

#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

class Agent; // global namespace; see GigaLearnInfer/Agent.h

namespace GGL { class ModelSet; }

namespace RocketForge
{
	/** Statistics a checkpoint saved next to its critic, needed to read the critic's output in real units. */
	struct CriticCheckpointStats
	{
		/** One entry per critic output row; empty for a single-output critic. */
		std::vector<ForgeDebug::ValueHeadStats> valueHeads;

		/** Scale of a single-output critic's predictions; 1 when the checkpoint has no return stat. */
		float returnStd = 1.f;
	};

	/** Reads a checkpoint folder's RUNNING_STATS.json. Empty when the file is missing, unparsable, or malformed. */
	std::optional<CriticCheckpointStats> LoadCriticCheckpointStats(const std::filesystem::path& checkpointFolder);

	struct CriticProbeLoadConfig
	{
		/** Supplies the network shapes; must outlive TryLoad only. */
		const Agent* agent = nullptr;

		std::filesystem::path checkpointFolder;
		int numActions = 0;

		/** The checkpoint does not save the PopArt sigma floor, so it comes from the learner config. */
		float popArtSigmaFloor = 0.01f;

		/** Rewards the value heads are matched to by name, for their weights and categories. */
		const std::vector<RLGC::WeightedReward>* rewards = nullptr;

		bool useGPU = true;
	};

	/** Evaluates a checkpoint's critic on single observations and reports the value in real return units. */
	class CriticProbe
	{
	public:
		/** Null, after logging why, when the checkpoint has no critic this probe can load. */
		static std::unique_ptr<CriticProbe> TryLoad(const CriticProbeLoadConfig& config);

		~CriticProbe();
		CriticProbe(const CriticProbe&) = delete;
		CriticProbe& operator=(const CriticProbe&) = delete;

		/** V(s) for one observation. rewards must be the list last given to TryLoad or RemapHeads; its weights may have changed since. */
		ForgeDebug::CriticDebugData Evaluate(const std::vector<float>& obs, const std::vector<RLGC::WeightedReward>& rewards) const;

		/** Re-matches the value heads to a replacement reward list, which may hold a different subset of rewards. */
		void RemapHeads(const std::vector<RLGC::WeightedReward>& rewards);

		bool IsDecomposed() const { return !_heads.empty(); }

	private:
		CriticProbe() = default;

		std::unique_ptr<GGL::ModelSet> _models;
		int _obsSize = 0;
		bool _useGPU = true;
		float _sigmaFloor = 0.01f;
		float _returnStd = 1.f;

		/** Critic row order. */
		std::vector<ForgeDebug::DecomposedCriticHead> _heads;

		/** Per head, the index of the same-named reward in the bound list, or -1. */
		std::vector<int> _headRewardIndices;
	};
}
