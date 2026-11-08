#pragma once

#include "PopArtStats.h"

#include <GigaLearnCPP/LearnerConfig.h>
#include <GigaLearnInfer/Models.h>

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace GGL {

	/** Fatal when the decomposed critic is enabled together with settings it can't work with. */
	void ValidateDecomposedCriticConfig(const LearnerConfig& config);

	/**
	 * Updates the PopArt stats from raw head targets [samples × heads],
	 * rescales the critic's output layer so its raw predictions are unchanged,
	 * and returns the targets normalized with the updated stats.
	 */
	torch::Tensor ApplyPopArtUpdate(
		PopArtStats& stats,
		Model& critic,
		const torch::Tensor& headTargets,
		const torch::Tensor& presence
	);

	/**
	 * Per-cell critic loss weights [samples × heads], scaled so mean(weights · squaredError) over the batch
	 * is the weighted average squared error over present cells.
	 * Equal weights every present cell the same; Contribution weights a cell by |w|·σ.
	 */
	torch::Tensor ComputeCriticLossWeights(
		const torch::Tensor& presence,
		const torch::Tensor& sampleWeights,
		const torch::Tensor& sigmas,
		CriticHeadLossWeighting weighting
	);

	/** Writes the PopArt head states into RUNNING_STATS JSON as "value_heads", in critic row order. */
	void SaveValueHeads(nlohmann::json& runningStats, const PopArtStats& stats);

	/** How a checkpoint's value heads relate to this run's. */
	struct ValueHeadsLoadResult {
		/** The checkpoint has different heads, or the same heads in a different row order. */
		bool layoutChanged = false;

		/** The checkpoint's head names in its critic row order; empty when the decomposed critic is off. */
		std::vector<std::string> savedHeadNames;
	};

	/**
	 * Reads "value_heads" into stats; stats is null when the decomposed critic is off.
	 * A changed layout remaps the stats by name, and the caller must remap the critic's output rows to match.
	 * Fatal when the checkpoint's critic is single-output and this run's is decomposed, or the reverse.
	 */
	ValueHeadsLoadResult LoadValueHeads(const nlohmann::json& runningStats, PopArtStats* stats);

	/**
	 * Copies a critic saved with a different value-head layout into critic.
	 * The trunk is copied unchanged and each output row is matched by head name.
	 * A head the checkpoint lacks gets a zero row and bias.
	 * Fatal when the trunks differ in shape.
	 */
	void RemapCriticHeads(
		Model& critic,
		Model& savedCritic,
		const std::vector<std::string>& savedHeadNames,
		const std::vector<std::string>& headNames
	);

	/** One-line summary of a head remap for the training log, e.g. "+1 added (C), -1 removed (B), critic optimizer reset". */
	std::string DescribeCriticHeadRemap(const std::vector<std::string>& savedHeadNames, const std::vector<std::string>& headNames);
}
