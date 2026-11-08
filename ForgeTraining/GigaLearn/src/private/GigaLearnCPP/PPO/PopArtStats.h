#pragma once

#include <GigaLearnInfer/FrameworkTorch.h>
#include <GigaLearnInfer/JsonArchive.h>

#include <string>
#include <vector>

namespace GGL {

	/** Running target statistics of one critic value head, in raw (unweighted) reward units. */
	struct PopArtHeadState {
		std::string name;
		bool initialized = false;
		double mean = 0.0;

		/** Running mean of squared targets; the head's spread is sqrt(meanSq - mean²). */
		double meanSq = 0.0;

		void Serialize(JsonArchive& ar);
	};

	/** The per-head scale in effect before an update, needed to keep outputs unchanged across it. */
	struct PopArtUpdateResult {
		torch::Tensor meansBefore;
		torch::Tensor sigmasBefore;

		/** [heads] bool: heads already initialized before the update, whose outputs must be preserved. */
		torch::Tensor preserveMask;
	};

	/**
	 * Per-head PopArt statistics (van Hasselt et al. 2016, arXiv 1602.07714).
	 * Each critic head outputs a normalized value n, and its raw value is σ·n + μ.
	 * Update() moves μ and σ towards a batch's targets;
	 * PreserveOutputs() then rescales the critic's output layer so raw predictions don't change.
	 */
	class PopArtStats {
	public:
		PopArtStats(const std::vector<std::string>& headNames, float beta, float sigmaFloor);

		int GetNumHeads() const { return (int)_heads.size(); }
		const std::vector<PopArtHeadState>& GetHeads() const { return _heads; }

		/** Replaces every head's state, e.g. from a checkpoint. Names must match this layout, in order. */
		void SetHeads(std::vector<PopArtHeadState> heads);

		/**
		 * Takes each head's state from the saved head with the same name, in any order.
		 * A head with no saved counterpart becomes uninitialized, so its next batch sets its stats directly.
		 */
		void RemapFrom(const std::vector<PopArtHeadState>& savedHeads);

		/** [heads] float CPU; an uninitialized head uses 0. */
		torch::Tensor GetMeans() const;

		/** [heads] float CPU, never below the floor; an uninitialized head uses 1. */
		torch::Tensor GetSigmas() const;

		/** Initialized heads whose unfloored target std is below the floor. */
		int CountHeadsAtFloor() const;

		/** [..., heads] normalized values to raw values. */
		torch::Tensor Denormalize(const torch::Tensor& normalized) const;

		/** [..., heads] raw values to normalized values. */
		torch::Tensor Normalize(const torch::Tensor& raw) const;

		/**
		 * Folds raw targets [samples × heads] into the stats, using only cells where presence is nonzero.
		 * A head's first batch sets its stats directly; later batches move them by beta.
		 */
		PopArtUpdateResult Update(const torch::Tensor& targets, const torch::Tensor& presence);

		/**
		 * Rescales an output layer's rows in place (weight [heads × inputs], bias [heads])
		 * so each preserved head's raw output under the current stats equals its raw output before the update.
		 */
		void PreserveOutputs(torch::Tensor weight, torch::Tensor bias, const PopArtUpdateResult& update) const;

	private:
		std::vector<PopArtHeadState> _heads;
		double _beta;
		double _sigmaFloor;

		double UnflooredSigma(const PopArtHeadState& head) const;
	};
}
