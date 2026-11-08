#pragma once

#include <GigaLearnInfer/Framework.h>

#include <torch/torch.h>

#include <cstdint>
#include <memory>

namespace GGL {
	class ModelSet;

	/** Runtime counters for fixed-slot recent-policy ensemble inference. **/
	struct RecentPolicyEnsembleStats {
		uint64_t captures = 0;
		uint64_t replays = 0;
		uint64_t fallbacks = 0;
		uint64_t captureFailures = 0;
		uint64_t weightCopies = 0;
		uint64_t graphEntries = 0;
		uint64_t graphEvictions = 0;
		uint64_t actualRows = 0;
		uint64_t evaluatedRows = 0;
	};

	/**
	 * Runs several architecture-compatible frozen policies as one fixed-slot batched
	 * network. Slot parameter storage never moves, so CUDA graphs remain valid when
	 * a new policy is copied into a draining slot.
	 **/
	class RG_IMEXPORT RecentPolicyEnsemble {
	public:
		explicit RecentPolicyEnsemble(int slotCount);
		~RecentPolicyEnsemble();

		RecentPolicyEnsemble(const RecentPolicyEnsemble&) = delete;
		RecentPolicyEnsemble& operator=(const RecentPolicyEnsemble&) = delete;

		int GetSlotCount() const;
		bool IsInitialized() const;
		bool IsCompatible(ModelSet& models) const;
		void SetSlot(int slot, ModelSet& models);
		void ClearSlot(int slot);

		/** Inputs are [active-slot, rows-per-slot, feature/action]. **/
		torch::Tensor InferPolicyProbs(
			torch::Tensor obs,
			torch::Tensor actionMasks,
			float temperature,
			float minActionProbability = 0,
			bool useCudaGraph = false);

		void InferActions(
			torch::Tensor obs,
			torch::Tensor actionMasks,
			bool deterministic,
			float temperature,
			torch::Tensor* outActions,
			float minActionProbability = 0,
			bool useCudaGraph = false);

		RecentPolicyEnsembleStats GetStats() const;
		void ResetStats();

	private:
		struct Impl;
		std::unique_ptr<Impl> impl;
	};
}
