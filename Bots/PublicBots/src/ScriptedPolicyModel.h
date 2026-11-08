#pragma once

#include <GigaLearnInfer/Models.h>

#include <torch/script.h>

namespace GGL
{
	/** Input layout expected by a packaged TorchScript policy. **/
	enum class ScriptedPolicyInput
	{
		Flat,
		PackedAttention
	};

	/**
	 * Inference-only policy backed by a public TorchScript artifact.
	 * PackedAttention splits [q, flattened kv, mask] before invoking the module.
	 **/
	class RG_IMEXPORT ScriptedPolicyModel final : public Model
	{
	public:
		ScriptedPolicyModel(
			std::string artifactName,
			int obsSize,
			int numActions,
			torch::Device device,
			ScriptedPolicyInput inputLayout,
			int attentionEntities = 0);

		torch::Tensor Forward(torch::Tensor input, bool halfPrec) override;
		torch::Tensor ForwardFeatures(torch::Tensor input, bool halfPrec) override;
		void Save(std::filesystem::path folder) override;
		void Load(std::filesystem::path folder, bool allowNotExist) override;
		torch::Tensor CopyParams() const override;

	private:
		std::string artifactName;
		ScriptedPolicyInput inputLayout;
		int attentionEntities;
		torch::jit::script::Module module;
		bool loaded = false;
	};
}
