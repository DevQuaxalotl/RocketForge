#pragma once

#include <GigaLearnInfer/Agent.h>

/** Inference-only RocketForge registration for the published Necto policy. **/
class NectoAgent final : public Agent
{
public:
	NectoAgent();
	std::unique_ptr<Agent> Clone() const override { return std::make_unique<NectoAgent>(*this); }
	void MakeInferenceModels(
		int obsSize,
		int numActions,
		const c10::Device& device,
		GGL::ModelSet& outModels) const override;
};

/** Inference-only RocketForge registration for the published Nexto policy. **/
class NextoAgent final : public Agent
{
public:
	NextoAgent();
	std::unique_ptr<Agent> Clone() const override { return std::make_unique<NextoAgent>(*this); }
	void MakeInferenceModels(
		int obsSize,
		int numActions,
		const c10::Device& device,
		GGL::ModelSet& outModels) const override;
};

/** Inference-only RocketForge registration for the published Element policy. **/
class ElementAgent final : public Agent
{
public:
	ElementAgent();
	std::unique_ptr<Agent> Clone() const override { return std::make_unique<ElementAgent>(*this); }
	void MakeInferenceModels(
		int obsSize,
		int numActions,
		const c10::Device& device,
		GGL::ModelSet& outModels) const override;
};
