#pragma once

#include <GigaLearnInfer/Agent.h>

class ExampleTrainingAgent : public Agent
{
public:
	ExampleTrainingAgent();

	std::unique_ptr<Agent> Clone() const override { return std::make_unique<ExampleTrainingAgent>(*this); }
};