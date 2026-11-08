#pragma once

#include <GigaLearnInfer/FrameworkTorch.h>

#include <torch/optim/sgd.h>

namespace GGL {
	using MagSGDOptions = torch::optim::SGDOptions;

	/** SGD with gradients normalized to a predetermined update magnitude. **/
	class MagSGD : public torch::optim::SGD {
	public:
		explicit MagSGD(std::vector<torch::Tensor> params, MagSGDOptions defaults)
			: SGD(params, defaults) {}

		torch::Tensor step(LossClosure closure = nullptr) override {
			RG_NO_GRAD;

			torch::Tensor loss = {};
			if (closure != nullptr) {
				at::AutoGradMode enableGrad(true);
				loss = closure();
			}

			float gradientMagnitude = 0;
			for (auto& group : param_groups())
				for (auto& parameter : group.params())
					if (parameter.grad().defined())
						gradientMagnitude += parameter.grad().detach().square().sum().cpu().item<float>();
			gradientMagnitude = sqrtf(gradientMagnitude);

			for (auto& group : param_groups()) {
				for (auto& parameter : group.params()) {
					if (!parameter.grad().defined())
						continue;
					parameter.mutable_grad() /= gradientMagnitude;
				}
			}

			return SGD::step(closure);
		}
	};
}
