#include "ModelOptimizer.h"

#include "MagSGD.h"

#include <torch/csrc/api/include/torch/serialize.h>
#include <torch/optim/adagrad.h>
#include <torch/optim/adam.h>
#include <torch/optim/adamw.h>
#include <torch/optim/rmsprop.h>

#include <fstream>
#include <unordered_map>

namespace {
	std::unique_ptr<torch::optim::Optimizer> MakeTorchOptimizer(
		GGL::ModelOptimType type,
		const std::vector<torch::Tensor>& parameters)
	{
		switch (type) {
		case GGL::ModelOptimType::ADAM:
			return std::make_unique<torch::optim::Adam>(parameters, 0);
		case GGL::ModelOptimType::ADAMW:
			return std::make_unique<torch::optim::AdamW>(parameters, 0);
		case GGL::ModelOptimType::ADAGRAD:
			return std::make_unique<torch::optim::Adagrad>(parameters, 0);
		case GGL::ModelOptimType::RMSPROP:
			return std::make_unique<torch::optim::RMSprop>(parameters, 0);
		case GGL::ModelOptimType::MAGSGD:
			return std::make_unique<GGL::MagSGD>(parameters, 0);
		}

		RG_ERR_CLOSE("Unknown optimizer type: " << static_cast<int>(type));
		return {};
	}

	class TorchModelOptimizer {
	public:
		TorchModelOptimizer(GGL::ModelOptimType type, const std::vector<torch::Tensor>& parameters)
			: type(type), optimizer(MakeTorchOptimizer(type, parameters)) {}

		void SetLearningRate(float learningRate) {
			for (auto& group : optimizer->param_groups()) {
				switch (type) {
				case GGL::ModelOptimType::ADAM:
					static_cast<torch::optim::AdamOptions&>(group.options()).lr(learningRate);
					break;
				case GGL::ModelOptimType::ADAMW:
					static_cast<torch::optim::AdamWOptions&>(group.options()).lr(learningRate);
					break;
				case GGL::ModelOptimType::ADAGRAD:
					static_cast<torch::optim::AdagradOptions&>(group.options()).lr(learningRate);
					break;
				case GGL::ModelOptimType::RMSPROP:
					static_cast<torch::optim::RMSpropOptions&>(group.options()).lr(learningRate);
					break;
				case GGL::ModelOptimType::MAGSGD:
					static_cast<GGL::MagSGDOptions&>(group.options()).lr(learningRate);
					break;
				}
			}
		}

		void Step() {
			optimizer->step();
			optimizer->zero_grad();
		}

		void Save(const std::filesystem::path& path) {
			torch::serialize::OutputArchive archive;
			optimizer->save(archive);
			archive.save_to(path.string());
		}

		void Load(const std::filesystem::path& path, const torch::Device& device) {
			torch::serialize::InputArchive archive;
			archive.load_from(path.string(), device);
			optimizer->load(archive);
		}

	private:
		GGL::ModelOptimType type;
		std::unique_ptr<torch::optim::Optimizer> optimizer;
	};
}

struct GGL::ModelOptimizerSet::Impl {
	std::unordered_map<Model*, std::unique_ptr<TorchModelOptimizer>> byModel;
};

GGL::ModelOptimizerSet::ModelOptimizerSet() : impl(std::make_unique<Impl>()) {}

GGL::ModelOptimizerSet::~ModelOptimizerSet() = default;

void GGL::ModelOptimizerSet::Attach(ModelSet& models) {
	impl->byModel.clear();
	for (Model* model : models) {
		impl->byModel.emplace(
			model,
			std::make_unique<TorchModelOptimizer>(model->config.optimType, model->parameters()));
	}
}

void GGL::ModelOptimizerSet::Clear() {
	impl->byModel.clear();
}

bool GGL::ModelOptimizerSet::Has(const Model& model) const {
	return impl->byModel.contains(const_cast<Model*>(&model));
}

void GGL::ModelOptimizerSet::SetLearningRate(Model& model, float learningRate) {
	auto itr = impl->byModel.find(&model);
	if (itr == impl->byModel.end())
		RG_ERR_CLOSE("ModelOptimizerSet::SetLearningRate(): model \"" << model.modelName << "\" has no optimizer");
	itr->second->SetLearningRate(learningRate);
}

void GGL::ModelOptimizerSet::Step(Model& model) {
	auto itr = impl->byModel.find(&model);
	if (itr == impl->byModel.end())
		RG_ERR_CLOSE("ModelOptimizerSet::Step(): model \"" << model.modelName << "\" has no optimizer");
	itr->second->Step();
	model._seqHalfOutdated = true;
}

void GGL::ModelOptimizerSet::StepAll(ModelSet& models) {
	for (Model* model : models)
		Step(*model);
}

void GGL::ModelOptimizerSet::Save(ModelSet& models, const std::filesystem::path& folder) {
	for (Model* model : models) {
		auto itr = impl->byModel.find(model);
		if (itr == impl->byModel.end())
			RG_ERR_CLOSE("ModelOptimizerSet::Save(): model \"" << model->modelName << "\" has no optimizer");
		itr->second->Save(model->GetSuffixedSavePath(folder, "_optim"));
	}
}

void GGL::ModelOptimizerSet::Load(ModelSet& models, const std::filesystem::path& folder) {
	for (Model* model : models) {
		auto itr = impl->byModel.find(model);
		if (itr == impl->byModel.end())
			RG_ERR_CLOSE("ModelOptimizerSet::Load(): model \"" << model->modelName << "\" has no optimizer");

		const std::filesystem::path path = model->GetSuffixedSavePath(folder, "_optim");
		if (!std::filesystem::exists(path)) {
			RG_LOG("WARNING: No optimizer found at " << path << ", optimizer will be reset");
			continue;
		}

		std::ifstream testStream(path, std::istream::ate | std::ios::binary);
		if (testStream.tellg() <= 0) {
			RG_LOG("WARNING: Saved optimizer at " << path << " is empty, optimizer will be reset");
			continue;
		}

		itr->second->Load(path, model->device);
	}
}
