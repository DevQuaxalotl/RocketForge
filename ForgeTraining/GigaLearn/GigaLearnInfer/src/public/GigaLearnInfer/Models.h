#pragma once
#include <RLGymCPP/Framework.h>
#include "FrameworkTorch.h"

#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/container/sequential.h>

#include "ModelConfig.h"

namespace GGL {

	inline void AddActivationFunc(torch::nn::Sequential& seq, ModelActivationType type) {
		switch (type) {
		case ModelActivationType::RELU:
			seq->push_back(torch::nn::ReLU());
			return;
		case ModelActivationType::LEAKY_RELU:
			seq->push_back(torch::nn::LeakyReLU());
			return;
		case ModelActivationType::SIGMOID:
			seq->push_back(torch::nn::Sigmoid());
			return;
		case ModelActivationType::TANH:
			seq->push_back(torch::nn::Tanh());
			return;
		}

		RG_ERR_CLOSE("Unknown activation function type: " << (int)type);
	}
	//////////////////////////

	class RG_IMEXPORT Model : public torch::nn::Module {
	public:
		const char* modelName;
		torch::Device device;
		torch::nn::Sequential seq;
		torch::nn::Sequential seqHalf;
		bool _seqHalfOutdated = true;
		ModelConfig config;

		Model() : config(PartialModelConfig{}), device({}), modelName(NULL) {} // Uninitialized init

		Model(
			const char* modelName,
			ModelConfig config,
			torch::Device device
		);

		virtual torch::Tensor Forward(torch::Tensor input, bool halfPrec);
		virtual torch::Tensor ForwardFeatures(torch::Tensor input, bool halfPrec);
		int GetFeatureSize() const;

		std::filesystem::path GetSuffixedSavePath(std::filesystem::path folder, std::string suffix) const {
			std::string filename = modelName + suffix ;
			for (char& c : filename)
				c = toupper(c);
			filename += ".lt";
			return folder / filename;
		}

		std::filesystem::path GetSavePath(std::filesystem::path folder) const {
			return GetSuffixedSavePath(folder, "");
		}

		virtual void Save(std::filesystem::path folder);
		virtual void Load(std::filesystem::path folder, bool allowNotExist);

		virtual torch::Tensor CopyParams() const;

		// NOTE: Resets parameters
		Model* MakeEmptyClone() {
			return new Model(modelName, config, device);
		}

		Model* MakeClone() {
			RG_NO_GRAD;

			Model* clone = MakeEmptyClone();
			auto fromParams = this->parameters();
			auto toParams = clone->parameters();
			for (int i = 0; i < fromParams.size(); i++)
				toParams[i].copy_(fromParams[i], true);
			return clone;
		}

		uint64_t GetParamCount() {
			uint64_t total = 0;
			for (auto& param : this->parameters()) {
				if (!param.requires_grad())
					continue;
				total += param.numel();
			}

			return total;
		}

		// NOTE: Releases any cached CUDA graphs captured from this model.
		virtual ~Model();
	};

	class RG_IMEXPORT ModelSet {
	public:
		std::map<std::string, Model*> map = {};

		Model* operator[](const std::string& name) { 
			auto itr = map.find(name);
			if (itr == map.end()) {
				return NULL;
			} else {
				return map[name];
			}
		};

		void Add(Model* model) {
			map[model->modelName] = model;
		}
		void Save(std::filesystem::path folder) {
			for (Model* model : *this)
				model->Save(folder);
		}

		void Load(std::filesystem::path folder, bool allowNotExist) {
			for (Model* model : *this)
				model->Load(folder, allowNotExist);
		}

		class ModelIterator : public std::iterator<std::forward_iterator_tag, Model*> {
		public:
			using MapItr = std::map<std::string, Model*>::iterator;
			MapItr _mapItr;

			ModelIterator(MapItr mapItr) : _mapItr(mapItr) {}

			ModelIterator& operator++() { ++_mapItr; return *this; }

			bool operator==(const ModelIterator& other) const { return _mapItr == other._mapItr; }
			bool operator!=(const ModelIterator& other) const { return _mapItr != other._mapItr; }

			Model*& operator*() const { return _mapItr->second; }
		};

		ModelIterator begin() {
			return map.begin();
		}

		ModelIterator end() {
			return map.end();
		}

		ModelSet CloneAll() {
			ModelSet clone = *this;
			for (Model*& model : clone)
				model = model->MakeClone();
			return clone;
		}

		void Free() {
			for (Model* model : *this)
				delete model;
			map.clear();
		}

	};
}