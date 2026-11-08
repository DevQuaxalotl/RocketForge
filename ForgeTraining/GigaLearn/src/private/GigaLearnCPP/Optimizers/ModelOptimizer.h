#pragma once

#include <GigaLearnInfer/Models.h>

#include <memory>

namespace GGL {
	/** Owns the optimizer state associated with a learner's models. **/
	class GGL_TRAIN_IMEXPORT ModelOptimizerSet {
	public:
		ModelOptimizerSet();
		~ModelOptimizerSet();

		ModelOptimizerSet(const ModelOptimizerSet&) = delete;
		ModelOptimizerSet& operator=(const ModelOptimizerSet&) = delete;

		void Attach(ModelSet& models);
		void Clear();
		bool Has(const Model& model) const;
		void SetLearningRate(Model& model, float learningRate);
		void Step(Model& model);
		void StepAll(ModelSet& models);
		void Save(ModelSet& models, const std::filesystem::path& folder);
		void Load(ModelSet& models, const std::filesystem::path& folder);

	private:
		struct Impl;
		std::unique_ptr<Impl> impl;
	};
}
