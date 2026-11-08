#pragma once

#include <GigaLearnInfer/JsonArchive.h>

#include <string>

namespace RocketForgeTraining {

	/** Kind of setting a StageParameterOverride replaces. */
	enum class StageParameterOverrideKind {
		RewardWeight,
		Hyperparameter,
	};

	/** One named override applied after the selected training stage builds its defaults. */
	struct StageParameterOverride {
		StageParameterOverrideKind kind = StageParameterOverrideKind::Hyperparameter;
		std::string group;
		std::string name;
		std::string value;

		void Serialize(GGL::JsonArchive& ar);
	};
}
