#pragma once

#include <GigaLearnInfer/Framework.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <limits>
#include <map>
#include <string>

namespace GGL {

	/** Key of the entry array in a Build's skill reference registry. **/
	inline constexpr const char* SKILL_REFERENCES_LIST_KEY = "skill_references";

	/** Key of the per-mode MMR object in a skill reference registry entry. **/
	inline constexpr const char* SKILL_REFERENCE_RATINGS_KEY = "skill_ratings";

	/**
	 * Reads a skill reference's stored MMR per mode from its "skill_ratings" object.
	 * A mode whose value is not a number that is finite as a float is left out, so it stays pending calibration;
	 * each one dropped is logged under referenceLabel, since the file is hand-editable.
	 **/
	inline std::map<std::string, float> ReadSkillReferenceRatings(const nlohmann::json& ratingsJson, const std::string& referenceLabel) {
		std::map<std::string, float> ratings = {};
		if (ratingsJson.is_null())
			return ratings;

		if (!ratingsJson.is_object()) {
			RG_LOG(" > Skill reference " << referenceLabel << " has a non-object " << SKILL_REFERENCE_RATINGS_KEY
				<< "; every mode is pending calibration");
			return ratings;
		}

		for (const auto& [mode, value] : ratingsJson.items()) {
			const double rating = value.is_number() ? value.get<double>() : NAN;
			if (!std::isfinite(rating) || std::abs(rating) > std::numeric_limits<float>::max()) {
				RG_LOG(" > Skill reference " << referenceLabel << " has an invalid " << mode << " rating (" << value.dump()
					<< "); that mode is pending calibration");
				continue;
			}

			ratings[mode] = static_cast<float>(rating);
		}

		return ratings;
	}
}
