#pragma once

#include <cstdlib>

namespace GGL {

	/** Exactly "1" enables and any other set value disables; an unset variable falls back to defaultEnabled. **/
	inline bool RuntimeOptionEnabled(const char* name, bool defaultEnabled = false) {
		const char* value = std::getenv(name);
		return value ? value[0] == '1' && value[1] == '\0' : defaultEnabled;
	}

}
