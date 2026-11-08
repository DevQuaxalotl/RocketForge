#pragma once

#include <GigaLearnInfer/Framework.h>

namespace GGL {
	namespace KeyPressDetector {
		GGL_TRAIN_IMEXPORT char GetPressedChar();

		// Returns the pressed char without blocking. Returns '\0' if no key is currently pressed.
		char TryGetPressedChar();
	}
}