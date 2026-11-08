#pragma once
#include "StateSetter.h"

namespace RLGC {
	class KickoffState : public StateSetter {
	public:
		KickoffState() { setterType = StateSetterType::Kickoff; }

		void ResetArena(Arena* arena) {
			arena->ResetToRandomKickoff();
		}
	};
}