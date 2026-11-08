#include "Reward.h"

#include <cstring>

#ifndef _MSC_VER
#include <cxxabi.h>
#include <cstdlib>
#endif

using namespace RLGC;

std::vector<float> RLGC::Reward::GetAllRewards(const GameState& state, bool isFinal) {

	std::vector<float> rewards = std::vector<float>(state.players.size());
	for (int i = 0; i < state.players.size(); i++) {
		rewards[i] = GetReward(state.players[i], state, isFinal);
	}

	return rewards;
}

void RLGC::Reward::GetAllRewardsInto(const GameState& state, bool isFinal, std::vector<float>& out) {
	out = GetAllRewards(state, isFinal);
}

std::string RLGC::GetRewardTypeName(const std::type_info& type) {

#ifndef _MSC_VER
	std::string rewardName;
	{
		int status = 0;
		char* demangled = abi::__cxa_demangle(type.name(), nullptr, nullptr, &status);
		rewardName = (status == 0 && demangled) ? demangled : type.name();
		free(demangled);
	}
#else
	std::string rewardName = type.name();
#endif

	// Trim the string to after cetain keys
	{
		constexpr const char* TRIM_KEYS[] = {
			"::", // Namespace separator
			" " // Any spaces
		};
		for (const char* key : TRIM_KEYS) {
			size_t idx = rewardName.rfind(key);
			if (idx == std::string::npos)
				continue;

			rewardName.erase(rewardName.begin(), rewardName.begin() + idx + strlen(key));
		}
	}

	return rewardName;
}

std::string RLGC::Reward::GetName() {
	if (_cachedName.empty())
		_cachedName = GetRewardTypeName(typeid(*this));

	return _cachedName;
}
