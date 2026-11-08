#include "DefaultAction.h"

#include <algorithm>

RLGC::DefaultAction::DefaultAction() {
	constexpr float
		// Boolean input
		R_B[] = { 0, 1 },

		R_F[] = { -1, 0, 1 };

	// TODO: Use std permutations here or whatever			

	// Ground
	for (float throttle : R_F) {
		for (float steer : R_F) {
			for (float boost : R_B) {
				for (float handbrake : R_B) {
					// Prevent useless throttle when boosting
					if (boost == 1 && throttle != 1)
						continue;

					actions.push_back(
						{
							throttle, steer, 0, steer, 0, 0, boost, handbrake
						}
					);
				}
			}
		}
	}

	int numGroundActions = actions.size();

	// Aerial
	for (float pitch : R_F) {
		for (float yaw : R_F) {
			for (float roll : R_F) {
				for (float jump : R_B) {
					for (float boost : R_B) {
						// Only need roll for sideflip
						if (jump == 1 && yaw != 0)
							continue;

						// Duplicate with ground
						if (pitch == roll && roll == jump && jump == 0)
							continue;

						// Enable handbrake for potential wavedashes
						float handbrake = (jump == 1) && (pitch != 0 || yaw != 0 || roll != 0);

						actions.push_back(
							{
								boost, yaw, pitch, yaw, roll, jump, boost, handbrake
							}
						);
					}
				}
			}
		}
	}

	groundMask.resize(actions.size());
	airMask.resize(actions.size());
	jumpMask.resize(actions.size());
	boostMask.resize(actions.size());

	for (int i = 0; i < actions.size(); i++) {
		Action& action = actions[i];

		if (action.jump)
			jumpMask[i] = true;

		if (action.boost)
			boostMask[i] = true;

		if (i < numGroundActions)
			groundMask[i] = true;

		if (i >= numGroundActions && !action.jump)
			airMask[i] = true;

		// Add additional yaw-only actions to air mask
		// These actions were skipped during air action generation to prevent duplicates
		if (i < numGroundActions) {
			if (action.throttle == action.boost && (action.yaw != 0) == (action.handbrake != 0)) {
				airMask[i] = true;
			}
		}
	}

	BuildCachedMasks();
}

std::vector<uint8_t> RLGC::DefaultAction::GetActionMask(const Player& player, const GameState& state) {
	std::vector<uint8_t> result;
	GetActionMaskInto(player, state, result);
	return result;
}

void RLGC::DefaultAction::GetActionMaskInto(const Player& player, const GameState& state, std::vector<uint8_t>& result) {
	bool isTurtled = player.worldContact.hasContact && player.worldContact.contactNormal.z > 0.9f;
	int maskIdx = (player.isOnGround ? 1 : 0)
		| (player.boost != 0 ? 2 : 0)
		| ((player.HasFlipOrJump() || isTurtled) ? 4 : 0);
	const std::vector<uint8_t>& cached = cachedMasks[maskIdx];
	result.resize(cached.size());
	std::copy(cached.begin(), cached.end(), result.begin());
}

void RLGC::DefaultAction::BuildCachedMasks() {
	auto fnBuildMask = [&](bool onGround, bool hasBoost, bool canJump) {
		std::vector<uint8_t> result(actions.size(), false);

		auto fnApplyMask = [&](const std::vector<uint8_t>& mask, bool add) {
			if (add) {
				for (int i = 0; i < actions.size(); i++)
					result[i] |= mask[i];
			} else {
				for (int i = 0; i < actions.size(); i++)
					result[i] &= ~mask[i];
			}
		};

		fnApplyMask(onGround ? groundMask : airMask, true);

		if (!hasBoost)
			fnApplyMask(boostMask, false);

		if (canJump)
			fnApplyMask(jumpMask, true);

		return result;
	};

	for (int onGround = 0; onGround <= 1; onGround++) {
		for (int hasBoost = 0; hasBoost <= 1; hasBoost++) {
			for (int canJump = 0; canJump <= 1; canJump++) {
				int maskIdx = onGround | (hasBoost << 1) | (canJump << 2);
				cachedMasks[maskIdx] = fnBuildMask(onGround != 0, hasBoost != 0, canJump != 0);
			}
		}
	}
}
