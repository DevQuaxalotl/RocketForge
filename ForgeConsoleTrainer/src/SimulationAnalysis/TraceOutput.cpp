#include "TraceOutput.h"

#include "SnapshotReplay.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>

namespace RocketForge::SimulationAnalysis
{
	using namespace RocketSim;
	using namespace RLGC;

	void WriteActionCsv(std::ostream& out, const Action& action)
	{
		out << action.throttle << ','
			<< action.steer << ','
			<< action.pitch << ','
			<< action.yaw << ','
			<< action.roll << ','
			<< action.jump << ','
			<< action.boost << ','
			<< action.handbrake;
	}

	bool ActionsEqual(const Action& a, const Action& b)
	{
		return a.throttle == b.throttle &&
			a.steer == b.steer &&
			a.pitch == b.pitch &&
			a.yaw == b.yaw &&
			a.roll == b.roll &&
			a.jump == b.jump &&
			a.boost == b.boost &&
			a.handbrake == b.handbrake;
	}

	int CountAllowedActions(const std::vector<uint8_t>& mask)
	{
		return (int)std::count_if(mask.begin(), mask.end(), [](uint8_t value) {
			return value != 0;
		});
	}

	uint64_t HashActionMask(const std::vector<uint8_t>& mask)
	{
		uint64_t hash = 1469598103934665603ull;
		for (uint8_t value : mask)
		{
			hash ^= value ? 1ull : 0ull;
			hash *= 1099511628211ull;
		}
		return hash;
	}

	std::string MaskToBits(const std::vector<uint8_t>& mask)
	{
		std::string result;
		result.reserve(mask.size());
		for (uint8_t value : mask)
			result.push_back(value ? '1' : '0');
		return result;
	}

	uint64_t HashObsValues(const FList& obs)
	{
		uint64_t hash = 1469598103934665603ull;
		for (float value : obs)
		{
			uint32_t bits = 0;
			std::memcpy(&bits, &value, sizeof(bits));
			for (int byte = 0; byte < 4; byte++)
			{
				hash ^= (uint64_t)((bits >> (byte * 8)) & 0xffu);
				hash *= 1099511628211ull;
			}
		}
		return hash;
	}

	std::string ObsToValues(const FList& obs)
	{
		std::ostringstream ss;
		ss << std::setprecision(9);
		for (int i = 0; i < (int)obs.size(); i++)
		{
			if (i > 0)
				ss << ';';
			ss << obs[i];
		}
		return ss.str();
	}

	int FindActionIdForControls(const std::vector<Action>& actions, const Action& action)
	{
		for (int i = 0; i < (int)actions.size(); i++)
			if (ActionsEqual(actions[i], action))
				return i;
		return -1;
	}

	std::string BuildMacroPlanString(
		const std::vector<ActionMacro>& macros,
		const std::vector<Action>& actionLookup)
	{
		std::ostringstream ss;
		for (int i = 0; i < (int)macros.size(); i++)
		{
			if (i > 0)
				ss << '|';
			ss << FindActionIdForControls(actionLookup, macros[i].action) << 'x' << macros[i].ticks;
		}
		return ss.str();
	}

	float ComputeDodgeTimeout(const CarState& state)
	{
		const bool hasDodged = state.hasFlipped || state.isFlipping;
		const bool flipAvailable =
			state.hasJumped &&
			!state.hasDoubleJumped &&
			!hasDodged &&
			!state.obsOnGround;
		return flipAvailable
			? std::max(0.f, RLConst::DOUBLEJUMP_MAX_DELAY - state.airTimeSinceJump)
			: 0.f;
	}

	bool IsWallContact(const CarState& state)
	{
		if (!state.worldContact.hasContact)
			return false;
		const Vec& normal = state.worldContact.contactNormal;
		const float horizontal = std::sqrt(normal.x * normal.x + normal.y * normal.y);
		return horizontal > 0.45f && std::abs(normal.z) < 0.75f;
	}

	bool IsNearWall(const CarState& state)
	{
		return std::abs(state.pos.x) > CommonValues::SIDE_WALL_X - 550.f ||
			std::abs(state.pos.y) > CommonValues::BACK_WALL_Y - 650.f;
	}

	int CountWheelContacts(const CarState& state)
	{
		int count = 0;
		for (bool hasContact : state.wheelsWithContact)
			if (hasContact)
				count++;
		return count;
	}

	std::string WheelContactBits(const CarState& state)
	{
		std::string bits;
		bits.reserve(4);
		for (bool hasContact : state.wheelsWithContact)
			bits.push_back(hasContact ? '1' : '0');
		return bits;
	}

	void WriteCsvHeader(std::ostream& out)
	{
		// Keep selected_* and applied_* columns separate. They differ whenever
		// actionDelay or a segmented macro action is active.
		out << "source,bot_index,bot_team,car_id,episode,physics_tick,policy_step,policy_tick,tick_skip,action_delay,ticks_elapsed,"
			<< "seconds_elapsed,game_time_remaining,delta_time,infer_action,apply_queued_action,"
			<< "selected_action_id,selected_is_macro,selected_macro_index,applied_action_id,applied_base_action_id,"
			<< "macro_segment_index,macro_segment_tick,macro_segment_ticks,macro_ticks_left,macro_plan,"
			<< "mask_allowed_count,mask_hash,mask_bits,"
			<< "mask_boost,mask_on_ground,mask_was_on_ground_during_step,mask_obs_on_ground,mask_is_jumping,mask_is_flipping,"
			<< "mask_has_jumped,mask_has_flipped,mask_has_dodged,mask_has_double_jumped,"
			<< "mask_air_state,mask_jump_time,mask_air_time,mask_air_time_since_jump,mask_dodge_timeout,"
			<< "obs_size,obs_hash,obs_values,"
			<< "selected_throttle,selected_steer,selected_pitch,selected_yaw,selected_roll,selected_jump,selected_boost,selected_handbrake,"
			<< "applied_throttle,applied_steer,applied_pitch,applied_yaw,applied_roll,applied_jump,applied_boost,applied_handbrake,"
			<< "prev_throttle,prev_steer,prev_pitch,prev_yaw,prev_roll,prev_jump,prev_boost,prev_handbrake,"
			<< "x,y,z,vx,vy,vz,ang_vel_x,ang_vel_y,ang_vel_z,speed,boost,on_ground,obs_on_ground,wheels_contact_count,wheels_contact_bits,"
			<< "world_contact,wall_contact,near_wall,contact_normal_x,contact_normal_y,contact_normal_z,"
			<< "is_jumping,is_flipping,has_jumped,has_flipped,has_dodged,has_double_jumped,"
			<< "air_state,jump_time,air_time,air_time_since_jump,dodge_timeout,raw_air_state,raw_dodge_timeout,raw_demo_timeout,"
			<< "frame_num,rlbot_last_input_throttle,rlbot_last_input_steer,rlbot_last_input_pitch,rlbot_last_input_yaw,"
			<< "rlbot_last_input_roll,rlbot_last_input_jump,rlbot_last_input_boost,rlbot_last_input_handbrake,"
			<< "raw_dodge_elapsed,raw_dodge_dir_x,raw_dodge_dir_y,"
			<< "jump_edge,ball_touched_tick,obs_state_snapshot,"
			<< "model_selected_action_id,model_selected_throttle,model_selected_steer,model_selected_pitch,model_selected_yaw,"
			<< "model_selected_roll,model_selected_jump,model_selected_boost,model_selected_handbrake,"
			<< "model_applied_action_id,model_applied_throttle,model_applied_steer,model_applied_pitch,model_applied_yaw,"
			<< "model_applied_roll,model_applied_jump,model_applied_boost,model_applied_handbrake,post_state_snapshot\n";
	}

	void WriteCsvRow(std::ostream& out, const TraceSample& sample)
	{
		out << sample.source << ','
			<< sample.botIndex << ','
			<< sample.botTeam << ','
			<< sample.carId << ','
			<< sample.episode << ','
			<< sample.physicsTick << ','
			<< sample.policyStep << ','
			<< sample.policyTick << ','
			<< sample.tickSkip << ','
			<< sample.actionDelay << ','
			<< sample.ticksElapsed << ','
			<< sample.secondsElapsed << ','
			<< sample.gameTimeRemaining << ','
			<< sample.deltaTime << ','
			<< (sample.inferAction ? 1 : 0) << ','
			<< (sample.applyQueuedAction ? 1 : 0) << ','
			<< sample.selectedActionId << ','
			<< (sample.selectedIsMacro ? 1 : 0) << ','
			<< sample.selectedMacroIndex << ','
			<< sample.appliedActionId << ','
			<< sample.appliedBaseActionId << ','
			<< sample.macroSegmentIndex << ','
			<< sample.macroSegmentTick << ','
			<< sample.macroSegmentTicks << ','
			<< sample.macroTicksLeft << ','
			<< sample.macroPlan << ','
			<< sample.maskAllowedCount << ','
			<< sample.maskHash << ','
			<< sample.maskBits << ','
			<< sample.maskBoost << ','
			<< (sample.maskOnGround ? 1 : 0) << ','
			<< (sample.maskWasOnGroundDuringStep ? 1 : 0) << ','
			<< (sample.maskObsOnGround ? 1 : 0) << ','
			<< (sample.maskIsJumping ? 1 : 0) << ','
			<< (sample.maskIsFlipping ? 1 : 0) << ','
			<< (sample.maskHasJumped ? 1 : 0) << ','
			<< (sample.maskHasFlipped ? 1 : 0) << ','
			<< (sample.maskHasDodged ? 1 : 0) << ','
			<< (sample.maskHasDoubleJumped ? 1 : 0) << ','
			<< sample.maskAirState << ','
			<< sample.maskJumpTime << ','
			<< sample.maskAirTime << ','
			<< sample.maskAirTimeSinceJump << ','
			<< sample.maskDodgeTimeout << ','
			<< sample.obsSize << ','
			<< sample.obsHash << ','
			<< sample.obsValues << ',';
		WriteActionCsv(out, sample.selectedAction);
		out << ',';
		WriteActionCsv(out, sample.appliedAction);
		out << ',';
		WriteActionCsv(out, sample.prevAction);
		out << ','
			<< sample.pos.x << ','
			<< sample.pos.y << ','
			<< sample.pos.z << ','
			<< sample.vel.x << ','
			<< sample.vel.y << ','
			<< sample.vel.z << ','
			<< sample.angVel.x << ','
			<< sample.angVel.y << ','
			<< sample.angVel.z << ','
			<< sample.speed << ','
			<< sample.boost << ','
			<< (sample.onGround ? 1 : 0) << ','
			<< (sample.obsOnGround ? 1 : 0) << ','
			<< sample.wheelsContactCount << ','
			<< sample.wheelsContactBits << ','
			<< (sample.worldContact ? 1 : 0) << ','
			<< (sample.wallContact ? 1 : 0) << ','
			<< (sample.nearWall ? 1 : 0) << ','
			<< sample.contactNormal.x << ','
			<< sample.contactNormal.y << ','
			<< sample.contactNormal.z << ','
			<< (sample.isJumping ? 1 : 0) << ','
			<< (sample.isFlipping ? 1 : 0) << ','
			<< (sample.hasJumped ? 1 : 0) << ','
			<< (sample.hasFlipped ? 1 : 0) << ','
			<< (sample.hasDodged ? 1 : 0) << ','
			<< (sample.hasDoubleJumped ? 1 : 0) << ','
			<< sample.airState << ','
			<< sample.jumpTime << ','
			<< sample.airTime << ','
			<< sample.airTimeSinceJump << ','
			<< sample.dodgeTimeout << ','
			<< sample.rawAirState << ','
			<< sample.rawDodgeTimeout << ','
			<< sample.rawDemoTimeout << ','
			<< sample.frameNum << ',';
		WriteActionCsv(out, sample.rlbotLastInput);
		out << ','
			<< sample.rawDodgeElapsed << ','
			<< sample.rawDodgeDirX << ','
			<< sample.rawDodgeDirY << ','
			<< (sample.jumpEdge ? 1 : 0) << ','
			<< (sample.ballTouchedTick ? 1 : 0) << ','
			<< sample.stateSnapshot << ','
			<< sample.modelSelectedActionId << ',';
		WriteActionCsv(out, sample.modelSelectedAction);
		out << ',' << sample.modelAppliedActionId << ',';
		WriteActionCsv(out, sample.modelAppliedAction);
		out << ',' << sample.postStateSnapshot << '\n';
	}

	TraceSample MakeTraceSample(
		int episode,
		uint32_t carId,
		int physicsTick,
		int policyStep,
		int policyTick,
		int tickSkip,
		int actionDelay,
		int selectedActionId,
		bool selectedIsMacro,
		int selectedMacroIndex,
		int appliedActionId,
		int appliedBaseActionId,
		int macroSegmentIndex,
		int macroSegmentTick,
		int macroSegmentTicks,
		int macroTicksLeft,
		const std::string& macroPlan,
		const std::vector<uint8_t>& actionMask,
		const FList& obs,
		const Player& maskPlayer,
		const Action& selectedAction,
		const Action& appliedAction,
		const Action& prevAction,
		const CarState& carState,
		const GameState& obsState,
		bool jumpEdge,
		bool ballTouchedTick)
	{
		// Convert RocketSim state into a stable CSV row shape. Extra mechanics
		// fields are included so later analysis can mine new behavior without
		// changing the trace format.
		TraceSample sample = {};
		sample.episode = episode;
		sample.botIndex = 0;
		sample.botTeam = (int)Team::BLUE;
		sample.carId = carId;
		sample.physicsTick = physicsTick;
		sample.policyStep = policyStep;
		sample.policyTick = policyTick;
		sample.tickSkip = tickSkip;
		sample.actionDelay = actionDelay;
		sample.secondsElapsed = (float)(physicsTick + 1) / 120.f;
		sample.gameTimeRemaining = -1.f;
		sample.deltaTime = 1.f / 120.f;
		sample.inferAction = policyTick == 0;
		sample.applyQueuedAction = policyTick == actionDelay;
		sample.selectedActionId = selectedActionId;
		sample.selectedIsMacro = selectedIsMacro;
		sample.selectedMacroIndex = selectedMacroIndex;
		sample.appliedActionId = appliedActionId;
		sample.appliedBaseActionId = appliedBaseActionId;
		sample.macroSegmentIndex = macroSegmentIndex;
		sample.macroSegmentTick = macroSegmentTick;
		sample.macroSegmentTicks = macroSegmentTicks;
		sample.macroTicksLeft = macroTicksLeft;
		sample.macroPlan = macroPlan;
		sample.maskAllowedCount = CountAllowedActions(actionMask);
		sample.maskHash = HashActionMask(actionMask);
		sample.maskBits = MaskToBits(actionMask);
		sample.maskBoost = maskPlayer.boost;
		sample.maskOnGround = maskPlayer.isOnGround;
		sample.maskWasOnGroundDuringStep = maskPlayer.wasOnGroundDuringStep;
		sample.maskObsOnGround = maskPlayer.obsOnGround;
		sample.maskIsJumping = maskPlayer.isJumping;
		sample.maskIsFlipping = maskPlayer.isFlipping;
		sample.maskHasJumped = maskPlayer.hasJumped;
		sample.maskHasFlipped = maskPlayer.hasFlipped;
		sample.maskHasDodged = maskPlayer.hasDodged;
		sample.maskHasDoubleJumped = maskPlayer.hasDoubleJumped;
		sample.maskAirState = maskPlayer.airState;
		sample.maskJumpTime = maskPlayer.jumpTime;
		sample.maskAirTime = maskPlayer.airTime;
		sample.maskAirTimeSinceJump = maskPlayer.airTimeSinceJump;
		sample.maskDodgeTimeout = maskPlayer.dodgeTimeout;
		sample.obsSize = (int)obs.size();
		sample.obsHash = HashObsValues(obs);
		sample.obsValues = ObsToValues(obs);
		sample.selectedAction = selectedAction;
		sample.modelSelectedActionId = selectedActionId;
		sample.modelSelectedAction = selectedAction;
		sample.modelAppliedActionId = appliedActionId;
		sample.modelAppliedAction = appliedAction;
		sample.appliedAction = appliedAction;
		sample.prevAction = prevAction;
		sample.pos = carState.pos;
		sample.vel = carState.vel;
		sample.angVel = carState.angVel;
		sample.speed = carState.vel.Length();
		sample.boost = carState.boost;
		sample.onGround = carState.isOnGround;
		sample.obsOnGround = carState.obsOnGround;
		sample.wheelsContactCount = CountWheelContacts(carState);
		sample.wheelsContactBits = WheelContactBits(carState);
		sample.worldContact = carState.worldContact.hasContact;
		sample.wallContact = IsWallContact(carState);
		sample.nearWall = IsNearWall(carState);
		sample.contactNormal = carState.worldContact.contactNormal;
		sample.isJumping = carState.isJumping;
		sample.isFlipping = carState.isFlipping;
		sample.hasJumped = carState.hasJumped;
		sample.hasFlipped = carState.hasFlipped;
		sample.hasDodged = carState.hasFlipped || carState.isFlipping;
		sample.hasDoubleJumped = carState.hasDoubleJumped;
		sample.airState =
			carState.obsOnGround ? 0 :
			carState.isJumping ? 1 :
			carState.hasDoubleJumped ? 2 :
			carState.isFlipping ? 3 :
			4;
		sample.jumpTime = carState.jumpTime;
		sample.airTime = carState.airTime;
		sample.airTimeSinceJump = carState.airTimeSinceJump;
		sample.dodgeTimeout = ComputeDodgeTimeout(carState);
		sample.frameNum = (uint32_t)(physicsTick + 1);
		sample.rlbotLastInput = appliedAction;
		sample.jumpEdge = jumpEdge;
		sample.ballTouchedTick = ballTouchedTick;
		sample.stateSnapshot = SerializeGameStateSnapshot(obsState);
		return sample;
	}
}
