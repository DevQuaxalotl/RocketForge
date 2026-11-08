#include "SnapshotReplay.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace RocketForge::SimulationAnalysis
{
	using namespace RocketSim;
	using namespace RLGC;

	void AppendVecSnapshot(std::ostream& out, const Vec& vec)
	{
		out << vec.x << ':' << vec.y << ':' << vec.z;
	}

	void AppendRotMatSnapshot(std::ostream& out, const RotMat& rotMat)
	{
		AppendVecSnapshot(out, rotMat.forward);
		out << ':';
		AppendVecSnapshot(out, rotMat.right);
		out << ':';
		AppendVecSnapshot(out, rotMat.up);
	}

	void AppendPhysSnapshot(std::ostream& out, const PhysState& state)
	{
		AppendVecSnapshot(out, state.pos);
		out << ':';
		AppendRotMatSnapshot(out, state.rotMat);
		out << ':';
		AppendVecSnapshot(out, state.vel);
		out << ':';
		AppendVecSnapshot(out, state.angVel);
	}

	void AppendActionSnapshot(std::ostream& out, const Action& action)
	{
		out << action.throttle << ':' << action.steer << ':' << action.pitch << ':' << action.yaw << ':'
			<< action.roll << ':' << action.jump << ':' << action.boost << ':' << action.handbrake;
	}

	void AppendControlsSnapshot(std::ostream& out, const CarControls& controls)
	{
		out << controls.throttle << ':' << controls.steer << ':' << controls.pitch << ':' << controls.yaw << ':'
			<< controls.roll << ':' << controls.jump << ':' << controls.boost << ':' << controls.handbrake;
	}

	void AppendPlayerSnapshot(std::ostream& out, const Player& player)
	{
		out << player.index << ':' << player.carId << ':' << (int)player.team << ':';
		AppendPhysSnapshot(out, player);
		out << ':'
			<< player.boost << ':'
			<< (player.isOnGround ? 1 : 0) << ':'
			<< (player.wasOnGroundDuringStep ? 1 : 0) << ':'
			<< (player.obsOnGround ? 1 : 0) << ':'
			<< (player.wheelsWithContact[0] ? 1 : 0) << ':'
			<< (player.wheelsWithContact[1] ? 1 : 0) << ':'
			<< (player.wheelsWithContact[2] ? 1 : 0) << ':'
			<< (player.wheelsWithContact[3] ? 1 : 0) << ':'
			<< (player.hasJumped ? 1 : 0) << ':'
			<< (player.hasDoubleJumped ? 1 : 0) << ':'
			<< (player.hasFlipped ? 1 : 0) << ':';
		AppendVecSnapshot(out, player.flipRelTorque);
		out << ':'
			<< player.jumpTime << ':'
			<< (player.isFlipping ? 1 : 0) << ':'
			<< player.flipTime << ':'
			<< (player.isJumping ? 1 : 0) << ':'
			<< player.airTime << ':'
			<< player.airTimeSinceJump << ':'
			<< (player.isBoosting ? 1 : 0) << ':'
			<< player.boostingTime << ':'
			<< player.timeSinceBoosted << ':'
			<< (player.isSupersonic ? 1 : 0) << ':'
			<< player.supersonicTime << ':'
			<< player.handbrakeVal << ':'
			<< (player.isAutoFlipping ? 1 : 0) << ':'
			<< player.autoFlipTimer << ':'
			<< player.autoFlipTorqueScale << ':'
			<< (player.worldContact.hasContact ? 1 : 0) << ':';
		AppendVecSnapshot(out, player.worldContact.contactNormal);
		out << ':'
			<< player.carContact.otherCarID << ':'
			<< player.carContact.cooldownTimer << ':'
			<< (player.isDemoed ? 1 : 0) << ':'
			<< player.demoRespawnTimer << ':';
		AppendControlsSnapshot(out, player.lastControls);
		out << ':';
		AppendActionSnapshot(out, player.prevAction);
		out << ':'
			<< (int)player.airState << ':'
			<< (player.hasDodged ? 1 : 0) << ':'
			<< player.dodgeTimeout << ':'
			<< player.demolishedTimeout << ':'
			<< (player.ballTouchedStep ? 1 : 0) << ':'
			<< (player.ballTouchedTick ? 1 : 0);
	}

	std::string SerializeGameStateSnapshot(const GameState& state)
	{
		std::ostringstream ss;
		ss << std::setprecision(9);
		ss << "v=2;gm=" << (int)state.gameMode
			<< ";dt=" << state.deltaTime
			<< ";goal=" << (state.goalScored ? 1 : 0)
			<< ";last_touch=" << state.lastTouchCarID
			<< ";last_tick=" << state.lastTickCount;
		if (!state.lastArena)
			throw std::runtime_error("cannot record the snapshot's engine for a state with no arena");
		ss << ";engine=" << static_cast<int>(state.lastArena->GetEngine());
		ss << ";ball=";
		AppendPhysSnapshot(ss, state.ball);

		ss << ";pads=";
		for (int i = 0; i < (int)state.boostPads.size(); i++)
		{
			if (i > 0)
				ss << '|';
			const float timer = i < (int)state.boostPadTimers.size() ? state.boostPadTimers[i] : 0.f;
			ss << (state.boostPads[i] ? 1 : 0) << ':' << timer;
		}

		ss << ";players=" << state.players.size();
		for (const Player& player : state.players)
		{
			ss << '|';
			AppendPlayerSnapshot(ss, player);
		}
		return ss.str();
	}

	std::vector<std::string> SplitString(const std::string& value, char delimiter)
	{
		std::vector<std::string> result;
		size_t start = 0;
		while (start <= value.size())
		{
			const size_t end = value.find(delimiter, start);
			if (end == std::string::npos)
			{
				result.push_back(value.substr(start));
				break;
			}
			result.push_back(value.substr(start, end - start));
			start = end + 1;
		}
		return result;
	}

	const std::string& NextSnapshotToken(const std::vector<std::string>& tokens, size_t& index)
	{
		if (index >= tokens.size())
			throw std::runtime_error("malformed obs_state_snapshot");
		return tokens[index++];
	}

	float ReadSnapshotFloat(const std::vector<std::string>& tokens, size_t& index)
	{
		return std::stof(NextSnapshotToken(tokens, index));
	}

	int ReadSnapshotInt(const std::vector<std::string>& tokens, size_t& index)
	{
		return std::stoi(NextSnapshotToken(tokens, index));
	}

	uint32_t ReadSnapshotUInt(const std::vector<std::string>& tokens, size_t& index)
	{
		return (uint32_t)std::stoul(NextSnapshotToken(tokens, index));
	}

	bool ReadSnapshotBool(const std::vector<std::string>& tokens, size_t& index)
	{
		return ReadSnapshotFloat(tokens, index) != 0.f;
	}

	Vec ReadSnapshotVec(const std::vector<std::string>& tokens, size_t& index)
	{
		const float x = ReadSnapshotFloat(tokens, index);
		const float y = ReadSnapshotFloat(tokens, index);
		const float z = ReadSnapshotFloat(tokens, index);
		return Vec(x, y, z);
	}

	PhysState ReadSnapshotPhys(const std::vector<std::string>& tokens, size_t& index)
	{
		PhysState state = {};
		state.pos = ReadSnapshotVec(tokens, index);
		state.rotMat.forward = ReadSnapshotVec(tokens, index);
		state.rotMat.right = ReadSnapshotVec(tokens, index);
		state.rotMat.up = ReadSnapshotVec(tokens, index);
		state.vel = ReadSnapshotVec(tokens, index);
		state.angVel = ReadSnapshotVec(tokens, index);
		return state;
	}

	Action ReadSnapshotAction(const std::vector<std::string>& tokens, size_t& index)
	{
		Action action = {};
		action.throttle = ReadSnapshotFloat(tokens, index);
		action.steer = ReadSnapshotFloat(tokens, index);
		action.pitch = ReadSnapshotFloat(tokens, index);
		action.yaw = ReadSnapshotFloat(tokens, index);
		action.roll = ReadSnapshotFloat(tokens, index);
		action.jump = ReadSnapshotFloat(tokens, index);
		action.boost = ReadSnapshotFloat(tokens, index);
		action.handbrake = ReadSnapshotFloat(tokens, index);
		return action;
	}

	CarControls ReadSnapshotControls(const std::vector<std::string>& tokens, size_t& index)
	{
		CarControls controls = {};
		controls.throttle = ReadSnapshotFloat(tokens, index);
		controls.steer = ReadSnapshotFloat(tokens, index);
		controls.pitch = ReadSnapshotFloat(tokens, index);
		controls.yaw = ReadSnapshotFloat(tokens, index);
		controls.roll = ReadSnapshotFloat(tokens, index);
		controls.jump = ReadSnapshotBool(tokens, index);
		controls.boost = ReadSnapshotBool(tokens, index);
		controls.handbrake = ReadSnapshotBool(tokens, index);
		return controls;
	}

	Player ReadSnapshotPlayer(const std::string& value, int snapshotVersion)
	{
		const std::vector<std::string> tokens = SplitString(value, ':');
		size_t index = 0;
		Player player = {};
		player.index = ReadSnapshotInt(tokens, index);
		player.carId = ReadSnapshotUInt(tokens, index);
		player.team = (Team)ReadSnapshotInt(tokens, index);
		static_cast<PhysState&>(player) = ReadSnapshotPhys(tokens, index);
		player.boost = ReadSnapshotFloat(tokens, index);
		player.isOnGround = ReadSnapshotBool(tokens, index);
		player.wasOnGroundDuringStep = snapshotVersion >= 2 ? ReadSnapshotBool(tokens, index) : player.isOnGround;
		player.obsOnGround = ReadSnapshotBool(tokens, index);
		for (bool& wheel : player.wheelsWithContact)
			wheel = ReadSnapshotBool(tokens, index);
		player.hasJumped = ReadSnapshotBool(tokens, index);
		player.hasDoubleJumped = ReadSnapshotBool(tokens, index);
		player.hasFlipped = ReadSnapshotBool(tokens, index);
		player.flipRelTorque = ReadSnapshotVec(tokens, index);
		player.jumpTime = ReadSnapshotFloat(tokens, index);
		player.isFlipping = ReadSnapshotBool(tokens, index);
		player.flipTime = ReadSnapshotFloat(tokens, index);
		player.isJumping = ReadSnapshotBool(tokens, index);
		player.airTime = ReadSnapshotFloat(tokens, index);
		player.airTimeSinceJump = ReadSnapshotFloat(tokens, index);
		player.isBoosting = ReadSnapshotBool(tokens, index);
		player.boostingTime = ReadSnapshotFloat(tokens, index);
		player.timeSinceBoosted = ReadSnapshotFloat(tokens, index);
		player.isSupersonic = ReadSnapshotBool(tokens, index);
		player.supersonicTime = ReadSnapshotFloat(tokens, index);
		player.handbrakeVal = ReadSnapshotFloat(tokens, index);
		player.isAutoFlipping = ReadSnapshotBool(tokens, index);
		player.autoFlipTimer = ReadSnapshotFloat(tokens, index);
		player.autoFlipTorqueScale = ReadSnapshotFloat(tokens, index);
		player.worldContact.hasContact = ReadSnapshotBool(tokens, index);
		player.worldContact.contactNormal = ReadSnapshotVec(tokens, index);
		player.carContact.otherCarID = ReadSnapshotUInt(tokens, index);
		player.carContact.cooldownTimer = ReadSnapshotFloat(tokens, index);
		player.isDemoed = ReadSnapshotBool(tokens, index);
		player.demoRespawnTimer = ReadSnapshotFloat(tokens, index);
		player.lastControls = ReadSnapshotControls(tokens, index);
		player.prevAction = ReadSnapshotAction(tokens, index);
		player.airState = (uint8_t)ReadSnapshotInt(tokens, index);
		player.hasDodged = ReadSnapshotBool(tokens, index);
		player.dodgeTimeout = ReadSnapshotFloat(tokens, index);
		player.demolishedTimeout = ReadSnapshotFloat(tokens, index);
		player.ballTouchedStep = ReadSnapshotBool(tokens, index);
		player.ballTouchedTick = ReadSnapshotBool(tokens, index);
		if (index != tokens.size())
			throw std::runtime_error("unexpected extra player snapshot fields");
		return player;
	}

	const std::string& SnapshotValueOrThrow(const std::map<std::string, std::string>& values, const std::string& key)
	{
		const auto it = values.find(key);
		if (it == values.end())
			throw std::runtime_error("obs_state_snapshot missing key: " + key);
		return it->second;
	}

	GameState ParseGameStateSnapshot(const std::string& snapshot, std::optional<EngineId> fallbackEngine, EngineId& engineOut)
	{
		std::map<std::string, std::string> values;
		for (const std::string& part : SplitString(snapshot, ';'))
		{
			const size_t equals = part.find('=');
			if (equals == std::string::npos)
				throw std::runtime_error("malformed obs_state_snapshot section");
			values[part.substr(0, equals)] = part.substr(equals + 1);
		}

		const int snapshotVersion = std::stoi(SnapshotValueOrThrow(values, "v"));
		if (snapshotVersion < 1 || snapshotVersion > 2)
			throw std::runtime_error("unsupported obs_state_snapshot version");

		const auto engineValue = values.find("engine");
		if (engineValue != values.end())
			engineOut = EngineFromVersion(std::stoi(engineValue->second));
		else if (fallbackEngine)
			engineOut = *fallbackEngine;
		else
			throw std::runtime_error("obs_state_snapshot records no engine and no agent engine is known");

		GameState state = {};
		state.gameMode = (GameMode)std::stoi(SnapshotValueOrThrow(values, "gm"));
		state.deltaTime = std::stof(SnapshotValueOrThrow(values, "dt"));
		state.goalScored = std::stoi(SnapshotValueOrThrow(values, "goal")) != 0;
		state.lastTouchCarID = std::stoi(SnapshotValueOrThrow(values, "last_touch"));
		state.lastTickCount = (uint64_t)std::stoull(SnapshotValueOrThrow(values, "last_tick"));

		std::vector<std::string> ballTokens = SplitString(SnapshotValueOrThrow(values, "ball"), ':');
		size_t ballIndex = 0;
		static_cast<PhysState&>(state.ball) = ReadSnapshotPhys(ballTokens, ballIndex);
		if (ballIndex != ballTokens.size())
			throw std::runtime_error("unexpected extra ball snapshot fields");

		const std::vector<std::string> padTokens = SplitString(SnapshotValueOrThrow(values, "pads"), '|');
		state.boostPads.assign(padTokens.size(), true);
		state.boostPadsInv.assign(padTokens.size(), true);
		state.boostPadTimers.assign(padTokens.size(), 0.f);
		state.boostPadTimersInv.assign(padTokens.size(), 0.f);
		for (int i = 0; i < (int)padTokens.size(); i++)
		{
			std::vector<std::string> padFields = SplitString(padTokens[i], ':');
			if (padFields.size() != 2)
				throw std::runtime_error("malformed boost pad snapshot");
			state.boostPads[i] = std::stof(padFields[0]) != 0.f;
			state.boostPadTimers[i] = std::stof(padFields[1]);
			const int invIdx = (int)padTokens.size() - i - 1;
			state.boostPadsInv[invIdx] = state.boostPads[i];
			state.boostPadTimersInv[invIdx] = state.boostPadTimers[i];
		}

		const std::vector<std::string> playerParts = SplitString(SnapshotValueOrThrow(values, "players"), '|');
		if (playerParts.empty())
			throw std::runtime_error("malformed players snapshot");
		const int playerCount = std::stoi(playerParts[0]);
		if (playerCount != (int)playerParts.size() - 1)
			throw std::runtime_error("players snapshot count mismatch");
		state.players.clear();
		state.players.reserve(playerCount);
		for (int i = 1; i < (int)playerParts.size(); i++)
			state.players.push_back(ReadSnapshotPlayer(playerParts[i], snapshotVersion));
		return state;
	}

	std::vector<std::string> SplitCsvRow(const std::string& row)
	{
		// The trace writer deliberately avoids commas inside complex fields, so a
		// plain split is enough and avoids bringing a CSV parser into this utility.
		return SplitString(row, ',');
	}

	std::string CsvFieldAt(const std::string& row, int targetColumn)
	{
		int column = 0;
		size_t start = 0;
		while (start <= row.size())
		{
			const size_t end = row.find(',', start);
			if (column == targetColumn)
				return end == std::string::npos ? row.substr(start) : row.substr(start, end - start);
			if (end == std::string::npos)
				break;
			start = end + 1;
			column++;
		}
		return {};
	}

	float ReadTraceFloat(
		const std::map<std::string, int>& columns,
		const std::string& row,
		const std::string& name)
	{
		const auto it = columns.find(name);
		if (it == columns.end())
			throw std::runtime_error("replay CSV missing column: " + name);
		return std::stof(CsvFieldAt(row, it->second));
	}

	float ReadTraceFloatOrDefault(
		const std::map<std::string, int>& columns,
		const std::string& row,
		const std::string& name,
		float fallback)
	{
		const auto it = columns.find(name);
		if (it == columns.end())
			return fallback;
		return std::stof(CsvFieldAt(row, it->second));
	}

	int ReadTraceInt(
		const std::map<std::string, int>& columns,
		const std::string& row,
		const std::string& name)
	{
		const auto it = columns.find(name);
		if (it == columns.end())
			throw std::runtime_error("replay CSV missing column: " + name);
		return std::stoi(CsvFieldAt(row, it->second));
	}

	uint32_t ReadTraceUInt(
		const std::map<std::string, int>& columns,
		const std::string& row,
		const std::string& name)
	{
		const auto it = columns.find(name);
		if (it == columns.end())
			throw std::runtime_error("replay CSV missing column: " + name);
		return (uint32_t)std::stoul(CsvFieldAt(row, it->second));
	}

	bool ReadTraceBool(
		const std::map<std::string, int>& columns,
		const std::string& row,
		const std::string& name)
	{
		return ReadTraceFloat(columns, row, name) != 0.f;
	}

	std::string ReadTraceString(
		const std::map<std::string, int>& columns,
		const std::string& row,
		const std::string& name)
	{
		const auto it = columns.find(name);
		if (it == columns.end())
			throw std::runtime_error("replay CSV missing column: " + name);
		return CsvFieldAt(row, it->second);
	}

	Action ReadTraceAction(
		const std::map<std::string, int>& columns,
		const std::string& row,
		const std::string& prefix)
	{
		Action action = {};
		action.throttle = ReadTraceFloat(columns, row, prefix + "throttle");
		action.steer = ReadTraceFloat(columns, row, prefix + "steer");
		action.pitch = ReadTraceFloat(columns, row, prefix + "pitch");
		action.yaw = ReadTraceFloat(columns, row, prefix + "yaw");
		action.roll = ReadTraceFloat(columns, row, prefix + "roll");
		action.jump = ReadTraceFloat(columns, row, prefix + "jump");
		action.boost = ReadTraceFloat(columns, row, prefix + "boost");
		action.handbrake = ReadTraceFloat(columns, row, prefix + "handbrake");
		return action;
	}

	ControlReplayFrame ReadControlReplayFrame(
		const std::map<std::string, int>& columns,
		const std::string& row)
	{
		ControlReplayFrame frame = {};
		frame.carId = ReadTraceUInt(columns, row, "car_id");
		frame.botIndex = ReadTraceInt(columns, row, "bot_index");
		frame.botTeam = ReadTraceInt(columns, row, "bot_team");
		frame.physicsTick = ReadTraceInt(columns, row, "physics_tick");
		frame.policyStep = ReadTraceInt(columns, row, "policy_step");
		frame.policyTick = ReadTraceInt(columns, row, "policy_tick");
		frame.tickSkip = ReadTraceInt(columns, row, "tick_skip");
		frame.actionDelay = ReadTraceInt(columns, row, "action_delay");
		frame.ticksElapsed = ReadTraceInt(columns, row, "ticks_elapsed");
		frame.secondsElapsed = ReadTraceFloat(columns, row, "seconds_elapsed");
		frame.gameTimeRemaining = ReadTraceFloatOrDefault(columns, row, "game_time_remaining", -1.f);
		frame.deltaTime = ReadTraceFloat(columns, row, "delta_time");
		frame.inferAction = ReadTraceBool(columns, row, "infer_action");
		frame.applyQueuedAction = ReadTraceBool(columns, row, "apply_queued_action");
		frame.selectedActionId = ReadTraceInt(columns, row, "selected_action_id");
		frame.selectedIsMacro = ReadTraceBool(columns, row, "selected_is_macro");
		frame.selectedMacroIndex = ReadTraceInt(columns, row, "selected_macro_index");
		frame.appliedActionId = ReadTraceInt(columns, row, "applied_action_id");
		frame.appliedBaseActionId = ReadTraceInt(columns, row, "applied_base_action_id");
		frame.macroSegmentIndex = ReadTraceInt(columns, row, "macro_segment_index");
		frame.macroSegmentTick = ReadTraceInt(columns, row, "macro_segment_tick");
		frame.macroSegmentTicks = ReadTraceInt(columns, row, "macro_segment_ticks");
		frame.macroTicksLeft = ReadTraceInt(columns, row, "macro_ticks_left");
		frame.macroPlan = ReadTraceString(columns, row, "macro_plan");
		frame.selectedAction = ReadTraceAction(columns, row, "selected_");
		frame.appliedAction = ReadTraceAction(columns, row, "applied_");
		frame.prevAction = ReadTraceAction(columns, row, "prev_");
		frame.frameNum = ReadTraceUInt(columns, row, "frame_num");
		frame.rlbotLastInput = ReadTraceAction(columns, row, "rlbot_last_input_");
		return frame;
	}

	void ApplySnapshotPolicyState(GameState& state, const ControlReplayFrame& frame)
	{
		for (Player& player : state.players)
		{
			if (player.carId == frame.carId)
			{
				player.lastControls = (CarControls)frame.appliedAction;
				player.prevAction = frame.prevAction;
				return;
			}
		}
		throw std::runtime_error("snapshot traced car was missing from obs_state_snapshot");
	}

	SnapshotReplaySeed LoadSnapshotReplaySeed(
		const std::filesystem::path& csvPath,
		int snapshotTick,
		int snapshotBotIndex,
		int64_t snapshotCarId,
		std::optional<EngineId> fallbackEngine)
	{
		std::ifstream in(csvPath);
		if (!in.good())
			throw std::runtime_error("failed to open snapshot CSV: " + csvPath.string());

		std::string headerLine;
		if (!std::getline(in, headerLine))
			throw std::runtime_error("snapshot CSV is empty: " + csvPath.string());
		const std::vector<std::string> header = SplitCsvRow(headerLine);

		std::map<std::string, int> columns;
		for (int i = 0; i < (int)header.size(); i++)
			columns[header[i]] = i;

		auto getColumn = [&](const std::string& name) {
			const auto it = columns.find(name);
			if (it == columns.end())
				throw std::runtime_error("snapshot CSV missing column: " + name);
			return it->second;
		};

		const int tickCol = getColumn("physics_tick");
		const int botIndexCol = getColumn("bot_index");
		const int policyStepCol = getColumn("policy_step");
		const int carIdCol = getColumn("car_id");
		const int snapshotCol = getColumn("obs_state_snapshot");

		std::map<uint32_t, std::string> previousRowsByCarId;
		std::string rowLine;
		while (std::getline(in, rowLine))
		{
			const std::string tickField = CsvFieldAt(rowLine, tickCol);
			if (tickField.empty())
				continue;
			const uint32_t rowCarId = (uint32_t)std::stoul(CsvFieldAt(rowLine, carIdCol));
			if (std::stoi(tickField) != snapshotTick)
			{
				previousRowsByCarId[rowCarId] = rowLine;
				continue;
			}
			if (snapshotBotIndex >= 0 && std::stoi(CsvFieldAt(rowLine, botIndexCol)) != snapshotBotIndex)
			{
				previousRowsByCarId[rowCarId] = rowLine;
				continue;
			}
			if (snapshotCarId >= 0 && std::stoll(CsvFieldAt(rowLine, carIdCol)) != snapshotCarId)
			{
				previousRowsByCarId[rowCarId] = rowLine;
				continue;
			}

			SnapshotReplaySeed seed = {};
			seed.tracedCarId = rowCarId;
			seed.tracedBotIndex = std::stoi(CsvFieldAt(rowLine, botIndexCol));
			seed.sourcePhysicsTick = std::stoi(tickField);
			seed.sourcePolicyStep = std::stoi(CsvFieldAt(rowLine, policyStepCol));
			seed.state = ParseGameStateSnapshot(CsvFieldAt(rowLine, snapshotCol), fallbackEngine, seed.engine);
			seed.sourceFrame = ReadControlReplayFrame(columns, rowLine);
			ApplySnapshotPolicyState(seed.state, seed.sourceFrame);
			const auto previousIt = previousRowsByCarId.find(seed.tracedCarId);
			if (previousIt != previousRowsByCarId.end())
			{
				seed.previousAppliedAction = ReadTraceAction(columns, previousIt->second, "applied_");
				seed.hasPreviousAppliedAction = true;
			}
			return seed;
		}

		throw std::runtime_error("no snapshot row found for physics_tick " + std::to_string(snapshotTick));
	}

	ControlReplayTrace LoadControlReplayTrace(
		const std::filesystem::path& csvPath,
		uint32_t tracedCarId,
		int startTick,
		int maxTicks)
	{
		std::ifstream in(csvPath);
		if (!in.good())
			throw std::runtime_error("failed to open control replay CSV: " + csvPath.string());

		std::string headerLine;
		if (!std::getline(in, headerLine))
			throw std::runtime_error("control replay CSV is empty: " + csvPath.string());
		const std::vector<std::string> header = SplitCsvRow(headerLine);

		std::map<std::string, int> columns;
		for (int i = 0; i < (int)header.size(); i++)
			columns[header[i]] = i;

		const int endTick = startTick + maxTicks;
		ControlReplayTrace trace = {};
		trace.startTick = startTick;
		trace.endTick = endTick;

		std::string rowLine;
		while (std::getline(in, rowLine))
		{
			if (rowLine.empty())
				continue;

			ControlReplayFrame frame = ReadControlReplayFrame(columns, rowLine);
			if (frame.physicsTick < startTick || frame.physicsTick >= endTick)
				continue;

			trace.framesByTick[frame.physicsTick][frame.carId] = frame;
			trace.exactRows++;
			if (frame.carId == tracedCarId)
				trace.tracedRows++;
		}

		if (trace.tracedRows == 0)
			throw std::runtime_error("control replay CSV has no rows for traced car_id " + std::to_string(tracedCarId));

		return trace;
	}


	int FindArenaBoostPadIndex(Arena* arena, const Vec& targetPos)
	{
		for (int i = 0; i < (int)arena->_boostPads.size(); i++)
		{
			if (arena->_boostPads[i]->config.pos.DistSq2D(targetPos) < 10.f)
				return i;
		}
		return -1;
	}

	void ApplySnapshotBoostPads(Arena* arena, const GameState& snapshotState)
	{
		const int padCount = snapshotState.GetBoostLocationAmount();
		const Vec* boostLocations = snapshotState.GetBoostLocations();
		if ((int)snapshotState.boostPads.size() != padCount)
			throw std::runtime_error("snapshot boost pad count does not match game mode");

		for (int i = 0; i < padCount; i++)
		{
			const int arenaPadIndex = FindArenaBoostPadIndex(arena, boostLocations[i]);
			if (arenaPadIndex < 0)
				throw std::runtime_error("failed to map snapshot boost pad to arena pad");

			BoostPadState padState = {};
			padState.isActive = snapshotState.boostPads[i];
			padState.cooldown = i < (int)snapshotState.boostPadTimers.size()
				? snapshotState.boostPadTimers[i]
				: 0.f;
			arena->_boostPads[arenaPadIndex]->SetState(padState);
		}
	}

	std::unique_ptr<Arena> BuildArenaFromSnapshot(
		const SnapshotReplaySeed& seed,
		const CarConfig& carConfig,
		Car*& tracedCarOut)
	{
		ArenaConfig arenaConfig = {};
		arenaConfig.engine = seed.engine;
		std::unique_ptr<Arena> arena(Arena::Create(seed.state.gameMode, arenaConfig));
		arena->tickCount = seed.state.lastTickCount;
		arena->ball->SetState(seed.state.ball);
		ApplySnapshotBoostPads(arena.get(), seed.state);

		tracedCarOut = nullptr;
		for (const Player& player : seed.state.players)
		{
			Car* car = arena->AddCar(player.team, carConfig);
			arena->_carIDMap.erase(car->id);
			car->id = player.carId;
			arena->_carIDMap[car->id] = car;
			arena->_lastCarID = std::max(arena->_lastCarID, car->id);

			CarState carState = static_cast<const CarState&>(player);
			car->SetState(carState);
			car->controls = player.lastControls;
			if (player.carId == seed.tracedCarId)
				tracedCarOut = car;
		}

		if (!tracedCarOut)
			throw std::runtime_error("snapshot traced car was missing from player list");
		return arena;
	}
}