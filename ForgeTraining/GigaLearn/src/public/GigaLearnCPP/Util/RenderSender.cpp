#include "RenderSender.h"

#include <nlohmann/json.hpp>

using namespace nlohmann;
using namespace RLGC;

GGL::RenderSender::RenderSender(float timeScale) : timeScale(timeScale) {
	RG_LOG("Initializing RenderSender...");

	// The embedded interpreter runs GIL-released (see InitializeTraining), so any thread
	// touching Python must hold the GIL. This scope covers the pyMod assignment below.
	pybind11::gil_scoped_acquire gil;

	try {
		RG_LOG("Current dir: " << std::filesystem::current_path());
		pyMod = pybind11::module::import("python_scripts.render_receiver");
		bool rsvisLaunched = pyMod.attr("init")(std::string(PY_EXEC_PATH)).cast<bool>();
		if (!rsvisLaunched)
			RG_LOG("Warning: RocketSimVis not found in build output. Re-run CMake, or start it manually on port 9273.");
	} catch (std::exception& e) {
		RG_ERR_CLOSE("RenderSender: Failed to import render receiver, exception: " << e.what());
	}

	RG_LOG(" > RenderSender initalized.");
}

void GGL::RenderSender::SetRuntimeTimeScale(std::atomic<float>* runtimeScale)
{
	_runtimeTimeScale = runtimeScale;
}

std::vector<std::string> GGL::RenderSender::PollCommands()
{
	std::vector<std::string> commands;

	try {
		pybind11::gil_scoped_acquire gil;
		for (const pybind11::handle& datagram : pyMod.attr("poll_commands")())
			commands.push_back(datagram.cast<std::string>());
	} catch (std::exception& e) {
		RG_LOG("RenderSender: failed to poll viewer commands: " << e.what());
	}

	return commands;
}

FList VecToList(const Vec& vec) {
	return FList({ vec.x, vec.y, vec.z });
}

json PhysToJSON(const PhysState& obj) {
	json j = {};

	j["pos"] = VecToList(obj.pos);

	j["forward"] = VecToList(obj.rotMat.forward);
	j["right"] = VecToList(obj.rotMat.right);
	j["up"] = VecToList(obj.rotMat.up);

	j["vel"] = VecToList(obj.vel);
	j["ang_vel"] = VecToList(obj.angVel);

	return j;
}

json PlayerToJSON(const Player& player) {
	json j = {};

	j["car_id"] = player.carId;
	j["team_num"] = (int)player.team;
	j["phys"] = PhysToJSON(player);
	j["is_demoed"] = player.isDemoed;
	j["on_ground"] = player.isOnGround;
	j["ball_touched"] = player.ballTouchedStep;
	j["has_flip"] = player.HasFlipOrJump();
	j["boost_amount"] = player.boost / 100;

	return j;
}

json GameStateToJSON(const GameState& state) {
	json j = {};
	
	j["ball"] = PhysToJSON(state.ball);

	std::vector<json> players;
	for (auto& player : state.players)
		players.push_back(PlayerToJSON(player));

	j["players"] = players;
	j["boost_pads"] = state.boostPads;

	return j;
}

std::vector<json> ActionSetToJSON(const std::vector<Action>& actions) {
	std::vector<json> js = {};
	for (auto& action : actions) {
		FList vals  = FList(action.begin(), action.end());
		js.push_back(json(vals));
	}

	return js;
}

json DebugLineToJSON(const RLGC::DebugLine& line) {
	json j = {};
	j["start"] = VecToList(line.start);
	j["end"] = VecToList(line.end);
	j["color"] = VecToList(line.color);
	return j;
}

json DebugSnapshotToJSON(const ForgeDebug::RewardDebugSnapshot& snapshot) {
	json j = {};
	j["frame"] = snapshot.frameNumber;
	j["norm_mode"] = (int)snapshot.normMode;
	j["total_continuous"] = snapshot.totalContinuousWeighted;
	j["total_event"] = snapshot.totalEventWeighted;
	j["total"] = snapshot.totalWeighted;
	j["player_index"] = snapshot.playerIndex;
	j["player_car_id"] = snapshot.playerCarId;
	j["player_team"] = snapshot.playerTeam;

	std::vector<json> rewardJsons;
	for (auto& rd : snapshot.rewards) {
		json rj = {};
		rj["name"] = rd.name;
		rj["category"] = (int)rd.category;
		rj["raw"] = rd.rawValue;
		rj["weight"] = rd.weight;
		rj["weighted"] = rd.weightedValue;
		rj["abs_sum"] = rd.episodeAbsWeightedSum;
		rj["normalized"] = rd.normalizedValue;
		rj["event_decay"] = rd.eventDecayValue;
		rj["event_fired"] = rd.eventFiredThisFrame;
		rj["range_min"] = rd.declaredRange.minValue;
		rj["range_max"] = rd.declaredRange.maxValue;

		if (!rd.debugLines.empty()) {
			std::vector<json> lineJsons;
			for (auto& line : rd.debugLines)
				lineJsons.push_back(DebugLineToJSON(line));
			rj["debug_lines"] = lineJsons;
		}

		rewardJsons.push_back(rj);
	}
	j["rewards"] = rewardJsons;
	return j;
}

void GGL::RenderSender::Send(
	const GameState& state,
	const ForgeDebug::RewardDebugSnapshot* debugSnapshot,
	const std::string& contextLabel,
	const RenderViewState& view) {
	const float currentScale = _runtimeTimeScale
		? _runtimeTimeScale->load(std::memory_order_relaxed)
		: timeScale;

	json j = {};
	j["gamemode"] = state.lastArena ? GAMEMODE_STRS[(int)state.lastArena->gameMode] : "soccar";
	j["state"] = GameStateToJSON(state);
	j["context_label"] = contextLabel;
	j["game_speed"] = currentScale;
	j["paused"] = view.paused;
	if (view.spectateIndex >= 0)
		j["spectate_index"] = view.spectateIndex;

	std::vector<Action> actions = {};
	for (auto& player : state.players)
		actions.push_back(player.prevAction);

	j["actions"] = ActionSetToJSON(actions);

	if (debugSnapshot)
		j["debug"] = DebugSnapshotToJSON(*debugSnapshot);

	std::string jStr = j.dump();

	// Only the Python call needs the GIL; the JSON build above and the delay below are C++,
	// so keep the lock scoped tightly (in particular, off the sleep below).
	try {
		pybind11::gil_scoped_acquire gil;
		pyMod.attr("render_state")(jStr);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("RenderSender: Failed to send gamestate, exception: " << e.what());
	}

	// Delay
	{
		namespace chr = std::chrono;

		// Determine the desired delay and the actual delay (in seconds)
		double targetDelay = state.deltaTime / currentScale;
		double realDelay = renderTimer.Elapsed();
		renderTimer.Reset();

		constexpr double CORRECTION_SCALE = 0.3f; // Portion of the error we wil compensate for each step
		double error = targetDelay - realDelay;

		if (adaptiveRenderDelay == -1) {
			// Just initialize render delay as target delay
			adaptiveRenderDelay = targetDelay;
		} else {
			adaptiveRenderDelay += error * CORRECTION_SCALE;
		}
		adaptiveRenderDelay = RS_CLAMP(adaptiveRenderDelay, 0, targetDelay);

		// Sleep for the new adaptive delay
		int64_t sleepMics = (int64_t)(adaptiveRenderDelay * 1'000'000);
		std::this_thread::sleep_for(chr::microseconds(sleepMics));
	}
}

GGL::RenderSender::~RenderSender() {
	pybind11::gil_scoped_acquire gil;
	try {
		pyMod.attr("_stop_rsvis")();
	} catch (...) {}
	// Drop the Python reference while the GIL is held. Otherwise the pyMod member's own
	// destructor (running after this body) would decref with the GIL released and crash.
	pyMod = pybind11::module();
}
