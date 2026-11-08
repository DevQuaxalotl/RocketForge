#include <RocketForge/ParityProbe/Protocol.h>

#include <RocketForge/ParityProbe/Hash.h>

#include <RLGymCPP/CommonValues.h>

#include <algorithm>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace RocketForge::ParityProbe
{
	namespace
	{
		using nlohmann::json;

		const json& RequireField(const json& object, const char* name)
		{
			if (!object.contains(name))
				throw std::runtime_error(std::string("parity fixture is missing required field: ") + name);
			return object.at(name);
		}

		Vec ReadVec(const json& value, const char* fieldName)
		{
			if (!value.is_array() || value.size() != 3)
				throw std::runtime_error(std::string("parity fixture field must be a three-value array: ") + fieldName);
			return { value.at(0).get<float>(), value.at(1).get<float>(), value.at(2).get<float>() };
		}

		RotMat ReadRotMat(const json& value)
		{
			if (!value.is_array() || value.size() != 3)
				throw std::runtime_error("parity fixture rot_mat must contain forward, right, and up columns");
			return {
				ReadVec(value.at(0), "rot_mat.forward"),
				ReadVec(value.at(1), "rot_mat.right"),
				ReadVec(value.at(2), "rot_mat.up")
			};
		}

		RLGC::Action ReadAction(const json& value, const char* fieldName)
		{
			if (!value.is_array() || value.size() != RLGC::Action::ELEM_AMOUNT)
				throw std::runtime_error(std::string("parity fixture action must contain eight values: ") + fieldName);
			return {
				value.at(0).get<float>(),
				value.at(1).get<float>(),
				value.at(2).get<float>(),
				value.at(3).get<float>(),
				value.at(4).get<float>(),
				value.at(5).get<float>(),
				value.at(6).get<float>(),
				value.at(7).get<float>()
			};
		}

		CarControls ReadControls(const json& value, const char* fieldName)
		{
			return static_cast<CarControls>(ReadAction(value, fieldName));
		}

		json ActionToJson(const RLGC::Action& action)
		{
			return json::array({
				action.throttle,
				action.steer,
				action.pitch,
				action.yaw,
				action.roll,
				action.jump,
				action.boost,
				action.handbrake
			});
		}

		void ReadPhysState(const json& source, PhysState& state)
		{
			state.pos = ReadVec(RequireField(source, "pos"), "pos");
			state.rotMat = ReadRotMat(RequireField(source, "rot_mat"));
			state.vel = ReadVec(RequireField(source, "vel"), "vel");
			state.angVel = ReadVec(RequireField(source, "ang_vel"), "ang_vel");
		}

		BallState ReadBallState(const json& source)
		{
			BallState ball = {};
			ReadPhysState(source, ball);
			ball.tickCountSinceUpdate = source.value("tick_count_since_update", uint64_t{ 0 });

			if (source.contains("heatseeker"))
			{
				const json& heatseeker = source.at("heatseeker");
				ball.hsInfo.yTargetDir = heatseeker.value("y_target_dir", 0.f);
				ball.hsInfo.curTargetSpeed = heatseeker.value("current_target_speed", ball.hsInfo.curTargetSpeed);
				ball.hsInfo.timeSinceHit = heatseeker.value("time_since_hit", 0.f);
			}

			if (source.contains("dropshot"))
			{
				const json& dropshot = source.at("dropshot");
				ball.dsInfo.chargeLevel = dropshot.value("charge_level", 1);
				ball.dsInfo.accumulatedHitForce = dropshot.value("accumulated_hit_force", 0.f);
				ball.dsInfo.yTargetDir = dropshot.value("y_target_dir", 0.f);
				ball.dsInfo.hasDamaged = dropshot.value("has_damaged", false);
				ball.dsInfo.lastDamageTick = dropshot.value("last_damage_tick", uint64_t{ 0 });
			}
			return ball;
		}

		RocketSim::Team ReadTeam(const json& value)
		{
			if (value.is_number_integer())
			{
				const int team = value.get<int>();
				if (team == 0)
					return RocketSim::Team::BLUE;
				if (team == 1)
					return RocketSim::Team::ORANGE;
			}
			else if (value.is_string())
			{
				const std::string team = value.get<std::string>();
				if (team == "blue")
					return RocketSim::Team::BLUE;
				if (team == "orange")
					return RocketSim::Team::ORANGE;
			}
			throw std::runtime_error("parity fixture team must be blue, orange, 0, or 1");
		}

		RocketSim::GameMode ReadGameMode(const json& value)
		{
			if (value.is_number_integer())
			{
				const int mode = value.get<int>();
				if (mode >= 0 && mode <= static_cast<int>(RocketSim::GameMode::THE_VOID))
					return static_cast<RocketSim::GameMode>(mode);
			}
			else if (value.is_string())
			{
				const std::string mode = value.get<std::string>();
				if (mode == "soccar")
					return RocketSim::GameMode::SOCCAR;
				if (mode == "hoops")
					return RocketSim::GameMode::HOOPS;
				if (mode == "heatseeker")
					return RocketSim::GameMode::HEATSEEKER;
				if (mode == "snowday")
					return RocketSim::GameMode::SNOWDAY;
				if (mode == "dropshot")
					return RocketSim::GameMode::DROPSHOT;
				if (mode == "void")
					return RocketSim::GameMode::THE_VOID;
			}
			throw std::runtime_error("parity fixture contains an unknown game_mode");
		}

		void ReadBallHitInfo(const json& source, BallHitInfo& info)
		{
			info.isValid = source.value("is_valid", false);
			info.relativePosOnBall = ReadVec(source.value("relative_pos_on_ball", json::array({ 0, 0, 0 })), "ball_hit_info.relative_pos_on_ball");
			info.ballPos = ReadVec(source.value("ball_pos", json::array({ 0, 0, 0 })), "ball_hit_info.ball_pos");
			info.extraHitVel = ReadVec(source.value("extra_hit_vel", json::array({ 0, 0, 0 })), "ball_hit_info.extra_hit_vel");
			info.tickCountWhenHit = source.value("tick_count_when_hit", ~uint64_t{ 0 });
			info.tickCountWhenExtraImpulseApplied = source.value("tick_count_when_extra_impulse_applied", ~uint64_t{ 0 });
		}

		RLGC::Player ReadPlayer(const json& source, int defaultIndex)
		{
			RLGC::Player player = {};
			ReadPhysState(source, player);
			player.index = source.value("index", defaultIndex);
			player.carId = RequireField(source, "car_id").get<uint32_t>();
			player.team = ReadTeam(RequireField(source, "team"));
			player.tickCountSinceUpdate = source.value("tick_count_since_update", uint64_t{ 0 });
			player.isOnGround = source.value("is_on_ground", true);
			player.wasOnGroundDuringStep = source.value("was_on_ground_during_step", player.isOnGround);
			player.groundContactGraceUsed = source.value("ground_contact_grace_used", false);
			player.obsOnGround = source.value("obs_on_ground", player.isOnGround);
			if (source.contains("wheels_with_contact"))
			{
				const json& wheels = source.at("wheels_with_contact");
				if (!wheels.is_array() || wheels.size() != 4)
					throw std::runtime_error("parity fixture wheels_with_contact must contain four values");
				for (size_t i = 0; i < 4; ++i)
					player.wheelsWithContact[i] = wheels.at(i).get<bool>();
			}
			player.hasJumped = source.value("has_jumped", false);
			player.hasDoubleJumped = source.value("has_double_jumped", false);
			player.hasFlipped = source.value("has_flipped", false);
			player.flipRelTorque = ReadVec(source.value("flip_rel_torque", json::array({ 0, 0, 0 })), "flip_rel_torque");
			player.jumpTime = source.value("jump_time", 0.f);
			player.flipTime = source.value("flip_time", 0.f);
			player.isFlipping = source.value("is_flipping", false);
			player.isJumping = source.value("is_jumping", false);
			player.airTime = source.value("air_time", 0.f);
			player.airTimeSinceJump = source.value("air_time_since_jump", 0.f);
			player.boost = source.value("boost", 0.f);
			player.timeSinceBoosted = source.value("time_since_boosted", 0.f);
			player.isBoosting = source.value("is_boosting", false);
			player.boostingTime = source.value("boosting_time", 0.f);
			player.isSupersonic = source.value("is_supersonic", false);
			player.supersonicTime = source.value("supersonic_time", 0.f);
			player.handbrakeVal = source.value("handbrake_value", 0.f);
			player.isAutoFlipping = source.value("is_auto_flipping", false);
			player.autoFlipTimer = source.value("auto_flip_timer", 0.f);
			player.autoFlipTorqueScale = source.value("auto_flip_torque_scale", 0.f);
			player.isDemoed = source.value("is_demoed", false);
			player.demoRespawnTimer = source.value("demo_respawn_timer", 0.f);
			player.ballTouchedStep = source.value("ball_touched_step", false);
			player.ballTouchedTick = source.value("ball_touched_tick", false);
			player.prevAction = source.contains("prev_action") ? ReadAction(source.at("prev_action"), "prev_action") : RLGC::Action{};
			player.airState = source.value("air_state", uint8_t{ 0 });
			player.hasDodged = source.value("has_dodged", false);
			player.dodgeTimeout = source.value("dodge_timeout", 0.f);
			player.demolishedTimeout = source.value("demolished_timeout", 0.f);
			player.lastControls = source.contains("last_controls") ? ReadControls(source.at("last_controls"), "last_controls") : CarControls{};
			player.jumpResetBlockTicks = source.value("jump_reset_block_ticks", 0);

			if (source.contains("world_contact"))
			{
				const json& contact = source.at("world_contact");
				player.worldContact.hasContact = contact.value("has_contact", false);
				player.worldContact.contactNormal = ReadVec(contact.value("normal", json::array({ 0, 0, 0 })), "world_contact.normal");
			}
			if (source.contains("car_contact"))
			{
				const json& contact = source.at("car_contact");
				player.carContact.otherCarID = contact.value("other_car_id", uint32_t{ 0 });
				player.carContact.cooldownTimer = contact.value("cooldown_timer", 0.f);
			}
			if (source.contains("ball_hit_info"))
				ReadBallHitInfo(source.at("ball_hit_info"), player.ballHitInfo);
			if (source.contains("event_state"))
			{
				const json& event = source.at("event_state");
				player.eventState.goal = event.value("goal", false);
				player.eventState.save = event.value("save", false);
				player.eventState.assist = event.value("assist", false);
				player.eventState.shot = event.value("shot", false);
				player.eventState.shotPass = event.value("shot_pass", false);
				player.eventState.bump = event.value("bump", false);
				player.eventState.bumped = event.value("bumped", false);
				player.eventState.demo = event.value("demo", false);
				player.eventState.demoed = event.value("demoed", false);
			}
			return player;
		}

		RLGC::GameState ReadGameState(const json& source)
		{
			RLGC::GameState state;
			state.gameMode = ReadGameMode(RequireField(source, "game_mode"));
			state.deltaTime = source.value("delta_time", 0.f);
			state.goalScored = source.value("goal_scored", false);
			state.lastTouchCarID = source.value("last_touch_car_id", -1);
			state.lastTickCount = source.value("last_tick_count", uint64_t{ 0 });
			state.ball = ReadBallState(RequireField(source, "ball"));
			state.lastArena = nullptr;

			state.ballPredData.clear();
			for (const json& prediction : source.value("ball_prediction", json::array()))
				state.ballPredData.push_back(ReadBallState(prediction));

			const json& boostPads = RequireField(source, "boost_pads");
			const json& boostTimers = RequireField(source, "boost_pad_timers");
			if (!boostPads.is_array() || !boostTimers.is_array() || boostPads.size() != boostTimers.size())
				throw std::runtime_error("parity fixture boost arrays must have equal lengths");
			const size_t expectedBoostCount = state.gameMode == RocketSim::GameMode::HOOPS
				? RLGC::CommonValues::BOOST_LOCATIONS_AMOUNT_HOOPS
				: RLGC::CommonValues::BOOST_LOCATIONS_AMOUNT;
			if (boostPads.size() != expectedBoostCount)
				throw std::runtime_error("parity fixture boost arrays do not match the game mode");
			state.boostPads = boostPads.get<std::vector<bool>>();
			state.boostPadTimers = boostTimers.get<std::vector<float>>();
			state.boostPadsInv.assign(state.boostPads.rbegin(), state.boostPads.rend());
			state.boostPadTimersInv.assign(state.boostPadTimers.rbegin(), state.boostPadTimers.rend());
			state.boostPadIndexMap.resize(expectedBoostCount);
			std::iota(state.boostPadIndexMap.begin(), state.boostPadIndexMap.end(), 0);

			const json& players = RequireField(source, "players");
			if (!players.is_array() || players.empty())
				throw std::runtime_error("parity fixture state must contain at least one player");
			state.players.reserve(players.size());
			for (size_t index = 0; index < players.size(); ++index)
				state.players.push_back(ReadPlayer(players.at(index), static_cast<int>(index)));
			return state;
		}

		json FloatVectorToJson(const std::vector<float>& values)
		{
			return {
				{ "size", values.size() },
				{ "float32_le_sha256", Sha256Float32LE(values) },
				{ "values", values }
			};
		}
	}

	FixtureDocument LoadFixtureDocument(const std::filesystem::path& path)
	{
		std::ifstream input(path);
		if (!input.good())
			throw std::runtime_error("failed to open parity fixture: " + path.string());
		json root = json::parse(input, nullptr, false);
		if (root.is_discarded() || !root.is_object())
			throw std::runtime_error("parity fixture is not valid JSON");
		if (root.value("schema", std::string{}) != FIXTURE_SCHEMA)
			throw std::runtime_error("unsupported parity fixture schema");

		FixtureDocument document;
		document.fixtureId = RequireField(root, "fixture_id");
		const json& sequences = RequireField(root, "sequences");
		if (!sequences.is_array() || sequences.empty())
			throw std::runtime_error("parity fixture must contain at least one sequence");

		for (const json& sequenceJson : sequences)
		{
			SequenceFixture sequence;
			sequence.sequenceId = RequireField(sequenceJson, "sequence_id");
			const json& frames = RequireField(sequenceJson, "frames");
			if (!frames.is_array() || frames.empty())
				throw std::runtime_error("parity fixture sequence must contain at least one frame");

			for (size_t frameIndex = 0; frameIndex < frames.size(); ++frameIndex)
			{
				const json& frameJson = frames.at(frameIndex);
				FrameFixture frame;
				frame.frameId = RequireField(frameJson, "frame_id");
				frame.targetCarId = RequireField(frameJson, "target_car_id").get<uint32_t>();
				frame.obsRngSeed = RequireField(frameJson, "obs_rng_seed").get<uint64_t>();
				frame.resetObsBuilder = frameJson.value("reset_obs_builder", frameIndex == 0);
				frame.source = frameJson;
				frame.state = ReadGameState(RequireField(frameJson, "state"));
				if (frameJson.contains("runtime") && frameJson.at("runtime").contains("applied_action"))
					frame.runtime.appliedAction = ReadAction(frameJson.at("runtime").at("applied_action"), "runtime.applied_action");

				const size_t targetCount = std::count_if(
					frame.state.players.begin(),
					frame.state.players.end(),
					[&](const RLGC::Player& player) { return player.carId == frame.targetCarId; });
				if (targetCount != 1)
					throw std::runtime_error("parity fixture target_car_id must identify exactly one player");
				sequence.frames.push_back(std::move(frame));
			}
			document.sequences.push_back(std::move(sequence));
		}
		return document;
	}

	nlohmann::json FrameResultToJson(const FrameResult& result, const std::string& producer)
	{
		json actionPlan = json::array();
		for (const ActionPlanEntry& entry : result.actionPlan)
			actionPlan.push_back({ { "action", ActionToJson(entry.action) }, { "ticks", entry.ticks } });

		return {
			{ "schema", RESULT_SCHEMA },
			{ "producer", producer },
			{ "fixture_id", result.fixtureId },
			{ "sequence_id", result.sequenceId },
			{ "frame_id", result.frameId },
			{ "target_car_id", result.targetCarId },
			{ "obs_rng_seed", result.obsRngSeed },
			{ "observation", FloatVectorToJson(result.observation) },
			{ "action_mask", {
				{ "size", result.actionMask.size() },
				{ "bits", result.actionMask },
				{ "sha256", Sha256Hex(result.actionMask) }
			} },
			{ "network", {
				{ "shared_features", FloatVectorToJson(result.sharedFeatures) },
				{ "raw_logits", FloatVectorToJson(result.rawLogits) },
				{ "masked_logits", FloatVectorToJson(result.maskedLogits) },
				{ "argmax_action_id", result.argmaxActionId },
				{ "top1_margin", result.top1Margin }
			} },
			{ "parsed_action", ActionToJson(result.parsedAction) },
			{ "action_plan", std::move(actionPlan) },
			{ "runtime", {
				{ "selected_action_id", result.argmaxActionId },
				{ "tick_skip", result.tickSkip },
				{ "action_delay", result.actionDelay },
				{ "action_ticks", result.actionTicks },
				{ "queued_action", ActionToJson(result.queuedAction) },
				{ "applied_action", ActionToJson(result.appliedAction) }
			} }
		};
	}
}
