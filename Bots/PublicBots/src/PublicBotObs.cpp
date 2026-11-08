#include "PublicBotObs.h"

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Gamestates/StateUtil.h>

#include <algorithm>
#include <cmath>

using namespace RLGC;
using namespace RocketSim;

namespace
{
	constexpr int ATTENTION_ENTITY_COUNT = 41;
	constexpr int PLAYER_SLOT_COUNT = 6;
	constexpr int ENTITY_WIDTH = 24;
	constexpr int QUERY_WIDTH = 32;
	constexpr int IS_SELF = 0;
	constexpr int IS_MATE = 1;
	constexpr int IS_OPPONENT = 2;
	constexpr int IS_BALL = 3;
	constexpr int IS_BOOST = 4;
	constexpr int POS = 5;
	constexpr int LIN_VEL = 8;
	constexpr int FORWARD = 11;
	constexpr int UP = 14;
	constexpr int ANG_VEL = 17;
	constexpr int BOOST = 20;
	constexpr int DEMO = 21;
	constexpr int ON_GROUND = 22;
	constexpr int HAS_FLIP = 23;
	constexpr float PI = 3.14159265358979323846f;

	using Entity = std::array<float, ENTITY_WIDTH>;
	using PlayerList = std::array<const Player*, PLAYER_SLOT_COUNT>;

	void SetVec(Entity& entity, int offset, const Vec& value)
	{
		entity[offset] = value.x;
		entity[offset + 1] = value.y;
		entity[offset + 2] = value.z;
	}

	int BuildOrderedPlayers(
		PlayerList& result,
		const Player& mainPlayer,
		const GameState& state)
	{
		int count = 0;
		result[count++] = &mainPlayer;
		for (const Player& other : state.players)
			if (count < PLAYER_SLOT_COUNT
				&& other.carId != mainPlayer.carId
				&& other.team == mainPlayer.team) {
				result[count++] = &other;
			}
		for (const Player& other : state.players)
			if (count < PLAYER_SLOT_COUNT && other.team != mainPlayer.team)
				result[count++] = &other;
		return count;
	}

	void NormalizeAttentionEntity(Entity& entity)
	{
		for (int i = POS; i < FORWARD; i++)
			entity[i] /= 2300.f;
		for (int i = ANG_VEL; i < BOOST; i++)
			entity[i] /= 5.5f;
	}

	void InvertAttentionEntity(Entity& entity)
	{
		for (int offset : { POS, LIN_VEL, FORWARD, UP, ANG_VEL }) {
			entity[offset] *= -1;
			entity[offset + 1] *= -1;
		}
		std::swap(entity[IS_MATE], entity[IS_OPPONENT]);
	}

	void AppendPacked(
		FList& out,
		const std::array<float, QUERY_WIDTH>& query,
		const std::array<Entity, ATTENTION_ENTITY_COUNT>& entities,
		const std::array<float, ATTENTION_ENTITY_COUNT>& mask)
	{
		out.reserve(QUERY_WIDTH + ATTENTION_ENTITY_COUNT * ENTITY_WIDTH + ATTENTION_ENTITY_COUNT);
		out.insert(out.end(), query.begin(), query.end());
		for (const Entity& entity : entities)
			out.insert(out.end(), entity.begin(), entity.end());
		out.insert(out.end(), mask.begin(), mask.end());
	}

	float DistSqToBall(const Player* player, const GameState& state)
	{
		return player->pos.DistSq(state.ball.pos);
	}

	void StableSortByBallDistance(
		PlayerList& players,
		int count,
		const GameState& state)
	{
		for (int i = 1; i < count; i++) {
			const Player* value = players[i];
			const float valueDistance = DistSqToBall(value, state);
			int destination = i;
			while (destination > 0
				&& DistSqToBall(players[destination - 1], state) > valueDistance) {
				players[destination] = players[destination - 1];
				destination--;
			}
			players[destination] = value;
		}
	}
}

void NectoObs::Reset(const GameState&)
{
	demoTimers.clear();
	boostTimers.fill(0);
	observedBoostTimers.fill(0);
	timerTick = UINT64_MAX;
}

void NectoObs::UpdateTimers(const GameState& state)
{
	if (timerTick == state.lastTickCount)
		return;
	timerTick = state.lastTickCount;

	for (const Player& player : state.players) {
		float& timer = demoTimers[player.carId];
		if (timer <= 0)
			timer = 3;
		else
			timer = std::max(timer - 8.f / 120.f, 0.f);
	}

	for (int i = 0; i < 34; i++) {
		const bool active = i < (int)state.boostPads.size() && state.boostPads[i];
		if (active && boostTimers[i] == 0)
			boostTimers[i] = 0.4f + 0.6f * (CommonValues::BOOST_LOCATIONS[i].z > 72);
		if (!active)
			boostTimers[i] = 0;
		observedBoostTimers[i] = boostTimers[i];
		boostTimers[i] = std::max(boostTimers[i] - 8.f / 1200.f, 0.f);
	}
}

// https://github.com/RLBot/RLBotPack/blob/0033a98d4e060670f334f60f7f8dc5ee21ba93bb/RLBotPack/Necto/Necto/necto_obs.py
FList NectoObs::BuildObs(const Player& player, const GameState& state)
{
	UpdateTimers(state);

	std::array<Entity, ATTENTION_ENTITY_COUNT> entities = {};
	std::array<float, ATTENTION_ENTITY_COUNT> mask = {};

	Entity& ball = entities[0];
	ball[IS_BALL] = 1;
	SetVec(ball, POS, state.ball.pos);
	SetVec(ball, LIN_VEL, state.ball.vel);
	SetVec(ball, ANG_VEL, state.ball.angVel);

	PlayerList orderedPlayers = {};
	const int playerCount = BuildOrderedPlayers(orderedPlayers, player, state);
	for (int i = 0; i < playerCount; i++) {
		const Player& source = *orderedPlayers[i];
		Entity& entity = entities[1 + i];
		entity[source.team == Team::BLUE ? IS_MATE : IS_OPPONENT] = 1;
		SetVec(entity, POS, source.pos);
		SetVec(entity, LIN_VEL, source.vel);
		SetVec(entity, FORWARD, source.rotMat.forward);
		SetVec(entity, UP, source.rotMat.up);
		SetVec(entity, ANG_VEL, source.angVel);
		entity[BOOST] = source.boost / 100.f;
		entity[DEMO] = demoTimers[source.carId] / 10.f;
		entity[ON_GROUND] = source.isOnGround;
		entity[HAS_FLIP] = !source.hasDoubleJumped;
	}
	for (int i = 1 + playerCount; i <= PLAYER_SLOT_COUNT; i++)
		mask[i] = 1;

	for (int i = 0; i < 34; i++) {
		Entity& entity = entities[1 + PLAYER_SLOT_COUNT + i];
		entity[IS_BOOST] = 1;
		SetVec(entity, POS, CommonValues::BOOST_LOCATIONS[i]);
		entity[BOOST] = 0.12f + 0.88f * (CommonValues::BOOST_LOCATIONS[i].z > 72);
		entity[DEMO] = observedBoostTimers[i];
	}

	for (Entity& entity : entities) {
		if (player.team == Team::ORANGE)
			InvertAttentionEntity(entity);
		NormalizeAttentionEntity(entity);
	}

	entities[1][IS_SELF] = 1;
	std::array<float, QUERY_WIDTH> query = {};
	std::copy(entities[1].begin(), entities[1].end(), query.begin());
	for (int i = 0; i < Action::ELEM_AMOUNT; i++)
		query[ENTITY_WIDTH + i] = player.prevAction[i];

	for (Entity& entity : entities) {
		for (int i = POS; i < FORWARD; i++)
			entity[i] -= query[i];
	}

	FList result;
	AppendPacked(result, query, entities, mask);
	return result;
}

// https://github.com/RLBot/RLBotPack/blob/0033a98d4e060670f334f60f7f8dc5ee21ba93bb/RLBotPack/Necto/Nexto/nexto_obs.py
FList NextoObs::BuildObs(const Player& player, const GameState& state)
{
	std::array<Entity, ATTENTION_ENTITY_COUNT> entities = {};
	std::array<float, ATTENTION_ENTITY_COUNT> mask = {};
	PlayerList orderedPlayers = {};
	const int playerCount = BuildOrderedPlayers(orderedPlayers, player, state);

	for (int i = 0; i < playerCount; i++) {
		const Player& source = *orderedPlayers[i];
		Entity& entity = entities[i];
		entity[source.team == Team::BLUE ? IS_MATE : IS_OPPONENT] = 1;
		if (i == 0)
			entity[IS_SELF] = 1;
		SetVec(entity, POS, source.pos);
		SetVec(entity, LIN_VEL, source.vel);
		SetVec(entity, FORWARD, source.rotMat.forward);
		SetVec(entity, UP, source.rotMat.up);
		SetVec(entity, ANG_VEL, source.angVel);
		entity[BOOST] = source.boost / 100.f;
		entity[DEMO] = source.isDemoed;
		entity[ON_GROUND] = source.isOnGround;
		entity[HAS_FLIP] = !source.hasDoubleJumped;
	}
	for (int i = playerCount; i < PLAYER_SLOT_COUNT; i++)
		mask[i] = 1;

	Entity& ball = entities[PLAYER_SLOT_COUNT];
	ball[IS_BALL] = 1;
	SetVec(ball, POS, state.ball.pos);
	SetVec(ball, LIN_VEL, state.ball.vel);
	SetVec(ball, ANG_VEL, state.ball.angVel);

	for (int i = 0; i < 34; i++) {
		Entity& entity = entities[PLAYER_SLOT_COUNT + 1 + i];
		entity[IS_BOOST] = 1;
		SetVec(entity, POS, CommonValues::BOOST_LOCATIONS[i]);
		entity[BOOST] = 0.12f + 0.88f * (CommonValues::BOOST_LOCATIONS[i].z > 72);
		entity[DEMO] = i < (int)state.boostPads.size() && state.boostPads[i];
	}

	for (Entity& entity : entities) {
		if (player.team == Team::ORANGE)
			InvertAttentionEntity(entity);
		NormalizeAttentionEntity(entity);
	}

	std::array<float, QUERY_WIDTH> query = {};
	std::copy(entities[0].begin(), entities[0].end(), query.begin());
	for (int i = 0; i < Action::ELEM_AMOUNT; i++)
		query[ENTITY_WIDTH + i] = player.prevAction[i];

	const float theta = std::atan2(query[FORWARD], query[FORWARD + 1]);
	const float cosine = std::cos(theta);
	const float sine = std::sin(theta);
	for (Entity& entity : entities) {
		for (int i = POS; i < LIN_VEL; i++)
			entity[i] -= query[i];
		for (int offset : { POS, LIN_VEL, FORWARD, UP, ANG_VEL }) {
			const float x = entity[offset];
			const float y = entity[offset + 1];
			entity[offset] = cosine * x - sine * y;
			entity[offset + 1] = sine * x + cosine * y;
		}
	}

	FList result;
	AppendPacked(result, query, entities, mask);
	return result;
}

// https://github.com/RLBot/RLBotPack/blob/0033a98d4e060670f334f60f7f8dc5ee21ba93bb/RLBotPack/Element/obs.py
FList ElementObs::BuildObs(const Player& player, const GameState& state)
{
	const bool inverted = player.team == Team::ORANGE;
	const PhysState ball = InvertPhys(state.ball, inverted);
	const auto& pads = state.GetBoostPads(inverted);

	FList result;
	result.reserve(107);
	result += ball.pos / 2300.f;
	result += ball.vel / 2300.f;
	result += ball.angVel / PI;
	for (int i = 0; i < Action::ELEM_AMOUNT; i++)
		result += player.prevAction[i];
	for (int i = 0; i < 34; i++)
		result += (float)pads[i];

	const PhysState playerCar = InvertPhys(player, inverted);
	AddPlayer(result, player, ball, inverted, nullptr);

	PlayerList teammates = {};
	PlayerList opponents = {};
	int teammateCount = 0;
	int opponentCount = 0;
	for (const Player& candidate : state.players) {
		if (candidate.team == player.team) {
			if (teammateCount < PLAYER_SLOT_COUNT)
				teammates[teammateCount++] = &candidate;
		} else if (opponentCount < PLAYER_SLOT_COUNT) {
			opponents[opponentCount++] = &candidate;
		}
	}
	StableSortByBallDistance(teammates, teammateCount, state);
	StableSortByBallDistance(opponents, opponentCount, state);

	int playerRank = 0;
	for (int i = 0; i < teammateCount; i++)
		if (teammates[i]->carId == player.carId)
			playerRank = i;
	if (opponentCount > 0) {
		const Player& opponent = *opponents[std::min(playerRank, opponentCount - 1)];
		AddPlayer(result, opponent, ball, inverted, &playerCar);
	}

	return result;
}

void ElementObs::AddPlayer(
	FList& out,
	const Player& player,
	const PhysState& ball,
	bool inverted,
	const PhysState* relativeTo)
{
	const PhysState car = InvertPhys(player, inverted);
	out += (ball.pos - car.pos) / 2300.f;
	out += (ball.vel - car.vel) / 2300.f;
	out += car.pos / 2300.f;
	out += car.rotMat.forward;
	out += car.rotMat.up;
	out += car.vel / 2300.f;
	out += car.angVel / PI;
	out += player.boost / 100.f;
	out += player.isOnGround;
	out += !player.hasDoubleJumped;
	out += player.isDemoed;

	if (relativeTo) {
		out += (car.pos - relativeTo->pos) / 2300.f;
		out += (car.vel - relativeTo->vel) / 2300.f;
	}
}
