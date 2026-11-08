#pragma once

#include <RLGymCPP/ActionParsers/ActionParser.h>
#include <RLGymCPP/ObsBuilders/ObsBuilder.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class Agent;

using ObsBuilderFactory = std::function<std::unique_ptr<RLGC::ObsBuilder>(const Agent&)>;
using ActionParserFactory = std::function<std::unique_ptr<RLGC::ActionParser>(const Agent&)>;

/**
 * Registers an observation builder factory by the setup name saved with trained models.
 * Factories receive the selected Agent so custom obs builders can use saved settings.
 **/
#define REGISTER_OBS_BUILDER(NAME, FACTORY)                                              \
namespace {                                                                               \
	[[maybe_unused]] const bool GGL_COMPONENT_CONCAT(g_obsBuilderRegistered_, __LINE__) = \
		[]() {                                                                            \
			ObsBuilderRegistry::Register(NAME, FACTORY);                                  \
			return true;                                                                  \
		}();                                                                              \
}

/**
 * Registers an action parser factory by the setup name saved with trained models.
 * Factories receive the selected Agent so custom parsers can use saved settings.
 **/
#define REGISTER_ACTION_PARSER(NAME, FACTORY)                                                \
namespace {                                                                                   \
	[[maybe_unused]] const bool GGL_COMPONENT_CONCAT(g_actionParserRegistered_, __LINE__) =   \
		[]() {                                                                                \
			ActionParserRegistry::Register(NAME, FACTORY);                                    \
			return true;                                                                      \
		}();                                                                                  \
}

#define GGL_COMPONENT_CONCAT_INNER(A, B) A##B
#define GGL_COMPONENT_CONCAT(A, B) GGL_COMPONENT_CONCAT_INNER(A, B)

/** Shared lookup table for named observation builders used by saved agent setup metadata. **/
class ObsBuilderRegistry
{
public:
	static void Register(const std::string& name, ObsBuilderFactory factory)
	{
		GetMap()[name] = std::move(factory);
	}

	static std::unique_ptr<RLGC::ObsBuilder> Create(const std::string& name, const Agent& agent)
	{
		auto& map = GetMap();
		auto it = map.find(name);
		if (it == map.end())
			throw std::runtime_error("Unknown obs builder: " + name);

		return it->second(agent);
	}

	static std::vector<std::string> GetAvailableObsBuilders()
	{
		std::vector<std::string> result;
		result.reserve(GetMap().size());

		for (const auto& [name, _] : GetMap())
			result.push_back(name);

		std::sort(result.begin(), result.end());
		return result;
	}

private:
	static std::unordered_map<std::string, ObsBuilderFactory>& GetMap()
	{
		static std::unordered_map<std::string, ObsBuilderFactory> map;
		return map;
	}
};

/** Shared lookup table for named action parsers used by saved agent setup metadata. **/
class ActionParserRegistry
{
public:
	static void Register(const std::string& name, ActionParserFactory factory)
	{
		GetMap()[name] = std::move(factory);
	}

	static std::unique_ptr<RLGC::ActionParser> Create(const std::string& name, const Agent& agent)
	{
		auto& map = GetMap();
		auto it = map.find(name);
		if (it == map.end())
			throw std::runtime_error("Unknown action parser: " + name);

		return it->second(agent);
	}

	static std::vector<std::string> GetAvailableActionParsers()
	{
		std::vector<std::string> result;
		result.reserve(GetMap().size());

		for (const auto& [name, _] : GetMap())
			result.push_back(name);

		std::sort(result.begin(), result.end());
		return result;
	}

private:
	static std::unordered_map<std::string, ActionParserFactory>& GetMap()
	{
		static std::unordered_map<std::string, ActionParserFactory> map;
		return map;
	}
};
