#include <gtest/gtest.h>

#include "TestSupport.h"

#include <GigaLearnInfer/Agent.h>
#include <GigaLearnInfer/AgentComponentRegistry.h>
#include <GigaLearnInfer/AgentRegistry.h>
#include <RLGymCPP/ActionParsers/ActionParser.h>
#include <RLGymCPP/ObsBuilders/ObsBuilder.h>

#include <algorithm>
#include <memory>
#include <string>

namespace
{
	struct RegistryTestObsBuilder : RLGC::ObsBuilder
	{
		explicit RegistryTestObsBuilder(int maxTeamSize) : maxTeamSize(maxTeamSize) {}

		void Reset(const RLGC::GameState& initialState) override
		{
			resetCalled = true;
		}

		RLGC::FList BuildObs(const RLGC::Player& player, const RLGC::GameState& state) override
		{
			if (!resetCalled)
				throw std::runtime_error("test observation builder was not reset");
			return { static_cast<float>(maxTeamSize) };
		}

		int maxTeamSize;
		bool resetCalled = false;
	};

	struct RegistryTestActionParser : RLGC::ActionParser
	{
		explicit RegistryTestActionParser(int tickSkip) : tickSkip(tickSkip) {}

		RLGC::Action ParseAction(int actionIdx, const RLGC::Player& player, const RLGC::GameState& state) override
		{
			return {};
		}

		int GetActionAmount() override
		{
			return tickSkip;
		}

		int tickSkip;
	};

	void AssertObsBuilderResolves(Agent& agent, const std::string& obsBuilderName)
	{
		agent.obsBuilderName = obsBuilderName;
		std::unique_ptr<RLGC::ObsBuilder> obsBuilder(agent.MakeObsBuilder());
		ASSERT_TRUE(obsBuilder != nullptr);
	}

	void AssertActionParserResolves(Agent& agent, const std::string& actionParserName)
	{
		agent.actionParserName = actionParserName;
		std::unique_ptr<RLGC::ActionParser> actionParser(agent.MakeActionParser());
		ASSERT_TRUE(actionParser != nullptr);
	}
}

REGISTER_OBS_BUILDER("RegistryTestObs", [](const Agent& agent) {
	return std::make_unique<RegistryTestObsBuilder>(agent.maxTeamSize);
})

REGISTER_ACTION_PARSER("RegistryTestAction", [](const Agent& agent) {
	return std::make_unique<RegistryTestActionParser>(agent.tickSkip);
})

TEST(AgentComponentRegistry, CustomComponentsResolveFromSavedNames)
{
	Agent agent;
	agent.obsBuilderName = "RegistryTestObs";
	agent.actionParserName = "RegistryTestAction";
	agent.maxTeamSize = 3;
	agent.tickSkip = 6;

	std::unique_ptr<RLGC::ObsBuilder> obsBuilder(agent.MakeObsBuilder());
	auto* registryObs = dynamic_cast<RegistryTestObsBuilder*>(obsBuilder.get());
	ASSERT_TRUE(registryObs != nullptr);
	ASSERT_EQ(registryObs->maxTeamSize, 3);

	std::unique_ptr<RLGC::ActionParser> actionParser(agent.MakeActionParser());
	auto* registryAction = dynamic_cast<RegistryTestActionParser*>(actionParser.get());
	ASSERT_TRUE(registryAction != nullptr);
	ASSERT_EQ(registryAction->tickSkip, 6);
}

TEST(AgentComponentRegistry, DefaultNamesStillResolve)
{
	Agent agent;
	agent.obsBuilderName.clear();
	agent.actionParserName.clear();

	std::unique_ptr<RLGC::ObsBuilder> obsBuilder(agent.MakeObsBuilder());
	std::unique_ptr<RLGC::ActionParser> actionParser(agent.MakeActionParser());
	ASSERT_TRUE(obsBuilder != nullptr);
	ASSERT_TRUE(actionParser != nullptr);

	const std::vector<std::string> obsNames = ObsBuilderRegistry::GetAvailableObsBuilders();
	ASSERT_TRUE(std::find(obsNames.begin(), obsNames.end(), "DefaultObsPadded") != obsNames.end());
	ASSERT_TRUE(std::find(obsNames.begin(), obsNames.end(), "DefaultObs") != obsNames.end());
	ASSERT_TRUE(std::find(obsNames.begin(), obsNames.end(), "AdvancedObs") != obsNames.end());
	ASSERT_TRUE(std::find(obsNames.begin(), obsNames.end(), "AdvancedObsPadded") != obsNames.end());

	const std::vector<std::string> actionNames = ActionParserRegistry::GetAvailableActionParsers();
	ASSERT_TRUE(std::find(actionNames.begin(), actionNames.end(), "DefaultAction") != actionNames.end());
}

TEST(AgentComponentRegistry, GetObsSizeResetsBuilderBeforeFirstObservation)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	Agent agent;
	agent.obsBuilderName = "RegistryTestObs";
	agent.maxTeamSize = 3;

	ASSERT_EQ(agent.GetObsSize(), 1);
}

TEST(AgentComponentRegistry, LinkedExampleBotComponentNamesResolve)
{
	std::unique_ptr<Agent> example = AgentRegistry::Create("example-agent");
	ASSERT_TRUE(example != nullptr);
	ASSERT_EQ(example->agentName, std::string("example-agent"));
	ASSERT_EQ(example->obsBuilderName, std::string("AdvancedObsPadded"));
	ASSERT_EQ(example->actionParserName, std::string("DefaultAction"));

	AssertObsBuilderResolves(*example, "AdvancedObsPadded");
	AssertActionParserResolves(*example, "DefaultAction");
}

TEST(AgentComponentRegistry, UnknownNamesThrow)
{
	Agent agent;
	agent.obsBuilderName = "MissingObsForRegistryTest";
	agent.actionParserName = "MissingActionForRegistryTest";

	bool sawObsThrow = false;
	try
	{
		std::unique_ptr<RLGC::ObsBuilder> obsBuilder(agent.MakeObsBuilder());
	}
	catch (const std::runtime_error& e)
	{
		sawObsThrow = std::string(e.what()).find("Unknown obs builder") != std::string::npos;
	}
	ASSERT_TRUE(sawObsThrow);

	bool sawActionThrow = false;
	try
	{
		std::unique_ptr<RLGC::ActionParser> actionParser(agent.MakeActionParser());
	}
	catch (const std::runtime_error& e)
	{
		sawActionThrow = std::string(e.what()).find("Unknown action parser") != std::string::npos;
	}
	ASSERT_TRUE(sawActionThrow);
}
