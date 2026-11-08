#pragma once

#include <GigaLearnInfer/Agent.h>
#include <RLGymCPP/ActionParsers/ActionParser.h>
#include <RLGymCPP/ObsBuilders/ObsBuilder.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace RocketForge::ParityProbe
{
	/** Mutable context exposed to optional, locally compiled fixture adapters. **/
	struct FrameHookContext
	{
		Agent& agent;
		RLGC::ObsBuilder& obsBuilder;
		RLGC::ActionParser& actionParser;
		const nlohmann::json& frameJson;
		RLGC::GameState& state;
		uint32_t targetCarId;
	};

	/** Generic extension point for state that cannot be represented by public GameState fields. **/
	class FrameAdapter
	{
	public:
		virtual ~FrameAdapter() = default;

		/** Runs after canonical state linking and before ObsBuilder::Reset(). **/
		virtual void BeforeReset(FrameHookContext&) {}

		/** Runs immediately before observation RNG seeding and BuildObsInto(). **/
		virtual void BeforeBuildObs(FrameHookContext&) {}
	};

	using FrameAdapterFactory = std::function<std::unique_ptr<FrameAdapter>()>;

	/** Registry populated by optional adapter translation units at static initialization. **/
	class FrameAdapterRegistry
	{
	public:
		static void Register(const std::string& name, FrameAdapterFactory factory);
		static std::unique_ptr<FrameAdapter> Create(const std::string& name);
		static std::vector<std::string> GetAvailableAdapters();
	};
}

#define REGISTER_PARITY_FRAME_ADAPTER(NAME, TYPE)                       \
namespace                                                               \
{                                                                       \
	struct TYPE##ParityFrameAdapterRegistrator                            \
	{                                                                     \
		TYPE##ParityFrameAdapterRegistrator()                               \
		{                                                                   \
			RocketForge::ParityProbe::FrameAdapterRegistry::Register(          \
				NAME, []() { return std::make_unique<TYPE>(); });                 \
		}                                                                   \
	};                                                                    \
	static TYPE##ParityFrameAdapterRegistrator                            \
		TYPE##parity_frame_adapter_registrator_instance;                    \
}
