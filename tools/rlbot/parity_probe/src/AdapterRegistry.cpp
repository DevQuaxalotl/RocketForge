#include <RocketForge/ParityProbe/AdapterRegistry.h>

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace RocketForge::ParityProbe
{
	namespace
	{
		std::unordered_map<std::string, FrameAdapterFactory>& GetAdapterMap()
		{
			static std::unordered_map<std::string, FrameAdapterFactory> adapters;
			return adapters;
		}
	}

	void FrameAdapterRegistry::Register(const std::string& name, FrameAdapterFactory factory)
	{
		if (name.empty())
			throw std::runtime_error("parity adapter name cannot be empty");
		if (!factory)
			throw std::runtime_error("parity adapter factory cannot be empty");
		GetAdapterMap()[name] = std::move(factory);
	}

	std::unique_ptr<FrameAdapter> FrameAdapterRegistry::Create(const std::string& name)
	{
		if (name.empty())
			return {};
		auto& adapters = GetAdapterMap();
		const auto iterator = adapters.find(name);
		if (iterator == adapters.end())
			throw std::runtime_error("unknown parity adapter: " + name);
		return iterator->second();
	}

	std::vector<std::string> FrameAdapterRegistry::GetAvailableAdapters()
	{
		std::vector<std::string> names;
		for (const auto& [name, factory] : GetAdapterMap())
			names.push_back(name);
		std::sort(names.begin(), names.end());
		return names;
	}
}
