#pragma once

#include "TrainingStage.h"

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <vector>

/**
* Registers a stage type to the StageRegistry via static variable.
* Static variables in anonymous namespaces are initialized before main() is called,
* allowing them to register with the global registry map before exe's search them.
*
* Note: Registration depends on static initialization.
*   Stage files are built as OBJECT libraries and explicitly included in the executable via CMake,
*   ensuring the registration runs even if unused. Keep registration in .cpp files; headers are not needed.
*/
#define REGISTER_STAGE(NAME, TYPE, COMPAT)                          \
namespace {                                                          \
    struct TYPE##StageRegistrator {                                  \
        TYPE##StageRegistrator() {                                   \
            RocketForge::StageRegistry::Register(NAME,               \
                [](std::unique_ptr<Agent> agent) {                   \
                    return RocketForge::TrainingStage::Create<TYPE>(std::move(agent)); \
                }, COMPAT);                                          \
        }                                                            \
    };                                                               \
    static TYPE##StageRegistrator TYPE##registrator_instance;        \
}

namespace RocketForge
{
    using StageFactory = std::function<std::unique_ptr<TrainingStage>(std::unique_ptr<Agent>)>;

    /** 
    * Information used to determine compatability between stages and agents
    * More data such as supported game modes can be added in the future
    */
    struct StageCompatInfo {
        int minAgentTeamSize = 1;
    };

    /** 
    * Wrapper for static map containing registered stages that executables can search for 
    * Each Stage class must register itself with REGISTER_STAGE to be added
    */
    class StageRegistry
    {
    public:
        static void Register(const std::string& name, StageFactory factory, StageCompatInfo compat)
        {
            GetMap()[name] = { factory, compat };
        }

        static std::unique_ptr<TrainingStage> Create(
            const std::string& name,
            std::unique_ptr<Agent> agent)
        {
            auto& map = GetMap();
            auto it = map.find(name);
            if (it == map.end())
                throw std::runtime_error("Unknown stage: " + name);

            return it->second.first(std::move(agent));
        }

        static std::vector<std::string> GetAvailableStages()
        {
            std::vector<std::string> result;
            result.reserve(GetMap().size());

            for (const auto& [name, _] : GetMap())
            {
                result.push_back(name);
            }

            std::sort(result.begin(), result.end());
            return result;
        }

        static std::vector<std::string> GetCompatibleStages(const Agent& agent)
        {
            std::vector<std::string> result;
            for (const auto& [name, entry] : GetMap())
            {
                if (entry.second.minAgentTeamSize <= agent.maxTeamSize)
                    result.push_back(name);
            }
            std::sort(result.begin(), result.end());
            return result;
        }

        static StageCompatInfo GetCompatInfo(const std::string& name)
        {
            auto& map = GetMap();
            auto it = map.find(name);
            if (it == map.end())
                return StageCompatInfo{};
            return it->second.second;
        }

    private:
        static std::unordered_map<std::string, std::pair<StageFactory, StageCompatInfo>>& GetMap()
        {
            static std::unordered_map<std::string, std::pair<StageFactory, StageCompatInfo>> map;
            return map;
        }
    };
}