#pragma once

#include "Agent.h"

#include <algorithm>
#include <string>
#include <vector>

/** 
* Registers an agent type to the AgentRegistry via static variable.
* Static variables in anonymous namespaces are initialized before main() is called,
* allowing them to register with the global registry map before exe's search them.
* 
* Note: Registration depends on static initialization. 
*   Agent files are built as OBJECT libraries and explicitly included in the executable via CMake, 
*   ensuring the registration runs even if unused. Keep registration in .cpp files; headers are not needed.
*/
#define REGISTER_AGENT(NAME, TYPE)                              \
namespace {                                                     \
    struct TYPE##Registrator {                                  \
        TYPE##Registrator() {                                   \
            AgentRegistry::Register(NAME, []() {                \
                return std::make_unique<TYPE>();                \
            });                                                 \
        }                                                       \
    };                                                          \
    static TYPE##Registrator TYPE##registrator_instance;        \
}

using AgentFactory = std::function<std::unique_ptr<Agent>()>;

/**
* Wrapper for static map containing registered agents that executables can search for
* Each Agent class must register itself with REGISTER_AGENT to be added
*/
class AgentRegistry
{
public:
    static void Register(const std::string& name, AgentFactory factory)
    {
        GetMap()[name] = factory;
    }

    static std::unique_ptr<Agent> Create(const std::string& name)
    {
        auto& map = GetMap();
        auto it = map.find(name);
        if (it == map.end())
            throw std::runtime_error("Unknown agent: " + name);

        return it->second();
    }

    static std::vector<std::string> GetAvailableAgents()
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

private:
    static std::unordered_map<std::string, AgentFactory>& GetMap()
    {
        static std::unordered_map<std::string, AgentFactory> map;
        return map;
    }
};