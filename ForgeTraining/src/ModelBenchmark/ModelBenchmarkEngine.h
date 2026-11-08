#pragma once

#include <RocketSim/Engine.h>

#include <algorithm>
#include <string>

namespace RocketForgeTraining {
    /** Provenance name of an engine's physics, such as "RocketSim-v3". **/
    inline std::string ModelBenchmarkEngineFamily(RocketSim::EngineId engine) {
        std::string family = RocketSim::GetEngineInfo(engine).displayName;
        std::replace(family.begin(), family.end(), ' ', '-');
        return family;
    }

    /** Order an engine steps cars in: v2 iterates a pointer-keyed set, v3 the order the cars were added. **/
    inline const char* ModelBenchmarkEngineCarOrder(RocketSim::EngineId engine) {
        return engine == RocketSim::EngineId::V2 ? "legacy-pointer-order" : "insertion-order";
    }
}
