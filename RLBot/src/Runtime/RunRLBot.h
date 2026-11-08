#pragma once

#include "RLBotClientExtension.h"

enum class RLBotLaunchMode {
    PORTABLE,
    ROCKET_FORGE_DEVELOPMENT,
};

int RunRLBot(
    int argc,
    char* argv[],
    RLBotLaunchMode mode,
    RLBotClientExtensionFactory extensionFactory = {});
