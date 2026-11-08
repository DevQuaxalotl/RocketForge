#include "RLBotDevelopmentExtension.h"
#include "Runtime/RunRLBot.h"

int main(int argc, char* argv[])
{
    return RunRLBot(
        argc,
        argv,
        RLBotLaunchMode::ROCKET_FORGE_DEVELOPMENT,
        CreateRLBotDevelopmentExtension);
}
