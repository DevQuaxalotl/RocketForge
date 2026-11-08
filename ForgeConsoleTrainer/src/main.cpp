#include "Cli/CommandTable.h"

#include <RocketSim/Engine.h>

#include <iostream>

int main(int argc, char* argv[])
{
	try {
		return RocketForge::Cli::RunCli(argc, argv);
	} catch (const RocketSim::EngineUnavailableError& error) {
		std::cerr << "Error: " << error.what() << "\n";
		return 1;
	}
}
