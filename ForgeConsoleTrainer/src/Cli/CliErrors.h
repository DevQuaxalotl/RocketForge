#pragma once

#include <stdexcept>

namespace RocketForge::Cli {

/** Bad command-line input or an unknown name; RunCli prints `<command>: <message>` and exits 2. */
struct CliUsageError : std::runtime_error {
	using std::runtime_error::runtime_error;
};

/** A command failed while acting on valid input; RunCli prints `<command>: <message>` and exits 1. */
struct CliRuntimeError : std::runtime_error {
	using std::runtime_error::runtime_error;
};

}
