#pragma once

#include "Cli/CliArgs.h"

namespace RocketForge::Cli {

/** `action-trace`: runs a saved policy offline and writes its per-tick actions to CSV plus a summary JSON. */
int RunActionTraceCommand(const CliArgs& args);

}
