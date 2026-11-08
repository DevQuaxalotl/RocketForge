#pragma once

#include <iosfwd>
#include <string>

namespace RocketForge::Cli {

enum class ConfirmResult { Confirmed, Declined, RefusedNonInteractive };

/** True when stdin is a terminal a person can answer prompts on. */
bool StdinIsInteractive();

/**
 * Prints summary, then confirms: yesFlag accepts without asking,
 * a non-interactive stdin refuses (so pipes and pods never delete unattended),
 * and otherwise only a "y" answer confirms.
 * Everything, including the refusal line, is written to out; commands pass std::cerr so their stdout stays machine-readable.
 */
ConfirmResult ConfirmDestructive(const std::string& summary, bool yesFlag, bool interactive, std::istream& in, std::ostream& out);

}
