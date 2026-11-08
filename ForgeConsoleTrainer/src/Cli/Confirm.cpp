#include "Confirm.h"

#include <cctype>
#include <cstdio>
#include <istream>
#include <ostream>

#ifdef _MSC_VER
#include <io.h>
#else
#include <unistd.h>
#endif

using namespace std;
using namespace RocketForge::Cli;

bool RocketForge::Cli::StdinIsInteractive()
{
#ifdef _MSC_VER
	return _isatty(_fileno(stdin)) != 0;
#else
	return isatty(fileno(stdin)) != 0;
#endif
}

ConfirmResult RocketForge::Cli::ConfirmDestructive(const string& summary, bool yesFlag, bool interactive, istream& in, ostream& out)
{
	out << summary << "\n";

	if (yesFlag)
		return ConfirmResult::Confirmed;

	if (!interactive) {
		out << "refusing without --yes (no interactive terminal)\n";
		return ConfirmResult::RefusedNonInteractive;
	}

	out << "Continue? [y/N] " << std::flush;

	string answer;
	if (!std::getline(in, answer))
		return ConfirmResult::Declined;

	for (char c : answer) {
		if (std::isspace(static_cast<unsigned char>(c)))
			continue;

		return (c == 'y' || c == 'Y') ? ConfirmResult::Confirmed : ConfirmResult::Declined;
	}

	return ConfirmResult::Declined;
}
