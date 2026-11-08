#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace RocketForge::Cli {

/** The options one command accepts; anything not listed is rejected with a suggestion. */
struct CliOptionSpec {
	/** Options that take the next argument as their value: "--name value". */
	std::vector<std::string> valueOptions;

	/** Options that stand alone: "--flag". */
	std::vector<std::string> switches;

	/** Arguments that aren't options, such as the subject of `list <subject>`. */
	size_t maxPositionals = 0;
};

/** One command's parsed arguments (everything after the command name). */
class CliArgs {
public:
	/** Throws CliUsageError for an unknown or repeated option, a missing or empty value, or too many positionals. */
	static CliArgs Parse(const std::vector<std::string>& args, const CliOptionSpec& spec);

	std::optional<std::string> Get(const std::string& option) const;
	bool Has(const std::string& switchName) const;

	/** Throws CliUsageError when the value isn't a non-negative whole number. */
	std::optional<uint64_t> GetUint64(const std::string& option) const;

	/** Throws CliUsageError when the value isn't a number. */
	std::optional<double> GetDouble(const std::string& option) const;

	const std::vector<std::string>& Positionals() const;

private:
	std::map<std::string, std::string> _values;
	std::set<std::string> _switches;
	std::vector<std::string> _positionals;
};

}
