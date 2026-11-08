#include "CliArgs.h"

#include "CliErrors.h"
#include "ClosestMatches.h"

#include <algorithm>
#include <cmath>

using namespace std;
using namespace RocketForge::Cli;

namespace {

	bool IsOptionToken(const string& token)
	{
		return token.rfind("--", 0) == 0;
	}

	bool Contains(const vector<string>& names, const string& name)
	{
		return std::find(names.begin(), names.end(), name) != names.end();
	}

	string JoinNames(const vector<string>& names)
	{
		string joined;
		for (const string& name : names)
			joined += (joined.empty() ? "" : ", ") + name;
		return joined;
	}

	[[noreturn]] void ThrowUnknownOption(const string& option, const CliOptionSpec& spec)
	{
		vector<string> known = spec.valueOptions;
		known.insert(known.end(), spec.switches.begin(), spec.switches.end());

		string message = "unknown option '" + option + "'";

		const vector<string> suggestions = ClosestMatches(option, known);
		if (!suggestions.empty())
			message += ". Did you mean: " + JoinNames(suggestions) + "?";

		throw CliUsageError(message);
	}

}

CliArgs CliArgs::Parse(const vector<string>& args, const CliOptionSpec& spec)
{
	CliArgs parsed;

	for (size_t i = 0; i < args.size(); i++) {
		const string& token = args[i];

		if (!IsOptionToken(token)) {
			if (parsed._positionals.size() >= spec.maxPositionals)
				throw CliUsageError("unexpected argument '" + token + "'");

			parsed._positionals.push_back(token);
			continue;
		}

		if (Contains(spec.switches, token)) {
			if (!parsed._switches.insert(token).second)
				throw CliUsageError(token + " given twice");
			continue;
		}

		if (!Contains(spec.valueOptions, token))
			ThrowUnknownOption(token, spec);

		if (i + 1 >= args.size() || IsOptionToken(args[i + 1]))
			throw CliUsageError(token + " requires a value");

		// Commands read an empty value as an absent option, so it would quietly select the default, such as the active segment for a delete.
		if (args[i + 1].empty())
			throw CliUsageError(token + " requires a non-empty value");

		if (!parsed._values.emplace(token, args[++i]).second)
			throw CliUsageError(token + " given twice");
	}

	return parsed;
}

optional<string> CliArgs::Get(const string& option) const
{
	const auto found = _values.find(option);
	if (found == _values.end())
		return nullopt;

	return found->second;
}

bool CliArgs::Has(const string& switchName) const
{
	return _switches.count(switchName) != 0;
}

optional<uint64_t> CliArgs::GetUint64(const string& option) const
{
	const optional<string> text = Get(option);
	if (!text.has_value())
		return nullopt;

	const bool allDigits = !text->empty() && std::all_of(text->begin(), text->end(), [](char c) { return c >= '0' && c <= '9'; });
	if (!allDigits)
		throw CliUsageError(option + " must be a whole number (got '" + *text + "')");

	try {
		return std::stoull(*text);
	} catch (const std::out_of_range&) {
		throw CliUsageError(option + " is too large (got '" + *text + "')");
	}
}

optional<double> CliArgs::GetDouble(const string& option) const
{
	const optional<string> text = Get(option);
	if (!text.has_value())
		return nullopt;

	size_t parsedChars = 0;
	double value = 0.0;
	try {
		value = std::stod(*text, &parsedChars);
	} catch (const std::exception&) {
		parsedChars = 0;
	}

	if (parsedChars == 0 || parsedChars != text->size() || !std::isfinite(value))
		throw CliUsageError(option + " must be a number (got '" + *text + "')");

	return value;
}

const vector<string>& CliArgs::Positionals() const
{
	return _positionals;
}
