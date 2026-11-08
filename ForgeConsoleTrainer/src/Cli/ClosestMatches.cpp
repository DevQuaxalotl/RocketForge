#include "ClosestMatches.h"

#include <algorithm>
#include <cctype>

using namespace std;

namespace {

	string ToLower(const string& text)
	{
		string lowered = text;
		for (char& c : lowered)
			c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
		return lowered;
	}

	size_t EditDistance(const string& from, const string& to)
	{
		vector<size_t> previousRow(to.size() + 1);
		vector<size_t> currentRow(to.size() + 1);
		for (size_t column = 0; column <= to.size(); column++)
			previousRow[column] = column;

		for (size_t row = 1; row <= from.size(); row++) {
			currentRow[0] = row;

			for (size_t column = 1; column <= to.size(); column++) {
				const size_t substitution = previousRow[column - 1] + (from[row - 1] == to[column - 1] ? 0 : 1);
				currentRow[column] = std::min({ previousRow[column] + 1, currentRow[column - 1] + 1, substitution });
			}

			std::swap(previousRow, currentRow);
		}

		return previousRow[to.size()];
	}

}

vector<string> RocketForge::Cli::ClosestMatches(const string& value, const vector<string>& candidates, size_t maxResults)
{
	// Lower rank sorts first: 0 = prefix, 1 = substring, 2 + distance = edit-distance hit.
	struct RankedCandidate {
		size_t rank;
		size_t order;
		const string* name;
	};

	const string loweredValue = ToLower(value);
	const size_t maxDistance = std::max<size_t>(2, value.size() / 3);

	vector<RankedCandidate> ranked;
	for (size_t i = 0; i < candidates.size(); i++) {
		const string loweredCandidate = ToLower(candidates[i]);
		const size_t foundAt = loweredValue.empty() ? string::npos : loweredCandidate.find(loweredValue);

		if (foundAt == 0) {
			ranked.push_back({ 0, i, &candidates[i] });
			continue;
		}

		if (foundAt != string::npos) {
			ranked.push_back({ 1, i, &candidates[i] });
			continue;
		}

		const size_t distance = EditDistance(loweredValue, loweredCandidate);
		if (distance <= maxDistance)
			ranked.push_back({ 2 + distance, i, &candidates[i] });
	}

	std::sort(ranked.begin(), ranked.end(), [](const RankedCandidate& a, const RankedCandidate& b) {
		return a.rank != b.rank ? a.rank < b.rank : a.order < b.order;
	});

	vector<string> matches;
	for (size_t i = 0; i < ranked.size() && i < maxResults; i++)
		matches.push_back(*ranked[i].name);

	return matches;
}
