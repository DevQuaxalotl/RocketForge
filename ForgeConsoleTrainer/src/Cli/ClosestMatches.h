#pragma once

#include <string>
#include <vector>

namespace RocketForge::Cli {

/**
 * Returns up to maxResults candidates that look like value, compared case-insensitively.
 * Candidates that start with or contain value come first, then those within an edit distance of max(2, value.size() / 3).
 */
std::vector<std::string> ClosestMatches(const std::string& value, const std::vector<std::string>& candidates, size_t maxResults = 3);

}
