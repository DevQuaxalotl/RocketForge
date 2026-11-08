#pragma once

#include <string>
#include <vector>

namespace RocketForge::Cli {

/** Left-aligned columns separated by two spaces, with a dashed rule under the headers. */
std::string FormatTable(const std::vector<std::string>& headers, const std::vector<std::vector<std::string>>& rows);

}
