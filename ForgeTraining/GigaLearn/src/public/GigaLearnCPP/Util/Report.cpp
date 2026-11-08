#pragma once

#include "Report.h"

void GGL::Report::Display(const std::vector<std::string>& keyRows) const {
	std::stringstream stream;
	stream << std::string(8, '\n');
	stream << RG_DIVIDER << std::endl;
	stream << "[Training metrics]" << std::endl;
	for (std::string row : keyRows) {
		if (!row.empty()) {

			int indentLevel = 0;
			while (row[0] == '-') {
				indentLevel++;
				row.erase(row.begin());
			}

			std::string prefix = {};
			if (indentLevel > 0) {
				prefix += std::string((indentLevel - 1) * 3, ' ');
				prefix += " - ";
			}
			if (Has(row)) {
				stream << prefix << SingleToString(row, true) << std::endl;
			} else {
				continue;
			}
		} else {
			stream << std::endl;
		}
	}

	stream << "[Training metrics end]" << std::endl;
	stream << std::string(4, '\n');

	// Readers tailing a redirected stdout apply a report only once its end marker arrives.
	std::cout << stream.str() << std::flush;
}
