#include "TextTable.h"

#include <algorithm>

using namespace std;

namespace {

	void AppendRow(string& table, const vector<string>& cells, const vector<size_t>& widths)
	{
		string line;
		for (size_t column = 0; column < widths.size(); column++) {
			const string cell = column < cells.size() ? cells[column] : string();
			line += cell;

			if (column + 1 < widths.size())
				line += string(widths[column] - cell.size() + 2, ' ');
		}

		// Short trailing cells would otherwise leave padding at the end of the line.
		line.erase(line.find_last_not_of(' ') + 1);
		table += line + "\n";
	}

}

string RocketForge::Cli::FormatTable(const vector<string>& headers, const vector<vector<string>>& rows)
{
	vector<size_t> widths(headers.size());
	for (size_t column = 0; column < headers.size(); column++)
		widths[column] = headers[column].size();

	for (const vector<string>& row : rows) {
		for (size_t column = 0; column < row.size() && column < widths.size(); column++)
			widths[column] = std::max(widths[column], row[column].size());
	}

	vector<string> rule;
	for (size_t width : widths)
		rule.push_back(string(width, '-'));

	string table;
	AppendRow(table, headers, widths);
	AppendRow(table, rule, widths);
	for (const vector<string>& row : rows)
		AppendRow(table, row, widths);

	return table;
}
