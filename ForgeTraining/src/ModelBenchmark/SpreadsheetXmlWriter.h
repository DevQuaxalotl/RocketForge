#pragma once

#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

namespace RocketForgeTraining
{
	/**
	 * Streams an Excel 2003 XML workbook (SpreadsheetML) whose cells are all strings.
	 * Worksheet names are sanitised to Excel's rules, and Lines() splits long lists across numbered sheets.
	 */
	class SpreadsheetXmlWriter
	{
	public:
		/** Excel 2003 caps a worksheet at 65536 rows; one is left for the header. */
		static constexpr size_t MAX_DATA_ROWS = 65535;

		/** Writes the workbook prologue and its Default and Header styles. */
		explicit SpreadsheetXmlWriter(std::ostream& out);

		void OpenWorksheet(const std::string& name);
		void Row(const std::vector<std::string>& cells, bool header = false);
		void CloseWorksheet();

		/** Writes one string per row under a header of the sheet name; a list too long for one sheet spans "<name> 1", "<name> 2"… */
		void Lines(const std::string& name, const std::vector<std::string>& lines);

		/** Closes the workbook. */
		void Finish();

	private:
		std::ostream& _out;
	};
}
