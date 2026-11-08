#include "SpreadsheetXmlWriter.h"

#include <algorithm>

using namespace RocketForgeTraining;

namespace
{
	constexpr size_t MAX_WORKSHEET_NAME_CHARACTERS = 31;

	std::string EscapeXml(const std::string& text)
	{
		std::string escaped;
		escaped.reserve(text.size());
		for (const char character : text) {
			switch (character) {
			case '&': escaped += "&amp;"; break;
			case '<': escaped += "&lt;"; break;
			case '>': escaped += "&gt;"; break;
			case '"': escaped += "&quot;"; break;
			default: escaped += character; break;
			}
		}

		return escaped;
	}

	std::string SanitizeWorksheetName(const std::string& name)
	{
		static const std::string invalidCharacters = "\\/?:*[]";

		std::string sanitized = name;
		std::replace_if(sanitized.begin(), sanitized.end(), [](char character) {
			return invalidCharacters.find(character) != std::string::npos;
		}, '_');

		if (sanitized.empty())
			sanitized = "Sheet";

		// Excel's limit counts characters, so never cut a UTF-8 sequence in half.
		size_t characters = 0;
		for (size_t byte = 0; byte < sanitized.size(); ++byte) {
			const bool continuationByte = (static_cast<unsigned char>(sanitized[byte]) & 0xC0) == 0x80;
			if (!continuationByte && characters++ == MAX_WORKSHEET_NAME_CHARACTERS)
				return sanitized.substr(0, byte);
		}

		return sanitized;
	}
}

SpreadsheetXmlWriter::SpreadsheetXmlWriter(std::ostream& out)
	: _out(out)
{
	_out << "<?xml version=\"1.0\"?>";
	_out << "<?mso-application progid=\"Excel.Sheet\"?>";
	_out << "<Workbook xmlns=\"urn:schemas-microsoft-com:office:spreadsheet\"";
	_out << " xmlns:o=\"urn:schemas-microsoft-com:office:office\"";
	_out << " xmlns:x=\"urn:schemas-microsoft-com:office:excel\"";
	_out << " xmlns:ss=\"urn:schemas-microsoft-com:office:spreadsheet\">";
	_out << "<Styles>";
	_out << "<Style ss:ID=\"Default\" ss:Name=\"Normal\"><Alignment ss:Vertical=\"Bottom\"/></Style>";
	_out << "<Style ss:ID=\"Header\"><Font ss:Bold=\"1\"/></Style>";
	_out << "</Styles>";
}

void SpreadsheetXmlWriter::OpenWorksheet(const std::string& name)
{
	_out << "<Worksheet ss:Name=\"" << EscapeXml(SanitizeWorksheetName(name)) << "\"><Table>";
}

void SpreadsheetXmlWriter::Row(const std::vector<std::string>& cells, bool header)
{
	_out << "<Row>";
	for (const std::string& cell : cells) {
		_out << "<Cell";
		if (header)
			_out << " ss:StyleID=\"Header\"";

		_out << "><Data ss:Type=\"String\">" << EscapeXml(cell) << "</Data></Cell>";
	}

	_out << "</Row>";
}

void SpreadsheetXmlWriter::CloseWorksheet()
{
	_out << "</Table></Worksheet>";
}

void SpreadsheetXmlWriter::Lines(const std::string& name, const std::vector<std::string>& lines)
{
	const size_t sheetCount = std::max<size_t>(1, (lines.size() + MAX_DATA_ROWS - 1) / MAX_DATA_ROWS);
	const bool headerRow = true;

	for (size_t sheetIndex = 0; sheetIndex < sheetCount; ++sheetIndex) {
		OpenWorksheet(sheetCount == 1 ? name : name + " " + std::to_string(sheetIndex + 1));
		Row({ name }, headerRow);

		const size_t begin = sheetIndex * MAX_DATA_ROWS;
		const size_t end = std::min(begin + MAX_DATA_ROWS, lines.size());
		for (size_t lineIndex = begin; lineIndex < end; ++lineIndex)
			Row({ lines[lineIndex] });

		CloseWorksheet();
	}
}

void SpreadsheetXmlWriter::Finish()
{
	_out << "</Workbook>";
}
