#include "SegmentVersioning.h"

#include <iomanip>
#include <regex>
#include <sstream>

std::string GGL::MakeSegmentVersionId(int major, int minor) {
	std::stringstream stream;
	stream << "v" << major << "." << std::setw(2) << std::setfill('0') << minor;
	return stream.str();
}

bool GGL::TryParseSegmentVersionId(const std::string& text, SegmentVersionNumber& out) {
	static const std::regex rx(R"(.*?v(\d+)\.(\d+).*)", std::regex_constants::icase);
	std::smatch match;
	if (!std::regex_match(text, match, rx))
		return false;

	out.major = std::stoi(match[1].str());
	out.minor = std::stoi(match[2].str());
	return true;
}
