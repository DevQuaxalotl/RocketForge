#pragma once
#include <QString>

namespace ForgeGUI {

// Formats a raw timestep count as a human-readable string: "1.50B", "500.00M", etc.
// Uses T/B/M/K scaling. Values below 1000 are returned as plain integers.
inline QString FormatTimesteps(quint64 timesteps)
{
	if (timesteps < 1000)
		return QString::number(static_cast<qulonglong>(timesteps));

	struct Unit { double divisor; const char* suffix; };
	static const Unit units[] = {
		{ 1e12, "T" },
		{ 1e9,  "B" },
		{ 1e6,  "M" },
		{ 1e3,  "K" }
	};
	for (const Unit& u : units) {
		if (static_cast<double>(timesteps) >= u.divisor)
			return QString("%1%2").arg(static_cast<double>(timesteps) / u.divisor, 0, 'f', 2).arg(u.suffix);
	}
	return QString::number(static_cast<qulonglong>(timesteps));
}

// Parses a user-entered timestep string (e.g. "1B", "1.5 B", "500M", "500 M", "1000")
// into a value in millions. Bare numbers are treated as millions.
// Sets ok=false if the string cannot be parsed.
inline double ParseTimestepsToMillions(const QString& text, bool& ok)
{
	ok = false;
	QString s = text.trimmed();
	if (s.isEmpty()) return 0.0;

	QChar last = s.back().toUpper();
	double mult = 1.0;
	bool hasSuffix = true;

	if      (last == 'T') mult = 1'000'000.0;
	else if (last == 'B') mult = 1'000.0;
	else if (last == 'M') mult = 1.0;
	else if (last == 'K') mult = 0.001;
	else                  hasSuffix = false;

	QString numPart = hasSuffix ? s.chopped(1).trimmed() : s;
	double val = numPart.toDouble(&ok);
	if (!ok) return 0.0;
	return val * mult;
}

} // namespace ForgeGUI
