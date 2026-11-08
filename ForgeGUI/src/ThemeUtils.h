#pragma once

#include <QColor>
#include <QCoreApplication>
#include <QRegularExpression>
#include <QString>
#include <QVariant>

namespace ForgeGUI
{
	inline constexpr const char* ColorsEnabledApplicationProperty = "rocketForgeColorsEnabled";

	/** Dynamic QLabel property matched by theme.qss: "muted" for secondary text, "caption" for small muted headings. **/
	inline constexpr const char* TextRoleProperty = "textRole";

	/**
	 * Theme colors for painted cells and item roles, which the style sheet cannot reach.
	 * They mirror the palette key at the top of resources/theme.qss.
	 * POSITIVE and NEGATIVE encode a value's sign, so they are never neutralized by the color preference.
	 **/
	namespace Theme
	{
		inline const QColor TEXT("#d8d0bf");
		inline const QColor TEXT_MUTED("#948c7c");
		inline const QColor TEXT_DISABLED("#6f746f");
		inline const QColor SURFACE_PANEL("#1d2223");
		inline const QColor SURFACE_SUNKEN("#202526");
		inline const QColor SURFACE_RAISED("#2c3436");
		inline const QColor SELECTION("#2b747a");
		inline const QColor AXIS("#6f746f");
		inline const QColor POSITIVE("#73c88f");
		inline const QColor NEGATIVE("#e47777");
		inline const QColor NEGATIVE_BRIGHT("#ff7d7d");
		inline const QColor COOL("#5aa9c4");
	}

	/** Mixes accentAmount (0..1) of accent into base. **/
	inline QColor BlendColor(const QColor& base, const QColor& accent, double accentAmount)
	{
		const double baseAmount = 1.0 - accentAmount;
		return QColor(
			(int)(base.red() * baseAmount + accent.red() * accentAmount),
			(int)(base.green() * baseAmount + accent.green() * accentAmount),
			(int)(base.blue() * baseAmount + accent.blue() * accentAmount));
	}

	/** Replaces decorative accent colors with the normal label text color when disabled. **/
	inline QColor ApplyColorPreference(const QColor& color, bool colorsEnabled)
	{
		if (colorsEnabled || !color.isValid())
			return color;

		switch (color.rgb() & 0x00ffffff) {
		case 0x00e47777:
		case 0x00ff7d7d:
		case 0x00d98bbd:
		case 0x00b78bd9:
		case 0x00b9cf45:
		case 0x0073c88f:
		case 0x0062c6cc:
		case 0x005aa9c4:
		case 0x00e2a84b: {
			QColor neutralText("#cfc5b2");
			neutralText.setAlpha(color.alpha());
			return neutralText;
		}
		default:
			return color;
		}
	}

	/** Neutralizes accent colors in a Qt style sheet while preserving its base theme. **/
	inline QString ApplyColorPreference(const QString& styleSheet, bool colorsEnabled)
	{
		if (colorsEnabled)
			return styleSheet;

		static const QRegularExpression hexColorPattern(QStringLiteral(R"(#[0-9A-Fa-f]{6}\b)"));
		QString result;
		qsizetype consumed = 0;
		QRegularExpressionMatchIterator matches = hexColorPattern.globalMatch(styleSheet);
		while (matches.hasNext()) {
			const QRegularExpressionMatch match = matches.next();
			result += styleSheet.mid(consumed, match.capturedStart() - consumed);
			result += ApplyColorPreference(QColor(match.captured()), false).name(QColor::HexRgb);
			consumed = match.capturedEnd();
		}
		result += styleSheet.mid(consumed);
		return result;
	}

	/** Returns the application-wide color preference, defaulting to enabled. **/
	inline bool ApplicationColorsEnabled()
	{
		QCoreApplication* application = QCoreApplication::instance();
		if (!application)
			return true;

		const QVariant preference = application->property(ColorsEnabledApplicationProperty);
		return !preference.isValid() || preference.toBool();
	}
}
