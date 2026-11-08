#include "TimestepSpinBox.h"
#include "FormatUtils.h"

#include <QColor>
#include <QIcon>
#include <QLineEdit>
#include <QPainter>
#include <QPen>
#include <QPixmap>
#include <QPointF>
#include <QResizeEvent>
#include <QSize>
#include <QToolButton>
#include <QTransform>

namespace
{
	constexpr int StepButtonSize = 22;
	constexpr int StepButtonGap = 2;
	constexpr int StepButtonRightMargin = 3;
	constexpr int StepButtonTextMargin = 18;

	QIcon MakeStepChevronIcon(bool up)
	{
		constexpr int iconSize = 16;
		QPixmap pixmap(iconSize, iconSize);
		pixmap.fill(Qt::transparent);

		{
			QPainter painter(&pixmap);
			painter.setRenderHint(QPainter::Antialiasing, true);
			QPen pen(QColor("#d8d0bf"), 2.0, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
			painter.setPen(pen);

			const QPointF left(4.0, 10.0);
			const QPointF center(8.0, 6.0);
			const QPointF right(12.0, 10.0);
			painter.drawLine(left, center);
			painter.drawLine(center, right);
		}

		if (!up)
			pixmap = pixmap.transformed(QTransform().rotate(180), Qt::SmoothTransformation);

		return QIcon(pixmap);
	}
}

TimestepSpinBox::TimestepSpinBox(QWidget* parent)
	: QDoubleSpinBox(parent)
{
	setButtonSymbols(QAbstractSpinBox::NoButtons);
	lineEdit()->setTextMargins(0, 0, (StepButtonSize * 2) + StepButtonGap + StepButtonTextMargin, 0);

	m_stepUpButton = new QToolButton(this);
	m_stepUpButton->setObjectName("targetTimestepsStepUpButton");
	m_stepUpButton->setFixedSize(StepButtonSize, StepButtonSize);
	m_stepUpButton->setIcon(MakeStepChevronIcon(true));
	m_stepUpButton->setIconSize(QSize(15, 15));
	m_stepUpButton->setAutoRepeat(true);
	m_stepUpButton->setFocusPolicy(Qt::NoFocus);
	m_stepUpButton->setToolTip("Increase target timesteps");
	connect(m_stepUpButton, &QToolButton::clicked, this, [this]() { stepBy(1); });

	m_stepDownButton = new QToolButton(this);
	m_stepDownButton->setObjectName("targetTimestepsStepDownButton");
	m_stepDownButton->setFixedSize(StepButtonSize, StepButtonSize);
	m_stepDownButton->setIcon(MakeStepChevronIcon(false));
	m_stepDownButton->setIconSize(QSize(15, 15));
	m_stepDownButton->setAutoRepeat(true);
	m_stepDownButton->setFocusPolicy(Qt::NoFocus);
	m_stepDownButton->setToolTip("Decrease target timesteps");
	connect(m_stepDownButton, &QToolButton::clicked, this, [this]() { stepBy(-1); });
}

QString TimestepSpinBox::textFromValue(double millions) const
{
	return ForgeGUI::FormatTimesteps(static_cast<quint64>(millions * 1'000'000.0));
}

double TimestepSpinBox::valueFromText(const QString& text) const
{
	bool ok = false;
	double v = ForgeGUI::ParseTimestepsToMillions(text, ok);
	return ok ? v : minimum();
}

void TimestepSpinBox::stepBy(int steps)
{
	const double v = value();
	double step;

	if (steps > 0) {
		// Going up: step size is determined by the tier we're currently in
		if      (v < 1'000.0)  step = 100.0;    // < 1B  → 100M steps
		else if (v < 10'000.0) step = 1'000.0;  // < 10B → 1B steps
		else                    step = 10'000.0; // ≥ 10B → 10B steps
	} else {
		// Going down: at a boundary, step into the smaller tier below it
		if      (v <= 1'000.0)  step = 100.0;    // ≤ 1B  → 100M steps
		else if (v <= 10'000.0) step = 1'000.0;  // ≤ 10B → 1B steps
		else                     step = 10'000.0; // > 10B → 10B steps
	}

	setValue(v + steps * step);
}

void TimestepSpinBox::resizeEvent(QResizeEvent* event)
{
	QDoubleSpinBox::resizeEvent(event);

	const int y = (height() - StepButtonSize) / 2;
	const int downX = width() - StepButtonRightMargin - StepButtonSize;
	const int upX = downX - StepButtonGap - StepButtonSize;

	if (m_stepUpButton)
		m_stepUpButton->setGeometry(upX, y, StepButtonSize, StepButtonSize);
	if (m_stepDownButton)
		m_stepDownButton->setGeometry(downX, y, StepButtonSize, StepButtonSize);
}

QValidator::State TimestepSpinBox::validate(QString& input, int& pos) const
{
	QString s = input.trimmed();
	if (s.isEmpty()) return QValidator::Intermediate;

	// Strip optional suffix letter
	QChar last = s.back().toUpper();
	if (last == 'T' || last == 'B' || last == 'M' || last == 'K')
		s = s.chopped(1).trimmed();

	if (s.isEmpty()) return QValidator::Intermediate;

	// Allow trailing decimal point by completing it before parsing
	QString numStr = s.endsWith('.') ? s + "0" : s;
	bool ok = false;
	numStr.toDouble(&ok);
	return ok ? QValidator::Acceptable : QValidator::Invalid;
}
