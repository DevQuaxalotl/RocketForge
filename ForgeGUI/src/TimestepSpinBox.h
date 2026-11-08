#pragma once
#include <QDoubleSpinBox>

class QResizeEvent;
class QToolButton;

// QDoubleSpinBox subclass that stores values internally in millions but
// displays and accepts input using auto-scaled suffixes: K, M, B, T.
// Examples: type "1.5B" to set 1.5 billion; displays "500.00M" for 500 million.
class TimestepSpinBox : public QDoubleSpinBox
{
	Q_OBJECT
public:
	explicit TimestepSpinBox(QWidget* parent = nullptr);

	QString textFromValue(double millions) const override;
	double  valueFromText(const QString& text) const override;
	QValidator::State validate(QString& input, int& pos) const override;
	void stepBy(int steps) override;

protected:
	void resizeEvent(QResizeEvent* event) override;

private:
	QToolButton* m_stepUpButton = nullptr;
	QToolButton* m_stepDownButton = nullptr;
};
