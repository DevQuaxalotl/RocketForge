#pragma once

#include <QObject>
#include <QString>

#include <streambuf>
#include <string>

/**
 * ConsoleCapture
 *
 * A std::streambuf subclass that intercepts everything written to std::cout
 * and re-emits each complete line as a Qt signal. This lets the GUI display
 * training output in a QPlainTextEdit without a visible Windows console.
 *
 * Usage:
 *   m_capture = new ConsoleCapture(this);
 *   std::cout.rdbuf(m_capture);      // redirect cout → ConsoleCapture
 *   connect(m_capture, &ConsoleCapture::LineReceived, ...);
 *
 * The original rdbuf should be restored before this object is destroyed
 * (e.g. call std::cout.rdbuf(nullptr) in the owner's destructor).
 *
 * Thread safety: LineReceived is emitted on whichever thread writes to cout.
 * Connect it with Qt::QueuedConnection when the slot lives on the UI thread.
 */
class ConsoleCapture : public QObject, public std::streambuf
{
    Q_OBJECT

public:
    explicit ConsoleCapture(QObject* parent = nullptr) : QObject(parent) {}

signals:
    /** Emitted once per complete line (newline stripped). */
    void LineReceived(const QString& line);

protected:
    /**
     * Called by the C++ runtime for single-character writes.
     * Appends the character to the internal buffer; emits LineReceived
     * whenever a newline is encountered.
     */
    int overflow(int c) override
    {
        if (c != EOF) {
            m_buffer += static_cast<char>(c);
            if (c == '\n')
                FlushLine();
        }
        return c;
    }

    /**
     * Called by the C++ runtime for bulk string writes (e.g. std::cout << str).
     * Processes each character individually so embedded newlines trigger a flush.
     */
    std::streamsize xsputn(const char* s, std::streamsize n) override
    {
        for (std::streamsize i = 0; i < n; ++i) {
            m_buffer += s[i];
            if (s[i] == '\n')
                FlushLine();
        }
        return n;
    }

    /**
     * Called when the stream is explicitly flushed (e.g. std::cout << std::flush).
     * Emits any partial line that has not yet ended with a newline.
     */
    int sync() override
    {
        if (!m_buffer.empty())
            FlushLine();
        return 0;
    }

private:
    /**
     * Strips trailing CR/LF from the buffer, emits LineReceived, then clears
     * the buffer for the next line.
     */
    void FlushLine()
    {
        QString line = QString::fromStdString(m_buffer);
        if (line.endsWith('\n'))  line.chop(1);
        if (line.endsWith('\r'))  line.chop(1);
        m_buffer.clear();
        emit LineReceived(line);
    }

    /** Accumulates characters until a newline terminates the current line. */
    std::string m_buffer;
};
