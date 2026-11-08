#include "KeyPressDetector.h"

#ifdef _MSC_VER
#include <conio.h>
#include <Windows.h>

char GGL::KeyPressDetector::GetPressedChar() {
	return _getch();
}

char GGL::KeyPressDetector::TryGetPressedChar() {
	// GetAsyncKeyState works from any thread without needing a console handle.
	// We track previous state to fire once per key press rather than every frame.
	static bool prevDown[256] = {};

	// Check A-Z and 0-9 (virtual key codes match ASCII for these ranges)
	for (BYTE vk = '0'; vk <= 'Z'; vk++) {
		if (vk > '9' && vk < 'A')
			continue;
		const bool isDown = (GetAsyncKeyState(vk) & 0x8000) != 0;
		const bool wasDown = prevDown[vk];
		prevDown[vk] = isDown;
		if (isDown && !wasDown)
			return static_cast<char>(vk);
	}
	return '\0';
}

#else
#include <unistd.h>
#include <termios.h>
#include <cstdlib>

// Capture the terminal state before any modification and restore it on exit.
// This fixes the "broken terminal after training" issue on Linux: if the process
// exits (clean exit or Ctrl+C caught by the trainer's save-and-quit path) while
// termios is in raw/no-echo mode, atexit restores the original settings.
// Only registered when stdin is an actual TTY — inside Docker or when piped,
// isatty() returns false and all termios manipulation is skipped.
namespace {
	struct termios s_originalTermios = {};
	bool s_termiosSaved = false;

	void EnsureTermiosRestored() {
		if (!isatty(STDIN_FILENO))
			return;
		if (!s_termiosSaved && tcgetattr(STDIN_FILENO, &s_originalTermios) == 0) {
			s_termiosSaved = true;
			std::atexit([]() {
				tcsetattr(STDIN_FILENO, TCSANOW, &s_originalTermios);
			});
		}
	}
}

char GGL::KeyPressDetector::GetPressedChar() {
	// No TTY (e.g. Docker / piped stdin): park this thread until a signal
	// arrives. pause() is zero-CPU and returns when interrupted by a signal
	// (e.g. SIGTERM from `docker stop`). The stop handler the host application installs
	// calls RequestStop(), so we don't need to inspect the return value here.
	if (!isatty(STDIN_FILENO)) {
		pause();
		return '\0';
	}

	EnsureTermiosRestored();

	// https://stackoverflow.com/questions/421860/capture-characters-from-standard-input-without-waiting-for-enter-to-be-pressed
	char buf = 0;
	struct termios old = { 0 };
	if (tcgetattr(STDIN_FILENO, &old) < 0)
		perror("tcsetattr()");
	old.c_lflag &= ~ICANON;
	old.c_lflag &= ~ECHO;
	old.c_cc[VMIN] = 1;
	old.c_cc[VTIME] = 0;
	if (tcsetattr(STDIN_FILENO, TCSANOW, &old) < 0)
		perror("tcsetattr ICANON");
	if (read(STDIN_FILENO, &buf, 1) < 0)
		perror("read()");
	old.c_lflag |= ICANON;
	old.c_lflag |= ECHO;
	if (tcsetattr(STDIN_FILENO, TCSADRAIN, &old) < 0)
		perror("tcsetattr ~ICANON");
	return buf;
}

char GGL::KeyPressDetector::TryGetPressedChar() {
	// No TTY (e.g. Docker / piped stdin): no key presses possible.
	if (!isatty(STDIN_FILENO))
		return '\0';

	EnsureTermiosRestored();

	char buf = 0;
	struct termios old = { 0 };
	if (tcgetattr(STDIN_FILENO, &old) < 0)
		return '\0';
	struct termios raw = old;
	raw.c_lflag &= ~(ICANON | ECHO);
	raw.c_cc[VMIN] = 0;  // Return immediately even with no input
	raw.c_cc[VTIME] = 0;
	tcsetattr(STDIN_FILENO, TCSANOW, &raw);
	read(STDIN_FILENO, &buf, 1);
	tcsetattr(STDIN_FILENO, TCSADRAIN, &old);
	return buf;
}

#endif