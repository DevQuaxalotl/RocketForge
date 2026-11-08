#include "MainWindow/mainwindow.h"

#include <QApplication>

#include <windows.h>
#include <io.h>
#include <cstdio>

int main(int argc, char* argv[])
{
    // Allocate a hidden console so that embedded Python has valid sys.stderr/sys.stdout.
    // Without this, Win32 GUI apps set sys.stderr = None, which crashes wandb on init.
    // The console window is immediately hidden — all output is captured by the GUI log panel.
    AllocConsole();
    if (HWND consoleWnd = GetConsoleWindow())
        ShowWindow(consoleWnd, SW_HIDE);

    FILE* f;
    freopen_s(&f, "CONOUT$", "w", stdout);
    freopen_s(&f, "CONOUT$", "w", stderr);
    freopen_s(&f, "CONIN$",  "r", stdin);

    // When a launcher (e.g. an IDE) starts us with redirected std handles, AllocConsole leaves them
    // untouched and the freopen_s calls above close the handles behind them, so GetStdHandle would
    // return dangling values. Child processes (RocketSimVis, pip, wandb) inherit those std handles,
    // and a dangling one that gets recycled can abort their startup. Point them at the console.
    SetStdHandle(STD_INPUT_HANDLE,  (HANDLE)_get_osfhandle(_fileno(stdin)));
    SetStdHandle(STD_OUTPUT_HANDLE, (HANDLE)_get_osfhandle(_fileno(stdout)));
    SetStdHandle(STD_ERROR_HANDLE,  (HANDLE)_get_osfhandle(_fileno(stderr)));

    QApplication app(argc, argv);

    MainWindow w;
    w.show();

    return app.exec();
}
