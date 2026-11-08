#pragma once

#include <chrono>

namespace RocketForge::Cli {

enum class ConsoleKeyAction { None, StopQueue, SkipStage };

/** Maps a pressed console key to the queue action it triggers. */
ConsoleKeyAction ConsoleKeyActionFor(char key);

/** Counts stop requests: the first is graceful, later ones fall through to the platform default. */
bool HandleStopRequest();

/** Clears the stop request count and the finished run so tests can exercise HandleStopRequest and WaitForRunFinished repeatedly. */
void ResetStopRequestCountForTests();

/** Call once the work the handlers stop has returned, so a waiting close handler lets the process end. */
void NotifyRunFinished();

/** True once NotifyRunFinished was called, waiting up to timeout for it. */
bool WaitForRunFinished(std::chrono::milliseconds timeout);

/**
 * Linux: SIGTERM/SIGINT -> stop, SIGUSR1 -> skip. Windows: Ctrl+C / Ctrl+Break -> stop.
 * Windows close, logoff and shutdown also request the stop, then hold the process until NotifyRunFinished or about 4.5 s,
 * because the OS ends the process when the handler returns and kills it at about 5 s.
 * That save is best-effort: one that takes longer is cut off.
 */
void InstallStopAndSkipHandlers();

/** Starts the one process-wide 'Q'/'S' reader when stdin is an interactive terminal; prints the keys. */
void StartTrainingKeyListener();

}
