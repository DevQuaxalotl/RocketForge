#include "ConsoleControl.h"

#include "Confirm.h"

#include <ForgeTraining.h>
#include <GigaLearnCPP/Util/KeyPressDetector.h>

#include <atomic>
#include <cctype>
#include <condition_variable>
#include <csignal>
#include <iostream>
#include <mutex>
#include <thread>

#ifdef _MSC_VER
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#endif

using namespace RocketForge::Cli;

static std::atomic<int> s_stopRequestCount{ 0 };

namespace {

	struct RunFinishedState {
		std::mutex mutex;
		std::condition_variable changed;
		bool finished = false;
	};

	// Never destroyed: a close handler may still be waiting on it while the process exits.
	RunFinishedState& RunFinished()
	{
		static RunFinishedState* state = new RunFinishedState();
		return *state;
	}

}

ConsoleKeyAction RocketForge::Cli::ConsoleKeyActionFor(char key)
{
	switch (std::toupper(static_cast<unsigned char>(key))) {
	case 'Q':
		return ConsoleKeyAction::StopQueue;
	case 'S':
		return ConsoleKeyAction::SkipStage;
	default:
		return ConsoleKeyAction::None;
	}
}

bool RocketForge::Cli::HandleStopRequest()
{
	if (s_stopRequestCount.fetch_add(1) != 0)
		return false;

	RocketForgeTraining::RequestStop();
	return true;
}

void RocketForge::Cli::ResetStopRequestCountForTests()
{
	s_stopRequestCount = 0;

	RunFinishedState& state = RunFinished();
	const std::lock_guard lock(state.mutex);
	state.finished = false;
}

void RocketForge::Cli::NotifyRunFinished()
{
	RunFinishedState& state = RunFinished();
	{
		const std::lock_guard lock(state.mutex);
		state.finished = true;
	}

	state.changed.notify_all();
}

bool RocketForge::Cli::WaitForRunFinished(std::chrono::milliseconds timeout)
{
	RunFinishedState& state = RunFinished();
	std::unique_lock lock(state.mutex);
	return state.changed.wait_for(lock, timeout, [&state] { return state.finished; });
}

#ifdef _MSC_VER

// Stays under the roughly 5 s the OS allows a handler before it kills the process.
static constexpr std::chrono::milliseconds CLOSE_SAVE_BUDGET{ 4500 };

// Returning FALSE hands the event to the next handler, which ends the process.
static BOOL WINAPI ConsoleCtrlHandler(DWORD controlType)
{
	switch (controlType) {
	case CTRL_C_EVENT:
	case CTRL_BREAK_EVENT:
		return HandleStopRequest() ? TRUE : FALSE;
	case CTRL_CLOSE_EVENT:
	case CTRL_LOGOFF_EVENT:
	case CTRL_SHUTDOWN_EVENT:
		// The process ends as soon as this returns, whatever it returns, so the save has to finish in here.
		HandleStopRequest();
		WaitForRunFinished(CLOSE_SAVE_BUDGET);
		return TRUE;
	default:
		return FALSE;
	}
}

void RocketForge::Cli::InstallStopAndSkipHandlers()
{
	SetConsoleCtrlHandler(ConsoleCtrlHandler, TRUE);
}

#else

static void StopSignalHandler(int sig)
{
	if (HandleStopRequest())
		return;

	std::signal(sig, SIG_DFL);
	std::raise(sig);
}

static void SkipSignalHandler(int /*sig*/)
{
	RocketForgeTraining::RequestSkipStage();
}

void RocketForge::Cli::InstallStopAndSkipHandlers()
{
	std::signal(SIGTERM, StopSignalHandler);
	std::signal(SIGINT, StopSignalHandler);
	std::signal(SIGUSR1, SkipSignalHandler);
}

#endif

void RocketForge::Cli::StartTrainingKeyListener()
{
	static std::once_flag started;

	std::call_once(started, [] {
		if (!StdinIsInteractive())
			return;

		std::cout << "Press 'Q' to save and stop the queue, 'S' to save and skip to the next stage.\n";

		// Detached: the blocking read cannot be interrupted, and the process outlives every stage.
		std::thread([] {
			while (true) {
				const ConsoleKeyAction action = ConsoleKeyActionFor(GGL::KeyPressDetector::GetPressedChar());

				if (action == ConsoleKeyAction::StopQueue) {
					if (HandleStopRequest())
						std::cout << "Save queued, will save and stop the queue.\n";
				} else if (action == ConsoleKeyAction::SkipStage) {
					std::cout << "Save queued, will save and skip to the next stage.\n";
					RocketForgeTraining::RequestSkipStage();
				}
			}
		}).detach();
	});
}
