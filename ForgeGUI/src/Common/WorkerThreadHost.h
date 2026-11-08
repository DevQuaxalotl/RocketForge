#pragma once

#include <QPointer>
#include <QThread>

#include <functional>

/**
 * Owns the QThread + worker lifecycle for one background job.
 *
 * Moc-free: this is not a QObject. The host manages thread creation, start, and teardown and
 * wires the worker's standard terminal signals (Completed/Cancelled/Failed) to quit the
 * thread. The owning widget wires its own typed slots (ProgressUpdated, Completed, ...) and
 * nulls its worker pointer on QObject::destroyed.
 *
 * _thread is a QPointer, so it auto-nulls when the QThread is destroyed -- the host needs no
 * self-capturing null-out lambda, and its own lifetime is irrelevant to the queued teardown.
 *
 * The worker type W must expose signals: Completed(...), Cancelled(), Failed(...).
 */
class WorkerThreadHost
{
public:
    WorkerThreadHost() = default;
    ~WorkerThreadHost() { Stop(); }

    WorkerThreadHost(const WorkerThreadHost&) = delete;
    WorkerThreadHost& operator=(const WorkerThreadHost&) = delete;

    /**
     * Create a thread, move `worker` onto it, wire started->startFn(worker),
     * terminal signals->quit, and finished->deleteLater (worker and thread), then start.
     * Ownership of `worker` transfers to the thread. Returns `worker` for typed
     * signal wiring by the caller. Must not be called while already running.
     */
    template <class W>
    W* Start(W* worker, std::function<void(W*)> startFn)
    {
        Q_ASSERT(!IsRunning());

        _thread = new QThread();
        worker->moveToThread(_thread);

        QThread* thread = _thread;
        QObject::connect(thread, &QThread::started, worker, [worker, startFn]() { startFn(worker); });

        QObject::connect(worker, &W::Completed, thread, &QThread::quit);
        QObject::connect(worker, &W::Cancelled, thread, &QThread::quit);
        QObject::connect(worker, &W::Failed, thread, &QThread::quit);

        // finished -> deleteLater for both worker and thread, each with itself as the
        // connection context so the host's lifetime is irrelevant. _thread is a QPointer,
        // so it auto-nulls when the QThread is destroyed -- no host-captured null-out lambda.
        QObject::connect(thread, &QThread::finished, worker, &QObject::deleteLater);
        QObject::connect(thread, &QThread::finished, thread, &QObject::deleteLater);

        thread->start();
        return worker;
    }

    /** Quit and join the thread. Safe when not running. */
    void Stop()
    {
        if (!_thread)
            return;
        _thread->quit();
        _thread->wait();
    }

    bool IsRunning() const { return !_thread.isNull(); }

private:
    QPointer<QThread> _thread;
};
