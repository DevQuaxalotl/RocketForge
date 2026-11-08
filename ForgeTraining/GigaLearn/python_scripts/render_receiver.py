import sys
import json
import traceback
import subprocess
import ctypes
import os
import time

import socket
import struct

# =======================
# Example implementation of render receiver, using RocketSimVis
# =======================

# Launch RocketSimVis as a subprocess.
# main.py lives at <build_root>/ForgeDebugger/RocketSimVis/src/main.py;
# this file is at <build_root>/python_scripts/render_receiver.py.
_rsvis_process = None
_rsvis_job = None  # Windows Job Object — kept alive so KILL_ON_JOB_CLOSE fires on parent exit

def _make_kill_on_close_job():
    """Create a Windows Job Object that terminates all member processes when the parent exits."""
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

    class _BasicInfo(ctypes.Structure):
        _fields_ = [
            ('PerProcessUserTimeLimit', ctypes.c_int64),
            ('PerJobUserTimeLimit',     ctypes.c_int64),
            ('LimitFlags',             ctypes.c_uint32),
            ('MinimumWorkingSetSize',  ctypes.c_size_t),
            ('MaximumWorkingSetSize',  ctypes.c_size_t),
            ('ActiveProcessLimit',     ctypes.c_uint32),
            ('Affinity',               ctypes.c_size_t),
            ('PriorityClass',          ctypes.c_uint32),
            ('SchedulingClass',        ctypes.c_uint32),
        ]

    class _ExtInfo(ctypes.Structure):
        _fields_ = [
            ('BasicLimitInformation', _BasicInfo),
            ('IoInfo',                ctypes.c_uint64 * 6),  # IO_COUNTERS
            ('ProcessMemoryLimit',    ctypes.c_size_t),
            ('JobMemoryLimit',        ctypes.c_size_t),
            ('PeakProcessMemoryUsed', ctypes.c_size_t),
            ('PeakJobMemoryUsed',     ctypes.c_size_t),
        ]

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None

    ext = _ExtInfo()
    ext.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(ext), ctypes.sizeof(ext)):
        kernel32.CloseHandle(job)
        return None

    return job

def _ensure_rsvis_deps(python_exe, requirements_txt):
    try:
        import moderngl  # Representative dep — if this imports, assume all are present
    except ImportError:
        print("Installing RocketSimVis dependencies...")
        subprocess.run(
            [python_exe, '-m', 'pip', 'install', '-r', requirements_txt, '--quiet'],
            check=True
        )

def _launch_rsvis(python_exe):
    global _rsvis_process, _rsvis_job
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_py = os.path.normpath(os.path.join(script_dir, '..', 'ForgeDebugger', 'RocketSimVis', 'src', 'main.py'))

    if not os.path.exists(main_py):
        return False

    requirements_txt = os.path.normpath(os.path.join(os.path.dirname(main_py), '..', 'requirements.txt'))
    _ensure_rsvis_deps(python_exe, requirements_txt)

    print("Launching RocketSimVis...")
    _rsvis_process = subprocess.Popen(
        [python_exe, main_py],
        cwd=os.path.dirname(main_py),  # src/ dir — needed for relative imports
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP  # Windows: clean signal isolation
    )

    # Tie the child's lifetime to the parent process via a Job Object.
    # If the parent exits for any reason, Windows closes all handles including
    # this job handle, and KILL_ON_JOB_CLOSE terminates the child automatically.
    job = _make_kill_on_close_job()
    if job:
        ctypes.WinDLL('kernel32').AssignProcessToJobObject(job, ctypes.c_void_p(_rsvis_process._handle))
        _rsvis_job = job  # hold reference so it is not garbage-collected

    time.sleep(1.5)  # wait for socket thread to bind port 9273
    print("RocketSimVis started.")
    return True

def _stop_rsvis():
    """Graceful shutdown — called explicitly from RenderSender destructor."""
    if _rsvis_process and _rsvis_process.poll() is None:
        print("Shutting down RocketSimVis...")
        _rsvis_process.terminate()
        try:
            _rsvis_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _rsvis_process.kill()

def init(python_exe):
    """Called by RenderSender with the real Python interpreter path before first use."""
    return _launch_rsvis(python_exe)

# Send to RocketSimVis
UDP_IP = "127.0.0.1"
UDP_PORT = 9273

# RocketSimVis sends control commands (focus / speed / pause) back on this port.
COMMAND_PORT = 9274

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP

_command_sock = None
_command_sock_failed = False

def _get_command_sock():
    global _command_sock, _command_sock_failed
    if _command_sock is None and not _command_sock_failed:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind((UDP_IP, COMMAND_PORT))
            s.setblocking(False)
            _command_sock = s
        except OSError as err:
            _command_sock_failed = True
            print("RocketSimVis command port {} unavailable, viewer keybinds disabled: {}".format(COMMAND_PORT, err))
    return _command_sock

def poll_commands():
    """Returns every pending viewer command datagram as a string. Never blocks."""
    s = _get_command_sock()
    if s is None:
        return []

    commands = []
    while True:
        try:
            data, _ = s.recvfrom(4096)
        except (BlockingIOError, ConnectionResetError):
            # Defensive: Windows can surface a stale ICMP error as a reset on recv; treat it as "nothing pending".
            break
        commands.append(data.decode("utf-8", errors="replace"))
    return commands

def send_data_to_rsvis(j, gamemode, debug=None, context_label="", view=None):
    json_out = {}
    json_out["gamemode"] = gamemode
    json_out["ball_phys"] = j['ball']
    json_out["ball_phys"].pop('forward')
    json_out["ball_phys"].pop('right')
    json_out["ball_phys"].pop('up')
    json_out["cars"] = []
    for player in j['players']:
        json_out["cars"].append(player)
    json_out["boost_pad_states"] = j['boost_pads']
    json_out["context_label"] = context_label

    if debug is not None:
        json_out["debug"] = debug

    if view is not None:
        if view.get("spectate_index") is not None:
            json_out["spectate_car_id"] = view["spectate_index"]
        json_out["game_speed"] = view.get("game_speed", 1.0)
        json_out["paused"] = view.get("paused", False)

    sock.sendto(json.dumps(json_out).encode(), (UDP_IP, UDP_PORT))

def render_state(state_json_str):
    j = json.loads(state_json_str)
    try:
        debug = j.get('debug', None)
        context_label = j.get('context_label', '')
        view = {
            "spectate_index": j.get('spectate_index'),
            "game_speed": j.get('game_speed', 1.0),
            "paused": j.get('paused', False),
        }
        if 'state' in j:
            send_data_to_rsvis(j['state'], j['gamemode'], debug, context_label, view)
        else:
            send_data_to_rsvis(j, debug=debug, context_label=context_label, view=view)
    except Exception as err:
        print("Exception while sending data:")
        traceback.print_exc()
