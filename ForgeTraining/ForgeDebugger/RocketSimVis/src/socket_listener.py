import socket

import state_manager
import json

import time
import traceback

class SocketListener:
    def __init__(self):
        self.has_received: bool = False
        self.buffer_size: int = 1024 * 1024
        self.should_run = True

    def run(self, port_num: int):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('127.0.0.1', port_num))
        sock.settimeout(0.5)
        print("Created socket on port {}, listening...".format(port_num))
        prev_recv_time = time.time()
        while self.should_run:
            try:
                data, addr = sock.recvfrom(self.buffer_size)
            except:
                continue

            self.has_received = True

            try:
                j = json.loads(data.decode("utf-8"))
            except json.decoder.JSONDecodeError as err:
                print("ERROR parsing received text to JSON:", err)

                view_range = 10
                start, stop = max(0, err.pos - view_range), min(err.pos + view_range, len(err.doc) - 1)
                snippet = err.doc[start:stop].replace('\r', '').replace('\n', ' ')
                snippet_prefix = "Received JSON: "
                underline = (' ' * (len(snippet)//2 + len(snippet_prefix))) + '^ HERE'
                print("\t" + snippet_prefix + snippet)
                print("\t" + underline)
                j = None

            if not (j is None):
                recv_time = time.time()

                try:
                    with state_manager.global_state_mutex:
                        previous = state_manager.global_state_manager.state
                    state = state_manager.GameState.from_json(j, previous)
                    state.recv_time = recv_time
                    state.recv_interval = recv_time - prev_recv_time
                    # Readers retain the old object. Published snapshots are never mutated.
                    with state_manager.global_state_mutex:
                        state_manager.global_state_manager.state = state
                    prev_recv_time = recv_time
                except Exception:
                    print("ERROR reading received JSON:")
                    traceback.print_exc()


    def stop_async(self):
        self.should_run = False