import copy
import json
import pathlib
import sys
import threading
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from states import GameState
import state_manager
from socket_listener import SocketListener
from test_snapshot_optimizations import packet, assert_physics_equal


class SocketPublicationTests(unittest.TestCase):
    def test_parse_is_outside_lock_and_invalid_packet_never_replaces_good_scene(self):
        listener = SocketListener()
        lock = threading.Lock()
        manager = state_manager.StateManager()
        data = packet(1)
        broken = copy.deepcopy(data)
        broken['ball_phys']['pos'] = [999, 999, 999]
        del broken['cars'][0]['phys']
        packets = [data, broken, packet(4)]
        observations = []
        sock = Mock()
        def receive(_):
            observations.append(manager.state)
            item = packets.pop(0)
            if not packets:
                listener.should_run = False
            return json.dumps(item).encode(), ('127.0.0.1', 10000)
        sock.recvfrom.side_effect = receive
        parse = GameState.from_json
        def checked_parse(data, previous):
            self.assertFalse(lock.locked())
            return parse(data, previous)
        with patch('socket_listener.socket.socket', return_value=sock), \
             patch.object(state_manager, 'global_state_manager', manager), \
             patch.object(state_manager, 'global_state_mutex', lock), \
             patch.object(GameState, 'from_json', side_effect=checked_parse), \
             patch('socket_listener.traceback.print_exc'), patch('builtins.print'):
            listener.run(10000)
        self.assertTrue(listener.has_received)
        self.assertIs(observations[1], observations[2])
        assert_physics_equal(observations[1].ball_state, parse(data, GameState()).ball_state)
        self.assertIsNot(manager.state, observations[2])
        self.assertGreater(manager.state.recv_time, 0)


if __name__ == '__main__':
    unittest.main()
