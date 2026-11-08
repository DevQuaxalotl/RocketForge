import copy
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from states import GameState
from state_manager import RenderState


def packet(tick):
    def physics(index):
        value = dict(pos=[tick, index * 300, 100], vel=[10, 20, 30],
                     ang_vel=[tick % 5 * .1, tick % 7 * .2, .3])
        if tick % 3 == 0:
            value.update(forward=[1, 0, 0], up=[0, 0, 1])
        return value
    return dict(ball_phys=physics(0), cars=[dict(team_num=i % 2,
        phys=physics(i), boost_amount=tick % 100, on_ground=True, is_demoed=False)
        for i in range((2, 4, 6)[tick % 3])])


def assert_physics_equal(left, right):
    for name in ("prev_pos", "next_pos", "prev_forward", "next_forward",
                 "prev_up", "next_up", "prev_vel", "next_vel", "ang_vel"):
        np.testing.assert_array_equal(np.asarray(getattr(left, name)),
                                      np.asarray(getattr(right, name)))
    assert left.has_rot == right.has_rot


class SnapshotOptimizationTests(unittest.TestCase):
    def test_render_rotation_matches_mutable_reference_with_mixed_packet_rates(self):
        reference, published = GameState(), GameState()
        display = RenderState()
        rng = np.random.default_rng(4791)
        tick = 0
        for _ in range(250):
            for _ in range(int(rng.integers(0, 8))):
                tick += 1
                data = packet(tick)
                reference.read_from_json(data)
                published = GameState.from_json(data, published)
            frozen = copy.deepcopy(published)
            if not reference.ball_state.has_rot:
                reference.ball_state.rotate_with_ang_vel(1 / 60)
            rendered = display.update(published, 1 / 60)
            assert_physics_equal(reference.ball_state, rendered.ball_state)
            assert_physics_equal(frozen.ball_state, published.ball_state)

    def test_new_and_malformed_packets_do_not_mutate_published_objects(self):
        published = GameState.from_json(packet(1), GameState())
        frozen = copy.deepcopy(published)
        for tick in range(2, 12):
            GameState.from_json(packet(tick), published)
        broken = packet(12)
        del broken["cars"][0]["phys"]
        with self.assertRaises(KeyError):
            GameState.from_json(broken, published)
        assert_physics_equal(frozen.ball_state, published.ball_state)
        self.assertEqual(len(frozen.car_states), len(published.car_states))
        for before, after in zip(frozen.car_states, published.car_states):
            assert_physics_equal(before.phys, after.phys)
            self.assertEqual(before.controls.__dict__, after.controls.__dict__)

    def test_controls_scene_metadata_and_independent_shells_survive_publication(self):
        data = packet(1)
        data.update(spectate_car_id=1, context_label="Example scene", game_speed=.5, paused=True,
                    boost_pad_locations=[[0, 0, 73]], boost_pad_states=[True],
                    debug={"rewards": [{"debug_lines": [{"start": [0, 0, 0], "end": [1, 2, 3], "color": [1, 0, 0]}]}]})
        data['cars'][0]['controls'] = dict(throttle=1, steer=0, pitch=0, yaw=0, roll=0, boost=True, jump=False, handbrake=False)
        initial = GameState()
        published = GameState.from_json(data, initial)
        self.assertEqual(initial.boost_pad_locations, [])
        display = RenderState()
        rendered = display.update(published, .01)
        for field in ('spectate_car_id', 'context_label', 'game_speed', 'paused'):
            self.assertEqual(getattr(rendered, field), data[field])
        self.assertEqual(len(rendered.debug_lines), 1)
        self.assertTrue(rendered.car_states[0].controls.boost)
        self.assertIs(display.update(published, .01), rendered)
        data['cars'][0]['controls']['boost'] = False
        data['paused'] = False
        next_packet = GameState.from_json(data, published)
        self.assertIsNot(next_packet.car_states[0].controls, published.car_states[0].controls)
        self.assertIsNot(next_packet.car_states[0].phys, published.car_states[0].phys)
        self.assertTrue(published.car_states[0].controls.boost)
        self.assertTrue(rendered.paused)
        self.assertFalse(display.update(next_packet, .01).paused)
        self.assertTrue(rendered.paused)

    def test_trail_updates_cannot_mutate_shared_teleport_positions(self):
        from ribbon import RibbonEmitter
        from pyrr import Vector3
        first = GameState.from_json(packet(1), GameState())
        data = packet(2)
        data['ball_phys']['pos'] = [9000, 0, 0]
        published = GameState.from_json(data, first)
        self.assertTrue(published.ball_state.is_teleporting())
        frozen = copy.deepcopy(published)
        ribbon = RibbonEmitter()
        ribbon.update(True, 0, published.ball_state.get_pos(.5), Vector3((100., 0., 0.)), 1., .1)
        self.assertFalse(np.shares_memory(ribbon.points[0].pos, published.ball_state.prev_pos))
        assert_physics_equal(frozen.ball_state, published.ball_state)


if __name__ == "__main__":
    unittest.main()
