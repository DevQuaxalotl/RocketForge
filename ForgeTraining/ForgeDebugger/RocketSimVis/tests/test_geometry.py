"""Compare optimized geometry and actual draw calls with the original scalar path."""
import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from pyrr import Vector3, Matrix44
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from debug_geometry import fill_debug_vertices, fill_ribbon_vertices
from states import DebugLine
from ribbon import RibbonEmitter, RibbonPoint


def scalar_quads(lines, camera):
    quads = []
    for line in lines:
        start, end = Vector3(line.start), Vector3(line.end)
        direction = end - start
        if direction.length < 1e-4:
            quads.append([start] * 4)
            continue
        direction = direction / direction.length
        toward = camera - (start + end) * .5
        toward = Vector3((0., 0., 1.)) if toward.length < 1e-4 else toward / toward.length
        side = direction.cross(toward)
        side = Vector3((0., 0., 10.)) if side.length < 1e-4 else side * (10. / side.length)
        quads.append([start - side, start + side, end - side, end + side])
    return np.asarray(quads, dtype='f4').reshape(-1, 4, 3)


def sample_lines(count):
    rng = np.random.default_rng(719)
    return [DebugLine(Vector3(rng.uniform(-300, 300, 3)),
                      Vector3(rng.uniform(-300, 300, 3)),
                      Vector3(rng.uniform(0, 1, 3))) for _ in range(count)]


class GeometryTests(unittest.TestCase):
    def test_batch_matches_scalar_geometry_colors_and_winding(self):
        camera = Vector3((0., 0., 400.))
        lines = sample_lines(1000) + [
            DebugLine(Vector3((1., 2., 3.)), Vector3((1., 2., 3.)), Vector3((1., 0., 0.))),
            DebugLine(Vector3((0., 0., 0.)), Vector3((0., 0., 200.)), Vector3((0., 1., 0.))),
            DebugLine(Vector3((-10., 0., 400.)), Vector3((10., 0., 400.)), Vector3((0., 0., 1.))),
        ]
        vertices = np.empty((len(lines) * 6, 6), dtype='f4')
        self.assertEqual(fill_debug_vertices(lines, camera, vertices), len(vertices))
        expected = scalar_quads(lines, camera)[:, (0, 1, 2, 2, 1, 3)]
        np.testing.assert_array_equal(vertices[:, :3].reshape(-1, 6, 3), expected)
        expected_colors = np.repeat(np.asarray([line.color for line in lines], dtype='f4'), 6, axis=0)
        np.testing.assert_array_equal(vertices[:, 3:], expected_colors)

    def test_batch_capacity_and_empty_inputs_do_not_touch_unused_storage(self):
        vertices = np.full((13, 6), -123., dtype='f4')
        self.assertEqual(fill_debug_vertices(sample_lines(10), Vector3((0., 0., 400.)), vertices), 12)
        np.testing.assert_array_equal(vertices[12], [-123.] * 6)
        before = vertices.copy()
        self.assertEqual(fill_debug_vertices([], Vector3(), vertices), 0)
        np.testing.assert_array_equal(vertices, before)
        self.assertEqual(fill_debug_vertices(sample_lines(1), Vector3(), vertices[:0]), 0)

    def test_ribbons_keep_live_strip_geometry_without_padding(self):
        camera = Vector3((0., 0., 400.))
        ribbon = RibbonEmitter()
        for index in range(70):
            point = RibbonPoint(Vector3((float(index), 0., 0.)), Vector3((10., 0., 0.)))
            point.time_active = index / 100.
            point.connected = index % 5 != 0
            ribbon.points.append(point)
        expected = []
        first = ribbon.points[0]
        sideways = first.vel.normalized.cross(-(first.pos - camera).normalized)
        for point in ribbon.points:
            if point.connected:
                scale = point.time_active / .1 if point.time_active < .1 else 1 - point.time_active
                offset = sideways * 20 * scale
                expected.extend([point.pos - offset, point.pos + offset])
        for capacity in (0, 10, 100, 150):
            with self.subTest(capacity=capacity):
                vertices = np.full((capacity, 3), -123., dtype='f4')
                count = fill_ribbon_vertices(ribbon, camera, 1., 20., .1, vertices)
                self.assertEqual(count, min(capacity, len(expected)))
                np.testing.assert_array_equal(vertices[:count], np.asarray(expected[:count], dtype='f4').reshape(-1, 3))
                self.assertTrue(np.all(vertices[count:] == -123.))
        ribbon.points.clear()
        self.assertEqual(fill_ribbon_vertices(ribbon, camera, 1., 20., .1, vertices), 0)

    def test_renderer_batches_lines_and_uploads_only_live_vertices(self):
        import main
        renderer = SimpleNamespace(last_camera_pos=Vector3((0., 0., 400.)),
            last_vp_mat=Matrix44.identity(), debug_line_vertices=np.empty((60, 6), dtype='f4'),
            debug_line_vbo=Mock(), debug_line_prog={'m_vp': Mock()}, debug_line_vao=Mock(), ctx=Mock())
        with patch.object(main, 'glDisable'), patch.object(main, 'glEnable'):
            main.QRSVGLWidget.render_debug_lines(renderer, sample_lines(3))
        renderer.debug_line_vao.render.assert_called_once_with(mode=main.moderngl.TRIANGLES, vertices=18)
        self.assertEqual(renderer.debug_line_vbo.write.call_args.args[0].nbytes, 18 * 6 * 4)
        ribbon = RibbonEmitter()
        ribbon.points.append(RibbonPoint(Vector3((0., 0., 0.)), Vector3((10., 0., 0.))))
        renderer.ribbon_verts = np.empty((100, 3), dtype='f4')
        renderer.ribbon_vbo, renderer.render_model, renderer.t_none = Mock(), Mock(), Mock()
        with patch.object(main, 'glDisable'), patch.object(main, 'glEnable'):
            main.QRSVGLWidget.render_ribbon(renderer, ribbon, renderer.last_camera_pos, 1., 20., .1, None)
        self.assertEqual(renderer.ribbon_vbo.write.call_args.args[0].nbytes, 2 * 3 * 4)
        self.assertEqual(renderer.render_model.call_args.kwargs['vertices'], 2)

    def test_model_scaling_does_not_mutate_snapshot_orientation(self):
        import main
        renderer = SimpleNamespace(pr_m_model=Mock(), pra_m_model=Mock(), pr_global_color=Mock(),
            ctx=Mock(), t_none=Mock(), vaos={'model': Mock()})
        position, forward, up = Vector3((0., 0., 0.)), Vector3((1., 0., 0.)), Vector3((0., 0., 1.))
        main.QRSVGLWidget.render_model(renderer, position, forward, up, 'model', None, scale=2.5)
        np.testing.assert_array_equal(np.asarray(forward), (1., 0., 0.))
        np.testing.assert_array_equal(np.asarray(up), (0., 0., 1.))
        renderer.vaos['model'].render.assert_called_once_with(main.moderngl.TRIANGLES, vertices=-1)


if __name__ == '__main__':
    unittest.main()
