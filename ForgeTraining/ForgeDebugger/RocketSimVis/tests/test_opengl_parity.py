"""Real offscreen OpenGL parity; set RF_REQUIRE_GL_TESTS=1 to require a driver."""
import os
import unittest
import numpy as np
import moderngl
from pyrr import Matrix44, Vector3
from test_geometry import scalar_quads, sample_lines
from debug_geometry import fill_debug_vertices
from shaders import VERT_SHADER, FRAG_SHADER, DEBUG_VERT_SHADER, DEBUG_FRAG_SHADER


class OpenGLParityTests(unittest.TestCase):
    def test_batch_pixels_match_individual_quads_with_depth_and_multisampling(self):
        try:
            ctx = moderngl.create_standalone_context(require=330)
        except Exception as error:
            if os.getenv('RF_REQUIRE_GL_TESTS') == '1':
                raise
            self.skipTest(str(error))
        try:
            old = ctx.program(vertex_shader=VERT_SHADER, fragment_shader=FRAG_SHADER)
            new = ctx.program(vertex_shader=DEBUG_VERT_SHADER, fragment_shader=DEBUG_FRAG_SHADER)
            texture = ctx.texture((1, 1), 4, bytes([255, 255, 255, 255]))
            texture.use()
            old['m_model'].write(Matrix44.identity(dtype='f4'))
            lines = sample_lines(64)
            for samples in (0, 4):
                target = ctx.simple_framebuffer((192, 192), components=4, samples=samples)
                resolved = ctx.simple_framebuffer((192, 192), components=4)
                for camera in (Vector3((0., -700., 400.)), Vector3((500., 400., 600.))):
                    with self.subTest(samples=samples, camera=list(camera)):
                        vp = Matrix44.perspective_projection(70., 1., .1, 3000.) * Matrix44.look_at(camera, (0., 0., 0.), (0., 0., 1.))
                        old['m_vp'].write(vp.astype('f4'))
                        old['cameraPos'].write(camera.astype('f4'))
                        new['m_vp'].write(vp.astype('f4'))
                        quads = scalar_quads(lines, camera)
                        vertices = np.empty((len(lines) * 6, 6), dtype='f4')
                        fill_debug_vertices(lines, camera, vertices)
                        old_buffer, new_buffer = ctx.buffer(quads), ctx.buffer(vertices)
                        old_vao = ctx.simple_vertex_array(old, old_buffer, 'in_position')
                        new_vao = ctx.vertex_array(new, [(new_buffer, '3f 3f', 'in_position', 'in_color')])
                        target.use()
                        ctx.viewport = (0, 0, 192, 192)
                        ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
                        ctx.disable(moderngl.CULL_FACE)
                        target.clear(0., 0., 0., 0.)
                        for index, line in enumerate(lines):
                            old['globalColor'].write(np.array([*line.color, 1.], dtype='f4'))
                            old_vao.render(moderngl.TRIANGLE_STRIP, vertices=4, first=index * 4)
                        ctx.copy_framebuffer(resolved, target)
                        expected = resolved.read(components=4)
                        target.clear(0., 0., 0., 0.)
                        new_vao.render(moderngl.TRIANGLES, vertices=len(vertices))
                        ctx.copy_framebuffer(resolved, target)
                        actual = resolved.read(components=4)
                        self.assertTrue(any(expected))
                        self.assertEqual(actual, expected)
                        old_vao.release(); new_vao.release()
                        old_buffer.release(); new_buffer.release()
                target.release(); resolved.release()
            texture.release(); old.release(); new.release()
        finally:
            ctx.release()


if __name__ == '__main__':
    unittest.main()
