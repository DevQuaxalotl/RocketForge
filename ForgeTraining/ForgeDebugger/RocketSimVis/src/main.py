import os
import math
import random
import threading
import sys
import argparse
import time
import pyrr.vector3
import json
import socket

from const import *
from shaders import *
from arena_shaders import *
from socket_listener import SocketListener
from state_manager import *
from ribbon import *
from debug_geometry import fill_debug_vertices, fill_ribbon_vertices
from outline_renderer import OutlineRenderer
import ui
from ui import get_ui, QUIBarWidget, QRSVWindow
from config import Config, ConfigVal

import moderngl
import moderngl_window
import moderngl_window.loaders.scene.wavefront as wvf
from moderngl_window import resources
from moderngl_window.meta import TextureDescription

from PyQt5 import QtOpenGL, QtWidgets
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QScreen, QColor

from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np

from pyrr import Quaternion, Matrix33, Matrix44, Vector3, Vector4

import pywavefront

# RocketForge render loop listens here for viewer control commands.
COMMAND_ADDR = ("127.0.0.1", 9274)
MAX_DIRECT_SELECT_PLAYERS = 8

# TODO: Move game logic out of here
class QRSVGLWidget(QtOpenGL.QGLWidget):
    def __init__(self, screen: QScreen):

        self.config = Config()

        self.spectate_count = 0
        self.spectate_idx = 0
        self.last_spectate_car_id = None
        self.bird_cam = False
        self.command_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.prev_interp_ratio = 0
        self.car_cam_time = 0
        self.last_render_time = time.time()
        self.fps_counter = 0
        self.last_fps = 0
        self.prev_state = None # type: GameState
        self.render_state = RenderState()

        ########################################################################

        self.samples = 4

        fmt = QtOpenGL.QGLFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QtOpenGL.QGLFormat.CoreProfile)
        fmt.setDepthBufferSize(24)
        fmt.setStencilBufferSize(8)
        fmt.setDoubleBuffer(True)
        fmt.setSwapInterval(1)

        if self.samples > 1:
            fmt.setSampleBuffers(True)
            fmt.setSamples(int(self.samples))

        super(QRSVGLWidget, self).__init__(fmt, None)

        self.setMouseTracking(True)

    def load_texture_2d(self, path: str) -> moderngl.Texture:
        return resources.textures.load(TextureDescription(path=path))

    def initializeGL(self):

        self.ctx = moderngl.create_context()
        moderngl_window.activate_context(None, self.ctx)

        ##########################################

        print("Creating shader programs...")

        self.prog = self.ctx.program(
            vertex_shader=VERT_SHADER,
            fragment_shader=FRAG_SHADER,
        )

        self.prog_arena = self.ctx.program(
            vertex_shader=ARENA_VERT_SHADER,
            fragment_shader=ARENA_FRAG_SHADER,
            geometry_shader=ARENA_GEOM_SHADER
        )


        '''
        # Make destination texture for lazy supersampling
        self.super_texture = self.ctx.texture((self.window_size[0] * 2, self.window_size[1] * 2), 4)
        self.super_texture.repeat_x = False
        self.super_texture.repeat_y = False
        self.super_framebuf = self.ctx.framebuffer(
            color_attachments=[self.super_texture],
        )
        '''

        print("Creating outline renderer...")
        #self.outline_renderer = OutlineRenderer(self.ctx, (self.width(), self.height())) # TODO: Fix resizing bugs
        self.outline_renderer = None # Disabled due to weird shader compilation issues

        print("Linking shader varaibles...")
        self.pr_m_vp = self.prog['m_vp']
        self.pr_m_model = self.prog['m_model']
        self.pr_global_color = self.prog['globalColor']
        self.pr_camera_pos = self.prog['cameraPos']

        self.pra_m_vp = self.prog_arena['m_vp']
        self.pra_m_model = self.prog_arena['m_model']
        self.pra_ball_pos = self.prog_arena['ballPos']

        ##########################################

        self.ball_ribbon = RibbonEmitter()
        self.car_ribbons = []

        print("Data path:", DATA_DIR_PATH)
        print("Loading models and textures...")

        self.vaos = {}
        self.load_vao("ArenaMeshCustom.obj", self.prog_arena)
        self.load_vao("ArenaMeshHoops.obj", self.prog_arena)

        self.load_vao("Octane.obj")
        self.load_vao("Ball.obj")

        self.load_vao("BoostPad_Small_0.obj")
        self.load_vao("BoostPad_Small_1.obj")
        self.load_vao("BoostPad_Big_0.obj")
        self.load_vao("BoostPad_Big_1.obj")

        self.ts_octane = [
            self.load_texture_2d(DATA_DIR_PATH + "T_Octane_B.png"),
            self.load_texture_2d(DATA_DIR_PATH + "T_Octane_O.png")
        ]
        self.t_ball = self.load_texture_2d(DATA_DIR_PATH + "T_Ball.png")
        self.t_boostpad = self.load_texture_2d(DATA_DIR_PATH + "T_BoostPad.png")
        self.t_boost_glow = self.load_texture_2d(DATA_DIR_PATH + "T_Boost_Glow.png")
        self.t_black = self.load_texture_2d(DATA_DIR_PATH + "T_Black.png")
        self.t_none = self.load_texture_2d(DATA_DIR_PATH + "T_None.png")

        ############################################

        # Make ribbon mesh
        self.ribbon_max_verts = 100
        self.ribbon_verts = np.empty((self.ribbon_max_verts, 3), dtype='f4')
        self.ribbon_vbo = self.ctx.buffer(reserve=self.ribbon_verts.nbytes, dynamic=True)
        self.ribbon_vao = self.ctx.simple_vertex_array(self.prog, self.ribbon_vbo, "in_position")
        self.vaos['ribbon'] = self.ribbon_vao

        self.debug_line_max_lines = 8192
        self.debug_line_vertices = np.empty((self.debug_line_max_lines * 6, 6), dtype='f4')
        self.debug_line_vbo = self.ctx.buffer(reserve=self.debug_line_vertices.nbytes, dynamic=True)
        self.debug_line_prog = self.ctx.program(
            vertex_shader=DEBUG_VERT_SHADER, fragment_shader=DEBUG_FRAG_SHADER)
        self.debug_line_vao = self.ctx.vertex_array(
            self.debug_line_prog,
            [(self.debug_line_vbo, '3f 3f', 'in_position', 'in_color')])

        ############################################

        # Auto-enable multisampling if we have multiple samples
        self.ctx.multisample = self.samples > 1

        ############################################

        print("Done.")

    def load_vao(self, model_name, program = None):
        loader = wvf.Loader(wvf.SceneDescription(path = DATA_DIR_PATH + "/" + model_name))
        model = loader.load()
        self.vaos[model_name] = model.root_nodes[0].mesh.vao.instance(self.prog if (program is None) else program)
        if not (self.outline_renderer is None):
            self.outline_renderer.load_vao(model_name, model)

    def render_model(self, pos, forward, up, model_name, texture, scale = 1.0, global_color = None, mode = moderngl.TRIANGLES, outline_color: Vector3 = None, vertices=-1):
        if pos is None:
            model_mat = Matrix44.identity()
        else:
            pos = Vector3(pos)
            # Scaling must not modify the shared snapshot orientation.
            forward = Vector3(forward).copy()
            up = Vector3(up).copy()
            right = Vector3(pyrr.vector3.cross(forward, up))

            forward *= scale
            right *= scale
            up *= scale

            model_mat = Matrix44([
                forward[0], forward[1], forward[2], 0,
                -right[0], -right[1], -right[2], 0,
                up[0], up[1], up[2], 0,
                pos[0], pos[1], pos[2], 1
            ])

        self.pr_m_model.write(model_mat.astype('f4'))
        self.pra_m_model.write(model_mat.astype('f4'))

        if global_color is None:
            global_color = Vector4((0, 0, 0, 0))
        self.pr_global_color.write(global_color.astype('f4'))

        if texture is not None:
            texture.use()
        else:
            self.t_none.use()

        self.ctx.screen.use()
        self.vaos[model_name].render(mode, vertices=vertices)

        if outline_color is not None:
            self.outline_renderer.use_framebuf()
            self.outline_renderer.pr_m_model.write(model_mat.astype('f4'))
            self.outline_renderer.pr_color.write(Vector4((outline_color.x, outline_color.y, outline_color.z, 1)).astype('f4'))
            self.outline_renderer.vaos[model_name].render(mode)

    def render_ribbon(self, ribbon: RibbonEmitter, camera_pos, lifetime, width, start_taper_time, color):
        count = fill_ribbon_vertices(
            ribbon, camera_pos, lifetime, width, start_taper_time, self.ribbon_verts)
        if count == 0:
            return
        self.ribbon_vbo.write(self.ribbon_verts[:count], 0)
        glDisable(GL_CULL_FACE)
        self.render_model(None, None, None, "ribbon", self.t_none, 20, color,
                          moderngl.TRIANGLE_STRIP, vertices=count)
        glEnable(GL_CULL_FACE)

    def render_debug_lines(self, debug_lines):
        camera_pos = getattr(self, 'last_camera_pos', None)
        if not debug_lines or camera_pos is None:
            return
        count = fill_debug_vertices(debug_lines, camera_pos, self.debug_line_vertices)
        self.debug_line_vbo.write(self.debug_line_vertices[:count])
        self.debug_line_prog['m_vp'].write(self.last_vp_mat.astype('f4'))
        self.ctx.screen.use()
        glDisable(GL_CULL_FACE)
        self.debug_line_vao.render(mode=moderngl.TRIANGLES, vertices=count)
        glEnable(GL_CULL_FACE)

    def calc_camera_state(self, state, interp_ratio, delta_time):
        pos = Vector3((-4000, 0, 1000))
        ball_pos = state.ball_state.get_pos(interp_ratio)

        cam_dir = (ball_pos - pos).normalized

        is_spectating_car = self.spectate_idx > -1 and len(state.car_states) > self.spectate_idx and not self.bird_cam
        if is_spectating_car:
            car_pos = state.car_states[self.spectate_idx].phys.get_pos(interp_ratio)
            car_vel = state.car_states[self.spectate_idx].phys.get_vel(interp_ratio)
            car_forward = state.car_states[self.spectate_idx].phys.get_forward(interp_ratio)

            # Calculate ball cam
            if True:
                height = self.config.camera_height.val
                dist = self.config.camera_distance.val

                ball_cam_offset_dir = ((ball_pos - car_pos).normalized * Vector3((1, 1, 0))).normalized

                # As we tilt up, move the camera down
                lean_scale = (ball_pos - car_pos).normalized.z
                height_clamp = abs(ball_pos.z - car_pos.z) / self.config.camera_lean_min_height_clamp.val
                if lean_scale > 0:
                    height *= 1 - min(lean_scale * self.config.camera_lean_height_scale.val, height_clamp)

                    # As we tilt up, move the camera closer
                    dist *= 1 - lean_scale * self.config.camera_lean_dist_scale.val

                ball_cam_offset = -ball_cam_offset_dir * dist
                ball_cam_offset.z += height

                # Make sure we are actually of the correct distance
                ball_cam_offset = ball_cam_offset.normalized * dist
                ball_cam_pos = car_pos + ball_cam_offset
                ball_cam_dir = (ball_pos - ball_cam_pos).normalized

            # Calculate car cam dir
            if True:
                if car_vel.length > 0:
                    car_cam_dir = (car_vel * Vector3((1, 1, 0))).normalized
                else:
                    car_cam_dir = (car_forward * Vector3((1, 1, 0)))
                    if car_cam_dir.length > 0:
                        car_cam_dir = car_cam_dir.normalized
                    else:
                        car_cam_dir = Vector3((1, 1, 0))
                car_cam_offset = -car_cam_dir * self.config.camera_distance.val
                car_cam_offset.z = self.config.camera_height.val
                car_cam_pos = car_pos + car_cam_offset

            # Determine if dribbling
            car_ball_delta = ball_pos - car_pos
            dribbling = False
            if car_vel.length > 500:
                if 90 < car_ball_delta.z < 200:
                    if (car_ball_delta * Vector3((1, 1, 0))).length < 135:
                        if ball_pos.z < 300:
                            dribbling = True

            car_cam_start_delay = 0.25
            car_cam_max_time = 0.65
            car_cam_inc_speed = 0.8
            car_cam_dec_speed = 0.5
            if car_ball_delta.length > 1000:
                car_cam_dec_speed *= 2 # Speed up when ball moved away quickly

            if dribbling:
                self.car_cam_time += car_cam_inc_speed * delta_time
            else:
                self.car_cam_time = min(self.car_cam_time, car_cam_max_time)
                self.car_cam_time = max(0, self.car_cam_time - car_cam_dec_speed * delta_time)

            # TODO: Auto car-cam is annoying, activates at bad times
            car_cam_ratio = np.clip((self.car_cam_time - car_cam_start_delay) / (car_cam_max_time - car_cam_start_delay), 0, 1)

            pos = ball_cam_pos*(1-car_cam_ratio) + car_cam_pos*car_cam_ratio
            cam_dir = (ball_cam_dir*(1-car_cam_ratio) + car_cam_dir*car_cam_ratio).normalized
        else:
            self.car_cam_time = 0

        return pos, pos + cam_dir, (self.config.camera_fov.val if is_spectating_car else self.config.camera_bird_fov.val)

    def paintGL(self):
        width, height = self.width(), self.height()
        self.ctx.viewport = (0, 0, width, height)

        cur_time = time.time()
        delta_time = cur_time - self.last_render_time
        self.render(cur_time, delta_time, width, height)

        self.fps_counter += 1

        if int(cur_time) > int(self.last_render_time):
            self.last_fps = self.fps_counter
            self.fps_counter = 2

        self.last_render_time = cur_time

        self.update()

    def render(self, total_time, delta_time, width, height):
        with global_state_mutex:
            published = global_state_manager.state
        state = self.render_state.update(published, delta_time)
        self.prev_state = state
        self.spectate_count = len(state.car_states)

        # Update spectate index from spectate_car_id if provided. Only apply it on change from the
        # last received value, or the echo of our own focus commands would keep overwriting a newer
        # local selection (request_focus) before that echo catches up.
        if (state.spectate_car_id is not None
                and 0 <= state.spectate_car_id < len(state.car_states)
                and state.spectate_car_id != self.last_spectate_car_id):
            self.spectate_idx = state.spectate_car_id
            self.last_spectate_car_id = state.spectate_car_id

        while len(self.car_ribbons) != len(state.car_states):
            if len(self.car_ribbons) < len(state.car_states):
                self.car_ribbons.append(RibbonEmitter())
            else:
                self.car_ribbons.pop()

        cur_time = time.time()
        interp_interval = max(state.recv_interval, 1e-6)
        interp_ratio = min(max((cur_time - state.recv_time) / interp_interval, 0), 1)

        if not (self.outline_renderer is None):
            self.outline_renderer.clear()

        self.ctx.clear(0, 0, 0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)

        self.ctx.cull_face = "back"
        self.ctx.front_face = "cw"
        self.ctx.enable(moderngl.CULL_FACE)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) # Normal blending
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR) # Use linear interpolation of pixels for supersampling

        camera_pos, camera_target_pos, camera_fov = self.calc_camera_state(state, interp_ratio, delta_time)
        proj = Matrix44.perspective_projection(camera_fov, -width/height, 0.1, 50 * 1000.0)
        lookat = Matrix44.look_at(
            camera_pos,
            camera_target_pos,
            (0.0, 0.0, 1.0),
        )

        m_vp = proj * lookat
        self.last_vp_mat = m_vp
        self.last_camera_pos = camera_pos
        self.pr_camera_pos.write(camera_pos.astype('f4'))
        self.pr_m_vp.write(m_vp.astype('f4'))
        self.pra_m_vp.write(m_vp.astype('f4'))
        if not (self.outline_renderer is None):
            self.outline_renderer.pr_m_vp.write((proj * lookat).astype('f4'))

        if not (state.boost_pad_states is None) and state.gamemode != "heatseeker": # Render boost pads
            for i in range(len(state.boost_pad_states)):
                is_big = state.is_boost_big(i)

                is_active = state.boost_pad_states[i]

                pos = state.boost_pad_locations[i].copy()
                pos.z = 0

                model_name = "BoostPad"
                model_name += "_Big_" if is_big else "_Small_"
                model_name += "1" if is_active else "0"
                model_name += ".obj"

                self.render_model(
                    pos, Vector3((1, 0, 0)), Vector3((0, 0, 1)),
                    model_name,
                    self.t_boostpad,
                    2.5,
                )


        if True: # Render ball
            ball_phys = state.ball_state
            ball_pos = ball_phys.get_pos(interp_ratio)
            self.render_model(
                ball_pos,
                ball_phys.get_forward(interp_ratio), ball_phys.get_up(interp_ratio), 'Ball.obj', self.t_ball,

                #outline_color = Vector4((1, 1, 1, 1))
            )

            if state.gamemode == "heatseeker": # Update and render ball ribbon
                ball_speed = ball_phys.get_vel(interp_ratio).length
                speed_frac = (max(0, min(1, ball_speed / 2800)) ** 2)
                ribbon_alpha = 0.75
                ribbon_lifetime = 0.8

                self.ball_ribbon.update(
                    ball_speed > 600,
                    0,
                    ball_pos,
                    Vector3((100,0,0)),
                    ribbon_lifetime,
                    delta_time
                )

                if ball_phys.is_teleporting():
                    self.ball_ribbon.points.clear()

                self.render_ribbon(
                    self.ball_ribbon,
                    camera_pos,
                    ribbon_lifetime,
                    50,
                    ribbon_lifetime / 10,
                    Vector4((1, 1, 1, ribbon_alpha))
                )

        if True: # Render cars
            for i in range(len(state.car_states)):
                car_state = state.car_states[i]
                if car_state.is_demoed:
                    continue

                car_ribbon = self.car_ribbons[i]

                car_pos = car_state.phys.get_pos(interp_ratio)
                car_forward = car_state.phys.get_forward(interp_ratio)
                car_up = car_state.phys.get_up(interp_ratio)
                outline_brightness = 1 - (1 / (1 + (car_pos - camera_pos).length / 1000))
                self.render_model(
                    car_pos,
                    car_forward, car_up, 'Octane.obj', self.ts_octane[car_state.team_num],

                    #outline_color = (Vector3((0, 0.5, 1)) if (car_state.team_num == 0) else Vector3((1, 0.5, 0))) * outline_brightness
                )

                if True: # Update and render car ribbon
                    RIBBON_LIFETIME = 0.3
                    ribbon_emit_pos = car_pos - (car_forward * 40) + (car_up * 10)
                    ribbon_vel = car_forward * -100
                    car_ribbon.update(
                        car_state.is_boosting,
                        0,
                        ribbon_emit_pos,
                        ribbon_vel,
                        RIBBON_LIFETIME,
                        delta_time
                    )

                    if car_state.phys.is_teleporting():
                        car_ribbon.points.clear()

                    self.render_ribbon(
                        car_ribbon,
                        camera_pos,
                        RIBBON_LIFETIME,
                        20,
                        RIBBON_LIFETIME / 10,
                        Vector4((1, 0.9, 0.4, 1))
                    )


        self.render_debug_lines(state.debug_lines)

        self.pra_ball_pos.write(state.ball_state.get_pos(interp_ratio).astype('f4'))

        if state.gamemode == "hoops":
            self.render_model(None, None, None, 'ArenaMeshHoops.obj', self.t_none, 1, Vector4((1,1,1,1)))
        else:
            self.render_model(None, None, None, 'ArenaMeshCustom.obj', self.t_none, 1, Vector4((1,1,1,1)))
            

        if not (self.outline_renderer is None):
            self.outline_renderer.render_quad()

        ###########################################

        ui_text = ""
        ui_text += "Render FPS: {}".format(self.last_fps) + "\n"
        ui_text += "Connected: {}".format(state.recv_time > 0) + "\n"
        if state.context_label:
            ui_text += state.context_label + "\n"
        if state.recv_interval > 0:
            ui_text += "Network rate: {:.2f}fps".format(1 / state.recv_interval) + "\n"
        ui_text += "Ball speed: {:.2f}kph".format(state.ball_state.prev_vel.length * (9 / 250)) + "\n"
        if state.game_speed is not None:
            ui_text += "Speed: {:.2f}x{}".format(state.game_speed, "  PAUSED" if state.paused else "") + "\n"
        if self.bird_cam:
            ui_text += "Bird cam" + "\n"
        ui_text += "Focus keys: ←/→ Tab 0-7, P nearest ball, B bird cam" + "\n"
        ui_text += "Time keys: -/= speed, Bksp 1x, Space pause" + "\n"
        get_ui().set_text(ui_text)

        ###########################################

        self.prev_interp_ratio = interp_ratio

    ####################################################

    def send_command(self, command):
        try:
            self.command_sock.sendto(json.dumps(command).encode(), COMMAND_ADDR)
        except OSError:
            pass  # Commands are best-effort; the viewer works without a trainer listening.

    def request_focus(self, index):
        if self.spectate_count == 0:
            return

        self.bird_cam = False
        self.spectate_idx = index % self.spectate_count
        self.send_command({"cmd": "focus_set", "index": self.spectate_idx})

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.request_focus(self.spectate_idx + 1)

    def keyPressEvent(self, event):
        key = event.key()
        shift = bool(event.modifiers() & Qt.ShiftModifier)

        if key in (Qt.Key_Right, Qt.Key_Tab) and not shift:
            self.request_focus(self.spectate_idx + 1)
        elif key in (Qt.Key_Left, Qt.Key_Backtab) or (key == Qt.Key_Tab and shift):
            self.request_focus(self.spectate_idx - 1)
        elif Qt.Key_0 <= key < Qt.Key_0 + MAX_DIRECT_SELECT_PLAYERS:
            index = key - Qt.Key_0
            if index < self.spectate_count:
                self.request_focus(index)
        elif key == Qt.Key_P:
            self.focus_closest_to_ball()
        elif key == Qt.Key_B:
            self.bird_cam = not self.bird_cam
        elif key in (Qt.Key_Equal, Qt.Key_Plus):
            self.send_command({"cmd": "speed_step", "dir": 1})
        elif key in (Qt.Key_Minus, Qt.Key_Underscore):
            self.send_command({"cmd": "speed_step", "dir": -1})
        elif key == Qt.Key_Backspace:
            self.send_command({"cmd": "speed_reset"})
        elif key == Qt.Key_Space:
            self.send_command({"cmd": "pause_toggle"})

    def focus_closest_to_ball(self):
        if self.prev_state is None:
            return

        closest_idx = -1
        closest_dist = 100_000
        for i in range(len(self.prev_state.car_states)):
            player = self.prev_state.car_states[i] # type: CarState
            dist_to_ball = (player.phys.next_pos - self.prev_state.ball_state.next_pos).length
            if dist_to_ball < closest_dist:
                closest_idx = i
                closest_dist = dist_to_ball

        if closest_idx >= 0:
            self.request_focus(closest_idx)


g_socket_listener = None
def run_socket_thread(port):
    global g_socket_listener
    g_socket_listener = SocketListener()
    g_socket_listener.run(port)

def main():
    #cmd_args = arg_parser.parse_args()
    port = 9273
    
    print("Starting RocketSimVis...")

    print("Starting socket thread...")
    socket_thread = threading.Thread(target=run_socket_thread, args=(int(port),))
    socket_thread.start()

    print("Starting visualizer window...")

    app = QtWidgets.QApplication([])
    ui.update_scaling_factor(app)

    window = QRSVWindow(QRSVGLWidget(app.primaryScreen()))
    window.showNormal()
    app.exec_()

    print("Shutting down...")
    g_socket_listener.stop_async()
    exit()

if __name__ == "__main__":
    main()
