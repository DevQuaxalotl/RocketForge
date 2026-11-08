"""Reusable vertex staging for independent debug quads and ribbon strips."""
from pyrr import Vector3
import numpy as np


def fill_debug_vertices(lines, camera_pos, vertices):
    count = min(len(lines), len(vertices) // 6)
    if count == 0:
        return 0
    # Do the geometry arithmetic in float64, as pyrr's scalar path did, then
    # round once into the existing float32 upload buffer.
    data = np.empty((count, 3, 3), dtype='f8')
    for i in range(count):
        data[i] = lines[i].start, lines[i].end, lines[i].color
    start, end = data[:, 0], data[:, 1]
    direction = end - start
    lengths = np.sqrt(np.sum(direction * direction, axis=1))
    degenerate = lengths < 1e-4
    direction /= np.where(degenerate, 1., lengths)[:, None]
    camera = np.asarray(camera_pos) - (start + end) * 0.5
    camera_lengths = np.sqrt(np.sum(camera * camera, axis=1))
    close = camera_lengths < 1e-4
    camera /= np.where(close, 1., camera_lengths)[:, None]
    camera[close] = (0., 0., 1.)
    side = np.cross(direction, camera)
    side_lengths = np.sqrt(np.sum(side * side, axis=1))
    parallel = side_lengths < 1e-4
    side *= (10. / np.where(parallel, 1., side_lengths))[:, None]
    side[parallel] = (0., 0., 10.)
    corners = np.stack((start - side, start + side, end - side, end + side), axis=1)
    corners[degenerate] = start[degenerate, None, :]
    output = vertices[:count * 6].reshape(count, 6, 6)
    # Preserve the two strip triangles' winding and rasterized diagonal.
    output[:, :, :3] = corners[:, (0, 1, 2, 2, 1, 3)]
    output[:, :, 3:] = data[:, 2, None, :]
    return count * 6


def fill_ribbon_vertices(ribbon, camera_pos, lifetime, width, start_taper_time, vertices):
    if not ribbon.points:
        return 0
    first = ribbon.points[0]
    cam_to_ribbon = -(first.pos - camera_pos).normalized
    sideways = first.vel.normalized.cross(cam_to_ribbon)
    count = 0
    for point in ribbon.points:
        if not point.connected:
            continue
        width_scale = (point.time_active / start_taper_time
                       if point.time_active < start_taper_time
                       else 1 - point.time_active / lifetime)
        offset = sideways * width * width_scale
        for sign in (-1, 1):
            if count == len(vertices):
                return count
            vertices[count] = point.pos + offset * sign
            count += 1
    return count
