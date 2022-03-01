import numpy as np


def convert_angle_coord(angles, distances):
    coord = np.zeros((angles.shape[0], 2))
    coord[:, 0] = distances * np.cos(angles)
    coord[:, 1] = distances * np.sin(angles)
    return coord


def unique(a):
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)
    return a[ui]


def bresenham2D(sx, sy, ex, ey):
    '''
    Bresenham's ray tracing algorithm in 2D.
    Inputs:
        (sx, sy)	start point of ray
        (ex, ey)	end point of ray
    '''
    sx = int(round(sx))
    sy = int(round(sy))
    ex = int(round(ex))
    ey = int(round(ey))
    dx = abs(ex - sx)
    dy = abs(ey - sy)
    steep = abs(dy) > abs(dx)
    if steep:
        dx, dy = dy, dx  # swap

    if dy == 0:
        q = np.zeros((dx + 1, 1))
    else:
        q = np.append(0, np.greater_equal(np.diff(np.mod(np.arange(np.floor(dx / 2), -dy * dx + np.floor(dx / 2) - 1, -dy), dx)), 0))
    if steep:
        if sy <= ey:
            y = np.arange(sy, ey + 1)
        else:
            y = np.arange(sy, ey - 1, -1)
        if sx <= ex:
            x = sx + np.cumsum(q)
        else:
            x = sx - np.cumsum(q)
    else:
        if sx <= ex:
            x = np.arange(sx, ex + 1)
        else:
            x = np.arange(sx, ex - 1, -1)
        if sy <= ey:
            y = sy + np.cumsum(q)
        else:
            y = sy - np.cumsum(q)
    return np.vstack((x, y))


def get_mapping(bot_pos, points):
    free_points = None
    block_points = None
    for p in points:
        point_space = bresenham2D(bot_pos[0], bot_pos[1], p[0], p[1]).T

        if free_points is None:
            free_points = point_space[:-1, :]
            block_points = point_space[-1:, :]
        else:
            free_points = np.concatenate((free_points, point_space[:-1, :]))
            block_points = np.concatenate((block_points, point_space[-1:, :]))
    if free_points is not None:
        new_free = unique(free_points).astype(np.int16)
    else:
        new_free = np.array([[]]).astype(np.int16)
    return new_free, block_points.astype(np.int16)


def get_valid(angles, ranges):
    indValid = np.logical_and((ranges < 60), (ranges > 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]
    return angles, ranges



