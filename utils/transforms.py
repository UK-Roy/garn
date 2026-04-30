"""Coordinate transforms and geometric utilities."""

import numpy as np


def world_to_robot_frame(pos_world, robot_pos, robot_theta):
    """Transform world-frame position to robot-centric frame."""
    dx = pos_world[0] - robot_pos[0]
    dy = pos_world[1] - robot_pos[1]
    cos_t = np.cos(-robot_theta)
    sin_t = np.sin(-robot_theta)
    x_r = cos_t * dx - sin_t * dy
    y_r = sin_t * dx + cos_t * dy
    return np.array([x_r, y_r])


def vel_world_to_robot_frame(vel_world, robot_theta):
    """Transform world-frame velocity to robot-centric frame."""
    cos_t = np.cos(-robot_theta)
    sin_t = np.sin(-robot_theta)
    vx_r = cos_t * vel_world[0] - sin_t * vel_world[1]
    vy_r = sin_t * vel_world[0] + cos_t * vel_world[1]
    return np.array([vx_r, vy_r])


def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def point_to_segment_dist(point, seg_start, seg_end):
    """Minimum distance from a point to a line segment."""
    px, py = point[0] - seg_start[0], point[1] - seg_start[1]
    dx, dy = seg_end[0] - seg_start[0], seg_end[1] - seg_start[1]
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-10:
        return np.hypot(px, py)
    t = max(0.0, min(1.0, (px * dx + py * dy) / seg_len_sq))
    proj_x = t * dx - px
    proj_y = t * dy - py
    return np.hypot(proj_x, proj_y)


def gaussian_personal_space(point, center, theta, a=0.45, b=0.25):
    """
    Asymmetric Gaussian personal space value at `point` relative to agent
    at `center` facing direction `theta`. Returns value in [0, 1].
    Matches paper Section III-A-2.
    """
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    # Rotate to agent frame
    x_local = cos_t * dx + sin_t * dy
    y_local = -sin_t * dx + cos_t * dy
    # Asymmetric: larger semi-axis in front
    sigma_x = a if x_local >= 0 else b
    sigma_y = b
    val = np.exp(-(x_local ** 2 / (2 * sigma_x ** 2) + y_local ** 2 / (2 * sigma_y ** 2)))
    return val


def convex_hull_2d(points):
    """Compute convex hull of 2D points. Returns hull vertices in order."""
    from scipy.spatial import ConvexHull
    points = np.array(points)
    if len(points) < 3:
        return points
    try:
        hull = ConvexHull(points)
        return points[hull.vertices]
    except Exception:
        return points


def point_in_convex_hull(point, hull_points):
    """Check if a point is inside a convex polygon (hull_points)."""
    if len(hull_points) < 3:
        return False
    from scipy.spatial import ConvexHull, Delaunay
    try:
        hull = ConvexHull(hull_points)
        deln = Delaunay(hull_points[hull.vertices])
        return deln.find_simplex(point) >= 0
    except Exception:
        return False


def dist_to_convex_hull(point, hull_points):
    """Minimum distance from a point to the boundary of a convex polygon."""
    if len(hull_points) < 2:
        if len(hull_points) == 1:
            return np.linalg.norm(np.array(point) - hull_points[0])
        return float('inf')

    hull_points = np.array(hull_points)
    n = len(hull_points)
    min_dist = float('inf')
    for i in range(n):
        seg_start = hull_points[i]
        seg_end = hull_points[(i + 1) % n]
        d = point_to_segment_dist(point, seg_start, seg_end)
        min_dist = min(min_dist, d)
    return min_dist


def action_to_velocity(action_idx, speeds, rotations, current_theta, v_pref):
    """Convert discrete action index to (vx, vy) velocity command."""
    n_rot = len(rotations)
    speed_idx = action_idx // n_rot
    rot_idx = action_idx % n_rot
    speed = speeds[speed_idx] * v_pref
    delta_theta = rotations[rot_idx]
    new_theta = current_theta + delta_theta
    vx = speed * np.cos(new_theta)
    vy = speed * np.sin(new_theta)
    return np.array([vx, vy]), new_theta
