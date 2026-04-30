"""Obstacle definitions: circular and rectangular (paper Section III-A-1)."""

import numpy as np


class CircularObstacle:
    """Circular obstacle: O^k_t = [px, py, r]."""

    def __init__(self, obs_id, pos, radius):
        self.id = obs_id
        self.type = 'circle'
        self.pos = np.array(pos, dtype=np.float64)
        self.radius = float(radius)
        self.theta = 0.0
        self.vel = np.zeros(2)

    def get_partial_state(self):
        """Partial state for attention extraction: [px, py, r, 0, 0] (padded to 5D)."""
        return np.array([self.pos[0], self.pos[1], self.radius, 0.0, 0.0], dtype=np.float32)

    def get_full_state(self):
        return np.array([self.pos[0], self.pos[1], self.radius, 0.0, 0.0], dtype=np.float32)

    def dist_to_point(self, point):
        return max(0.0, np.linalg.norm(self.pos - point) - self.radius)

    def to_dict(self):
        return {'type': 'circle', 'pos': self.pos.copy(), 'radius': self.radius}


class RectangularObstacle:
    """Rectangular obstacle: O^k_t = [px, py, theta, w, l]."""

    def __init__(self, obs_id, pos, theta, width, length):
        self.id = obs_id
        self.type = 'rect'
        self.pos = np.array(pos, dtype=np.float64)
        self.theta = float(theta)
        self.width = float(width)
        self.length = float(length)
        self.vel = np.zeros(2)

    def get_partial_state(self):
        """Partial state: [px, py, theta, w, l]."""
        return np.array([self.pos[0], self.pos[1], self.theta,
                         self.width, self.length], dtype=np.float32)

    def get_full_state(self):
        return self.get_partial_state()

    def get_corners(self):
        """Return 4 corners of the rectangle in world frame."""
        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)
        hw = self.width / 2
        hl = self.length / 2
        corners_local = np.array([
            [-hw, -hl], [hw, -hl], [hw, hl], [-hw, hl]
        ])
        corners_world = []
        for c in corners_local:
            x = cos_t * c[0] - sin_t * c[1] + self.pos[0]
            y = sin_t * c[0] + cos_t * c[1] + self.pos[1]
            corners_world.append([x, y])
        return np.array(corners_world)

    def dist_to_point(self, point):
        from utils.transforms import point_to_segment_dist
        corners = self.get_corners()
        n = len(corners)
        min_dist = float('inf')
        for i in range(n):
            d = point_to_segment_dist(point, corners[i], corners[(i + 1) % n])
            min_dist = min(min_dist, d)
        return min_dist

    def to_dict(self):
        return {
            'type': 'rect', 'pos': self.pos.copy(),
            'theta': self.theta, 'width': self.width, 'length': self.length
        }
