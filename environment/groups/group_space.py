"""
Group space representation: personal space (Gaussian ellipse) and
group space (convex hull of personal spaces). Paper Section III-A-2.
"""

import numpy as np
from utils.transforms import (convex_hull_2d, point_in_convex_hull,
                               dist_to_convex_hull, gaussian_personal_space)


class GroupSpace:
    """
    Represents the social space of a pedestrian group.
    - Static groups: convex hull of member positions (members face inward)
    - Dynamic groups: convex hull of member positions (shared direction)
    """

    def __init__(self, group_id, members):
        """
        group_id: int
        members: list of Pedestrian objects
        """
        self.group_id = group_id
        self.members = members
        self.hull_pts = None
        self.center = None
        self.vel = None
        self.n_members = len(members)
        self._update()

    def _update(self):
        if not self.members:
            return
        positions = np.array([m.pos for m in self.members])
        self.center = np.mean(positions, axis=0)
        vels = np.array([m.vel for m in self.members])
        self.vel = np.mean(vels, axis=0)
        self.n_members = len(self.members)
        if len(positions) >= 3:
            self.hull_pts = convex_hull_2d(positions)
        else:
            self.hull_pts = positions

    def update_members(self, members):
        self.members = members
        self._update()

    def contains_point(self, point):
        """Return True if point is inside the group convex hull."""
        if self.hull_pts is None or len(self.hull_pts) < 3:
            return False
        return point_in_convex_hull(point, self.hull_pts)

    def dist_to_boundary(self, point):
        """Shortest distance from point to group convex hull boundary."""
        if self.hull_pts is None:
            return float('inf')
        return dist_to_convex_hull(point, self.hull_pts)

    def get_state(self):
        """
        Group state GP^m_t = [gx, gy, gvx, gvy, n_members, hull_area, dist_placeholder].
        Returns float32 array of length 7.
        """
        hull_area = self._hull_area()
        return np.array([
            self.center[0], self.center[1],
            self.vel[0], self.vel[1],
            float(self.n_members),
            hull_area,
            0.0  # dist_to_robot filled in by environment
        ], dtype=np.float32)

    def get_partial_state(self):
        """Partial state for attention extraction: [gx, gy, gvx, gvy]."""
        return np.array([
            self.center[0], self.center[1],
            self.vel[0], self.vel[1]
        ], dtype=np.float32)

    def _hull_area(self):
        if self.hull_pts is None or len(self.hull_pts) < 3:
            return 0.0
        pts = self.hull_pts
        n = len(pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1]
            area -= pts[j][0] * pts[i][1]
        return abs(area) / 2.0

    def is_approaching(self, robot_pos, robot_vel, threshold=1.5):
        """True if robot is moving toward this group from the front."""
        diff = self.center - robot_pos
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            return False
        approach_dir = diff / dist
        speed = np.linalg.norm(robot_vel)
        if speed < 1e-6:
            return False
        dot = np.dot(robot_vel / speed, approach_dir)
        return dot > 0.5 and dist < threshold
