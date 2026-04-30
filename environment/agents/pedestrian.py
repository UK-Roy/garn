"""Pedestrian agent state."""

import numpy as np


class Pedestrian:
    """Single pedestrian with personal space and group membership."""

    def __init__(self, ped_id, pos, vel, theta, group_id=0,
                 radius=0.3, v_pref=1.0,
                 ps_a=0.45, ps_b=0.25):
        self.id = ped_id
        self.pos = np.array(pos, dtype=np.float64)
        self.vel = np.array(vel, dtype=np.float64)
        self.theta = float(theta)
        self.group_id = int(group_id)  # 0 = individual
        self.radius = radius
        self.v_pref = v_pref
        self.ps_a = ps_a   # personal space Gaussian major semi-axis
        self.ps_b = ps_b   # personal space Gaussian minor semi-axis
        self.goal = None

    def set_goal(self, goal):
        self.goal = np.array(goal, dtype=np.float64)

    def update(self, new_pos, new_vel):
        self.pos = np.array(new_pos, dtype=np.float64)
        self.vel = np.array(new_vel, dtype=np.float64)
        speed = np.linalg.norm(self.vel)
        if speed > 1e-6:
            self.theta = np.arctan2(self.vel[1], self.vel[0])

    def get_partial_state(self):
        """Partial state for attention extraction: [px, py, theta, vx, vy]."""
        return np.array([self.pos[0], self.pos[1], self.theta,
                         self.vel[0], self.vel[1]], dtype=np.float32)

    def get_full_state(self):
        """
        Full state P_t^i = [px, py, theta, vx, vy, personal_space, group_id, radius].
        personal_space is encoded as (ps_a, ps_b) ellipse semi-axes.
        """
        return np.array([
            self.pos[0], self.pos[1], self.theta,
            self.vel[0], self.vel[1],
            self.ps_a, self.ps_b,
            float(self.group_id), self.radius
        ], dtype=np.float32)

    def is_in_personal_space(self, point, threshold=0.8):
        """
        Check if point is within the asymmetric Gaussian personal space.
        Returns True if Gaussian value > threshold.
        """
        from utils.transforms import gaussian_personal_space
        val = gaussian_personal_space(point, self.pos, self.theta,
                                      self.ps_a, self.ps_b)
        return val > threshold
