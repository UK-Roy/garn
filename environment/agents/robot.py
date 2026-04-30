"""Robot agent state and kinematics."""

import numpy as np
from utils.transforms import wrap_angle


class Robot:
    """Differential-drive robot with holonomic velocity control (paper assumption)."""

    def __init__(self, cfg):
        self.radius = cfg['robot']['radius']
        self.v_pref = cfg['robot']['v_pref']
        self.start = np.array(cfg['robot']['start'], dtype=np.float64)
        self.goal = np.array(cfg['robot']['goal'], dtype=np.float64)
        self.dt = cfg['world']['dt']

        self.pos = self.start.copy()
        self.vel = np.zeros(2)
        self.theta = 0.0

    def reset(self, start=None, goal=None):
        if start is not None:
            self.start = np.array(start, dtype=np.float64)
        if goal is not None:
            self.goal = np.array(goal, dtype=np.float64)
        self.pos = self.start.copy()
        self.vel = np.zeros(2)
        self.theta = np.arctan2(self.goal[1] - self.start[1],
                                self.goal[0] - self.start[0])

    def step(self, action_vel):
        """Apply velocity action and advance one timestep."""
        self.vel = np.array(action_vel, dtype=np.float64)
        speed = np.linalg.norm(self.vel)
        if speed > self.v_pref:
            self.vel = self.vel / speed * self.v_pref
        self.pos = self.pos + self.vel * self.dt
        if speed > 1e-6:
            self.theta = np.arctan2(self.vel[1], self.vel[0])

    def get_goal_dist(self):
        return np.linalg.norm(self.goal - self.pos)

    def get_goal_dir(self):
        """Unit vector from robot to goal."""
        diff = self.goal - self.pos
        norm = np.linalg.norm(diff)
        if norm < 1e-8:
            return np.zeros(2)
        return diff / norm

    def get_partial_state(self):
        """Partial state for attention extraction: [px, py, theta, vx, vy]."""
        return np.array([self.pos[0], self.pos[1], self.theta,
                         self.vel[0], self.vel[1]], dtype=np.float32)

    def get_full_state(self):
        """
        Full state S_t = [px, py, theta, vx, vy, r, vpref, dgx, dgy].
        dgx, dgy: goal vector components.
        """
        goal_vec = self.goal - self.pos
        return np.array([
            self.pos[0], self.pos[1], self.theta,
            self.vel[0], self.vel[1],
            self.radius, self.v_pref,
            goal_vec[0], goal_vec[1]
        ], dtype=np.float32)
