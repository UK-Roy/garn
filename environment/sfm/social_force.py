"""
Extended Social Force Model for pedestrian and group simulation.
Matches the paper's "custom Python version of the extended SFM"
(paper Section IV-A-1).
"""

import numpy as np


class SocialForceModel:
    """
    Extended SFM that simulates individuals and groups.
    Groups add cohesion and alignment forces on top of standard SFM.
    """

    def __init__(self, cfg):
        sfm = cfg.get('sfm', {})
        self.tau = sfm.get('tau', 0.5)
        self.A_ped = sfm.get('A_ped', 2.0)
        self.B_ped = sfm.get('B_ped', 0.3)
        self.A_wall = sfm.get('A_wall', 10.0)
        self.B_wall = sfm.get('B_wall', 0.1)
        self.group_cohesion = sfm.get('group_cohesion', 0.5)
        self.world_size = cfg['world']['size']
        self.dt = cfg['world']['dt']

    def step(self, pedestrians, obstacles):
        """
        Update all pedestrians one timestep using SFM forces.
        pedestrians: list of Pedestrian objects
        obstacles:   list of obstacle objects
        """
        forces = {p.id: self._driving_force(p) for p in pedestrians}

        for p in pedestrians:
            for q in pedestrians:
                if p.id == q.id:
                    continue
                forces[p.id] += self._ped_repulsion(p, q)

        for p in pedestrians:
            forces[p.id] += self._boundary_force(p)

        # Group cohesion and alignment forces
        group_map = {}
        for p in pedestrians:
            if p.group_id > 0:
                group_map.setdefault(p.group_id, []).append(p)

        for gid, members in group_map.items():
            center = np.mean([m.pos for m in members], axis=0)
            avg_vel = np.mean([m.vel for m in members], axis=0)
            for p in members:
                forces[p.id] += self._group_cohesion_force(p, center)
                forces[p.id] += self._group_alignment_force(p, avg_vel)

        for p in pedestrians:
            acc = forces[p.id]
            new_vel = p.vel + acc * self.dt
            speed = np.linalg.norm(new_vel)
            if speed > p.v_pref:
                new_vel = new_vel / speed * p.v_pref
            new_pos = p.pos + new_vel * self.dt
            new_pos = np.clip(new_pos, -self.world_size / 2 + p.radius,
                              self.world_size / 2 - p.radius)
            p.update(new_pos, new_vel)

    def _driving_force(self, ped):
        """Force toward pedestrian's goal."""
        if ped.goal is None:
            return np.zeros(2)
        diff = ped.goal - ped.pos
        dist = np.linalg.norm(diff)
        if dist < 0.1:
            return np.zeros(2)
        desired_vel = (diff / dist) * ped.v_pref
        return (desired_vel - ped.vel) / self.tau

    def _ped_repulsion(self, ped_i, ped_j):
        """Repulsive force between two pedestrians."""
        diff = ped_i.pos - ped_j.pos
        dist = np.linalg.norm(diff)
        r_ij = ped_i.radius + ped_j.radius
        if dist < 1e-6:
            diff = np.random.randn(2) * 0.01
            dist = np.linalg.norm(diff)
        magnitude = self.A_ped * np.exp((r_ij - dist) / self.B_ped)
        direction = diff / dist
        return magnitude * direction

    def _boundary_force(self, ped):
        """Repulsive force from world boundaries."""
        half = self.world_size / 2
        force = np.zeros(2)
        # Left / right walls (x boundaries, centered at 0)
        dist_left = ped.pos[0] - (-half)
        dist_right = half - ped.pos[0]
        force[0] += self.A_wall * np.exp(-dist_left / self.B_wall)
        force[0] -= self.A_wall * np.exp(-dist_right / self.B_wall)
        # Bottom / top walls (y)
        dist_bottom = ped.pos[1] - 0.0
        dist_top = self.world_size - ped.pos[1]
        force[1] += self.A_wall * np.exp(-dist_bottom / self.B_wall)
        force[1] -= self.A_wall * np.exp(-dist_top / self.B_wall)
        return force

    def _group_cohesion_force(self, ped, group_center):
        """Pull group members toward the group center."""
        diff = group_center - ped.pos
        dist = np.linalg.norm(diff)
        if dist < 0.1:
            return np.zeros(2)
        return self.group_cohesion * diff / dist

    def _group_alignment_force(self, ped, avg_vel):
        """Align pedestrian velocity with group average velocity."""
        return 0.3 * (avg_vel - ped.vel)
