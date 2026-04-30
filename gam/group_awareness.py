"""
Group-Awareness Module (GAM): wraps group detection and group space
representation into a single interface used by the environment and trainer.
"""

import numpy as np
from environment.groups.group_detector import GroupManager
from environment.groups.group_space import GroupSpace


class GroupAwarenessModule:
    """
    Implements the Group-Awareness Mechanism from Section III-A.
    Provides group detection, space representation, and group state for STGAN.
    """

    def __init__(self, env_cfg):
        self.group_manager = GroupManager(env_cfg)
        self.group_spaces = {}
        self._member_history = {}  # for dynamic group tracking

    def update(self, pedestrians):
        """
        Detect groups and update group spaces.
        Called every timestep from the environment.
        """
        self.group_manager.update(pedestrians)
        all_groups = self.group_manager.get_all_groups()
        ped_map = {p.id: p for p in pedestrians}

        new_spaces = {}
        for gid, member_ids in all_groups.items():
            members = [ped_map[pid] for pid in member_ids if pid in ped_map]
            if members:
                new_spaces[gid] = GroupSpace(gid, members)
        self.group_spaces = new_spaces
        return self.group_spaces

    def get_group_states(self, robot_pos, max_groups=6):
        """
        Return padded array of group partial states for attention network.
        Shape: (max_groups, 4) = [gx, gy, gvx, gvy]
        """
        states = np.zeros((max_groups, 4), dtype=np.float32)
        for i, grp in enumerate(list(self.group_spaces.values())[:max_groups]):
            states[i] = grp.get_partial_state()
        return states

    def get_group_full_states(self, robot_pos, max_groups=6):
        """
        Return padded array of full group states for relation modeling.
        Shape: (max_groups, 7)
        """
        states = np.zeros((max_groups, 7), dtype=np.float32)
        for i, grp in enumerate(list(self.group_spaces.values())[:max_groups]):
            gs = grp.get_state()
            gs[6] = float(np.linalg.norm(robot_pos - grp.center))
            states[i] = gs
        return states

    def robot_is_intruding(self, robot_pos):
        """Check if robot is inside any group's convex hull."""
        for grp in self.group_spaces.values():
            if grp.contains_point(robot_pos):
                return True, grp.group_id
        return False, None

    def approaching_groups(self, robot_pos, robot_vel):
        """Return list of group_ids for groups the robot is approaching."""
        return [
            gid for gid, grp in self.group_spaces.items()
            if grp.is_approaching(robot_pos, robot_vel)
        ]

    def n_groups(self):
        return len(self.group_spaces)
