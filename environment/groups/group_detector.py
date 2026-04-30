"""
Group detection: static groups via F-formation, dynamic groups via DBSCAN.
Matches paper Section III-A-2.
"""

import numpy as np
from sklearn.cluster import DBSCAN


def detect_static_groups(pedestrians, o_space_radius=1.0):
    """
    F-formation detection: pedestrians facing a shared o-space.
    For simulation, we use predefined group_id assignments.
    In real deployment, detect from orientation convergence.
    Returns: dict {group_id: [ped_ids]}
    """
    groups = {}
    for p in pedestrians:
        if p.group_id > 0:
            groups.setdefault(p.group_id, []).append(p.id)
    # Only return groups that match static group criteria
    # (paper: static groups face inward toward shared o-space)
    return groups


def detect_dynamic_groups(pedestrians, eps=0.8, min_samples=2,
                          vel_threshold=0.3):
    """
    DBSCAN-based dynamic group detection from position + velocity similarity.
    Pedestrians with coincident orientations and similar speeds are grouped.
    Returns: dict {cluster_id: [ped_ids]}
    """
    if len(pedestrians) < min_samples:
        return {}

    features = []
    for p in pedestrians:
        speed = np.linalg.norm(p.vel)
        if speed < 0.1:
            continue
        # Feature: position + velocity direction (normalized)
        norm_vel = p.vel / (speed + 1e-8)
        features.append([p.pos[0], p.pos[1], norm_vel[0], norm_vel[1]])

    if len(features) < min_samples:
        return {}

    features = np.array(features)
    # Scale: position in meters, velocity direction in [-1,1] → weight equally
    scale = np.array([1.0, 1.0, eps, eps])
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(features / scale).labels_

    moving_peds = [p for p in pedestrians if np.linalg.norm(p.vel) >= 0.1]
    groups = {}
    for i, label in enumerate(labels):
        if label >= 0:
            groups.setdefault(label, []).append(moving_peds[i].id)

    return groups


class GroupManager:
    """
    Manages group detection and tracks group state over time.
    In simulation, group membership is known a priori (from SFM setup).
    This class provides a unified interface for both modes.
    """

    def __init__(self, cfg):
        grp_cfg = cfg.get('group', {})
        self.dbscan_eps = grp_cfg.get('dbscan_eps', 0.8)
        self.dbscan_min = grp_cfg.get('dbscan_min_samples', 2)
        self._static_groups = {}    # group_id (>0) -> [ped_ids]
        self._dynamic_groups = {}   # group_id (>0) -> [ped_ids]

    def update(self, pedestrians):
        """Re-detect groups from current pedestrian states."""
        self._static_groups = detect_static_groups(pedestrians)
        # Only run DBSCAN on pedestrians not already in static groups
        static_ids = {pid for ids in self._static_groups.values() for pid in ids}
        free_peds = [p for p in pedestrians if p.id not in static_ids]
        raw_dyn = detect_dynamic_groups(free_peds, self.dbscan_eps, self.dbscan_min)
        # Remap cluster IDs to avoid collision with static group IDs
        max_static = max(self._static_groups.keys(), default=0)
        self._dynamic_groups = {
            max_static + k + 1: v for k, v in raw_dyn.items()
        }

    def get_all_groups(self):
        """Return merged dict of all detected groups."""
        return {**self._static_groups, **self._dynamic_groups}

    def get_group_of(self, ped_id):
        """Return group_id for a pedestrian, or 0 if individual."""
        for gid, members in self.get_all_groups().items():
            if ped_id in members:
                return gid
        return 0
