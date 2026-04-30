"""
Adapter: CrowdSimVarNum observation dict → GARN 8-key net_obs dict.

Usage:
    adapter = CustomEnvAdapter(speeds=[0.0, 0.5, 1.0],
                               rotations=[-0.5236, -0.2618, 0.0, 0.2618, 0.5236],
                               max_obstacles=12, max_pedestrians=20, max_groups=3)
    net_obs = adapter.obs_to_net(custom_obs)
    vx, vy  = adapter.action_to_vel(action_idx, robot_theta)

Custom env obs keys:
    robot_node:        (1,7) [px, py, r, gx, gy, v_pref, theta]
    temporal_edges:    (2,)  [vx, vy]  — robot's absolute velocity
    spatial_edges:     (20,2) [px_rel, py_rel]   sentinel=15.0 if invisible
    velocity_edges:    (20,2) [vx_rel, vy_rel]   sentinel=15.0 if invisible
    visible_masks:     (20,) bool
    detected_human_num:(1,) float32
    clusters:          (20,) int32  (-1 = individual)
    group_members:     dict  {group_id: [human_ids]}
    group_centroids:   (3,2) [cx,cy]  sentinel=15.0 if no group
    group_radii:       (3,)  float64  sentinel=15.0 if no group
    grp:               bool

GARN net_obs keys (8+3 counts):
    robot_partial: (5,)       [px, py, theta, vx, vy]
    obs_partial:   (K,5)      zero (obstacles not separated in custom env)
    ped_partial:   (I,5)      [px_rel, py_rel, theta_est, vx_rel, vy_rel]
    grp_partial:   (M,4)      [gx, gy, gvx, gvy]
    robot_full:    (9,)       [px, py, theta, vx, vy, r, vpref, dgx, dgy]
    obs_full:      (K,5)      zero
    ped_full:      (I,9)      [px, py, theta, vx, vy, ps_a, ps_b, group_id, r]
    grp_full:      (M,7)      [gx, gy, gvx, gvy, n_members, hull_area, dist_to_robot]
    n_o, n_p, n_g: int counts
"""

import numpy as np

SENTINEL = 15.0
PED_RADIUS = 0.3
PED_PS_A = 0.45   # personal space major semi-axis (match GARN pedestrian default)
PED_PS_B = 0.25   # personal space minor semi-axis


class CustomEnvAdapter:
    """Translates CrowdSimVarNum obs ↔ GARN net_obs and action conventions."""

    def __init__(self, speeds, rotations,
                 max_obstacles=12, max_pedestrians=20, max_groups=3):
        self.speeds = speeds
        self.rotations = rotations
        self.n_actions = len(speeds) * len(rotations)
        self.max_obstacles = max_obstacles
        self.max_pedestrians = max_pedestrians
        self.max_groups = max_groups

    # ------------------------------------------------------------------
    # Observation translation
    # ------------------------------------------------------------------

    def obs_to_net(self, obs):
        """
        obs: dict returned by custom env's reset() or step()
        Returns: net_obs dict expected by STGAN.forward() / model.get_action()
        """
        rn = np.asarray(obs['robot_node']).flatten()     # (7,)
        te = np.asarray(obs['temporal_edges']).flatten() # (2,)
        se = np.asarray(obs['spatial_edges'])            # (20,2)
        ve = np.asarray(obs['velocity_edges'])           # (20,2)
        vm = np.asarray(obs['visible_masks']).flatten()  # (20,) bool
        cl = np.asarray(obs['clusters']).flatten()       # (20,) int
        gc = np.asarray(obs['group_centroids'])          # (3,2)
        gr = np.asarray(obs['group_radii']).flatten()    # (3,)
        gm = obs.get('group_members', {})                # dict

        # Robot absolute position and velocity
        robot_px, robot_py = rn[0], rn[1]
        robot_r            = rn[2]
        robot_gx, robot_gy = rn[3], rn[4]
        robot_vpref        = rn[5]
        robot_theta        = rn[6]
        robot_vx, robot_vy = te[0], te[1]

        # --- robot_partial (5,) ---
        robot_partial = np.array(
            [robot_px, robot_py, robot_theta, robot_vx, robot_vy],
            dtype=np.float32)

        # --- robot_full (9,) ---
        dgx = robot_gx - robot_px   # goal vector x
        dgy = robot_gy - robot_py   # goal vector y
        robot_full = np.array(
            [robot_px, robot_py, robot_theta, robot_vx, robot_vy,
             robot_r, robot_vpref, dgx, dgy],
            dtype=np.float32)

        # --- obstacles: not explicitly separated in custom env → zero (n_o=0) ---
        obs_partial = np.zeros((self.max_obstacles, 5), dtype=np.float32)
        obs_full    = np.zeros((self.max_obstacles, 5), dtype=np.float32)
        n_o = 0

        # --- pedestrians ---
        ped_partial = np.zeros((self.max_pedestrians, 5), dtype=np.float32)
        ped_full    = np.zeros((self.max_pedestrians, 9), dtype=np.float32)
        n_visible = 0

        for i in range(min(len(vm), self.max_pedestrians)):
            if not vm[i]:
                continue  # invisible — leave as zero (masked out by attention)

            px_rel = se[i, 0]
            py_rel = se[i, 1]
            vx_rel = ve[i, 0]
            vy_rel = ve[i, 1]

            # Skip if sentinel slipped through (shouldn't happen if vm is correct)
            if abs(px_rel) >= SENTINEL or abs(py_rel) >= SENTINEL:
                continue

            # Estimate heading from relative velocity direction
            speed_rel = np.hypot(vx_rel, vy_rel)
            theta_est = np.arctan2(vy_rel, vx_rel) if speed_rel > 0.05 else 0.0

            ped_partial[i] = [px_rel, py_rel, theta_est, vx_rel, vy_rel]

            # Absolute position = robot pos + relative offset
            px_abs = robot_px + px_rel
            py_abs = robot_py + py_rel
            vx_abs = robot_vx + vx_rel
            vy_abs = robot_vy + vy_rel
            group_id = float(cl[i]) if cl[i] >= 0 else 0.0

            ped_full[i] = [px_abs, py_abs, theta_est, vx_abs, vy_abs,
                           PED_PS_A, PED_PS_B, group_id, PED_RADIUS]
            n_visible += 1

        n_p = n_visible

        # --- groups ---
        grp_partial = np.zeros((self.max_groups, 4), dtype=np.float32)
        grp_full    = np.zeros((self.max_groups, 7), dtype=np.float32)
        n_g = 0

        for m in range(min(len(gr), self.max_groups)):
            cx, cy = gc[m, 0], gc[m, 1]
            radius = gr[m]

            if abs(cx) >= SENTINEL or abs(cy) >= SENTINEL:
                continue  # group not detected this step

            # group_id in gm dict is 0-indexed key
            group_key = m  # centroids are ordered by group index
            member_ids = gm.get(group_key, [])
            n_members = len(member_ids)

            # Approximate hull area as circle: π r²
            hull_area = np.pi * radius ** 2

            dist_to_robot = np.hypot(cx - robot_px, cy - robot_py)

            # Group velocity not available in custom env obs → 0
            grp_partial[m] = [cx, cy, 0.0, 0.0]
            grp_full[m]    = [cx, cy, 0.0, 0.0, float(n_members), hull_area, dist_to_robot]
            n_g += 1

        return {
            'robot_partial': robot_partial,
            'obs_partial':   obs_partial,
            'ped_partial':   ped_partial,
            'grp_partial':   grp_partial,
            'robot_full':    robot_full,
            'obs_full':      obs_full,
            'ped_full':      ped_full,
            'grp_full':      grp_full,
            'n_o': n_o,
            'n_p': n_p,
            'n_g': n_g,
        }

    # ------------------------------------------------------------------
    # Action translation: GARN discrete index → custom env (vx, vy)
    # ------------------------------------------------------------------

    def action_to_vel(self, action_idx, robot_theta):
        """
        Map GARN discrete action index → continuous (vx, vy) for custom env.

        action_idx: int in [0, n_actions)
        robot_theta: current robot heading (radians)

        Returns: (vx, vy) as list — pass directly to custom_env.step()
        """
        n_rot = len(self.rotations)
        speed_idx = action_idx // n_rot
        rot_idx   = action_idx  % n_rot

        speed   = self.speeds[speed_idx]      # fraction of v_pref (≤1.0)
        d_theta = self.rotations[rot_idx]     # heading change (radians)

        heading = robot_theta + d_theta
        vx = speed * np.cos(heading)
        vy = speed * np.sin(heading)
        return [vx, vy]
