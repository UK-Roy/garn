"""
Group-Aware reward function: R = R_gl + R_obs + R_prox + R_grp.
R_grp = R_intru + R_ot/flw + R_cops  (paper Section III-B, Eq. 3-8)
"""

import numpy as np


class RewardCalculator:
    def __init__(self, reward_cfg, env_cfg):
        self.goal_reward = reward_cfg.get('goal_reward', 1.0)
        self.collision_penalty = reward_cfg.get('collision_penalty', -0.25)
        self.potential_weight = reward_cfg.get('potential_weight', 1.0)
        self.discomfort_dist = reward_cfg.get('discomfort_dist', 0.2)
        self.discomfort_penalty = reward_cfg.get('discomfort_penalty', -0.1)
        self.c1 = reward_cfg.get('c1', 0.05)
        self.c2 = reward_cfg.get('c2', 0.025)
        self.group_intru_penalty = reward_cfg.get('group_intrusion_penalty', -0.25)
        self.time_penalty = reward_cfg.get('time_penalty', -0.001)
        self.d_frt = env_cfg.get('group', {}).get('d_frt', 2.0)
        self.robot_radius = env_cfg['robot']['radius']

        # Tracking state for temporal rewards
        self._prev_robot_pos = None
        self._prev_dist_to_goal = None
        self._prev_group_centers = {}

    def reset(self):
        self._prev_robot_pos = None
        self._prev_dist_to_goal = None
        self._prev_group_centers = {}

    def compute(self, robot, pedestrians, obstacles, group_spaces, t, dt):
        """
        Compute total reward and info dict.
        Returns (reward, info).
        """
        info = {
            'reached_goal': False,
            'collision': False,
            'n_discomfort': 0,
            'n_intrusion': 0,
            'did_cooperative_pass': False,
            'did_follow': False,
            'did_overtake': False,
        }

        curr_dist = robot.get_goal_dist()

        r_gl = self._goal_reward(robot, curr_dist, info)
        if info['reached_goal']:
            self._prev_robot_pos = robot.pos.copy()
            self._prev_dist_to_goal = curr_dist
            self._prev_group_centers = self._extract_group_centers(group_spaces)
            return r_gl, info

        r_obs = self._obstacle_reward(robot, obstacles, pedestrians, info)
        if info['collision']:
            self._prev_robot_pos = robot.pos.copy()
            self._prev_dist_to_goal = curr_dist
            self._prev_group_centers = self._extract_group_centers(group_spaces)
            return r_obs, info

        r_prox = self._proximity_reward(robot, pedestrians, info)
        r_grp = self._group_reward(robot, group_spaces, info)

        # Potential-based shaping: dense reward for moving toward goal every step
        r_potential = 0.0
        if self._prev_dist_to_goal is not None:
            r_potential = self.potential_weight * (self._prev_dist_to_goal - curr_dist)

        reward = r_gl + r_obs + r_prox + r_grp + r_potential + self.time_penalty

        self._prev_robot_pos = robot.pos.copy()
        self._prev_dist_to_goal = curr_dist
        self._prev_group_centers = self._extract_group_centers(group_spaces)

        return reward, info

    # ------------------------------------------------------------------
    # R_gl: goal reward
    # ------------------------------------------------------------------
    def _goal_reward(self, robot, curr_dist, info):
        if curr_dist < robot.radius + 0.1:
            info['reached_goal'] = True
            return self.goal_reward
        return 0.0

    # ------------------------------------------------------------------
    # R_obs: obstacle / collision reward
    # ------------------------------------------------------------------
    def _obstacle_reward(self, robot, obstacles, pedestrians, info):
        # Check collisions with obstacles
        for obs in obstacles:
            dist = obs.dist_to_point(robot.pos)
            if dist < robot.radius:
                info['collision'] = True
                return self.collision_penalty

        # Check collisions with pedestrians
        for ped in pedestrians:
            dist = np.linalg.norm(robot.pos - ped.pos)
            if dist < robot.radius + ped.radius:
                info['collision'] = True
                return self.collision_penalty

        return 0.0

    # ------------------------------------------------------------------
    # R_prox: proximity / discomfort reward
    # ------------------------------------------------------------------
    def _proximity_reward(self, robot, pedestrians, info):
        reward = 0.0
        for ped in pedestrians:
            dist = np.linalg.norm(robot.pos - ped.pos) - robot.radius - ped.radius
            if dist < self.discomfort_dist:
                info['n_discomfort'] += 1
                reward += self.discomfort_penalty * (self.discomfort_dist - dist)
        return reward

    # ------------------------------------------------------------------
    # R_grp = R_intru + R_ot/flw + R_cops  (paper Eq. 4-8)
    # ------------------------------------------------------------------
    def _group_reward(self, robot, group_spaces, info):
        r_intru = self._intrusion_reward(robot, group_spaces, info)
        r_otflw = self._overtaking_following_reward(robot, group_spaces, info)
        r_cops = self._cooperative_passing_reward(robot, group_spaces, info)
        return r_intru + r_otflw + r_cops

    def _intrusion_reward(self, robot, group_spaces, info):
        """R_intru: -0.25 for each group whose convex hull the robot is inside. (Eq. 5)"""
        reward = 0.0
        for grp in group_spaces.values():
            if grp.contains_point(robot.pos):
                reward += self.group_intru_penalty  # -0.25
                info['n_intrusion'] += 1
        return reward

    def _overtaking_following_reward(self, robot, group_spaces, info):
        """
        R_ot/flw: incentivize overtaking and following groups walking same direction.
        Penalize lagging behind while walking in the same direction. (Eq. 6-7)
        """
        if self._prev_robot_pos is None:
            return 0.0

        reward = 0.0
        v_pref = 1.0  # robot preferred speed
        v_robot = np.linalg.norm(robot.vel)
        if v_robot < 1e-6:
            return 0.0
        v_robot_dir = robot.vel / v_robot

        # M1: groups walking in same direction as robot
        M1 = []
        for gid, grp in group_spaces.items():
            grp_speed = np.linalg.norm(grp.vel)
            if grp_speed < 0.1:
                continue
            grp_dir = grp.vel / grp_speed
            if np.dot(v_robot_dir, grp_dir) > 0.5:
                dist = np.linalg.norm(robot.pos - grp.center)
                if dist <= self.d_frt:
                    M1.append(grp)

        for grp in M1:
            # Displacement of robot projected onto robot's direction
            robot_disp = robot.pos - self._prev_robot_pos
            grp_disp = grp.center - self._prev_group_centers.get(grp.group_id, grp.center)

            delta_robot = np.dot(robot_disp, v_robot_dir)
            delta_grp = np.dot(grp_disp, v_robot_dir)

            # ρ^m: 1 if robot is positioned in front of group center
            rho = 1.0 if np.dot(robot.pos - grp.center, v_robot_dir) > 0 else 0.0
            # 0_{d_m}: 1 if robot is NOT inside group hull
            not_intruding = 0.0 if grp.contains_point(robot.pos) else 1.0

            term = (delta_robot - delta_grp) / (v_pref + 1e-8)
            dot_proj = np.dot(robot_disp, v_robot_dir / (np.linalg.norm(v_robot_dir) + 1e-8))
            reward += not_intruding * self.c1 * term * np.sign(dot_proj)

        if len(M1) > 0:
            info['did_overtake'] = True
            info['did_follow'] = True

        return reward

    def _cooperative_passing_reward(self, robot, group_spaces, info):
        """
        R_cops: reward cooperative passing — moving past approaching groups from front. (Eq. 8)
        """
        if self._prev_robot_pos is None:
            return 0.0

        reward = 0.0
        v_robot = np.linalg.norm(robot.vel)
        if v_robot < 1e-6:
            return 0.0
        v_robot_dir = robot.vel / v_robot

        # M2: groups approaching robot from front
        M2 = []
        for gid, grp in group_spaces.items():
            grp_speed = np.linalg.norm(grp.vel)
            if grp_speed < 0.1:
                continue
            grp_dir = grp.vel / grp_speed
            # Approaching: group moving toward robot direction
            toward_robot = robot.pos - grp.center
            if (np.dot(grp_dir, toward_robot / (np.linalg.norm(toward_robot) + 1e-8)) > 0.3
                    and np.linalg.norm(toward_robot) < self.d_frt):
                M2.append(grp)

        for grp in M2:
            robot_disp = robot.pos - self._prev_robot_pos
            prev_center = self._prev_group_centers.get(grp.group_id, grp.center)
            grp_disp = grp.center - prev_center

            delta_robot = np.dot(robot_disp, v_robot_dir)
            delta_grp = np.dot(grp_disp, v_robot_dir)

            rho = 1.0 if np.dot(robot.pos - grp.center, v_robot_dir) > 0 else 0.0
            not_intruding = 0.0 if grp.contains_point(robot.pos) else 1.0

            term = (delta_robot - delta_grp) / (1.0 + 1e-8)
            reward += not_intruding * self.c2 * rho * abs(term)

        if len(M2) > 0:
            info['did_cooperative_pass'] = True

        return reward

    def _extract_group_centers(self, group_spaces):
        return {gid: grp.center.copy() for gid, grp in group_spaces.items()}
