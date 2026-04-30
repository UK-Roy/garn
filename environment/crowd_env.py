"""
CrowdNav environment compatible with Gymnasium.
Implements the 10m×10m simulation used in the GARN paper (Section IV-A-1).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from environment.agents.robot import Robot
from environment.agents.pedestrian import Pedestrian
from environment.agents.obstacle import CircularObstacle, RectangularObstacle
from environment.sfm.social_force import SocialForceModel
from environment.groups.group_detector import GroupManager
from environment.groups.group_space import GroupSpace
from gam.reward import RewardCalculator


class CrowdNavEnv(gym.Env):
    """
    Crowd navigation environment with group-aware pedestrian simulation.

    Observation: dict with keys matching STGAN inputs
        robot_state: (9,)  full robot state
        obstacles:   (K, 5) obstacle states (padded)
        pedestrians: (I, 9) pedestrian states (padded)
        groups:      (M, 7) group states (padded)

    Action: integer index into discrete action space (speed × rotation)
    """

    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, env_cfg, model_cfg, train_cfg, scenario='s1'):
        super().__init__()
        self.env_cfg = env_cfg
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.scenario = scenario.lower()

        self.dt = env_cfg['world']['dt']
        self.t_limit = env_cfg['world']['t_limit']
        self.world_size = env_cfg['world']['size']

        # Action space
        self.speeds = model_cfg['action']['speeds']
        self.rotations = model_cfg['action']['rotations']
        self.n_actions = len(self.speeds) * len(self.rotations)
        self.action_space = spaces.Discrete(self.n_actions)

        # Observation space (we use dict; actual tensors built in _get_obs)
        # Max agents for padding
        self.max_obstacles = 10
        self.max_pedestrians = 15
        self.max_groups = 6
        self.observation_space = spaces.Dict({
            'robot_state': spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32),
            'obstacles': spaces.Box(-np.inf, np.inf,
                                    shape=(self.max_obstacles, 5), dtype=np.float32),
            'pedestrians': spaces.Box(-np.inf, np.inf,
                                      shape=(self.max_pedestrians, 9), dtype=np.float32),
            'groups': spaces.Box(-np.inf, np.inf,
                                  shape=(self.max_groups, 7), dtype=np.float32),
            'n_obstacles': spaces.Discrete(self.max_obstacles + 1),
            'n_pedestrians': spaces.Discrete(self.max_pedestrians + 1),
            'n_groups': spaces.Discrete(self.max_groups + 1),
        })

        # Sub-systems
        self.robot = Robot(env_cfg)
        self.sfm = SocialForceModel(env_cfg)
        self.group_manager = GroupManager(env_cfg)
        self.reward_calc = RewardCalculator(train_cfg['reward'], env_cfg)

        self.pedestrians = []
        self.obstacles = []
        self.group_spaces = {}
        self.t = 0.0
        self.episode_steps = 0
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.t = 0.0
        self.episode_steps = 0

        self.robot.reset()

        scen_cfg = self.env_cfg[f'scenario_{self.scenario}']
        self.obstacles = self._spawn_obstacles(scen_cfg['n_obstacles'])
        self.pedestrians = self._spawn_pedestrians(scen_cfg)
        self._update_groups()

        self.reward_calc.reset()

        return self._get_obs(), {}

    def step(self, action):
        from utils.transforms import action_to_velocity
        vel, new_theta = action_to_velocity(
            action, self.speeds, self.rotations,
            self.robot.theta, self.robot.v_pref)
        self.robot.step(vel)

        # Advance pedestrians
        self.sfm.step(self.pedestrians, self.obstacles)
        self._respawn_reached_goals()
        self._update_groups()

        self.t += self.dt
        self.episode_steps += 1

        obs = self._get_obs()
        reward, info = self.reward_calc.compute(
            self.robot, self.pedestrians, self.obstacles,
            self.group_spaces, self.t, self.dt)

        terminated = info.get('reached_goal', False) or info.get('collision', False)
        truncated = self.t >= self.t_limit

        return obs, reward, terminated, truncated, info

    def render(self):
        pass  # rendering handled externally via visualization.py

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self):
        robot_state = self.robot.get_full_state()

        obs_arr = np.zeros((self.max_obstacles, 5), dtype=np.float32)
        for i, obs in enumerate(self.obstacles[:self.max_obstacles]):
            state = obs.get_full_state()
            obs_arr[i, :len(state)] = state

        ped_arr = np.zeros((self.max_pedestrians, 9), dtype=np.float32)
        for i, ped in enumerate(self.pedestrians[:self.max_pedestrians]):
            state = ped.get_full_state()
            ped_arr[i, :len(state)] = state

        grp_arr = np.zeros((self.max_groups, 7), dtype=np.float32)
        groups = list(self.group_spaces.values())
        for i, grp in enumerate(groups[:self.max_groups]):
            gs = grp.get_state()
            # Fill dist_to_robot
            gs[6] = np.linalg.norm(self.robot.pos - grp.center)
            grp_arr[i] = gs

        return {
            'robot_state': robot_state,
            'obstacles': obs_arr,
            'pedestrians': ped_arr,
            'groups': grp_arr,
            'n_obstacles': min(len(self.obstacles), self.max_obstacles),
            'n_pedestrians': min(len(self.pedestrians), self.max_pedestrians),
            'n_groups': min(len(groups), self.max_groups),
        }

    def _update_groups(self):
        self.group_manager.update(self.pedestrians)
        all_groups = self.group_manager.get_all_groups()
        self.group_spaces = {}
        ped_map = {p.id: p for p in self.pedestrians}
        for gid, member_ids in all_groups.items():
            members = [ped_map[pid] for pid in member_ids if pid in ped_map]
            if members:
                self.group_spaces[gid] = GroupSpace(gid, members)

    def _safe_pos(self, low, high, clearance_pos, min_clearance=1.5, max_tries=50):
        """Sample a position that is at least min_clearance away from clearance_pos."""
        pos = self._rng.uniform(low, high)
        for _ in range(max_tries):
            if np.linalg.norm(pos - clearance_pos) >= min_clearance:
                return pos
            pos = self._rng.uniform(low, high)
        return pos

    def _spawn_obstacles(self, n_obstacles):
        obstacles = []
        half = self.world_size / 2
        robot_start = np.array(self.env_cfg['robot']['start'])
        for i in range(n_obstacles):
            pos = self._safe_pos(
                [-half + 1, 1], [half - 1, self.world_size - 1],
                robot_start, min_clearance=1.5)
            if self._rng.random() < 0.6:
                radius = self._rng.uniform(0.2, 0.5)
                obstacles.append(CircularObstacle(i, pos, radius))
            else:
                theta = self._rng.uniform(0, np.pi)
                w = self._rng.uniform(0.3, 0.8)
                l = self._rng.uniform(0.5, 1.5)
                obstacles.append(RectangularObstacle(i, pos, theta, w, l))
        return obstacles

    def _spawn_pedestrians(self, scen_cfg):
        pedestrians = []
        ped_id = 0
        half = self.world_size / 2
        robot_start = np.array(self.env_cfg['robot']['start'])

        # Individual pedestrians
        for _ in range(scen_cfg['n_individuals']):
            pos, goal = self._random_start_goal(half, avoid=robot_start)
            vel = self._rng.uniform(-0.5, 0.5, size=2)
            p = Pedestrian(ped_id, pos, vel, group_id=0,
                           theta=np.arctan2(vel[1] + 1e-8, vel[0] + 1e-8))
            p.set_goal(goal)
            pedestrians.append(p)
            ped_id += 1

        # Static group members
        group_id = 1
        for grp in scen_cfg.get('static_groups', []):
            center = self._safe_pos(
                [-half + 2, 2], [half - 2, self.world_size - 2],
                robot_start, min_clearance=2.0)
            for j in range(grp['members']):
                offset = self._rng.uniform(-0.5, 0.5, size=2)
                pos = center + offset
                toward_center = center - pos
                theta = np.arctan2(toward_center[1], toward_center[0])
                vel = np.zeros(2)
                p = Pedestrian(ped_id, pos, vel, group_id=group_id, theta=theta)
                p.set_goal(center)
                pedestrians.append(p)
                ped_id += 1
            group_id += 1

        # Dynamic group members
        for grp in scen_cfg.get('dynamic_groups', []):
            start, goal = self._random_start_goal(half, avoid=robot_start)
            shared_theta = np.arctan2(goal[1] - start[1], goal[0] - start[0])
            for j in range(grp['members']):
                offset = self._rng.uniform(-0.4, 0.4, size=2)
                pos = start + offset
                vel = np.array([
                    self._rng.uniform(0.6, 1.0) * np.cos(shared_theta),
                    self._rng.uniform(0.6, 1.0) * np.sin(shared_theta)
                ])
                p = Pedestrian(ped_id, pos, vel, group_id=group_id, theta=shared_theta)
                p.set_goal(goal + self._rng.uniform(-0.3, 0.3, size=2))
                pedestrians.append(p)
                ped_id += 1
            group_id += 1

        return pedestrians

    def _random_start_goal(self, half, avoid=None, min_clearance=1.5):
        """Random position/goal pair, optionally keeping min_clearance from avoid point."""
        pos = self._rng.uniform([-half + 0.5, 0.5], [half - 0.5, self.world_size - 0.5])
        if avoid is not None:
            for _ in range(50):
                if np.linalg.norm(pos - avoid) >= min_clearance:
                    break
                pos = self._rng.uniform([-half + 0.5, 0.5],
                                        [half - 0.5, self.world_size - 0.5])
        goal = self._rng.uniform([-half + 0.5, 0.5],
                                 [half - 0.5, self.world_size - 0.5])
        while np.linalg.norm(goal - pos) < 2.0:
            goal = self._rng.uniform([-half + 0.5, 0.5],
                                     [half - 0.5, self.world_size - 0.5])
        return pos, goal

    def _respawn_reached_goals(self):
        """Give new goals to pedestrians who reached their current goal."""
        half = self.world_size / 2
        for p in self.pedestrians:
            if p.goal is not None and np.linalg.norm(p.pos - p.goal) < 0.3:
                _, new_goal = self._random_start_goal(half)
                p.set_goal(new_goal)

    def get_group_spaces(self):
        return self.group_spaces

    def get_state_for_network(self):
        """
        Return structured tensors for STGAN:
        (robot_partial, obstacles_partial, pedestrians_partial, groups_partial,
         robot_full, obstacles_full, pedestrians_full, groups_full, counts)
        """
        obs = self._get_obs()
        n_o = obs['n_obstacles']
        n_p = obs['n_pedestrians']
        n_g = obs['n_groups']

        robot_partial = np.array([
            self.robot.pos[0], self.robot.pos[1], self.robot.theta,
            self.robot.vel[0], self.robot.vel[1]
        ], dtype=np.float32)

        obs_partial = np.zeros((self.max_obstacles, 5), dtype=np.float32)
        for i, ob in enumerate(self.obstacles[:self.max_obstacles]):
            s = ob.get_partial_state()
            obs_partial[i] = s

        ped_partial = np.zeros((self.max_pedestrians, 5), dtype=np.float32)
        for i, p in enumerate(self.pedestrians[:self.max_pedestrians]):
            ped_partial[i] = p.get_partial_state()

        grp_partial = np.zeros((self.max_groups, 4), dtype=np.float32)
        for i, grp in enumerate(list(self.group_spaces.values())[:self.max_groups]):
            grp_partial[i] = grp.get_partial_state()

        return {
            'robot_partial': robot_partial,
            'obs_partial': obs_partial,
            'ped_partial': ped_partial,
            'grp_partial': grp_partial,
            'robot_full': obs['robot_state'],
            'obs_full': obs['obstacles'],
            'ped_full': obs['pedestrians'],
            'grp_full': obs['groups'],
            'n_o': n_o, 'n_p': n_p, 'n_g': n_g,
        }
