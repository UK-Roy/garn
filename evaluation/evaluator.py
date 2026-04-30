"""
Evaluator: runs N test episodes and computes metrics matching paper Table I.

Metrics:
  r_s  : success rate (reached goal without collision)
  r_c  : collision rate
  t_s  : average navigation time (successful episodes only)
  d_o  : average minimum distance to any obstacle
  n_dis: total individual discomfort instances across all episodes
  n_i  : total group intrusion count
  n_cp : total cooperative pass count
  n_flw: total following count
  n_ot : total overtaking count
"""

import numpy as np
import torch

from environment.crowd_env import CrowdNavEnv


class Evaluator:
    def __init__(self, env_cfg, model_cfg, train_cfg, device):
        self.env_cfg = env_cfg
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.device = device

    def evaluate(self, model, scenario, n_episodes=500, render=False):
        """Run n_episodes test episodes and return aggregated metrics dict."""
        env = CrowdNavEnv(self.env_cfg, self.model_cfg, self.train_cfg,
                          scenario=scenario)
        model.eval()

        successes = 0
        collisions = 0
        nav_times = []
        min_dists = []
        n_dis_total = 0
        n_i_total = 0
        n_cp_total = 0
        n_flw_total = 0
        n_ot_total = 0

        with torch.no_grad():
            for ep in range(n_episodes):
                _, _ = env.reset()
                hx = model.reset_hidden(batch_size=1, device=self.device)
                ep_min_dist = float('inf')
                ep_dis = 0
                ep_ni = 0
                ep_cp = 0
                ep_flw = 0
                ep_ot = 0
                t = 0.0
                reached = False
                collided = False

                while True:
                    net_obs = env.get_state_for_network()
                    action, hx = model.get_action(net_obs, hx, self.device, epsilon=0.0)
                    _, reward, terminated, truncated, info = env.step(action)

                    t += env.dt

                    # Track minimum obstacle distance
                    robot_pos = env.robot.pos
                    for obs in env.obstacles:
                        d = obs.dist_to_point(robot_pos) - env.robot.radius
                        ep_min_dist = min(ep_min_dist, max(0.0, d))
                    for ped in env.pedestrians:
                        d = (np.linalg.norm(robot_pos - ped.pos)
                             - env.robot.radius - ped.radius)
                        ep_min_dist = min(ep_min_dist, max(0.0, d))

                    ep_dis += info.get('n_discomfort', 0)
                    ep_ni += info.get('n_intrusion', 0)
                    ep_cp += int(info.get('did_cooperative_pass', False))
                    ep_flw += int(info.get('did_follow', False))
                    ep_ot += int(info.get('did_overtake', False))

                    if info.get('reached_goal', False):
                        reached = True
                    if info.get('collision', False):
                        collided = True

                    if terminated or truncated:
                        break

                if reached and not collided:
                    successes += 1
                    nav_times.append(t)
                if collided:
                    collisions += 1

                if ep_min_dist < float('inf'):
                    min_dists.append(ep_min_dist)
                n_dis_total += ep_dis
                n_i_total += ep_ni
                n_cp_total += ep_cp
                n_flw_total += ep_flw
                n_ot_total += ep_ot

        metrics = {
            'r_s': successes / n_episodes,
            'r_c': collisions / n_episodes,
            't_s': float(np.mean(nav_times)) if nav_times else float('nan'),
            'd_o': float(np.mean(min_dists)) if min_dists else 0.0,
            'n_dis': n_dis_total,
            'n_i': n_i_total,
            'n_cp': n_cp_total,
            'n_flw': n_flw_total,
            'n_ot': n_ot_total,
        }

        model.train()
        return metrics

    @staticmethod
    def print_table(metrics_s1, metrics_s2):
        """Print results in the same format as paper Table I."""
        print("\n" + "=" * 80)
        print("GARN Evaluation Results (Table I format)")
        print("=" * 80)
        hdr = f"{'Metric':<10} {'S1':>10} {'S2':>10}   {'Target S1':>10} {'Target S2':>10}"
        print(hdr)
        print("-" * 80)
        targets_s1 = {'r_s': 1.00, 'r_c': 0.00, 't_s': 12.87, 'd_o': 0.22,
                      'n_i': 29, 'n_cp': 8, 'n_flw': 25}
        targets_s2 = {'r_s': 0.97, 'r_c': 0.00, 't_s': 20.83, 'd_o': 0.21,
                      'n_i': 72, 'n_cp': 23, 'n_flw': 19}
        for k in ['r_s', 'r_c', 't_s', 'd_o', 'n_dis', 'n_i', 'n_cp', 'n_flw', 'n_ot']:
            v1 = metrics_s1.get(k, float('nan'))
            v2 = metrics_s2.get(k, float('nan'))
            t1 = targets_s1.get(k, '—')
            t2 = targets_s2.get(k, '—')
            print(f"{k:<10} {v1:>10.3f} {v2:>10.3f}   {str(t1):>10} {str(t2):>10}")
        print("=" * 80 + "\n")
