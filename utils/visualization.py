"""Visualization utilities for crowd navigation episodes."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
from matplotlib.collections import LineCollection


def render_episode(history, env_cfg, save_path=None, show=False):
    """
    Render a full episode trajectory.
    history: list of dicts with keys:
        robot_pos, robot_theta, pedestrian_states, obstacle_states, group_states
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    world_size = env_cfg.get('world', {}).get('size', 10.0)
    ax.set_xlim(-world_size / 2, world_size / 2)
    ax.set_ylim(0, world_size)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('GARN Episode Trajectory')
    ax.grid(True, alpha=0.3)

    robot_positions = np.array([h['robot_pos'] for h in history])
    ax.plot(robot_positions[:, 0], robot_positions[:, 1],
            'b-', linewidth=2, label='Robot path', zorder=5)
    ax.plot(robot_positions[0, 0], robot_positions[0, 1],
            'bs', markersize=8, label='Robot start', zorder=6)
    ax.plot(robot_positions[-1, 0], robot_positions[-1, 1],
            'b*', markersize=12, label='Robot end', zorder=6)

    if history:
        frame = history[-1]
        for obs in frame.get('obstacle_states', []):
            if obs['type'] == 'circle':
                c = plt.Circle(obs['pos'], obs['radius'],
                               color='gray', fill=True, alpha=0.7, zorder=3)
                ax.add_patch(c)
            else:
                rect = patches.Rectangle(
                    (obs['pos'][0] - obs['width'] / 2, obs['pos'][1] - obs['length'] / 2),
                    obs['width'], obs['length'],
                    angle=np.degrees(obs['theta']),
                    color='gray', fill=True, alpha=0.7, zorder=3)
                ax.add_patch(rect)

        group_colors = plt.cm.Set1(np.linspace(0, 1, 9))
        for gidx, grp in enumerate(frame.get('group_states', [])):
            color = group_colors[gidx % len(group_colors)]
            if grp.get('hull_pts') is not None and len(grp['hull_pts']) >= 3:
                from matplotlib.patches import Polygon
                hull = np.array(grp['hull_pts'])
                poly = Polygon(hull, closed=True, fill=True,
                               facecolor=(*color[:3], 0.15),
                               edgecolor=(*color[:3], 0.8),
                               linewidth=2, linestyle='--', zorder=2)
                ax.add_patch(poly)

        for ped in frame.get('pedestrian_states', []):
            ax.add_patch(plt.Circle(ped['pos'], 0.3, color='orange', alpha=0.8, zorder=4))
            ax.annotate(f"P{ped['id']}", ped['pos'], fontsize=6,
                        ha='center', va='center', color='black', zorder=7)

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def plot_training_curves(log_path, save_path=None):
    """Plot training reward and success rate curves from tensorboard logs."""
    import pandas as pd
    try:
        from tensorboard.backend.event_processing import event_accumulator
        ea = event_accumulator.EventAccumulator(log_path)
        ea.Reload()
        tags = ea.Tags().get('scalars', [])

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('GARN Training Curves')

        plot_map = {
            'reward': (axes[0, 0], 'Episode Reward', 'Reward'),
            'success_rate': (axes[0, 1], 'Success Rate', 'Rate'),
            'collision_rate': (axes[1, 0], 'Collision Rate', 'Rate'),
            'nav_time': (axes[1, 1], 'Navigation Time (s)', 'Time (s)'),
        }
        for tag, (ax, title, ylabel) in plot_map.items():
            if tag in tags:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                vals = [e.value for e in events]
                ax.plot(steps, vals)
                ax.set_title(title)
                ax.set_xlabel('Episode')
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    except ImportError:
        print("tensorboard not available for curve plotting.")
