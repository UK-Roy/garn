"""
Test / evaluation entry point for GARN.

Usage:
    conda activate garn
    python test.py --checkpoint checkpoints/garn_ep20000.pt
    python test.py --checkpoint checkpoints/garn_ep20000.pt --episodes 500
    python test.py --checkpoint checkpoints/garn_ep20000.pt --render
"""

import argparse
import yaml
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.stgan import STGAN
from evaluation.evaluator import Evaluator


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Test GARN')
    parser.add_argument('--checkpoint', required=True, help='Path to .pt checkpoint')
    parser.add_argument('--env-cfg', default='config/env_config.yaml')
    parser.add_argument('--model-cfg', default='config/model_config.yaml')
    parser.add_argument('--train-cfg', default='config/train_config.yaml')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--scenario', default='both', choices=['s1', 's2', 'both'])
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    env_cfg = load_config(args.env_cfg)
    model_cfg = load_config(args.model_cfg)
    train_cfg = load_config(args.train_cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Test] Using device: {device}")

    n_speeds = len(model_cfg['action']['speeds'])
    n_rots = len(model_cfg['action']['rotations'])
    n_actions = n_speeds * n_rots

    model = STGAN(model_cfg, n_actions).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['online_net'])
    model.eval()
    print(f"[Test] Loaded checkpoint: {args.checkpoint}")

    evaluator = Evaluator(env_cfg, model_cfg, train_cfg, device)

    scenarios = ['s1', 's2'] if args.scenario == 'both' else [args.scenario]
    results = {}
    for scen in scenarios:
        print(f"\n[Test] Evaluating scenario {scen.upper()} ({args.episodes} episodes)...")
        metrics = evaluator.evaluate(model, scen, n_episodes=args.episodes,
                                     render=args.render)
        results[scen] = metrics

    if 'both' == args.scenario or len(results) == 2:
        Evaluator.print_table(results.get('s1', {}), results.get('s2', {}))
    else:
        scen = scenarios[0]
        print(f"\nScenario {scen.upper()} results:")
        for k, v in results[scen].items():
            print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
