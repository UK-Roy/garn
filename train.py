"""
Main training entry point for GARN.

Usage:
    conda activate garn
    python train.py                          # train from scratch
    python train.py --resume checkpoints/garn_ep5000.pt
    python train.py --scenario s2           # train on S2 (default: s1)
    python train.py --episodes 20000        # override episode count
"""

import argparse
import yaml
import os
import sys

# Allow running from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.trainer import Trainer


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train GARN')
    parser.add_argument('--env-cfg', default='config/env_config.yaml')
    parser.add_argument('--model-cfg', default='config/model_config.yaml')
    parser.add_argument('--train-cfg', default='config/train_config.yaml')
    parser.add_argument('--resume', default=None, help='Path to checkpoint')
    parser.add_argument('--scenario', default='s1', choices=['s1', 's2'])
    parser.add_argument('--episodes', type=int, default=None)
    args = parser.parse_args()

    env_cfg = load_config(args.env_cfg)
    model_cfg = load_config(args.model_cfg)
    train_cfg = load_config(args.train_cfg)

    # Override scenario config if specified
    if args.scenario != 's1':
        print(f"[Note] Training scenario overridden to: {args.scenario}")
    if args.episodes is not None:
        train_cfg['training']['n_episodes'] = args.episodes

    trainer = Trainer(env_cfg, model_cfg, train_cfg)
    trainer.train(resume_path=args.resume)


if __name__ == '__main__':
    main()
